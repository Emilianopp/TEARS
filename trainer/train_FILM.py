import sys 
import os
PATH = '/home/user/NEW_MODEL_CACHE/'
os.environ['TRANSFORMERS_CACHE'] = '/home/mila/e/emiliano.penaloza/scratch/'
os.environ['HF_HOME'] = '/home/mila/e/emiliano.penaloza/scratch/models'
os.environ['HF_DATASETS_CACHE'] = '/home/mila/e/emiliano.penaloza/scratch/models'
os.environ['TORCH_HOME'] = '/home/mila/e/emiliano.penaloza/scratch/models'
from trainer.transformer_utilts import *
sys.path.append("..")
import torch.optim as optim

from tqdm import tqdm
import pandas as pd
import time 
import torch
from dotenv import load_dotenv
from model.MF import get_model,get_tokenizer,get_EASE
from helper.dataloader import *
import wandb
from tqdm import tqdm
from peft import LoraConfig, TaskType, PeftModel,get_peft_model
from transformers import T5Tokenizer ,AutoTokenizer
import torch.multiprocessing as mp

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoConfig
from transformers import AutoModel
from model.eval_model import LargeScaleEvaluator
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_flash_sdp(False)
import os
def set_seed(seed: int = 2024) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

set_seed()




args = parse_args(notebook=False)

debug_string  = "debug" if args.debug else ""
args.debug_prompts = False
args.loss = 'bce_softmax'


def cleanup():
    dist.destroy_process_group()
    

def setup(rank,world_size):
    assert torch.distributed.is_available()
    print("PyTorch Distributed available.")
    print("  Backends:")
    print(f"    Gloo: {torch.distributed.is_gloo_available()}")
    print(f"    NCCL: {torch.distributed.is_nccl_available()}")
    print(f"    MPI:  {torch.distributed.is_mpi_available()}")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
 
    torch.distributed.init_process_group(
         backend="nccl",
        init_method="env://",
         world_size=world_size,
         rank=rank,
     )

def train_fun(rank,world_size):
    setup(rank, world_size)
    set_seed(args.seed)
    start_time = time.time()
    #get unique hash for both gpus
    if not args.debug:
        tags = [args.data_name,os.environ["SLURM_JOB_ID"]]
        wandb.init(project='llm4rec', name=args.model_log_name,
                   group=time.strftime("%d-%m-%Y_%H:%M:%S", time.localtime()),
                   tags = tags,config=args)
        wandb.config.update(args)
        wandb.watch_called = False  # To avoid re-watching the model
        wandb.define_metric("batch")
        wandb.define_metric("epoch")
        wandb.define_metric("total_steps")
        wandb.define_metric("val_recall", step_metric="epoch")
        wandb.define_metric("val_ndcg", step_metric="epoch")
        wandb.define_metric("bce_loss", step_metric="total_steps")
        wandb.define_metric("loss", step_metric="total_steps")
        wandb.define_metric("lr", step_metric="epochs")

    torch.cuda.set_device(rank)
    
    
    tokenizer = get_tokenizer(args)
    prompts,rec_dataloader,augmented_dataloader,num_movies,val_dataloader,test_dataloader,val_data_tr,test_data_tr= load_data(args,tokenizer,rank,world_size)
    if args.EASE:
        train_items = []
        target_items = []
        for b in rec_dataloader: 
            train_items.append(b['labels_tr'])
            target_items.append(b['labels'])
        train_items = torch.concat(train_items)
        target_items = torch.concat(target_items)

        
        val_items = []
        val_target_items = []
        for b in val_dataloader: 
            val_items.append(b['labels_tr'])
            val_target_items.append(b['labels'])
        val_target_items = torch.concat(val_target_items)
        val_items = torch.concat(val_items)
        
        test_items = []
        test_target_items = []
        for b in test_dataloader: 
            test_items.append(b['labels_tr'])
            test_target_items.append(b['labels'])
        test_target_items = torch.concat(test_target_items)
        test_items = torch.concat(test_items)
        
        
        train_matrix = 0
        l2_regs = np.linspace(1,10000,50)

        for l2_reg in (pbar:=tqdm(l2_regs[1:2])):
            model = get_EASE(args,num_movies,l2_reg)
            print(f"{model=}")
            model(target_items)
            metrics = model.eval(val_items,val_target_items)
            recall = metrics['ndcg@50']
            pbar.set_description(f"Recall@50 = {recall}, l2_reg = {l2_reg}")
            # print(f"Recall@50 = {recall}")
            if recall > train_matrix:
                train_matrix = recall
                best_l2_reg = l2_reg
        print(f"Best L2 Reg = {best_l2_reg}")
        test_recall = model.eval(test_items,test_target_items)
        print(f"{test_recall=}")
        exit()
    
    model,lora_config = get_model(args, tokenizer, num_movies, rank, world_size)
    model.to(rank)
    if args.eval_control:
        item_title_dict = map_id_to_title(args.data_name)
        item_genre_dict = map_id_to_genre(args.data_name)
        large_change_evaluator = LargeScaleEvaluator(model,item_title_dict,item_genre_dict,tokenizer, 0,args,alpha =0.5)
        large_change_evaluator.set_alpha2(0.5)

    if False and args.warmup > 0 and args.mask == 0 and args.embedding_module != 'VAE':
        #freeze everything that is not the classifier
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        dir_name = args.scratch + '/' + args.embedding_module.replace('/','_') + f"/{args.data_name}/"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        precomputed_embeddings = get_embeddings(model, rec_dataloader, rank, world_size, num_movies, tokenizer, save_path= dir_name , save_name= 'embeddings.pt', save=True,bfloat16 = False)
        val_embeddings = get_embeddings(model, val_dataloader, rank, world_size, num_movies, tokenizer, save_path= dir_name , save_name= 'val_embeddings.pt', save=True,bfloat16 = False )
        test_embeddings = get_embeddings(model, test_dataloader, rank, world_size, num_movies, tokenizer, save_path= dir_name , save_name= 'test_embeddings.pt', save=True,bfloat16 = False )
    else:
        precomputed_embeddings, val_embeddings, test_embeddings = None,None,None
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    if lora_config is not None:

        model = get_peft_model(model, lora_config)           
    
    if args.warmup> 0 :
            for name, param in model.named_parameters():
                if 'lora' in name:
                    param.requires_grad = False
    mods = ['OTRecVAE','FT5RecVAE','T5Vae','MacridTEARS','RecVAEGenreVAE']
    if args.warmup <= 0 and args.embedding_module in mods:
        for name, param in model.vae.named_parameters():

                print('IN LOOP')
                
                # if name in ['encoder','item_embeddings','prototype_embeddings']:
                if 'q' in name or  'encoder' in name or 'item_embeddings' in name or 'prototype_embeddings' in name  :
                    print('TURNED OFF ENCODER')
                    param.requires_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    model.to(rank)
    
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False if 'RecVAE' in args.embedding_module else True)
    print(f"Number of Batches = {len(rec_dataloader)}")
    #get a subset of model parameters that require grad to feed into the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr,weight_decay=args.l2_lambda) if args.embedding_module != 'RecVAE' else optim.Adam(model.parameters(),lr = args.lr ,weight_decay = args.l2_lambda)
    scheduler = get_scheduler(optimizer, args)
    min_val_recall = -np.inf
    patience_counter = 0
    last_saved_epoch = 0
    loss_f = get_loss(args.loss)
    val_recall = 0

    

    enabled_lora = False
    for e in (pbar := tqdm(range(args.epochs))):
        model.train()
        model.share_memory()
        if args.warmup > 0 and e > args.warmup and enabled_lora == False:
            
            for name, param in model.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(params, lr=args.lr2,weight_decay=args.l2_lambda)
            scheduler = get_scheduler(optimizer, args)
            
            enabled_lora = True

        if args.embedding_module in mods:
            trainT5Vae(args, rec_dataloader, model, optimizer, scheduler,pbar,e,rank,precomputed_embeddings if e < args.warmup else None)
        
        else: 
            
            train_distill(args, rec_dataloader, model, optimizer, scheduler,pbar,e,rank,precomputed_embeddings if e < args.warmup else None)
            
        if e % 1 == 0: 
            torch.cuda.set_device(rank)

            val_loss,outputs = eval_model(args,model,val_dataloader,rank,val_data_tr,val_embeddings if e < args.warmup else None ,loss_f = loss_f,world_size = world_size,vae = True)
            if not args.debug :
                val_outputs = {f'val_{k}': v for k,v in outputs.items()}
                wandb.log({**val_outputs})
                wandb.log({'val_l':val_loss})
            if args.eval_control:
                if world_size > 1:
                        all_up = [None for _ in range(world_size)]
                        all_down = [None for _ in range(world_size)]
                        delta_up, delta_down = large_change_evaluator.evaluate(test_dataloader, prompts, 20,rank = rank)
                        dist.all_gather_object(all_up, delta_up)
                        dist.all_gather_object(all_down, delta_down)

                        delta_up = np.mean(all_up)
                        delta_down = np.mean(all_down)
                else:
                        delta_up, delta_down = large_change_evaluator.evaluate(test_dataloader, prompts, 20,rank=rank) 
                #loge detla_up and delta_down
                wandb.log({'delta_up':delta_up})
                wandb.log({'delta_down':delta_down})
            eval_metric = outputs['ndcg@50'] if 'ndcg@50_avg' not in outputs else outputs['ndcg@50_avg']
            if eval_metric > min_val_recall:
                min_val_recall = eval_metric
                best_e = e 
                
                # torch.save(model.module.state_dict(), f'{args.scratch}/saved_model/{args.data_name}/{args.model_log_name}.pt')
                best_model = model.module.state_dict()
                torch.save(model.module.state_dict(), f'{args.scratch}/saved_model/{args.data_name}/{args.model_log_name}.pt')
                print('saved_model')
                

                last_saved_epoch = e
                wandb.log({'ndcg@50_max': min_val_recall})
                wandb.log({'last_saved_epoch': last_saved_epoch})
                print(f"Early stopping training at epoch {e}")
                test_loss,test_outputs = eval_model(args,model,test_dataloader,rank,test_data_tr ,test_embeddings if e < args.warmup else None,loss_f = loss_f,world_size = world_size,vae = True)
                test_outputs = {f'test_{k}': v for k,v in test_outputs.items()}
                
                wandb.log({**test_outputs})

                if args.patience == patience_counter:
                    print(f"Early stopping training at epoch {e}")
                    
             

            pbar.set_postfix({'val_loss': val_loss,'val_ndcg': outputs['ndcg@50'],'recall': outputs['recall@50'],'last_saved_epoch': last_saved_epoch})


    model,lora_config = get_model(args, tokenizer, num_movies, rank, world_size)
    if lora_config is not None:

        model = get_peft_model(model, lora_config)           

    model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False if 'RecVAE' not in args.embedding_module else True)
    state_dict = torch.load( f'{args.scratch}/saved_model/{args.data_name}/{args.model_log_name}.pt')
    model.module.load_state_dict(state_dict)
    
    model.to(rank)


    val_loss,val_outputs =eval_model(args,model,val_dataloader,rank,val_data_tr, val_embeddings if best_e < args.warmup else None ,loss_f = loss_f,world_size = world_size,vae = True)
    test_loss,test_outputs = eval_model(args,model,test_dataloader,rank,test_data_tr,test_embeddings if best_e < args.warmup else None,loss_f = loss_f,world_size = world_size,vae = True)
    test_outputs = {f'test_{k}': v for k,v in test_outputs.items()}
    print(f"{test_outputs=}")
    val_outputs = {f'val_{k}': v for k,v in val_outputs.items()}
    print(f"{val_outputs=}")
    
    # wandb.log(})
    #save model
    # torch.save(model.module.state_dict(), f'{args.scratch}/saved_model/{args.data_name}/{args.model_log_name}.pt')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time} seconds")

    out_metrics = defaultdict(dict)
    ups = defaultdict(list)
    downs = defaultdict(list)
    ndcgs20 = defaultdict(list)
    alphas = [x/10 for x in range(0,11)]


    module = args.embedding_module
    if args.eval_control:
        if args.embedding_module in ['OTRecVAE','T5Vae','VariationalT5','MacridTEARS']:
            if module in ['VariationalT5']:
                eval_model_obj = LargeScaleEvaluator(model,item_title_dict,item_genre_dict,tokenizer, 0,args,alpha = 0 )
                metrics = eval_model(args,model,test_dataloader,rank,val_data_tr , None,loss_f = loss_f,world_size = world_size,vae = True,mult_process = False,alpha = alpha)[1]
                down,up = eval_model_obj.evaluate(test_dataloader, prompts,20,rank = 0)
                ups[module].append(up)
                downs[module].append(-down)

            else: 
                ndcgs = {}
                n20 = []
                eval_model_obj = LargeScaleEvaluator(model,item_title_dict,item_genre_dict,tokenizer, 0,args,alpha = 0 )
                
                for alpha in alphas:
            
                    metrics = eval_model(args,model,val_dataloader,rank,val_data_tr , None,loss_f = loss_f,world_size = world_size,vae = True,mult_process = False,alpha = alpha)[1]
                    n2_track = eval_model(args,model,test_dataloader,rank,test_data_tr , None,loss_f = loss_f,world_size = world_size,vae = True,mult_process = False,alpha = alpha)[1]['ndcg@20']
                    
                    eval_model_obj.alpha = alpha
                    eval_model_obj.alpha2 = alpha
                    up,down = eval_model_obj.evaluate(test_dataloader, prompts,20,rank = 0)
                    ndcgs20[module].append(n2_track)
                    ups[module].append(-up)
                    downs[module].append(down)
                    
                    ndcgs[alpha] = metrics['ndcg@20']

                max_alpha = max(ndcgs, key=ndcgs.get)
                metrics = eval_model(args,model,test_dataloader,rank,test_data_tr , None,loss_f = loss_f,world_size = world_size,vae = True,mult_process = False,alpha = max_alpha)[1]

            out_metrics[module] = metrics

    # Save metrics to a dataframe
                

    
    log_results = pd.DataFrame({'model_name': [args.model_log_name],**test_outputs,**val_outputs,
                                'ups': [[ups[module]]], 'downs': [[downs[module]]], 'ndcgs': [[ndcgs20[module]]]
                                   }).round(4)
    
    # log_results = pd.concat([log_results], axis=1)

    csv_path = f"./model_logs/{args.data_name}/{args.embedding_module}/parameter_sweep.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    print(f'Saved to {csv_path}')
    # Check if the CSV file already exists
    if os.path.exists(csv_path):
        existing_data = pd.read_csv(csv_path)
        updated_data = pd.concat([existing_data, log_results], ignore_index=True)
        updated_data.to_csv(csv_path, index=False)
    else:
        log_results.to_csv(csv_path, index=False)
    cleanup()
if __name__ == "__main__":
    set_seed()
    world_size = torch.cuda.device_count()

    mp.spawn(train_fun,
             args = (world_size,),nprocs = world_size)

    model_name = f"{args.embedding_module}-large"
    
    

    
