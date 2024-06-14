import sys 
import os
PATH = '/home/user/NEW_MODEL_CACHE/'
os.environ['TRANSFORMERS_CACHE'] = '/home/mila/e/emiliano.penaloza/scratch/'
os.environ['HF_HOME'] = '/home/mila/e/emiliano.penaloza/scratch/models'
os.environ['HF_DATASETS_CACHE'] = '/home/mila/e/emiliano.penaloza/scratch/models'
os.environ['TORCH_HOME'] = '/home/mila/e/emiliano.penaloza/scratch/models'
from trainer.transformer_utilts import load_data,train_fineTuning,eval_model,get_scheduler,get_loss, get_embeddings,parse_args,eval_model
sys.path.append("..")
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import time 
import torch
from dotenv import load_dotenv
from model.MF import get_model,get_tokenizer
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
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
import os
from model.eval_model import LargeScaleEvaluator





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
    model,lora_config = get_model(args, tokenizer, num_movies, rank, world_size)
    model.to(rank)
    
    if args.warmup > 0 and args.mask == 0 and args.embedding_module != 'VAE':
        dir_name = args.scratch + '/' + args.embedding_module.replace('/','_') + f"/{args.data_name}/"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        precomputed_embeddings = get_embeddings(model, rec_dataloader, rank, world_size, num_movies, tokenizer, save_path= dir_name , save_name= 'embeddings.pkl', save=True,bfloat16 = False if args.embedding_module == 'google-t5/t5-3b' else True)
        val_embeddings = get_embeddings(model, val_dataloader, rank, world_size, num_movies, tokenizer, save_path= dir_name , save_name= 'val_embeddings.pkl', save=True,bfloat16 = False if args.embedding_module == 'google-t5/t5-3b' else True)
        test_embeddings = get_embeddings(model, test_dataloader, rank, world_size, num_movies, tokenizer, save_path= dir_name , save_name= 'test_embeddings.pkl', save=True,bfloat16 = False if args.embedding_module == 'google-t5/t5-3b' else True)
    else:
        precomputed_embeddings = None
            
    model = get_peft_model(model, lora_config)           
    
    if args.warmup> 0 :
            for name, param in model.named_parameters():
                if 'lora' in name:
                    param.requires_grad = False

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    print(f"Number of Batches = {len(rec_dataloader)}")
    #get a subset of model parameters that require grad to feed into the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr,weight_decay=args.l2_lambda)
    scheduler = get_scheduler(optimizer, args)
    min_val_recall = -np.inf
    patience_counter = 0
    last_saved_epoch = 0
    loss_f = get_loss(args.loss)
    item_title_dict = map_id_to_title(args.data_name)
    item_genre_dict = map_id_to_genre(args.data_name)
    large_change_evaluator = LargeScaleEvaluator(model,item_title_dict,item_genre_dict,tokenizer, rank,args)

    

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
            precomputed_embeddings = val_embeddings = test_embeddings  = None 
            
            enabled_lora = True
        
        train_fineTuning(args, rec_dataloader,augmented_dataloader, model, optimizer, scheduler,pbar,e,rank,tokenizer,precomputed_embeddings if e < args.warmup else None)
        if e % 1 == 0: 
            torch.cuda.set_device(rank)

            val_loss,outputs = eval_model(args,model,val_dataloader,rank,val_data_tr,val_embeddings if e < args.warmup else None ,loss_f = loss_f,world_size = world_size)
            if not args.debug :
                val_outputs = {f'val_{k}': v for k,v in outputs.items()}
                wandb.log({**val_outputs,'validation_loss': val_loss})

        if args.eval_control:
            if world_size > 1:
                    all_up = [None for _ in range(world_size)]
                    all_down = [None for _ in range(world_size)]
                    delta_up, delta_down = large_change_evaluator.evaluate(val_dataloader, prompts, 20,rank = rank)
                    dist.all_gather_object(all_up, delta_up)
                    dist.all_gather_object(all_down, delta_down)

                    delta_up = np.mean(all_up)
                    delta_down = np.mean(all_down)
            else:
                    delta_up, delta_down = large_change_evaluator.evaluate(val_dataloader, prompts, 20,rank = rank) 
            
            wandb.log({'delta_up': delta_up})
            wandb.log({'delta_down': delta_down})

            if outputs['ndcg@50'] > min_val_recall:
                min_val_recall = outputs['ndcg@50']
                best_e = e 

                # torch.save(model.module.state_dict(), f'{args.scratch}/saved_model/{args.data_name}/{args.model_log_name}.pt')
                best_model = model.module.state_dict()
                torch.save(model.module.state_dict(), f'{args.scratch}/saved_model/{args.data_name}/{args.model_log_name}.pt')
                
                last_saved_epoch = e
                wandb.log({'ndcg@50_max': min_val_recall})
                wandb.log({'last_saved_epoch': last_saved_epoch})
                print(f"Early stopping training at epoch {e}")
                test_loss,test_outputs = eval_model(args,model,test_dataloader,rank,test_data_tr ,test_embeddings if e < args.warmup else None,loss_f = loss_f,world_size = world_size)
                test_outputs = {f'test_{k}': v for k,v in test_outputs.items()}
                
                wandb.log({**test_outputs})

                if args.patience == patience_counter:
                    print(f"Early stopping training at epoch {e}")
                    
             

            pbar.set_postfix({'val_loss': val_loss,'val_ndcg': outputs['ndcg@50'],'recall': outputs['recall@50'],'last_saved_epoch': last_saved_epoch})
            
    model.module.load_state_dict(best_model)
    torch.save(model.module.state_dict(), f'{args.scratch}/saved_model/{args.data_name}/{args.model_log_name}.pt')
    model.to(rank)

    val_loss,val_outputs =eval_model(args,model,val_dataloader,rank,val_data_tr, val_embeddings if best_e < args.warmup else None ,loss_f = loss_f,world_size = world_size)

    test_loss,test_outputs = eval_model(args,model,test_dataloader,rank,test_data_tr,test_embeddings if best_e < args.warmup else None,loss_f = loss_f,world_size = world_size)
    test_outputs = {f'test_{k}': v for k,v in test_outputs.items()}
    
    wandb.log({'test_recall@10': test_outputs['recall@10'],
               'test_recall@20': test_outputs['recall@20'],
               'test_recall@50': test_outputs['recall@50'],
               'test_ndcg@10': test_outputs['ndcg@10'],
               'test_ndcg@20': test_outputs['ndcg@20'],
               'test_ndcg@50': test_outputs['ndcg@50'],
               'test_MRR@10': test_outputs['mrr@10'],
               'test_MRR@20': test_outputs['mrr@20'],
               'test_MRR@50': test_outputs['mrr@50']})
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time} seconds")


    log_results = pd.DataFrame({'model_name': [args.model_log_name],
                                    'test_recall@10': [test_outputs['recall@10']],
                                    'test_recall@20': [test_outputs['recall@20']],
                                    'test_recall@50': [test_outputs['recall@50']],
                                    'test_ndcg@10': [test_outputs['ndcg@10']],
                                    'test_ndcg@20': [test_outputs['ndcg@20']],
                                    'test_ndcg@50': [test_outputs['ndcg@50']],
                                    'test_MRR@10' : [test_outputs['mrr@10']],
                                    'test_MRR@20' : [test_outputs['mrr@20']],
                                    'test_MRR@50' : [test_outputs['mrr@50']],
                                    'val_recall@20': [val_outputs['recall@20']],
                                    'val_recall@50': [val_outputs['recall@50']],
                                    'val_ndcg@20': [val_outputs['ndcg@20']],
                                    'val_ndcg@50': [val_outputs['ndcg@50']],
                                    'train_time_min': [elapsed_time / 60]}).round(4)

    csv_path = f"./model_logs/{args.data_name}/parameter_sweep.csv"
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
    
    world_size = torch.cuda.device_count()
    mp.spawn(train_fun,
             args = (world_size,),nprocs = world_size)

    model_name = f"{args.embedding_module}-large"
    
    

    
