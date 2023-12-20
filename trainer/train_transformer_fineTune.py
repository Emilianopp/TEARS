import pickle
from peft import get_peft_model
import json
from tqdm import tqdm
import openai
import pandas as pd
from data.dataloader import get_dataloader
from helper.dataloader import load_pickle, map_title_to_id, map_id_to_title
import re
from torch.utils.data import DataLoader, Subset
import torch
import os
from dotenv import load_dotenv
from model.MF import MatrixFactorizationLLM,sentenceT5Classification
from model.decoderMLP import decoderMLP, decoderAttention
from model.transformerModel import movieTransformer
import argparse
from torch.optim.lr_scheduler import LambdaLR
from helper.eval_metrics import precision_at_k, recall_at_k, mapk, ndcg_k
from helper.dataloader import *
from .training_utils import *
import wandb
from tqdm import tqdm
import math
from torch.nn.parallel import DataParallel
from .transformer_utilts import *
from peft import LoraConfig, TaskType
from transformers import T5ForSequenceClassification
#import mp 
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP



args = parse_args()



debug_string  = "debug" if args.debug else ""

if args.debugger:
    import debugpy
    debugpy.listen(5678)
    print('attach please')
    debugpy.wait_for_client()
    
def cleanup():
    dist.destroy_process_group()
    
def setup(rank,world_size):
    assert torch.distributed.is_available()
    print("PyTorch Distributed available.")
    print("  Backends:")
    print(f"    Gloo: {torch.distributed.is_gloo_available()}")
    print(f"    NCCL: {torch.distributed.is_nccl_available()}")
    print(f"    MPI:  {torch.distributed.is_mpi_available()}")
 
    # NOTE: the env:// init method uses FileLocks, which sometimes causes deadlocks due to the
    # distributed filesystem configuration on the Mila cluster.
    # For multi-node jobs, use the TCP init method instead.
   

     # DDP Job is being run via `srun` on a slurm cluster.
    # rank = int(os.environ["SLURM_PROCID"])
    # local_rank = int(os.environ["SLURM_LOCALID"])
    # world_size = int(os.environ["SLURM_NTASKS"])
 
     # SLURM var -> torch.distributed vars in case needed
     # NOTE: Setting these values isn't exactly necessary, but some code might assume it's
     # being run via torchrun or torch.distributed.launch, so setting these can be a good idea.
    os.environ["RANK"] = str(rank)
    # os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
 
    torch.distributed.init_process_group(
         backend="nccl",
        init_method="env://",
         world_size=world_size,
         rank=rank,
     )


def get_embeddings(prompts,max_l):
    if os.path.isfile(f'{args.scratch}/{args.data_name}/embedding_module_transformer{debug_string}.pt'):
        embeddings = torch.load(f'{args.scratch}/{args.data_name}/embedding_module{debug_string}.pt')
    else:
        print("Making Embeddings")
        prompt_dataset = PromptDataset(prompts)
        prompt_data_loader = DataLoader(prompt_dataset, batch_size=len(prompts)//2, shuffle=True)

        
        embeddings = make_embeddings(args.embedding_module,prompt_data_loader,max_l)
        torch.save(embeddings,f'{args.sctratch}/data_preprocessed/{args.data_name}/embedding_module{debug_string}.pt')
    return embeddings

def get_dataloaders(data,rank,num_movies,world_size,bs):
    rec_dataset = RecDatasetNegatives(data,num_negatives=1) if args.neg_sample else RecDatasetFull(data,num_movies=num_movies)
    sampler = DistributedSampler(rec_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    rec_dataloader = DataLoader(rec_dataset, batch_size=bs, collate_fn=(custom_collate if args.neg_sample else None) , num_workers = 0,pin_memory=False,
                                sampler = sampler) 
    return rec_dataloader

def load_data(args,rank,world_size):
    # 1. Data Loading & Preprocessing

    train_data = pd.read_csv(f'./data_preprocessed/{args.data_name}/train_leave_one_out_timestamped.csv')
    valid_data = pd.read_csv(f'./data_preprocessed/{args.data_name}/validation_leave_one_out_timestamped.csv')
    test_data = pd.read_csv(f'./data_preprocessed/{args.data_name}/test_leave_one_out_timestamped.csv')
    strong_generalization_set = pd.read_csv(f'./data_preprocessed/{args.data_name}/strong_generalization_set_timestamped.csv')


    
    test_items = dict(test_data.sort_values(by='userId').groupby('userId')['movieId'].agg(list).reset_index().values)

    val_items = dict(valid_data.sort_values(by='userId').groupby('userId')['movieId'].agg(list).reset_index().values)
    
    #do a training and testing split on the strong_generalization_set
    strong_generalization_set_val = strong_generalization_set.sample(frac=0.5,random_state=42)
    strong_generalization_set_test = strong_generalization_set.drop(strong_generalization_set_val.index)
    strong_generalization_set_val_items = dict(strong_generalization_set_val.sort_values(by='userId').groupby('userId')['movieId'].agg(list).reset_index().values)
    strong_generalization_set_test_items = dict(strong_generalization_set_test.sort_values(by='userId').groupby('userId')['movieId'].agg(list).reset_index().values)
    

    num_movies = max(set(train_data.movieId) | set(valid_data.movieId) | set(test_data.movieId)) + 1
    strong_generalization_set_val_dataloader = get_dataloaders(strong_generalization_set_val,rank,num_movies,world_size,args.bs)
    strong_generalization_set_test_dataloader = get_dataloaders(strong_generalization_set_test,rank,num_movies,world_size,args.bs)
    rec_dataloader = get_dataloaders(train_data,rank,num_movies,world_size,args.bs)
    test_dataloader = get_dataloaders(test_data,rank,num_movies,world_size,args.bs)
    val_dataloader = get_dataloaders(valid_data,rank,num_movies,world_size,args.bs)

   
    
    
    with open(f'./saved_user_summary/{args.data_name}/user_summary_gpt4_{debug_string}.json','r') as f:
        prompts = json.load(f)
        #make int 

    #sort prompt dict in ascending order 
    prompts = {int(float(k)): v for k, v in sorted(prompts.items(), key=lambda item: float(item[0]))}
    return prompts,rec_dataloader,num_movies,val_items,test_items,val_dataloader,test_dataloader, strong_generalization_set_val_dataloader, strong_generalization_set_test_dataloader,\
        strong_generalization_set_val_items,strong_generalization_set_test_items
        
    
def train_model(rank,world_size):
    setup(rank, world_size)
    start_time = time.time()
    
    if not args.debug:

        wandb.init(project='llm4rec', name=args.model_log_name)
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
        
        
            
    pprint(vars(args))        
    torch.cuda.set_device(rank)
    data = {
        'tensor': torch.ones(3,device=rank) + rank,
        'list': [1,2,3] ,
        'dict': {'rank':rank}   
    }
    
    # we have to create enough room to store the collected objects
    outputs = [None for _ in range(world_size)]
    # the first argument is the collected lists, the second argument is the data unique in each process
    dist.all_gather_object(outputs, data)
    # we only want to operate on the collected objects at master node
    if rank == 0:
        print(outputs)
        
    torch.cuda.set_device(rank)
  
    
    prompts,rec_dataloader,num_movies,val_items,test_items,val_dataloader,test_dataloader, strong_generalization_set_val_dataloader, strong_generalization_set_test_dataloader,strong_generalization_set_val_items,strong_generalization_set_test_items\
        = load_data(args,rank,world_size)
    #print number of batches in dataloader 
    print(f"Number of Batches = {len(rec_dataloader)}")
    model_name = f"{args.embedding_module}-large"
    tokenizer = T5Tokenizer.from_pretrained('t5-large')
    if args.embedding_module == 'sentence-t5':
        if args.lora: 

            lora_model = SentenceTransformer('sentence-transformers/sentence-t5-large')
            lora_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.1,
                                    target_modules=["q", "v"])
            lora_model = get_peft_model(lora_model, lora_config)
            
        print('LOADING SENTENCE T5')
        model = sentenceT5Classification(lora_model,num_movies,args)
        max_length = None

    elif args.embedding_module == 't5':
        model = T5ForSequenceClassification.from_pretrained(model_name, num_labels=num_movies)
        max_length = max([tokenizer.encode(text, return_tensors="pt").shape[1] for text in prompts.values()])
        if args.lora: 
            
            lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.1,
                                    target_modules=["q", "v"])
            model = get_peft_model(model, lora_config)
            for name, param in model.named_parameters():
                if 'classification_head' in name:
                    param.requires_grad = True
        
    else:
        # Freeze all the parameters # not in classifciation head
        for name,param in model.named_parameters() :
            if "classification_head" not in name:
                param.requires_grad = False

        # Unfreeze the last block of the decoder
        for param in model.transformer.decoder.block[-1].parameters():
            param.requires_grad = True

    if args.no_bias: 
        for name,param in model.named_parameters() :
            if "bias" in name:
                param.requires_grad = False 
    model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr, max_lr=args.lr*5,
    #                                         step_size_up=(args.epochs//4) *3, cycle_momentum=False)
    #use a cosine decay scheduler instead 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    #enable classification head gradients
    
    train_fun = train_fineTuning 

    min_val_recall = -np.inf
    patience_counter = 0
    last_saved_epoch = 0

    unfrozen = False
    for e in (pbar := tqdm(range(args.epochs))):
        model.train()
        model.share_memory()
        if e < args.warmup_mlp and not unfrozen:
            for name, param in model.module.named_parameters():  # Access the `module` attribute here
                if 'classification_head' not in name:  # 'classification_head' is the name of the classification layer
                    param.requires_grad = False
                else:
                    print(f"Unfreezing {name}")
           
        # Unfreeze all layers after the first 10 epochs
        elif e >= args.warmup_mlp and not unfrozen:
            # if lora is set then only unfreeze the lora layer
            if args.lora:
                for name, param in model.module.named_parameters():
                    if 'lora' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            else:
                for param in model.module.parameters():  # Access the `module` attribute here
                    param.requires_grad = True
                unfrozen = True  # Set the flag to True after unfreezing        
        train_fun(args, rec_dataloader, model, prompts, optimizer, scheduler, val_items,tokenizer,max_length,pbar,e,rank)
        if e % 2 == 0: 
            torch.cuda.set_device(rank)
            
            ranked_movies = generate_preds_fineTuning(args,model,prompts,val_dataloader,tokenizer,rank,20)
            val_recall = recall_at_k(val_items,ranked_movies , topk=20)
            val_ndcg = ndcg_k(val_items,ranked_movies , topk=20)
            strong_generalization_set_ranked_movies = generate_preds_fineTuning(args,model,prompts,strong_generalization_set_val_dataloader,tokenizer,rank,20)
            strong_generalization_set_val_recall = recall_at_k(strong_generalization_set_val_items,strong_generalization_set_ranked_movies , topk=20)
            strong_generalization_set_val_ndcg = ndcg_k(strong_generalization_set_val_items,strong_generalization_set_ranked_movies , topk=20)
            if not args.debug:
                wandb.log({'val_recall': val_recall,
                            'val_ndcg': val_ndcg,
                            "strong_generalization_set_val_recall": strong_generalization_set_val_recall,
                            "strong_generalization_set_val_ndcg": strong_generalization_set_val_ndcg,
                            'epoch': e})

            val_recall = 0
            val_ndcg=0
            if val_recall > min_val_recall:
                min_val_recall = val_recall
                torch.save(model.state_dict(), f'{args.scratch}/saved_model/{args.data_name}/{args.model_log_name}.pt')
                last_saved_epoch = e
                best_val = val_recall
                best_ndcg = val_ndcg
                if args.patience == patience_counter:
                    print(f"Early stopping training at epoch {e}")
                    break

            pbar.set_description(f"Epoch {e}: val_recall: {val_recall} ndcg: {val_ndcg} last_saved_epoch: {last_saved_epoch}")
           # model = T5ForSequenceClassification.from_pretrained(model_name, num_labels=num_movies)
           
        
    model.load_state_dict(torch.load(f'{args.scratch}/saved_model/{args.data_name}/{args.model_log_name}.pt'))
    model.to(rank)
    # setup(rank, world_size)
    # model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    model.eval()
    ranked_movies = generate_preds_fineTuning(args,model,prompts,test_dataloader,tokenizer,rank,20)

    test_recall = recall_at_k(test_items,ranked_movies , topk=20)
    test_ndcg = ndcg_k(test_items,ranked_movies , topk=20)
    
    if not args.debug:
        wandb.log({'test_recall': test_recall,
                   'test_ndcg': test_ndcg})
    print(f"test_recall: {test_recall}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time} seconds")
    
    
    log_results = pd.DataFrame({'model_name': [args.model_log_name],
                                    'test_recall': [test_recall],
                                    'test_ndcg': [test_ndcg],
                                    'val_recall': [best_val],
                                    'val_ndcg': [best_ndcg],
                                    'train_time_min': [elapsed_time / 60]}).round(3)
    
    
    csv_path = f"./model_logs/{args.data_name}/{args.model_name}.csv"

    # Check if the CSV file already exists
    if os.path.exists(csv_path):
        existing_data = pd.read_csv(csv_path)
        updated_data = pd.concat([existing_data, log_results], ignore_index=True)
        updated_data.to_csv(csv_path, index=False)
    else:
        log_results.to_csv(csv_path, index=False)
    
    cleanup()
    

if __name__ == "__main__":

    load_dotenv(".env")
    openai.api_key = os.getenv("OPEN-AI-SECRET")
   

    # Call the train_model function
    world_size = torch.cuda.device_count()
   
    
    mp.spawn(train_model,
             args = (world_size,),nprocs = world_size)

    # Record end time

    # Calculate the elapsed time

    # Print the elapsed time
    model_name = f"{args.embedding_module}-large"
    
    #evaluation: 
    
    # model = T5ForSequenceClassification.from_pretrained(model_name, num_labels=num_movies)
    # model.load_state_dict(torch.load(f'{args.scratch}/saved_model/{args.data_name}/{args.model_log_name}.pt'))
    # model.eval()
    # ranked_movies = generate_preds_fineTuning(args,model,prompts,tokenizer = tokenizer , max_l=max_length)

    # test_recall = recall_at_k(test_items,ranked_movies , topk=20)
    # test_ndcg = ndcg_k(test_items,ranked_movies , topk=20)
    
    

    
