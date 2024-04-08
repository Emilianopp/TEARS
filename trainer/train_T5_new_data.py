import sys 
sys.path.append("..")
import torch.optim as optim
from peft import get_peft_model
from tqdm import tqdm
import pandas as pd
import time 
import torch
import os
from dotenv import load_dotenv
from model.MF import sentenceT5Classification,sentenceT5ClassificationFrozen
from helper.dataloader import *
import wandb
from tqdm import tqdm
from trainer.transformer_utilts import *
from peft import LoraConfig, TaskType
from transformers import T5Tokenizer
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


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
        tags = [args.data_name]
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
    

    tokenizer = T5Tokenizer.from_pretrained('t5-large')
    prompts,rec_dataloader,num_movies,val_dataloader,test_dataloader,val_data_tr,test_data_tr,non_bin_dataloader= load_data(args,tokenizer,rank,world_size)
    model_name = f"{args.embedding_module}-large"
    
    if args.embedding_module == 't5_classification':
        model = sentenceT5Classification.from_pretrained('t5-large', num_labels=num_movies, classifier_dropout = args.dropout)
        lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.dropout,
                                target_modules=["q", "v"],
                                modules_to_save=['classification_head'])
        model = get_peft_model(model, lora_config)
    elif args.embedding_module == 't5_frozen':
        model = sentenceT5ClassificationFrozen.from_pretrained('t5-large', num_labels=num_movies, classifier_dropout = args.dropout)
        for name, param in model.named_parameters():
            if 'classification_head' not in name:
                param.requires_grad = False
                
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
    val_recall = 0
    
    for e in (pbar := tqdm(range(args.epochs))):
        model.train()
        model.share_memory()
        train_fineTuning(args, rec_dataloader, model, optimizer, scheduler,pbar,e,rank)
        if e % 1 == 0: 
            torch.cuda.set_device(rank)

            val_loss,outputs = eval_model(model,val_dataloader,rank,val_data_tr,loss_f = loss_f,world_size = world_size)
            if not args.debug :
                wandb.log({'val_ndcg@20': outputs['ndcg@20'],
                            'val_recall@20': outputs['recall@20'],
                            'val_ndcg@50': outputs['ndcg@50'],
                            'val_recall@50': outputs['recall@50'],
                            'validation_loss': val_loss,
                            'epoch': e})

            if outputs['ndcg@50'] > min_val_recall:
                min_val_recall = outputs['ndcg@50']
                torch.save(model.module.state_dict(), f'{args.scratch}/saved_model/{args.data_name}/{args.model_log_name}.pt')
                last_saved_epoch = e
                wandb.log({'ndcg@50_max': min_val_recall})
                wandb.log({'last_saved_epoch': last_saved_epoch})
                print(f"Early stopping training at epoch {e}")
                if args.patience == patience_counter:
                    print(f"Early stopping training at epoch {e}")
                    
             

            pbar.set_postfix({'val_loss': val_loss,'val_ndcg': outputs['ndcg@50'],'recall': outputs['recall@50'],'last_saved_epoch': last_saved_epoch})
            
    model.module.load_state_dict(torch.load(f'{args.scratch}/saved_model/{args.data_name}/{args.model_log_name}.pt'))
    model.to(rank)

    val_loss,val_outputs =eval_model(model,val_dataloader,rank,val_data_tr,loss_f = loss_f,world_size = world_size)

    test_loss,test_outputs = eval_model(model,test_dataloader,rank,test_data_tr,loss_f = loss_f,world_size = world_size)
    
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time} seconds")


    log_results = pd.DataFrame({'model_name': [args.model_log_name],
                                    'test_recall@10': [test_outputs['recall@10']],
                                    'test_recall@20': [test_outputs['recall@20']],
                                    'test_recall@50': [test_outputs['recall@50']],
                                    'test_ndcg@20': [test_outputs['ndcg@10']],
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
    
    

    
