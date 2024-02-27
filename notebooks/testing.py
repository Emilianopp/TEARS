# %%

import sys 
sys.path.append("..")
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
from trainer.training_utils import *
import wandb
from tqdm import tqdm
import math
from trainer.transformer_utilts import *
from torch.nn.parallel import DataParallel
from peft import LoraConfig, TaskType
from transformers import T5ForSequenceClassification
#import mp 
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import hashlib
from model.MF import T5MappingVAE

#set export MASTER_ADDR="127.0.0.1" as in os 
os.environ['MASTER_ADDR'] = '127.0.0.1'

#same with master port  export MASTER_PORT=$(expr 10000)

os.environ['MASTER_PORT'] = '10000'

    


args = parse_args(notebook=True)

debug_string  = "debug" if args.debug else ""
args.debug_prompts = True  
args.loss = 'bce_softmax'

if args.debugger:
    import debugpy
    debugpy.listen(5678)
    print('attach please')
    debugpy.wait_for_client()




# %%



def get_dataloaders(data,rank,num_movies,world_size,bs,encodings):
    rec_dataset = RecDatasetNegatives(data,num_negatives=1) if args.neg_sample else RecDatasetFull(data,encodings,num_movies=num_movies)
    sampler = DistributedSampler(rec_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    rec_dataloader = DataLoader(rec_dataset, batch_size=bs, collate_fn=(custom_collate if args.neg_sample else None) , num_workers = 0,pin_memory=False,
                                sampler = sampler) 
    return rec_dataloader
def load_data(args,tokenizer, rank=1,world_size=1):
    # 1. Data Loading & Preprocessing

    train_data = pd.read_csv(f'../data_preprocessed/{args.data_name}/train_leave_one_out_timestamped.csv')
    valid_data = pd.read_csv(f'../data_preprocessed/{args.data_name}/validation_leave_one_out_timestamped.csv')
    test_data = pd.read_csv(f'../data_preprocessed/{args.data_name}/test_leave_one_out_timestamped.csv')
    strong_generalization_set = pd.read_csv(f'../data_preprocessed/{args.data_name}/strong_generalization_set_timestamped.csv')
    
    training_items = dict(train_data.sort_values(by='userId').groupby('userId')['movieId'].agg(list).reset_index().values)
    test_items = dict(test_data.sort_values(by='userId').groupby('userId')['movieId'].agg(list).reset_index().values)

    val_items = dict(valid_data.sort_values(by='userId').groupby('userId')['movieId'].agg(list).reset_index().values)
    
    #do a training and testing split on the strong_generalization_set
    strong_generalization_set_val = strong_generalization_set.sample(frac=0.5,random_state=42)
    strong_generalization_set_test = strong_generalization_set.drop(strong_generalization_set_val.index)
    strong_generalization_set_val_items = dict(strong_generalization_set_val.sort_values(by='userId').groupby('userId')['movieId'].agg(list).reset_index().values)
    strong_generalization_set_test_items = dict(strong_generalization_set_test.sort_values(by='userId').groupby('userId')['movieId'].agg(list).reset_index().values)
    
    with open(f'../saved_user_summary/{args.data_name}/user_summary_gpt4_.json','r') as f:
        prompts = json.load(f)
        #make int 

    #sort prompt dict in ascending order 
    promp_list = [v for k,v in prompts.items()]
    max_l = max([len(i) for i in promp_list])
    print("Max Prompt Length",max_l)
    if not args.debug_prompts:
        prompts = {int(float(k)): v for k, v in sorted(prompts.items(), key=lambda item: float(item[0]))} 
        encodings = tokenizer(promp_list,padding=True, truncation=True,max_length=max_l)
    else: 
        input_ids = [[i,1] for i in range(len(prompts)) if i % 10 == 0]
        decoderAttention = [[1,1] for i in range(len(prompts)) if i % 10 == 0]
        encodings = {'input_ids':input_ids,'attention_mask':decoderAttention}


    
    
    
    # if args.debug_prompts take a subset of 10% of the users and run the test on those
    if args.debug_prompts:
 
        #get the same subset of users for val test and train sets 
        train_data = train_data[train_data.userId % 10 == 0]
        valid_data = valid_data[valid_data.userId % 10 == 0]
        test_data = test_data[test_data.userId % 10 == 0]
        strong_generalization_set_val = strong_generalization_set_val[strong_generalization_set_val.userId % 10 == 0]
        strong_generalization_set_test = strong_generalization_set_test[strong_generalization_set_test.userId % 10 == 0]

    

    num_movies = max(set(train_data.movieId) | set(valid_data.movieId) | set(test_data.movieId)) + 1
    strong_generalization_set_val_dataloader = get_dataloaders(strong_generalization_set_val,rank,num_movies,world_size,args.bs,encodings)
    strong_generalization_set_test_dataloader = get_dataloaders(strong_generalization_set_test,rank,num_movies,world_size,args.bs,encodings)
    rec_dataloader = get_dataloaders(train_data,rank,num_movies,world_size,args.bs,encodings)
    test_dataloader = get_dataloaders(test_data,rank,num_movies,world_size,args.bs,encodings)
    val_dataloader = get_dataloaders(valid_data,rank,num_movies,world_size,args.bs,encodings)
 
    return prompts,rec_dataloader,num_movies,training_items, val_items,test_items,val_dataloader,test_dataloader, strong_generalization_set_val_dataloader, strong_generalization_set_test_dataloader,\
        strong_generalization_set_val_items,strong_generalization_set_test_items


# %%
start_time = time.time()


args.bs = 128
args.l2_lambda = 0
args.debug = False



tokenizer = T5Tokenizer.from_pretrained('t5-base')
prompts,rec_dataloader,num_movies,training_items,val_items,test_items,val_dataloader,test_dataloader, strong_generalization_set_val_dataloader, strong_generalization_set_test_dataloader,strong_generalization_set_val_items,strong_generalization_set_test_items\
    = load_data(args,tokenizer,0,1)

# %%
def get_teacher_values(teacher,dataloader,rank):
    with torch.no_grad():
        for b in dataloader:
            inputs = b['labels'].to(rank)
            outputs = teacher(inputs)[0].cpu()
            recall = Recall_at_k_batch(outputs,b['labels'],k=10)
    return recall

# %%

#print number of batches in dataloader 
args.lora_r = 32
print(f"Number of Batches = {len(rec_dataloader)}")
model_name = f"{args.embedding_module}-large"
if args.embedding_module == 'sentence-t5':
    if args.lora: 

        lora_model = SentenceTransformer('sentence-transformers/sentence-t5-base')
        lora_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0,
                                target_modules=["q", "v"])
        lora_model = get_peft_model(lora_model, lora_config)
        
    print('LOADING SENTENCE T5')
    model = sentenceT5Classification(lora_model,num_movies,args)
    max_length = None

elif args.embedding_module == 't5':
    model =  T5MappingVAE.from_pretrained('t5-large', num_labels=num_movies)

    lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0,
                            target_modules=["q", "v"],
                            modules_to_save=['classification_head'])
    model = get_peft_model(model, lora_config)

model.to(rank)
model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

#get a subset of model parameters that require grad to feed into the optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.AdamW(params, lr=.5, weight_decay=.00)

#scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.05, total_iters=1/)
#do a decay scheduler 
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))
#enable classification head gradients


# %%
teacher = torch.load("/home/mila/e/emiliano.penaloza/LLM4REC/notebooks/vae-cf-pytorch/model.pt").to(rank)

teacher.eval()
teacher_recall = get_teacher_values(teacher,strong_generalization_set_val_dataloader,rank)
print(f"{teacher_recall=}")

