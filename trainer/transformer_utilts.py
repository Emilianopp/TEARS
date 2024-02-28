import argparse
from typing import List
import torch.nn.functional as F
import sys
import torch.distributed as dist
import numpy as np
import json
import os 
import pandas as pd
from collections import defaultdict
from helper.eval_metrics import Recall_at_k_batch,NDCG_binary_at_k_batch,MRR_at_k
import wandb
from helper.dataloader import DataMatrix,MatrixDataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import sys 
sys.path.append('../')
from vae import data
import torch
from torch.nn.utils.rnn import pad_sequence
import pickle


def load_data(args,tokenizer= None, rank=0,world_size=1):
    data_path = f'./data_preprocessed/{args.data_name}/'
    
    loader = MatrixDataLoader(data_path)
    
    train_data = loader.load_data('train')
    vad_data_tr, valid_data = loader.load_data('validation')
    test_data_tr, test_data = loader.load_data('test')
    num_users = train_data.shape[0]
    num_movies = train_data.shape[1]

    nonzer_indeces_train = {i:v for i,v in enumerate(set(train_data.sum(axis =1 ).nonzero()[0]))}
    nonzer_indeces_valid = {i:v for i,v in enumerate(set(valid_data.sum(axis =1 ).nonzero()[0]))}
    nonzer_indeces_test = {i:v for i,v in enumerate(set(test_data.sum(axis = 1).nonzero()[0]))}

    with open (f'{data_path}/profile2id.pkl','rb') as f:
        profile2id = pickle.load(f)
    with open(f'./saved_user_summary/{args.data_name}/user_summary_gpt4_.json','r') as f:
        prompts = json.load(f)
    prompts = {profile2id[int(float(k))]:v for k,v in prompts.items() } 
    promp_list = [v for k,v in prompts.items()]
    max_l = max([len(i.split()) for i in promp_list])
    print("Max Prompt Length",max_l)
    encodings = {k: tokenizer([v],padding='max_length', return_tensors='pt',truncation=True,max_length=max_l) for k, v in sorted(prompts.items())} 
    encodings = {k: {k1: v1.squeeze(0) for k1, v1 in v.items()} for k, v in encodings.items()}
    print(f"Number of Users is {num_users=}")
    print(f"Number of Movies is {num_movies=}")
    
    
    rec_dataloader = get_dataloader(train_data,rank,world_size,args.bs,encodings,nonzer_indeces_train)
    val_dataloader = get_dataloader(valid_data,rank,world_size,args.bs,encodings,nonzer_indeces_valid)
    test_dataloader = get_dataloader(test_data,rank,world_size,args.bs,encodings,nonzer_indeces_test)

    return prompts,rec_dataloader,num_movies,val_dataloader,test_dataloader,vad_data_tr,test_data_tr


def get_dataloader(data,rank,world_size,bs,encodings,nonzer_indeces_train):
    rec_dataset =  DataMatrix(data,encodings,nonzer_indeces_train)
    sampler = DistributedSampler(rec_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    rec_dataloader = DataLoader(rec_dataset, batch_size=bs, collate_fn= None , num_workers = 0,pin_memory=False,
                                sampler = sampler) 
    return rec_dataloader

def get_loss(loss_name):
    if loss_name == 'bce':
        return torch.nn.BCELoss()
    elif loss_name == 'bce_softmax':
        return torch.nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError


def eval_model(model, test_dataloader, rank, test_data_tr,loss_f = None, mult_process=True, world_size=2):
    torch.cuda.set_device(rank)
    metrics = defaultdict(list)
    output_metrics = defaultdict(float)
    user_id_set = set()
    user_ids_l = []
    model.to(rank)
    rolling_loss = 0 

    with torch.no_grad():
        model.eval()
        for b, item in enumerate(test_dataloader):
            user_ids = sum(item.pop('idx').cpu().tolist(), [])
            user_id_set.update(user_ids)
            item = {k: v.to(rank) for k, v in item.items()}
            movie_emb_clean = model(**item)[0]
            rolling_loss += loss_f(movie_emb_clean, item['labels'].to(rank)).item()
            user_ids_l.append(user_ids)
            masked_rows = test_data_tr[user_ids].toarray()
            movie_emb_clean[np.where(masked_rows > 0)] = -torch.inf
            labels = item['labels'].cpu().numpy()
            recon = movie_emb_clean.cpu().numpy()
            metrics['ndcg@10'].append(NDCG_binary_at_k_batch(recon, labels, k=10).mean().tolist())
            metrics['ndcg@20'].append(NDCG_binary_at_k_batch(recon, labels, k=20).mean().tolist())
            metrics['ndcg@50'].append(NDCG_binary_at_k_batch(recon, labels, k=50).mean().tolist())
            metrics['mrr@10'].append(MRR_at_k(recon, labels, k=10))
            metrics['mrr@20'].append(MRR_at_k(recon, labels, k=20))
            metrics['mrr@50'].append(MRR_at_k(recon, labels, k=50))
            metrics['recall@10'].append(Recall_at_k_batch(recon, labels, k=10))
            metrics['recall@20'].append(Recall_at_k_batch(recon, labels, k=20))
            metrics['recall@50'].append(Recall_at_k_batch(recon, labels, k=50))

        if mult_process:
            ranked_movies_all, losses = [None for _ in range(world_size)], [None for _ in range(world_size)]
            dist.all_gather_object(ranked_movies_all, {})
            dist.all_gather_object(losses, rolling_loss / len(test_dataloader))
            for key in metrics.keys():
                gathered_metrics = [None for _ in range(world_size)]
                dist.all_gather_object(gathered_metrics, metrics[key])
                output_metrics[key] = np.concatenate(gathered_metrics).mean()

    return sum(losses) / len(losses),dict(output_metrics)


def train_fineTuning(args, rec_dataloader, model, optimizer, scheduler,pbar,epoch,rank):
    rec_dataloader.sampler.set_epoch(epoch)
    loss_f = get_loss(args.loss)
    scaler = torch.cuda.amp.GradScaler()
    rolling_loss = []

    for b,items in enumerate(rec_dataloader):
            model.train()
            optimizer.zero_grad(set_to_none = True )
            movie_emb = model(input_ids = items['input_ids'], attention_mask = items['attention_mask'])[0]
            loss = loss_f( movie_emb,items['labels'].to(rank))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            rolling_loss.append( loss.item())
            scaler.update()
            torch.cuda.empty_cache() 
            pbar.set_description(f"Epoch {epoch}: Batch: {b} loss: {np.mean(rolling_loss)} current lr: {scheduler.get_last_lr()[0]}")
            break
            
    if not args.debug  :

        wandb.log({'loss': np.mean(rolling_loss),
                    "epoch": epoch,
                    "total_steps": epoch*len(rec_dataloader) + b,
                    'lr': scheduler.get_last_lr()[0],
                    })
        
    pbar.set_description(f"Batch {epoch}: loss: {np.mean(rolling_loss)} current lr: {scheduler.get_last_lr()[0]}")
    scheduler.step()



def get_scheduler(optimizer,args):
    scheduler = args.scheduler
    if scheduler == 'linear_decay':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 - epoch / args.epochs)
    elif scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= args.epochs)
    elif scheduler == 'cosine_warmup':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0= args.epochs)
    else:
        raise NotImplementedError
    

def parse_args(notebook = False):  # Parse command line arguments
    parser = argparse.ArgumentParser(description="LLM4RecSys")
    parser.add_argument("--data_name", default='ml-1m', type=str)
    parser.add_argument("--log_file", default= 'model_logs/ml-100k/logging_llmMF.csv', type=str)
    parser.add_argument("--model_name", default='Transformer', type=str)
    parser.add_argument("--emb_type", default='attn', type=str)
    parser.add_argument("--summary_style", default='topM', type=str)
    parser.add_argument("--embedding_module", default='t5', type=str)
    parser.add_argument("--scheduler", default='linear_decay', type=str)
    parser.add_argument("--embedding_dim" , default=1536, type=int)
    parser.add_argument("--bs" , default=8, type=int)
    parser.add_argument("--patience" , default=100, type=int)
    parser.add_argument("--output_emb" , default=256, type=int)
    parser.add_argument("--top_for_rerank" , default=50, type=int)
    parser.add_argument("--num_layers" , default=3, type=int)
    parser.add_argument("--num_layers_transformer" , default=3, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--topk", default=20, type=int)

    parser.add_argument("--attention_emb", default=512, type=int)
    parser.add_argument("--train", default=True, type=bool)
    parser.add_argument("--num_neg", default=1, type=int)
    parser.add_argument("--num_heads", default=4, type=int)
    parser.add_argument("--lr", default=.0001, type=float)
    parser.add_argument("--l2_lambda", default=.0001, type=float)
    parser.add_argument("--dropout", default=.1, type=float)
    parser.add_argument("--temp", default=2, type=float)


    parser.add_argument("--wd", default=0, type=float)
    parser.add_argument('--make_embeddings', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debugger', action='store_true')
    parser.add_argument('--cosine', action='store_true')
    parser.add_argument('--make_augmented', action='store_true')
    parser.add_argument('--neg_sample', action='store_true')
    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--make_data', action='store_true')
    parser.add_argument('--loss', default='bce_softmax', type=str, choices=['bce','bce_softmax'])
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--total_steps", type=int, default=1)
    parser.add_argument("--beta", type=float, default=1)
    
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--warmup_mlp", type=int, default=10)
    parser.add_argument('--no_bias', action='store_true')
    parser.add_argument('--debug_prompts', action='store_true')

    args = parser.parse_args() if not notebook else parser.parse_args(args=[])
    args.recon = False
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    args.model_save_path = f'/home/mila/e/emiliano.penaloza/scratch/saved_model/ml-100k/{args.model_name}'


    args.model_save_name = f"{args.model_save_path}/best_model_lr_{args.lr}_embedding_module_{args.embedding_module}_epochs_{args.epochs}_l2_lambda_{args.l2_lambda}_lora_alpha_{args.lora_alpha}_lora_r_{args.lora_r}_{args.loss}__bias_{args.no_bias}.pth"
    print(f"MODEL NAME = {args.model_save_name}")

    args.model_log_name = f"{args.model_name}_embedding_module_{args.embedding_module}_l2_lambda_{args.l2_lambda}_lora_r_{args.lora_r}_scheduler_{args.scheduler}_{args.lr}_{args.dropout}.csv"
    
    directory_path = "./scratch"

    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        print(f"The directory '{directory_path}' exists. Will save all weights and models there")
        args.scratch = './scratch'
        
    else:
        args.scratch = '.'
    args.scratch = '/home/mila/e/emiliano.penaloza/scratch'

    return args