import random 
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
from trainer.losses.loss import get_loss
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm 





def load_data(args,tokenizer= None, rank=0,world_size=1):
    data_path = f'./data_preprocessed/{args.data_name}/'
    loader = MatrixDataLoader(data_path)
    
    train_data,non_binary_data = loader.load_data('train')
    vad_data_tr, valid_data,_,_ = loader.load_data('validation')
    test_data_tr, test_data,_,_ = loader.load_data('test')
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
        #add an <eos> token to the end of the prompt
        # for k,v in prompts.items():
        #     prompts[k] = v + ' <eos>'
            
    promp_list = [v for k,v in prompts.items()]
    max_l = max([len(i.split()) for i in promp_list])
    max_token_l = max([len(tokenizer.encode(v)) for v in promp_list])
    
    print("Max Prompt Length",max_l)
    encodings = {k: tokenizer([v],padding='max_length', return_tensors='pt',truncation=True,max_length=max_token_l) for k, v in sorted(prompts.items())} 

    encodings = {k: {k1: v1.squeeze(0) for k1, v1 in v.items()} for k, v in encodings.items()}
    print(f"Number of Users is {num_users=}")
    print(f"Number of Movies is {num_movies=}")
    
    #half the batch if we are doubleing the training data by making sure the original is in there
    rec_dataloader = get_dataloader(train_data,rank,world_size,args.bs//2 if  args.masked_and_original else args.bs,encodings,nonzer_indeces_train, None,prompts,args.mask)
    augmented_dataloader = get_dataloader(train_data,rank,world_size,args.bs//2 if  args.masked_and_original else args.bs,encodings,nonzer_indeces_train, tokenizer,prompts,args.mask)
    val_dataloader = get_dataloader(valid_data,rank,world_size,args.bs,encodings,nonzer_indeces_valid)
    test_dataloader = get_dataloader(test_data,rank,world_size,args.bs,encodings,nonzer_indeces_test)
    non_bin_dataloader = get_dataloader(non_binary_data,rank,world_size,args.bs,encodings,nonzer_indeces_train)
    

    return prompts,rec_dataloader,augmented_dataloader,num_movies,val_dataloader,test_dataloader,vad_data_tr,test_data_tr,non_bin_dataloader


def load_data_items_as_tokens(args,tokenizer= None, rank=0,world_size=1):
    data_path = f'./data_preprocessed/books/'
    print(f"{data_path=}")

    loader = MatrixDataLoader(data_path)

    train_data,non_binary_data = loader.load_data('train')
    vad_data_tr, valid_data,rating_vad_data_tr, rating_valid_data = loader.load_data('validation')
    test_data_tr, test_data,rating_test_data_tr, rating_test_data = loader.load_data('test')

    num_users = train_data.shape[0]
    num_movies = train_data.shape[1]
    #binarize non_binary_data
    rating_all = non_binary_data + rating_vad_data_tr + rating_test_data_tr

    nonzer_indeces_train = {i:v for i,v in enumerate(set(train_data.sum(axis =1 ).nonzero()[0]))}
    nonzer_indeces_valid = {i:v for i,v in enumerate(set(valid_data.sum(axis =1 ).nonzero()[0]))}
    nonzer_indeces_test = {i:v for i,v in enumerate(set(test_data.sum(axis = 1).nonzero()[0]))}

    with open (f'{data_path}/profile2id.pkl','rb') as f:
        profile2id = pickle.load(f)
    with open(f'./saved_user_summary/{args.data_name}/user_summary_gpt4_.json','r') as f:
        prompts = json.load(f)
        

    prompts = {profile2id[int(float(k))]:v for k,v in prompts.items() } 
    new_tokens = [f'<item_id_{i}>' for i in range(num_movies)] + [f'<rating_{k}>' for k in range(1,6)]

    tokenizer.add_tokens(new_tokens)
    for u in range(num_users): 
        nonzero_columns = set(rating_all[u].nonzero()[1])
        #make the prompt with the tokens 
        prompts[u] = ''
        for i in nonzero_columns:
            rating = rating_all[u,i]
            prompts[u] += f' <item_id_{i}> <rating_{int(rating)}>\n'
 

    promp_list = [v for k,v in prompts.items()]
    max_l = max([len(i.split()) for i in promp_list])
    #calculate the max_token_length by tokenizing all the prompts 
    max_token_l = max([len(tokenizer.encode(v)) for v in promp_list])
    encodings = {k: tokenizer([v],padding='max_length', return_tensors='pt',truncation=True,max_length=max_token_l ) for k, v in sorted(prompts.items())} 

    encodings = {k: {k1: v1.squeeze(0) for k1, v1 in v.items()} for k, v in encodings.items()}
    print(f"Number of Users is {num_users=}")
    print(f"Number of Movies is {num_movies=}")


    rec_dataloader = get_dataloader(train_data,rank,world_size,args.bs,encodings,nonzer_indeces_train)
    val_dataloader = get_dataloader(valid_data,rank,world_size,args.bs,encodings,nonzer_indeces_valid)
    test_dataloader = get_dataloader(test_data,rank,world_size,args.bs,encodings,nonzer_indeces_test)
    non_bin_dataloader = get_dataloader(non_binary_data,rank,world_size,args.bs,encodings,nonzer_indeces_train)

    return prompts,rec_dataloader,num_movies,val_dataloader,test_dataloader,vad_data_tr,test_data_tr,non_bin_dataloader,tokenizer



def get_dataloader(data,rank,world_size,bs,encodings,nonzer_indeces_train,tokenizer=None,prompts=None,mask=None):
    rec_dataset =  DataMatrix(data,encodings,nonzer_indeces_train,tokenizer,prompts,mask)
    sampler = DistributedSampler(rec_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    rec_dataloader = DataLoader(rec_dataset, batch_size=bs, collate_fn= rec_dataset.custom_collator if tokenizer is not None else None  , num_workers = 0,pin_memory=False,
                                sampler = sampler) 
    return rec_dataloader


def get_embeddings(model, dataloader, rank, world_size, num_movies, tokenizer, save_path, save_name, save=True):
    model.eval()
    if os.path.exists(os.path.join(save_path, save_name)):
        with open(save_path + '/' +save_name , 'rb') as f:

            embeddings = pickle.load(f)

    else:

        model.eval()
        embeddings = {}
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Making embeddings"):
                input_ids = batch['input_ids'].to(rank)
                attention_mask = batch['attention_mask'].to(rank)
                idx = batch['idx']
                batch_embeddings = model.llm_forward(input_ids=input_ids, attention_mask=attention_mask)

                for i in range(len(idx)):
                    embeddings[idx[i].item()] = batch_embeddings[i].float().cpu().numpy()


        if world_size > 1:
            embeddings = dist.gather(embeddings, dst=0)
            if rank == 0:
                embeddings = {k: v for emb in embeddings for k, v in emb.items()}

        if rank == 0 and save:
            os.makedirs(save_path, exist_ok=True)

            with open(os.path.join(save_path, save_name), 'wb') as f:
                pickle.dump(embeddings, f)

    return embeddings

def eval_model(model, test_dataloader, rank, test_data_tr,precompute_embeddings,loss_f = None, mult_process=True, world_size=2):
    torch.cuda.set_device(rank)
    metrics = defaultdict(list)
    output_metrics = defaultdict(float)
    user_id_set = set()
    user_ids_l = []
    model.to(rank)
    rolling_loss = 0 
    metrics = defaultdict(list)
    with torch.no_grad():
        model.eval()
        for b, item in enumerate(test_dataloader):
            user_ids = sum(item.pop('idx').cpu().tolist(), [])
            user_id_set.update(user_ids)
            item = {k: v.to(rank) for k, v in item.items()}
            if precompute_embeddings is not None: 

                hidden_states = [precompute_embeddings[idx] for idx in user_ids]
                hidden_states = torch.tensor(hidden_states).to(rank).bfloat16()
                movie_emb_clean = model.module.classifier_forward(hidden_states)
            else:
                movie_emb_clean = model(**item)[0]
            masked_rows = test_data_tr[user_ids].toarray()
            movie_emb_clean[np.where(masked_rows > 0)] = np.log(.0001)

            rolling_loss += loss_f(movie_emb_clean, item['labels'].to(rank)).item()
            # movie_emb_clean[np.where(masked_rows > 0)] = -torch.inf
            user_ids_l.append(user_ids)
            labels = item['labels'].cpu().numpy()
            recon = movie_emb_clean.float().cpu().numpy()
            k_values = [10, 20, 50]

            for k in k_values:
                metrics[f'ndcg@{k}'].append(NDCG_binary_at_k_batch(recon, labels, k=k).tolist())

                metrics[f'mrr@{k}'].append(MRR_at_k(recon, labels, k=k,mean = False).tolist())
                metrics[f'recall@{k}'].append(Recall_at_k_batch(recon, labels, k=k,mean = False).tolist())





        
        if mult_process:
            ranked_movies_all, losses = [None for _ in range(world_size)], [None for _ in range(world_size)]
            dist.all_gather_object(ranked_movies_all, {})
            dist.all_gather_object(losses, rolling_loss / len(test_dataloader))
            for key in metrics.keys():
                gathered_metrics = [None for _ in range(world_size)]
                dist.all_gather_object(gathered_metrics, metrics[key])

                output_metrics[key] = np.mean(sum(sum(gathered_metrics,[]),[]))
            losses = sum(losses) / len(losses)
        
        else: 
            for key in metrics.keys():
                output_metrics[key] = np.mean(metrics[key])
                losses = rolling_loss / len(test_dataloader)
            
    return losses,dict(output_metrics)





def mask_tokens_with_attention(input_ids, attention_mask, eos_token_id, tokenizer):
    # Define the mask token ID using a token unlikely to appear in your data, like <extra_id_0>
    mask_token_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    
    # Calculate the number of tokens to mask in each sequence
    # Excluding EOS and padding from the masking candidates
    valid_tokens_mask = (input_ids != eos_token_id) & (attention_mask == 1)
    num_valid_tokens = valid_tokens_mask.sum(dim=1)
    # Ensure at least one token is available for masking to avoid zero division
    num_valid_tokens = torch.clamp(num_valid_tokens, min=1)
    num_to_mask = (num_valid_tokens.float() * torch.rand(num_valid_tokens.size()) * 0.2).floor().long()

    # Prepare a masked input copy
    masked_input_ids = input_ids.clone()

    # For each sequence, randomly select tokens to mask
    for i in range(input_ids.size(0)):
        valid_indices = valid_tokens_mask[i].nonzero().view(-1)
        indices_to_mask = valid_indices[torch.randperm(valid_indices.size(0))[:num_to_mask[i]]]
        masked_input_ids[i, indices_to_mask] = mask_token_id

    return masked_input_ids

def pad_tensors(tensor1,tensor2,pad_id):
    max_len = max(tensor1.size(1),tensor2.size(1))
    tensor1 = F.pad(tensor1, (0, max_len - tensor1.size(1)), value=pad_id)
    tensor2 = F.pad(tensor2, (0, max_len - tensor2.size(1)), value=pad_id)
    
    return torch.cat([tensor1,tensor2])

def train_fineTuning(args, rec_dataloader,augmented_dataloader, model, optimizer, scheduler,pbar,epoch,rank,tokenizer,precompute_embeddings):
    rec_dataloader.sampler.set_epoch(epoch)
    loss_f = get_loss(args.loss)
    scaler = torch.cuda.amp.GradScaler()
    rolling_loss = []

    for b, (items,augmented_items) in enumerate(zip(rec_dataloader,augmented_dataloader)):
        model.train()
        
        # with torch.cuda.amp.autocast():
            
        if args.mask == 0: 
            labels = items['labels']
            input_ids = items['input_ids']
            attention_mask = items['attention_mask']
        else:
            input_ids = pad_tensors(items['input_ids'],augmented_items['input_ids'],tokenizer.pad_token_id)
            labels = torch.cat((items['labels'],augmented_items['labels']))
            attention_mask = pad_tensors(items['attention_mask'], augmented_items['attention_mask'], 0)
        
        
        if precompute_embeddings is not None: 

            hidden_states = [precompute_embeddings[idx.item()] for idx in items['idx'].cpu().numpy()]
            hidden_states = torch.tensor(hidden_states).to(rank).bfloat16()
            movie_emb = model.module.classifier_forward(hidden_states)
        

        else: 
            movie_emb = model(input_ids=input_ids, attention_mask=attention_mask)[0]
        loss = loss_f(movie_emb, labels.to(rank)) / args.update_every
    
        loss.backward()
        
        if (b + 1) % args.update_every == 0:
            optimizer.step()
            # scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        rolling_loss.append(loss.item() * args.update_every)

        torch.cuda.empty_cache() 
        pbar.set_description(f"Epoch {epoch}: Batch: {b} loss: {np.mean(rolling_loss)} current lr: {scheduler.get_last_lr()[0]}")
         
            
    if not args.debug  :

        wandb.log({'loss': np.mean(rolling_loss),
                    "epoch": epoch,
                    "total_steps": epoch*len(rec_dataloader) + b,
                    'lr': scheduler.get_last_lr()[0],
                    })
        
    pbar.set_description(f"Batch {epoch}: loss: {np.mean(rolling_loss)} current lr: {scheduler.get_last_lr()[0]}")
    scheduler.step()

def train_distill(args, rec_dataloader, model, optimizer, scheduler,pbar,epoch,rank,non_binary_data,teacher):
    rec_dataloader.sampler.set_epoch(epoch)

    
    loss_f = get_loss('kl',args)
    scaler = torch.cuda.amp.GradScaler()
    rolling_loss = []
    kl_roll = []
    

    for b,(items,non_bin_items) in enumerate(zip(rec_dataloader,non_binary_data)):
            model.train()

            labels = items['labels'].to(rank)
            optimizer.zero_grad(set_to_none = True )
            movie_emb = model(input_ids = items['input_ids'], attention_mask = items['attention_mask'])[0]
            #mask half of the labels which are one 
            teacher_logits = teacher(non_bin_items['labels'])
            total_loss,kl,loss = loss_f(movie_emb,items['labels'].to(rank),teacher_logits)
            loss_f.anneal_beta(epoch, args.epochs)
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            rolling_loss.append( loss.item())
            kl_roll.append(kl.item())
            scaler.update()
            torch.cuda.empty_cache() 
            pbar.set_description(f"Epoch {epoch}: Batch: {b} loss: {np.mean(rolling_loss)} current lr: {scheduler.get_last_lr()[0]}")
            
    if not args.debug  :

        wandb.log({'loss': np.mean(rolling_loss),
                   'kl_loss': np.mean(kl_roll),
                    "epoch": epoch,
                    "total_steps": epoch*len(rec_dataloader) + b,
                    'lr': scheduler.get_last_lr()[0],
                    })
        
    pbar.set_description(f"Batch {epoch}: loss: {np.mean(rolling_loss)} current lr: {scheduler.get_last_lr()[0]}")
    scheduler.step()

def get_eos_token(prompts):
    # Find the indices of non-zero elements
    non_zero_indices = torch.nonzero(prompts, as_tuple=True)[0]

    # Select the last non-zero element
    eos_token = prompts[non_zero_indices[-1]]

    return eos_token

def get_scheduler(optimizer,args):
    scheduler = args.scheduler
    if scheduler == 'linear_decay':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 - epoch / args.epochs)
    elif scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= args.epochs)
    elif scheduler == 'cosine_warmup':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0= args.epochs)
    elif scheduler == 'None':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1)
    else:
        raise NotImplementedError
    

def parse_args(notebook = False):  # Parse command line arguments
    parser = argparse.ArgumentParser(description="LLM4RecSys")
    parser.add_argument("--data_name", default='ml-1m', type=str)
    parser.add_argument("--log_file", default= 'model_logs/ml-100k/logging_llmMF.csv', type=str)
    parser.add_argument("--model_name", default='Transformer', type=str)
    parser.add_argument("--emb_type", default='attn', type=str)

    parser.add_argument("--save_teacher_path", default='./vae/', type=str)
    parser.add_argument("--embedding_module", default="microsoft/phi-2", type=str)
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
    parser.add_argument("--update_every", default=16, type=int)
    parser.add_argument("--attention_emb", default=512, type=int)
    parser.add_argument("--train", default=True, type=bool)
    parser.add_argument("--num_neg", default=1, type=int)
    parser.add_argument("--num_heads", default=4, type=int)
    parser.add_argument("--lr", default=.0001, type=float)
    parser.add_argument("--lr2", default=.00001, type=float)
    parser.add_argument("--l2_lambda", default=.0001, type=float)
    parser.add_argument("--mask", default=0 , type=float)
    parser.add_argument("--dropout", default=.1, type=float)
    parser.add_argument("--temp", default=2, type=float)
    parser.add_argument("--anneal", default=True, type=bool)
    parser.add_argument("--masked_and_original", action = 'store_true')
    parser.add_argument("--wd", default=0, type=float)
    parser.add_argument('--make_embeddings', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debugger', action='store_true')
    parser.add_argument('--cosine', action='store_true')
    parser.add_argument('--make_augmented', action='store_true')
    parser.add_argument('--neg_sample', action='store_true')
    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--make_data', action='store_true')
    parser.add_argument('--loss', default='bce_softmax', type=str, choices=['bce','bce_softmax','kl'])
    parser.add_argument('--pooling', default='eos_token', type=str, choices=['weighted_mean','eos_token','mean','bos_token'])
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--total_steps", type=int, default=1)
    parser.add_argument("--beta", type=float, default=1)
    
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument('--no_bias', action='store_true')
    parser.add_argument('--debug_prompts', action='store_true')

    args = parser.parse_args() if not notebook else parser.parse_args(args=[])
    args.recon = False
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    args.model_save_path = f'/home/mila/e/emiliano.penaloza/scratch/saved_model/{args.data_name}/{args.model_name}'

    
    args.model_save_name = f"{args.model_save_path}/best_model_lr_{args.lr}_embedding_module_{args.embedding_module.replace('/','_')}_epochs_{args.epochs}_l2_lambda_{args.l2_lambda}_lora_alpha_{args.lora_alpha}_lora_r_{args.lora_r}_{args.loss}__bias_{args.no_bias}.pth"
    print(f"MODEL NAME = {args.model_save_name}")

    args.model_log_name = f"{args.model_name}_embedding_module_{args.embedding_module.replace('/','_')}_l2_lambda_{args.l2_lambda}_lora_r_{args.lora_r}_scheduler_{args.scheduler}_{args.lr}_{args.dropout}.csv"
    print(f"{args.model_log_name=}")

    
    directory_path = "./scratch"

    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        print(f"The directory '{directory_path}' exists. Will save all weights and models there")
        args.scratch = './scratch'
        
    else:
        args.scratch = '.'
    args.scratch = '/home/mila/e/emiliano.penaloza/scratch'

    return args