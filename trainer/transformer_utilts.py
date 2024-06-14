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
import datetime
update_count =0



 

def load_data(args,tokenizer= None, rank=0,world_size=1,prompt_p = None ):
    data_path = f'/home/mila/e/emiliano.penaloza/LLM4REC/data_preprocessed/{args.data_name}/'
    if prompt_p is None:
        prompt_p = f'./saved_user_summary/{args.data_name}/user_summary_gpt4_.json'
    

    loader = MatrixDataLoader(data_path,args)
    
    train_data_tr,train_data = loader.load_data('train')
    valid_data_tr, valid_data = loader.load_data('validation')
    test_data_tr, test_data = loader.load_data('test')
    num_users = train_data.shape[0]
    num_movies = train_data.shape[1]

    nonzer_indeces_train = {i:v for i,v in enumerate(set(train_data.sum(axis =1 ).nonzero()[0]))}
    nonzer_indeces_valid = {i:v for i,v in enumerate(set(valid_data.sum(axis =1 ).nonzero()[0]))}
    nonzer_indeces_test = {i:v for i,v in enumerate(set(test_data.sum(axis = 1).nonzero()[0]))}

    with open (f'{data_path}/profile2id.pkl','rb') as f:
        profile2id = pickle.load(f)
    with open(prompt_p,'r') as f:
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
    rec_dataloader = get_dataloader(train_data,train_data_tr,rank,world_size,args.bs//2 if  args.masked_and_original else args.bs,encodings,nonzer_indeces_train, None,prompts,args.mask)
    augmented_dataloader = get_dataloader(train_data,train_data_tr,rank,world_size,args.bs//2 if  args.masked_and_original else args.bs,encodings,nonzer_indeces_train, tokenizer,prompts,args.mask)
    val_dataloader = get_dataloader(valid_data,valid_data_tr,rank,world_size,args.bs,encodings,nonzer_indeces_valid)
    test_dataloader = get_dataloader(test_data,test_data_tr,rank,world_size,args.bs,encodings,nonzer_indeces_test)
    # non_bin_dataloader = get_dataloader(train_data_tr,rank,world_size,args.bs,encodings,nonzer_indeces_train)
    

    return prompts,rec_dataloader,augmented_dataloader,num_movies,val_dataloader,test_dataloader,valid_data_tr,test_data_tr




def get_dataloader(data,train_data_tr,rank,world_size,bs,encodings,nonzer_indeces_train,tokenizer=None,prompts=None,mask=None,user_id_to_row=None):
    rec_dataset =  DataMatrix(data,train_data_tr,encodings,nonzer_indeces_train,tokenizer,prompts,mask,user_id_to_row)
    sampler = DistributedSampler(rec_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    rec_dataloader = DataLoader(rec_dataset, batch_size=bs, collate_fn= rec_dataset.custom_collator if tokenizer is not None else None  , num_workers = 0,pin_memory=False,
                                sampler = sampler) 
    return rec_dataloader


def get_embeddings(model, dataloader, rank, world_size, num_movies, tokenizer, save_path, save_name, save=True,bfloat16 = False):
    model.eval()
    if os.path.exists(os.path.join(save_path, save_name)):

        embeddings = torch.load(os.path.join(save_path, save_name))

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
                    embeddings[idx[i].item()] = batch_embeddings[i].cpu()

                    #assert there are no nans in the embeddings
                    assert not torch.isnan(embeddings[idx[i].item()]).any()
        if world_size > 1:
            embeddings = dist.gather(embeddings, dst=0)
            if rank == 0:
                embeddings = {k: v for emb in embeddings for k, v in emb.items()}

        if rank == 0 and save:
            os.makedirs(save_path, exist_ok=True)
            torch.save(embeddings, os.path.join(save_path, save_name))
            

        # if bfloat16:
        #     embeddings = {k: torch.tensor(v).bfloat16() for k, v in embeddings.items()}
        # else: 
        #     embeddings = {k: torch.tensor(v) for k, v in embeddings.items()}
 
    return embeddings

def eval_model(args,model, test_dataloader, rank, test_data_tr,precompute_embeddings,loss_f = None, mult_process=True, world_size=2,vae = False,alpha = .5,
               to_mask = None):
    torch.cuda.set_device(rank)
    metrics = defaultdict(list)
    output_metrics = defaultdict(float)
    user_id_set = set()
    user_ids_l = []
    model.to(rank)
    rolling_loss = 0 
    metrics = defaultdict(list)
    logits_rec = None
    loss = None
    with torch.no_grad():
        model.eval()
        for b, item in enumerate(test_dataloader):
            user_ids = sum(item.pop('idx').cpu().tolist(), [])
            user_id_set.update(user_ids)
            item = {k: v.to(rank) for k, v in item.items()}  
            labels = item['labels']
            labels[labels >=1] = 1     
            if args.embedding_module == 'RecVAE':

                movie_emb_clean,loss = model(item['labels_tr'],labels ,calculate_loss = True)
                # print(f"LOSS FROM MODEL {loss=}")
            elif args.embedding_module == 'MacridVAE':
                movie_emb_clean,loss = model(item['labels_tr'],labels )
                
            elif args.embedding_module == 'T5Vae' or args.embedding_module =='OTRecVAE' or args.embedding_module =='MacridTEARS' or args.embedding_module =='RecVAEGenreVAE' :

                movie_emb_clean,logits_rec,logits_text,*_ = model(data_tensor=item['labels_tr'], input_ids = item['input_ids'],attention_mask = item['attention_mask'],alpha  =alpha)
            elif args.embedding_module in ['OTVae' ,'FT5RecVAE']:
                _,_,movie_emb_clean,*_ = model(data_tensor=item['labels_tr'], input_ids = item['input_ids'],attention_mask = item['attention_mask'],alpha  =alpha)
            
            else:     
                if precompute_embeddings is not None: 
                    hidden_states = [precompute_embeddings[idx] for idx in user_ids]
                    hidden_states = torch.stack(hidden_states).to(rank)
                    if vae: 
                        train_items = item['labels_tr']
                        movie_emb_clean,mu,logvar = model.module.classifier_forward(train_items, hidden_states)
                    else:
                        movie_emb_clean = model.module.classifier_forward(hidden_states)  
                else:
                    if vae: 
                        train_items = item['labels_tr']
                        movie_emb_clean,mu,logvar = model(data_tensor=train_items, input_ids = item['input_ids'],attention_mask = item['attention_mask'])
                        
                    else:
                        movie_emb_clean = model(**item)[0]
                                   
            masked_rows = item['labels_tr'].cpu().numpy() if to_mask is None else to_mask
            movie_emb_clean[np.where(masked_rows > 0)] = -1e20
            if logits_rec is not None:
                logits_rec[np.where(masked_rows > 0)] = -1e20
                logits_text[np.where(masked_rows > 0)] = -1e20
                
                recon_rec =   logits_rec.float().cpu().numpy()
                recon_text =  logits_text.float().cpu().numpy()

            

            # print(f"{loss_f(movie_emb_clean, labels).item()=}")
            # print(f"{movie_emb_clean.sum()=}")
            rolling_loss += loss_f(movie_emb_clean, labels).item()
            # exit()

            # movie_emb_clean[np.where(masked_rows > 0)] = -torch.inf
            user_ids_l.append(user_ids)
            recon = movie_emb_clean.float().cpu().numpy() 
            k_values = [10, 20, 50]
            #binarize labels 
            labels = labels.cpu().numpy()
            for k in k_values:
                metrics[f'ndcg@{k}'].append(NDCG_binary_at_k_batch(recon, labels, k=k).tolist())
                # metrics[f'mrr@{k}'].append(MRR_at_k(recon, labels, k=k,mean = False).tolist())
                metrics[f'recall@{k}'].append(Recall_at_k_batch(recon, labels, k=k,mean = False).tolist())
                #for logits_rect nmow 
                if logits_rec is not None:
                    metrics[f'text_ndcg@{k}'].append(NDCG_binary_at_k_batch(recon_text, labels, k=k).tolist())
                    # metrics[f'mrr@{k}'].append(MRR_at_k(recon_text , labels, k=k,mean = False).tolist())
                    metrics[f'text_recall@{k}'].append(Recall_at_k_batch(recon_text , labels, k=k,mean = False).tolist())
                    
                    metrics[f'rec_ndcg@{k}'].append(NDCG_binary_at_k_batch(recon_rec, labels, k=k).tolist())
                    # metrics[f'rec_mrr@{k}'].append(MRR_at_k(recon_rec, labels, k=k,mean = False).tolist())
                    metrics[f'rec_recall@{k}'].append(Recall_at_k_batch(recon_rec, labels, k=k,mean = False).tolist())
                

            
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
                output_metrics[key] = np.mean(sum(metrics[key],[]))
                losses = rolling_loss / len(test_dataloader)
                
        if logits_rec is not None:

            output_metrics['ndcg@50_avg'] = np.mean([output_metrics['ndcg@50'] ,output_metrics['rec_ndcg@50'],output_metrics['text_ndcg@50']])
   
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
    global update_count
    

    for b, (items,augmented_items) in enumerate(zip(rec_dataloader,augmented_dataloader)):
        model.train()
        

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
    
            hidden_states = torch.stack(hidden_states).to(rank)
            movie_emb,mu,logvar = model.module.classifier_forward(hidden_states)
        

        else: 
            movie_emb,mu,logvar = model(input_ids=input_ids.to(rank), attention_mask=attention_mask.to(rank))
        if 200000 > 0:
            anneal = min(.5,
                    1. * update_count / 20000)
        else:
            anneal = args.anneal_cap

        loss = loss_f(movie_emb, labels.to(rank),mu=mu,logvar = logvar,anneal=anneal) / args.update_every
    
        loss.backward()
        
        if (b + 1) % args.update_every == 0:
            
            optimizer.step()
            update_count += 1

            optimizer.zero_grad(set_to_none=True)
        
        rolling_loss.append(loss.item() * args.update_every)
         
        torch.cuda.empty_cache() 
        pbar.set_description(f"Epoch {epoch}: Batch: {b} loss: {np.mean(rolling_loss)} current lr: {scheduler.get_last_lr()[0]} beta {anneal}")

            
    if not args.debug  :

        wandb.log({'loss': np.mean(rolling_loss),
                    "epoch": epoch,
                    "total_steps": epoch*len(rec_dataloader) + b,
                    'lr': scheduler.get_last_lr()[0],
                    })
        
    pbar.set_description(f"Batch {epoch}: loss: {np.mean(rolling_loss)} current lr: {scheduler.get_last_lr()[0]}")
    scheduler.step()


def trainT5Vae(args, rec_dataloader, model, optimizer, scheduler,pbar,epoch,rank,precompute_embeddings):
    rec_dataloader.sampler.set_epoch(epoch)
    if 'RecVAE'  in args.embedding_module:
        loss_f = get_loss('RecVAE_loss',args)
    elif 'MacridTEARS' in args.embedding_module:
        loss_f = get_loss('Macrid_loss',args)
    else:
        loss_f = get_loss('prior_bce')
    # scaler = torch.cuda.amp.GradScaler()
    rolling_loss = []
    kl_roll = []
    global update_count

    bces = []
    klds = []
    bces_mergeds = []
    bces_text = []
    bces_rec = []
    for b,items in enumerate(rec_dataloader):
        model.train()
        labels = items['labels'].to(rank)
        # print(f"{labels.max(axis =1 )=}")
        train_items = items['labels_tr'].to(rank)
        optimizer.zero_grad(set_to_none=True)

        movie_emb, logits_rec, logits_text, mu, logvar, prior_mu, prior_logvar, z_rec, S = model(
            data_tensor=train_items, input_ids=items['input_ids'], attention_mask=items['attention_mask'])

        if 10000 > 0:
            anneal = min(.5,
                 1. * update_count / 10000)
        else:
            anneal = args.anneal_cap
        if 'RecVAE' in  args.embedding_module :
            try:
                loss, BCE, wasserstein_loss, BCE_rec, BCE_text, BCE_merged = loss_f(movie_emb,labels,z_rec,mu,logvar,anneal,model.module.vae.modules_to_save['default'].prior,True,
                            logits_text = logits_text,logits_rec = logits_rec,prior_mu = prior_mu,prior_logvar = prior_logvar,gamma = args.gamma,train_items=train_items,epsilon  = args.epsilon)
            except:

                loss, BCE, wasserstein_loss, BCE_rec, BCE_text, BCE_merged = loss_f(movie_emb,labels,z_rec,mu,logvar,anneal,model.module.vae.prior,True,
                            logits_text = logits_text,logits_rec = logits_rec,prior_mu = prior_mu,prior_logvar = prior_logvar,train_items=train_items,epsilon=args.epsilon)
        
        elif 'MacridTEARS' in args.embedding_module:
                loss, BCE, wasserstein_loss, BCE_rec, BCE_text, BCE_merged = loss_f(movie_emb,labels,z_rec,mu,logvar,anneal,model.module.vae,True,
                            logits_text = logits_text,logits_rec = logits_rec,prior_mu = prior_mu,prior_logvar = prior_logvar,gamma = args.gamma,train_items=train_items,epsilon  = args.epsilon)
            
        elif 'GenreVAE' in args.embedding_module:
            loss, BCE, wasserstein_loss, BCE_rec, BCE_text, BCE_merged = loss_f(movie_emb, logits_rec, logits_text,
                                             labels, mu, logvar, prior_mu, prior_logvar,
                                             None, S, anneal,args.epsilon)
        else:
            loss, BCE, wasserstein_loss, BCE_rec, BCE_text, BCE_merged = loss_f(movie_emb, logits_rec, logits_text,
                                             labels, mu, logvar, prior_mu, prior_logvar,
                                             None, S, anneal,args.epsilon)

        loss.backward()
        optimizer.step()
        bces_mergeds.append(BCE_merged.item())
        bces_text.append(BCE_text.item())
        bces_rec.append(BCE_rec.item())
        rolling_loss.append(loss.item())
        bces.append(BCE.item())
        klds.append(wasserstein_loss.item())
        
        torch.cuda.empty_cache()
        pbar.set_description(
            f"Epoch {epoch}: Batch: {b} loss: {np.mean(rolling_loss)} bce: {np.mean(bces)} WL: {np.mean(klds)} current lr: {scheduler.get_last_lr()[0]} BCE_rec: {np.mean(bces_rec)} BCE_text: {np.mean(bces_text)} BCE_merged: {np.mean(bces_mergeds)}")
        
        if (b + 1) % args.update_every == 0:    
            # print(f"{anneal=}")
            update_count += 1
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
         
        
    if not args.debug  :
        wandb.log({'loss': np.mean(rolling_loss),
                    "epoch": epoch,
                    "total_steps": epoch*len(rec_dataloader) + b,
                    'lr': scheduler.get_last_lr()[0],
                    })
        
    pbar.set_description(f"Batch {epoch}: loss: {np.mean(rolling_loss)} current lr: {scheduler.get_last_lr()[0]}")
    scheduler.step()
    

def train_distill(args, rec_dataloader, model, optimizer, scheduler,pbar,epoch,rank,precompute_embeddings):
    rec_dataloader.sampler.set_epoch(epoch)

    loss_f = get_loss(args.loss) if 'RecVAE' not in args.embedding_module else get_loss('RecVAE_loss',args)
    scaler = torch.cuda.amp.GradScaler()
    rolling_loss = []
    kl_roll = []
    
    global update_count
    update_count = 0

    for b,items in enumerate(rec_dataloader):
        loss = None
        model.train()
        labels = items['labels'].to(rank)
        train_items = items['labels_tr'].to(rank)
        optimizer.zero_grad(set_to_none = True )
        # print(f"{labels.sum(axis=1)=}")
        # print(f"{train_items.sum(axis=1)=}")



        if 200000 > 0:
            anneal = min(.5, 
                        1. * update_count / 20000)
        else:
            anneal = args.anneal_cap
        if precompute_embeddings is not None:
            hidden_states = [precompute_embeddings[idx.item()] for idx in items['idx'].cpu().numpy()]
            hidden_states = torch.stack(hidden_states).to(rank)
            movie_emb,mu,logvar = model.module.classifier_forward(train_items, hidden_states)
            
        if args.embedding_module =='MacridVAE':
            movie_emb,loss = model(train_items,labels,anneal= anneal)
        elif args.embedding_module == 'RecVAE':
            movie_emb,loss = model(train_items,labels ,calculate_loss = True)
        
        else: 
            hidden_states = None 
            movie_emb,mu,logvar = model(data_tensor=train_items, input_ids = items['input_ids'],attention_mask = items['attention_mask'])

        if loss is  None:
            if args.embedding_module == 'RecVAE':
                loss = loss_f(movie_emb,labels,z,mu,logvar,anneal,model.module.prior,gamma = args.gamma,train_items = train_items)
            else: 
                loss = loss_f(movie_emb,labels,mu,logvar,anneal)

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        #if the loss is nan break 
        # if torch.isnan(loss):
        #     exit()
        loss.backward()
        optimizer.step()
        rolling_loss.append( loss.item())
        # torch.cuda.empty_cache() 
        update_count += 1
        pbar.set_description(f"Epoch {epoch}: Batch: {b} loss: {np.mean(rolling_loss)} current lr: {scheduler.get_last_lr()[0]}")
        
            
        scheduler.step()

    if not args.debug  :

        wandb.log({'loss': np.mean(rolling_loss),
                    "epoch": epoch,
                    "total_steps": epoch*len(rec_dataloader) + b,
                    'lr': scheduler.get_last_lr()[0],
                    })
        
    pbar.set_description(f"Batch {epoch}: loss: {np.mean(rolling_loss)} current lr: {scheduler.get_last_lr()[0]}")

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
    parser.add_argument("--regularization_type", default='OT', type=str)
    parser.add_argument("--embedding_dim" , default=1536, type=int)
    parser.add_argument("--bs" , default=8, type=int)
    parser.add_argument("--seed" , default=2024, type=int)
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
    parser.add_argument("--epsilon", default=1, type=float)
    parser.add_argument("--text_tau", default=1, type=float)
    parser.add_argument("--tau", default=1, type=float)
    parser.add_argument("--rec_tau", default=1, type=float)
    parser.add_argument("--recon_tau", default=1, type=float)
    parser.add_argument("--lr2", default=.00001, type=float)
    parser.add_argument("--gamma", default=.0035, type=float)
    parser.add_argument("--kfac", default=10, type=int)
    parser.add_argument("--dfac", default=100, type=int)
    parser.add_argument("--l2_lambda", default=.0001, type=float)
    parser.add_argument("--mask", default=0 , type=float)
    parser.add_argument("--dropout", default=.1, type=float)
    parser.add_argument("--keep_prob", default=.5, type=float)
    parser.add_argument("--temp", default=2, type=float)
    parser.add_argument("--anneal", default=True, type=bool)
    parser.add_argument("--masked_and_original", action = 'store_true')
    parser.add_argument("--wd", default=0, type=float)
    parser.add_argument('--make_embeddings', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--DAE', action='store_true')
    parser.add_argument('--debugger', action='store_true')
    parser.add_argument('--std', type=float, default=0.075,
                 help='Standard deviation of the Gaussian prior.')
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
    parser.add_argument('--train_encoder', action='store_true')
    parser.add_argument('--concat', action='store_true')
    parser.add_argument('--binarize', action='store_true')
    parser.add_argument('--eval_control', action='store_true')
    parser.add_argument('--debug_prompts', action='store_true')
    parser.add_argument('--EASE', action='store_true')
    parser.add_argument('--no_merged', action='store_true')
    parser.add_argument('--no_text', action='store_true')
    parser.add_argument('--no_rec', action='store_true')
    parser.add_argument('--nogb', action='store_true')
    parser.add_argument('--KLD', action='store_true')
    parser.add_argument('--mask_control_labels', action='store_true')

    args = parser.parse_args() if not notebook else parser.parse_args(args=[])
    args.recon = False
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    args.model_save_path = f'/home/mila/e/emiliano.penaloza/scratch/saved_model/{args.data_name}/{args.model_name}'

    
    args.model_save_name = f"{args.model_save_path}/best_model_lr_{args.lr}_embedding_module_{args.embedding_module.replace('/','_')}_epochs_{args.epochs}_l2_lambda_{args.l2_lambda}_lora_alpha_{args.lora_alpha}_lora_r_{args.lora_r}_{args.loss}__bias_{args.no_bias}.pth"
    print(f"MODEL NAME = {args.model_save_name}")

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.model_log_name = f"{args.model_name}_{args.data_name}_embedding_module_{args.embedding_module.replace('/','_')}_{current_time}_{args.seed}.csv"
    print(f"{args.model_log_name=}")

    directory_path = "./scratch"

    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        print(f"The directory '{directory_path}' exists. Will save all weights and models there")
        args.scratch = './scratch'
        
    else:
        args.scratch = '.'
    args.scratch = '/home/mila/e/emiliano.penaloza/scratch'

    return args