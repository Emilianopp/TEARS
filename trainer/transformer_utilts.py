from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from pprint import pprint
from model.decoderMLP import decoderMLP, decoderAttention, movieTransformer
from tqdm import tqdm
import openai
import pickle 
import argparse
from typing import List
import torch.nn.functional as F
import sys
import logging
import numpy as np
import torch
import json
import time
import torch.optim as optim
import torch
from scipy.sparse import csr_matrix
import os 
import pandas as pd
from collections import defaultdict
from helper.sampler import NegSampler, negsamp_vectorized_bsearch_preverif
from argparse import ArgumentParser
from model.MF import MatrixFactorization
from helper.eval_metrics import precision_at_k, recall_at_k, mapk, ndcg_k
from helper.dataloader import load_dataset, map_title_to_id, convert_titles_to_ids, \
    create_train_matrix_and_actual_lists
from transformers import T5EncoderModel
import wandb
from peft import LoraConfig, TaskType
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.distributed as dist


lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, r=1, lora_alpha=1, lora_dropout=0.1
)




T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
# device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def precompute_embeddings(args,model,tokenizer,rec_dataloader,prompts):
    #use only the parts of the model that are frozen
    model.eval()
    model = model.encoder
    model.eval()
    prompt_embeddings = {}
    

def pad_tensor(tens,max_l):
    return torch.cat([tens,torch.zeros(max_l-tens.shape[0],tens.shape[1])],dim=0)

def make_embeddings(embedding_module,prompt_dataset,max_l):
    if embedding_module == 't5':
        print('Loading T5 embeddings')
        model_name = "t5-large"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5EncoderModel.from_pretrained(model_name).to(device)
        out_dict = {}
        model.eval()

        for i,batch in enumerate(prompt_dataset):
            prompts = batch['text'].to(device)
            user_id = batch['user_id'].to(device)

            input_ids = torch.stack([tokenizer.encode(x, return_tensors="pt",padding = 'max_length', max_length = max_l) for x in prompts]).squeeze(1).to(device)
            # Pass the input through the encoder
            with torch.no_grad():
                encoder_output = model(input_ids).last_hidden_state




            out_dict  = {**out_dict,**dict(zip(user_id,encoder_output))}
        return out_dict
    
        

def bpr_loss(positive_scores, negative_scores):
    """
    Bayesian Personalized Ranking (BPR) loss.

    Parameters:
        - positive_scores: Tensor of shape (batch_size, 1) representing the scores for positive samples.
        - negative_scores: Tensor of shape (batch_size, num_negatives) representing the scores for negative samples.

    Returns:
        - loss: BPR loss.
    """
    # Calculate the difference between positive and negative scores
    scores_diff = negative_scores - positive_scores.unsqueeze(1)

    # Calculate the sigmoid of the score differences
    exp_scores_diff = torch.exp(-scores_diff)

    # Calculate the BPR loss
    loss = -torch.sum(F.logsigmoid(scores_diff))

    return loss

def binary_cross_entropy_loss(embeddings,positives):

    # log_probs = torch.log(torch.softmax(embeddings,axis = 1))

    # positive_scores = log_probs @ positives.T.float()
    
    # print(f"{embeddings.shape=}")
    scores = F.binary_cross_entropy_with_logits(embeddings,positives.float(), reduction='none')
  

    return scores.mean(axis = 1).mean()

def binary_cross_entropy_loss_softmax(embeddings,positives):

    probs = torch.softmax(embeddings,axis = 1)

    # positive_scores = log_probs @ positives.T.float()
    
    # print(f"{embeddings.shape=}")
    scores = torch.nn.BCELoss(embeddings,positives.float(), reduction='none')
  

    return scores.mean(axis = 1).mean()
    
def get_loss(loss_name):
    if loss_name == 'bce':
        return binary_cross_entropy_loss
    elif loss_name == 'bce_softmax':
        return binary_cross_entropy_loss_softmax
    else:
        raise ValueError(f"loss {loss_name} not supported")

def generate_preds(model, prompt_embeddings):

    top_movies = []
    for user_id, prompt_embedding in prompt_embeddings.items():

        prompt_embedding = prompt_embedding.to(device)
        prompt_embedding = prompt_embedding.unsqueeze(0)
        prompt_embedding = prompt_embedding.to(device)
        model.eval()
        with torch.no_grad():
            movie_emb = model(prompt_embedding)
            movie_emb = movie_emb.cpu()
            movie_emb = movie_emb.squeeze(0)
            movie_emb = movie_emb.numpy()
            movie_emb = np.argsort(movie_emb)
            movie_emb = movie_emb.tolist()
            #return index of top 20 movies
            top_movies.append(movie_emb[-20:])

    return top_movies



def generate_preds_fineTuning(args, model, prompts,rec_dataloader, tokenizer,rank,topk=None,world_size =2 ,mult_process = True):
    num_users = max(prompts.keys())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(rank)
    top_movies = {}

    with torch.no_grad():
        for b,(user_ids, positive_movies) in tqdm(enumerate(rec_dataloader)):
            texts = [prompts[user_id.item()] if not args.debug else prompts[list(prompts.keys())[0]] for user_id in user_ids]

                # Tokenize the batch of prompts

            model.eval()
            input_ids = tokenizer.batch_encode_plus(
                texts, padding=True, return_tensors="pt", truncation=True
            )["input_ids"].to(device)
            movie_emb = model(input_ids)[0]
            movie_emb = torch.argsort(movie_emb)
            # Extract top 20 movies for each user into the top_movies dict 
            for i, user_id in enumerate(user_ids):
                top_movies[user_id.item()] = movie_emb[i][-20:].cpu().numpy()
    if mult_process:
        ranked_movies_all = [None for _ in range(world_size)]
        dist.all_gather_object(ranked_movies_all,top_movies)
        #make into a single dict 
        ranked_movies_all = {k: v for d in ranked_movies_all for k, v in d.items()}
    else: 
        ranked_movies_all = top_movies
    return ranked_movies_all


    

def train_bpr(args, rec_dataloader, model, prompt_embeddings, optimizer, scheduler,prompts, val_items):
    for user_ids, positive_samples, negative_samples in rec_dataloader:
            model.train()
            
            # Extract prompt embeddings for the current batch of users
            batch_prompt_embeddings = torch.stack(
                [(prompt_embeddings[x] if not args.debug else prompt_embeddings[list(prompts.keys())[0]]) for x in user_ids]
            ).to(args.device)
            batch_prompt_embeddings = batch_prompt_embeddings.to(args.device)
            
            # Move samples to the specified device
            positive_samples = positive_samples.to(args.device)
            negative_samples = negative_samples.to(args.device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Get movie embeddings using the model
            movie_emb = model(batch_prompt_embeddings)
            
            # Extract positive and negative movie embeddings
            movie_emb_pos = movie_emb[:, positive_samples]
            movie_emb_neg = movie_emb[:, negative_samples]
            
            # Calculate BPR loss
            loss = bpr_loss(movie_emb_pos, movie_emb_neg)
            if args.l2_lambda > 0:
                l2_reg = torch.tensor(0.).to(args.device)
                for param in model.parameters():
                    l2_reg += torch.norm(param)
            
                # Add L2 regularization to the loss
                loss += args.l2_lambda * l2_reg
        
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            scheduler.step()
            

            if not args.debug:
                wandb.log({'loss': loss.item()})
            else:
                break
            #clear torch cache 
            torch.cuda.empty_cache()
            
            
    model.eval()
    ranked_movies = generate_preds(model,prompt_embeddings)

    val_recall = recall_at_k(val_items,ranked_movies , topk=20)
    val_ndcg = ndcg_k(val_items,ranked_movies , topk=20)
    if not args.debug:
        wandb.log({'val_recall': val_recall,
                    'val_ndcg': val_ndcg})
    return val_recall,val_ndcg


def train_softmax(args, rec_dataloader, model, prompt_embeddings, optimizer, scheduler,prompts, val_items):
    for user_ids, positive_movies in rec_dataloader:
            model.train()
            
            # Extract prompt embeddings for the current batch of users

            batch_prompt_embeddings = torch.stack(
                [(prompt_embeddings[x] if not args.debug else prompt_embeddings[list(prompts.keys())[0]]) for x in user_ids]
            ).to(args.device)
            batch_prompt_embeddings = batch_prompt_embeddings.to(args.device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Get movie embeddings using the model

            movie_emb = model(batch_prompt_embeddings)
            
            # Calculate BPR loss
            loss_bce = binary_cross_entropy_loss( movie_emb,positive_movies)
            if args.l2_lambda > 0:
                l2_reg = None
                for param in model.parameters():
                    if param.requires_grad:
                        if l2_reg is None:
                            l2_reg = param.norm()
                        else:
                            l2_reg = l2_reg + param.norm()
            
                # Add L2 regularization to the loss
                loss = args.l2_lambda * l2_reg + loss_bce
            
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            scheduler.step()
            


            if not args.debug:
                wandb.log({'loss': loss_bce.item()})
            else:
                break
    model.eval()
    ranked_movies = generate_preds(model,prompt_embeddings)

    val_recall = recall_at_k(val_items,ranked_movies , topk=20)
    val_ndcg = ndcg_k(val_items,ranked_movies , topk=20)
    if not args.debug:
        wandb.log({'val_recall': val_recall,
                    'val_ndcg': val_ndcg})
    return val_recall,val_ndcg
  



def train_fineTuning(args, rec_dataloader, model, prompts, optimizer, scheduler, val_items,tokenizer,max_l,pbar,epoch,device):
    rec_dataloader.sampler.set_epoch(epoch)
    loss = get_loss(args.loss)

    for b,(user_ids, positive_movies) in enumerate(rec_dataloader):
       
            model.train()
            
            optimizer.zero_grad(set_to_none = True )
            # Extract prompt embeddings for the current batch of users
            texts = [prompts[user_id.item()] if not args.debug else prompts[list(prompts.keys())[0]] for user_id in user_ids]
        
            
            if tokenizer is not None:
                tokenized =tokenizer.batch_encode_plus(texts, return_tensors="pt",padding = True)
                input_ids = tokenized["input_ids"].to(device)
                attention_mask = tokenized["attention_mask"].to(device)
                movie_emb = model(input_ids)[0]
            else: 
                movie_emb = model(texts)


            loss_bce = binary_cross_entropy_loss( movie_emb,positive_movies.to(device))

            if args.l2_lambda > 0:
                num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

                l2_reg = 0
                for name, param in model.named_parameters():
                    if param.requires_grad and ('classification_head' in name or 'bias' in name):
                            l2_reg = l2_reg + param.norm()
                # Add L2 regularization to the loss
                loss = args.l2_lambda * l2_reg + loss_bce
            else:
                loss = loss_bce

            
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            pbar.set_description(f"Batch {b}: loss: {loss.item()} bce_loss: {loss_bce.item()} current lr: {scheduler.get_last_lr()[0]}")

            if not args.debug:
                wandb.log({'loss': loss.item(),
                           'bce_loss': loss_bce.item(),
                           'lr': scheduler.get_last_lr()[0],
                           "epoch": epoch,
                           "batch": b,
                           "total_steps": epoch*len(rec_dataloader) + b})
            else:
                break
            #clear cuda cache 
            torch.cuda.empty_cache()
    scheduler.step()
    
            
    

def parse_args(notebook = False):  # Parse command line arguments
    parser = argparse.ArgumentParser(description="LLM4RecSys")
    parser.add_argument("--data_name", default='ml-1m', type=str)
    parser.add_argument("--log_file", default= 'model_logs/ml-100k/logging_llmMF.csv', type=str)
    parser.add_argument("--model_name", default='Transformer', type=str)
    parser.add_argument("--emb_type", default='attn', type=str)
    parser.add_argument("--summary_style", default='topM', type=str)
    parser.add_argument("--embedding_module", default='t5', type=str)
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
    parser.add_argument("--total_steps", default=1000, type=int)
    parser.add_argument("--attention_emb", default=512, type=int)
    parser.add_argument("--train", default=True, type=bool)
    parser.add_argument("--num_neg", default=1, type=int)
    parser.add_argument("--num_heads", default=4, type=int)
    parser.add_argument("--lr", default=.0001, type=float)
    parser.add_argument("--l2_lambda", default=.0001, type=float)
    parser.add_argument("--wd", default=0, type=float)
    parser.add_argument('--make_embeddings', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debugger', action='store_true')
    parser.add_argument('--cosine', action='store_true')
    parser.add_argument('--make_augmented', action='store_true')
    parser.add_argument('--neg_sample', action='store_true')
    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--make_data', action='store_true')
    parser.add_argument('--loss', default='bce', type=str, choices=['bce','bce_softmax'])
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--warmup_mlp", type=int, default=10)
    parser.add_argument('--no_bias', action='store_true')

    args = parser.parse_args() if not notebook else parser.parse_args(args=[])
    args.recon = False
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    args.model_save_path = f'/home/mila/e/emiliano.penaloza/scratch/saved_model/ml-100k/{args.model_name}'


    args.model_save_name = f"{args.model_save_path}/best_model_lr_{args.lr}_embedding_module_{args.embedding_module}_epochs_{args.epochs}_l2_lambda_{args.l2_lambda}_lora_alpha_{args.lora_alpha}_lora_r_{args.lora_r}_{args.loss}__bias_{args.no_bias}.pth"
    print(f"MODEL NAME = {args.model_save_name}")

    args.model_log_name = f"{args.model_name}_lr_{args.lr}_embedding_module_{args.embedding_module}_epochs_{args.epochs}_l2_lambda_{args.l2_lambda}_lora_alpha_{args.lora_alpha}_lora_r_{args.lora_r}_{args.loss}_bias_{args.no_bias}.csv"
    
    directory_path = "../scratch"

    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        print(f"The directory '{directory_path}' exists. Will save all weights and models there")
        args.scratch = '../scratch'
        
    else:
        args.scratch = '.'
    args.scratch = '/home/mila/e/emiliano.penaloza/scratch'

    return args