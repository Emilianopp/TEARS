import pickle
from sentence_transformers import SentenceTransformer

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
from model.MF import MatrixFactorizationLLM
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

args = parse_args()
model_name = f"{args.embedding_module}-large"

tokenizer = T5Tokenizer.from_pretrained(model_name)


debug_string  = "debug" if args.debug else ""
def get_embeddings(prompts,max_l):
    if os.path.isfile(f'./data_preprocessed/{args.data_name}/embedding_module_transformer{debug_string}.pt'):
        embeddings = torch.load(f'../data_preprocessed/{args.data_name}/embedding_module{debug_string}.pt')
    else:
        print(f"{len(prompts)=}")
        prompt_dataset = PromptDataset(prompts)
        prompt_data_loader = DataLoader(prompt_dataset, batch_size=len(prompts)//2, shuffle=True)

        
        embeddings = make_embeddings(args.embedding_module,prompt_data_loader,max_l)
        torch.save(embeddings,f'./data_preprocessed/{args.data_name}/embedding_module{debug_string}.pt')
    return embeddings

def train_model():
    t_start = time.time()

    if not args.debug:
        wandb.init(project='llm4rec', name=args.model_log_name)
        wandb.config.update(args)
        wandb.watch_called = False  # To avoid re-watching the model

    # 1. Data Loading & Preprocessing
    train_data = pd.read_csv(f'./data_preprocessed/{args.data_name}/train_leave_one_out_timestamped.csv')
    valid_data = pd.read_csv(f'./data_preprocessed/{args.data_name}/validation_leave_one_out_timestamped.csv')
    test_data = pd.read_csv(f'./data_preprocessed/{args.data_name}/test_leave_one_out_timestamped.csv')
    movie_title_to_id = map_title_to_id("./data/ml-100k/movies.dat")

    #Plus one since movielens ids start at 1
    num_movies = max(set(train_data.movieId) | set(valid_data.movieId) | set(test_data.movieId)) + 1
    train_matrix= binary_train_matrix_pandas(train_data,num_movies  )

    rec_dataset = RecDatasetNegatives(train_data,num_negatives=1) if args.negative_sample else RecDatasetFull(train_data,num_movies)
    rec_dataloader = DataLoader(rec_dataset, batch_size=3, collate_fn=(custom_collate if args.negative_sample else None), shuffle=True)

    
  
  
    # prompts = ../saved_user_summary/ml-100k/user_summary_gpt4_{"debug" if args.debug else ""}.json'
    with open(f'./saved_user_summary/{args.data_name}/user_summary_gpt4_{debug_string}.json','r') as f:
        prompts = json.load(f)


    max_length = max([tokenizer.encode(text, return_tensors="pt").shape[1] for text in prompts.values()])

    prompt_embeddings = get_embeddings(prompts,max_length)

    
    text_embedding_shape = prompt_embeddings[list(prompts.keys())[0]].shape[1]


    
    model = movieTransformer(attention_dim=text_embedding_shape,
                             num_heads=args.num_heads, 
                             num_layers_mlp=args.num_layers, 
                             output_emb_mlp=num_movies , 
                             num_layers=args.num_layers_transformer)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))
    model = model.to(args.device)

    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
        
    for e in (pbar := tqdm(range(args.epochs))):
        for user_ids ,positive_samples, negative_samples in rec_dataloader:
            model.train()
            batch_prompt_embeddings = torch.stack([(prompt_embeddings[x] if not args.debug else prompt_embeddings[list(prompts.keys())[0]]) for x in user_ids ]).to(args.device)
            batch_prompt_embeddings = batch_prompt_embeddings.to(args.device)
            positive_samples = positive_samples.to(args.device)
            negative_samples = negative_samples.to(args.device)
            optimizer.zero_grad()
            movie_emb = model(batch_prompt_embeddings)
            movie_emb_pos = movie_emb[:,positive_samples]
            movie_emb_neg = movie_emb[:,positive_samples]
            loss = bpr_loss(movie_emb_pos, movie_emb_neg)
            
            print(f"{movie_emb.shape=}")
            exit(0)
            

            loss.backward()
            optimizer.step()
            scheduler.step()
            wandb.log({'loss': loss.item()})

        model.eval()
        with torch.no_grad():
            val_recall = recall_at_k(model, prompt_embeddings, val_items, k=20)
            test_recall = recall_at_k(model, prompt_embeddings, test_items, k=210)
            wandb.log({'val_recall': val_recall})
            wandb.log({'test_recall': test_recall})
            print(f"Epoch {e}: val_recall: {val_recall}, test_recall: {test_recall}")
            pbar.set_description(f"Epoch {e}: val_recall: {val_recall}, test_recall: {test_recall}")

            
            
         



if __name__ == "__main__":
    load_dotenv(".env")
    openai.api_key = os.getenv("OPEN-AI-SECRET")

    start_time = time.time()

    # Call the train_model function
    train_model()

    # Record end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"Elapsed Time: {elapsed_time} seconds")

    
