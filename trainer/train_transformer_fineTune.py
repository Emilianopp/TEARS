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
from peft import LoraConfig, TaskType
from transformers import T5ForSequenceClassification



args = parse_args()



debug_string  = "debug" if args.debug else ""


    

def get_embeddings(prompts,max_l):
    if os.path.isfile(f'./data_preprocessed/{args.data_name}/embedding_module_transformer{debug_string}.pt'):
        embeddings = torch.load(f'../data_preprocessed/{args.data_name}/embedding_module{debug_string}.pt')
    else:
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


    
    test_items = test_data.sort_values(by='userId').groupby('userId')['movieId'].agg(list).reset_index().movieId.values.tolist()
    val_items = valid_data.sort_values(by='userId').groupby('userId')['movieId'].agg(list).reset_index().movieId.values.tolist()


    
    
    movie_title_to_id = map_title_to_id("./data/ml-100k/movies.dat")

    #Plus one since movielens ids start at 1
    num_movies = max(set(train_data.movieId) | set(valid_data.movieId) | set(test_data.movieId)) + 1
    train_matrix= binary_train_matrix_pandas(train_data,num_movies  )

    rec_dataset = RecDatasetNegatives(train_data,num_negatives=1) if args.neg_sample else RecDatasetFull(train_data,num_movies=num_movies)
    rec_dataloader = DataLoader(rec_dataset, batch_size=args.bs, collate_fn=(custom_collate if args.neg_sample else None) , shuffle=True) 
    # prompts = ../saved_user_summary/ml-100k/user_summary_gpt4_{"debug" if args.debug else ""}.json'
    with open(f'./saved_user_summary/{args.data_name}/user_summary_gpt4_{debug_string}.json','r') as f:
        prompts = json.load(f)
        #make int 

    #sort prompt dict in ascending order 
    prompts = {int(float(k)): v for k, v in sorted(prompts.items(), key=lambda item: float(item[0]))}





    

    lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.1)
    model_name = f"{args.embedding_module}-large"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForSequenceClassification.from_pretrained(model_name, num_labels=num_movies)
    max_length = max([tokenizer.encode(text, return_tensors="pt").shape[1] for text in prompts.values()])
    model = get_peft_model(model, lora_config)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))
    model = model.to(args.device)
    
    train_fun = train_fineTuning 

    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    min_val_recall = -np.inf
    patience_counter = 0
    last_saved_epoch = 0
    
    for e in (pbar := tqdm(range(args.epochs))):
       
        val_recall, val_ndcg = train_fun(args, rec_dataloader, model, prompts, optimizer, scheduler, val_items,tokenizer,max_length)
        if val_recall > min_val_recall:
            min_val_recall = val_recall
            torch.save(model.state_dict(), f'./saved_model/{args.data_name}/{args.model_log_name}.pt')
            last_saved_epoch = e
            best_val = val_recall
            best_ndcg = val_ndcg
            if args.patience == patience_counter:
                print(f"Early stopping training at epoch {e}")
                break
        pbar.set_description(f"Epoch {e}: val_recall: {val_recall} ndcg: {val_ndcg} last_saved_epoch: {last_saved_epoch}")
    #evaluate test set 
    #load best model 
    
    model.load_state_dict(torch.load(f'./saved_model/{args.data_name}/{args.model_log_name}.pt'))
    model.eval()
    ranked_movies = generate_preds_fineTuning(args,model,prompts,tokenizer = tokenizer , max_l=max_length)

    test_recall = recall_at_k(test_items,ranked_movies , topk=20)
    test_ndcg = ndcg_k(test_items,ranked_movies , topk=20)
    if not args.debug:
        wandb.log({'test_recall': test_recall,
                   'test_ndcg': test_ndcg})
    print(f"test_recall: {test_recall}")
    
   

    return test_recall,test_ndcg,best_val,best_ndcg

if __name__ == "__main__":
    load_dotenv(".env")
    openai.api_key = os.getenv("OPEN-AI-SECRET")

    start_time = time.time()

    # Call the train_model function
    test_recall,test_ndcg,best_val,best_ndcg = train_model()

    # Record end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
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

    
