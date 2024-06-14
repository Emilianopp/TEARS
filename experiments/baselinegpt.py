# %%
import os



# %%

import sys 
sys.path.append('../')
import time
import pandas as pd 
import numpy as np
import torch
import openai
from tqdm import tqdm
from dotenv import load_dotenv
from trainer.transformer_utilts import *
from helper.dataloader import * 
import pickle
from peft import get_peft_model
from model.MF import sentenceT5Classification
import argparse
from tqdm import tqdm
from peft import LoraConfig, TaskType
import logging
args = parse_args(notebook=True)



# %%
id_title_map = map_id_to_title()
id_genre = map_id_to_genre()
genre_title_map = {}
for movie_id in id_genre:
    genre = id_genre[movie_id]
    title = id_title_map[movie_id]
    genre_title_map[title] = genre
    


# %%
from model.MF import *
tokenizer = get_tokenizer(args)
args.bs = 250
prompts,rec_dataloader,augmented_dataloader,num_movies,val_dataloader,test_dataloader,val_data_tr,test_data_tr= load_data(args,tokenizer,0,1)


# %%
movie_list = [f" movie: {movie} id : {id}\n " for id ,movie in id_title_map.items()]



# %%
load_dotenv()
key = os.getenv("OPEN-AI-SECRET")
openai.api_key = key = os.getenv("OPEN-AI-SECRET")
k= 50

# %%
import pandas as pd
message = []
k_values = [10,20,50]
error = 0 
for b in test_dataloader:
    user_ids = b['idx']

    preds = torch.zeros_like(b['labels'])

    metrics = defaultdict(list)
    k=100
    gpt_outputs = {}
    movies_dict = {}
    for i,user_id in enumerate(user_ids):
        not_done = True
        while not_done:
            user_summary = prompts[user_id.item()]
            labels = b['labels']
            ind = torch.nonzero(b['labels_tr'][i]).flatten()
            seen_movies = '\n'.join([ id_title_map[x.item()] for x in ind])
            
            prompt =\
            f'''
            User summary: 

            {user_summary}

            Here are the available movies: 

            {"".join(movie_list)}

            Important, do not recommend the following movies as they have already been seen by the user:
            
            {seen_movies}
            
            Please only output the top {k} movies. Simply print their id do not use the title out of a total of {len(movie_list)} movies in the format:
            id1, id2, ... idn
            
            '''

            system_prompt = f'''
            You are a movie recommendation system, please recommendation system, please recommend a movie based on the following user preferences
            

            '''
            msg = [
                        {
                            "role": "system",
                            "content": system_prompt

                        },
                        {
                            "role": "user",
                            "content": prompt
                        },                
            ]
            outputs = openai.ChatCompletion.create(
                        model='gpt-4-1106-preview',
                        messages=msg,
                        # max_tokens=300,
                        # temperature=1,
                        seed=2024,
                    )['choices'][0]['message']['content']

            try:
                movies =[int(x) for x in  outputs.split(',')]
                movies_dict[user_id.item()] = movies
                gpt_outputs[user_id.item()] = outputs
                max_val = len(movies)
                for j,mov in enumerate(movies):
                    
                    preds[i,mov] = (max_val - j) + 1 
                #map back to movie title 
                movie_titles = [id_title_map[movie] for movie in movies]
                print(f"{movie_titles=}")
                
                
                message.append(msg)
                not_done = False
            except Exception:
                error += 1
                print(f"{error=}")
                pass
            
        
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()
    for k in k_values:
            metrics[f'ndcg@{k}'].append(NDCG_binary_at_k_batch(preds, labels, k=k).tolist())
            metrics[f'mrr@{k}'].append(MRR_at_k(preds, labels, k=k,mean = False).tolist())
            metrics[f'recall@{k}'].append(Recall_at_k_batch(preds, labels, k=k,mean = False).tolist())

            metrics_df = pd.DataFrame(metrics)
    #log the outputs and movies to a dataframe as well 
    outputs_df = pd.DataFrame(gpt_outputs.items(),columns=['user_id','output'])
    movies_df = pd.DataFrame(movies_dict.items(),columns=['user_id','movies'])
    #merge into one dataframe 
    outputs_df = outputs_df.merge(movies_df,on='user_id')
    #save 
outputs_df.to_csv('./results/peft_openai_outputs.csv',index=False)
#save the metrics dataframe 
metrics_df.to_csv('./results/peft_openai.csv',index=False)
        

# %%
outputs_df

# %%
for k in k_values:
            metrics[f'ndcg@{k}'].append(NDCG_binary_at_k_batch(preds.cpu().numpy(), labels.cpu().numpy(), k=k))
            metrics[f'mrr@{k}'].append(MRR_at_k(preds, labels, k=k,mean = False))
            # metrics[f'recall@{k}'].append(Recall_at_k_batch(preds, labels, k=k,mean = False))

