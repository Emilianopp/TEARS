import os
import sys 
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
from transformers import AutoTokenizer
from collections import Counter



logging.basicConfig(filename=f'./logs/{time.time()}.log',level=logging.INFO)
logging.info(f'Experiment started at {time.time()}')


def parse_args(notebook = False):  # Parse command line arguments
    parser = argparse.ArgumentParser(description="LLM4RecSys")
    parser.add_argument("--data_name", default='ml-1m', type=str)
    parser.add_argument("--direction", default='up', type=str)
    parser.add_argument("--bs", default=32, type=int)
    parser.add_argument("--target_index", default=10, type=int)
    parser.add_argument("--num_samples", default=100, type=int)
    parser.add_argument("--max_l", default=1778, type=int)
    parser.add_argument("--debug_prompts", default=False, type=bool)
    parser.add_argument("--binarize", default=True, type=bool)
    parser.add_argument("--masked_and_original", default=True, type=bool)
    parser.add_argument("--mask", default=0, type=int)
    args = parser.parse_args() if not notebook else parser.parse_args(args=[])
    args.scratch = '/home/mila/e/emiliano.penaloza/scratch'
    return args

args = parse_args()


load_dotenv()
max_l = args.max_l
key = os.getenv("OPEN-AI-SECRET")
openai.api_key = key
tokenizer = AutoTokenizer.from_pretrained("t5-large")
prompts,rec_dataloader,augmented_dataloader,num_movies,val_dataloader,test_dataloader,valid_data_tr,test_data_tr= load_data(args,tokenizer,0,1)

# Load model and mapping dicts
path = './model/weights/TEARS_FineTuned.pt'
model = sentenceT5Classification.from_pretrained('t5-large', num_labels=num_movies)

lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=32, lora_alpha=16, lora_dropout=0,
                            target_modules=["q", "v"],
                            modules_to_save=['classification_head'])
model = get_peft_model(model, lora_config)
state_dict = torch.load(path ,map_location=torch.device('cpu'))

#remove original_module from the dict superkey 
state_dict = {k.replace('classification_head.','classifier.'):v for k,v in state_dict.items()}
model.load_state_dict(state_dict)
model.to(0)


def map_title_to_id(movies_file):
    data = pd.read_csv(movies_file,sep="::",names=["movieId","title","genre"],encoding='ISO-8859-1')
    mapping = {}
    for index, row in data.iterrows():
        mapping[row['title']] = row['movieId']
    return mapping

def map_id_to_title(data_file,data = 'ml-1m'):
    if data == 'ml-1m':
        data = pd.read_csv(data_file,sep="::",names=["itemId","title","genre"],encoding='ISO-8859-1')

        mapping = {}
        for index, row in data.iterrows():
            
            mapping[row['itemId']] = row['title']

        return mapping
    elif data == 'books': 
        df = pd.read_csv(data_file)
        return df.set_index('sid').to_dict()['title']
        
        
        

def map_id_to_genre(path= './data/ml-1m/movies.dat',data='ml-1m'):
    if data == 'ml-1m':
        data = pd.read_csv(path,sep="::",names=["itemId","title","genre"],encoding='ISO-8859-1')

        mapping = {}
        for index, row in data.iterrows():
            
            mapping[row['itemId']] = row['genre']

        return mapping
    elif data =='books': 
        df = pd.read_csv(path)
        return df.set_index('sid').to_dict()['genres']


  

with open('./data_preprocessed/ml-1m/show2id.pkl','rb') as f:
    movie_id_map = pickle.load(f)
    movie_id_map = {v:k for k,v in movie_id_map.items()}

movie_id_to_title = map_id_to_title('./data/ml-1m/movies.dat')
movie_id_to_genre = map_id_to_genre('./data/ml-1m/movies.dat')
movie_title_to_genre = {movie_title: movie_id_to_genre[movie_id] for movie_id, movie_title in movie_id_to_title.items()}
for movie_id, genres in movie_title_to_genre.items():
    movie_title_to_genre[movie_id] = genres.split('|')

#easy tokenization function 
def tokenize_prompt(tokenizer,prompt,max_l):
    encodings = tokenizer([prompt],padding=True, truncation=True,max_length=max_l,
                          return_tensors='pt')
    return encodings


def genrewise_ndcg(genre_movies, genre, min_k=0, max_k=None):
    relevance = [1 if genre in g   else 0 for g in genre_movies[min_k:max_k]]
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
    idcg = sum(1 / np.log2(i + 2) for i in range(len(relevance)))
    ndcg = dcg / idcg if idcg > 0 else 0
    
    return ndcg

# gets diffs in recommendations between two summaries
def get_difs(s1,s2,device = 0,genre = 'Action' ,max_k = 20 ):

    s_1_encodings = tokenize_prompt(tokenizer,s1,max_l)
    s_2_encodings = tokenize_prompt(tokenizer,s2,max_l)
    for k in s_1_encodings:
        s_1_encodings[k] = s_1_encodings[k].to(device)
    for k in s_2_encodings:
        s_2_encodings[k] = s_2_encodings[k].to(device)
    out_s1 = model(**s_1_encodings)
    out_s2 = model(**s_2_encodings)
    l = torch.topk(torch.softmax(out_s1.logits,dim = 1),1000)
    indices_1 = l.indices[0]
    movie_titles_1 = [movie_id_to_title[movie_id_map[index_1.item()]] for index_1 in indices_1]
    l2 = torch.topk(torch.softmax(out_s2.logits,dim = 1),1000)
    indices_2 = l2.indices[0]
    movie_titles_2 = [movie_id_to_title[movie_id_map[index_2.item()]] for index_2 in indices_2]
    genre_movies_1 = [movie_title_to_genre[movie_title] for movie_title in movie_titles_1]
    genre_movies_2 = [movie_title_to_genre[movie_title] for movie_title in movie_titles_2]
    ndcg_1 = genrewise_ndcg(genre_movies_1,genre,min_k = 0,max_k = max_k)
    ndcg_2 = genrewise_ndcg(genre_movies_2,genre,min_k = 0,max_k = max_k)
    delta = ndcg_1 - ndcg_2    

    rankings1 = [(movie_title,genre) for movie_title,genre in zip(movie_titles_1,genre_movies_1)]
    rankings2 = [(movie_title,genre) for movie_title,genre in zip(movie_titles_2,genre_movies_2)]



    return delta,rankings1,rankings2

counts = Counter(sum([v.split("|") for v in movie_id_to_genre.values()],[]))
#keep counts if above 200 
counts = {k:v for k,v in counts.items() if v > 200}
genre_set = list(counts.keys())
genre_set = ', '.join(genre_set)
k_list= list(prompts.keys())
k_sample = random.sample(k_list,500)


np.random.seed(2024)
# Prompt GPT to change summaries 
model.eval()
moved_up = []
moved_down = []
average_medians_down = []
average_medians_up = []
move_up_genres =[]
move_down_genres = []
outputs = []
keys = []
deltas_ndcg_up = []
deltas_ndcg_down = []
deltas_ndcg_up = []
deltas_ndcg_down = []
for k in (pbar:=tqdm(k_sample)): 
    s = prompts[k]
        
    prompt = \
                f"""
                You are a professional editor please identify the users preferred genres from the following:
                {genre_set}
                """
    user_prompt =\
                f"""
                Please identify the users most favorite genre from the following summary and the least favorite genre: 
                in the format Favorite: [genre]\n Least Favorite: [genre]
                {s}.
                """
                
    msg = [
                {
                    "role": "system",
                    "content": prompt

                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]

    genres = openai.ChatCompletion.create(
                    model='gpt-4-1106-preview',
                    messages=msg,
                    max_tokens=300,
                    temperature=0.001,
                    seed=2024,
                )['choices'][0]['message']['content']
    lines = genres.split('\n')
    favorite_genre = lines[0].split(': ')[1]
    logging.info(f"{favorite_genre=}")
    least_favorite_genre = lines[1].split(': ')[1]
    logging.info(f"{least_favorite_genre=}")
    
    msg = [
                {
                    "role": "system",
                    "content": prompt

                },
                {
                    "role": "user",
                    "content": user_prompt
                },
                {'role': 'assistant',
                'content': genres},
                {'role':'user',
                'content':
                f'Now using this setup write the a new summary in the same style that reflects that {favorite_genre} is your least favorite\
                    and {least_favorite_genre} is your favorite only output the full summary keep the format and length the same' }
            ]

    gpt_output = openai.ChatCompletion.create(
                    model='gpt-4-1106-preview',
                    messages=msg,
                    max_tokens=300,
                    temperature=0.0000001,
                    seed=2024,
                )['choices'][0]['message']['content']


    delta_down,rankings1_down,rankings2_down = get_difs(s,gpt_output,genre = favorite_genre)
    delta_up,rankings1_up,rankings2_up = get_difs(s,gpt_output,genre = least_favorite_genre)
    move_down_genres.append(favorite_genre)
    move_up_genres.append(least_favorite_genre)
    outputs.append(gpt_output)  
    keys.append(k)
    deltas_ndcg_down.append(delta_down)
    deltas_ndcg_up.append(delta_up)

    
    pbar.set_description(f"average ndcgs up {np.mean(deltas_ndcg_up)} average ndcg down {np.mean(deltas_ndcg_down)}")    
    logging.info(f"average ndcgs up {np.mean(deltas_ndcg_up)} average ndcg down {np.mean(deltas_ndcg_down)}")
    

data = {
    'move_down_genres': move_down_genres,
    'move_up_genres': move_up_genres,
    'outputs': outputs,
    'keys': keys,
    'deltas_ndcg_down': deltas_ndcg_down,
    'deltas_ndcg_up': deltas_ndcg_up,
}


df = pd.DataFrame(data)
df.to_csv(f'./results/{args.data_name}/gpt4_results_large_{args.direction}.csv')
logging.info(f'./results/{args.data_name}/gpt4_results_large_{args.direction}.csv')

