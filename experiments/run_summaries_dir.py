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
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoTokenizer


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
    args = parser.parse_args() if not notebook else parser.parse_args(args=[])
    args.scratch = '/home/mila/e/emiliano.penaloza/scratch'
    return args

args = parse_args()


load_dotenv()
max_l = args.max_l
key = os.getenv("OPEN-AI-SECRET")
openai.api_key = key
tokenizer = AutoTokenizer.from_pretrained("t5-large")
prompts,rec_dataloader,num_movies,val_dataloader,test_dataloader,_,_= load_data(args,tokenizer,0,1)

# Load model and mapping dicts
path = './model/weights/TEARS_FineTuned.pt'
model = sentenceT5Classification.from_pretrained('t5-large', num_labels=num_movies)

lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=32, lora_alpha=16, lora_dropout=0,
                            target_modules=["q", "v"],
                            modules_to_save=['classification_head'])
model = get_peft_model(model, lora_config)
model.load_state_dict(torch.load(path ,map_location=torch.device('cpu')))
model.to(0)

with open('./data_preprocessed/ml-1m/show2id.pkl','rb') as f:
    movie_id_map = pickle.load(f)
    movie_id_map = {v:k for k,v in movie_id_map.items()}

movie_id_to_title = map_id_to_title('./data/ml-1m/movies.dat')
movie_id_to_genre = map_id_to_genre('./data/ml-1m/movies.dat')


#easy tokenization function 
def tokenize_prompt(tokenizer,prompt,max_l):
    encodings = tokenizer([prompt],padding=True, truncation=True,max_length=max_l,
                          return_tensors='pt')
    return encodings

# gets diffs in recommendations between two summaries
def get_difs(s1,s2,device = 0,target_rank = 10 ):
    rankings = []
    genres = []
    s_1_encodings = tokenize_prompt(tokenizer,s1,max_l)
    s_2_encodings = tokenize_prompt(tokenizer,s2,max_l)
    for k in s_1_encodings:
        s_1_encodings[k] = s_1_encodings[k].to(device)
    for k in s_2_encodings:
        s_2_encodings[k] = s_2_encodings[k].to(device)
    out_s1 = model(**s_1_encodings)
    out_s2 = model(**s_2_encodings)
    l = torch.topk(torch.softmax(out_s1.logits,dim = 1),2000)
    indices_1 = l.indices[0]
    l2 = torch.topk(torch.softmax(out_s2.logits,dim = 1),2000)
    indices_2 = l2.indices[0]
    movie_titles_1 = [movie_id_to_title[movie_id_map[index_1.item()]] for index_1 in indices_1]
    movie_titles_2 = [movie_id_to_title[movie_id_map[index_2.item()]] for index_2 in indices_2]
    target_index_s1 = indices_1[target_rank].item()
    pos = torch.where(indices_2 == target_index_s1)[0].item()
    delta = target_rank - pos
    for i in range(len(indices_2)):
            index_2 = indices_2[i].item()
            movie_id2 = movie_id_map[index_2]
            movie_title = movie_id_to_title[movie_id2] 
            genres.append(movie_id_to_genre[movie_id2])
            rankings.append(movie_id_to_title[movie_id2])
    return delta,pos,rankings,genres

#get recommendations for a single summary
def get_recs(s1,device = 0,ranking = 10): 
    s = ''
    rankings = []
    genres = []
    s1_encodings = tokenize_prompt(tokenizer,s1,max_l)
    #move encodings to device
    for k in s1_encodings:
        s1_encodings[k] = s1_encodings[k].to(device)
    out_s1 = model(**s1_encodings)
    l = torch.topk(torch.softmax(out_s1.logits,dim = 1),1000)
    indices_1 = l.indices[0]
    for i in range(len(indices_1)):
            index_1 = indices_1[i].item()
            movie_id1 = movie_id_map[index_1]
            movie_title = movie_id_to_title[movie_id1]
            rankings.append(movie_id_to_title[movie_id1])
            genres.append(movie_id_to_genre[movie_id1])
            if i == ranking:
                target_movie = movie_title

    return target_movie,s,rankings ,genres

np.random.seed(2024)
sampled_keys = np.random.choice(list(prompts.keys()),args.num_samples , replace=False)
gpt_outputs_d = defaultdict(list)
original_rankings = defaultdict(list)
changed_rankings = defaultdict(list)
original_genres = defaultdict(list)
changed_genres = defaultdict(list)
original_prompts = {}
user_target_movie ={}
deltas = {}
running_deltas = []
bleu_scores = {}
running_bleu_scores = []
direction = args.direction
model.eval()
taget_rank = args.target_index

# Prompt GPT to change summaries 
for k in (pbar:=tqdm(sampled_keys)):
    target_movie, ranking,rankings,o_genres  = get_recs(prompts[k],ranking = taget_rank)
    logging.info(f"{target_movie=}")

    prompt = \
                f"""
                You will be fed user profiles which summarize user's movie preferences. Please make edits to them that such that you push
                the input movie as far {direction} as possible. Please maintain the summaries in the following format: 
                Summary: [Specific details about genres the user enjoys]. [Specific details of plot points the user seems to enjoy]. 
                [Specific details about genres the user does not enjoy]. [Specific details of plot points the user does not enjoy but other users may].
                You will be fed the summaries, the target movie, and the rankings. Please only output the full summaries keep them as close to the input summary as possible.
                """
    user_prompt =\
                f"""
                Remember do not include any information about the target movie in the summary.\n
                Please only output the full summary make edit at most 10 words. 
                Target movie to move in ranking {direction}: {target_movie}\n
                {prompts[k]}\n 
                """
    while True:
        try:
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

            gpt_output = openai.ChatCompletion.create(
                model='gpt-4-1106-preview',
                messages=msg,
                max_tokens=300,
                temperature=0.01,
                seed=2024
            )['choices'][0]['message']['content']

            # If the API call is successful and does not throw an error, break the loop
            break
        except Exception as e:
            # If an error occurs, print the error and continue with the next iteration of the loop
            print(f"An error occurred: {e}")
            
    gpt_outputs_d[k].append(gpt_output)
    original_prompts[k] = prompts[k]
    delta, new_rank,new_rankings,generes = get_difs(s1 = prompts[k],s2 = gpt_output,target_rank = taget_rank)
    user_target_movie[k] = target_movie
    original_rankings[k].append(rankings)
    changed_rankings[k].append(new_rankings)
    deltas[k] = delta
    changed_genres[k].append(generes)   
    original_genres[k].append(o_genres)
    running_deltas.append(delta)
    reference = [word for word in prompts[k].split()]
    candidate = [word for word in gpt_output.split()]
    bleu_score = sentence_bleu([reference], candidate)
    bleu_scores[k] = bleu_score
    running_bleu_scores.append(bleu_score)

    pbar.set_description(f'Original Ranking: Delta: {delta} , Running Delta: {np.mean(running_deltas):.3f},Running Delta: {np.median(running_deltas)}, Mean BLEU Score: {np.mean(running_bleu_scores):.3f}')
    logging.info(f'Original Ranking: Delta: {delta} , Running Delta: {np.mean(running_deltas):.3f},Running Delta: {np.median(running_deltas)}, Mean BLEU Score: {np.mean(running_bleu_scores):.3f}')
    

#make a dataframe for the dicts 
blue_scores_df = pd.DataFrame.from_dict(bleu_scores,orient = 'index')
gpt_outputs_df = pd.DataFrame.from_dict(gpt_outputs_d,orient = 'index')
original_rankings_df = pd.DataFrame.from_dict(original_rankings,orient = 'index')
changed_rankings_df = pd.DataFrame.from_dict(changed_rankings,orient = 'index')
deltas_df = pd.DataFrame.from_dict(deltas,orient = 'index')
original_prompts_df = pd.DataFrame.from_dict(original_prompts,orient = 'index')

 
#save dataframe 
all_df = pd.concat([original_prompts_df,gpt_outputs_df,original_rankings_df,changed_rankings_df,deltas_df,blue_scores_df],axis = 1)
all_df.columns = ['Original Summary','changed Summary','Original Ranking','Changed Ranking','Delta','blue_scores']

all_df.to_csv(f'./results/{args.data_name}/gpt4_results_{taget_rank}_{direction}.csv')
logging.info(f'Saved results to ./results/{args.data_name}/gpt4_results_{taget_rank}_{direction}.csv')

