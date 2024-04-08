# %%


# %%
import os
# os.chdir('../')

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
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoTokenizer


logging.basicConfig(filename=f'./logs/{time.time()}.log',level=logging.INFO)
logging.info(f'Experiment started at {time.time()}')


def parse_args(notebook = False):  # Parse command line arguments
    parser = argparse.ArgumentParser(description="LLM4RecSys")
    parser.add_argument("--data_name", default='ml-1m', type=str)
    parser.add_argument("--direction", default='up', type=str)
    parser.add_argument("--bs", default=32, type=int)
    parser.add_argument("--target_index", default=19, type=int)
    parser.add_argument("--num_samples", default=250, type=int)
    parser.add_argument("--max_l", default=260, type=int)
    parser.add_argument("--debug_prompts", default=False, type=bool)
    parser.add_argument("--metadata_path",default='./data_preprocessed/books/train.csv',type=str)
    args = parser.parse_args() if not notebook else parser.parse_args(args=[])
    args.scratch = '/home/mila/e/emiliano.penaloza/scratch'
    return args
args = parse_args(True)
args.data_name = 'books'


load_dotenv()
max_l = args.max_l
key = os.getenv("OPEN-AI-SECRET")
openai.api_key = key
tokenizer = AutoTokenizer.from_pretrained("t5-large")
prompts,rec_dataloader,num_movies,val_dataloader,test_dataloader,_,_= load_data(args,tokenizer,0,1)

# Load model and mapping dicts
path = f'{args.scratch}/saved_model/{args.data_name}/t5_classification_embedding_module_t5_classification_l2_lambda_0.0_lora_r_32_scheduler_cosine_warmup_0.001_0.2.csv.pt'
model = sentenceT5Classification.from_pretrained('t5-large', num_labels=num_movies)

lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=32, lora_alpha=16, lora_dropout=0,
                            target_modules=["q", "v"],
                            modules_to_save=['classification_head'])
model = get_peft_model(model, lora_config)
model.load_state_dict(torch.load(path ,map_location=torch.device('cpu')))
model.to(0)



with open(f'./data_preprocessed/{args.data_name}/show2id.pkl','rb') as f:
    movie_id_map = pickle.load(f)
    movie_id_map = {v:k for k,v in movie_id_map.items()}
    


    

item_id_to_title = map_id_to_title( args.metadata_path,data = 'books')
item_id_to_genre = map_id_to_genre(args.metadata_path,data = 'books')


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
    target_index_s1 = indices_1[target_rank].item()
    pos = torch.where(indices_2 == target_index_s1)[0].item()
    delta = target_rank - pos
    for i in range(len(indices_2)):
            index_2 = indices_2[i].item()
            movie_id2 = movie_id_map[index_2] if args.data_name != 'books' else index_2
            genres.append(item_id_to_genre[movie_id2])
            rankings.append(item_id_to_title[movie_id2])
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
            movie_id1 = movie_id_map[index_1] if args.data_name != 'books' else index_1
            movie_title = item_id_to_title[movie_id1]
            rankings.append(item_id_to_title[movie_id1])
            genres.append(item_id_to_genre[movie_id1])
            if i == ranking:
                target_movie = movie_title

    return target_movie,s,rankings ,genres

np.random.seed(2024)
sampled_keys = np.random.choice(list(prompts.keys()),args.num_samples , replace=False)
#shuffle the list 
np.random.shuffle(sampled_keys)

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

    rankings =  {"\n".join([f'{i+1,item}' for i,item in enumerate(rankings[:args.target_index+1])])}
    prompt = \
                f"""
                You will be fed user profiles which summarize user's preferences. Please make edits to them that such that you push
                the input movie as far {direction} as possible. Please maintain the summaries in the following format: 
                You will be fed the summaries, the target item, and the rankings. Please only output the full summaries keep them as close to the input summary as possible.
                """
    user_prompt =\
                f"""
                The rankings are:
                {rankings} \n
                Taking into account the rankings make slight changes to the genres and plot points the user enjoys and does not enjoy.\n
                You can also simply remove words\n 
                You can change words to move other items up or down such that they move the target item {direction}\n
                Do not just bold or emphasize words\n 
                Remember do not include any information about the target item in the summary.\n
                Please only output the full summary make edit/remove at most 10-15 words. 
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
                temperature=0.001,
                max_tokens=300,
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

    pbar.set_description(f'Target item: {target_movie}, Original Ranking: Delta: {delta} , Running mean Delta: {np.mean(running_deltas):.3f},Running median Delta: {np.median(running_deltas)}, Mean BLEU Score: {np.mean(running_bleu_scores):.3f}')
    logging.info(f'Target item: {target_movie}, Original Ranking: Delta: {delta} , Running Delta: {np.mean(running_deltas):.3f},Running Delta: {np.median(running_deltas)}, Mean BLEU Score: {np.mean(running_bleu_scores):.3f}')
    

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



# %%
from pprint import pprint
pprint(user_prompt)

# %%
s = set(movie_id_map.keys())
val = set(movie_id_map.values())

# %%
max(list(movie_id_map.keys()))


