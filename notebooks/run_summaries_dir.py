# %%
import os
import sys 
sys.path.append('../')

import time
from transformers import  AutoTokenizer 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import torch
import openai
# from tenacity import retry, wait_exponential, stop_after_attempt
from tqdm import tqdm
from dotenv import load_dotenv
from helper.in_context_learning_batch import in_context_user_summary, in_context_retrieval
from helper.builder import build_context, build_movie_candidates

import time
import os
import pandas as pd 
from dotenv import load_dotenv
from trainer.transformer_utilts import *

import sys 
sys.path.append("..")
import pickle
from peft import get_peft_model

import pandas as pd
from data.dataloader import get_dataloader
from helper.dataloader import load_pickle, map_title_to_id, map_id_to_title
from torch.utils.data import DataLoader, Subset
from dotenv import load_dotenv
from model.MF import MatrixFactorizationLLM,sentenceT5Classification
from model.decoderMLP import decoderMLP, decoderAttention
from model.transformerModel import movieTransformer
import argparse
from torch.optim.lr_scheduler import LambdaLR
from helper.eval_metrics import *
from helper.metrics import genrewise_ndcg,genrewise_recall
from helper.dataloader import *
from trainer.training_utils import *
import wandb
from tqdm import tqdm
import math
from trainer.transformer_utilts import *
from torch.nn.parallel import DataParallel
from peft import LoraConfig, TaskType
from transformers import T5ForSequenceClassification
#import mp 
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import hashlib
from model.MF import T5MappingVAE
from helper.metrics import genrewise_ndcg,genrewise_recall,genrewise_average_rankings
#import logging 
import logging
from nltk.translate.bleu_score import sentence_bleu
from pprint import pprint
#set up logging add the time of the experiment to the log file
logging.basicConfig(filename=f'./logs/{time.time()}.log',level=logging.INFO)
logging.info(f'Experiment started at {time.time()}')


os.chdir('../')


# %%


#set export MASTER_ADDR="127.0.0.1" as in os 
os.environ['MASTER_ADDR'] = '127.0.0.1'

#same with master port  export MASTER_PORT=$(expr 10000)

os.environ['MASTER_PORT'] = '10000'

    


def parse_args(notebook = False):  # Parse command line arguments
    parser = argparse.ArgumentParser(description="LLM4RecSys")
    parser.add_argument("--data_name", default='ml-1m', type=str)
    parser.add_argument("--direction", default='up', type=str)
    parser.add_argument("--bs", default=32, type=int)
    parser.add_argument("--target_index", default=10, type=int)
    parser.add_argument("--num_samples", default=100, type=int)
    parser.add_argument("--debug_prompts", default=False, type=bool)

    args = parser.parse_args() if not notebook else parser.parse_args(args=[])
    args.scratch = '/home/mila/e/emiliano.penaloza/scratch'
    
    return args
args = parse_args()


    
def cleanup():
    dist.destroy_process_group()
    
def setup(rank,world_size):
    assert torch.distributed.is_available()
    print("PyTorch Distributed available.")
    print("  Backends:")
    print(f"    Gloo: {torch.distributed.is_gloo_available()}")
    print(f"    NCCL: {torch.distributed.is_nccl_available()}")
    print(f"    MPI:  {torch.distributed.is_mpi_available()}")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
 
    torch.distributed.init_process_group(
         backend="nccl",
        init_method="env://",
         world_size=world_size,
         rank=rank,
     )
rank = 0
world_size = 1


setup(rank, world_size)




# %%
load_dotenv()

key = os.getenv("OPEN-AI-SECRET")
openai.api_key = key


# %%
tokenizer = AutoTokenizer.from_pretrained("t5-large")
prompts,rec_dataloader,num_movies,val_dataloader,test_dataloader= load_data_vae_way(args,tokenizer,rank,world_size)

path = f'{args.scratch}/saved_model/{args.data_name}/t5_classification_please_embedding_module_t5_classification_l2_lambda_0.01_lora_r_32_scheduler_cosine_warmup_0.001.csv.pt'

model = sentenceT5Classification.from_pretrained('t5-large', num_labels=num_movies)

lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=32, lora_alpha=16, lora_dropout=0,
                            target_modules=["q", "v"],
                            modules_to_save=['classification_head'])
model = get_peft_model(model, lora_config)


model.load_state_dict(torch.load(path ,map_location=torch.device('cpu')))
model.to(rank)



# %%
length_list = []
for i,k in prompts.items():
    length_list.append(len(k.split())) 

# %%
with open('/home/mila/e/emiliano.penaloza/LLM4REC/vae/ml-1m/pro_sg_text/show2id.pkl','rb') as f:
    movie_id_map = pickle.load(f)
    #reverse the map
    movie_id_map = {v:k for k,v in movie_id_map.items()}
#reverse the map

movie_id_to_title = map_id_to_title('/home/mila/e/emiliano.penaloza/LLM4REC/data/ml-1m/movies.dat')
movie_id_to_genre = map_id_to_genre('/home/mila/e/emiliano.penaloza/LLM4REC/data/ml-1m/movies.dat')
#create a movie title to genre map
movie_title_to_genre = {movie_title: movie_id_to_genre[movie_id] for movie_id, movie_title in movie_id_to_title.items()}
 
for movie_id, genres in movie_title_to_genre.items():
    movie_title_to_genre[movie_id] = genres.split('|')


# %%
train_data = pd.read_csv(f'./data_preprocessed/{args.data_name}/train_leave_one_out_timestamped.csv')
train_data = train_data[train_data.rating >= 4]
valid_data = pd.read_csv(f'./data_preprocessed/{args.data_name}/validation_leave_one_out_timestamped.csv')
valid_data = valid_data[valid_data.userId.isin(train_data.userId.unique())]
test_data = pd.read_csv(f'./data_preprocessed/{args.data_name}/test_leave_one_out_timestamped.csv')
test_data = test_data[test_data.userId.isin(train_data.userId.unique())]
strong_generalization_set = pd.read_csv(f'./data_preprocessed/{args.data_name}/strong_generalization_set_timestamped.csv')
strong_generalization_set = strong_generalization_set[strong_generalization_set.rating >= 4]
strong_generalization_set = strong_generalization_set.groupby('userId').filter(lambda x: len(x) > 5)
data = pd.concat([train_data, strong_generalization_set])


# %%


def tokenize_prompt(tokenizer,prompt,max_l):
    encodings = tokenizer([prompt],padding=True, truncation=True,max_length=max_l,
                          return_tensors='pt')

    return encodings
max_l = 1778



"""
when pushing stuff down you want the diff to be positive 
when pushing stuff up you want the diff to be negative
"""
def print_difs_genre(s1,s2,device = 0,genre = 'Action' ):
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
    l = torch.topk(torch.softmax(out_s1.logits,dim = 1),1000)
    indices_1 = l.indices[0]
    movie_titles_1 = [movie_id_to_title[movie_id_map[index_1.item()]] for index_1 in indices_1]
    l2 = torch.topk(torch.softmax(out_s2.logits,dim = 1),1000)
    indices_2 = l2.indices[0]
    movie_titles_2 = [movie_id_to_title[movie_id_map[index_2.item()]] for index_2 in indices_2]
    genre_movies_1 = [movie_title_to_genre[movie_title] for movie_title in movie_titles_1]
    genre_movies_2 = [movie_title_to_genre[movie_title] for movie_title in movie_titles_2]
    ndcg_1 = genrewise_ndcg(genre_movies_1,genre,min_k = 0,max_k = 100)
    ndcg_2 = genrewise_ndcg(genre_movies_2,genre,min_k = 0,max_k = 100)
    delta = ndcg_1 - ndcg_2    
    average_rankings1 = genrewise_average_rankings(genre_movies_1,genre,topk= 100)
    average_rankings2 = genrewise_average_rankings(genre_movies_2,genre,topk = 100)
    rankings1 = [(movie_title,genre) for movie_title,genre in zip(movie_titles_1,genre_movies_1)]
    rankings2 = [(movie_title,genre) for movie_title,genre in zip(movie_titles_2,genre_movies_2)]
    delta_rankings = average_rankings1 - average_rankings2
    

    return delta,delta_rankings,rankings1,rankings2



def print_recs(s1,device = 0,ranking = 10): 
    s = ''
    rankings = []
    genres = []
    s1_encodings = tokenize_prompt(tokenizer,s1,max_l)
    #move encodings to device
    for k in s1_encodings:
        s1_encodings[k] = s1_encodings[k].to(device)
    out_s1 = model(**s1_encodings)
    l = torch.topk(torch.softmax(out_s1.logits,dim = 1),200)
    indices_1 = l.indices[0]

    for i in range(len(indices_1)):
            index_1 = indices_1[i].item()
            movie_id1 = movie_id_map[index_1]
            # print(i+1,movie_id_to_title[movie_id1])
            movie_title = movie_id_to_title[movie_id1]
            rankings.append(movie_id_to_title[movie_id1])

            genres.append(movie_id_to_genre[movie_id1])
            if i == ranking:
                target_movie = movie_title
            # print('='*20)
    return target_movie,s,rankings ,genres


def print_recs_mask(s1,device = 0,ranking = 10): 
    s = ''
    rankings = []
    genres = []
    s1_encodings = tokenize_prompt(tokenizer,s1,max_l)
    #move encodings to device
    for k in s1_encodings:
        s1_encodings[k] = s1_encodings[k].to(device)
    out_s1 = model(**s1_encodings)
    l = torch.topk(torch.softmax(out_s1.logits,dim = 1),200)
    indices_1 = l.indices[0]

    for i in range(len(indices_1)):
            index_1 = indices_1[i].item()
            movie_id1 = movie_id_map[index_1]
            # print(i+1,movie_id_to_title[movie_id1])
            movie_title = movie_id_to_title[movie_id1]
            rankings.append(movie_id_to_title[movie_id1])

            genres.append(movie_id_to_genre[movie_id1])
            if i == ranking:
                target_movie = movie_title
            # print('='*20)
    return target_movie,s,rankings ,genres


def print_masked(s1,s2,device = 0,genre = 'Action' ,user_id = 1):
    rankings = []
    genres = []
    data_movies = data[data.userId == user_id].movieId.unique()

    #convet movie ids to indeces 
    reverse_movie_id_map = {v: k for k, v in movie_id_map.items()}
    data_movies = [reverse_movie_id_map[movie_id] for movie_id in data_movies]
    s_1_encodings = tokenize_prompt(tokenizer,s1,max_l)
    s_2_encodings = tokenize_prompt(tokenizer,s2,max_l)
    for k in s_1_encodings:
        s_1_encodings[k] = s_1_encodings[k].to(device)
    for k in s_2_encodings:
        s_2_encodings[k] = s_2_encodings[k].to(device)
    out_s1 = model(**s_1_encodings)
    out_s2 = model(**s_2_encodings)
    #make the watched movies = 0
    out_s1.logits[:,data_movies] = -np.inf
    out_s2.logits[:,data_movies] = -np.inf
    l = torch.topk(out_s1.logits,1000)
    indices_1 = l.indices[0]
    movie_titles_1 = [movie_id_to_title[movie_id_map[index_1.item()]] for index_1 in indices_1]
    l2 = torch.topk(out_s2.logits,1000)
    indices_2 = l2.indices[0]
    movie_titles_2 = [movie_id_to_title[movie_id_map[index_2.item()]] for index_2 in indices_2]
    genre_movies_1 = [movie_title_to_genre[movie_title] for movie_title in movie_titles_1]
    genre_movies_2 = [movie_title_to_genre[movie_title] for movie_title in movie_titles_2]
    ndcg_1 = genrewise_ndcg(genre_movies_1,genre,min_k = 0,max_k = 50)
    ndcg_2 = genrewise_ndcg(genre_movies_2,genre,min_k = 0,max_k = 50)
    delta = ndcg_1 - ndcg_2    
    average_rankings1 = genrewise_average_rankings(genre_movies_1,genre,topk= 50)
    average_rankings2 = genrewise_average_rankings(genre_movies_2,genre,topk = 50)
    average_recalls1 = genrewise_recall(genre_movies_1,genre,min_k = 0,max_k = 50)
    average_recalls2 = genrewise_recall(genre_movies_2,genre,min_k = 0,max_k = 50)
    rankings1 = [(movie_title,genre) for movie_title,genre in zip(movie_titles_1,genre_movies_1)]
    rankings2 = [(movie_title,genre) for movie_title,genre in zip(movie_titles_2,genre_movies_2)]
    delta_rankings = average_rankings1 - average_rankings2
    delta_recalls = average_recalls1 - average_recalls2
    
    
    return delta,delta_rankings,delta_recalls, rankings1,rankings2


# %%
def print_difs(s1,s2,device = 0,target_rank = 10 ):
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
    #get delta 
    delta = target_rank - pos
    for i in range(len(indices_2)):
            index_2 = indices_2[i].item()
            movie_id2 = movie_id_map[index_2]
            movie_title = movie_id_to_title[movie_id2] 
            genres.append(movie_id_to_genre[movie_id2])
            rankings.append(movie_id_to_title[movie_id2])

    return delta,pos,rankings,genres


def print_recs(s1,device = 0,ranking = 10): 
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
            # print(i+1,movie_id_to_title[movie_id1])
            movie_title = movie_id_to_title[movie_id1]
            rankings.append(movie_id_to_title[movie_id1])
            genres.append(movie_id_to_genre[movie_id1])
            if i == ranking:
                target_movie = movie_title
            # print('='*20)
    return target_movie,s,rankings ,genres





# %%


def print_rankings(rankings1,rankings2):
    k = 1
    for (movie1,ranking1),(movie_2,ranking_2) in zip(rankings1,rankings2):
        print(f"{k} {movie1=} {ranking1=} \n{movie_2=} {ranking_2=}")
        k+=1
        print('='*20)
        if k == 20:
            break
        

def print_rankings_seperate(rankings1,rankings2):
    k = 1
    for (movie1,ranking1) in rankings1:
        print(f"{k} {movie1=} {ranking1=}")
        k+=1
        print('='*20)
        if k == 20:
            break
    for (movie_2,ranking_2) in rankings2:
        print(f"{k} {movie_2=} {ranking_2=}")
        k+=1
        print('='*20)
        if k == 20:
            break

# %%

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


for k in (pbar:=tqdm(sampled_keys)):
    target_movie, ranking,rankings,o_genres  = print_recs(prompts[k],ranking = taget_rank)
    
    print(f"{target_movie=}")
    # rankings_s = '\n'.join(rankings[:taget_rank])
    
    # pprint(rankings[:20])
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
                # current top 20 movies: {rankings_s}
                
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
    delta, new_rank,new_rankings,generes = print_difs(s1 = prompts[k],s2 = gpt_output,target_rank = taget_rank)
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

#all dataframe 
all_df = pd.concat([original_prompts_df,gpt_outputs_df,original_rankings_df,changed_rankings_df,deltas_df,blue_scores_df],axis = 1)
all_df.columns = ['Original Summary','changed Summary','Original Ranking','Changed Ranking','Delta','blue_scores']

#save dataframe 
all_df.to_csv(f'/home/mila/e/emiliano.penaloza/LLM4REC/results/{args.data_name}/gpt4_results_{taget_rank}_{direction}.csv')
logging.info(f'Saved results to /home/mila/e/emiliano.penaloza/LLM4REC/results/{args.data_name}/gpt4_results_{taget_rank}_{direction}.csv')


deltas = [delta for delta in deltas.values()]

# all_df = pd.read_csv(f'/home/mila/e/emiliano.penaloza/LLM4REC/results/{args.data_name}/gpt4_results_20_down.csv')
sns.set_style('whitegrid')
sns.set_context('paper')
plt.figure(figsize = (10,10))
sns.histplot(deltas)
plt.axvline(np.mean(deltas),color = 'red',label = f'Mean: {np.mean(deltas)}')
plt.axvline(np.median(deltas),color = 'green',label = f'Median: {np.median(deltas)}')
plt.legend()
plt.savefig(f'/home/mila/e/emiliano.penaloza/LLM4REC/results/ml-1m/figures/gpt4_20_down.png')
#grid off 
plt.grid(False)
#add a title


