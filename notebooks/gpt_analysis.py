# %%
import os
import sys 
sys.path.append('../')
import json
import time
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import pipeline
import torch
import re
import openai
# from tenacity import retry, wait_exponential, stop_after_attempt
from tqdm import tqdm
from dotenv import load_dotenv
from helper.in_context_learning_batch import in_context_user_summary, in_context_retrieval
from helper.builder import build_context, build_movie_candidates
import torch
import time
import json
from torch.utils.data import Dataset, DataLoader
from data.dataloader import get_dataloader
from argparse import ArgumentParser
import os
import pandas as pd 
import debugpy
from dotenv import load_dotenv
from trainer.transformer_utilts import *
import sys 
sys.path.append("..")
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
from model.MF import MatrixFactorizationLLM,sentenceT5Classification
from model.decoderMLP import decoderMLP, decoderAttention
from model.transformerModel import movieTransformer
import argparse
from torch.optim.lr_scheduler import LambdaLR
from helper.eval_metrics import precision_at_k, recall_at_k, mapk, ndcg_k
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
os.chdir('../')

#set export MASTER_ADDR="127.0.0.1" as in os 
os.environ['MASTER_ADDR'] = '127.0.0.1'

#same with master port  export MASTER_PORT=$(expr 10000)

os.environ['MASTER_PORT'] = '10000'

    


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
    # os.environ["LOCAL_RANK"] = str(local_rank)
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



global_path = '/home/mila/e/emiliano.penaloza/LLM4REC'



# %%
load_dotenv()

key = os.getenv("OPEN-AI-SECRET")
openai.api_key = key

# %%
tokenizer = AutoTokenizer.from_pretrained("t5-large")
prompts,rec_dataloader,num_movies,training_items, val_items,test_items,val_dataloader,test_dataloader, strong_generalization_set_val_dataloader, strong_generalization_set_test_dataloader,\
        strong_generalization_set_val_items,strong_generalization_set_test_items\
    = load_data(args,tokenizer,rank,world_size)
path = f'{args.scratch}/saved_model/{args.data_name}/t5_classification_embedding_module_t5_classification_l2_lambda_0.01_lora_r_32_scheduler_cosine_warmup.csv.pt'
model = sentenceT5Classification.from_pretrained('t5-large', num_labels=num_movies)

lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=32, lora_alpha=16, lora_dropout=.2,
                            target_modules=["q", "v"],
                            modules_to_save=['classification_head'])
model = get_peft_model(model, lora_config)


model.load_state_dict(torch.load(path ,map_location=torch.device('cpu')))
model.to(rank)



# %%
model.eval()
movie_id_map =rec_dataloader.dataset.movie_id_map
#reverse the map
movie_id_map = {v: k for k, v in movie_id_map.items()}
movie_id_to_title = map_id_to_title('/home/mila/e/emiliano.penaloza/LLM4REC/data/ml-1m/movies.dat')
movie_id_to_genre = map_id_to_genre('/home/mila/e/emiliano.penaloza/LLM4REC/data/ml-1m/movies.dat')

def tokenize_prompt(tokenizer,prompt,max_l):
    encodings = tokenizer([prompt],padding=True, truncation=True,max_length=max_l,
                          return_tensors='pt')

    return encodings
max_l = 1778



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
    l = torch.topk(torch.softmax(out_s1.logits,dim = 1),1000)
    indices_1 = l.indices[0]
    l2 = torch.topk(torch.softmax(out_s2.logits,dim = 1),1000)
    indices_2 = l2.indices[0]

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
#randomly sample n keys from the prompts dic 
np.random.seed(2024)
sampled_keys = np.random.choice(list(prompts.keys()), args.num_samples, replace=False)
print(f"{args.num_samples=}")


gpt_outputs_d = defaultdict(list)
original_rankings = defaultdict(list)
changed_rankings = defaultdict(list)
original_genres = defaultdict(list)
changed_genres = defaultdict(list)
original_prompts = {}
user_target_movie ={}
deltas = {}
running_deltas = []
for k in (pbar:=tqdm(sampled_keys)):
    target_movie, ranking,rankings,o_genres  = print_recs(prompts[k],ranking = args.target_index)
    prompt = \
                f"""
                You will be fed user profiles which summarize user's movie preferences. Please make edits to them that such that you push
                the input movie as far {args.direction} as possible. Please maintain the summaries in the following format: 
                Summary: [Specific details about genres the user enjoys]. [Specific details of plot points the user seems to enjoy]. 
                [Specific details about genres the user does not enjoy]. [Specific details of plot points the user does not enjoy but other users may].
                You will be fed the summaries, the target movie, and the rankings. Please only output the full summaries.
                """
    user_prompt =\
                f"""
                Please only output the full summary with as little changes as possible
                Target movie to move {args.direction}: {target_movie}\n
                Summary: {prompts[k]}\n 
                Rankings: {ranking[:args.target_index]}\n
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
    

    gpt_output = openai.ChatCompletion.create(
                    model='gpt-4-1106-preview',
                    messages=msg,
                    max_tokens=300
                )['choices'][0]['message']['content']
    gpt_outputs_d[k].append(prompts[k])
    original_prompts[k] = prompts[k]
    delta, new_rank,new_rankings,generes = print_difs(s1 = prompts[k],s2 = gpt_output,target_rank = args.target_index)
    user_target_movie[k] = target_movie
    original_rankings[k].append(rankings)
    changed_rankings[k].append(new_rankings)
    deltas[k] = delta
    changed_genres[k].append(generes)   
    original_genres[k].append(o_genres)
    
    running_deltas.append(delta)
    
    pbar.set_description(f'Original Ranking: Delta: {delta} , Running Delta: {np.mean(running_deltas)}')
    

#make a dataframe for the dicts 
gpt_outputs_df = pd.DataFrame.from_dict(gpt_outputs_d,orient = 'index')
original_rankings_df = pd.DataFrame.from_dict(original_rankings,orient = 'index')
changed_rankings_df = pd.DataFrame.from_dict(changed_rankings,orient = 'index')
deltas_df = pd.DataFrame.from_dict(deltas,orient = 'index')
original_prompts_df = pd.DataFrame.from_dict(original_prompts,orient = 'index')

#all dataframe 
all_df = pd.concat([original_prompts_df,gpt_outputs_df,original_rankings_df,changed_rankings_df,deltas_df],axis = 1)
all_df.columns = ['Original Summary','changed Summary','Original Ranking','Changed Ranking','Delta']

#save dataframe 
all_df.to_csv(f'/home/mila/e/emiliano.penaloza/LLM4REC/results/{args.data_name}/gpt4_results{args.target_index}_{args.direction}.csv')



