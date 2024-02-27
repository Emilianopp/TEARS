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
from model.MF import MatrixFactorizationLLM,T5MappingVAE,sentenceT5Classification,sentenceT5ClassificationFrozen


os.chdir('../')


#set export MASTER_ADDR="127.0.0.1" as in os 
os.environ['MASTER_ADDR'] = '127.0.0.1'

#same with master port  export MASTER_PORT=$(expr 10000)

os.environ['MASTER_PORT'] = '10000'

    


args = parse_args(notebook=True)

debug_string  = "debug" if args.debug else ""
args.debug_prompts = True  
args.loss = 'bce_softmax'

if args.debugger:
    import debugpy
    debugpy.listen(5678)
    print('attach please')
    debugpy.wait_for_client()
    
def cleanup():
    dist.destroy_process_group()
    
def setup(rank,world_size):
    assert torch.distributed.is_available()
    print("PyTorch Distributed available.")
    print("  Backends:")
    print(f"    Gloo: {torch.distributed.is_gloo_available()}")
    print(f"    NCCL: {torch.distributed.is_nccl_available()}")
    print(f"    MPI:  {torch.distributed.is_mpi_available()}")
 
    # NOTE: the env:// init method uses FileLocks, which sometimes causes deadlocks due to the
    # distributed filesystem configuration on the Mila cluster.
    # For multi-node jobs, use the TCP init method instead.
   

     # DDP Job is being run via `srun` on a slurm cluster.
    # rank = int(os.environ["SLURM_PROCID"])
    # local_rank = int(os.environ["SLURM_LOCALID"])
    # world_size = int(os.environ["SLURM_NTASKS"])
 
     # SLURM var -> torch.distributed vars in case needed
     # NOTE: Setting these values isn't exactly necessary, but some code might assume it's
     # being run via torchrun or torch.distributed.launch, so setting these can be a good idea.
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


# setup(rank, world_size)



global_path = '/home/mila/e/emiliano.penaloza/LLM4REC'

args = parse_args(notebook=True)
load_dotenv()

key = os.getenv("OPEN-AI-SECRET")
openai.api_key = key

tokenizer = AutoTokenizer.from_pretrained("t5-large")
prompts,rec_dataloader,num_movies,val_dataloader,test_dataloader= load_data_vae_way(args,tokenizer,rank,world_size)

# path = f'{args.scratch}/saved_model/{args.data_name}/t5_classification_embedding_module_t5_frozen_l2_lambda_0.0001_lora_r_32_scheduler_cosine_warmup.csv.pt'

# model = sentenceT5Classification.from_pretrained('t5-large', num_labels=num_movies)

# lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=32, lora_alpha=args.lora_alpha, lora_dropout=args.dropout,
#                             target_modules=["q", "v"],
#                             modules_to_save=['classification_head'])
# model = get_peft_model(model, lora_config)


# path = f'{args.scratch}/saved_model/{args.data_name}/t5_classification_embedding_module_t5_frozen_l2_lambda_0.0001_lora_r_32_scheduler_cosine_warmup.csv.pt'
# # model = sentenceT5Classification.from_pretrained('t5-large', num_labels=num_movies)
# model = sentenceT5ClassificationFrozen.from_pretrained('t5-large', num_labels=num_movies, classifier_dropout = args.dropout)


# # lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=32, lora_alpha=args.lora_alpha, lora_dropout=args.dropout,
# #                             target_modules=["q", "v"],
# #                             modules_to_save=['classification_head'])
# # model = get_peft_model(model, lora_config)


# model.load_state_dict(torch.load(path ,map_location=torch.device('cpu')))
# # model.to(rank)


path = f'{args.scratch}/saved_model/{args.data_name}/t5_classification_embedding_module_t5_frozen_l2_lambda_0.0001_lora_r_32_scheduler_cosine_warmup.csv.pt'

# model = sentenceT5Classification.from_pretrained('t5-large', num_labels=num_movies)
model = sentenceT5ClassificationFrozen.from_pretrained('t5-large', num_labels=num_movies, classifier_dropout = args.dropout)
model.eval()
model.to(rank)
model.load_state_dict(torch.load(path ,map_location=torch.device('cpu')))




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

def tokenize_prompt(tokenizer,prompt,max_l):
    encodings = tokenizer([prompt],padding=True, truncation=True,max_length=max_l,
                          return_tensors='pt')

    return encodings

# %%

# %%
import torch
from tqdm import tqdm
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns

import pandas as pd
# args.data_name = 'ml-1m'
# all_df = pd.read_csv(f'/home/mila/e/emiliano.penaloza/LLM4REC/results/ml-1m/gpt4_results_20_up.csv')
p = '/home/mila/e/emiliano.penaloza/LLM4REC/results/ml-1m/'

df_list = [p + 'gpt4_results_20_up.csv',p + 'gpt4_results_19_down.csv',p + 'gpt4_results_49_up.csv',p + 'gpt4_results_49_down.csv']
ks = [18,18,48,48]
directions = ['up','down','up','down']

# %%
# deltas = all_df.Delta
#do a hist of running deltas put an avline in the mean and median put in the legend the values of the mena and media n
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
def plot_deltas(deltas, direction , k,title='Histogram of Deltas',color = 'skyblue'):
    # Set seaborn style and context
    deltas = deltas[deltas > np.percentile(deltas,2.5)]

    sns.set_style('white')
    sns.set_context('paper')
    #get data 90th percentiles on the lower tail 

    # Create a figure
    plt.figure(figsize=(10, 10))

    # Create a histogram of deltas with seaborn, adjust opacity with alpha
    sns.histplot(deltas, color=color, alpha=0.5, kde=True)

    # Add vertical lines for the mean and median
    plt.axvline(np.mean(deltas), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(deltas):.2f}')
    plt.axvline(np.median(deltas), color='green', linestyle='dashed', linewidth=2, label=f'Median: {np.median(deltas):.2f}')
    plt.legend()
    plt.xticks(fontsize=14)

    plt.yticks(fontsize=14)
    plt.grid(False)
    plt.xlabel('Change in Ranking Position', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.xlabel('Change in Ranking Position')
    #save the plot to /home/mila/e/emiliano.penaloza/LLM4REC/results/ml-1m/figures
    plt.savefig(f'/home/mila/e/emiliano.penaloza/LLM4REC/results/ml-1m/figures/hist_deltas_{direction}_{k}.png')
    plt.show()

# %%
import nltk
from nltk.translate.bleu_score import sentence_bleu
tokenizer = T5Tokenizer.from_pretrained('t5-large')

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

    return delta
max_l = 1778

def bleu_score(x):
    return sentence_bleu([x['Original Summary'].split()],x['changed Summary'].split())



def edit_distance_words(str1, str2):
    words1 = str1.split()
    words2 = str2.split()

    edit_distance = nltk.edit_distance(words1, words2)
    
    return edit_distance


def examine_rest_of_rankings(df,k):
    user_deltas = defaultdict(list)
    for m,(original_summary,changed_summary )in tqdm(enumerate(zip(df['Original Summary'],df['changed Summary']))):

        for i in range(k):
            user_deltas[m].append(print_difs(original_summary,changed_summary,device = rank,target_rank = i))
    return user_deltas
def tokenize_prompt(tokenizer,prompt,max_l):
    encodings = tokenizer([prompt],padding=True, truncation=True,max_length=max_l,
                          return_tensors='pt')

    return encodings
def get_paper_metrics( dfs,ks,directions,plot = False,debug =False,model_type ='fine-tuned'): 
    out_dict= defaultdict(dict)
    for i,path in enumerate(dfs):
        df = pd.read_csv(path)
        if debug: 
            #get a subset of 5 rows from the dataframe 
            df = df.sample(2)
        outputs = df['changed Summary']
        original = df['Original Summary']
        deltas = []
        for ori,chan in zip(original,outputs):
            delta = print_difs(ori,chan,ks[i]+1)
            deltas.append(delta)
        out_dict[path]['delta'] = deltas
        out_dict[path]['delta_mean'] = np.mean(deltas)
        out_dict[path]['delta_std'] = np.std(deltas)
        out_dict[path]['delta_median'] = np.median(deltas)
        if plot: 
            plot_deltas(deltas,k = ks[i],direction=directions[i],title =model_type)
        df['blue']= df.apply(bleu_score,axis=1)
        out_dict[path]['bleu_mean'] = df['blue'].mean()
        df['edit_distance'] = df.apply(lambda x: edit_distance_words(x['Original Summary'],x['changed Summary']),axis=1)
        out_dict[path]['edit_distance_mean'] = df['edit_distance'].mean()
        out_dict[path]['edit_distance_median'] = df['edit_distance'].median()
        
        rest_of_rankings = examine_rest_of_rankings(df,ks[i])
        #flatten the dict of lists into a list 
        rest_of_rankings = [v for k,v in rest_of_rankings.items()]
        #get the median mean and also take the absolute value and then take the median and mean report all 4 values in outpud_dict 
        rest_of_rankings = np.array(rest_of_rankings)
        out_dict[path]['rest_of_rankings_mean'] = rest_of_rankings.mean()
        out_dict[path]['rest_of_rankings_median'] = np.median(rest_of_rankings)
        out_dict[path]['rest_of_rankings_mean_abs'] = np.abs(rest_of_rankings).mean()
        out_dict[path]['rest_of_rankings_median_abs'] = np.median(np.abs(rest_of_rankings))
    return out_dict
        
        


        



# %%
path = f'{args.scratch}/saved_model/{args.data_name}/t5_classification_embedding_module_t5_frozen_l2_lambda_0.0001_lora_r_32_scheduler_cosine_warmup.csv.pt'
# model = sentenceT5Classification.from_pretrained('t5-large', num_labels=num_movies)
model = sentenceT5ClassificationFrozen.from_pretrained('t5-large', num_labels=num_movies, classifier_dropout = args.dropout)


# %%
model.to(rank)
model.eval()
out_dict = get_paper_metrics(df_list,ks,directions,plot = True,model_type='frozen')

# %%
#pickle out the out_df 
with open(f'/home/mila/e/emiliano.penaloza/LLM4REC/results/ml-1m/frozen_metrics.pkl','wb') as f:
    pickle.dump(out_dict,f)
#save the out dict as a dataframe with the column with the name of the direction and k 
out_df = pd.DataFrame(out_dict).T
out_df.to_csv(f'/home/mila/e/emiliano.penaloza/LLM4REC/results/ml-1m/frozen_metrics.csv')


# out_df


# path = f'{args.scratch}/saved_model/{args.data_name}/t5_classification_please_embedding_module_t5_classification_l2_lambda_0.01_lora_r_32_scheduler_cosine_warmup_0.001.csv.pt'

# model = sentenceT5Classification.from_pretrained('t5-large', num_labels=num_movies)

# lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=32, lora_alpha=16, lora_dropout=0,
#                             target_modules=["q", "v"],
#                             modules_to_save=['classification_head'])
# model = get_peft_model(model, lora_config)


# model.load_state_dict(torch.load(path ,map_location=torch.device('cpu')))
# model.to(rank)
# model.eval()
# out_dict = get_paper_metrics(df_list,ks,directions,plot = True,model_type='fine-tuned')


# with open(f'/home/mila/e/emiliano.penaloza/LLM4REC/results/ml-1m/frozen_metrics.pkl','wb') as f:
#     pickle.dump(out_dict,f)
# #save the out dict as a dataframe with the column with the name of the direction and k 
# out_df = pd.DataFrame(out_dict).T
# out_df.to_csv(f'/home/mila/e/emiliano.penaloza/LLM4REC/results/ml-1m/fine_tuned_metrics.csv')
