# %%
import os
import sys 
sys.path.append('../')
# print(f"{os.getcwd()=}")
from trainer.transformer_utilts import *


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
import torch
import time
import json
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd 
import debugpy
from dotenv import load_dotenv


import sys 
# sys.path.append("..")
import pickle
from peft import get_peft_model
import json
from tqdm import tqdm
import openai
import pandas as pd

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

args = parse_args(notebook=True)

debug_string  = "debug" if args.debug else ""
args.debug_prompts = True  
args.loss = 'bce_softmax'


rank = 0
world_size = 1


# setup(rank, world_size)



global_path = '/home/mila/e/emiliano.penaloza/LLM4REC'

args = parse_args(notebook=True)
load_dotenv()


tokenizer = AutoTokenizer.from_pretrained("t5-large")
prompts,rec_dataloader,num_movies,val_dataloader,test_dataloader= load_data_vae_way(args,tokenizer,rank,world_size)

path = f'{args.scratch}/saved_model/{args.data_name}/t5_classification_please_embedding_module_t5_classification_l2_lambda_0.01_lora_r_32_scheduler_cosine_warmup_0.001.csv.pt'
path = f'{args.scratch}/saved_model/{args.data_name}/t5_classification_embedding_module_t5_classification_l2_lambda_0.0001_lora_r_16_scheduler_cosine_warmup.csv.pt'
# path  = f'{args.scratch}/saved_model/{args.data_name}/t5_classification_embedding_module_t5_classification_l2_lambda_0.001_lora_r_16_scheduler_cosine_warmup_0.001_0.2.csv.pt'
# path = f'{args.scratch}/saved_model/{args.data_name}/t5_classification_please_embedding_module_t5_classification_l2_lambda_0.01_lora_r_32_scheduler_cosine_warmup_0.001.csv.pt'

#this is the best model
path = f'{args.scratch}/saved_model/{args.data_name}/t5_classification_please_embedding_module_t5_classification_l2_lambda_0.01_lora_r_32_scheduler_cosine_warmup_0.001.csv.pt'


model = sentenceT5Classification.from_pretrained('t5-large', num_labels=num_movies)

lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=32, lora_alpha=16, lora_dropout=0,
                            target_modules=["q", "v"],
                            modules_to_save=['classification_head'])
model = get_peft_model(model, lora_config)


model.load_state_dict(torch.load(path ,map_location=torch.device('cuda')))
model.to(0)





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
# path = f'{args.scratch}/saved_model/{args.data_name}/t5_classification_embedding_module_t5_frozen_l2_lambda_0.0001_lora_r_32_scheduler_cosine_warmup.csv.pt'
# # path = f'{args.scratch}/saved_model/{args.data_name}/t5_classification_please_embedding_module_t5_classification_l2_lambda_0.01_lora_r_32_scheduler_cosine_warmup_0.001.csv.pt'
# model = sentenceT5ClassificationFrozen.from_pretrained('t5-large', num_labels=num_movies, classifier_dropout = args.dropout)
# model.load_state_dict(torch.load(path ,map_location=torch.device('cpu')))


# %%
def MRR_at_k(X_pred, heldout_batch, k=100, mean = True):
    '''
    Mean Reciprocal Rank@k
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]

    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)

    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    tp = 1. / (np.arange(1, k + 1))

    RR = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                        idx_topk] * tp).max(axis=1)

    return np.mean(RR) if mean else RR

# %%
def MAP_at_k(X_pred, heldout_batch, k=100):
    '''
    Mean Average Precision@k
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]

    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)

    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    tp = np.cumsum(heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk], axis=1) / (np.arange(1, k + 1))
    AP_at_k = (heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk] * tp).sum(axis=1) / (heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk].sum(axis=1) + 1e-10)

    return np.mean(AP_at_k)

# %%
#reverse profil2id
# id2profile = {v:k for k,v in profile2id.items()}
with open('/home/mila/e/emiliano.penaloza/LLM4REC/vae/ml-1m/pro_sg_text/profile2id.pkl','rb') as f:
    profile2id = pickle.load(f)
    #reverse the map
    id2profile = {v:k for k,v in profile2id.items()}
with open('/home/mila/e/emiliano.penaloza/LLM4REC/vae/ml-1m/pro_sg_text/show2id.pkl','rb') as f:
    movie_id_map = pickle.load(f)
    #reverse the map

    movie_id_map = {v:k for k,v in movie_id_map.items()}
movie_set = set([x for x in movie_id_map.keys()])


# %%


def get_dataloader(data,rank,world_size,bs,encodings,nonzer_indeces_train):
    rec_dataset =  DataMatrix(data,encodings,nonzer_indeces_train)
    rec_dataloader = DataLoader(rec_dataset, batch_size=bs, collate_fn= None , num_workers = 0,pin_memory=False,
                                shuffle=True) 
    return rec_dataloader



# %%
k=30

loader = data.DataLoader('/home/mila/e/emiliano.penaloza/LLM4REC/vae/ml-1m')

train_data = loader.load_data('train')
vad_data_tr, vad_data_te = loader.load_data('validation')

args.bs = 249
#concat both the vad and test set 
valid_data =  vad_data_te

test_data_tr, test_data_te = loader.load_data('test')


test_data =   test_data_te

num_users = train_data.shape[0]
num_movies = train_data.shape[1]

nonzer_indeces_train = {i:v for i,v in enumerate(set(train_data.sum(axis =1 ).nonzero()[0]))}
nonzer_indeces_valid = {i:v for i,v in enumerate(set(valid_data.sum(axis =1 ).nonzero()[0]))}
nonzer_indeces_test = {i:v for i,v in enumerate(set(test_data.sum(axis = 1).nonzero()[0]))}



#pickle movie_id_map if it doesnt exist to saved_iser_summary 
with open ('/home/mila/e/emiliano.penaloza/LLM4REC/vae/ml-1m/pro_sg/profile2id.pkl','rb') as f:
    profile2id = pickle.load(f)

with open(f'./saved_user_summary/{args.data_name}/user_summary_gpt4_.json','r') as f:
    prompts = json.load(f)
    #make int 
# prompts = {profile2id[int(float(k))]:v for k,v in prompts.items() } 
prompts = {profile2id[int(float(k))]:v for k,v in prompts.items() } 

with open (f'/home/mila/e/emiliano.penaloza/LLM4REC/saved_user_summary/ml-100k/user_summary_gpt4_30_new.json','r') as f:
    subset_prompts = json.load(f)
    #make int
    subset_prompts = {profile2id[(float(k))]:v for k,v in subset_prompts.items() }

#with they keys in subset prompts replace the values of the keys in prompts
if k != 50:
    prompts.update(subset_prompts)

subset_keys = set(subset_prompts.keys())



#sort prompt dict in ascending order 
promp_list = [v for k,v in prompts.items()]
max_l = max([len(i.split()) for i in promp_list])
print("Max Prompt Length",max_l)

encodings = {k: tokenizer([v],padding='max_length', return_tensors='pt',truncation=True,max_length=max_l) for k, v in sorted(prompts.items())} 
#squeze the encodings 
encodings = {k: {k1: v1.squeeze(0) for k1, v1 in v.items()} for k, v in encodings.items()}


print(f"Number of Users is {num_users=}")
print(f"Number of Movies is {num_movies=}")
rec_dataloader = get_dataloader(train_data,rank,world_size,args.bs,encodings,nonzer_indeces_train)
val_dataloader = get_dataloader(valid_data,rank,world_size,args.bs,encodings,nonzer_indeces_valid)
test_dataloader = get_dataloader(test_data,rank,world_size,args.bs,encodings,nonzer_indeces_test)


    

# %%
torch.cuda.set_device(rank)
metrics = defaultdict(list)
# user_idx_map = val_dataloader.dataset.user_idx_map
ndcgs_20 = {}
prompt_set_df = pd.read_csv(f'/home/mila/e/emiliano.penaloza/LLM4REC/data_preprocessed/ml-1m/prompt_set_timestamped.csv')
# prompt_set_df = prompt_set_df[prompt_set_df.rating > 3]
movie_set = set([x for x in movie_id_map.keys()])
user_id_set = set()
user_ids_l = []
ss = 0
lengths = []
u_ids_text = {}
#set torch seed 
model.to(rank)
torch.manual_seed(2024)
rn20_user = {}

num_items ={ }
with torch.no_grad():
    model.eval()
    for b,item in enumerate(test_dataloader):
        
        user_ids = sum(item.pop('idx').cpu().tolist(),[])
  
        user_id_set.update( user_ids)
        #move to device
        item = {k:v.to(rank) for k,v in item.items()}
        # labels_pos = torch.where(item['labels'][i] == 1)[0]
        movie_emb_clean = model(**item)[0]
        # movie_emb_clean[:,labels_pos] = -torch.inf
        #copy the labels 
        user_ids_l.append(user_ids)
        test_rows = test_data_tr[user_ids].toarray()
      

        #make it zero where the user has seen the movie
        movie_emb_clean[np.where(test_rows > 0)] = -torch.inf
        
        for i in range(len(user_ids)):
            
            
            user_id = id2profile[user_ids[i]]
            # assert counts[user_ids[i]] == item['labels'][i].sum().item(), f"{counts[user_ids[i]]=} {item['labels'][i].sum().item()=}"
            u_ids_text[user_id] = user_id
            # user_id = i
            lengths.append(len(prompts[user_ids[i]].split()))
            prompt_set = prompt_set_df[(prompt_set_df.userId == user_id)].movieId.tolist()

            num_items[user_id] = len(prompt_set)
            # num_items_eval[user_id] = item['labels'][i].sum().item()
            # num_items.append( item['labels'][i].sum().item())
    
        
        labels =item['labels'].cpu().numpy()

        recon = movie_emb_clean.cpu().numpy()
        metrics['ndcgs_10'].append(NDCG_binary_at_k_batch(recon,labels,k=10).mean().tolist())
        metrics['ndcgs_20'].append(NDCG_binary_at_k_batch(recon,labels,k=20))
        metrics['r20_no'].append(Recall_at_k_batch(recon,labels,k=20,mean=False))
        metrics['ndcgs_50'].append(NDCG_binary_at_k_batch(recon,labels,k=50).mean().tolist())

        # metrics['ndcgs_10'].append(NDCG_binary_at_k_batch(recon,test_rows,k=10).mean().tolist())
        # metrics['ndcgs_20'].append(NDCG_binary_at_k_batch(recon,test_rows,k=20).mean().tolist())
        # metrics['ndcgs_50'].append(NDCG_binary_at_k_batch(recon,test_rows,k=50).mean().tolist())

        # metrics['ndcgs_10'].append(NDCG_binary_at_k_batch(recon,labels,k=10).mean().tolist())
        # metrics['ndcgs_20'].append(NDCG_binary_at_k_batch(recon,labels,k=20).mean().tolist())
        # metrics['ndcgs_50'].append(NDCG_binary_at_k_batch(recon,labels,k=50).mean().tolist())
        metrics['mrr@10'].append(MRR_at_k(recon,labels,k=10))
        metrics['mrr@20'].append(MRR_at_k(recon,labels,k=20))
        metrics['mrr@50'].append(MRR_at_k(recon,labels,k=50))
        # metrics['map@10'].append(MAP_at_k(recon,labels,k=10))
        # metrics['map@20'].append(MAP_at_k(recon,labels,k=20))
        # metrics['map@50'].append(MAP_at_k(recon,labels,k=50))
        # metrics['ru_10_u'].append(Recall_at_k_batch_unormalized(recon,labels,k=10))
        # metrics['ru_20_u'].append(Recall_at_k_batch_unormalized(recon,labels,k=20))
        # metrics['ru_50_u'].append(Recall_at_k_batch_unormalized(recon,labels,k=50))
        metrics['rn_10_n'].append(Recall_at_k_batch(recon,labels,k=10))
        metrics['rn_20_n'].append(Recall_at_k_batch(recon,labels,k=20))
        metrics['rn_50_n'].append(Recall_at_k_batch(recon,labels,k=50))
        
        #make a dict that goes user_id:ndcg@20 
        for i,user_id in enumerate(user_ids):
            ndcgs_20[user_id] = metrics['ndcgs_20'][0][i]

            rn20_user[user_id] = metrics['r20_no'][0][i]    



    

# %%
# subset_keys
#extractt he users in subset keys from ndcg@20 
ndcg_20_subset = {k:v for k,v in ndcgs_20.items() if k in subset_keys}

# %%

np.mean([v for k,v in ndcg_20_subset.items()])
print(f"{np.mean([v for k,v in ndcg_20_subset.items()])=}")

# %%
# subset_keys
#extractt he users in subset keys from ndcg@20 
rn20_subset  = {k:v for k,v in rn20_user.items() if k in subset_keys}
np.mean([v for k,v in rn20_subset.items()])
print(f"{np.mean([v for k,v in rn20_subset.items()])=}")

