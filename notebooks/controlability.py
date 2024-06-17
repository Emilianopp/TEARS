import sys 
import os
PATH = '/home/user/NEW_MODEL_CACHE/'
os.environ['TRANSFORMERS_CACHE'] = '/home/mila/e/emiliano.penaloza/scratch/'
os.environ['HF_HOME'] = '/home/mila/e/emiliano.penaloza/scratch/models'
os.environ['HF_DATASETS_CACHE'] = '/home/mila/e/emiliano.penaloza/scratch/models'
os.environ['TORCH_HOME'] = '/home/mila/e/emiliano.penaloza/scratch/models'
# os.chdir('../')
from trainer.transformer_utilts import *
sys.path.append("..")
import torch.optim as optim
from collections import Counter
from tqdm import tqdm
import pandas as pd
import time 
import torch
from dotenv import load_dotenv
from model.MF import get_model,get_tokenizer
from helper.dataloader import *
import wandb
from tqdm import tqdm
from peft import LoraConfig, TaskType, PeftModel,get_peft_model
from transformers import T5Tokenizer ,AutoTokenizer
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoConfig
from transformers import AutoModel
from model.eval_model import *
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_flash_sdp(False)
import os

rank =0
world_size = 1 
args = parse_args(notebook=True)
args.embedding_module="t5_film"


# %%
tokenizer = get_tokenizer(args)
prompts,rec_dataloader,augmented_dataloader,num_movies,val_dataloader,test_dataloader,val_data_tr,test_data_tr= load_data(args,tokenizer,rank,world_size)
model,lora_config = get_model(args, tokenizer, num_movies, 0, world_size)
model.to(rank)


# %%
model_p ='t5_classification_fixed_data_embedding_module_t5_film_l2_lambda_0.01_lora_r_16_scheduler_None_0.0001_0.9.csv.pt'
p = f'{args.scratch}/saved_model/{args.data_name}/' + model_p
model.load_state_dict(torch.load(p))

# %%
item_title_dict = map_id_to_title(args.data_name)
item_genre_dict = map_id_to_genre(args.data_name)
eval_model = LargeScaleEvaluator(model,item_title_dict,item_genre_dict,tokenizer, 0,args)

# %%
counts = Counter(sum([v.split("|") for v in item_genre_dict.values()],[]))
#keep counts if above 200 
counts = {k:v for k,v in counts.items() if v > 100}
genre_set = list(counts.keys())
genre_set = ', '.join(genre_set)
k_list= list(prompts.keys())
k_sample = random.sample(k_list,500)

# %%
import random


user_labels = {}
for b in val_dataloader:
    user_ids = b['idx']
    for i,u in enumerate(user_ids):
        user_labels [u.item()] = b['labels_tr'][i].to(rank)

sampled_keys = random.sample(user_labels.keys(), 250)


# %%
eval_model.promptGPT(sampled_keys,prompts,user_labels)

# %%



