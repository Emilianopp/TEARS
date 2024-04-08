
import sys
sys.path.append('../')
import pickle
from peft import get_peft_model
import json
from tqdm import tqdm
import openai
import pandas as pd
from helper.dataloader import load_pickle, map_title_to_id, map_id_to_title
from data.dataloader import get_dataloader
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
from torch.nn.parallel import DataParallel
from trainer.transformer_utilts import *
from peft import LoraConfig, TaskType
from transformers import T5ForSequenceClassification
#import mp 
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import accelerate
from trainer.losses.loss import *
# set visible cuda devices to one 


# %%

args = parse_args(notebook=True)

debug_string  = "debug" if args.debug else ""


def get_embeddings(prompts,max_l):
    if os.path.isfile(f'{args.scratch}/{args.data_name}/embedding_module_transformer{debug_string}.pt'):
        embeddings = torch.load(f'{args.scratch}/{args.data_name}/embedding_module{debug_string}.pt')
    else:
        print("Making Embeddings")
        prompt_dataset = PromptDataset(prompts)
        prompt_data_loader = DataLoader(prompt_dataset, batch_size=len(prompts)//2, shuffle=True)

        
        embeddings = make_embeddings(args.embedding_module,prompt_data_loader,max_l)
        torch.save(embeddings,f'{args.sctratch}/data_preprocessed/{args.data_name}/embedding_module{debug_string}.pt')
    return embeddings

def get_dataset(data,  num_movies,bs,encodings):
    rec_dataset =  RecDatasetFull(data,encodings, num_movies=num_movies)
    return rec_dataset

def load_datasets(args,tokenizer):
    # 1. Data Loading & Preprocessing

    train_data = pd.read_csv(f'../data_preprocessed/{args.data_name}/train_leave_one_out_timestamped.csv')
    valid_data = pd.read_csv(f'../data_preprocessed/{args.data_name}/validation_leave_one_out_timestamped.csv')
    test_data = pd.read_csv(f'../data_preprocessed/{args.data_name}/test_leave_one_out_timestamped.csv')
    strong_generalization_set = pd.read_csv(f'../data_preprocessed/{args.data_name}/strong_generalization_set_timestamped.csv')

    test_items = dict(test_data.sort_values(by='userId').groupby('userId')['movieId'].agg(list).reset_index().values)

    val_items = dict(valid_data.sort_values(by='userId').groupby('userId')['movieId'].agg(list).reset_index().values)
    strong_generalization_set_val = strong_generalization_set.sample(frac=0.5,random_state=42)
    strong_generalization_set_test = strong_generalization_set.drop(strong_generalization_set_val.index)
    strong_generalization_set_val_items = dict(strong_generalization_set_val.sort_values(by='userId').groupby('userId')['movieId'].agg(list).reset_index().values)
    strong_generalization_set_test_items = dict(strong_generalization_set_test.sort_values(by='userId').groupby('userId')['movieId'].agg(list).reset_index().values)


    with open(f'../saved_user_summary/{args.data_name}/user_summary_gpt4_{debug_string}.json','r') as f:
        prompts = json.load(f)

    #sort prompt dict in ascending order 
    prompts = {int(float(k)): v for k, v in sorted(prompts.items(), key=lambda item: float(item[0]))}
    #get max length of prompt
    max_l = max([len(v) for k,v in prompts.items()])
    
    #tokenize prompts
    promp_list = [v for k,v in prompts.items()]
    encodings = tokenizer(promp_list,padding=True, truncation=True)

    #do a training and testing split on the strong_generalization_set

    num_movies = max(set(train_data.movieId) | set(valid_data.movieId) | set(test_data.movieId)) + 1
    strong_generalization_set_val_data = get_dataset(strong_generalization_set_val,  num_movies, args.bs,encodings)
    strong_generalization_set_test_data = get_dataset(strong_generalization_set_test, num_movies, args.bs,encodings)
    train_dataset = get_dataset(train_data,  num_movies,  args.bs,encodings)
  
    test_dataset = get_dataset(test_data,  num_movies, args.bs,encodings)
    val_dataset = get_dataset(valid_data,  num_movies, args.bs,encodings)

    return prompts, train_dataset, num_movies, val_items, test_items, val_dataset, test_dataset, strong_generalization_set_val_data, strong_generalization_set_test_data, strong_generalization_set_val_items, strong_generalization_set_test_items

# %%
class costumTrainer(Trainer):
    def __init__(self,loss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = loss
    def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss = self.loss(logits,labels)
        return loss

# %%
tokenizer = T5Tokenizer.from_pretrained('t5-base')
    
prompts,rec_dataloader,num_movies,val_items,test_items,val_dataloader,test_dataloader, strong_generalization_set_val_dataloader, strong_generalization_set_test_dataloader,strong_generalization_set_val_items,strong_generalization_set_test_items\
        = load_datasets(args,tokenizer)

# %%

print(f"Number of Batches = {len(rec_dataloader)}")
model_name = f"{args.embedding_module}-base"
    

lora_model = AutoModelForSequenceClassification.from_pretrained('sentence-transformers/sentence-t5-base', num_labels=num_movies)
lora_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.1,
                                target_modules=["q", "v"],
                                bias="none",
                                modules_to_save=["classifier"])
lora_model = get_peft_model(lora_model, lora_config)


# %%
from torch.nn.utils.rnn import pad_sequence

class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):

        # Separate the dictionary elements of each item in the batch
        keys = batch[0].keys()
        batch_dict = {key: [item[key] for item in batch] for key in keys}

        # Pad the sequences in the 'input_ids' and 'attention_mask' fields
        for key in ['input_ids', 'attention_mask']:
            batch_dict[key] = pad_sequence(batch_dict[key], batch_first=True, padding_value=self.pad_id)

        # Stack the 'labels' tensors
        batch_dict['labels'] = torch.stack(batch_dict['labels'])

        return batch_dict

# Usage:
# collator = Collator(pad_id=0)
# data_loader = DataLoader(dataset, batch_size=32, collate_fn=collator)

# %%
def compute_metrics(pred):
    labels = pred.label_ids
    # print(f"{labels.shape=}")
    preds = pred.predictions
    # print(f"{preds=}")
    movie_emb = torch.argsort(movie_emb)
            # Extract top 20 movies for each user into the top_movies dict 

    movie_emb =  movie_emb[:][-20:].cpu().numpy()

    recall = recall_at_k(movie_emb, preds, 20)
    ndcg_at_10 = ndcg_k(movie_emb, preds, 10)
    return {
        'recall_at_10': recall,
        'ndcg_at_10': ndcg_at_10
    }

# %%
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=50,
    remove_unused_columns=False,
    evaluation_strategy="steps",
    eval_steps = 1
)


trainer = costumTrainer(
    loss = get_loss('bce_softmax'),
    model=lora_model,
    args=training_args,
    train_dataset=rec_dataloader,
    data_collator=Collator(0),
    eval_dataset=strong_generalization_set_val_dataloader,
    compute_metrics= compute_metrics

    
)


# %%
trainer.train()
