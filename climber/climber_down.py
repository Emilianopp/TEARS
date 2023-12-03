# %%

import sys
sys.path.append('../')
import os
import re 
import logging
import json
from pprint import pprint as pp
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.sparse import csr_matrix
from typing import List
from tqdm import tqdm
from argparse import ArgumentParser
import datetime
from dateutil import tz
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch
import pickle
import math
import warnings
from sentence_transformers import SentenceTransformer
from model.decoderMLP import decoderMLP, decoderAttention, movieTransformer
from helper.sampler import NegSampler, negsamp_vectorized_bsearch_preverif
from model.MF import MatrixFactorization, MatrixFactorizationLLM
from trainer.training_utils import *
from helper.eval_metrics import *
from helper.dataloader import *
from StyleTransfer.scorer import *
from StyleTransfer.editor import RobertaEditor
from StyleTransfer.config import get_args
import random
import json
from pprint import pprint as pp


import debugpy


def get_args():

    parser = argparse.ArgumentParser(description="model parameters")
    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generator')
    
    parser.add_argument('--output_dir', type=str, default="output/", help='Output directory path to store checkpoints.')
    parser.add_argument('--class_name',default='../EleutherAI/gpt-neo-1.3B',type=str)
    parser.add_argument('--topk', default=20, type=int,help="top-k words in masked out word prediction")
    parser.add_argument("--fluency_weight", type=int, default=1, help='fluency')
    parser.add_argument("--sem_weight",type=int, default=1, help='semantic similarity')
    parser.add_argument("--style_weight", type=int, default=8, help='style')
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--bs",type=int,default=4,help="batch size")
    parser.add_argument('--keyword_pos', default=True, type=bool)
    parser.add_argument("--early_stop",default=True, type=bool)
    parser.add_argument("--data_name", default='ml-100k', type=str)
    parser.add_argument("--embedding_module", default='t5', type=str)
    parser.add_argument('--debugger', action='store_true')
    parser.add_argument('--make_rankings', action='store_true')

    args, _ = parser.parse_known_args() 
  

    args.device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')) 

    return args


args = get_args()

if args.debugger: 
    debugpy.listen(5678)
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
    
def find_string_differences(str1, str2):
    # Find the indices where the strings differ
    str1 = str1.split()
    str2 = str2.split()
    
    diff_indices = [i for i, (c1, c2) in enumerate(zip(str1, str2)) if c1 != c2]

    # Print 5 indices before and after each difference
    for index in diff_indices:
        start_index = max(0, index - 30)
        end_index = min(len(str1), index + 30)


        print(f"Difference at index {index}:")

        print(f"String 2: {str2[start_index:end_index]}")
        print(f"String 1: {str1[start_index:end_index]}")
        print()
        

# Split the text into words
def do_not_edit(text):
    text = text[0] if not isinstance(text,str) else text
    
    words = text.split()

    # Initialize a list to store the indices of words with ':'
    indices_of_words_with_colon = []

    # Iterate through the words to find the indices of words with ':'
    for index, word in enumerate(words):
        if ':' in word:
            # Add the index to the list
            indices_of_words_with_colon.append(index)

    # Print the list of indices of words with ':'
    return indices_of_words_with_colon
        
def get_preds(summaries ,USER_INDEX): 
    args.embedding_module = 't5'
    topk= args.topk
    embs = get_genrewise_embeddings(summaries,args, model= transformer_model )

    genre_list = get_genres()
    embs_tens = model.user_embeddings.prepare_input(embs,genre_list).to(args.device)


    rating_pred = model.predict(embs_tens.unsqueeze(0)).cpu().detach().numpy()
    
    rating_pred[train_matrix[USER_INDEX].toarray() > 0] = 0


    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
    ind = np.argpartition(rating_pred, -topk)
    ind = ind[:, -topk:]
    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
    

    ranked_items = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
    recall_val = recall_at_k_one(actual_list_val[3], ranked_items[0].tolist(), 20)


    reversed_movie_title_to_id = {v: k for k, v in movie_title_to_id.items()}
    movie_titles_ranked = [f'{index} : {reversed_movie_title_to_id[i+1]} {id_genre_map[i+1]}' for index,i in enumerate(ranked_items[0][:20])]



    return torch.tensor(ranked_items).to(args.device),torch.tensor(rating_pred).to(args.device)

def make_string_dict(data):
    
    genre_summary_dict = {}
    data = data[0].lower().replace('\n',' ').replace('-','').replace('summary:','summary')

    # Use regular expression to find genre and summary information
    matches = re.finditer(r'(\w+): (.+?)(?=\w+:|$)', data)

    # Iterate through the matches and extract genre and summary information
    for match in matches:
        genre, summary = match.group(1), match.group(2)
        genre_summary_dict[genre] = summary.replace('summary','summary:').strip()


    # Convert the dictionary to JSON
    return genre_summary_dict

def make_dict_string(data):
    data_out = []
    for d in data:
        user_str = ''
        for k,v in d.items():
            user_str += f" {k}: {v}"
        data_out.append(user_str)
        
    return data_out

            
def find_different_word_indices(s1, s2):
    # Split the strings into lists of words
    words_s1 = s1.split()
    words_s2 = s2.split()

    # Find the minimum length of the two lists
    min_len = min(len(words_s1), len(words_s2))

    # Initialize a list to store the indices of differences
    different_indices = []

    # Iterate through each word and compare
    i = 0
    k=0
    while i < len(words_s1):
        print(f"{words_s1[i]=}")
        print(f"{ words_s2[k]=}",k)
        while  i < len(words_s1) and words_s1[i] != words_s2[k]:
            different_indices.append(i)
            i += 1
        i+=1
        k+=1

    return different_indices



def generate_rankings(data):
    result_list = []
    for i, x in enumerate(data):

        preds = get_preds(x, i)[0][:19].detach().cpu().numpy()
        result_list.append(preds)
    
    return np.array(result_list)

    
# %%

def style_ranker(text,movie_id =57 ,user_id=None,original_index =16,model= None ):
    preds_new,scores = get_preds(make_string_dict(text),user_id)
    if movie_id in preds_new: 
        return scores[:,movie_id],preds_new[0]
    else:
        return torch.tensor([float('inf')]).to(args.device),torch.tensor([float('inf')]).to(args.device)
    
    
# %%
lr= 0.00001
epochs = 500
num_heads = 8
cosine = False
num_layers = 2
output_emb = 256
embedding_dim = 768


saved_path = f'/home/mila/e/emiliano.penaloza/scratch/saved_model/ml-100k/attention_best_model_{lr}_{epochs}_{num_heads}_{cosine}_{num_layers}.pth'

model_path = saved_path + '_best_model.pth'
embedder_path = saved_path + '_embedder.pth'
item_embeddings_path = saved_path + '_item_embeddings.pth'
user_embeddings_path = saved_path + '_user_embeddings.pth'
model_rankings_path = saved_path + '_rankings_matrix.npy'
id_genre_map = map_id_to_genre('../data/ml-100k/movies.dat')

# 1. Data Loading & Preprocessing
train_data = load_dataset("../data_preprocessed/ml-100k/data_split/train_set_leave_one.json")
valid_data = load_dataset("../data_preprocessed/ml-100k/data_split/valid_set_leave_one.json")
test_data = load_dataset("../data_preprocessed/ml-100k/data_split/test_set_leave_one.json")
movie_title_to_id = map_title_to_id("../data/ml-100k/movies.dat")
ids_to_genres = convert_ids_to_genres("../data/ml-100k/movies.dat")
ids_to_title = map_id_to_title("../data/ml-100k/movies.dat")
train_data = convert_titles_to_ids(train_data, movie_title_to_id)
valid_data = convert_titles_to_ids(valid_data, movie_title_to_id)
test_data = convert_titles_to_ids(test_data, movie_title_to_id)


train_matrix, actual_list_val, actual_list_test = create_train_matrix_and_actual_lists(train_data, valid_data,
                                                                                        test_data, movie_title_to_id)
train_matrix = csr_matrix(train_matrix)  # Convert train_matrix to a CSR matrix

num_users, num_items = train_matrix.shape
args.output_emb = 256
user_embedder = decoderAttention(embedding_dim,num_heads,num_layers,output_emb, 0  )
model = MatrixFactorizationLLM(num_users, user_embedder,num_items, args).to(args.device)
# rankings_true = np.load(model_rankings_path)

model.load_state_dict(torch.load(model_path,map_location=torch.device('cuda')))
model.eval()

transformer_model = SentenceTransformer('sentence-transformers/sentence-t5-large').to(args.device) 

with open('../saved_user_summary/ml-100k/user_summary_gpt3.5_in1_title0_full.json','r') as j:
    data = json.load(j)
    data = {int(key): value for key, value in data.items()}
  



# rankings = generate_rankings(data) if args.make_rankings else np.load('./rankings.npy' ).squeeze(1)
data =[v for k,v in data.items()]
rankings = generate_rankings(data[:10]).squeeze(1)
print(f"{rankings.shape=}")
print(args.make_rankings)
last_items = rankings[:,19]

data = make_dict_string(data)





args.max_len = 514
# print(f"{args.max_len=}")
# exit(0)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# %%
editor = RobertaEditor(args).to(device)
sahc = SteepHC(args, editor).to(device)   


# %%


of_dir = 'results/' + args.output_dir

if not os.path.exists(of_dir):
    os.makedirs(of_dir)


bsz = args.bs


tzone = tz.gettz('')
timestamp = datetime.datetime.now().astimezone(tzone).strftime('%Y-%m-%d_%H:%M:%S')

output_file =f'{timestamp}_{dst}_seed={str(args.seed)}_{str(args.style_weight)}.txt'

log_txt_path=os.path.join(of_dir, output_file.split('.txt')[0] + '.log')



for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(format='',
                    filename=log_txt_path,
                    filemode='w',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

word_pairs ={"ca n't": "can not", "wo n't": "will not"}
logging.info(args)

def print_es():
    print("Early Stopping!")
    logging.info("Early Stopping!")
    
num_batches = len(data)//bsz
result_d = {}


# %%
data = data[:5]

with open(of_dir + output_file, 'w', encoding='utf8') , torch.no_grad():
    for i in range(len(data)):
        batch_data = data[i]


        for k, v in word_pairs.items():
            batch_data = batch_data.strip().lower().replace(k, v)
        
        ref_oris = ref_olds = [batch_data]
        state_vec, _ = editor.state_vec(ref_olds)

        break_flag = False
        min_score=np.inf
        step_min_score_list=[np.inf]

        max_len=len(batch_data.split())
        select_sent = None
        movie_id = rankings[i,0]
        
        print(f"Boosting Movie: {movie_id} id, title = {ids_to_title[movie_id+1]} with genres {ids_to_genres[movie_id+1]}")
        
        action_list  = []
        select_pos = []
        for step in range(args.max_steps):
            indices_of_words_with_colon = do_not_edit(ref_olds)
            sampled_indices = list(range(max_len))
            input_tuples = [[ref_olds,[ops],[positions],bsz,max_len]
                                                    for positions in sampled_indices if positions not in indices_of_words_with_colon for ops in [0,1,2]]

            ref_news = [editor.edit(*inp) for inp in tqdm(input_tuples,desc = "Making Edits")]
            select_step = None
            for idx in ( pbar := tqdm(range(len(ref_news)),desc = 'going through styles')):
                ref_new_batch_data=ref_news[idx]
                # Calculating the acceptance probability
                ref_old_score, ref_new_score, new_style_labels,_,pos_new,pos_old \
                    = sahc.acceptance_prob_lower(ref_new_batch_data, ref_olds, ref_oris, state_vec,style_ranker,movie_id = movie_id,user_id = i)
                ref_hat = ref_new_batch_data
                new_style_label=new_style_labels
                
                # Updating the maximum score and selected sentence
           
                if ref_new_score<min_score and ref_new_score<ref_old_score:
                    select_sent = ref_hat
                    min_score=ref_new_score
                    select_step = idx
                    print(f"New Score {ref_new_score=}")
                    
                pbar.set_description(f'score = {ref_new_score}')
                if args.early_stop == True and new_style_label == 1:
                        print(f"{new_style_label=}")
                        select_sent = ref_hat
                        print_es()
                        break_flag = True
                        break
            # Checking if the current score is larger than previous max score
            if min_score<=step_min_score_list[step]: 
                print("hill climbing!")
                logging.info("hill climbing!")
                if select_sent is None: 
                    random_draw = random.sample(range(len(input_tuples)),1)
                    print('randomly drawing')
                    select_sent = ref_news[random_draw[0]]

                step_min_score_list.append(min_score)
                if select_step is not None:
                    select_pos.append(input_tuples[idx][2][0])
                    action_list.append(input_tuples[select_step][1])
                else:
                    action_list.append('nothing')
                    select_pos.append('nothing')
                result_d[i] = {'old' : ref_oris, 'new':select_sent,'movie' : movie_id,'steps':step
                               ,'title':ids_to_title[movie_id+1], 'genres' : ids_to_genres[movie_id+1],
                               'user_id': i, 'scores':step_min_score_list,'break':break_flag,'actions':action_list,'pos_dict': select_pos}
            else:
                print("don't climb, stop!")
                logging.info("don't climb, stop!")
                break_flag=True
            if break_flag:
                steps = step
                break
        if break_flag:
            select_sent = select_sent

        logging.info('climb {} steps, the selected sentence is: {}'.format(step+1,select_sent))
        print('climb {} steps, the selected sentence is: {}'.format(step+1,select_sent))
        print(f'The original sentence is: {ref_oris} ')
         
    
with open('./result_dict_down.pkl','wb+') as f:
    pickle.dump(result_d,f)




