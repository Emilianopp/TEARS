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
import ray
from pprint import pprint as pp


import debugpy

# 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1

# %%
lr= 0.00001
epochs = 400
num_heads = 6
cosine = False
num_layers = 3
output_emb = 64
embedding_dim = 768
saved_path = f'../saved_model/ml-100k/attn_best_model_{lr}_{epochs}_{num_heads}_{cosine}_{num_layers}.pth'

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

train_data = convert_titles_to_ids(train_data, movie_title_to_id)
valid_data = convert_titles_to_ids(valid_data, movie_title_to_id)
test_data = convert_titles_to_ids(test_data, movie_title_to_id)

train_matrix, actual_list_val, actual_list_test = create_train_matrix_and_actual_lists(train_data, valid_data,
                                                                                        test_data, movie_title_to_id)
train_matrix = csr_matrix(train_matrix)  # Convert train_matrix to a CSR matrix


# 2. Model Creation
num_users, num_items = train_matrix.shape
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

# Example usage:

# %%
args = parse_args()

args.output_emb = 64

if args.debugger: 
    debugpy.listen(5678)
    print("Waiting for debugger attach")
    debugpy.wait_for_client()

user_embedder = decoderAttention(embedding_dim,num_heads,num_layers,output_emb, 0  ,bias = True)

model = MatrixFactorizationLLM(num_users, user_embedder,num_items, args).to(args.device)


# %%
rankings_true = np.load(model_rankings_path)

# %%
model.load_state_dict(torch.load(model_path,map_location=torch.device('cuda')))
user_embedder.load_state_dict(torch.load(user_embeddings_path,map_location=torch.device('cuda')))
model.user_embeddings = user_embedder

model.eval()

transformer_model = SentenceTransformer('sentence-transformers/sentence-t5-large').to(args.device) 
USER_INDEX = 3
# %%
summaries = {
"Comedy": "Summary: This list of comedy movies includes adventure, drama, science fiction, and romance. Set in various locations, these films portray the trials, tribulations, and humorous situations of diverse characters in different settings.",
"Romance": "Summary: A collection of lighthearted romantic comedies and dramas set in various locations and time periods, showcasing diverse relationships and the challenges they face.",
"Drama": "Summary: A collection of drama films from the 1990s, exploring various themes such as historical mysteries, trials and tribulations of life in poverty-stricken neighborhoods, LGBT-related stories, romantic comedies, biographical tales, and epic journeys.",
"Action": "Summary: Action movies with a dark and gritty tone, featuring intense crime stories and strong performances from the cast."
    }
  
topk = 20 
MOVIE_ID = 561
def get_preds(summaries ): 
    args.embedding_module = 't5'
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



    return torch.tensor(ranked_items[:20]).to(args.device),torch.tensor(rating_pred).to(args.device)

# %%
data  = ''' Drama: Summary: A collection of drama films that explore themes of love, personal growth, and societal challenges. Each film tells a unique story with memorable characters and emotional depth.
Romance: Summary: A collection of romance films with diverse themes, including a Cuban refugee pretending to be a family, an erotic romantic thriller, a fantasy comedy about reliving the same day, a mistaken identity romantic comedy, a classic screwball comedy, a Shakespeare
Children: Summary: These children's movies are filled with fantasy, adventure, and heartwarming stories. From animated tales to magical adventures, these films are perfect for young viewers.
Comedy: Summary: A collection of comedy films spanning different eras and styles, featuring romantic entanglements, humorous misadventures, and comedic performances from an ensemble of actors.
Action: Summary: Action-packed films with a mix of genres including Western, medical disaster, buddy cop, science fiction, and horror anthology. The movies feature intense action sequences and diverse plots involving epic space battles, covert operations, and post-apocalyptic worlds.
Thriller: Summary: A collection of thrilling films with elements of psychological drama, legal suspense, science fiction, and horror. Featuring talented actors and directors, these movies keep audiences on the edge of their seats with intense storylines and unexpected twists.
Crime: Summary: A collection of crime films with various themes, including violence, drama, comedy, and thrillers, featuring notable directors and actors.
Adventure: Summary: Adventure movies filled with thrilling journeys, time travel, and epic battles. From exploring other planets through a mystical portal to going back in time or diving into a fantastical Arthurian world, these films take us on incredible escapades.
Sci-Fi: Summary: A collection of science fiction films that transport audiences to different worlds and challenge their perceptions of reality. These movies explore themes of adventure, disaster, and the power of technology in intriguing and unexpected ways.
Fantasy: Summary: Adventure comedies with a fantasy twist that provide an enjoyable escape from reality.
'''


# data  = ''' Drama: Summary: A collection of drama films that explore themes of love, personal growth, and societal challenges. Each film tells a unique story with memorable characters and emotional depth.
# Romance: Summary: A collection of romance films with diverse themes, including a Cuban refugee pretending to be a 
# '''


# data  = ''' Drama: Summary: A collection of drama'''
# %%

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

# %%

''''
RIGHT NOW IT TAKES IN A LIST VALUE AND ONLY INDECES THE FIRST ENTRY THIS IS A BANDAID FIX
'''
def make_string_dict(data):
    
    genre_summary_dict = {}
    data = data[0].lower().replace('\n',' ').replace('-','').replace('summary:','summary')

    # Use regular expression to find genre and summary information
    matches = re.finditer(r'(\w+): (.+?)(?=\w+:|$)', data)

    # Iterate through the matches and extract genre and summary information
    for match in matches:
        genre, summary = match.group(1), match.group(2)
        genre_summary_dict[genre] = summary.strip()

    # Convert the dictionary to JSON
    return genre_summary_dict
# %%

def style_ranker(text,movie_id =57 ,original_index =16,model= None ):
    preds_new,scores = get_preds(make_string_dict(text))
    if MOVIE_ID in preds_new: 

        return scores[:,MOVIE_ID],preds_new[0]
    else:
        return torch.tensor([0]).to(args.device),torch.tensor([0]).to(args.device)
    
    
   


data =[data]



os.environ["TOKENIZERS_PARALLELISM"] = "false"

args = get_args()
args.device =  'cuda' if torch.cuda.is_available() else 'cpu'

editor = RobertaEditor(args).to(device)

sahc = SteepHC(args, editor).to(device)
of_dir = 'results/' + args.output_dir

if not os.path.exists(of_dir):
    os.makedirs(of_dir)

if args.direction == '0-1': postfix = '0'
else: postfix = '1'


bsz = args.bsz
max_len=len(data[0].split())

tzone = tz.gettz('')
timestamp = datetime.datetime.now().astimezone(tzone).strftime('%Y-%m-%d_%H:%M:%S')

output_file =f'{timestamp}_{dst}_seed={str(args.seed)}_{str(args.style_weight)}_{args.direction}.txt'

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

with open(of_dir + output_file, 'w', encoding='utf8') , torch.no_grad():
    for i in range(num_batches):
        batch_data = data[bsz * i:bsz * (i + 1)]

        ref_oris = []
        for d in batch_data:
            for k, v in word_pairs.items():
                    d=d.strip().lower().replace(k, v)
            ref_oris.append(d)

        ref_olds=ref_oris.copy()
        state_vec, _ = editor.state_vec(ref_olds)

        break_flag = False
        max_score=-np.inf
        step_max_score_list=[-np.inf]
        seq_len=[len(line.split()) for line in ref_olds]
        max_seq_len=max(seq_len)
        args.max_steps = 100
        select_sent = None
        
        
        
        for step in range(args.max_steps):
            #get the whole candidate list
            max_len = len(ref_olds[0].split())


            indices_of_words_with_colon = do_not_edit(ref_olds[0])
            sampled_indices = random.sample(range(max_len), 5)
            input_tuples = [[ref_olds,[ops]*bsz,[positions]*bsz,bsz,max_len]
                                                    for positions in sampled_indices if positions not in indices_of_words_with_colon for ops in [0,1,2]]
            ref_news = [editor.edit(*inp)for inp in tqdm(input_tuples,desc = "Making Edits")]
            for idx in ( pbar := tqdm(range(len(ref_news)),desc = 'going through styles')):

                ref_new_batch_data=ref_news[idx]

                # Calculating the acceptance probability

                ref_old_score, ref_new_score, new_style_labels,_,pos_new,pos_old \
                    = sahc.acceptance_prob(ref_new_batch_data, ref_olds, ref_oris, state_vec,style_ranker,movie_id = MOVIE_ID)

                ref_hat = ref_new_batch_data

               
                new_style_label=new_style_labels
                
                # Updating the maximum score and selected sentence
           
                if ref_new_score>max_score and ref_new_score>ref_old_score:
                    select_sent = ref_hat
                    max_score=ref_new_score
                    print(f"New Score {ref_new_score=}")
                pbar.set_description(f'score = {ref_new_score}')
                # the style is changed!
                if args.early_stop == True:
                    if (args.direction == '0-1' and new_style_label == 1) or \
                            (args.direction == '1-0' and new_style_label == 0) :
                        print(f"{new_style_label=}")
                        select_sent = ref_hat
                        print_es()
                        break_flag = True
                        break
            # Checking if the current score is larger than previous max score
       
            if max_score>=step_max_score_list[step]: 
                print("hill climbing!")
                logging.info("hill climbing!")
                if select_sent is None: 
                    random_draw = random.sample(range(len(input_tuples)),1)
                    print('randomly drawing')
                    select_sent = ref_news[random_draw[0]]

                    # find_string_differences(select_sent[0],ref_oris[0])

            



                ref_olds = select_sent

                pp(pos_new)
                pp(pos_old)

                    
                step_max_score_list.append(max_score)
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
        
    with open('./results.txt', 'w+') as file:
        # Write the first string to the file
        
        file.write(f'ORIGINAL SENTENCE {steps}\n\n\n'+ref_oris[0].replace('\n',' ')+ '\n')  # Add a newline character to separate the strings
        file.write('Augmented SENTENCE \n\n\n'+select_sent[0] + '\n')  # Add a newline character to separate the strings

        # Write the second string to the file

        
    logging.info('\n')
