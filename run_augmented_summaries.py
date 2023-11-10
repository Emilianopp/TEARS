from tqdm import tqdm
from IPython.display import display
from dotenv import load_dotenv
import pandas as pd 
import sys
import logging
import numpy as np
import torch
import json
import time
import torch.optim as optim
import torch
from scipy.sparse import csr_matrix
import os 
import pandas as pd
import wandb
from helper.sampler import NegSampler, negsamp_vectorized_bsearch_preverif
from argparse import ArgumentParser
from model.MF import MatrixFactorization
from helper.eval_metrics import precision_at_k, recall_at_k, mapk, ndcg_k
from helper.metrics import ndcg_k_genre
from helper.dataloader import *
from trainer.training_utils import *
from model.MF import MatrixFactorizationLLM
from model.decoderMLP import decoderMLP
import pickle 
from itertools import islice


def get_summary_embeddings(augmented = False):

    embeddings_path = f"saved_summary_embeddings/ml-100k/embeddings_{args.model_name}_augmented={augmented}.pt"

    original_embeddings_path = f'saved_summary_embeddings/ml-100k/embeddings_{args.model_name}.pt'
    excluded_genres_dict = f"saved_summary_embeddings/ml-100k/excluded_genres.pkl"
    user_summaries_path = 'saved_user_summary/ml-100k/user_summary_gpt3.5_in1_title0_full.json'
    
    embedding_function = get_open_ai_embeddings if args.embedding_module == 'openai' else get_t5_embeddings 
    #only load the first 100 summaries

    user_summaries_full ={int(k):v for k,v in  load_dataset(user_summaries_path).items()}
    user_summaries = dict(islice(user_summaries_full.items(), 100))

    if augmented:
        if args.make_augmented:
            embeddings, prompt_dict = embedding_function(user_summaries,args,prompt_dict= None,augmented = augmented)
            with open('saved_summary_embeddings/ml-100k/prompt_dict_augmented.json', 'w') as fp:
                json.dump(prompt_dict, fp)
            torch.save(torch.tensor(embeddings), embeddings_path)
        else: 
            prompt_dict = load_dataset('saved_summary_embeddings/ml-100k/prompt_dict_augmented.json')
            prompt_dict = {int(k):v for k,v in prompt_dict.items()}
            embeddings = torch.load(embeddings_path)
    else: 
        embeddings = torch.load(original_embeddings_path)[:100]


    return torch.tensor(embeddings)




def get_recs(args,augmented = False):
    t_start = time.time()
    experiment_name = f"{args.model_name}_{time.strftime('%Y-%m-%d %H:%M:%S')}"
  

    # 1. Data Loading & Preprocessing

    train_data = load_dataset("./data_preprocessed/ml-100k/data_split/train_set_leave_one.json")
    valid_data = load_dataset("./data_preprocessed/ml-100k/data_split/valid_set_leave_one.json")
    test_data = load_dataset("./data_preprocessed/ml-100k/data_split/test_set_leave_one.json")
    
    
    movie_title_to_id = map_title_to_id("./data/ml-100k/movies.dat")

    test_data = convert_titles_to_ids(test_data, movie_title_to_id)

    train_matrix, actual_list_val, actual_list_test = create_train_matrix_and_actual_lists(train_data, valid_data,
                                                                                           test_data, movie_title_to_id)
    full_data  = create_full_data(train_data,valid_data,test_data)
    
    data_matrix = create_full_dataset_matrix(full_data,movie_title_to_id)
    train_matrix = csr_matrix(train_matrix)  # Convert train_matrix to a CSR matrix
    num_users, num_items = train_matrix.shape
    
    print("train_matrix:", train_matrix.shape)

    
    user_embeddings  = get_summary_embeddings(augmented = augmented)
    embedding_dim = user_embeddings.shape[1]
    print(f"{embedding_dim=}")
    

    user_embedder = decoderMLP(embedding_dim, args.num_layers ,args.output_emb) 

    
    
    model = MatrixFactorizationLLM(num_users, user_embedder,num_items, args).to(args.device)
    model.load_state_dict(torch.load(f"{args.model_save_path}/{args.model_name}_best_model.pth"))
    user_embedder.load_state_dict(torch.load(f"{args.model_save_path}/{args.model_name}_user_embedder.pth"))


    model.eval()

    pred_list_test = generate_pred_list(model, train_matrix,args,user_embeddings, topk=args.topk,top100 = True)
    
    return pred_list_test,np.array(data_matrix)
 

def make_result_df(user_genres_augmented,user_genres_original,excluded_genres):
    
    df_augmented = pd.DataFrame.from_dict(user_genres_augmented, orient='index').add_suffix('_augmented').fillna(0)
    df_original = pd.DataFrame.from_dict(user_genres_original, orient='index').add_suffix('_original').fillna(0)
    concat_df = pd.concat([df_augmented, df_original], axis=1)
    unique_column_names = set(col.split('_')[0] for col in concat_df.columns if '_augmented' in col or '_original' in col)
    # Create new columns with '_augmented' and '_original' suffixes
    for column_name in unique_column_names:
        augmented_column_name = f'{column_name}_augmented'
        original_column_name = f'{column_name}_original'
        
        if augmented_column_name not in concat_df.columns:
            concat_df[augmented_column_name] = 0
            
        if original_column_name not in concat_df.columns:
            concat_df[original_column_name] = 0
    difs = [[] for _ in range(len(excluded_genres[0]))]
    genre_list = [[] for _ in range(concat_df.shape[0])]
    max_length = max([len(x) for i,x in excluded_genres.items()])
    excluded_genres_for_df = max([len(x) for i,x in excluded_genres.items()])    
    for (index, row),(_,excluded_genres) in zip(concat_df.iterrows(),excluded_genres.items()):

        for i in range(max_length):
            if i >= len(excluded_genres) :
                difs[i].append(0)
                continue
            genre = excluded_genres[i]
                
            if f'{genre}_original' not in concat_df.columns and f'{genre}_augmented' not in concat_df.columns:
                difs[i].append(-1)
                continue
            
            if f'{genre}_augmented' not in concat_df.columns:
                augmented_count = 0 
            else:
                augmented_count = row[[f'{genre}_augmented']].values[0]

            if row[[f'{genre}_original']].values[0] == 0:
                difs[i].append( augmented_count)
                continue
                
            difs[i].append( (augmented_count - row[[f'{genre}_original']].values[0])/row[[f'{genre}_original']].values[0])
                
        genre_list[index].append(f'{excluded_genres}')



            
            # excluded_genres_for_df[i].append('{ge}')


    for i,dif in enumerate(difs): 
        print(f"{len(dif)=}")
        concat_df[f'dif_{i}'] = dif
    print(f"{genre_list=}")
    concat_df['excluded_genres'] = genre_list
    # concat_df['excluded_genres'] = [x for _,x in excluded_genres.items()]
    return concat_df


def count_user_genres(rec_array,movie_genres_dict):

    user_genres = {}
    for user in range(rec_array.shape[0]):
        user_genres[user] = {}
        recs = rec_array[user]
        for movie in recs:
 
            genres = movie_genres_dict[movie+1]
            for genre in genres:
                if genre not in user_genres[user]:
                    user_genres[user][genre] = 1
                user_genres[user][genre] += 1
    return user_genres

def convert_recs_to_genres(recs,genre_dict):
    output_recs = recs.copy().tolist()
    recs = recs.tolist()
    for i,user_recs in enumerate(recs):

        #map the user recs to genres
        output_recs[i] = [genre_dict[rec+1] for rec in user_recs]
    return output_recs

def convert_ids_to_genres(filename):
    """
    Parses movies.dat file and returns a dictionary that maps movie IDs to their genres.

    The expected format of each line in movies.dat:
        movie_id::movie_title (year)::genre1|genre2|...
    """
    movie_id_to_genres = {}
    with open(filename, 'r', encoding='ISO-8859-1') as file:  # encoding might need to be adjusted
        for line in file:
            tokens = line.strip().split('::')
            movie_id = int(tokens[0])
            genres = tokens[2].split('|')

            # Replacing "Children's" with "Children"
            genres = ['Children' if genre == "Children's" else genre for genre in genres]

            movie_id_to_genres[movie_id] = genres
    return movie_id_to_genres



if __name__ == '__main__':
    load_dotenv(".env")

    openai.api_key = os.getenv("OPEN-AI-SECRET")
    args = parse_args()
    recs_augmented,data_matrix = get_recs(args,augmented = True)
    excluded_genres = {int(k):v for k,v in load_pickle('saved_summary_embeddings/ml-100k/excluded_genres.pkl').items()}
    print('GOT AUGMENTED SUMMARIES')
    
    recs_original,_ = get_recs(args)
    #save recs to the file #rec_list.pickle 
    with open('saved_summary_embeddings/rec_list.pickle', 'wb') as handle:
        pickle.dump(recs_augmented, handle, protocol=pickle.HIGHEST_PROTOCOL)
    movie_id_to_genres = convert_ids_to_genres("./data/ml-100k/movies.dat")
    user_genres_augmented_count = count_user_genres(recs_augmented,movie_id_to_genres)
    
    user_genres_original_count = count_user_genres(recs_original,movie_id_to_genres)
    
    user_genres_true = count_user_genres(data_matrix,movie_id_to_genres)
    row_indices, col_indices = np.where(data_matrix== 1)

    user_genres_true = convert_recs_to_genres(data_matrix,movie_id_to_genres)
    user_genres_original = convert_recs_to_genres(recs_original,movie_id_to_genres)
    user_genres_augmented = convert_recs_to_genres(recs_augmented,movie_id_to_genres)
    
    out_original = ndcg_k_genre(user_genres_original[:100],user_genres_true[:100],excluded_genres,args.topk,1)
    out_augmented = ndcg_k_genre(user_genres_augmented[:100],user_genres_true[:100],excluded_genres,args.topk,1)
   
    # make a df with out_original and out_augmented 
    df_augmented = pd.DataFrame(out_augmented, columns=["genre_1_augmented"])
    df_original= pd.DataFrame(out_original, columns=["genre_original"])
    merged_df = pd.concat([df_augmented, df_original], axis=1)
    merged_df['excluded_genres'] = [x for _,x in excluded_genres.items()]
    merged_df.to_csv(f'results_ndcg_{args.model_name}.csv')

    
    result_df = make_result_df(user_genres_augmented_count,user_genres_original_count,excluded_genres)
    
    result_df.to_csv(f'results_{args.model_name}.csv')

    
    



    

