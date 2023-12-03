from sentence_transformers import SentenceTransformer
from model.decoderMLP import decoderMLP, decoderAttention, movieTransformer
from tqdm import tqdm
import openai
import pickle 
import argparse
from typing import List
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
from collections import defaultdict
from helper.sampler import NegSampler, negsamp_vectorized_bsearch_preverif
from argparse import ArgumentParser
from model.MF import MatrixFactorization
from helper.eval_metrics import precision_at_k, recall_at_k, mapk, ndcg_k
from helper.dataloader import load_dataset, map_title_to_id, convert_titles_to_ids, \
    create_train_matrix_and_actual_lists

#flatten a nested dict to get all the counts 

def get_max_counts(d):
    max_counts = []
    for user_id, genre_counts in d.items():
        min_count = max(genre_counts.values())
        max_counts.append(min_count)
    return max_counts

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


def neg_item_pre_sampling(train_matrix, num_neg_candidates=500):
    num_users, num_items = train_matrix.shape
    user_neg_items = []
    for user_id in range(num_users):
        pos_items = train_matrix[user_id].indices

        u_neg_item = negsamp_vectorized_bsearch_preverif(pos_items, num_items, num_neg_candidates)
        user_neg_items.append(u_neg_item)

    user_neg_items = np.asarray(user_neg_items)

    return user_neg_items


def generate_pred_list(model, train_matrix,args,user_embeddings, topk=20,summary_encoder = None,top100 = False,print_emb = False):
    num_users = train_matrix.shape[0]
    batch_size = 1024
    num_batches = int(num_users / batch_size) + 1
    print(f"{num_batches=}")
    user_indexes = np.arange(num_users)
    pred_list = None

    for batchID in range(num_batches):
        start = batchID * batch_size
        end = start + batch_size

        if batchID == num_batches - 1:
            if start < num_users:
                end = num_users
            else:
                break
        batch_user_index = user_indexes[start:end ] if not top100 else user_indexes[:100]
        
   

        batch_user_emb = user_embeddings[batch_user_index].to(args.device)

       
        rating_pred = model.predict_recon(batch_user_emb,summary_encoder) if args.recon else  model.predict(batch_user_emb,print_emb = print_emb)

       


        rating_pred = rating_pred.cpu().data.numpy().copy()
        rating_pred[train_matrix[batch_user_index].toarray() > 0] = 0

        # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
        ind = np.argpartition(rating_pred, -topk)
        ind = ind[:, -topk:]
        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

        if batchID == 0:
            pred_list = batch_pred_list
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)

    return pred_list

def generate_pred_list_attention(model, train_matrix,args,user_embeddings, topk=20,summary_encoder = None,top100 = False,print_emb = False):
    num_users = train_matrix.shape[0]
    batch_size = 4
    num_batches = int(num_users / batch_size) + 1
    print(f"{num_batches=}")
    user_indexes = np.arange(num_users)
    pred_list = None

    for batchID in range(num_batches):
        start = batchID * batch_size
        end = start + batch_size

        if batchID == num_batches - 1:
            if start < num_users:
                end = num_users
            else:
                break
            
        genre_list = get_genres()
        batch_user_index = user_indexes[start:end ] if not top100 else user_indexes[:100] 

        
   
        all_user_genre_dict = [user_embeddings[u+1] for u in batch_user_index]

        user_list = [ model.user_embeddings.prepare_input(user_genre_dict, genre_list) for user_genre_dict in all_user_genre_dict]
        user_tensor = torch.stack(user_list).to(args.device)

       
        rating_pred =   model.predict(user_tensor,print_emb = print_emb)

        rating_pred = rating_pred.cpu().data.numpy().copy()
        rating_pred[train_matrix[batch_user_index].toarray() > 0] = 0

        # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
        ind = np.argpartition(rating_pred, -topk)
        ind = ind[:, -topk:]
        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

        if batchID == 0:
            pred_list = batch_pred_list
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)

    return pred_list

def compute_metrics(test_set, pred_list, topk=20):
    precision, recall, MAP, ndcg = [], [], [], []
    for k in [5, 10, 15, 20]:
        precision.append(precision_at_k(test_set, pred_list, k))
        recall.append(recall_at_k(test_set, pred_list, k))
        MAP.append(mapk(test_set, pred_list, k))
        ndcg.append(ndcg_k(test_set, pred_list, k))

    return precision, recall, MAP, ndcg


def generate_rankings_for_all_users(model,user_emb, num_users, num_items,args):
    
    rankings = np.zeros((num_users, num_items), dtype=int)

    # For each user, generate scores for all items, then rank the items
    for user_id in range(num_users):


        user_id_tensor = user_emb[user_id].to(args.device)


        scores = model.predict(user_id_tensor)

        # Get the items' indices sorted by their scores in descending order
        ranked_items = np.argsort(scores)[::-1]

        # Store the ranked item IDs in the matrix
        rankings[user_id] = ranked_items

    return rankings

def generate_rankings_for_all_users_attention(model,user_emb, num_users, num_items,args):
    rankings = np.zeros((num_users, num_items), dtype=int)

    # For each user, generate scores for all items, then rank the items
    genre_list = get_genres()
    
    for user_id in range(num_users):

        user_id_tensor = user_emb[user_id+1]

        user_id_tensor = model.user_embeddings.prepare_input(user_id_tensor, genre_list).unsqueeze(0).to(args.device)

        scores = model.predict(user_id_tensor).squeeze().cpu().detach().numpy()

        # Get the items' indices sorted by their scores in descending order
        ranked_items = np.argsort(scores)[::-1]

        # Store the ranked item IDs in the matrix
        rankings[user_id] = ranked_items

    return rankings

def generate_and_save_rankings_json(rankings_matrix, topk, top_for_rerank, movie_id_to_genres, filename,
                                    user_genre_file):
    # Load the user_genre.json file
    with open(user_genre_file, 'r') as f:
        user_genres_data = json.load(f)

    rankings_dict = {}
    num_users, _ = rankings_matrix.shape

    for user_id in range(num_users):
        user_real_id = user_id + 1  # Assuming user IDs start from 1
        rankings = rankings_matrix[user_id][:topk]

        # Get the genres specific to the user from user_genre.json
        user_allowed_genres = user_genres_data.get(str(user_real_id), [])

        # Initialize the genre_dict with genres from user_genre.json, keeping the order intact
        genre_dict = {genre: [] for genre in user_allowed_genres}

        for movie_idx in rankings:
            movie_real_id = movie_idx + 1  # Assuming movie IDs start from 1
            genres = movie_id_to_genres.get(movie_real_id, [])
            for genre in genres:
                if genre in genre_dict:  # Note that we don't need to check if genre is in user_allowed_genres since genre_dict is built from it
                    genre_dict[genre].append(movie_real_id)

        # Trim the list of movies for each genre to top "top_for_rerank"
        for genre, movies in genre_dict.items():
            genre_dict[genre] = movies[:top_for_rerank]  # This was changed from 1 to 50

        rankings_dict[str(user_real_id)] = genre_dict

    print("rankings_dict:", rankings_dict['941'])

    def int64_converter(o):
        if isinstance(o, np.int64):
            return int(o)
        raise TypeError

    with open(filename, 'w') as file:
        json.dump(rankings_dict, file, indent=4, default=int64_converter)


def log_results_csv(log_file,log_data):
    if not os.path.isfile(log_file):
        # Create a new CSV file with headers
        df = pd.DataFrame([log_data])
        df.to_csv(log_file, index=False)
    else:
        # Append to the existing CSV file
        df = pd.read_csv(log_file)
        df = df._append(log_data, ignore_index=True)
        df.to_csv(log_file, index=False)
 
 

def json_to_list(data):
    summaries = []
    for key, genre_summaries in data.items():
            summaries.append(genre_summaries)
    return summaries

def get_summary_function(summary_style, user_summary_subset ):
    if summary_style == 'original':
        return summarize_summaries_prompt(user_summary_subset)
    elif summary_style == 'topM':
        user_genre_counts_full = load_pickle('./saved_summary_embeddings/ml-100k/genre_counts.pkl')
        user_genre_counts = trim_dictionary(user_genre_counts_full, 5)
        return summarize_summaries_prompt_topM(user_summary_subset,user_genre_counts)
    elif summary_style == 'augmented':
        return summarize_summaries_augmented_prompt(user_summary_subset)
    

def summarize_summaries_prompt(user_summaries): 
    role_prompt = 'The following are the summaries of movies watched by the user, in the format genre: summary. \n'
    prompts = []
    for user in user_summaries:
        summary_prompt = ""
        for genre, summary in user.items():
            summary_prompt += f"{genre}: {summary}\n"
        prompts += [role_prompt + summary_prompt]
    with open("saved_summary_embeddings/ml-100k/prompts_original.txt", "w") as f:
        for prompt in prompts:
            f.write(prompt)
            f.write("\n")    
    
    return prompts,None

def summarize_summaries_prompt_topM(user_summaries,genre_counts): 
    role_prompt = 'The following are the summaries of movies watched by the user, in the format genre: summary. \n'
    prompts = []
    for k,user in enumerate(user_summaries):
        summary_prompt = ""
        for genre, summary in user.items():
            if genre in list(genre_counts[k].keys()):
                summary_prompt += f"{genre}: {summary}\n"
        prompts += [role_prompt + summary_prompt]
    with open("saved_summary_embeddings/ml-100k/prompts_topM.txt", "w") as f:
        for prompt in prompts:
            f.write(prompt)
            f.write("\n")    
    
    return prompts,None


def summarize_summaries_augmented_prompt(user_summaries,sentiment = '',excluded_genres=3,max_exclude = True): 
    role_prompt = 'The following are the summaries of movies watched by the user, in the format genre: summary. \n'
    prompts = []
    excluded_genre_users =defaultdict(list)
    user_key_lengths = []
    # load saved_summary_embeddings/ml-100k/genre_counts.pkl
    with open("saved_summary_embeddings/ml-100k/genre_counts.pkl", "rb") as f:
        genre_counts = pickle.load(f)

    for i,user in enumerate(user_summaries):

    
        summary_prompt = ""
        '''
        Test this works once the model is trained
        '''
        random_genre = np.random.choice(list(user.keys()),excluded_genres) if not max_exclude else [max(genre_counts[i], key=genre_counts[i].get)] 
        

        sentiment_prompt = 'The user has a really negative opinion about this genre, as well as a negative opinion of the summary. \n' if sentiment != '' else ''


        for genre, summary in user_summaries[user].items():

            if  genre in random_genre:

                excluded_genre_users[i].append( genre)
                if max_exclude:

                    continue
                if sentiment == '':
                    continue
                elif sentiment == 'negative':
                    sentiment_prompt += f"{genre}: {summary}\n"
                    continue
            summary_prompt += f"{genre}: {summary}\n"
        
        prompts += [role_prompt + summary_prompt + sentiment_prompt]
    #save prompts to a text file 
    with open("saved_summary_embeddings/ml-100k/prompts_augmented.txt", "w") as f:
        for prompt in prompts:
            f.write(prompt)
            f.write("\n")    
   


    return prompts, excluded_genre_users


def get_embedding_module(embedding_module):
    if embedding_module == 'openai':
        return get_open_ai_embeddings
    elif embedding_module == 't5':
        return get_t5_embeddings
    

def get_model(model_name):
    if  model_name == 'transformer': 
        return movieTransformer
    elif model_name == 'attention':
        return decoderAttention       

def get_genrewise_embeddings(user_genre_summaries,args,model):
        embedding_module = get_embedding_module(args.embedding_module)
        genre_embeddings = {}
        
        for genre, summaries in user_genre_summaries.items():

            embeddings = embedding_module(summaries,args,model)
            genre_embeddings[genre] = embeddings
            
        return genre_embeddings
    
    
def trim_dictionary(data, thresh):
    trimmed_data = {
        user: {genre: count for genre, count in genres.items() if count > thresh}
        for user, genres in data.items()
    }
    return trimmed_data

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
    
def get_prompts(user_summaries,args,augmented):
    user_summary_subset = user_summaries
    if augmented: 
        summary_style = 'augmented'
    else:
        summary_style = args.summary_style
        
    summarize_summaries_prompt_list,excluded_genres = get_summary_function(summary_style, user_summary_subset)
    if excluded_genres is not None: 
        excluded_genres_dict = f"saved_summary_embeddings/ml-100k/excluded_genres.pkl"
        with open(excluded_genres_dict, "wb") as f:
            pickle.dump(excluded_genres, f)

    text = [x.replace("\n", " ") for x in summarize_summaries_prompt_list ]
    promp_dict = {}

    for i,t in (pbar := tqdm(enumerate(text))):
        
        msg =[{
                "role": "system",
                "content": "Provide a description of these movie summaries. Only summarize the text and suggest types of movies that the user might like. \
                    Do not directly include any movie titles or genres. When you can describe two genres the user likes together do so. Be concise with about 200 characters but descriptive.\
                    This is an example of inputs:\
                    Action: Summary: Action-packed movies from the 90s with a mix of cyberpunk, post-apocalyptic, and martial arts themes. These films feature intense action sequences and are directed by renowned filmmakers.,\
                    Drama: Summary: A collection of drama films ranging from disaster survival, epic Western, historical biographical, romantic, and psychological thriller. These films explore themes of love, loss, personal struggles, and the human condition.,\
                    Romance: Summary: A collection of romantic movies from various genres including drama, comedy, musical, and fantasy. These films explore themes of love and relationships, featuring a mix of well-known actors and diverse storylines.,\
                    Sci-Fi: Summary: A collection of sci-fi films ranging from space operas to post-apocalyptic adventures, with elements of cyberpunk and dystopia. The movies explore themes of technology, survival, and thrilling action, featuring memorable characters and imaginative settings.,\
                    Thriller: Summary: A collection of thrilling movies with elements of crime, suspense, and science fiction. The films feature a variety of genres including erotic thriller, cyberpunk, and independent drama. The cast includes notable actors such as Ray Liotta, Linda Fiorent,\
                    Crime: Summary: A collection of crime films from the mid-90s, including gangster comedies, epic crime dramas, cyberpunk thrillers, and action-packed sci-fi flicks.,\
                    Comedy: Summary: A collection of comedy films spanning different themes and genres, including children's comedy, adventure comedy fantasy, prank calls, historical comedy-drama, action comedy, romantic comedy, and farce black comedy.,\
                    Children: Summary: A collection of coming-of-age, fantasy, adventure, comedy, and family films that will entertain children and provide a fun and enjoyable experience.,\
                    Adventure: Summary: Adventure movies from the 1990s with a mix of drama, fantasy, and superhero themes.,\
                    Animation: Summary: Animated films that belong to the Animation genre, providing entertainment with adventure, music, and historical themes.,\
                    Fantasy: Summary: A collection of fantasy films ranging from family adventures to magical realism, including stories about mythical creatures, mystical worlds, and extraordinary quests.,\
                    Documentary: Summary: A collection of documentary films exploring various subjects, including basketball dreams, mockumentary, hip-hop music, Hollywood's infamous madam, fashion models, and notable individuals like Nico and Jean Seberg. The films offer a glimpse into different worlds,\
                    Mystery: Summary: A collection of mysterious and suspenseful films that explore crime, psychological thrillers, and science fiction. These movies delve into the world of detectives, investigations, and the after-effects of unexpected events.,\
                    Horror: Summary: A collection of horror films that explore vampires, psychological thrillers, and supernatural elements. A vampire gothic horror, a vampire black comedy, a psychological thriller, an action horror, a fantasy thriller, a direct-to-video horror, and a,\
                    War: Summary: A collection of war-inspired films from various genres, including action thrillers, historical dramas, biographical dramas, satirical comedies, and dramatic portrayals of real-life events.\
                    This is the resulting output summary: \
                    Based on your movie preferences, you enjoy a variety of genres. You seem to enjoy dramas that tackle various themes such as romance, religion, comedy, feminism, and social issues. You also enjoy intense thrillers with elements of horror, mystery, crime, and cyberpunk. Adventure movies with thrilling escapades, mysterious secrets, and a touch of danger appeal to you as well. Additionally, you seem to be a fan of romance films from the 1990s that explore love, relationships, and personal growth. Comedies with various themes, including romantic comedy, road trips, screwball antics, and drag queens are also favorites of yours. Crime films with thrilling mysteries, engaging drama, and a touch of cybercrime catch your interest. Lastly, action-packed films from the 1990s with elements of crime, cyberpunk, and thrilling adventures also seem to be enjoyable for you."},            
              {
                "role": "user",
                "content": text[i]
            }
            ]
        try:
            prompts = openai.ChatCompletion.create(model="gpt-3.5-turbo",   
                                                messages=msg,                                    
                                    max_tokens=200  # limit the token length of response
                                )['choices'][0]['message']['content']
            promp_dict[i] = prompts
            pbar.set_description(f"Prompt {i + 1}/{len(text)}")
    
        except Exception as e:
            prompts = None
            j = 0 
            while prompts is None:
                try:
                    print(f"{e=} exception occured have tried {j} times")
                    pbar.set_description(f"Timeout exception: {e}. Sleeping for a minute and retrying.")
                    time.sleep(60)
                    prompts = openai.ChatCompletion.create(model="gpt-3.5-turbo",   
                                                    messages=msg,
                                        max_tokens=200  # limit the token length of response
                                    )['choices'][0]['message']['content']

                    pbar.set_description(f"Request {i} completed")
                    j+=1
                except Exception as e:
                    prompts = None
    
    return promp_dict

def dump_json(path,data):
      with open(path, "w") as f:
        json.dump(data , f, indent=4)
    
    
def get_encoder_inputs( user_summaries,args,prompt_dict=None,augmented =False):
    prompt_dict = get_prompts( user_summaries,args,augmented) if prompt_dict is None else prompt_dict
    prompts = [x for i,x in prompt_dict.items()]
    return prompts,prompt_dict
    
    
def get_t5_embeddings( prompts,args,model = None): 

    
    model = SentenceTransformer('sentence-transformers/sentence-t5-large').to(args.device) if model is None else model

    embeddings = model.encode(prompts,show_progress_bar =False)

    return torch.tensor(embeddings)

def get_t5():
    model = SentenceTransformer('sentence-transformers/sentence-t5-large')
    return model
    

def get_open_ai_embeddings( prompts, args,model = None): 

    embeddings_response = openai.Embedding.create(input = prompts, model="text-embedding-ada-002")

    embeddings = [x['embedding'] for x in embeddings_response['data']]


    return torch.tensor(embeddings)

def get_genres():
    genre_dict = {
    "unknown": 0,
    "Action": 1,
    "Adventure": 2,
    "Animation": 3,
    "Children's": 4,
    "Comedy": 5,
    "Crime": 6,
    "Documentary": 7,
    "Drama": 8,
    "Fantasy": 9,
    "Film-Noir": 10,
    "Horror": 11,
    "Musical": 12,
    "Mystery": 13,
    "Romance": 14,
    "SciFi": 15,
    "Thriller": 16,
    "War": 17,
    "Western": 18
    }
    # Convert keys to lowercase
    genre_dict_lower = {key.lower(): value for key, value in genre_dict.items()}

    return genre_dict_lower





def parse_args(notebook = False):  # Parse command line arguments
    parser = argparse.ArgumentParser(description="LLM4RecSys")
    parser.add_argument("--data_name", default='ml-100k', type=str)
    parser.add_argument("--log_file", default= 'model_logs/ml-100k/logging_llmMF.csv', type=str)
    parser.add_argument("--model_name", default='MFLLM', type=str)
    parser.add_argument("--emb_type", default='attn', type=str)
    parser.add_argument("--summary_style", default='topM', type=str)
    parser.add_argument("--embedding_module", default='openai', type=str)
    parser.add_argument("--embedding_dim" , default=1536, type=int)
    parser.add_argument("--bs" , default=4, type=int)
    parser.add_argument("--output_emb" , default=256, type=int)
    parser.add_argument("--top_for_rerank" , default=50, type=int)
    parser.add_argument("--num_layers" , default=3, type=int)
    parser.add_argument("--num_layers_transformer" , default=3, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--topk", default=20, type=int)
    parser.add_argument("--total_steps", default=1000, type=int)
    parser.add_argument("--attention_emb", default=512, type=int)
    parser.add_argument("--train", default=True, type=bool)
    parser.add_argument("--num_neg", default=1, type=int)
    parser.add_argument("--num_heads", default=4, type=int)
    parser.add_argument("--lr", default=.0001, type=float)
    parser.add_argument("--wd", default=0, type=float)
    parser.add_argument('--make_embeddings', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debugger', action='store_true')
    parser.add_argument('--cosine', action='store_true')
    parser.add_argument('--make_augmented', action='store_true')
    parser.add_argument("--max_steps", type=int, default=5)

    args = parser.parse_args() if not notebook else parser.parse_args(args=[])
    args.recon = False
    args.device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu') )
    args.model_save_path = f'/home/mila/e/emiliano.penaloza/scratch/saved_model/ml-100k/{args.model_name}'
    args.model_save_name = f"{args.model_save_path}_best_model_{args.lr}_{args.output_emb}_{args.num_heads}_{args.cosine}_{args.num_layers}.pth"

    args.model_log_name = f'{args.model_name}_{args.lr}_{args.output_emb}_{args.num_heads}_{args.cosine}_{args.num_layers}'
    return args
