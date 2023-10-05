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

from helper.sampler import NegSampler, negsamp_vectorized_bsearch_preverif
from argparse import ArgumentParser
from model.MF import MatrixFactorization
from helper.eval_metrics import precision_at_k, recall_at_k, mapk, ndcg_k
from helper.dataloader import load_dataset, map_title_to_id, convert_titles_to_ids, \
    create_train_matrix_and_actual_lists

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


def generate_pred_list(model, train_matrix,args,user_embeddings, topk=20,summary_encoder = None):
    num_users = train_matrix.shape[0]
    batch_size = 1024
    num_batches = int(num_users / batch_size) + 1
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

        batch_user_index = user_indexes[start:end]
    
       

        batch_user_emb = user_embeddings[batch_user_index].to(args.device)

        rating_pred = model.predict_recon(batch_user_emb,summary_encoder) if args.recon else  model.predict(batch_user_emb)
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
        df = df.append(log_data, ignore_index=True)
        df.to_csv(log_file, index=False)
 