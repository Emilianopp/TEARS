import sys
import logging
import numpy as np
import torch
import json
import time
import torch.optim as optim
import torch
from scipy.sparse import csr_matrix

from helper.sampler import NegSampler, negsamp_vectorized_bsearch_preverif
from argparse import ArgumentParser
from model.MF import MatrixFactorization
from helper.eval_metrics import precision_at_k, recall_at_k, mapk, ndcg_k
from helper.dataloader import load_dataset, map_title_to_id, convert_titles_to_ids, \
    create_train_matrix_and_actual_lists

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True
logger = logging.getLogger(__name__)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


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


def generate_pred_list(model, train_matrix, topk=20):
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
        batch_user_ids = torch.from_numpy(np.array(batch_user_index)).type(torch.LongTensor).to(args.device)

        rating_pred = model.predict(batch_user_ids)
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


def generate_rankings_for_all_users(model, num_users, num_items):
    rankings = np.zeros((num_users, num_items), dtype=int)

    # For each user, generate scores for all items, then rank the items
    for user_id in range(num_users):
        user_id_tensor = torch.tensor([user_id], device=model.device)
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


def main_MF(args):
    t_start = time.time()
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
    print("train_matrix:", train_matrix.shape)

    # Negative item pre-sampling (assuming you have a method for this)
    user_neg_items = neg_item_pre_sampling(train_matrix)
    pre_samples = {'user_neg_items': user_neg_items}
    print("Pre sampling time:{}".format(time.time() - t_start))

    # 2. Model Creation
    num_users, num_items = train_matrix.shape
    model = MatrixFactorization(num_users, num_items, args).to(args.device)
    optimizer = optim.Adam(model.myparameters, lr=args.lr, weight_decay=args.wd)

    # Create negative sampler
    neg_sampler = NegSampler(train_matrix, pre_samples, batch_size=args.batch_size, num_neg=args.num_neg)
    num_batches = train_matrix.count_nonzero() // args.batch_size

    # Early stopping parameters
    best_recall = 0.0
    patience = 10  # Number of epochs to wait for recall to improve
    counter = 0
    model_save_path = f'../saved_model/ml-100k/{args.model_name}'

    # 3. Training Loop
    model.train()
    for epoch in range(args.epochs):
        loss = 0.0
        for _ in range(num_batches):
            batch_user_id, batch_item_id, neg_samples = neg_sampler.next_batch()
            user, pos, neg = batch_user_id, batch_item_id, np.squeeze(neg_samples)

            user_emb, pos_emb, neg_emb = model(user, pos, neg)

            batch_loss = model.bpr_loss(user_emb, pos_emb, neg_emb)
            batch_loss = torch.mean(batch_loss)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()

        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {loss / num_batches}")

        # Check Recall@20 on validation
        model.eval()
        pred_list_val = generate_pred_list(model, train_matrix, topk=20)
        recall_val = recall_at_k(actual_list_val, pred_list_val, 20)
        # print(f"Validation Recall@20: {recall_val}")

        # Early stopping check
        if recall_val > best_recall:
            best_recall = recall_val
            counter = 0
            # Save the model
            torch.save(model.state_dict(), f"{model_save_path}_best_model.pth")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break
    # 4. Evaluation
    # Load best model for evaluation
    model.load_state_dict(torch.load(f"{model_save_path}_best_model.pth"))
    model.eval()

    # Validation set results (optional: you might want to skip this as we've evaluated during training)
    # pred_list_val = generate_pred_list(model, train_matrix, topk=args.topk)
    # precision_val = precision_at_k(actual_list_val, pred_list_val, args.topk)
    # recall_val = recall_at_k(actual_list_val, pred_list_val, args.topk)
    # ndcg_val = ndcg_k(actual_list_val, pred_list_val, args.topk)
    # print(f"Validation Precision@{args.topk}: {precision_val}")
    # print(f"Validation Recall@{args.topk}: {recall_val}")
    # print(f"Validation NDCG@{args.topk}: {ndcg_val}")

    # Test set results
    pred_list_test = generate_pred_list(model, train_matrix, topk=args.topk)
    actual_list_test = actual_list_test  # Adjust based on your data format
    precision_test = precision_at_k(actual_list_test, pred_list_test, args.topk)
    recall_test = recall_at_k(actual_list_test, pred_list_test, args.topk)
    ndcg_test = ndcg_k(actual_list_test, pred_list_test, args.topk)
    print(f"Test Precision@{args.topk}: {precision_test}")
    print(f"Test Recall@{args.topk}: {recall_test}")
    print(f"Test NDCG@{args.topk}: {ndcg_test}")

    # Save pretrained embeddings
    torch.save(model.user_embeddings.weight.data, f"{model_save_path}_user_embeddings.pth")
    torch.save(model.item_embeddings.weight.data, f"{model_save_path}_item_embeddings.pth")

    rankings_matrix = generate_rankings_for_all_users(model, num_users=num_users, num_items=num_items)
    np.save(f"{model_save_path}_rankings_matrix.npy", rankings_matrix)
    print("rankings_matrix:", rankings_matrix)

    movie_id_to_genres = convert_ids_to_genres("../data/ml-100k/movies.dat")
    generate_and_save_rankings_json(rankings_matrix, args.topk, args.top_for_rerank, movie_id_to_genres,
                                    f"{model_save_path}_user_genre_rankings.json",
                                    "../data_preprocessed/ml-100k/user_genre.json")

    return model, rankings_matrix


if __name__ == "__main__":
    class Arguments:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dim = 64  # embedding dimension
        lr = 1e-3  # learning rate
        wd = 0  # wd
        epochs = 1000
        batch_size = 1024
        num_neg = 1
        topk = 1682
        top_for_rerank = 50
        model_name = 'BPRMF'


    args = Arguments()
    model, rankings_matrix = main_MF(args)
