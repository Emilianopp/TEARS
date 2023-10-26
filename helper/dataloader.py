import json
import pickle
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict

# Load datasets
def load_dataset(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
# Create a mapping from movie titles to movie IDs
def map_title_to_id(movies_file):
    with open(movies_file, 'r') as f:
        lines = f.readlines()

    mapping = {}
    for line in lines:
        tokens = line.split("::")
        movie_id = int(tokens[0])
        title = tokens[1]
        mapping[title] = movie_id

    return mapping

def map_id_to_title(movies_file):
    with open(movies_file, 'r') as f:
        lines = f.readlines()

    mapping = {}
    for line in lines:
        tokens = line.split("::")
        movie_id = int(tokens[0])
        title = tokens[1]
        mapping[movie_id] = title

    return mapping


# Convert movie titles in a dataset to their corresponding IDs
def convert_titles_to_ids(dataset, title_to_id_map):
    for user_id, genre_data in dataset.items():
        for genre, items in genre_data.items():
            if isinstance(items, list):
                for item in items:
                    item['movieId'] = title_to_id_map[item['title']]
                    del item['title']
            else:
                items['movieId'] = title_to_id_map[items['title']]
                del items['title']
    return dataset


def create_full_dataset_matrix(full_data,movie_title_to_id):

    num_users = max([int(user) for user in full_data.keys()])
    num_movies = max(movie_title_to_id.values())

    # Create an empty matrix with dimensions (num_users, num_movies)
    full_matrix = [[0 for _ in range(num_movies)] for _ in range(num_users)]

    for user_id, genre_data in full_data.items():
        for genre, items in genre_data.items():
            if isinstance(items, list):
                for item in items:
                    full_matrix[int(user_id) - 1][item['movieId'] - 1] = 1
            else:
                full_matrix[int(user_id) - 1][items['movieId'] - 1] = 1
    return full_matrix
def create_full_data(train_data,val_data,test_data):
    output_dict = defaultdict(dict)
    for user,item in train_data.items():
        for genre,movie_list in item.items():

            output_dict[user][genre] = movie_list +[ val_data[user][genre],test_data[user][genre]]
    return output_dict 
    

def create_train_matrix_and_actual_lists(train_data, valid_data, test_data, movie_title_to_id):
    num_users = max([int(user) for user in train_data.keys()])
    num_movies = max(movie_title_to_id.values())

    # Create an empty matrix with dimensions (num_users, num_movies)
    train_matrix = [[0 for _ in range(num_movies)] for _ in range(num_users)]

    for user_id, genre_data in train_data.items():
        for genre, items in genre_data.items():
            if isinstance(items, list):
                for item in items:
                    train_matrix[int(user_id) - 1][item['movieId'] - 1] = 1
            else:
                train_matrix[int(user_id) - 1][items['movieId'] - 1] = 1

    actual_list_val = {}
    actual_list_test = {}

    # For validation data
    for user_id, genre_data in valid_data.items():
        actual_list_val[user_id] = []
        for genre, items in genre_data.items():
            if isinstance(items, list):
                for item in items:
                    actual_list_val[user_id].append(item['movieId'] - 1)
            else:
                actual_list_val[user_id].append(items['movieId'] - 1)
        actual_list_val[user_id] = list(set(actual_list_val[user_id]))

    # For test data
    for user_id, genre_data in test_data.items():
        actual_list_test[user_id] = []
        for genre, items in genre_data.items():
            if isinstance(items, list):
                for item in items:
                    actual_list_test[user_id].append(item['movieId'] - 1)
            else:
                actual_list_test[user_id].append(items['movieId'] - 1)
        actual_list_test[user_id] = list(set(actual_list_test[user_id]))

    actual_list_val = list(actual_list_val.values())
    actual_list_test = list(actual_list_test.values())

    return train_matrix, actual_list_val, actual_list_test


def main():
    # Load datasets
    train_data = load_dataset("./data_preprocessed/ml-100k/data_split/train_set_leave_one.json")
    valid_data = load_dataset("./data_preprocessed/ml-100k/data_split/valid_set_leave_one.json")
    test_data = load_dataset("./data_preprocessed/ml-100k/data_split/test_set_leave_one.json")

    # Create movie title to ID mapping
    movie_title_to_id = map_title_to_id("./data/ml-100k/movies.dat")

    # Convert titles to IDs in each dataset
    train_data = convert_titles_to_ids(train_data, movie_title_to_id)
    valid_data = convert_titles_to_ids(valid_data, movie_title_to_id)
    test_data = convert_titles_to_ids(test_data, movie_title_to_id)

    # Create train matrix and actual lists
    train_matrix, actual_list_val, actual_list_test = create_train_matrix_and_actual_lists(train_data, valid_data,
                                                                                           test_data, movie_title_to_id)
    # train_matrix = csr_matrix(train_matrix)

    # Merge all datasets into a single dictionary
    full_data = {
        'train': train_data,
        'valid': valid_data,
        'test': test_data,
        'actual_list_val': actual_list_val,
        'actual_list_test': actual_list_test,
        'train_matrix': train_matrix
    }
    print("actual_list_val:", actual_list_val)
    print("actual_list_test:", actual_list_test)
    print("train_matrix:", np.array(train_matrix).shape)

    # Save consolidated data
    with open("./data_preprocessed/ml-100k/full_data.json", 'w') as f:
        json.dump(full_data, f, indent=4)


if __name__ == "__main__":
    main()
