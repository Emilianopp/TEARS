import json
import pickle
import numpy as np
from torch.nn.functional import one_hot
from scipy.sparse import csr_matrix
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
import random

class PromptDataset(Dataset):
    def __init__(self, prompts):
        self.data = list(prompts.items())
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_id, text = self.data[idx]
        # You can preprocess the text or perform any other necessary steps here
        return {'user_id': user_id, 'text': text}

class RecDatasetNegatives(Dataset):
    def __init__(self, train_matrix, num_negatives=3):
        self.train_matrix = train_matrix
        self.num_negatives = num_negatives
        self.users, self.movies, self.ratings = train_matrix['userId'], train_matrix['movieId'], train_matrix['rating']
        self.movie_set = set(train_matrix['movieId'].unique())
        self.neg_sampler = self._initialize_negative_sampler()

    def _initialize_negative_sampler(self):
        neg_sampler = {}
        for user_id in self.users.unique():
            neg_samples = self._sample_negatives(user_id)
            np.random.shuffle(neg_samples)
            neg_samples = iter(neg_samples)
            neg_sampler[user_id] = self._infinite_generator(neg_samples)
        return neg_sampler

    def _infinite_generator(self, iterator):
        # Infinitely yield items from the iterator and reinsert them when exhausted
        items = []
        while True:
            try:
                item = next(iterator)
                items.append(item)
                yield item
            except StopIteration:
                np.random.shuffle(items)
                iterator = iter(items)

    def __len__(self):
        return len(self.train_matrix)

    def __getitem__(self, idx):
        user_id, movie_id, rating = self.users[idx], self.movies[idx], self.ratings[idx]

        # Positive sample
        positive_sample = torch.tensor([ movie_id], dtype=torch.float32)

        # Negative samples
        negative_samples = self.neg_sampler[user_id]

        return user_id,positive_sample.int(), next(negative_samples).int()

    def _sample_negatives(self, user_id):
        # Sample negative items that the user has not interacted with
        negative_samples = []
        for _ in range(self.num_negatives):
            negative_item = torch.randint(1, len(self.movie_set) + 1, (1,)).item()
            while negative_item in self.train_matrix[self.train_matrix['userId'] == user_id]['movieId'].values:
                negative_item = torch.randint(1, len(self.movie_set) + 1, (1,)).item()
            negative_samples.append(negative_item)

        return torch.tensor(negative_samples, dtype=torch.long)

    def next_batch(self, batch_size):
        for start in range(0, len(self.train_matrix), batch_size):
            end = min(start + batch_size, len(self.train_matrix))
            batch_data = self.__getitem__(slice(start, end))
            yield batch_data
            
            

from torch.nn.utils.rnn import pad_sequence
class RecDatasetFull(Dataset):
    def __init__(self, train_matrix, num_movies):
        self.train_matrix = train_matrix
        self.num_movies = num_movies
        self.user_movie_dict = self.create_user_movie_dict()
        self.movie_one_hot_dict = self.create_movie_one_hot_dict()

    def create_user_movie_dict(self):
        user_movie_dict = {}
        for index, row in self.train_matrix.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            if user_id in user_movie_dict:
                user_movie_dict[user_id].append(movie_id)
            else:
                user_movie_dict[user_id] = [movie_id]
        return user_movie_dict

    def create_movie_one_hot_dict(self):
        movie_one_hot_dict = {}
        for movie_id in self.train_matrix['movieId'].unique():
            one_hot_encoded = one_hot(torch.tensor([movie_id], dtype=torch.long), num_classes=self.num_movies).squeeze()
            movie_one_hot_dict[movie_id] = one_hot_encoded
        return movie_one_hot_dict

    def __len__(self):
        return len(self.user_movie_dict)

    def __getitem__(self, idx):
        user_id = list(self.user_movie_dict.keys())[idx]
        movie_ids = self.user_movie_dict[user_id]

        # Use precomputed one-hot encoded tensors for each movie
        movie_ids_one_hot = sum(self.movie_one_hot_dict[movie_id] for movie_id in movie_ids)

        return  torch.tensor(user_id, dtype=torch.long), movie_ids_one_hot

def custom_collate(batch):
    # Custom collate function to handle batches with different sizes
    user_ids , positive_samples, negative_samples = zip(*batch)
    positive_samples = torch.stack(positive_samples)
    negative_samples = torch.stack(negative_samples)
    user_ids = torch.tensor(user_ids)
    return user_ids,positive_samples, negative_samples


    
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

def map_id_to_genre(movies_file):
    with open(movies_file, 'r') as f:
        lines = f.readlines()

    mapping = {}
    for line in lines:
        tokens = line.split("::")
        movie_id = int(tokens[0])
        title = tokens[2].strip('\n')
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
    
def rating_train_matrix_pandas(data):
    
    # Create a mapping of unique user and movie IDs to matrix indices
    unique_users = sorted(data['userId'].unique())
    unique_movies = sorted(data['movieId'].unique())

    user_to_index = {user_id: index for index, user_id in enumerate(unique_users)}
    movie_to_index = {movie_id: index for index, movie_id in enumerate(unique_movies)}

    # Create an empty user-item matrix filled with NaN values
    num_users = len(unique_users)
    num_movies = len(unique_movies)
    user_item_matrix = np.full((num_users, num_movies), np.nan, dtype=float)

    # Fill in the matrix with ratings from your training data
    for _, row in data.iterrows():
        user_index = user_to_index[row['userId']]
        movie_index = movie_to_index[row['movieId']]
        user_item_matrix[user_index, movie_index] = row['rating']

    return user_item_matrix


def binary_train_matrix_pandas(data,num_movies=1682,plus_one_users=True):
    
    # Create a mapping of unique user and movie IDs to matrix indices
    
    unique_users = sorted(data['userId'].unique())
    unique_movies = sorted(data['movieId'].unique())

    # Create an empty user-item matrix filled with NaN values
    num_users = len(unique_users) + (1 if plus_one_users else 0) 
    user_item_matrix = np.zeros((num_users, num_movies), dtype=float)

    # Fill in the matrix with ratings from your training data
    for _, row in data.iterrows():
        user_index = row['userId'] 
        movie_index = row['movieId'] 
        user_item_matrix[user_index, movie_index] = 1

    return user_item_matrix

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
