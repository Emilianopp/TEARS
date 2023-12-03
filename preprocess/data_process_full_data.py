import pandas as pd
import json
import random
import os
from collections import OrderedDict
import  debugpy
import argparse 



parser = argparse.ArgumentParser(description='Generate train, validation, and test splits for MovieLens dataset.')
parser.add_argument('--data_name', type=str, choices=['ml-1m', 'ml-100k'], default='ml-100k',
                        help='Name of the MovieLens dataset (ml-1m or ml-100k). Default is ml-1m.')
parser.add_argument('--timestamp', action='store_true')


args = parser.parse_args()
data_name = args.data_name

def save_sorted_json(data, filename):
    sorted_data = OrderedDict(sorted(data.items(), key=lambda t: t[0]))
    with open(filename, 'w') as f:
        json.dump(sorted_data, f, indent=4)


def extract_unique_movies(user_data):
    movies = set()
    for user_movies in user_data.values():
        for genre_movies in user_movies.values():
            if isinstance(genre_movies, list):  # normal train, valid, test set
                for movie in genre_movies:
                    movies.add(movie['title'])
            else:  # leave-one-out valid and test set
                movies.add(genre_movies['title'])
    return movies


def extract_user_genres(data):
    """Extract genres a user has interacted with."""
    user_genres = {}

    for user_id, genres in data.items():
        user_genres[user_id] = list(genres.keys())

    return user_genres

def load_movie_titles_from_dat(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    movie_titles = set()
    for line in lines:
        tokens = line.split("::")
        title = tokens[1]
        movie_titles.add(title)
    return movie_titles

def filter_movie_titles_by_valid_set(movies, valid_movie_titles):
    return [movie for movie in movies if movie['title'] in valid_movie_titles]

def sample_random(user_movies):
    validation_movie = user_movies.sample(n=1)

    user_movies = user_movies.drop(validation_movie.index)
    

    # Randomly choose one movie for test and remove it from user's ratings
    test_movie = user_movies.sample(n=1)
    user_movies = user_movies.drop(test_movie.index)
    return validation_movie,test_movie,user_movies

def sample_most_recent(user_movies):
    # Sort the DataFrame by the 'timestamp' column in descending order
    user_movies = user_movies.sort_values(by='timestamp', ascending=False)

    # Sample the two most recent movies
    most_recent_movies = user_movies.head(2)

    # Drop the two most recent movies from the user's ratings
    user_movies = user_movies.drop(most_recent_movies.index)

    # Split the two most recent movies into validation_movie and test_movie
    validation_movie = most_recent_movies.head(1)
    test_movie = most_recent_movies.tail(1)

    return validation_movie, test_movie, user_movies

    

def generate_train_val_test_splits(ratings, k, movie_metadata):
    # Count the number of ratings for each user
    user_counts = ratings['userId'].value_counts()

    # Filter out users with fewer than k ratings
    eligible_users = user_counts[user_counts > k].index
  

    # Initialize empty lists for training, validation, and test data
    train_data = []
    val_data = []
    test_data = []

    # Iterate through eligible users
    non_users = []
    for user_id in eligible_users:
        # Extract user's ratings
        user_ratings = ratings[ratings['userId'] == user_id]
        user_movies = user_ratings.merge(movie_metadata, left_on='movieId', right_on='movielens_id')
        user_movies = user_movies.dropna(subset=['summary'])  # Remove movies with 'NaN' summary
        user_movies = user_movies[user_movies['title'].isin(valid_movie_titles)]
        if len(user_movies) <= 3:
            non_users.append(user_id)
            continue

        # Randomly choose one movie for validation and remove it from user's ratings
        validation_movie, test_movie, user_movies = sample_random(user_movies) if not args.timestamp else sample_most_recent(user_movies)

        # Append user's training data
        train_data.append(user_movies)

        # Append user's validation data
        validation_movie['split'] = 'validation'
        val_data.append(validation_movie)

        # Append user's test data
        test_movie['split'] = 'test'
        test_data.append(test_movie)
 
    # Concatenate the dataframes to get the final splits
    train_data = pd.concat(train_data)
    val_data = pd.concat(val_data)
    test_data = pd.concat(test_data)
    return train_data, val_data, test_data,non_users



if __name__ == "__main__":
        


    valid_movie_titles = load_movie_titles_from_dat('../data/ml-100k/movies.dat')

    if data_name == 'ml-1m':
        ratings_file = '../data/ml-1m/ratings.dat'
        separator = "::"
        header = None
        rating_columns = ['userId', 'movieId', 'rating', 'timestamp']
        movie_metadata_file = '../data/ml-1m/movies.dat'
    elif data_name == 'ml-100k':
        ratings_file = '../data/ml-100k/u.data'
        separator = "\t"
        header = None
        rating_columns = ['userId', 'movieId', 'rating', 'timestamp']
        movie_metadata_file = '../data/ml-100k/u.item'


    ratings = pd.read_csv(ratings_file, sep=separator, header=header, encoding='ISO-8859-1')
    ratings.columns = ['userId', 'movieId', 'rating', 'timestamp']

    # Load movie metadata
    movie_metadata = pd.read_csv('../data/merged_asin_movielens_summary.csv')


    k = 10  # threshold for history length

    train_data, val_data, test_data,non_users = generate_train_val_test_splits(ratings, k, movie_metadata)

    valid_movie_titles = list(valid_movie_titles)
    train_data = train_data[train_data['title'].isin(valid_movie_titles)]


    val_data = val_data[val_data['title'].isin(valid_movie_titles)]
    test_data = test_data[test_data['title'].isin(valid_movie_titles)]

    # Check for overlapping user-movie pairs again after filtering
    train_user_movie_pairs = set(zip(train_data['userId'], train_data['movieId']))
    val_user_movie_pairs = set(zip(val_data['userId'], val_data['movieId']))
    test_user_movie_pairs = set(zip(test_data['userId'], test_data['movieId']))

    overlap_train_val = train_user_movie_pairs.intersection(val_user_movie_pairs)
    overlap_train_test = train_user_movie_pairs.intersection(test_user_movie_pairs)
    overlap_val_test = val_user_movie_pairs.intersection(test_user_movie_pairs)
    num_users_train = len(set(train_data['userId']))
    num_users_val = len(set(val_data['userId']))
    num_users_test = len(set(test_data['userId']))

    
    ### Error checking 
    assert not overlap_train_val, f"Overlap between train and validation sets in rows:\n{train_data[train_data[['userId', 'movieId']].apply(tuple, axis=1).isin(overlap_train_val)]}"
    assert not overlap_train_test, f"Overlap between train and test sets in rows:\n{train_data[train_data[['userId', 'movieId']].apply(tuple, axis=1).isin(overlap_train_test)]}"
    assert not overlap_val_test, f"Overlap between validation and test sets in rows:\n{val_data[val_data[['userId', 'movieId']].apply(tuple, axis=1).isin(overlap_val_test)]}"
    print("No overlap found after filtering by valid_movie_titles.")
    assert num_users_test == num_users_train and num_users_test == num_users_val, f'{num_users_val=} {num_users_train=} {num_users_test=}'
    print(f"All sets have the same number of users = {num_users_train}")

    #save data
   
    train_data.to_csv(f'../data_preprocessed/{data_name}/train_leave_one_out_{("timestamped" if args.timestamp else "")}.csv', index=False)
    val_data.to_csv(f'../data_preprocessed/{data_name}/validation_leave_one_out_{"timestamped" if args.timestamp else ""}.csv',index=False)
    test_data.to_csv(f'../data_preprocessed/{data_name}/test_leave_one_out_{"timestamped" if args.timestamp else ""}.csv', index=False)

    # Print a message
    print("data saved to CSVs to data_preprocessed folder")



