import pandas as pd
import json
import random
import os
from collections import OrderedDict


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


def sample_user_history(movie_metadata, data_name, num_users=1):
    valid_movie_titles = load_movie_titles_from_dat('../data/ml-100k/movies.dat')

    if data_name == 'ml-1m':
        ratings_file = '../data/ml-1m/ratings.dat'
        separator = "::"
        header = None
        rating_columns = ['userId', 'movieId', 'rating', 'timestamp']
    elif data_name == 'ml-100k':
        ratings_file = '../data/ml-100k/u.data'
        separator = "\t"
        header = None
        rating_columns = ['userId', 'movieId', 'rating', 'timestamp']

    ratings = pd.read_csv(ratings_file, sep=separator, header=header, encoding='ISO-8859-1')
    ratings.columns = ['userId', 'movieId', 'rating', 'timestamp']

    k = 10  # threshold for history length

    # Count the number of ratings for each user
    user_counts = ratings['userId'].value_counts()
    # eligible_users = user_counts[(user_counts > 10) & (user_counts < 20)].index
    eligible_users = user_counts[user_counts > k].index

    # If there are not enough eligible users, raise an exception
    if len(eligible_users) < num_users:
        print("We will use all users.")
        user_ids = list(eligible_users)
    else:
        # Randomly sample user IDs
        user_ids = random.sample(list(eligible_users), num_users)

    print("number of users:", len(user_ids))

    # Dictionaries to store training, holdout and test sets for each user
    user_data_train = {}
    user_data_valid = {}
    user_data_test = {}
    user_data_leave_one_train = {}
    user_data_leave_one_valid = {}
    user_data_leave_one_test = {}
    user_genre_interactions = {}  # Dictionary to store user-genre interactions

    for user_id in user_ids:
        user_ratings = ratings[ratings['userId'] == user_id]  # filter the ratings for the user
        user_ratings = user_ratings.sort_values('timestamp')  # sort by timestamp

        # Merge the user's ratings with the movie_metadata. Then obtain history_leave_one
        user_movies = user_ratings.merge(movie_metadata, left_on='movieId', right_on='movielens_id')
        user_movies = user_movies.dropna(subset=['summary'])  # Remove movies with 'NaN' summary
        user_history_leave_one = user_movies[['title', 'movieId', 'genres', 'summary']].to_dict('records')

        """Cluster user history based on tags"""
        unique_genres = set()
        genre_movies = {}
        genre_movie_count = {}

        # Loop over the list of movie dictionaries
        for movie in user_history_leave_one:
            genres = movie['genres'].split('|')  # Split the genres string into a list of genres
            unique_genres.update(genres)  # Add the genres to the unique_genres set

            # Loop over the genres for this movie
            for genre in genres:
                # If this genre is not yet a key in genre_movies, add it with an empty list as its value
                if genre not in genre_movies:
                    genre_movies[genre] = []
                    genre_movie_count[genre] = 0

                # Append this movie to the list of movies for this genre
                genre_movies[genre].append(movie)
                genre_movie_count[genre] += 1


        # Create training, holdout and test sets
        train_set = {}
        valid_set = {}
        test_set = {}
        leave_one_train_set = {}
        leave_one_valid_set = {}
        leave_one_test_set = {}

        # Loop over the genre_movies dictionary
        for genre, movies in genre_movies.items():
            movies = filter_movie_titles_by_valid_set(movies, valid_movie_titles)
            count = len(movies)
            # Ignore genres with less than 5 movies
            if count < 5:
                continue

            if count < 10:
                holdout_count = test_count = 1
                training_count = count - 2
            else:
                # Divide 80% of items for training, 10% for holdout and 10% for test
                training_count = int(count * 0.8)
                holdout_count = test_count = int(count * 0.1)

            train_set[genre] = movies[:training_count]
            valid_set[genre] = movies[training_count:training_count + holdout_count]
            test_set[genre] = movies[training_count + holdout_count:]

            leave_one_train_set[genre] = movies[:-2]
            leave_one_valid_set[genre] = movies[-2]
            leave_one_test_set[genre] = movies[-1]

        # Store the training, holdout and test sets for the user in separate dictionaries
        user_data_train[user_id] = train_set
        user_data_valid[user_id] = valid_set
        user_data_test[user_id] = test_set
        user_data_leave_one_train[user_id] = leave_one_train_set
        user_data_leave_one_valid[user_id] = leave_one_valid_set
        user_data_leave_one_test[user_id] = leave_one_test_set

    train_movies = extract_unique_movies(user_data_leave_one_train)
    valid_movies = extract_unique_movies(user_data_leave_one_valid)
    test_movies = extract_unique_movies(user_data_leave_one_test)

    all_unique_movies = train_movies.union(valid_movies, test_movies)
    print("Total number of unique movies across train, validation, and test sets:", len(all_unique_movies))

    user_genres = extract_user_genres(user_data_leave_one_test)

    # Convert sets to lists before JSON serialization as sets are not JSON serializable
    for user_id, genre_set in user_genres.items():
        user_genres[user_id] = list(genre_set)

    # Save the training, holdout, and test sets to JSON
    save_sorted_json(user_data_train, '../data_preprocessed/{}/data_split/train_set.json'.format(data_name))
    save_sorted_json(user_data_valid, '../data_preprocessed/{}/data_split/valid_set.json'.format(data_name))
    save_sorted_json(user_data_test, '../data_preprocessed/{}/data_split/test_set.json'.format(data_name))
    save_sorted_json(user_data_leave_one_train, '../data_preprocessed/{}/data_split/train_set_leave_one.json'.format(data_name))
    save_sorted_json(user_data_leave_one_valid, '../data_preprocessed/{}/data_split/valid_set_leave_one.json'.format(data_name))
    save_sorted_json(user_data_leave_one_test, '../data_preprocessed/{}/data_split/test_set_leave_one.json'.format(data_name))

    save_sorted_json(user_genres, '../data_preprocessed/{}/user_genre.json'.format(data_name))


if __name__ == "__main__":
    print(os.getcwd())
    movie_metadata = pd.read_csv('../data/merged_asin_movielens_summary.csv')
    num_users = 10000  # Specify the number of users to sample
    data_name = 'ml-100k'
    sample_user_history(movie_metadata, data_name, num_users)
