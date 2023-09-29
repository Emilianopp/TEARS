import pandas as pd
from collections import defaultdict
import json
import heapq
import os


def convert_ml100k_uitem_format(input_file, output_file):
    # Define column names
    movie_columns = ['movieId', 'title', 'release date', 'video release date', 'IMDb URL',
                     'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime',
                     'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                     'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    # Read the u.item file
    movies_df = pd.read_csv(input_file, sep='|', header=None, encoding='ISO-8859-1', names=movie_columns)

    # For MovieLens 100k, the genres are separate columns with binary values
    genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime',
                  'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                  'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    # Convert binary genre columns into a list of genres
    movies_df['genres'] = movies_df[genre_cols].apply(lambda x: '|'.join(x.index[x.astype(bool)]), axis=1)

    # Select only the 'movieId', 'title', and 'genres' columns
    movies_df = movies_df[['movieId', 'title', 'genres']]

    # Write to file line by line with the required format
    with open(output_file, 'w') as f:
        for _, row in movies_df.iterrows():
            f.write(f"{row['movieId']}::{row['title']}::{row['genres']}\n")


def build_similar_movies(data_name):
    if data_name == 'ml-1m':
        movies_file = '../data/ml-1m/movies.dat'
        ratings_file = '../data/ml-1m/ratings.dat'
        output_file = '../data_preprocessed/ml-1m/similar_movies.json'
        separator = "::"
        header = None
        movie_columns = ['movieId', 'title', 'genres']
        rating_columns = ['userId', 'movieId', 'rating', 'timestamp']
    elif data_name == 'ml-100k':
        movies_file = '../data/ml-100k/movies.dat'
        ratings_file = '../data/ml-100k/u.data'
        output_file = '../data_preprocessed/ml-100k/similar_movies.json'
        separator = "\t"
        header = None
        # the u.item file has 24 columns in total
        movie_columns = ['movieId', 'title', 'release date', 'video release date', 'IMDb URL',
                         'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime',
                         'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                         'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        rating_columns = ['userId', 'movieId', 'rating', 'timestamp']

    movies_df = pd.read_csv(movies_file, sep="::", header=header, encoding='ISO-8859-1')
    movies_df.columns = ['movieId', 'title', 'genres']
    movies_df['genres'] = movies_df['genres'].apply(lambda x: x.split("|"))

    # Load all user ratings
    ratings_df = pd.read_csv(ratings_file, sep=separator, header=header, encoding='ISO-8859-1')
    ratings_df.columns = ['userId', 'movieId', 'rating', 'timestamp']

    print("For each movie, list all users who have watched it")
    movie_users = defaultdict(set)
    for index, row in ratings_df.iterrows():
        movie_users[row['movieId']].add(row['userId'])

    print("For each pair of movies, count the number of shared users")
    shared_users = defaultdict(lambda: defaultdict(int))
    for movie1 in movie_users.keys():
        for movie2 in movie_users.keys():
            if movie1 != movie2:
                shared_users[movie1][movie2] = len(movie_users[movie1].intersection(movie_users[movie2]))

    print("For each genre, find the most similar movies")
    similar_movies = defaultdict(lambda: defaultdict(list))
    for movie in shared_users.keys():
        genres = movies_df[movies_df['movieId'] == movie]['genres'].values[0]
        current_title = movies_df[movies_df['movieId'] == movie]['title'].values[0]

        for genre in genres:
            # Find other movies of the same genre
            other_movies = \
                movies_df[(movies_df['genres'].apply(lambda x: genre in x)) & (movies_df['movieId'] != movie)][
                    'movieId'].values

            # Find the top 19 most similar movies
            top_similar_movies = heapq.nlargest(19, other_movies, key=lambda x: shared_users[movie][x])

            # Store the current movie's title
            similar_movies[current_title][genre].append(current_title)

            # Store the titles of the top similar movies
            for top_movie in top_similar_movies:
                title = movies_df[movies_df['movieId'] == top_movie]['title'].values[0]
                similar_movies[current_title][genre].append(title)  # Using the movie title as the key

    # Save the similar movies dictionary to a JSON file
    with open(output_file, 'w') as f:
        json.dump(similar_movies, f, indent=4)


if __name__ == "__main__":
    print(os.getcwd())
    build_similar_movies('ml-100k')
    # convert_ml100k_uitem_format('../data/ml-100k/u.item', '../data/ml-100k/movies.dat')
