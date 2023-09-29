import pandas as pd
import json
import os
from preprocess.data_process import preprocess_data_summary, retrieve_data_reddit
import pandas as pd
import json
from collections import Counter
import pandas as pd
import json
from collections import defaultdict
import random


def build_context(global_path):
    # Load the CSV data into a pandas dataframe
    merged_df = pd.read_csv(
        f'{global_path}/data/merged_asin_movielens_summary.csv')
    # Prepare a dictionary with asin as key and movie details as value
    movie_dict = merged_df.set_index('asin')[['title', 'genres', 'summary']].to_dict(orient='index')
    # Load the reddit data
    with open(f'{global_path}/data/reddit.json') as f:
        reddit_data = json.load(f)

    # pre-process data
    if os.path.exists(
            f'{global_path}/data/merged_asin_movielens_summary.csv'):
        pass
    else:
        preprocess_data_summary()

    # obtain demonstration
    if os.path.exists(f'{global_path}/data/built_context.json'):
        with open(f'{global_path}/data/built_context.json', 'r') as file:
            built_context = json.load(file)
    else:
        built_context = retrieve_data_reddit(reddit_data, movie_dict)
        # Open the file in write mode
        with open(f'{global_path}/data/built_context.json', 'w') as json_file:
            json.dump(built_context, json_file)
    return built_context


# def build_movie_candidates(user_id, genre, sim_file, eval_file):
#     # load the eval_file
#     with open(eval_file, 'r') as f:
#         eval_movie_data = json.load(f)
#
#     # load the similarity file
#     with open(sim_file, 'r') as f:
#         sim_file = json.load(f)
#
#     # Get the last movie directly
#     eval_movie = eval_movie_data[str(user_id)][genre]
#     eval_movie_title = eval_movie['title']
#
#     # get the retrieval candidates and shuffle
#     movie_candidates = sim_file[eval_movie_title][genre]
#     random.shuffle(movie_candidates)
#
#     return movie_candidates

def build_movie_candidates(user_id_genre_pairs, sim_file, eval_file):
    # load the eval_file
    with open(eval_file, 'r') as f:
        eval_movie_data = json.load(f)

    # load the similarity file
    with open(sim_file, 'r') as f:
        sim_file_data = json.load(f)

    all_movie_candidates = []
    for user_id, genre in user_id_genre_pairs:
        # Get the last movie directly
        eval_movie = eval_movie_data[str(user_id)][genre]
        eval_movie_title = eval_movie['title']

        # get the retrieval candidates and shuffle
        movie_candidates = sim_file_data[eval_movie_title][genre]
        random.shuffle(movie_candidates)

        all_movie_candidates.append(movie_candidates)

    return all_movie_candidates
