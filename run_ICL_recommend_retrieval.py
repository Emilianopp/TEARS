from helper.in_context_learning import in_context_user_summary, in_context_recommendation, in_context_retrieval
from helper.builder import build_context, build_movie_candidates

import torch
import time

from argparse import ArgumentParser


def parse_args():  # Parse command line arguments
    parser = ArgumentParser(description="LLM4RecSys")
    parser.add_argument(
        "--model_name", default='gpt2', type=str,
        help="model_type: ['gpt2', 'llama-3b', 'llama-7b', 'falcon-7b-instruct', 'llama2']"
    )
    parser.add_argument(
        "--mode", default='ret', type=str, choices=['rec', 'ret']
    )

    return parser.parse_args()



if __name__ == "__main__":
    print("Start time:", time.asctime())
    args = parse_args()
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    built_context = build_context()

    # sample a user and genre
    user_id = 4169  # Specify the user ID
    genre = "Action"  # Specify the movie genre
    user_summary_list = in_context_user_summary(built_context, user_id, genre, args.model_name, device)

    # Recommendation or Retrieval
    if args.mode == 'rec':
        in_context_recommendation(built_context, user_id, genre, user_summary_list, args.model_name, device)
    elif args.mode == 'ret':
        # movies_file = 'data/ml-latest/movies.csv'
        # ratings_file = 'data/ml-latest/ratings.csv'
        # history_file = 'data/user_history/full_train_set.json'
        eval_file = '/home/haolun/projects/ctb-lcharlin/haolun/LLM4Rec_User_Summary/data_preprocessed/ml-1m/valid_set_leave_one.json'
        sim_file = '/home/haolun/projects/ctb-lcharlin/haolun/LLM4Rec_User_Summary/data_preprocessed/ml-1m/similar_movies.json'
        movie_candidates = build_movie_candidates([(user_id, genre)], sim_file, eval_file)
        in_context_retrieval(built_context, user_id, genre, user_summary_list, movie_candidates, args.model_name,
                             device)

    print("End time:", time.asctime())
    print("*********")
    print("Overall time:", time.time() - start_time)
