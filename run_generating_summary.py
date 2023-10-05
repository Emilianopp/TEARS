from helper.in_context_learning_batch import in_context_user_summary, in_context_retrieval
from helper.builder import build_context, build_movie_candidates
import torch
import time
import json
from torch.utils.data import Dataset, DataLoader
from data.dataloader import get_dataloader
from argparse import ArgumentParser
import os


def parse_args():  # Parse command line arguments
    parser = ArgumentParser(description="LLM4RecSys")
    parser.add_argument("--data_name", default='ml-100k', type=str)
    parser.add_argument(
        "--model_name", default='gpt3.5', type=str,
        help="model_type: ['gpt2', 'llama','llama-3b', 'llama-7b', 'falcon', 'llama2', 'gpt3.5', 'gpt4']"
    )
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_users", default=3, type=int)
    parser.add_argument("--mode", default='sum', type=str, choices=['sum', 'rec', 'ret', 'rerank'])
    # setting for summary generation
    parser.add_argument("--in_context", default=1, type=int)
    parser.add_argument("--only_title", default=0, type=int)

    return parser.parse_args()


if __name__ == "__main__":
    print("Start time:", time.asctime())
    args = parse_args()
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu') )
    # device = torch.device('cpu')
    # global_path = '/home/haolun/projects/ctb-lcharlin/haolun/LLM4Rec_User_Summary'
    # global_path = '/Users/haolunwu/Documents/GitHub/LLM4Rec_User_Summary'
    #set the global path enviroment variable as the path to the project not dependant on the machine
    global_path = os.path.dirname(os.path.abspath(__file__))
  

    built_context = build_context(global_path)

    user_genre_file = f"{global_path}/data_preprocessed/{args.data_name}/user_genre.json"

    user_genre_dataloader = get_dataloader(user_genre_file, batch_size=args.batch_size, num_users=args.num_users, user_start=764, user_end=1000)
    num_of_batch = len(user_genre_dataloader)
    print("Num of batch:", num_of_batch)

    if args.mode == 'sum':
        user_summary_list = in_context_user_summary(global_path , built_context, args.model_name, device, args.data_name,
                                                    user_genre_dataloader, args.in_context, args.only_title)

    # for batch in dataloader:
    #     user_id_genre_pairs = list(zip(batch["user_id"], batch["genre"]))
    #     print("user_id_genre_pairs:", user_id_genre_pairs)
    #     # # sample user_id_genre_pairs
    #     # user_id_genre_pairs = [(405, "Drama"), (405, "Romance"), (450, "Drama"), (450, "Romance")]
    #     # # user_id_genre_pairs = [(4169, "Action")]

    # elif args.mode == 'ret':
    #     # movies_file = 'data/ml-latest/movies.csv'
    #     # ratings_file = 'data/ml-latest/ratings.csv'
    #     # history_file = 'data/user_history/full_train_set.json'
    #     eval_file = '/home/haolun/projects/ctb-lcharlin/haolun/LLM4Rec_User_Summary/data_preprocessed/{args.data_name}/valid_set_leave_one.json'
    #     sim_file = '/home/haolun/projects/ctb-lcharlin/haolun/LLM4Rec_User_Summary/data_preprocessed/{args.data_name}/similar_movies.json'
    #     movie_candidates_list = build_movie_candidates(user_id_genre_pairs, sim_file, eval_file)
    #     in_context_retrieval(built_context, user_id_genre_pairs, user_summary_list, movie_candidates_list,
    #                          args.model_name, device, args.data_name, batch=True)

    print("End time:", time.asctime())
    print("*********")
    print("Overall time:", time.time() - start_time)
