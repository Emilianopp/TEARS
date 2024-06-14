import os
import sys 
sys.path.append('../')
import json
import time
import random 
from transformers import pipeline
import torch
import re
from collections import defaultdict 
import openai
# from tenacity import retry, wait_exponential, stop_after_attempt
from tqdm import tqdm
import torch
import time
import json

from argparse import ArgumentParser
import os
import pandas as pd 
import debugpy
from dotenv import load_dotenv

global_path = '/home/mila/e/emiliano.penaloza/LLM4REC'


# Load environment variables from the .env file into the script's environment
load_dotenv()
key = os.getenv("OPEN-AI-SECRET")
parser = ArgumentParser(description="Your script description here")
parser.add_argument("--gpt_version", default="gpt-4-1106-preview", help="GPT model version")
parser.add_argument("--max_tokens", type=int, default=600, help="Maximum number of tokens for the response")
parser.add_argument("--len", type=int, default=400, help="Maximum number of tokens for the response")
parser.add_argument("--debug", action='store_true', help="Whether to run in debug mode")
parser.add_argument("--data_name", default = 'ml-1m', help="Name of the dataset to use")
parser.add_argument("--save_path", default = '', help="Name of the dataset to use")
parser.add_argument("--books", action='store_true', help="Whether to run in books prompt")

args = parser.parse_args()
train_data = f'../data_preprocessed/{args.data_name}/prompt_set_timestamped.csv'
args = parser.parse_args()

openai.api_key = key



def load_existing_data(file_path):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r') as fp:
            existing_data = json.load(fp)
        return existing_data

    return {}


def generate_prompts_recent(train_data,only_title=1,num_movies = 30): 
    prompts = {}
    for id in train_data.userId.unique():        

        user_data = train_data[train_data['userId'] == id]

        # Sort the user data by the 'timestamp' column in descending order
        user_data = user_data.sort_values(by='timestamp', ascending=False)

        # Take the top 30 most recent movies
        user = user_data.head(num_movies)
        
        movie_titles = user['title'].tolist()

        # movie_summaries= user['summary'].tolist()
        
        ratings = user['rating'].tolist()
        genres = user['genres'].tolist()

        prompt = ""

        for (title,rating),genres in zip(zip(movie_titles,ratings),genres):


            prompt += f"\n{title}"

            # else:
            #     prompt += f"\n{title}: {description}"
                
            prompt += f'\nRating: {rating}\n'
            prompt += f'\Genres: {genres}\n'
        prompts[id] = prompt    
    return prompts


        


if __name__ == "__main__":
    random.seed(2024)
    data = pd.read_csv(args.train_data)
    strong_generalization_set = pd.read_csv(args.test_data)
    strong_gen_val_user_set = random.sample(list(strong_generalization_set.userId.unique()),int(len(strong_generalization_set.userId.unique())/2))
    strong_generalization_set_val =strong_generalization_set[strong_generalization_set.userId.isin(strong_gen_val_user_set)]
    test = strong_generalization_set[~strong_generalization_set.userId.isin(strong_gen_val_user_set)]
    data = data[data.userId.isin(test.userId.unique())]
    print(f"Number of users = {len(data.userId.unique())=}")

    prompt = "Task: You will now help me generate a highly detailed summary based on the broad common elements of movies.\n"
    prompt += "Do not comment on the year of production. Do not mention any specific movie titles.\n"
    prompt += 'Do not comment on the ratings but use qualitative speech such as the user likes, or the user does not enjoy\n'
    prompt += 'Remember you are an expert crafter of these summaries so any other expert should be able to craft a similar summary to yours given this task\n'
    prompt += f"Keep the summary short at about {args.len} words. The summary should have the following format:\n"
    prompt += "Summary: {Specific details about genres the user enjoys}. {Specific details of plot points the user seems to enjoy}. \
        {Specific details about genres the user does not enjoy}. {Specific details of plot points the user does not enjoy but other users may}."
    # prompt += " The following is an example:\n"
    # prompt += demo_str  # add demonstration

    prompts = generate_prompts_recent(data, 1,args.num_movies)

    print(f"{len(prompts)=}")
        
    existing_data = load_existing_data(f'../saved_user_summary/ml-100k/user_summary_gpt4_new{"debug" if args.debug else ""}.json')

    #make existing data keys back into ints 

    #if existing data is empty dont do anything else fill it uo 
    if existing_data:
        existing_data = {int(k):v for k,v in existing_data.items()}

    prompts_to_process = {}

    for i in range(args.num_samples):
        if i in existing_data:
            cur_prompts = existing_data[i]
        else:
            cur_prompts = {}
        prompts_to_process[i] = {idx: user_prompt for idx, user_prompt in prompts.items() if float(idx) not in cur_prompts}
    
    return_dict = existing_data
    for m in range(args.num_samples): 
        print('='*50)
        print("Running sample ", m+1)
        print('='*50)
        for i, (idx, user_prompt) in (pbar := tqdm(enumerate(prompts_to_process[m].items()))):
            msg = [
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
            while True:
                try:
                    return_dict[m][float(idx)] = openai.ChatCompletion.create(
                        model=args.gpt_version,
                        messages=msg,
                        max_tokens=args.max_tokens
                    )['choices'][0]['message']['content']
                    # If the request is successful, break out of the loop
                    break
                except Exception as e:
                    print(f"{e=}")
                    pbar.set_description(f"An error occurred: {e} ,Retrying in 30 seconds...")
                    time.sleep(10*3)
            if args.debug and i == 5:
                break
            # Save updated dict as a JSON file
            pbar.set_description(f"Saving user {idx} summary...")
            with open(f'../saved_user_summary/ml-100k/user_summary_gpt4_new{"debug" if args.debug else ""}_{args.num_movies}.json', 'w') as fp:
                json.dump(return_dict, fp, indent=4)

