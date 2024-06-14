# %%
import os
import sys 
sys.path.append('../')
import json
import time

from transformers import pipeline
import openai
from tqdm import tqdm
from dotenv import load_dotenv
import time
import json
from argparse import ArgumentParser
import os
import pandas as pd 
from dotenv import load_dotenv
import random 

global_path = '/home/mila/e/emiliano.penaloza/LLM4REC'


# Load environment variables from the .env file into the script's environment
load_dotenv()
key = os.getenv("OPEN-AI-SECRET")
parser = ArgumentParser(description="Your script description here")
parser.add_argument("--data_name", default = 'ml-1m', help="Name of the dataset to use")
parser.add_argument("--gpt_version", default="gpt-4-1106-preview", help="GPT model version")
parser.add_argument("--max_tokens", type=int, default=300, help="Maximum number of tokens for the response")
parser.add_argument("--num_samples", type=int, default=1, help="Maximum number of tokens for the response")
parser.add_argument("--debug", action='store_true', help="Whether to run in debug mode")
parser.add_argument("--num_movies", type=int, default=50, help="Maximum number of tokens for the response")
parser.add_argument("--num_thresh", type=int, default=50, help="Maximum number of tokens for the response")
parser.add_argument("--len", type=int, default=400, help="Maximum number of tokens for the response")

args = parser.parse_args()
train_data = f'./data_preprocessed/{args.data_name}/prompt_set_timestamped.csv'
test_data = f'./data_preprocessed/{args.data_name}/strong_generalization_set_.csv'



openai.api_key = key 



def load_existing_data(file_path):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r') as fp:
            existing_data = json.load(fp)
        return existing_data

    return {}


def generate_prompts_recent(train_data,num_movies = 30): 
    prompts = {}
    for id in train_data.userId.unique():        

        user_data = train_data[train_data['userId'] == id]

        # Sort the user data by the 'timestamp' column in descending order
        user_data = user_data.sort_values(by='timestamp', ascending=False)


        # Take the top 30 most recent movies
        user = user_data.head(num_movies)
        movie_titles = user['title'].tolist()
        ratings = user['rating'].tolist()
        genres = user['genres'].tolist()
        # assert len(movie_titles) == num_movies

        prompt = ""

        for (title,rating),genres in zip(zip(movie_titles,ratings),genres):


            prompt += f"\n{title}"

            # else:
            #     prompt += f"\n{title}: {description}"
                
            prompt += f'\nRating: {rating}\n'
            prompt += f'\Genres: {genres}\n'
        prompts[id] = prompt    
    return prompts
    

random.seed(2024)

strong_generalization_set = pd.read_csv(test_data)
print(f"{strong_generalization_set=}")

prompt_set = pd.read_csv(f'./data_preprocessed/{args.data_name}/prompt_set_timestamped.csv')
strong_gen_val_user_set = random.sample(list(strong_generalization_set.userId.unique()),int(len(strong_generalization_set.userId.unique())/2))
strong_generalization_set_val =strong_generalization_set[strong_generalization_set.userId.isin(strong_gen_val_user_set)]
test_users = strong_generalization_set[~strong_generalization_set.userId.isin(strong_gen_val_user_set)].userId.unique()
data = prompt_set[prompt_set.userId.isin(test_users)]
print(f"{data=}")

#make sure the test has more than the num_movies 
# grouped_test = test.groupby('userId').count()
# user_set = grouped_test[grouped_test['movieId'] >= args.num_movies].index







prompt = "Task: You will now help me generate a highly detailed summary based on the broad common elements of movies.\n"
prompt += "Do not comment on the year of production. Do not mention any specific movie titles.\n"
prompt += 'Do not comment on the ratings but use qualitative speech such as the user likes, or the user does not enjoy\n'
prompt += 'Remember you are an expert crafter of these summaries so any other expert should be able to craft a similar summary to yours given this task\n'
prompt += f"Keep the summary short at about {args.len} words. The summary should have the following format:\n"
prompt += "Summary: {Specific details about genres the user enjoys}. {Specific details of plot points the user seems to enjoy}. \
    {Specific details about genres the user does not enjoy}. {Specific details of plot points the user does not enjoy but other users may}."


prompts = generate_prompts_recent(data, args.num_thresh)
print(f"{len(prompts)=}")
    
existing_data = load_existing_data(f'./saved_user_summary/{args.data_name}/user_summary_gpt4_{args.num_thresh}_{args.num_movies}_new{"debug" if args.debug else ""}.json')

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


print('='*50)
# print("Running sample ", m+1)
print('='*50)
for i, (idx, user_prompt) in (pbar := tqdm(enumerate(prompts_to_process[0].items()))):
    if idx in return_dict:
        continue
    print(f"{i=}")
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
            return_dict[int(idx)] = openai.ChatCompletion.create(
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
    with open(f'./saved_user_summary/{args.data_name}/user_summary_gpt4_sum_length_{args.len}.json', 'w') as fp:
        #make keys int 
        return_dict = {int(k):v for k,v in return_dict.items()}
        json.dump(return_dict, fp, indent=4)