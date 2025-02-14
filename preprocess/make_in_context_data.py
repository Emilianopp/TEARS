import os
import sys 
sys.path.append('../')
import json
import time
import openai
# from tenacity import retry, wait_exponential, stop_after_attempt
from tqdm import tqdm
from dotenv import load_dotenv
import torch
import time
import json
from argparse import ArgumentParser
import os
import pandas as pd 
import debugpy



# Load environment variables from the .env file into the script's environment
load_dotenv()

parser = ArgumentParser(description="Your script description here")
parser.add_argument("--model", default="gpt-4-1106-preview", help="GPT model version")

parser.add_argument("--max_tokens", type=int, default=300, help="Maximum number of tokens for the response")
parser.add_argument("--debug", action='store_true', help="Whether to run in debug mode")
parser.add_argument("--data_name", default = 'ml-1m', help="Name of the dataset to use")
parser.add_argument("--save_path", default = '', help="Name of the dataset to use")
parser.add_argument("--books", action='store_true', help="Whether to run in books prompt")

args = parser.parse_args()
train_data = f'./data_preprocessed/{args.data_name}/prompt_set_timestamped.csv'
key = os.getenv("OPEN-AI-SECRET") if 'gpt' in args.model else os.getenv("LLAMA-KEY")
    
openai.api_key = key 
if 'gpt' not in  args.model : 
 openai.api_base = "https://api.llama-api.com" 






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
        genres = user['genres'].tolist() if args.data_name == 'netflix' else user['genre'].tolist()


        prompt = ""

        for (title,rating),genres in zip(zip(movie_titles,ratings),genres):
         

            prompt += f"\n{title}"
                
            prompt += f'\nRating: {rating}\n'
            prompt += f'\Genres: {genres}\n'
        prompts[id] = prompt  
        if 'gpt' not in args.model: 
            prompt += 'Do not comment on the ratings or specific titles but use qualitative speech such as the user likes, or the user does not enjoy\n'
            prompt += 'Do not comment mention any actor/director names\n'

        else: 
            prompt += 'Never make statments like such as  "Films like" or the "user has high ratings for"\n'
            prompt += 'Make no comments on the not the ratings or specific titles but use qualitative speech such as the user likes, or the user does not enjoy\n'
            prompt += 'Do not comment mention any actor/director names or any movie names in the summary.\n'
  
    return prompts



def generate_prompts_recent_books(train_data,num_movies = 30): 
    prompts = {}
    for id in train_data.userId.unique():        

        user_data = train_data[train_data['userId'] == id]

        # Take the top 30 most recent movies
        user = user_data.head(num_movies)
        
        movie_titles = user['title'].tolist()


        
        ratings = user['rating'].tolist()
        authors = user['authors'].tolist()

        prompt = ""

        for (title,rating),author in zip(zip(movie_titles,ratings),authors):
         

            prompt += f"Book : \n{title}"
                
            prompt += f'\nRating: {rating}\n'
            prompt += f'\Authors: {author}\n'
            
        prompt += 'Do not comment on the ratings or specific titles but use qualitative speech such as the user likes, or the user does not enjoy\n'
        prompt += 'Do not comment mention any author names\n'

        prompts[id] = prompt    
   
    return prompts


if __name__ == "__main__":
    data = pd.read_csv(train_data)
    if not args.books :

        prompt = "Task: You will now help me generate a highly detailed summary based on the broad common elements of movies.\n"
        prompt += "Do not comment on the year of production. Do not mention any specific movie titles or actors.\n"
        prompt += 'Do not comment on the ratings but use qualitative speech such as the user likes, or the user does not enjoy\n'
        prompt += 'Remember you are an expert crafter of these summaries so any other expert should be able to craft a similar summary to yours given this task\n'
        prompt += "Keep the summary short at about 200 words. The summary should have the following format:\n"
        prompt += "Summary: {Specific details about genres the user enjoys}. {Specific details of plot points the user seems to enjoy}. \
            {Specific details about genres the user does not enjoy}. {Specific details of plot points the user does not enjoy but other users may}."

    else:
        prompt = "Task: You will now help me generate a highly detailed summary based on the broad common elements of books.\n"
        prompt += "Do not comment on the year of release. Do not mention any specific book titles or authors.\n"
        prompt += 'Do not comment on the ratings but use qualitative speech such as the user likes, or the user does not enjoy\n'
        prompt += 'Remember you are an expert crafter of these summaries so any other expert should be able to craft a similar summary to yours given this task\n'
        prompt += "Keep the summary short at about 200 words. The summary should have the following format:\n"
        prompt += "Summary: {Specific details about genres the user enjoys}. {Specific details of plot points the user seems to enjoy}. \
            {Specific details about genres the user does not enjoy}. {Specific details of plot points the user does not enjoy but other users may}"
        
        
    # prompt += " The following is an example:\n"
    # prompt += demo_str  # add demonstration

    prompts = generate_prompts_recent(data, 50) if not args.books else generate_prompts_recent_books(data, 50)
        
    existing_data = load_existing_data(f'./saved_user_summary/{args.data_name}{args.save_path}/user_summary_{args.model.replace("/","_")}_{"debug" if args.debug else ""}.json')

    #make existing data keys back into ints 
    existing_data = {float(k):v for k,v in existing_data.items()}

    prompts_to_process = {idx: user_prompt for idx, user_prompt in prompts.items() if float(idx) not in existing_data}
   
    return_dict = existing_data
    for i, (idx, user_prompt) in (pbar := tqdm(enumerate(prompts_to_process.items()))):

        if 'gpt' in args.model or 'llama' in args.model:
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

                    return_dict[float(idx)] = openai.ChatCompletion.create(
                        model=args.model,
                        messages=msg,
                        max_tokens=args.max_tokens,
                        seed = 0,
                        temperature = 0 if 'gpt' in args.model else 0,
                    )['choices'][0]['message']['content']
                    


                    # If the request is successful, break out of the loop
                    break
                except Exception as e:
                    pbar.set_description(f"An error occurred: {e} ,Retrying in 30 seconds...")
                    time.sleep(60*3)

            if args.debug and i == 5:
                break                        


        # Save updated dict as a JSON file
        pbar.set_description(f"Saving user {idx} summary...")
        os.makedirs(f'./saved_user_summary/{args.data_name}{args.save_path}', exist_ok=True)
        with open(f'./saved_user_summary/{args.data_name}{args.save_path}/user_summary_{args.model.replace("/","_")}_{"debug" if args.debug else ""}.json', 'w+') as fp:

            json.dump(return_dict, fp, indent=4)
