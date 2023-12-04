import os
import sys 
sys.path.append('../')
import json
import time
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import pipeline
import torch
import re
import openai
# from tenacity import retry, wait_exponential, stop_after_attempt
from tqdm import tqdm
from dotenv import load_dotenv
from helper.in_context_learning_batch import in_context_user_summary, in_context_retrieval
from helper.builder import build_context, build_movie_candidates
import torch
import time
import json
from torch.utils.data import Dataset, DataLoader
from data.dataloader import get_dataloader
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
parser.add_argument("--train_data", default="../data_preprocessed/ml-100k/train_leave_one_out_timestamped.csv", help="Path to the training data CSV file")
parser.add_argument("--gpt_version", default="gpt-4-1106-preview", help="GPT model version")
parser.add_argument("--max_tokens", type=int, default=300, help="Maximum number of tokens for the response")
parser.add_argument("--debug", action='store_true', help="Whether to run in debug mode")
args = parser.parse_args()

 
openai.api_key = key


def generate_prompts_recent(train_data,only_title=1,num_movies = 30): 
    prompts = {}
    for id in train_data.userId.unique():        

        user_data = train_data[train_data['userId'] == id]

        # Sort the user data by the 'timestamp' column in descending order
        user_data = user_data.sort_values(by='timestamp', ascending=False)

        # Take the top 30 most recent movies
        user = user_data.head(num_movies)
        
        movie_titles = user['title'].tolist()

        movie_summaries= user['summary'].tolist()
        
        ratings = user['rating'].tolist()
        genres = user['genres'].tolist()

        prompt = ""

        for ((description,title),rating),genres in zip(zip(zip(movie_summaries,movie_titles),ratings),genres):
            if only_title == 1:

                prompt += f"\n{title}"

            else:
                prompt += f"\n{title}: {description}"
                
            prompt += f'\nRating: {rating}\n'
            prompt += f'\Genres: {rating}\n'
        prompts[id] = prompt    
    return prompts


        


if __name__ == "__main__":
    data = pd.read_csv(args.train_data)
    

    prompt = f"Task: You will now help me generate a simply worded highly detailed summary of the following genre summaries\n"
    prompt += "Instructions: Always start with 'Summary:'. Please be constrain the length within 200 tokens for the summary by being concise and not including irrelevant details.\n"
    prompt += "The format of the summary should be:\n"
    prompt += "Summary: {Specific details about genres the user enjoys}. {Specific details of plot points the user seemns to enjoy}.  {Specific details about genres the user does not enjoy}. {Specific details of plot points the user does not enjoy}."
    prompt += "The following is an example of a generated summary"
    prompt += """Summary: The user enjoys action, crime, thriller, adventure, and comedy films, with preference for high-stakes confrontations, complex schemes, suspense, humorous moments, and films set in advanced technology scenarios or future settings. However, they do not like war films, historical dramas, movies with slow-paced narratives, heavy dialogue, films set in surrealistic settings or involve anthropomorphic characters, and films that heavily rely on abstract concepts and complexity. Films exploring intense dramatic scenes or serious existential themes are also less appealing."""    
    
    prompt = f"Task: You will now help me generate a highly detailed summary based on the broad common elements of movies.\n"
    prompt += f"Do not comment on the year of production. Do not mention any specific movie titles.\n"
    prompt += 'Do not comment on the ratings but use qualitative speech such as the user likes, or the user does not enjoy\n'
    prompt += 'Remember you are an expert crafter of these summaries so any other exper should be able to craft a similar summary to yours given this task\n'
    prompt += "Keep the summary short at about 200 words. The summary should have the following format:\n"
    prompt += "Summary: {Specific details about genres the user enjoys}. {Specific details of plot points the user seemns to enjoy}.  {Specific details about genres the user does not enjoy}. {Specific details of plot points the user does not enjoy but other users may}."
    # prompt += " The following is an example:\n"
    # prompt += demo_str  # add demonstration

    prompts = generate_prompts_recent(data, 1,30)
    

    return_dict = {}
    for i,(idx,user_prompt) in enumerate(prompts.items()):
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
                        model=args.gpt_version,
                        messages=msg,
                        max_tokens=args.max_tokens
                    )['choices'][0]['message']['content']

                    # If the request is successful, break out of the loop
                    break
                except Exception as e:
                    print(f"An error occurred: {e}")
                    print("Retrying in 30 seconds...")
                    time.sleep(30)
            if args.debug and i == 5:
                    break 

    #save out dict as a json
    with open(f'../saved_user_summary/ml-100k/user_summary_gpt4_{"debug" if args.debug else ""}.json', 'w') as fp:

        print(f"{return_dict=}")
        json.dump(return_dict, fp, indent=4)
