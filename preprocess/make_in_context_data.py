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

global_path = '/home/mila/e/emiliano.penaloza/LLM4REC'

from dotenv import load_dotenv

# Load environment variables from the .env file into the script's environment
load_dotenv()

# Access the environment variables using the os module
import os

# Example: Get the value of the 'DATABASE_URL' environment variable
key = os.getenv("OPEN-AI-SECRET")


 
openai.api_key = key

prompt = f"Task: You will now help me generate a simply worded highly detailed summary of the following genre summaries\n"
prompt += "Instructions: Always start with 'Summary:'. Please be constrain the length within 200 tokens for the summary by being concise and not including irrelevant details.\n"
prompt += "The format of the summary should be:\n"
prompt += "Summary: {Specific details about genres the user enjoys}. {Specific details of plot points the user seemns to enjoy}.  {Specific details about genres the user does not enjoy}. {Specific details of plot points the user does not enjoy}."
prompt += "The following is an example of a generated summary"
prompt += """Summary: The user enjoys action, crime, thriller, adventure, and comedy films, with preference for high-stakes confrontations, complex schemes, suspense, humorous moments, and films set in advanced technology scenarios or future settings. However, they do not like war films, historical dramas, movies with slow-paced narratives, heavy dialogue, films set in surrealistic settings or involve anthropomorphic characters, and films that heavily rely on abstract concepts and complexity. Films exploring intense dramatic scenes or serious existential themes are also less appealing."""
def summarize_summaries(summaries,context):
    out_dict = {}
    for user_id,sums in summaries.items():

        sum_prompt = ""
        for genre,genre_sum in sums.items(): 
            sum_prompt+= f"genre: {genre} \n {genre_sum}\n"
        

        msg = [{
                "role": "system",
                "content": context},
                {"role":"user",
                "content": sum_prompt}
            ]

        out = openai.ChatCompletion.create(model="gpt-4",   
                                        messages=msg,                                    
                                max_tokens=300  # limit the token length of response
                            )['choices'][0]['message']['content']
        out_dict[user_id] = out
    return out_dict
    
d = summarize_summaries(out,prompt)