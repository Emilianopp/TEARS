
import os
import sys 
sys.path.append('../')
os.chdir('../')
import pickle
import json
import numpy as np


#pickle movie_id_map if it doesnt exist to saved_iser_summary 
with open ('/home/mila/e/emiliano.penaloza/LLM4REC/vae/ml-1m/pro_sg/profile2id.pkl','rb') as f:
    profile2id = pickle.load(f)


with open('/home/mila/e/emiliano.penaloza/LLM4REC/saved_user_summary/ml-1m/user_summary_gpt4_.json','r') as f:
    base_prompts = json.load(f) 
    #make int
base_prompts = {profile2id[int(float(k))]:v for k,v in base_prompts.items() }


with open('/home/mila/e/emiliano.penaloza/LLM4REC/saved_user_summary/ml-1m/user_summary_gpt4_new.json','r') as f:
    new_prompts = json.load(f)

    new_prompts = {int(float(k)):v for k,v in new_prompts.items() }
    
for k,v in new_prompts.items():
    new_prompts[k] = {profile2id[int(float(k1))]:v1 for k1,v1 in v.items()}

base_prompts = {k:v for k,v in base_prompts.items() if k in new_prompts[0].keys()}




# %%
#import bleu score from nltk
from nltk.translate.bleu_score import sentence_bleu
#import edit distance from nltk
from nltk.metrics import edit_distance
from tqdm import tqdm

bleus = np.zeros((len(new_prompts[0]),len(new_prompts),len(new_prompts)))
edit_distances = np.zeros((len(new_prompts[0]),len(new_prompts),len(new_prompts)))
for k,user in tqdm(enumerate(new_prompts[0].keys())):
    for i in new_prompts.keys():
        for j in new_prompts.keys():
            if i == j:
                continue
            if bleus[k,i,j] != 0 or edit_distances[k,i,j] != 0:
                continue
            #calculate the edit distance and Bleu score between the current and prior dicts prompgs 
            bleus[k,i,j] = sentence_bleu(new_prompts[i][user],new_prompts[j][user])

            edit_distances[k,i,j] = edit_distance(new_prompts[i][user].split(),new_prompts[j][user].split())


print(f"{bleus.mean()=}")
print(f"{edit_distances.mean()=}")

#save both arrays 
np.save(f'./bleus_same_sum.npy',bleus)
np.save(f'./edit_same_sum.npy',edit_distances)
  
  