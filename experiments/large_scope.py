import pickle 
import json
import sys 
sys.path.append('./')
from helper.dataloader import  map_id_to_title
from collections import defaultdict
from model.MF import sentenceT5Classification
import torch 
from peft import LoraConfig, TaskType,get_peft_model
import pandas as pd 
from trainer.transformer_utilts import *
from transformers import T5Tokenizer
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Process some integers.")

# Add the arguments
parser.add_argument('--num_movies', type=int, default=3522, help='Number of movies')
parser.add_argument('--rank', type=int, default=0, help='Rank of the process')
parser.add_argument('--world_size', type=int, default=1, help='Size of the process group')

# Parse the arguments
args = parser.parse_args()

# Access the arguments
num_movies = args.num_movies
rank = args.rank
world_size = args.world_size

with open('./data_preprocessed/ml-1m/show2id.pkl','rb') as f:
    movie_id_map = pickle.load(f)
    #reverse the map
    movie_id_map = {v:k for k,v in movie_id_map.items()}


movie_id_to_title = map_id_to_title('./data/ml-1m/movies.dat')

#pickle movie_id_map if it doesnt exist to saved_iser_summary 
with open ('data_preprocessed/ml-1m/profile2id.pkl','rb') as f:
    profile2id = pickle.load(f)


with open('./saved_user_summary/ml-1m/user_summary_gpt4_.json','r') as f:
    base_prompts = json.load(f) 
    #make int
base_prompts = {profile2id[int(float(k))]:v for k,v in base_prompts.items() }

with open('./saved_user_summary/ml-1m/user_summary_gpt4_new.json','r') as f:
    new_prompts = json.load(f)

    new_prompts = {int(float(k)):v for k,v in new_prompts.items() }
    
for k,v in new_prompts.items():
    new_prompts[k] = {profile2id[int(float(k1))]:v1 for k1,v1 in v.items()}
# first_promtp = {profile2id[float(k)]:v for k,v in new_prompts[2].items()}

#take the subset of the base prompt that has the same keys as first prompt
base_prompts = {k:v for k,v in base_prompts.items() if k in new_prompts[0].keys()}

new_prompts[4] = base_prompts





model = sentenceT5Classification.from_pretrained('t5-large', num_labels=num_movies)

lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=32, lora_alpha=16, lora_dropout=0,
                            target_modules=["q", "v"],
                            modules_to_save=['classification_head'])
model = get_peft_model(model, lora_config)

path = f'./model/weights/TEARS_FineTuned.pt'
model.load_state_dict(torch.load(path ,map_location=torch.device('cuda')))
model.to(0)


metrics = defaultdict(list)
ndcgs_50 = []
prompt_set_df = pd.read_csv(f'./data_preprocessed/ml-1m/prompt_set_timestamped.csv')

movie_set = set([x for x in movie_id_map.keys()])
user_id_set = set()
user_ids_l = []
ss = 0
lengths = []
u_ids_text = {}
torch.manual_seed(2024)
num_items ={ }
metrics_std={}

# Initialize a dictionary to store the sum of each metric for each m value
metrics_sum = defaultdict(list)

# Initialize a dictionary to store the average of each metric for each m value
metrics_avg = {}
tokenizer = T5Tokenizer.from_pretrained('t5-large')

prompts,rec_dataloader,num_movies,val_dataloader,test_dataloader,val_data_tr,test_data_tr= load_data(args,tokenizer,rank,world_size)



for m in range(len(new_prompts)):

    cur_prompts = new_prompts[m]
    encodings = {k: tokenizer([v],padding='max_length', return_tensors='pt',truncation=True,max_length=262) for k, v in sorted(cur_prompts.items())} 
    encodings = {k: {k1: v1.squeeze(0) for k1, v1 in v.items()} for k, v in encodings.items()}


    with torch.no_grad():
        model.eval()
        for b,item in enumerate(test_dataloader):
            user_ids = sum(item.pop('idx').cpu().tolist(),[])
            user_id_set.update( user_ids)
            #move to device
            item = {k:v.to(0) for k,v in item.items()}
            # labels_pos = torch.where(item['labels'][i] == 1)[0]
            movie_emb_clean = model(**item)[0]
            # movie_emb_clean[:,labels_pos] = -torch.inf
            #copy the labels 
            user_ids_l.append(user_ids)
            test_rows = test_data_tr[user_ids].toarray()
            #make it zero where the user has seen the movie
            movie_emb_clean[np.where(test_rows > 0)] = -torch.inf
            

            labels =item['labels'].cpu().numpy()

            recon = movie_emb_clean.cpu().numpy()
            metrics['ndcgs_10'].append(NDCG_binary_at_k_batch(recon,labels,k=10).mean().tolist())
            metrics['ndcgs_20'].append(NDCG_binary_at_k_batch(recon,labels,k=20).mean().tolist())
            metrics['ndcgs_50'].append(NDCG_binary_at_k_batch(recon,labels,k=50).mean().tolist())

 
            metrics['mrr@10'].append(MRR_at_k(recon,labels,k=10))
            metrics['mrr@20'].append(MRR_at_k(recon,labels,k=20))
            metrics['mrr@50'].append(MRR_at_k(recon,labels,k=50))

            metrics['rn_10_n'].append(Recall_at_k_batch(recon,labels,k=10))
            metrics['rn_20_n'].append(Recall_at_k_batch(recon,labels,k=20))
            metrics['rn_50_n'].append(Recall_at_k_batch(recon,labels,k=50))
            
            for key in metrics.keys():
                metrics_sum[key].append(np.mean(metrics[key]))

for key in metrics_sum.keys():

    metrics_avg[key] = np.mean(metrics_sum[key]) 
    metrics_std[key] = np.std(metrics[key])
    print(f'{key} : {metrics_avg[key]} +- {metrics_std[key]}')


    

