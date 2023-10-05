import json
from tqdm import tqdm
import openai
import pandas as pd
from data.dataloader import get_dataloader
from helper.dataloader import map_title_to_id, map_id_to_title
import re
from torch.utils.data import DataLoader, Subset
import torch
import os
from dotenv import load_dotenv
from model.MF import MatrixFactorizationLLM
from model.decoderMLP import decoderMLP
import argparse
from helper.eval_metrics import precision_at_k, recall_at_k, mapk, ndcg_k
from helper.dataloader import load_dataset, map_title_to_id, convert_titles_to_ids, \
    create_train_matrix_and_actual_lists
from .training_utils import *
import wandb
from tqdm import tqdm

def parse_args():  # Parse command line arguments
    parser = argparse.ArgumentParser(description="LLM4RecSys")
    parser.add_argument("--data_name", default='ml-100k', type=str)
    parser.add_argument("--log_file", default= 'model_logs/ml-100k/logging_llmMF.csv', type=str)
    parser.add_argument("--model_name", default='MFLLM', type=str)
    parser.add_argument("--embedding_dim" , default=1536, type=int)
    parser.add_argument("--output_emb" , default=64, type=int)
    parser.add_argument("--top_for_rerank" , default=50, type=int)
    parser.add_argument("--num_layers" , default=4, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--topk", default=20, type=int)
    parser.add_argument("--train", default=True, type=bool)

    
    parser.add_argument("--num_neg", default=1, type=int)
    parser.add_argument("--lr", default=.001, type=float)
    parser.add_argument("--wd", default=0, type=float)
    args = parser.parse_args()
    args.recon = False
    args.device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu') )
    d = vars(args)
    return args



def get_open_ai_embeddings(num_users_retrieved, user_summaries,model="text-embedding-ada-002"):
     
    user_summary_subset = user_summaries[num_users_retrieved:]
    summarize_summaries_prompt_list = summarize_summaries_prompt(user_summary_subset)
    text = [x.replace("\n", " ") for x in summarize_summaries_prompt_list ]
    return openai.Embedding.create(input = text, model=model)['data']


def json_to_list(data):
    summaries = []
    for key, genre_summaries in data.items():
            summaries.append(genre_summaries)
    return summaries

def summarize_summaries_prompt(user_summaries): 
    role_prompt = 'The following are the summaries of movies watched by the user, in the format genre: summary. \n'
    prompts = []
    for user in user_summaries:
        summary_prompt = ""
        for genre, summary in user.items():
            summary_prompt += f"{genre}: {summary}\n"
        prompts += [role_prompt + summary_prompt]
        
    return prompts

def get_summary_embeddings():
    embeddings_path = "saved_summary_embeddings/ml-100k/embeddings.json"
    user_summaries = 'saved_user_summary/ml-100k/user_summary_gpt3.5_in1_title0_full.json'
    
    with open(user_summaries, "r") as f:
            user_summaries = json.load(f)
            user_summaries = json_to_list(user_summaries)
            total_users = len(user_summaries)
            
    if os.path.exists(embeddings_path):
        with open(embeddings_path, "r") as f:
            num_users_retrieved = len(json.load(f))
    else: 
        num_users_retrieved = 0
    if num_users_retrieved == total_users:
        print('Loading saved embeddings')
        return json.load(open(embeddings_path, "r"))
    else:
        new_embeddings = get_open_ai_embeddings(num_users_retrieved, user_summaries)
        if num_users_retrieved < total_users:
            with open(embeddings_path, "a") as f:
                json.dump(new_embeddings, f, indent=4)
        elif 0==num_users_retrieved:
            with open(embeddings_path, "w") as f:
                json.dump(new_embeddings, f, indent=4)
        print('Wrote Embeddings')
        return new_embeddings



# def train_decoder(model):
    
def train_model(args):
    t_start = time.time()
    experiment_name = f"{args.model_name}_{time.strftime('%Y-%m-%d %H:%M:%S')}"
    wandb.init(project='llm4rec', name=experiment_name)
    wandb.config.update(args)
    wandb.watch_called = False  # To avoid re-watching the model

    # 1. Data Loading & Preprocessing
    train_data = load_dataset("./data_preprocessed/ml-100k/data_split/train_set_leave_one.json")
    valid_data = load_dataset("./data_preprocessed/ml-100k/data_split/valid_set_leave_one.json")
    test_data = load_dataset("./data_preprocessed/ml-100k/data_split/test_set_leave_one.json")
    movie_title_to_id = map_title_to_id("./data/ml-100k/movies.dat")

    train_data = convert_titles_to_ids(train_data, movie_title_to_id)
    valid_data = convert_titles_to_ids(valid_data, movie_title_to_id)
    test_data = convert_titles_to_ids(test_data, movie_title_to_id)

    train_matrix, actual_list_val, actual_list_test = create_train_matrix_and_actual_lists(train_data, valid_data,
                                                                                           test_data, movie_title_to_id)
    train_matrix = csr_matrix(train_matrix)  # Convert train_matrix to a CSR matrix
    print("train_matrix:", train_matrix.shape)

    # Negative item pre-sampling (assuming you have a method for this)
    user_neg_items = neg_item_pre_sampling(train_matrix)
    pre_samples = {'user_neg_items': user_neg_items}
    print("Pre sampling time:{}".format(time.time() - t_start))

    # 2. Model Creation
    num_users, num_items = train_matrix.shape
    
    user_embedder = decoderMLP(args.embedding_dim, args.num_layers ,args.output_emb) 
    
    model = MatrixFactorizationLLM(num_users, user_embedder,num_items, args).to(args.device)
    print(f"{model.parameters()=}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Create negative sampler
    neg_sampler = NegSampler(train_matrix, pre_samples, batch_size=args.batch_size, num_neg=args.num_neg)
    num_batches = train_matrix.count_nonzero() // args.batch_size

    # Early stopping parameters
    best_recall = 0.0
    patience = 10  # Number of epochs to wait for recall to improve
    counter = 0
    model_save_path = f'./saved_model/ml-100k/{args.model_name}'
    user_embeddings = get_summary_embeddings()

    user_embeddings= torch.tensor([user_embeddings[i]['embedding'] for i in range(len(user_embeddings))])

    recall_val = 0


    # 3. Training Loop
    if args.train:
        model.train()
        for epoch in (pbar := tqdm(range(args.epochs))):
            loss = 0.0
            for batch in range(num_batches):
                batch_user_id, batch_item_id, neg_samples = neg_sampler.next_batch()
                # user_inp = 
                
                user_id, pos, neg = batch_user_id, batch_item_id, np.squeeze(neg_samples)
                user = user_embeddings[user_id].to(args.device)

                user_emb, pos_emb, neg_emb = model(user, pos, neg)

                batch_loss = model.bpr_loss(user_emb, pos_emb, neg_emb)
                batch_loss = torch.mean(batch_loss)

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                loss += batch_loss.item()
                pbar.set_description(f"Epoch {batch + 1}/{num_batches}, Loss: {loss / (batch+1)}")
                
            
                # print(f"Epoch {_ + 1}/{num_batches}, Loss: {loss /( _ + 1)}")
            wandb.log({'Loss': loss / num_batches})


            print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {loss / num_batches}")

            # Check Recall@20 on validation
            model.eval()
            pred_list_val = generate_pred_list(model, train_matrix,args,user_embeddings, topk=20)
            recall_val = recall_at_k(actual_list_val, pred_list_val, 20)
            wandb.log({'Validation Recall@20': recall_val})

            # print(f"Validation Recall@20: {recall_val}")

            # Early stopping check
            if recall_val > best_recall:
                best_recall = recall_val
                counter = 0
                # Save the model
                torch.save(model.state_dict(), f"{model_save_path}_best_model.pth")
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered.")
                    
            
        # 4. Evaluation
        # Load best model for evaluation
        model.load_state_dict(torch.load(f"{model_save_path}_best_model.pth"))
        model.eval()
    else:
        model.load_state_dict(torch.load(f"{model_save_path}_best_model.pth"))
        model.eval()


    # Test set results
    pred_list_test = generate_pred_list(model, train_matrix,args,user_embeddings, topk=args.topk)
    actual_list_test = actual_list_test  # Adjust based on your data format
    precision_test = precision_at_k(actual_list_test, pred_list_test, args.topk)
    recall_test = recall_at_k(actual_list_test, pred_list_test, args.topk)
    ndcg_test = ndcg_k(actual_list_test, pred_list_test, args.topk)
    print(f"Test Precision@{args.topk}: {precision_test}")
    print(f"Test Recall@{args.topk}: {recall_test}")
    print(f"Test NDCG@{args.topk}: {ndcg_test}")
    
    log_data = {
        'model_name_date_time_ran': f"{args.model_name}_{time.strftime('%Y-%m-%d %H:%M:%S')}",
        'args.topk': args.topk,
        'precision_test': precision_test,
        'recall_test': recall_test,
        'recall_val': recall_val,
        'args.num_layers': args.num_layers,
        'args.lr': args.lr
    }
    log_results_csv( args.log_file,log_data)
       

    # Save pretrained embeddings
    torch.save(model.user_embeddings.state_dict(), f"{model_save_path}_user_embeddings.pth")
    torch.save(model.item_embeddings.weight.data, f"{model_save_path}_item_embeddings.pth")

    rankings_matrix = generate_rankings_for_all_users(model, num_users=num_users, num_items=num_items,user_emb=user_embeddings,args=args)
    np.save(f"{model_save_path}_rankings_matrix.npy", rankings_matrix)
    print("rankings_matrix:", rankings_matrix)

    movie_id_to_genres = convert_ids_to_genres("./data/ml-100k/movies.dat")
    generate_and_save_rankings_json(rankings_matrix, args.topk, args.top_for_rerank, movie_id_to_genres,
                                    f"{model_save_path}_user_genre_rankings.json",
                                    "./data_preprocessed/ml-100k/user_genre.json")
    wandb.finish()

    return model, rankings_matrix 


if __name__ == "__main__":
    load_dotenv(".env")
    args = parse_args()
    openai.api_key = os.getenv("OPEN-AI-SECRET")
    # embeddings = get_summary_embeddings()
    loss = MatrixFactorizationLLM.bpr_loss

    

    train_model(args)
    

    
    