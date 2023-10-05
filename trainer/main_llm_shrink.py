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
    parser.add_argument("--log_file", default= 'model_logs/ml-100k/logging_recon.csv', type=str)
    parser.add_argument("--model_name", default='BPRMF', type=str)
    parser.add_argument("--embedding_dim" , default=1536, type=int)
    parser.add_argument("--output_emb" , default=64, type=int)
    parser.add_argument("--top_for_rerank" , default=50, type=int)
    parser.add_argument("--num_layers" , default=6, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--train_recon", default=False, type=bool)


    parser.add_argument("--topk", default=20, type=int)

    parser.add_argument("--train", default=True, type=bool)
    
    parser.add_argument("--num_neg", default=1, type=int)
    parser.add_argument("--lr", default=.0001, type=float)
    parser.add_argument("--wd", default=0, type=float)
    

    
    args = parser.parse_args()
    args.recon = True
    args.model_save_path = f'./saved_model/ml-100k/{args.model_name}'
    args.device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu') )

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




def train_recon_model(args):
    experiment_name = f"{args.model_name}_reconstruction_{time.strftime('%Y-%m-%d %H:%M:%S')}"
    
    wandb.init(project='llm4rec', name=experiment_name)
    wandb.config.update(args)
    wandb.watch_called = False  # To avoid re-watching the model
    
    target = torch.load(f"{args.model_save_path}_user_embeddings.pth")
  
    # print(f"{int(len(target)*.9)=}")
    train_target = target[:int(len(target)*.9)]
    val_target = target[int(len(target)*.9):int(len(target)*.95)]
    test_target = target[int(len(target)*.95):]
    patience = 10  # Number of epochs to wait for recall to improve
    
    
    user_embeddings = get_summary_embeddings()

    user_embeddings= torch.tensor([user_embeddings[i]['embedding'] for i in range(len(user_embeddings))])
    
    #make the same split for the user_embeddings
    train_summary_embeddings = user_embeddings[:int(len(user_embeddings)*.9)]
    val_summary_embeddings = user_embeddings[int(len(user_embeddings)*.9):int(len(user_embeddings)*.95)]
    test_summary_embeddings = user_embeddings[int(len(user_embeddings)*.95):]
    
    assert len(train_summary_embeddings) == len(train_target), "Train summary and target lengths do not match: {} != {}".format(len(train_summary_embeddings), len(train_target))
    assert len(val_summary_embeddings) == len(val_target), "Validation summary and target lengths do not match: {} != {}".format(len(val_summary_embeddings), len(val_target))
    assert len(test_summary_embeddings) == len(test_target), "Test summary and target lengths do not match: {} != {}".format(len(test_summary_embeddings), len(test_target))

    
    #make dataloaders with a baths size of 16 
    target_train_dataloader = DataLoader(train_target, batch_size=args.batch_size)
    target_summary_dataloader = DataLoader(train_summary_embeddings, batch_size=args.batch_size)
    num_batches = len(target_train_dataloader)
    model = decoderMLP(args.embedding_dim, args.num_layers ,args.output_emb).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Create negative sampler


    model.train()
    best_loss = 0
    for epoch in (pbar := tqdm(range(args.epochs))):
        loss = 0.0
        batch_user_id_min = 0
        batch_user_id_max = 0
        for i,(summary_batch,target_batch) in enumerate(zip(target_summary_dataloader, target_train_dataloader)):
            batch_user_id_max += args.batch_size
           
           
            user_embeddings_model = model(summary_batch.to(args.device))

            batch_loss = model.mse_loss(user_embeddings_model, target_batch.to(args.device))
            
            # batch_loss = torch.mean(batch_loss)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()
            pbar.set_description(f"Batch {i + 1}/{num_batches}, Loss: {loss / (i+1)}")
            batch_user_id_min = batch_user_id_max
        
            # print(f"Epoch {_ + 1}/{num_batches}, Loss: {loss /( _ + 1)}")
            
         
        wandb.log({'Loss': loss / num_batches})


        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {loss / num_batches}")

        # Check Recall@20 on validation
        model.eval()
        val_output = model(val_summary_embeddings.to(args.device))
        val_loss = model.mse_loss(val_output, val_target.to(args.device))
        wandb.log({'Validation MSE ': val_loss})

        # print(f"Validation Recall@20: {recall_val}")

        # Early stopping check
        if val_loss > best_loss:
            best_loss = val_loss
            counter = 0
            # Save the model
            torch.save(model.state_dict(), f"{args.model_save_path}_recon_best_model.pth")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break 
        
                
        
    # 4. Evaluation
    # Load best model for evaluation
    
    model.load_state_dict(torch.load(f"{args.model_save_path}_recon_best_model.pth"))
    model.eval()

    test_outputs = model(test_summary_embeddings.to(args.device))
    test_loss = model.mse_loss(test_outputs, test_target.to(args.device))
    
    
    print(f"Test MSE loss: {test_loss}")

    
    log_data = {
        'model_name_date_time_ran': f"{args.model_name}_{time.strftime('%Y-%m-%d %H:%M:%S')}",
        'args.topk': args.topk,
        'batch_size': args.batch_size,
        'mse_loss': test_loss.detach().cpu().numpy(),
        'args.num_layers': args.num_layers,
        'args.lr': args.lr
    }
    log_results_csv( args.log_file,log_data)
       

    # Save pretrained embeddings
    torch.save(model.state_dict(), f"{args.model_save_path}_recon_best_model.pth")




    wandb.finish()

    return model 

    

def eval_recon(args):
    user_embedding_model = decoderMLP(args.embedding_dim, args.num_layers ,args.output_emb).to(args.device)

    user_embedding_model.load_state_dict(torch.load(f"{args.model_save_path}_recon_best_model.pth"))
    user_embedding_model.eval()
    
    user_embeddings = get_summary_embeddings()
    user_embeddings= torch.tensor([user_embeddings[i]['embedding'] for i in range(len(user_embeddings))])
    

    test_data = load_dataset("./data_preprocessed/ml-100k/data_split/test_set_leave_one.json")
    train_data = load_dataset("./data_preprocessed/ml-100k/data_split/train_set_leave_one.json")
    valid_data = load_dataset("./data_preprocessed/ml-100k/data_split/valid_set_leave_one.json")
    test_data = load_dataset("./data_preprocessed/ml-100k/data_split/test_set_leave_one.json")
    movie_title_to_id = map_title_to_id("./data/ml-100k/movies.dat")

    train_matrix, actual_list_val, actual_list_test = create_train_matrix_and_actual_lists(train_data, valid_data,
                                                                                           test_data, movie_title_to_id)

    train_matrix = csr_matrix(train_matrix)  # Convert train_matrix to a CSR matrix

    num_users, num_items = train_matrix.shape
    
    model = MatrixFactorization(num_users, num_items, args).to(args.device)
    
    model.load_state_dict(torch.load(f"{args.model_save_path}_best_model.pth"))
    
    pred_list_test = generate_pred_list(model, train_matrix,args,user_embeddings, topk=args.topk,summary_encoder= user_embedding_model)
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
        'args.num_layers': args.num_layers,
        'args.lr': args.lr
    }
    log_results_csv( 'model_logs/ml-100k/logging_MF_w_recon.csv',log_data)
       

    # Save pretrained embeddings







  
    wandb.finish()
    
    
    
    
    
    
    
    


if __name__ == "__main__":
    load_dotenv(".env")
    args = parse_args()
    openai.api_key = os.getenv("OPEN-AI-SECRET")
    # embeddings = get_summary_embeddings()


    
    
    if args.train_recon:
        train_recon_model(args)
    eval_recon(args)
    

    
    