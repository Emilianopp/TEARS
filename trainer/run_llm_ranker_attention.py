import pickle
from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm
import openai
import pandas as pd
from data.dataloader import get_dataloader
from helper.dataloader import load_pickle, map_title_to_id, map_id_to_title
import re
from torch.utils.data import DataLoader, Subset
import torch
import os
from dotenv import load_dotenv
from model.MF import MatrixFactorizationLLM
from model.decoderMLP import decoderMLP, decoderAttention
import argparse
from torch.optim.lr_scheduler import LambdaLR
from helper.eval_metrics import precision_at_k, recall_at_k, mapk, ndcg_k
from helper.dataloader import load_dataset, map_title_to_id, convert_titles_to_ids, \
    create_train_matrix_and_actual_lists
from .training_utils import *
import wandb
from tqdm import tqdm
import math

def get_user_genre_embeddings(args):
    if args.make_embeddings:   
        with open('saved_user_summary/ml-100k/user_summary_gpt3.5_in1_title0_full.json','r') as f: 
            user_summaries = json.load(f)
            user_summaries = {int(k):v for k,v in user_summaries.items()}
        user_embeddings = {}
        print ('getting embeddings')
        model = get_t5() if args.embedding_module == 't5' else None
        for user, summaries in tqdm(user_summaries.items()):
            user_embeddings[user] = get_genrewise_embeddings(summaries,args,model)
        print ('got embeddings')
        torch.save(user_embeddings, f"saved_summary_embeddings/ml-100k/user_embeddings_{args.model_name}.pt")
    else:
        user_embeddings = torch.load(f"saved_summary_embeddings/ml-100k/user_embeddings_{args.model_name}.pt")
    return user_embeddings
        
            



# def train_decoder(model):
    
def train_model(args):
    t_start = time.time()
    experiment_name = f"{args.model_name}_{time.strftime('%Y-%m-%d %H:%M:%S')}"
    if not args.debug:
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
    # user_embeddings = torch.tensor(get_summary_embeddings(args))
    user_embeddings  = get_user_genre_embeddings(args)

    print('made embeddings')




    
    embedding_dim = user_embeddings[1][list(user_embeddings[1].keys())[0]].shape[0]
    
    user_embedder = decoderAttention(embedding_dim,args.num_heads,args.num_layers ,args.output_emb) 
    
    model = MatrixFactorizationLLM(num_users, user_embedder,num_items, args).to(args.device)

    lr_lambda = lambda step: 0.5 * (1 + math.cos(step / args.total_steps * math.pi))  # Cosine decay

    # Create a LambdaLR scheduler that adjusts the learning rate
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = LambdaLR(optimizer, lr_lambda)

    # Create negative sampler
    neg_sampler = NegSampler(train_matrix, pre_samples, batch_size=args.batch_size, num_neg=args.num_neg)
    num_batches = train_matrix.count_nonzero() // args.batch_size

    # Early stopping parameters
    best_recall = 0.0
    patience = 50  # Number of epochs to wait for recall to improve
    counter = 0


    # user_embeddings= torch.tensor([user_embeddings[i]['embedding'] for i in range(len(user_embeddings))])

    recall_val = 0
    genre_list = get_genres()




    # 3. Training Loop
    if args.train:
        for epoch in (pbar := tqdm(range(args.epochs))):
            model.train()
            loss = 0.0
            for batch in range(num_batches):
                batch_user_id, batch_item_id, neg_samples = neg_sampler.next_batch()
                # user_inp = 
                
                user_id, pos, neg = batch_user_id, batch_item_id, np.squeeze(neg_samples)

                
                #this needs to change to a dict of user embeddings



                all_user_genre_dict = [user_embeddings[u+1] for u in user_id]

                user_list = [ model.user_embeddings.prepare_input(user_genre_dict, genre_list) for user_genre_dict in all_user_genre_dict]
                user_tensor = torch.stack(user_list).to(args.device)

                user_emb, pos_emb, neg_emb = model.forward(user_tensor, pos, neg)

                batch_loss = model.bpr_loss(user_emb, pos_emb, neg_emb)
                batch_loss = torch.mean(batch_loss)

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                loss += batch_loss.item()
                if args.cosine:
                    scheduler.step
                pbar.set_description(f"Epoch {batch + 1}/{num_batches}, Loss: {loss / (batch+1)}")
                
                if args.debug:
                    break
                # print(f"Epoch {_ + 1}/{num_batches}, Loss: {loss /( _ + 1)}")
            if not args.debug:
                wandb.log({'Loss': loss / num_batches})


            print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {loss / num_batches}")

            # Check Recall@20 on validation
            model.eval()
            pred_list_val = generate_pred_list_attention(model, train_matrix,args,user_embeddings, topk=20, print_emb=True)
            recall_val = recall_at_k(actual_list_val, pred_list_val, 20)
            if not args.debug:
                wandb.log({'Validation Recall@20': recall_val}) 
            
            print(f"Validation Recall@20: {recall_val}")

            # print(f"Validation Recall@20: {recall_val}")

            # Early stopping check
            if recall_val > best_recall:
                best_recall = recall_val
                counter = 0
                best_epoch = epoch
                # Save the model

                #write the pred list val to a pickle file 
                with open(f"saved_summary_embeddings/ml-100k/pred_list_val.pkl", "wb+") as f:
                    pickle.dump(pred_list_val, f)
                torch.save(model.state_dict(), f"{args.model_save_name}_best_model.pth")
                torch.save(model.user_embeddings.state_dict(), f"{args.model_save_name}_embedder.pth")
                print(f'wrote model to {args.model_save_name}')
            else:
                counter += 1
                
                if counter >= patience or torch.isnan(batch_loss).any():
                    print("Early stopping triggered.",f'{batch_loss=}')
                    break
                    
            if args.debug:
                break
        # 4. Evaluation
        # Load best model for evaluation

        model.load_state_dict(torch.load(f"{args.model_save_name}_best_model.pth"))
        model.eval()
    else:
        model.load_state_dict(torch.load(f"{args.model_save_name}_best_model.pth"))
        model.eval()


    # Test set results
    pred_list_test = generate_pred_list_attention(model, train_matrix,args,user_embeddings, topk=args.topk)
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
        'args.lr': args.lr,
        'args.num_heads':args.num_heads,
        'args.epochs' : args.epochs
    }
    log_results_csv( args.log_file,log_data)
       

    # Save pretrained embeddings
    torch.save(model.user_embeddings.state_dict(), f"{args.model_save_name}_user_embeddings.pth")
    torch.save(model.item_embeddings.weight.data, f"{args.model_save_name}_item_embeddings.pth")

    

    rankings_matrix = generate_rankings_for_all_users_attention(model, num_users=num_users, num_items=num_items,user_emb=user_embeddings,args=args)
    np.save(f"{args.model_save_name}_rankings_matrix.npy", rankings_matrix)
    print("rankings_matrix:", rankings_matrix)

    movie_id_to_genres = convert_ids_to_genres("./data/ml-100k/movies.dat")
    generate_and_save_rankings_json(rankings_matrix, args.topk, args.top_for_rerank, movie_id_to_genres,
                                    f"{args.model_save_name}_user_genre_rankings.json",
                                    "./data_preprocessed/ml-100k/user_genre.json")
    
    if not args.debug:
        wandb.finish()

    return model, rankings_matrix 


if __name__ == "__main__":
    load_dotenv(".env")
    args = parse_args()
    openai.api_key = os.getenv("OPEN-AI-SECRET")
    
    train_model(args)
    

    
    