from tqdm import tqdm
import datetime
from trainer.losses.loss import get_loss
import pickle
import torch
import argparse
from typing import List
import torch.nn.functional as F
import sys
import torch.distributed as dist
import numpy as np
import json
import os
from collections import defaultdict
from helper.eval_metrics import Recall_at_k_batch, NDCG_binary_at_k_batch, MRR_at_k
import wandb
from helper.dataloader import DataMatrix, MatrixDataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import sys
sys.path.append('../')
update_count = 0


def load_data(args, tokenizer, rank=0, world_size=1, prompt_p=None):
    data_path = f'./data_preprocessed/{args.data_name}/'
    if prompt_p is None:
        prompt_p = f'./saved_user_summary/{args.data_name}/user_summary_gpt4_.json'

    loader = MatrixDataLoader(data_path, args)

    train_data_tr, train_data = loader.load_data('train')
    valid_data_tr, valid_data = loader.load_data('validation')
    test_data_tr, test_data = loader.load_data('test')
    num_users = train_data.shape[0]
    num_movies = train_data.shape[1]

    # All matrices are shape NxI so we need to subset the non zero indices
    nonzer_indeces_train = {i: v for i, v in enumerate(
        set(train_data.sum(axis=1).nonzero()[0]))}
    nonzer_indeces_valid = {i: v for i, v in enumerate(
        set(valid_data.sum(axis=1).nonzero()[0]))}
    nonzer_indeces_test = {i: v for i, v in enumerate(
        set(test_data.sum(axis=1).nonzero()[0]))}

    with open(f'{data_path}/profile2id.pkl', 'rb') as f:
        profile2id = pickle.load(f)
    with open(prompt_p, 'r') as f:
        prompts = json.load(f)
        prompts = {profile2id[int(float(k))]: v for k, v in prompts.items()}

    promp_list = [v for k, v in prompts.items()]
    max_l = max([len(i.split()) for i in promp_list])
    max_token_l = max([len(tokenizer.encode(v)) for v in promp_list])

    # Preprocess the prompt encodings
    encodings = {k: tokenizer([v], padding='max_length', return_tensors='pt',
                              truncation=True, max_length=max_token_l) for k, v in sorted(prompts.items())}
    encodings = {k: {k1: v1.squeeze(0) for k1, v1 in v.items()}
                 for k, v in encodings.items()}

    rec_dataloader = get_dataloader(
        train_data, train_data_tr, rank, world_size, args.bs, encodings, nonzer_indeces_train)
    val_dataloader = get_dataloader(
        valid_data, valid_data_tr, rank, world_size, args.bs, encodings, nonzer_indeces_valid)
    test_dataloader = get_dataloader(
        test_data, test_data_tr, rank, world_size, args.bs, encodings, nonzer_indeces_test)

    print("Max Prompt Length", max_l)
    print(f"Number of Users is {num_users=}")
    print(f"Number of Movies is {num_movies=}")

    return prompts, rec_dataloader, num_movies, val_dataloader, test_dataloader


def get_dataloader(data, train_data_tr, rank, world_size, bs, encodings, nonzer_indeces_train, user_id_to_row=None):
    rec_dataset = DataMatrix(
        data, train_data_tr, encodings, nonzer_indeces_train, user_id_to_row)
    sampler = DistributedSampler(
        rec_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    rec_dataloader = DataLoader(rec_dataset, batch_size=bs, num_workers=0, pin_memory=False,
                                sampler=sampler)
    return rec_dataloader


def get_embeddings(args, model, item,  alpha=0.5):
    logits_rec, logits_text = None, None

    if args.embedding_module == 'TearsBase':
        movie_emb_clean,  *_ = model(
            data_tensor=item['labels_tr'], input_ids=item['input_ids'], attention_mask=item['attention_mask'], alpha=alpha
        )

    elif 'Tears' in args.embedding_module:
        movie_emb_clean, logits_rec, logits_text, *_ = model(
            data_tensor=item['labels_tr'], input_ids=item['input_ids'], attention_mask=item['attention_mask'], alpha=alpha
        )
    else:
        movie_emb_clean, *_ = model(item['labels_tr'], item['labels'])

    return movie_emb_clean, logits_rec, logits_text


def calculate_metrics(movie_emb_clean, labels, k_values, logits_rec=None, logits_text=None):
    metrics = defaultdict(list)
    logits_rec = logits_rec.cpu().numpy() if logits_rec is not None else None
    logits_text = logits_text.cpu().numpy() if logits_text is not None else None
    for k in k_values:
        metrics[f'ndcg@{k}'].append(NDCG_binary_at_k_batch(
            movie_emb_clean, labels, k=k).tolist())
        metrics[f'recall@{k}'].append(Recall_at_k_batch(
            movie_emb_clean, labels, k=k, mean=False).tolist())
        if logits_rec is not None and logits_text is not None:
            metrics[f'text_ndcg@{k}'].append(
                NDCG_binary_at_k_batch(logits_text, labels, k=k).tolist())
            metrics[f'text_recall@{k}'].append(Recall_at_k_batch(
                logits_text, labels, k=k, mean=False).tolist())
            metrics[f'rec_ndcg@{k}'].append(
                NDCG_binary_at_k_batch(logits_rec, labels, k=k).tolist())
            metrics[f'rec_recall@{k}'].append(Recall_at_k_batch(
                logits_rec, labels, k=k, mean=False).tolist())
    return metrics


def gather_metrics(metrics, mult_process, world_size):
    if mult_process:
        gathered_metrics = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_metrics, metrics)
        return np.mean(sum(sum(gathered_metrics, []), []))
    else:
        return np.mean(sum(metrics, []))


def eval_model(args, model, test_dataloader, rank,  mult_process=True, world_size=2,  alpha=0.5):
    torch.cuda.set_device(rank)
    model.to(rank)
    metrics = defaultdict(list)
    output_metrics = defaultdict(float)
    k_values = [10, 20, 50]

    with torch.no_grad():
        model.eval()
        for b, item in enumerate(test_dataloader):
            item = {k: v.to(rank) for k, v in item.items()}
            labels = item['labels']
            movie_emb_clean, logits_rec, logits_text = get_embeddings(
                args, model, item, alpha)
            mask = np.where(item['labels_tr'].cpu().numpy() > 0)
            recon = movie_emb_clean.float().cpu().numpy()
            recon[mask] = -1e20
            if logits_rec is not None and logits_text is not None:
                logits_rec[mask] = -1e20
                logits_text[mask] = -1e20
            labels = labels.cpu().numpy()
            metrics_batch = calculate_metrics(
                recon, labels, k_values, logits_rec, logits_text)
            for key, value in metrics_batch.items():
                metrics[key].extend(value)
            

        for key in metrics.keys():
            output_metrics[key] = gather_metrics(
                metrics[key], mult_process, world_size)

        if logits_rec is not None:
            output_metrics['ndcg@50_avg'] = np.mean(
                [output_metrics['ndcg@50'], output_metrics['rec_ndcg@50'], output_metrics['text_ndcg@50']])

    return output_metrics


def TrainTears(args, rec_dataloader, model, optimizer, scheduler, pbar, epoch, rank):
    rec_dataloader.sampler.set_epoch(epoch)
    global update_count

    loss_f = get_loss('ot_loss', args)
    losses = defaultdict(list)
    model.train()

    for b, items in enumerate(rec_dataloader):

        if args.max_anneal_steps > update_count:
            anneal = min(args.min_anneal,
                         1. * update_count / args.max_anneal_steps)
        else:
            anneal = args.anneal_cap

        labels = items['labels'].to(rank)
        train_items = items['labels_tr'].to(rank)

        movie_emb, logits_rec, logits_text, text_mu, text_logvar, rec_mu, rec_logvar = model(
            data_tensor=train_items, input_ids=items['input_ids'], attention_mask=items['attention_mask'])

        loss, BCE, wasserstein_loss, BCE_rec, BCE_text, BCE_merged =\
            loss_f(recon_x=movie_emb, x=labels, logits_text=logits_text, logits_rec=logits_rec,  mu=text_mu, logvar=text_logvar, rec_mu=rec_mu, rec_logvar=rec_logvar, anneal=anneal,
                   epsilon=args.epsilon)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        losses['rolling_loss'].append(loss.item())
        losses['bces'].append(BCE.item())
        losses['klds'].append(wasserstein_loss.item())
        losses['bces_mergeds'].append(BCE_merged.item())
        losses['bces_text'].append(BCE_text.item())
        losses['bces_rec'].append(BCE_rec.item())

        torch.cuda.empty_cache()

        pbar.set_description(
            f"Epoch {epoch}: Batch {b} | Loss: {np.mean(losses['rolling_loss']):.4f} | BCE: {np.mean(losses['bces']):.4f} | "
            f"WL: {np.mean(losses['klds']):.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | "
            f"BCE_rec: {np.mean(losses['bces_rec']):.4f} | BCE_text: {np.mean(losses['bces_text']):.4f} | "
            f"BCE_merged: {np.mean(losses['bces_mergeds']):.4f}"
        )

    if args.wandb:
        wandb.log({
            'loss': np.mean(losses['rolling_loss']),
            'epoch': epoch,
            'total_steps': epoch * len(rec_dataloader) + b,
            'lr': scheduler.get_last_lr()[0],
        })


def trainAEs(args, rec_dataloader, model, optimizer, scheduler, pbar, epoch, rank):
    rec_dataloader.sampler.set_epoch(epoch)

    loss_f = get_loss('bce_softmax')
    rolling_loss = []

    global update_count
    update_count = 0

    for b, items in enumerate(rec_dataloader):
        loss = None
        model.train()
        labels = items['labels'].to(rank)
        train_items = items['labels_tr'].to(rank)
        optimizer.zero_grad(set_to_none=True)
        if args.max_anneal_steps > 0:
            anneal = min(args.min_anneal,
                         update_count / args.max_anneal_steps)
        else:
            anneal = args.anneal_cap

        if args.embedding_module == 'MacridVAE':
            movie_emb, loss = model(train_items, labels, anneal=anneal)
        elif args.embedding_module == 'RecVAE':
            movie_emb, loss = model(train_items, labels, calculate_loss=True)
        else:
            movie_emb, mu, logvar = model(
                data_tensor=train_items, input_ids=items['input_ids'], attention_mask=items['attention_mask'])
        if loss is None:
            loss = loss_f(movie_emb, labels, mu, logvar, anneal)

        loss.backward()
        optimizer.step()
        rolling_loss.append(loss.item())
        update_count += 1
        pbar.set_description(
            f"Epoch {epoch}: Batch: {b} loss: {np.mean(rolling_loss)} current lr: {scheduler.get_last_lr()[0]}")
        scheduler.step()

    if not args.debug:

        wandb.log({'loss': np.mean(rolling_loss),
                   "epoch": epoch,
                   "total_steps": epoch*len(rec_dataloader) + b,
                   'lr': scheduler.get_last_lr()[0],
                   })

    pbar.set_description(
        f"Batch {epoch}: loss: {np.mean(rolling_loss)} current lr: {scheduler.get_last_lr()[0]}")


def get_scheduler(optimizer, args):

    if args.scheduler == 'linear_decay':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 - epoch / args.epochs)
    elif args.scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'cosine_warmup':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs)
    elif args.scheduler == 'None':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1)
    else:
        raise NotImplementedError


def parse_args(notebook=False):  # Parse command line arguments
    parser = argparse.ArgumentParser(description="TEARS")
    parser.add_argument("--data_name", default='ml-1m', type=str)
    parser.add_argument("--vae_path", type=str)
    parser.add_argument("--model_name", default='Transformer', type=str)
    parser.add_argument("--embedding_module",default="MVAE", type=str)
    parser.add_argument("--scheduler", default='None', type=str)
    parser.add_argument("--bs", default=64, type=int)
    parser.add_argument("--seed", default=2024, type=int)
    parser.add_argument("--max_anneal_steps", default=10000, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--topk", default=20, type=int)
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--epsilon", default=1, type=float)
    parser.add_argument("--tau", default=1, type=float)
    parser.add_argument("--text_tau", default=1, type=float)
    parser.add_argument("--rec_tau", default=1, type=float)
    parser.add_argument("--recon_tau", default=1, type=float)
    parser.add_argument("--emb_dim", default=400, type=float)
    parser.add_argument("--gamma", default=.0035, type=float)
    parser.add_argument("--min_anneal", default=.5, type=float)
    parser.add_argument("--lr", default=.001, type=float)
    parser.add_argument("--l2_lambda", default=.00, type=float)
    parser.add_argument("--kfac", default=10, type=int)
    parser.add_argument("--dfac", default=100, type=int)
    parser.add_argument("--dropout", default=.1, type=float)
    parser.add_argument("--anneal", default=True, type=bool)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--loss', default='bce_softmax',type=str, choices=['bce', 'bce_softmax', 'kl'])
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--total_steps", type=int, default=1)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument('--eval_control', action='store_true')
    parser.add_argument('--binarize', action='store_true')
    parser.add_argument('--nogb', action='store_true')
    parser.add_argument('--KLD', action='store_true')
    parser.add_argument('--scratch', action='store_true')
    parser.add_argument('--mask_control_labels', action='store_true')
    args = parser.parse_args() if not notebook else parser.parse_args(args=[])
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.model_log_name = f"{args.embedding_module.replace('/','_')}_{args.data_name}_{current_time}_{args.seed}"
    print(f'Model will be saved under the name {args.model_log_name }')

    directory_path = "../scratch"

    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        print(
            f"The directory '{directory_path}' exists. Will save all weights and models there")
        args.scratch = '../scratch'
        #automatically set transformer cache to scratch directory
        os.environ['TRANSFORMERS_CACHE'] = args.scratch
        os.environ['HF_HOME'] = args.scratch
        os.environ['HF_DATASETS_CACHE'] = args.scratch
        os.environ['TORCH_HOME'] = args.scratch
    else:
        args.scratch = '.'

    return args
