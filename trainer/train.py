from model.MF import get_model, get_tokenizer, EASE
import time
from model.eval_model import LargeScaleEvaluator
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from helper.dataloader import *
import torch
import pandas as pd
from tqdm import tqdm
import torch.optim as optim
from trainer.transformer_utilts import *
import sys
import os
import random 
sys.path.append("..")


def set_seed(seed: int = 2024) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def cleanup():
    dist.destroy_process_group()


def setup(rank, world_size):
    assert torch.distributed.is_available()
    print("PyTorch Distributed available.")
    print("  Backends:")
    print(f"    Gloo: {torch.distributed.is_gloo_available()}")
    print(f"    NCCL: {torch.distributed.is_nccl_available()}")
    print(f"    MPI:  {torch.distributed.is_mpi_available()}")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )


def train_fun(rank, world_size, args):
    setup(rank, world_size)
    set_seed(args.seed)
    start_time = time.time()
    # get unique hash for both gpus
    if args.wandb:
        tags = [args.data_name, os.environ["SLURM_JOB_ID"]]
        wandb.init(project='TEARS', name=args.model_log_name,
                   group=time.strftime("%d-%m-%Y_%H:%M:%S", time.localtime()),
                   tags=tags, config=args)
        wandb.config.update(args)
        wandb.watch_called = False  # To avoid re-watching the model
        wandb.define_metric("batch")
        wandb.define_metric("epoch")
        wandb.define_metric("total_steps")
        wandb.define_metric("val_recall", step_metric="epoch")
        wandb.define_metric("val_ndcg", step_metric="epoch")
        wandb.define_metric("bce_loss", step_metric="total_steps")
        wandb.define_metric("loss", step_metric="total_steps")
        wandb.define_metric("lr", step_metric="epochs")
    else:
        wandb.log = lambda x: None

    torch.cuda.set_device(rank)
    tokenizer = get_tokenizer()
    mods = ['TearsRecVAE', 'TearsMVAE', 'TearsMacrid', 'RecVAEGenreVAE']

    prompts, rec_dataloader, num_movies, val_dataloader, test_dataloader = load_data(
        args, tokenizer, rank, world_size)
    if args.embedding_module == 'EASE':
        # if ease directly train
        EASE.train(args, rec_dataloader, val_dataloader,
                   test_dataloader, num_movies)
    model = get_model(args, num_movies)
    model.to(rank)
    item_title_dict = map_id_to_title(args.data_name)
    item_genre_dict = map_id_to_genre(args.data_name)

    # Make sure encoder is kept frozen through optimization process and print trainable parameters
    if args.embedding_module in mods:
        for name, param in model.vae.named_parameters():
            if 'q' in name or 'encoder' in name or 'item_embeddings' in name or 'prototype_embeddings' in name:
                param.requires_grad = False
            else:
                print(name)

    model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank,
                find_unused_parameters=False if 'RecVAE' in args.embedding_module else True)

    print(f"Number of Batches = {len(rec_dataloader)}")

    params = [p for p in model.parameters() if p.requires_grad]
    # use AdamW optimizer for LLMs and Adam for VAEs (as per their respective papers)
    optimizer = optim.AdamW(params, lr=args.lr) if args.embedding_module in mods else optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.l2_lambda)
    scheduler = get_scheduler(optimizer, args)
    min_val_metric = -np.inf
    last_saved_epoch = 0

    for e in (pbar := tqdm(range(args.epochs))):
        model.train()
        model.share_memory()
        if args.embedding_module in mods:
            TrainTears(args, rec_dataloader, model,
                       optimizer, scheduler, pbar, e, rank)
        else:
            trainAEs(args, rec_dataloader, model,
                     optimizer, scheduler, pbar, e, rank)

        if e % 1 == 0:

            outputs = eval_model(args, model, val_dataloader,
                                 rank, world_size=world_size)
            if not args.debug:
                val_outputs = {f'val_{k}': v for k, v in outputs.items()}
                wandb.log({**val_outputs})
          

            eval_metric = outputs['ndcg@50'] if 'ndcg@50_avg' not in outputs else outputs['ndcg@50_avg']

            if eval_metric > min_val_metric:
                print(f"Checkpointing at epoch {e}")
                min_val_metric = eval_metric
                torch.save(model.module.state_dict(
                ), f'{args.scratch}/saved_model/{args.data_name}/{args.model_log_name}.pt')
                print(
                    f'Saved model to {args.scratch}/saved_model/{args.data_name}/{args.model_log_name}.pt')
                last_saved_epoch = e
                wandb.log({'ndcg@50_max': min_val_metric})
                wandb.log({'last_saved_epoch': last_saved_epoch})

            pbar.set_postfix(
                {'val_ndcg': outputs['ndcg@50'], 'recall': outputs['recall@50'], 'last_saved_epoch': last_saved_epoch})

    # Load model for evaluation
    model = get_model(args, num_movies)
    model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank,
                find_unused_parameters=False if 'RecVAE' not in args.embedding_module else True)
    state_dict = torch.load(
        f'{args.scratch}/saved_model/{args.data_name}/{args.model_log_name}.pt')
    model.module.load_state_dict(state_dict)

    
    val_outputs = eval_model(args, model, val_dataloader,
                             rank, world_size=world_size)
    test_outputs = eval_model(
        args, model, test_dataloader, rank, world_size=world_size)
    test_outputs = {f'test_{k}': v for k, v in test_outputs.items()}
    val_outputs = {f'val_{k}': v for k, v in val_outputs.items()}
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time} seconds")

    out_metrics = defaultdict(dict)
    ups = defaultdict(list)
    downs = defaultdict(list)
    ndcgs20 = defaultdict(list)
    eval_genre_up = defaultdict(dict)
    eval_genre_down = defaultdict(dict)
    best_alphas = {}
    alphas = [x/10 for x in range(0, 11)]
    module = args.embedding_module
    max_alpha = 0
    if args.eval_control:
        large_change_evaluator = LargeScaleEvaluator(
            model, item_title_dict, item_genre_dict, tokenizer, 0, args, alpha=0.5, split='test')


        if args.embedding_module in (mods + ['TearsBase']):
            # Get controllability metrics for logging
            if module == 'TearsBase': 
                eval_model_obj = LargeScaleEvaluator(
                    model, item_title_dict, item_genre_dict, tokenizer, 0, args, alpha=0)
                max_alpha = 0
                metrics = eval_model(args, model, test_dataloader, rank,
                                     world_size=world_size, alpha=0)
                up, down = eval_model_obj.evaluate(
                    test_dataloader, prompts, 20, rank=rank)
                ups[module].append(-up)
                downs[module].append(down)

            # For TearsVAE models Sweep through alphas in the validation set and report result with that alpha on the test set
            else:
                ndcgs = {}
                eval_model_obj = LargeScaleEvaluator(
                    model, item_title_dict, item_genre_dict, tokenizer, 0, args, alpha=0)
                for alpha in alphas:

                    n2_track = eval_model(args, model, val_dataloader, rank, world_size=world_size,
                                          alpha=alpha)['ndcg@20']
                    eval_model_obj.alpha = alpha
                    eval_model_obj.alpha2 = alpha
                    up, down = eval_model_obj.evaluate(
                        test_dataloader, prompts, 20, rank=rank)
                    ndcgs20[module].append(n2_track)
                    ups[module].append(-up)
                    downs[module].append(down)
                    ndcgs[alpha] = n2_track
                    

            max_alpha = max(ndcgs, key=ndcgs.get)
            metrics = eval_model(args, model, test_dataloader, rank,
                                    world_size=world_size,  alpha=max_alpha)
            eval_model_obj.alpha = 1
            eval_model_obj.alpha2=.5

            eval_genre_up[module] = eval_model_obj.evaluate_genre(model,test_dataloader,20,rank = rank)
            eval_genre_up[module] = {f'{x}_up' : eval_genre_up[module][x] for x in eval_genre_up[module]}

            eval_genre_down[module] = eval_model_obj.evaluate_genre(model,test_dataloader,20,neg = True,rank = rank)
            eval_genre_down[module] = {f'{x}_down' : eval_genre_down[module][x] for x in eval_genre_down[module]}
            max_alpha = max(ndcgs, key=ndcgs.get)
            best_alphas[module] = max_alpha
            metrics = eval_model(args, model, val_dataloader, rank, world_size=world_size,
                                                    alpha=max_alpha)
            downs_all = {f'down_.{x}' : downs[module][x] for x in range(len(downs[module]))}
            ups_all = {f'ups_.{x}' : ups[module][x] for x in range(len(downs[module]))}
            ndcg20_list = {f'ndcgs_.{x}' : ndcgs20[module][x] for x in range(len(downs[module]))}
            metrics = {**metrics,**ndcg20_list, **ups_all, **downs_all, 
                    'eval_genre_up':eval_genre_up[module],'eval_genre_down':eval_genre_down[module],'best_alpha':max_alpha}

            out_metrics[module] = metrics

    log_results = pd.DataFrame({'model_name': [args.model_log_name], **test_outputs, **val_outputs,
                                'ups': [[ups[module]]], 'downs': [[downs[module]]], 'ndcgs': [[ndcgs20[module]]], 'max_alpha': [max_alpha] if args.eval_control else 'N/A',
                                }).round(4)

    csv_path = f"./model_logs/{args.data_name}/{args.embedding_module}/results_.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    print(f'Saved to {csv_path}')

    if os.path.exists(csv_path):
        existing_data = pd.read_csv(csv_path)
        updated_data = pd.concat(
            [existing_data, log_results], ignore_index=True)
        updated_data.to_csv(csv_path, index=False)
    else:
        log_results.to_csv(csv_path, index=False)
    cleanup()


if __name__ == "__main__":

    args = parse_args(notebook=False)
    world_size = torch.cuda.device_count()
    print('World Size:', world_size)
    mp.spawn(train_fun,
             args=(world_size, args), nprocs=world_size)
