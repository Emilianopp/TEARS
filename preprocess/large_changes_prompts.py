import sys 
import argparse
from trainer.transformer_utilts import *
sys.path.append("..")
from model.MF import get_model, get_tokenizer
from helper.dataloader import *
from model.eval_model import *



def main():

    rank = 0
    world_size = 1

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='netflix', help='Name of the dataset')
    parser.add_argument('--split', type=str, default='', help='Name of the dataset')
    args = parser.parse_args()
    item_genre_dict = map_id_to_genre(args.data_name)
    item_title_dict = map_id_to_title(args.data_name)
    tokenizer = get_tokenizer(args)
    eval_model_obj = LargeScaleEvaluator(None, item_title_dict, item_genre_dict, tokenizer, 0, args, alpha=0)
    args.bs = 250
    args.binarize = True
    prompts, rec_dataloader, num_movies, val_dataloader, test_dataloader = load_data(args, tokenizer, rank, world_size)

    subset_keys = []
    labels = {}
    for b in (val_dataloader if args.split == '' else test_dataloader):
        subset_keys.append([x.item() for x in b['idx'].flatten()])
        for x, k in zip(b['labels'], b['idx'].flatten()):
            labels[k.item()] = x.tolist()

    subset_keys = sum(subset_keys, [])

    eval_model_obj.promptGPT(subset_keys, prompts, labels, split='')

if __name__ == "__main__":
    main()
