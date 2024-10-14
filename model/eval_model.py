from helper.dataloader import *
from trainer.transformer_utilts import *
from model.MF import get_model, get_tokenizer
import argparse
import random
from collections import defaultdict
import pandas as pd
from dotenv import load_dotenv
import openai
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from collections import Counter
import os
from torch.nn.parallel import DistributedDataParallel


class LargeScaleEvaluator:
    def __init__(
        self,
        model,
        item_title_dict,
        item_genre_dict,
        tokenizer,
        rank,
        args,
        alpha=1,
        split="test",
    ):
        super().__init__()
        self.model = model
        self.item_title_dict = item_title_dict
        self.item_genre_dict = item_genre_dict
        self.tokenizer = tokenizer
        self.rank = rank
        self.args = args
        self.alpha = alpha
        self.alpha2 = None
        self.split = split

        # Consolidating genre list creation into a helper method
        self.genre_list, self.genre_set,self.counts = self._create_genre_list(
            item_genre_dict)

        # Checking if CSV file exists
        if os.path.exists(
            f"./saved_user_summary/{self.args.data_name}/results_large_genre_{split}_{args.llm_backbone}.csv"
        ):
            self.df = pd.read_csv(
                f"./saved_user_summary/{self.args.data_name}/results_large_genre_{split}_{args.llm_backbone}.csv"
            )

    def _create_genre_list(self, item_genre_dict):
        """Helper function to create a list and set of genres."""
        counts = Counter(sum([v for v in item_genre_dict.values()], []))
        counts = {k: v for k, v in counts.items() if v > 100}
        genre_list = [x.lower().replace("-", " ") for x in counts.keys()]
        genre_set = ", ".join(genre_list)
        return genre_list, genre_set,counts

    def genrewise_ndcg(self, genre_movies, genre, min_k=0, max_k=None):
        """Helper function to calculate NDCG for a genre."""
        relevance = [1 if genre in g else 0 for g in genre_movies[min_k:max_k]]
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
        idcg = sum(1 / np.log2(i + 2) for i in range(len(relevance)))
        return dcg / idcg if idcg > 0 else 0

    def getGenreDeltaBatch(
        self, s1, s2, labels, topk, genre_1, genre_2, rank, neg=False, max_delta=False,proportional=False,small = False
    ):
        if self.args.mask_control_labels:
            labels = torch.zeros_like(labels)

        generate_recs = (
            self.model.module.generate_recommendations_batch
            if isinstance(self.model, DistributedDataParallel)
            else self.model.generate_recommendations_batch
        )

        topk1 = generate_recs(
            s1, self.tokenizer, labels, topk, rank, alpha=self.alpha, neg=neg
        )
        topk2 = generate_recs(
            s2,
            self.tokenizer,
            labels,
            topk,
            rank,
            alpha=self.alpha2 if self.alpha2 else self.alpha,
            neg=neg,
        )

        movie_genres1 = [
            [self.item_genre_dict[i] for i in sublist] for sublist in topk1
        ]
        movie_genres2 = [
            [self.item_genre_dict[i] for i in sublist] for sublist in topk2
        ]


        change_up_list = []
        change_down_list = []


        for i in range(len(movie_genres1)):
            change_up = self.genrewise_ndcg(
                movie_genres1[i], genre_2[i], min_k=0, max_k=topk
            ) - self.genrewise_ndcg(movie_genres2[i], genre_2[i], min_k=0, max_k=topk)
            change_up_list.append(change_up)
            change_down = self.genrewise_ndcg(
                movie_genres1[i], genre_1[i], min_k=0, max_k=topk
            ) - self.genrewise_ndcg(movie_genres2[i], genre_1[i], min_k=0, max_k=topk)
            change_down_list.append(change_down)

        return change_down_list, change_up_list

    def _set_api_key(self):
        """Helper method to set the OpenAI or Llama API key."""
        if "gpt" not in self.args.llm_backbone:
            openai.api_base = "https://api.llama-api.com"
        key = (
            os.getenv("OPEN-AI-SECRET")
            if "gpt" in self.args.llm_backbone
            else os.getenv("LLAMA-KEY")
        )
        openai.api_key = key

    def promptGPT(self, k_sample, prompts, labels, seed=2024, save_every_iter=False):
        np.random.seed(seed)
        load_dotenv()

        # Reuse API key setting method
        self._set_api_key()

        (
            deltas_ndcg_up,
            deltas_ndcg_down,
            move_up_genres,
            move_down_genres,
            outputs,
            keys,
        ) = [],[],[],[],[],[]

        for k in (pbar := tqdm(k_sample)):
            s = prompts[k]
            labs = labels[k]

            prompt = \
                        f"""
                        You are a professional editor please identify the users preferred genres from the following:
                        {self.genre_set}
                        """

            user_prompt =\
                        f"""
                        Please identify the users most favorite genre from the following summary and the least favorite genre: 
                        in the format Favorite: [genre]\n Least Favorite: [genre]
                        {s}.
                        Remember the genre you pick must be in this set of genres {self.genre_set} and in the format Favorite: [genre]\n Least Favorite: [genre]
                        """

            msg = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_prompt},
            ]

        
            genres = openai.ChatCompletion.create(
                model=self.args.llm_backbone,
                messages=msg,
                max_tokens=300,
                temperature=1 if "gpt" in self.args.llm_backbone else 0,
                seed=2024,
            )["choices"][0]["message"]["content"]

            lines = genres.split("\n")

            try:
                favorite_genre = lines[0].split(
                    ": ")[1].lower().replace("-", " ").rstrip(' ')
                least_favorite_genre = lines[1].split(
                    ": ")[1].lower().replace("-", " ").rstrip(' ')
            except:
                continue

            if (
                favorite_genre not in self.genre_list
                or least_favorite_genre not in self.genre_list
            ):

                pbar.set_description(
                    f"pass {k} {favorite_genre} {least_favorite_genre}"
                )

                continue

            msg = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": genres},
                {
                    "role": "user",
                    "content": f"Now using this setup write the a new summary in the same style that reflects that {favorite_genre} is your least favorite\
                            and {least_favorite_genre} is your favorite only output the full summary keep the format and length the same",
                },
            ]


            gpt_output = openai.ChatCompletion.create(
                model=self.args.llm_backbone,
                messages=msg,
                max_tokens=300,
                temperature=1 if "gpt" in self.args.llm_backbone else 0,
                seed=2024,
            )["choices"][0]["message"]["content"]
            
            

                
            delta1, delta2 = self.getGenreDeltaBatch(
                s,
                gpt_output,
                labs,
                self.args.topk,
                [favorite_genre],
                [least_favorite_genre],
                rank=self.rank,
            )
            move_down_genres.append(favorite_genre)
            move_up_genres.append(least_favorite_genre)
            outputs.append(gpt_output)
            keys.append(k)
            deltas_ndcg_down.append(delta1)
            deltas_ndcg_up.append(delta2)

            pbar.set_description(
                f"gen {favorite_genre}, average ndcgs up {np.mean(deltas_ndcg_up)} average ndcg down {np.mean(deltas_ndcg_down)}"
            )


            data = {
                "move_down_genres": move_down_genres,
                "move_up_genres": move_up_genres,
                "outputs": outputs,
                "keys": keys,
                "deltas_ndcg_down": deltas_ndcg_down,
                "deltas_ndcg_up": deltas_ndcg_up,
            }

            if save_every_iter:
                new_data = pd.DataFrame(data)
                self.df = pd.concat([self.df, new_data])
                self.df.to_csv(
                    f"./saved_user_summary/{self.args.data_name}/results_large_genre_{self.split}_{self.args.llm_backbone}.csv"
                )
                print("WROTE DF")


        if not save_every_iter:
            new_data = pd.DataFrame(data)
            self.df = pd.concat([self.df, new_data])
            self.df.to_csv(
                f"./saved_user_summary/{self.args.data_name}/results_large_genre_{self.split}_{self.args.llm_backbone}.csv"
            )
            print("WROTE DF")

    def evaluate(self, dataloader, prompts, topk, rank=None, max_delta=False,proportional = False,save = False):
        self.model.eval()

        self.delta_up , self.delta_down, self.least_fav,self.most_fav = [],[],[],[]

        for b in tqdm(dataloader, desc='Controllability'):
            uids = b['idx'].flatten().tolist()  
            labels = b['labels_tr'].to(rank)  
            valid_uids = [uid for uid in uids if uid in self.df['keys'].values]
            valid_indices = [uids.index(uid) for uid in valid_uids]
            valid_labels = labels[valid_indices]
            valid_prompts = [prompts[uid] for uid in valid_uids]
            if valid_uids:
                sub_df = self.df[self.df['keys'].isin(valid_uids)]
                gpt_outputs = sub_df['outputs'].values.tolist()
                genre1_list = sub_df['move_down_genres'].apply(lambda x: x.lower().replace('-', ' ')).values
                genre2_list = sub_df['move_up_genres'].apply(lambda x: x.lower().replace('-', ' ')).values

                self.least_fav.extend(genre2_list)
                self.most_fav.extend(genre1_list)
                
                with torch.no_grad():
                    down, up, *_ = self.getGenreDeltaBatch(valid_prompts, gpt_outputs, valid_labels, topk, genre1_list, genre2_list, rank=rank,proportional = proportional)


                self.delta_up.extend(up)
                self.delta_down.extend(down)


            avg_up = np.mean(self.delta_up) if self.delta_up else 0
            avg_down = np.mean(self.delta_down) if self.delta_down else 0
            tqdm.write(f"Average NDCG up: {avg_up} | Average NDCG down: {avg_down}")
            

        # Final averages
        final_avg_up = np.mean(self.delta_up) if self.delta_up else 0
        final_avg_down = np.mean(self.delta_down) if self.delta_down else 0
        tqdm.write(f"Final average NDCG up: {final_avg_up} | Final average NDCG down: {final_avg_down}")
        if save:
            data = {
                "delta_up": self.delta_up,
                "delta_down": self.delta_down,
                "least_fav": self.least_fav,
                "most_fav": self.most_fav,
            }
            df = pd.DataFrame(data)
            df.to_csv(
                f"./saved_user_summary/{self.args.data_name}/evaluation_results_large_{self.split}_{self.args.llm_backbone}.csv",
                index=False,
            )
        # Return the final averages
        return final_avg_up, final_avg_down

    
    def evaluate_genre(self, model, dataloader, topk, rank=None,  neg=False, genre_changed=None,save = False):

        if genre_changed is not None:
            genres = [genre_changed]
        else:
            # Get the top 10 genres from self.counts (already sorted)
            self.counts = dict(sorted(self.counts.items(), key=lambda item: item[1], reverse=True))
            genres = [x for x in list(self.counts.keys())[:10]]
            
        item = 'books' if self.args.data_name == 'goodbooks' else 'movies'

        out_metrics = {}
        for b in tqdm(dataloader, desc='Controllability'):
            uids = b['idx'].flatten().tolist()
            labels = b['labels_tr'].to(rank)

            for genre in genres:
                genre_up ,batch_s1 , batch_s2 = [],[],[]

                for j in uids:
                    # Create genre-specific summaries
                    rep_s1 = f'Summary: {genre} {item} are the users favourite they really enjoy {genre} content '
                    #doesnt matter what this is
                    rep_s2 = f''

                    batch_s1.append(rep_s1)
                    batch_s2.append(rep_s2)
                    
                genre_list = [genre] * len(batch_s1)
                genre_dummie = ['action'] * len(batch_s1)
                _, up, *_ = self.getGenreDeltaBatch(batch_s2, batch_s1, labels, topk, genre_dummie, genre_list, rank=rank, neg=neg)                           
                
                genre_up.append(up)
                
                out_metrics[genre] = [np.mean(genre_up)]


                
            tqdm.write(f"Genre {genre}: {np.mean(genre_up)}")
            
        if save: 
            df = pd.DataFrame(out_metrics)
            df.to_csv(
                f"./saved_user_summary/{self.args.data_name}/evaluation_results_large_{self.split}_{self.args.llm_backbone}_{neg}.csv",
                index=False,
            )

        return out_metrics



        

class FineGrainedEvaluator():

    def __init__(
        self,
        item_title_dict,
        item_genre_dict,
        tokenizer,
        rank,
        args,
        models_dict,
        uids,
        split="test",
    ):
        """
        Class to run the fine-graind experiment in the paper.

        Parameters:
            model: The recommendation model.
            item_title_dict: Dictionary of item titles.
            item_genre_dict: Dictionary of item genres.
            tokenizer: Tokenizer used for processing text.
            rank: Rank or device rank for distributed computing.
            args: Additional arguments for the evaluator.
            alpha: Default value for alpha tuning parameter (optional, default = 0.5).
            split: Dataset split (optional, default = 'test').
        """
        super().__init__()
        self.item_title_dict = item_title_dict
        self.item_genre_dict = item_genre_dict
        self.num_movies = len(item_title_dict)
        self.uids = uids
        self.tokenizer = tokenizer
        self.rank = rank
        self.models_dict = models_dict
        self.split = split
        counts = Counter(sum([v for v in item_genre_dict.values()], []))

        self.counts = {k: v for k, v in counts.items() if v > 100}
        self.genre_list = list(self.counts.keys())
        self.genre_list = [x.lower().replace("-", " ")
                           for x in self.genre_list]
        self.genre_set = ", ".join(self.genre_list)
        self.args = args

        if os.path.exists(
            f"./saved_user_summary/{args.data_name}/results_small_{split}_{args.llm_backbone}_median.csv"
        ):
            self.df = pd.read_csv(
                f"./saved_user_summary/{args.data_name}/results_small_{split}_{args.llm_backbone}_median.csv"
            )
        else:
            self.df = pd.DataFrame()

    def _set_api_key(self):
        """Helper method to set the OpenAI or Llama API key."""
        if "gpt" not in self.args.llm_backbone:
            openai.api_base = "https://api.llama-api.com"
        key = (
            os.getenv("OPEN-AI-SECRET")
            if "gpt" in self.args.llm_backbone
            else os.getenv("LLAMA-KEY")
        )
        openai.api_key = key
    def get_user_subset(self, uids, labels_d, labels_eval,prompts):
        """Loops over all models and alpha values to find eligable items lying between ranks 100 and 500
        grabs all users that satisfy that condition and returns a dictionary of such users
        """
        movies_u = {}
        for user in tqdm(uids):
            user_sets = []
            for i,(model_name, model) in enumerate(self.models_dict.items()):
                user_labs_in = labels_d[user]
                user_labs_target = labels_eval[user]
                for alpha in [x / 10 for x in range(1, 11)]:
                    s = prompts[user]
                    labels_tens = torch.tensor(user_labs_in)
                    preds = model.generate_recommendations(
                        s,
                        tokenizer=self.tokenizer,
                        topk=self.num_movies,
                        data_tensor=labels_tens.to(self.rank),
                        alpha=alpha,
                    )
                    labels_set = set(user_labs_target)
                    in_set = set(preds[100:500])
                    trainer_set = set.intersection(labels_set, in_set)
                    user_sets.append(trainer_set)
            
            intersection = set.intersection(*user_sets)
            movies_u[user] = intersection
        

        self.non_empty_users_d = {k: v for k,
                                  v in movies_u.items() if len(v) > 0}

    def find_movie_rank(self, movie_title, labels):
        for i, lab in enumerate(labels):
            if lab == movie_title:
                return i
        return None

    def prompt_LLM_small(
        self, prompts, labels_d, models_dict, save_every_iter=True, num_attemps=3
    ):
        """Function for producing text Edits with GPT/LLaMA make sure to have an API set in a .env file
        you can obtain API keys from OpenAI or LLaMA:

            +OpenAI: https://platform.openai.com/docs/overview
            +LLama: https://www.llama-api.com/

        We used GPT-4-1106-preview and LLaMA-3.1 405b as it was the best and cheapest option at the time but
        you can use any model you want as long as it is supported through these two APIs, else you'll have to edit the code.
        """

        np.random.seed(10)
        random.seed(10)
        load_dotenv()
        self._set_api_key()
        

        for j, k in (pbar := tqdm(enumerate(self.non_empty_users_d.keys()))):

            if 'uid' in self.df.columns and k in self.df.uid.values:
                pbar.set_description(f"Skipping {k}")
                continue

            s = prompts[k]
            labs = labels_d[k]
            idx_title = self.item_title_dict[
                random.sample(self.non_empty_users_d[k], 1)[0]
            ]

            model_ranks_inner = defaultdict(list)
            gpt_outputs_per_model, model_ranks = {},{}
            outputs_l, out_iter = [],[]
            for i in range(num_attemps):
                try:
                    prompt = f"""
                                You are a professional editor please summarise the movie in 5 into 5 words only refer to plot points/themes
                                """
                    user_prompt = f"""
                                movie = {idx_title}
                                """

                    msg = [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user_prompt},
                    ]

                    plot_points = openai.ChatCompletion.create(
                        model=self.args.llm_backbone,
                        messages=msg,
                        max_tokens=300,
                        temperature=1 if "gpt" in self.args.llm_backbone else 0,
                        seed=(2020 + i),
                    )["choices"][0]["message"]["content"]

                    msg = [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": plot_points},
                        {
                            "role": "user",
                            "content": f"""Now using this edit those 5 words into this summary, replace a sentence where it makes sense, only output the summary \n {s} 
                                \n only output the new summary make sure the 5 new comma sepearted words are in a new sentence replacing an old one somehwere together in the new summary""",
                        },
                    ]
                    gpt_output = openai.ChatCompletion.create(
                        model=self.args.llm_backbone,
                        messages=msg,
                        max_tokens=300,
                        temperature=1 if "gpt" in self.args.llm_backbone else 0,
                        seed=(2020 + i),
                    )["choices"][0]["message"]["content"]
                    out_iter.append(gpt_output)


                except Exception as e:
                    # The openai api sometimes times out for not reason, unless you want to have unexpected interruptions leave this in
                    print(e)
                    continue

                for iter_model, (model_name, model) in enumerate(self.models_dict.items()):
                    with torch.no_grad():
                        
                        m1 = model.generate_recommendations(
                                s,
                                tokenizer=self.tokenizer,
                                topk=self.num_movies,
                                data_tensor=labs.to(self.rank),
                                alpha=0,
                            )

                        m2 = model.generate_recommendations(
                            gpt_output,
                            tokenizer=self.tokenizer,
                            topk=self.num_movies,
                            data_tensor=labs.to(0),
                            alpha=0,
                        )

                    titles_flat = [self.item_title_dict[x] for x in m1]
                    rank_titles = self.find_movie_rank(idx_title, titles_flat)                    
                    titles2_flat = [self.item_title_dict[x] for x in m2]
                    rank_titles2 = self.find_movie_rank(idx_title, titles2_flat)
                    outputs_l.append(gpt_output)
                    ranking_difference = rank_titles - rank_titles2

                    model_ranks_inner[model_name].append(ranking_difference)


            for model_name, model in models_dict.items():
                median_index = np.where(
                    model_ranks_inner[model_name]
                    == np.median(model_ranks_inner[model_name])
                )[0][0]
                model_ranks[model_name] = np.median(
                    model_ranks_inner[model_name])
                gpt_output_median = out_iter[median_index]
                gpt_outputs_per_model[f"gpt_chosen_{model_name}"] = (
                    gpt_output_median
                )

                pbar.set_description(
                    f"Difference in ranking: {model_name}", model_ranks[model_name]
                )

                model_ranks_inner_df = {
                    f"{model_name}_ranks": v for k, v in model_ranks_inner.items()
                }
            self.df = pd.concat(
                [
                    self.df,
                    pd.DataFrame(
                        [
                            {
                                "uid": k,
                                "Movie Title": idx_title,
                                **gpt_outputs_per_model,
                                "original_rankings": titles_flat,
                                "augmented_rankings": titles2_flat,
                                "plot_points": plot_points,
                                **model_ranks,
                                **model_ranks_inner_df,
                                "all_gpt_outputs": out_iter,
                            }
                        ]
                    ),
                ]
            )

            if save_every_iter:
                self.df.to_csv(
                    f"./saved_user_summary/{self.args.data_name}/results_small_{self.split}_{self.args.llm_backbone}_median.csv"
                )

        if not save_every_iter:
            self.df.to_csv(
                f"./saved_user_summary/{self.args.data_name}/results_small_{self.split}_{self.args.llm_backbone}_median.csv"
            )
    def evaluate(self,prompts,user_ids_data_tensor,tokenizer, save = False):

        labels_subset = torch.concat([user_ids_data_tensor[u] for u in self.df.uid])

        s_list = [prompts[k] for k in self.df.uid]
        movie_titles = self.df['Movie Title'].tolist()
        alphas = [x/10 for x in range(11)]

        
        d_err, d_props = {},{}
        for module,models_iter in (pbar:=tqdm(self.models_dict.items())):
            gpt_outputs = self.df[f'gpt_chosen_{module}'].tolist()
            diff = defaultdict(list)
            
            for alpha in alphas:
                with torch.no_grad():
                    recs1 = models_iter.generate_recommendations_batch(s_list,tokenizer = tokenizer,topk = self.num_movies,data_tensor=labels_subset,alpha = alpha) 
                    recs2 = models_iter.generate_recommendations_batch(gpt_outputs,tokenizer = tokenizer,topk = self.num_movies,data_tensor=labels_subset,alpha = alpha)  

                for r1,r2,m in zip(recs1,recs2,movie_titles):

                    
                    titles  = [self.item_title_dict[x] for x in r1]
                    titles2 = [self.item_title_dict[x] for x in r2]

                    rank_titles = self.find_movie_rank(m,titles)
                    rank_titles2 = self.find_movie_rank(m,titles2)

                    ranking_difference = rank_titles - rank_titles2
                    diff[alpha].append( ranking_difference)
                    
                    if rank_titles > 500:
                        tqdm.write(f"Ranking Difference: {rank_titles} for model {module} {ranking_difference=}")

                    pbar.set_description(f"{module=}, {alpha=}")
                
            d_props[module] = {alp:np.mean(x) for alp,x in  diff.items()}
            d_err[module] = {alp:np.std(x)/len(s_list) for alp,x in  diff.items()}
        sub = pd.DataFrame(d_props)
        dfs_err = pd.DataFrame(d_err)
        if save:
            sub.to_csv(f"./saved_user_summary/{self.args.data_name}/results_small_{self.split}_{self.args.llm_backbone}_mean.csv")
            dfs_err.to_csv(f"./saved_user_summary/{self.args.data_name}/results_small_{self.split}_{self.args.llm_backbone}_err.csv")
    



def parse_args(notebook=False):  # Parse command line arguments
    parser = argparse.ArgumentParser(description="TEARS")
    parser.add_argument("--data_name", default='ml-1m', type=str)
    parser.add_argument("--llm_backbone", default='gpt-4-1106-preview', type=str)
    parser.add_argument("--vae_path", type=str)
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
    parser.add_argument("--kfac", default=2, type=int)
    parser.add_argument("--dfac", default=100, type=int)
    parser.add_argument("--dropout", default=.1, type=float)
    parser.add_argument("--anneal", default=True, type=bool)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--loss', default='bce_softmax',type=str, choices=['bce', 'bce_softmax', 'kl'])
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--total_steps", type=int, default=1)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument('--eval_control', action='store_true')
    parser.add_argument('--binarize', action='store_true')
    parser.add_argument('--nogb', action='store_true')
    parser.add_argument('--KLD', action='store_true')
    parser.add_argument('--scratch', action='store_true')
    parser.add_argument('--mask_control_labels', action='store_true')
    
    '''Parameters specific to these experiments'''
    parser.add_argument("--alpha", default=0, type=float)
    parser.add_argument("--split",default="test",choices=['val','test'], type=str)
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--experiment",default="large_prompt",choices=['large_eval','guided','small_prompt','large_prompt','small_eval'], type=str)
    '''====================================================================================================='''
    ''' When entering paths make sure they are in the same order as the name list'''
    model_names = ['TearsBase', 'TearsMVAE', 'TearsMacrid','TearsRecVAE']
    parser.add_argument('-l', '--model_list', help='delimited list of model_paths', type=str)
    parser.add_argument('-vl', '--vae_list', help='delimited list of model_paths', type=str)
    '''====================================================================================================='''
    
    args = parser.parse_args() if not notebook else parser.parse_args(args=[])

    print(f"{args.model_list=}")
    args.model_list = {name:item for name,item in zip(model_names,args.model_list.split(','))}
    args.vae_list = {name:item for name,item in zip(model_names[1:],args.vae_list.split(','))}
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    directory_path = "../scratch"
    
    #you can readjust this if the full evaluation set does not fit into memory usually it does
    args.bs = 1000 if args.data_name != 'ml-1m' else 250

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


def load_all_models(paths,vae_paths,rank):
    models = {}
    for module,path in paths.items():
        if path == '': continue
        if module != 'TearsBase':
            args.vae_path = vae_paths[module]
        p =  path +'.pt'
        args.embedding_module = module
        model = get_model(args,num_movies)
        model.to(rank)
        state_dict = torch.load(p, map_location=torch.device('cuda'))
        model.load_state_dict(state_dict)
        model.eval()
        models[module] = model
        print(f'{module} loaded')
    return models

if __name__ == "__main__":
    args = parse_args()
    tokenizer = get_tokenizer()
    # I recommend not using distributed here as this can call GPT/LLaMA multiple times and create errors
    rank = 0
    world_size = 1
    prompts, rec_dataloader, num_movies, val_dataloader, test_dataloader = load_data(
        args, tokenizer, rank, world_size)
    dataloader = val_dataloader if args.split == 'val' else test_dataloader
    #this assumes the eval data is loaded in as ingle batch
    for b in dataloader:
        uids = b['idx'].flatten().tolist()
        labels = torch.tensor(b['labels_tr']).to(0)
        labels_eval = b['labels'].numpy()

        
    labels_u_dict = {k: v.unsqueeze(0) for k, v in zip(uids, labels)}
    labels_eval_dict = {k:np.nonzero(v)[0] for k,v in zip(uids,labels_eval)}
    

    item_title_dict = map_id_to_title(args.data_name)
    item_genre_dict = map_id_to_genre(args.data_name)
    if 'large' in args.experiment or 'guided' in args.experiment:
    
        #for this we will set the model as the first one in the dict as we only need a single one.
        # Modify the input dict if you want to try different ones
        model = list(load_all_models(args.model_list,args.vae_list,rank).values())[1]

        large_change_evaluator = LargeScaleEvaluator(
            model, item_title_dict, item_genre_dict, tokenizer, 0, args, alpha=args.alpha, split='test')

        if args.experiment == 'large_prompt':
            large_change_evaluator.promptGPT(
                uids, prompts,labels_u_dict, seed=2024, save_every_iter= args.save)
        
        elif args.experiment == 'large_eval': 
            large_change_evaluator.evaluate(
                labels_u_dict, prompts, args.topk, rank=rank,save = args.save)
            
        elif args.experiment == 'guided':
            large_change_evaluator.alpha = 1
            large_change_evaluator.alpha2=.5
            
            #delta_up,genre
            large_change_evaluator.evaluate_genre(
                model, dataloader, args.topk, rank=rank, neg=False,save = args.save)
            #delta_down,genre
            large_change_evaluator.evaluate_genre(
                model, dataloader, args.topk, rank=rank, neg=True,save = args.save)
            
    elif 'small' in args.experiment:
        models_dict = load_all_models(args.model_list,args.vae_list,rank)
        evaluator = FineGrainedEvaluator(
            
            item_title_dict, item_genre_dict, tokenizer, rank, args, split='test',uids=uids,models_dict=models_dict)
        if args.experiment == 'small_prompt':
            evaluator.get_user_subset(uids, labels_u_dict,labels_eval_dict, prompts)
            evaluator.prompt_LLM_small(
                prompts, labels_u_dict, models_dict, save_every_iter=args.save)
        elif args.experiment == 'small_eval':
            evaluator.evaluate(
                prompts, labels_u_dict,  tokenizer,save = args.save)
        
    else:
        raise ValueError('Invalid experiment type')
    