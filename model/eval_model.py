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
class LargeScaleEvaluator(nn.Module):
    def __init__(self,model,item_title_dict,item_genre_dict,tokenizer,rank,args,alpha = .5,split='test'):
        super().__init__()
        self.model = model
        self.item_title_dict = item_title_dict
        self.item_genre_dict = item_genre_dict
        self.tokenizer = tokenizer
        self.rank = rank
        counts = Counter(sum([v for v in item_genre_dict.values()],[]))

        #keep counts if above 200 
        self.counts = {k:v for k,v in counts.items() if v > 100}
        self.genre_list = list(self.counts.keys())
        # if args.data_name == 'goodbooks':
        self.genre_list = [x.lower().replace('-', ' ') for x in self.genre_list]
        
        self.genre_set = ', '.join(self.genre_list)
        self.args = args
        if os.path.exists(f'./results/{self.args.data_name}/gpt4_results_large_genre_{split}.csv'):
            self.df = pd.read_csv(f'./results/{self.args.data_name}/gpt4_results_large_genre_{split}.csv')


        self.alpha = alpha 
        self.alpha2 = None
    def set_alpha2(self,alpha2):
        self.alpha2 = alpha2   
    def getGenreDelta(self, s1,s2,labels,topk,genre_1 , genre_2,rank ,neg=False):
        # print(f"{s1=}")
        # print(f"{s2=}")
        if self.args.mask_control_labels:
        # if True:
            labels = torch.zeros_like(labels)
        # labels = torch.zeros_like(labels)
        # print(f"{labels.sum()=}")
        if  self.args.embedding_module not in  ['RecVAEGenreVAE','GenreTEARS']:
            if isinstance(self.model, DistributedDataParallel):
                topk1 = self.model.module.generate_recommendations(s1, self.tokenizer, labels, topk, rank,alpha = self.alpha,neg = neg)
                topk2 = self.model.module.generate_recommendations(s2, self.tokenizer, labels, topk, rank,alpha = self.alpha if self.alpha2 is not None else self.alpha2,neg = neg)

            else:
                topk1 = self.model.generate_recommendations(s1, self.tokenizer, labels, topk, rank,alpha = self.alpha,neg =neg)
                
                topk2 = self.model.generate_recommendations(s2, self.tokenizer, labels, topk, rank,alpha = self.alpha if self.alpha2 is not None else self.alpha2,neg = neg)
        else: 
            if isinstance(self.model, DistributedDataParallel):
                topk1 = self.model.module.generate_recommendations(  topk =topk, rank=rank,alpha = self.alpha,neg = neg)
                topk2 = self.model.module.generate_recommendations(mask_genre = genre_2, topk = topk, rank =rank,alpha = self.alpha if self.alpha2 is not None else self.alpha2,neg = neg)

            else:
                topk1 = self.model.generate_recommendations( data_tensor = labels,topk =topk, rank=rank,alpha = self.alpha,neg = neg)

                
                topk2 = self.model.generate_recommendations(data_tensor = labels,mask_genre = genre_2, topk = topk, rank =rank,alpha = self.alpha if self.alpha2 is not None else self.alpha2,neg = neg)


        movie_titles1 = [self.item_title_dict[i] for i in topk1]
        movie_titles2 = [self.item_title_dict[i] for i in topk2]
        # print(f"{movie_titles1=}")
        # print(f"{movie_titles2=}")

        movie_genres1 = [self.item_genre_dict[i] for i in topk1]
        movie_genres2 = [self.item_genre_dict[i] for i in topk2]
        
        change_down = self.genrewise_ndcg(movie_genres1,genre_1,min_k = 0,max_k = topk) - self.genrewise_ndcg(movie_genres2,genre_1,min_k = 0,max_k = topk)
        # print(f"{genre_1=}")
        # print(f"{self.genrewise_ndcg(movie_genres1,genre_1,min_k = 0,max_k = topk)=}")
        # print(f"{change_down=}")
        # print(f"{self.genrewise_ndcg(movie_genres2,genre_1,min_k = 0,max_k = topk)=}")
        change_up =  self.genrewise_ndcg(movie_genres1,genre_2,min_k = 0,max_k = topk) - self.genrewise_ndcg(movie_genres2,genre_2,min_k = 0,max_k = topk)
        # print(f"{genre_2=}")
        # print(f"{change_up=}")
        # raise Exception
        
        return change_down,change_up,movie_titles1,movie_titles2
    def genrewise_ndcg(self,genre_movies, genre, min_k=0, max_k=None):
        
        relevance = [1 if genre in g   else 0 for g in genre_movies[min_k:max_k]]
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
        idcg = sum(1 / np.log2(i + 2) for i in range(len(relevance)))
        ndcg = dcg / idcg if idcg > 0 else 0
        
        return ndcg
    
    def print_ranking_diffences(self,titles1,titles2): 
        for i,(title1, title2) in enumerate(zip(titles1, titles2)):
            print(f"{i} {title1:<80} {title2}")

    def promptGPT(self,k_sample, prompts, labels, seed =2024,split='val'):
    
        np.random.seed(seed)
        # self.model.eval()
        load_dotenv()

        self.genre_list = [x.lower().replace('-', ' ') for x in self.genre_list]

        print(f"{self.genre_list=}")
        c = 0
        openai.api_key  = os.getenv("OPEN-AI-SECRET")
        move_up_genres =[]
        move_down_genres = []
        outputs = []
        keys = []
        deltas_ndcg_up = []
        deltas_ndcg_down = []
        deltas_ndcg_up = []
        deltas_ndcg_down = []
        for k in (pbar:=tqdm(k_sample)): 
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
                        {
                            "role": "system",
                            "content": prompt

                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ]

            genres = openai.ChatCompletion.create(
                            model='gpt-4o',
                            # model='gpt-4o',
                            messages=msg,
                            max_tokens=300,
                            temperature=0.001,
                            seed=2024,
                        )['choices'][0]['message']['content']
            lines = genres.split('\n')
            try:
                favorite_genre = lines[0].split(': ')[1].lower().replace('-', ' ')
                least_favorite_genre = lines[1].split(': ')[1].lower().replace('-', ' ')
            except:
                continue
            if favorite_genre not in self.genre_list or least_favorite_genre not in self.genre_list:
                c+= 1
                pbar.set_description(f"pass {c} {favorite_genre} {least_favorite_genre}")    
                
                continue
            
            msg = [
                        {
                            "role": "system",
                            "content": prompt

                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        },
                        {'role': 'assistant',
                        'content': genres},
                        {'role':'user',
                        'content':
                        f'Now using this setup write the a new summary in the same style that reflects that {favorite_genre} is your least favorite\
                            and {least_favorite_genre} is your favorite only output the full summary keep the format and length the same' }
                    ]

            gpt_output = openai.ChatCompletion.create(
                            model='gpt-4-1106-preview',
                            # model='gpt-4o',
                            messages=msg,
                            max_tokens=300,
                            temperature=0.001,
                            seed=2024,
                        )['choices'][0]['message']['content']


            # delta1,delta2,movies1,movies2 = self.getGenreDelta(s,gpt_output,labs,20,favorite_genre,least_favorite_genre,rank = self.rank)
            move_down_genres.append(favorite_genre)
            move_up_genres.append(least_favorite_genre)
            outputs.append(gpt_output)  
            keys.append(k)
            deltas_ndcg_down.append(0)
            deltas_ndcg_up.append(0)

            
            pbar.set_description(f"gen {favorite_genre}, average ndcgs up {np.mean(deltas_ndcg_up)} average ndcg down {np.mean(deltas_ndcg_down)}")    

            

            data = {
                'move_down_genres': move_down_genres,
                'move_up_genres': move_up_genres,
                'outputs': outputs,
                'keys': keys,
                'deltas_ndcg_down': deltas_ndcg_down,
                'deltas_ndcg_up': deltas_ndcg_up,
            }


        df = pd.DataFrame(data)
        df.to_csv(f'./results/{self.args.data_name}/gpt4_results_large_genre_{split}.csv')
        print('WROTE DF')
        self.df = df
            

    
    def evaluate(self,dataloader,prompts,topk,rank = None,max_delta = False):
        self.model.eval()



        # df = pd.read_csv(f'./results/{self.args.data_name}/gpt4_results_large_genre.csv')
        self.delta_up = []
        self.delta_down = []
        # print(self.model.device)
        # print(rank)
        # exit()
        c = 0

        for b in (pbar:=tqdm(dataloader,desc = 'Controlability')):
            uids = b['idx'].flatten().tolist()
            #if rank is not set set the rank
            labels = b['labels_tr'].to(rank)
            self.least_fav = []
            self.most_fav = []   
            

            for uid,label in zip(uids,labels):
                if uid in self.df['keys'].values:

                    
                    sub_df = self.df[self.df['keys'] == uid]
                    gpt_output = sub_df['outputs'].values[0]
                    genre1 = sub_df['move_down_genres'].values[0].lower().replace('-', ' ')
                    genre2 = sub_df['move_up_genres'].values[0].lower().replace('-', ' ')
                    self.least_fav.append(genre2)
                    self.most_fav.append(genre1)
                        
                        
                    down,up,movies1,movies2 = self.getGenreDelta(prompts[uid],gpt_output,label,topk,genre1 ,genre2 ,rank =rank) if not max_delta else self.getMaxDelta(prompts[uid],gpt_output,label,topk,genre1 ,genre2 ,rank =rank)
                    # print(f"{up=}")
                    # print(f"{down=}")

                    
                    # print(f"{up=}")
                    # print(f"{down=}")
                    self.delta_up.append(up)
                    # print(f"{self.delta_up=}")
                    self.delta_down.append(down)
               


            print(f"average ndcgs up {np.mean(self.delta_up)} average ndcg down {np.mean(self.delta_down)}")
            pbar.set_description(f"average ndcgs up {np.mean(self.delta_up)} average ndcg down {np.mean(self.delta_down)}")

        # raise Exception
        return np.mean(self.delta_up),np.mean(self.delta_down)
    
    def evaluate_genre(self,model,dataloader,prompts,topk,rank = None,dir = 'more',neg = False,return_arr = False,genre_changed = None):
            model.eval()
            # print(self.genre_set)
            # raise Exception
            if genre_changed is not None: 
                genres = [genre_changed]
            else:
                genres = self.counts.keys()
            # df = pd.read_csv(f'./results/{self.args.data_name}/gpt4_results_large_genre.csv')
            self.delta_up = []
            self.delta_down = []
            # print(self.model.device)
            # print(rank)
            # exit()
            
            
            
            for b in (pbar:=tqdm(dataloader,desc = 'Controlability')):
                uids = b['idx'].flatten().tolist()
                #if rank is not set set the rank
                labels = b['labels_tr'].to(rank)
                mean_per_user = []
                out_metrics = []
                for uid,label in zip(uids,labels):

                    for genre in genres:
                        rep_s = f' Summary: {dir} {genre} films'
                        down,up,movies1,movies2 = self.getGenreDelta('',rep_s,label,topk,genre ,'Action' ,rank =rank,neg=neg)


                        self.delta_up.append(up)
                        # print(f"{delta_up=}")
                        self.delta_down.append(down)
                        

                        mean_per_user.append(down)
                    out_metrics.append(np.mean(mean_per_user))

                    pbar.set_description(f"average ndcgs up {np.median(self.delta_up)} average ndcg down {np.mean(self.delta_down)}")
            if return_arr:
                
                return out_metrics,out_metrics
            else: 
                return np.mean(self.delta_up),np.mean(self.delta_down)
                        

            
    def getMaxDelta(self, s1,s2,labels,topk,genre_1 , genre_2,rank ,neg=False):

        # print(f"{s1=}")
        # print(f"{s2=}")
        if self.args.mask_control_labels:
        # if True:
            labels = torch.zeros_like(labels)
        # labels = torch.zeros_like(labels)
        # print(f"{labels.sum()=}")
        if  self.args.embedding_module not in  ['RecVAEGenreVAE','GenreTEARS']:
            if isinstance(self.model, DistributedDataParallel):
                topk1 = self.model.module.generate_recommendations(s1, self.tokenizer, labels, topk, rank,alpha = self.alpha,neg = neg)
                topk2 = self.model.module.generate_recommendations(s2, self.tokenizer, labels, topk, rank,alpha = self.alpha if self.alpha2 is not None else self.alpha2,neg = neg)

            else:
                topk1 = self.model.generate_recommendations(s1, self.tokenizer, labels, topk, rank,alpha = self.alpha,neg =neg)
                
                topk2 = self.model.generate_recommendations(s2, self.tokenizer, labels, topk, rank,alpha = self.alpha if self.alpha2 is not None else self.alpha2,neg = neg)
        else: 
            if isinstance(self.model, DistributedDataParallel):
                topk1 = self.model.module.generate_recommendations(  topk =topk, rank=rank,alpha = self.alpha,neg = neg)
                topk2 = self.model.module.generate_recommendations(mask_genre = genre_2, topk = topk, rank =rank,alpha = self.alpha if self.alpha2 is not None else self.alpha2,neg = neg)

            else:
                topk1 = self.model.generate_recommendations( data_tensor = labels,topk =topk, rank=rank,alpha = self.alpha,neg = neg)

                
                topk2 = self.model.generate_recommendations(data_tensor = labels,mask_genre = genre_2, topk = topk, rank =rank,alpha = self.alpha if self.alpha2 is not None else self.alpha2,neg = neg)


        movie_titles1 = [self.item_title_dict[i] for i in topk1]
        movie_titles2 = [self.item_title_dict[i] for i in topk2]
        # print(f"{movie_titles1=}")
        # print(f"{movie_titles2=}")

        movie_genres1 = [self.item_genre_dict[i] for i in topk1]
        movie_genres2 = [self.item_genre_dict[i] for i in topk2]
        
        change_down = self.genrewise_ndcg(movie_genres1,genre_1,min_k = 0,max_k = topk) - 0
        # print(f"{genre_1=}")
        # print(f"{self.genrewise_ndcg(movie_genres1,genre_1,min_k = 0,max_k = topk)=}")
        # print(f"{change_down=}")
        # print(f"{self.genrewise_ndcg(movie_genres2,genre_1,min_k = 0,max_k = topk)=}")
        change_up =  self.genrewise_ndcg(movie_genres1,genre_2,min_k = 0,max_k = topk) - 1
        # print(f"{genre_2=}")
        # print(f"{change_up=}")
        # raise Exception
        
        return change_down,change_up,movie_titles1,movie_titles2

        
    def get_qualitative_metrics(self,):
        return {'fav_genres': self.most_fav, 'least_fav_genres': self.least_fav,'down': self.delta_down, 'up': self.delta_up}