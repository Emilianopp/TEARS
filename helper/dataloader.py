import json
import ast

import pandas as pd 
import pickle
import numpy as np
from torch.nn.functional import one_hot
from scipy.sparse import csr_matrix
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
import random
import os 
from scipy import sparse

def drop_words(prompts, augmented_data):
    dropped_prompts = []

    for prompt in prompts:

        words = prompt.split()
        num_words = len(words)
        num_dropped = int(num_words * augmented_data)
        if num_dropped > 0:
            dropped_words = random.sample(words, num_dropped)
            dropped_prompt = ' '.join([word for word in words if word not in dropped_words])
            dropped_prompts.append(dropped_prompt)
        else:
            dropped_prompts.append(prompt)

    
    return dropped_prompts



# class DataMatrix(Dataset):
#     def __init__(self, train_matrix,encodings,indecis,tokenizer,prompts,mask):
#         self.train_matrix = naive_sparse2tensor(train_matrix)
#         self.indices = indecis
#         self.prompts = prompts
#         self.tokenizer =tokenizer
#         self.mask = mask


#         self.encodings = encodings
#     def __len__(self):

#         return len(self.indices)

#     def __getitem__(self, idx):

#         row = self.indices[idx]
#         if self.tokenizer is not None: 

#             agumented_prompts = drop_words([self.prompts[row]], self.mask)
#             item = self.tokenizer(agumented_prompts, add_special_tokens=True,return_tensors = 'pt')

#             item = self.tokenizer([self.prompts[row]], add_special_tokens=True,return_tensors = 'pt')

#         else:    
#             item = self.encodings[row]
        
#         label_tensor = self.train_matrix[row]

#         item['idx'] = torch.tensor([row])
#         item['labels'] = label_tensor

#         return  item


    


class DataMatrix(Dataset):
    def __init__(self, train_matrix,train_matrix_tr,encodings,indecis,tokenizer,prompts,mask,user_id_to_row=None):
        self.train_matrix = naive_sparse2tensor(train_matrix)

        self.train_matrix_tr = naive_sparse2tensor(train_matrix_tr)
        self.indices = indecis
        self.prompts = prompts
        self.tokenizer =tokenizer
        self.mask = mask
        self.encodings = encodings
        self.user_id_to_row = user_id_to_row
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = self.indices[idx]


        if self.tokenizer is not None: 

            agumented_prompts = drop_words([self.prompts[row]], self.mask)
            item = self.tokenizer(agumented_prompts, add_special_tokens=True,return_tensors = 'pt')

            item = self.tokenizer([self.prompts[row]], add_special_tokens=True,return_tensors = 'pt')

        else:    
            item = self.encodings[row]
        
        label_tensor = self.train_matrix[row]
        item['labels_tr']  = self.train_matrix_tr[row]
        if self.user_id_to_row is not None:
            item['idx'] = torch.tensor([self.user_id_to_row[row]])
        else:
            item['idx'] = torch.tensor([row])

        item['labels'] = label_tensor

        return  item


    def custom_collator(self,batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        idx = [item['idx'] for item in batch]

        # Determine the maximum length from all sequences
        max_length = max(ids.shape[1] for ids in input_ids)  # Assuming ids have shape [1, seq_length]

        # Function to pad tensors to the max_length, handling 2D tensors
        def pad_tensor(tensor, length):

            return torch.cat([tensor, torch.tensor([self.tokenizer.pad_token_id] * (length - tensor.shape[1])).unsqueeze(0)], dim=1).long()
            
   

        # Pad input_ids and attention_mask, ensuring they are 2D tensors of shape [1, max_length]
        padded_input_ids = torch.stack([pad_tensor(ids, max_length) for ids in input_ids], dim=0).squeeze(1)
        padded_attention_mask = torch.stack([pad_tensor(mask, max_length) for mask in attention_mask], dim=0).squeeze(1)

        # Handling labels and idx, assuming they are already appropriately shaped or scalar values
        labels_tensor = torch.tensor(labels, dtype=torch.long) if not all(isinstance(l, torch.Tensor) for l in labels) else torch.stack(labels)
        idx_tensor = torch.tensor(idx, dtype=torch.long) if not all(isinstance(i, torch.Tensor) for i in idx) else torch.stack(idx)

        return {
            'input_ids': padded_input_ids,
            'attention_mask': padded_attention_mask,
            'labels': labels_tensor,
            'idx': idx_tensor
        }
        
    def custom_collator(self,batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        idx = [item['idx'] for item in batch]

        # Determine the maximum length from all sequences
        max_length = max(ids.shape[1] for ids in input_ids)  # Assuming ids have shape [1, seq_length]

        # Function to pad tensors to the max_length, handling 2D tensors
        def pad_tensor(tensor, length):

            return torch.cat([tensor, torch.tensor([self.tokenizer.pad_token_id] * (length - tensor.shape[1])).unsqueeze(0)], dim=1).long()
            
   

        # Pad input_ids and attention_mask, ensuring they are 2D tensors of shape [1, max_length]
        padded_input_ids = torch.stack([pad_tensor(ids, max_length) for ids in input_ids], dim=0).squeeze(1)
        padded_attention_mask = torch.stack([pad_tensor(mask, max_length) for mask in attention_mask], dim=0).squeeze(1)

        # Handling labels and idx, assuming they are already appropriately shaped or scalar values
        labels_tensor = torch.tensor(labels, dtype=torch.long) if not all(isinstance(l, torch.Tensor) for l in labels) else torch.stack(labels)
        idx_tensor = torch.tensor(idx, dtype=torch.long) if not all(isinstance(i, torch.Tensor) for i in idx) else torch.stack(idx)

        return {
            'input_ids': padded_input_ids,
            'attention_mask': padded_attention_mask,
            'labels': labels_tensor,
            'idx': idx_tensor
        }



# class MatrixDataLoader():
#     '''
#     Load Movielens-20m dataset
#     '''
#     def __init__(self, path,args):
#         self.pro_dir = path
#         self.args = args
#         assert os.path.exists(self.pro_dir), "Preprocessed files does not exist. Run data.py"

#         self.n_items = self.load_n_items()
    
#     def load_data(self, datatype='train'):
#         if datatype == 'train':
#             return self._load_train_data()
#         elif datatype == 'validation':
#             return self._load_tr_te_data(datatype)
#         elif datatype == 'test':
#             return self._load_tr_te_data(datatype)
#         else:
#             raise ValueError("datatype should be in [train, validation, test]")
        
#     def load_n_items(self):
#         unique_sid = list()
#         with open(os.path.join(self.pro_dir, 'unique_sid.txt'), 'r') as f:
#             for line in f:
#                 unique_sid.append(line.strip())
#         n_items = len(unique_sid)
#         return n_items
    
#     def _load_train_data(self):
#         path = os.path.join(self.pro_dir, 'train.csv')        
#         tr_path = os.path.join(self.pro_dir, '{}_tr.csv'.format('train'))
#         te_path = os.path.join(self.pro_dir, '{}_te.csv'.format('train'))
        

        
#         tr = pd.read_csv(tr_path)
#         tr_grouped = tr.groupby('uid').count()

#         #get max count 
#         max_count = tr_grouped.sid.max()
#         assert max_count <= 50
#         te = pd.read_csv(te_path)
#         te = te[te['rating'] > 3]
        
#         print(f"{te=}")
#         print(f"{te_path=}")

#         self.n_users = int(tr['uid'].max() + 1)


#         rows, cols,ratings= tr['uid'], tr['sid'],tr['rating']

#         data = sparse.csr_matrix((ratings,
#                                  (rows, cols)), dtype='float64',
#                                  shape=(self.n_users, self.n_items))
        
#         rows, cols,ratings= te['uid'], te['sid'],te['rating']
#         data_te = sparse.csr_matrix((np.ones_like(rows),
#                                  (rows, cols)), dtype='float64',
#                                  shape=(self.n_users, self.n_items))

#         #binarize the data_te to be 1 if the rating is higher than 3 and 0 otherwise

        
#         #assert that all rows are in data_te are bigger smaller than 1 
#         assert data_te.max() <= 1,data_te.max()
#         print('Number of users: {}, Number of items: {}'.format(self.n_users, self.n_items))
#         data_te = data_te[data_te.sum(axis = 1).nonzero()[0]]
#         data = data[data.sum(axis = 1).nonzero()[0]]
#         #assert in each row of data ther are at most 50 nonzero elements
#         for row in data: 
#             assert len(row.nonzero()[1]) <= 50,len(row.nonzero()[1])
#         return data_te,data
        
#     def _load_tr_te_data(self, datatype='test'):
#         promt_dataset = pd.read_csv(f'./data_preprocessed/{self.args.data_name}/prompt_set_timestamped.csv')
#         with open(os.path.join(self.pro_dir, 'show2id.pkl'), 'rb') as f:
#             show2id = pickle.load(f)
#             #reverse 
#             id2show = {v:k for k,v in show2id.items()}
#         with open(os.path.join(self.pro_dir,  'profile2id.pkl'), 'rb') as f:
#             profile2id = pickle.load(f)
#             #reverse
#             id2profile = {v:k for k,v in profile2id.items()}
            
        
#         tr_path = os.path.join(self.pro_dir, '{}_tr.csv'.format(datatype))
#         te_path = os.path.join(self.pro_dir, '{}_te.csv'.format(datatype))

#         tp_tr = pd.read_csv(tr_path)
#         tp_te = pd.read_csv(te_path)
#         print(f"{tp_te=}")
#         print(f"{tp_te=}")
#         #make sure all the users in tr are in te 
#         tp_te = tp_te.loc[tp_te['uid'].isin(tp_tr['uid'])]
#         #and viceversa
#         tp_tr = tp_tr.loc[tp_tr['uid'].isin(tp_te['uid'])]

#         start_idx = int(min(tp_tr['uid'].min(), tp_te['uid'].min()))
#         end_idx = int(max(tp_tr['uid'].max(), tp_te['uid'].max()))

#         rows_tr, cols_tr,rating_tr = tp_tr['uid'] - start_idx, tp_tr['sid'],tp_tr['rating']
#         rows_te, cols_te,rating_te = tp_te['uid'] - start_idx, tp_te['sid'],tp_te['rating']

#         data_tr = sparse.csr_matrix((rating_tr,
#                                     (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, self.n_items))
#         # no need to make these the ratings as they are used for evaluation anyways
#         data_te = sparse.csr_matrix((np.ones_like(rows_te),
#                                     (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, self.n_items))
#         #make a map of user_id to row index
#         #remove nonzero rows 
#         user_id_to_row = {row_idx: user_id for row_idx, user_id in enumerate(tp_tr['uid'].unique())}
#         data_te = data_te[data_te.sum(axis = 1).nonzero()[0]]
#         data_tr = data_tr[data_tr.sum(axis = 1).nonzero()[0]]
        
#         #assert the have the same dimensions 
#         assert data_te.shape == data_tr.shape
#         #assert data_tr has 50 nonzero rows for each user
        

#         #assert all users in the user_id_row are in the prompt_dataset 
#         for user_idx in user_id_to_row.values():
#             user_id = id2profile[user_idx]
#             if user_id not in promt_dataset.userId.unique():
#                 print(f"{user_id=}")
#                 print(f"{promt_dataset.userId.unique()=}")
#                 raise AssertionError("Not all users in the user_id_row are in the prompt_dataset")

#         #assert that tr only has 50 movies 
#         if datatype != 'train':
#             for row_idx in range(data_tr.shape[0]):
#                 if len(data_tr[row_idx].nonzero()[1]) > 50:
#                     print(f"{len(data_tr[row_idx].nonzero()[1])=}")
#                     print(f"{data_tr[row_idx].nonzero()[1]=}")
#                     print(f"{row_idx=}")
#                     raise ValueError("data_tr has more than 50 nonzero rows for a user")
            

#         return  data_tr,data_te,user_id_to_row


class MatrixDataLoader():
    '''
    Load Movielens-20m dataset
    '''
    def __init__(self, path,args):
        self.pro_dir = path
        self.args = args

        assert os.path.exists(self.pro_dir), "Preprocessed files does not exist. Run data.py"

        self.n_items = self.load_n_items()
    
    def load_data(self, datatype='train',head = None):
        if datatype == 'train':
            return self._load_train_data(head = head)
        elif datatype == 'validation':
            return self._load_tr_te_data(datatype,head = head)
        elif datatype == 'test':
            return self._load_tr_te_data(datatype,head=head)
        else:
            raise ValueError("datatype should be in [train, validation, test]")
        
    def load_n_items(self):
        unique_sid = list()
        with open(os.path.join(self.pro_dir, 'unique_sid.txt'), 'r') as f:
            for line in f:
                unique_sid.append(line.strip())
        n_items = len(unique_sid)
        return n_items
    
    def _load_train_data(self,head = 50):
        if head is not None:
            path = os.path.join(self.pro_dir, 'train.csv')
        else: 
            path = os.path.join(self.pro_dir, 'train.csv')
            
        tp = pd.read_csv(path)
        self.n_users = tp['uid'].max() + 1

        
        tp_tr = tp.groupby('uid').apply(lambda x: x.head(head)).reset_index(drop=True)
        rows, cols,rating = tp_tr['uid'], tp_tr['sid'],tp_tr['rating']
        data_tr = sparse.csr_matrix((np.ones_like(rows) if self.args.binarize else rating,
                                 (rows, cols)), dtype='float64',
                                 shape=(self.n_users, self.n_items))
        
        tp = tp[tp['rating'] > 3]   
        rows, cols = tp['uid'], tp['sid']
        data_te = sparse.csr_matrix((np.ones_like(rows),
                                 (rows, cols)), dtype='float64',
                                 shape=(self.n_users, self.n_items))

     

        return data_tr,data_te
    
    def _load_tr_te_data(self, datatype='test',head = None):
        tr_path = os.path.join(self.pro_dir, '{}_tr.csv'.format(datatype))
        te_path = os.path.join(self.pro_dir, '{}_te.csv'.format(datatype))
        tp_tr = pd.read_csv(tr_path)
        tp_te = pd.read_csv(te_path)

        if head is not None:
            tp_tr = pd.concat([tp_tr,tp_te])
            # print(f"{tp_tr.groupby('uid').size()=}")
            # print(f"{tp_tr=}")
            grouped_test = tp_tr.groupby('uid').count()
            user_set = grouped_test[grouped_test['sid'] > 100].index
            tp_tr = tp_tr[tp_tr.uid.isin(user_set)]
            tp_tr_head = tp_tr.groupby('uid').apply(lambda x: x.head(100)).reset_index(drop=True)

            # Step 2: Find the indices of the rows that are not in the top 100 for each group and update tp_te
            # First, we'll get the indices of tp_tr_head to exclude them from tp_tr
            indices_to_exclude = tp_tr_head.index

            # Now, select rows in tp_tr that are not in tp_tr_head
            tp_te = tp_tr.loc[~tp_tr.index.isin(indices_to_exclude)]
            # print(f"{tp_te.groupby('uid').size()=}")
            

            # Update tp_tr to be just the head

            tp_tr = tp_tr.groupby('uid').apply(lambda x: x.head(head)).reset_index(drop=True)
            # print(f"{head=}")
            tp_tr = tp_tr[tp_tr.uid.isin(tp_te.uid.unique())]
            # print(f"{tp_tr.groupby('uid').size()=}")
            # print(f"{len(tp_tr.uid.unique())=}")
            # print(f"{len(tp_te.uid.unique())=}")




        rows_tr, cols_tr,rating_tr = tp_tr['uid'] , tp_tr['sid'], tp_tr['rating']

        rows_te, cols_te,rating_te = tp_te['uid'], tp_te['sid'], tp_te['rating']
 


        data_tr = sparse.csr_matrix((np.ones_like(rows_tr) if self.args.binarize else rating_tr,
                                    (rows_tr, cols_tr)), dtype='float64', shape=(self.n_users, self.n_items))
        data_te = sparse.csr_matrix((np.ones_like(rows_te) ,
                                    (rows_te, cols_te)), dtype='float64', shape=(self.n_users, self.n_items))
        # data_tr = data_tr[data_tr.sum(axis=1).nonzero()[0]]
        # print(f"{(data_tr>0).sum(axis =1 )=}")
        # print(f"{data_tr.sum(axis =1 ).shape=}")
        return data_tr, data_te

def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())

   

def map_title_to_id(movies_file):
    data = pd.read_csv(movies_file,sep="::",names=["movieId","title","genre"],encoding='ISO-8859-1')
    mapping = {}
    for index, row in data.iterrows():
        mapping[row['title']] = row['movieId']
    return mapping

def map_id_to_title(data = 'ml-1m'):
    if data == 'ml-1m':
        
        data = pd.read_csv('./data/ml-1m/movies.dat',sep="::",names=["itemId","title","genre"],encoding='ISO-8859-1')
        with open('/home/mila/e/emiliano.penaloza/LLM4REC/data_preprocessed/ml-1m/show2id.pkl','rb') as f :
            item_id_map = pickle.load(f)
            #reverse the mapping 
            # item_id_map = {v:k for k,v in item_id_map.items()}
        mapping = {}
        
        
        for index, row in data.iterrows():
            #item may have been removed at post processing so check if it was
            if row['itemId'] in item_id_map:
            
                mapping[item_id_map[row['itemId']]] = row['title']
        print(f"{len(mapping)=}")

        return mapping
    elif data == 'netflix':
        with open('/home/mila/e/emiliano.penaloza/LLM4REC/data_preprocessed/netflix/id_title_map.pkl','rb') as f :
            item_id_map = pickle.load(f)
        return item_id_map

    elif data =='goodbooks': 
        df = pd.read_csv('/home/mila/e/emiliano.penaloza/LLM4REC/data/goodbooks/genres.csv')
        df.genres = df.genres.apply(ast.literal_eval)

        mapping = {}
        with open('/home/mila/e/emiliano.penaloza/LLM4REC/data_preprocessed/goodbooks/show2id.pkl','rb') as f :
            item_id_map = pickle.load(f)
        for index, row in df.iterrows():
            if row['book_id'] in item_id_map:
                    mapping[item_id_map[row['book_id']]] = row['title']



    return mapping
        
        
        

def map_id_to_genre(data='ml-1m'):
    if data == 'ml-1m':
        data = pd.read_csv('./data/ml-1m/movies.dat',sep="::",names=["itemId","title","genre"],encoding='ISO-8859-1')
        with open('/home/mila/e/emiliano.penaloza/LLM4REC/data_preprocessed/ml-1m/show2id.pkl','rb') as f :
            item_id_map = pickle.load(f)
        print(f"{len(item_id_map)=}")

        mapping = {}
        for index, row in data.iterrows():
            #item may have been removed at post processing so check if it was
            if row['itemId'] in item_id_map:

                genres = row['genre'].lower().replace('-', ' ').split('|')
                mapping[item_id_map[row['itemId']]] = genres


    elif data=='netflix':
        
        with open('/home/mila/e/emiliano.penaloza/LLM4REC/data_preprocessed/netflix/show2id.pkl','rb') as f :
            item_id_map = pickle.load(f)


        df = pd.read_csv('/home/mila/e/emiliano.penaloza/LLM4REC/data/netflix/netflix_genres.csv')

        mapping = {}

        for index, row in df.iterrows():
            if row['movieId'] in item_id_map:
                mapping[item_id_map[row['movieId']]] = row['genres'].lower().replace('-', ' ').split('|')
        
    elif data =='goodbooks': 
        df = pd.read_csv('/home/mila/e/emiliano.penaloza/LLM4REC/data/goodbooks/genres.csv')
        df.genres = df.genres.apply(ast.literal_eval)

        mapping = {}
        with open('/home/mila/e/emiliano.penaloza/LLM4REC/data_preprocessed/goodbooks/show2id.pkl','rb') as f :
            item_id_map = pickle.load(f)
        for index, row in df.iterrows():
            if row['book_id'] in item_id_map:
                    mapping[item_id_map[row['book_id']]] = row['genres']



    return mapping
  