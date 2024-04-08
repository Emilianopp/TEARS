import json
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



class DataMatrix(Dataset):
    def __init__(self, train_matrix,encodings,indecis,tokenizer,prompts,mask):
        self.train_matrix = naive_sparse2tensor(train_matrix)
        self.indices = indecis
        self.prompts = prompts
        self.tokenizer =tokenizer
        self.mask = mask


        self.encodings = encodings
    def __len__(self):

        return len(self.indices)

    def __getitem__(self, idx):

        row = self.indices[idx]
        if self.tokenizer is not None: 

            agumented_prompts = drop_words([self.prompts[row]], self.mask)
            item = self.tokenizer(agumented_prompts, add_special_tokens=True,return_tensors = 'pt')

            original = self.tokenizer([self.prompts[row]], add_special_tokens=True,return_tensors = 'pt')

            item['original_input_ids'] = original['input_ids']
            item['original_attention_mask'] = original['attention_mask']
        else:    
            item = self.encodings[row]
        
        label_tensor = self.train_matrix[row]

        item['idx'] = torch.tensor([row])
        item['labels'] = label_tensor

        return  item


def custom_collator(batch):
    input_ids = [item['input_ids'] for item in batch]

 
    
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    idx = [item['idx'] for item in batch]
    if 'original_input_ids' in batch[0]:
        original_input_ids = [item['original_input_ids'] for item in batch]

        original_mask = [item['original_attention_mask'] for item in batch]
        

        input_ids = input_ids   + original_input_ids
        attention_mask = attention_mask + original_mask
        labels = labels + labels
        idx = idx+ idx



    # Determine the maximum length from all sequences
    max_length = max(ids.shape[1] for ids in input_ids)  # Assuming ids have shape [1, seq_length]

    # Function to pad tensors to the max_length, handling 2D tensors
    def pad_tensor(tensor, length):
        # Calculate padding length
        pad_length = length - tensor.shape[1]
        # Pad the tensor to the right along the sequence length dimension
        return torch.cat([tensor, torch.zeros(1, pad_length, dtype=tensor.dtype)], dim=1)

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





class MatrixDataLoader():
    '''
    Load Movielens-20m dataset
    '''
    def __init__(self, path):
        self.pro_dir = path
        assert os.path.exists(self.pro_dir), "Preprocessed files does not exist. Run data.py"

        self.n_items = self.load_n_items()
    
    def load_data(self, datatype='train'):
        if datatype == 'train':
            return self._load_train_data()
        elif datatype == 'validation':
            return self._load_tr_te_data(datatype)
        elif datatype == 'test':
            return self._load_tr_te_data(datatype)
        else:
            raise ValueError("datatype should be in [train, validation, test]")
        
    def load_n_items(self):
        unique_sid = list()
        with open(os.path.join(self.pro_dir, 'unique_sid.txt'), 'r') as f:
            for line in f:
                unique_sid.append(line.strip())
        n_items = len(unique_sid)
        return n_items
    
    def _load_train_data(self):
        path = os.path.join(self.pro_dir, 'train.csv')

        
        tp = pd.read_csv(path)

        self.n_users = tp['uid'].max() + 1

        rows, cols,rating  = tp['uid'], tp['sid'],tp['rating']
    
        non_binary_data = sparse.csr_matrix((rating,
                                 (rows, cols)), dtype='float64',
                                 shape=(self.n_users, self.n_items))
        

        tp = tp[tp['rating'] > 3]   
        #get the most recent 50 items for each user
        #for each user sample 50 random movies 
        
        # tp = tp.groupby('uid').apply(lambda x: x.head(50)).reset_index(drop=True)

        

        rows, cols = tp['uid'], tp['sid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                 (rows, cols)), dtype='float64',
                                 shape=(self.n_users, self.n_items))
        return data,non_binary_data
    
    def _load_tr_te_data(self, datatype='test'):
        tr_path = os.path.join(self.pro_dir, '{}_tr.csv'.format(datatype))
        te_path = os.path.join(self.pro_dir, '{}_te.csv'.format(datatype))

        tp_tr = pd.read_csv(tr_path)
        tp_te = pd.read_csv(te_path)
        
        rows_tr, cols_tr,rating_tr = tp_tr['uid'] , tp_tr['sid'], tp_tr['rating']
        rows_te, cols_te,rating_te = tp_te['uid'], tp_te['sid'], tp_te['rating']


        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                    (rows_tr, cols_tr)), dtype='float64', shape=(self.n_users, self.n_items))
        data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                    (rows_te, cols_te)), dtype='float64', shape=(self.n_users, self.n_items))
        data_tr_rating = sparse.csr_matrix((rating_tr,
                                    (rows_tr, cols_tr)), dtype='float64', shape=(self.n_users, self.n_items))
        data_te_rating = sparse.csr_matrix((rating_te,
                                    (rows_te, cols_te)), dtype='float64', shape=(self.n_users, self.n_items))
        
        return data_tr, data_te,data_tr_rating,data_te_rating

def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())

   

def map_title_to_id(movies_file):
    data = pd.read_csv(movies_file,sep="::",names=["movieId","title","genre"],encoding='ISO-8859-1')
    mapping = {}
    for index, row in data.iterrows():
        mapping[row['title']] = row['movieId']
    return mapping

def map_id_to_title(data_file,data = 'ml-1m'):
    if data == 'ml-1m':
        data = pd.read_csv(data_file,sep="::",names=["itemId","title","genre"],encoding='ISO-8859-1')

        mapping = {}
        for index, row in data.iterrows():
            
            mapping[row['itemId']] = row['title']

        return mapping
    elif data == 'books': 
        df = pd.read_csv(data_file)
        return df.set_index('sid').to_dict()['title']
        
        
        

def map_id_to_genre(path= './data/ml-1m/movies.dat',data='ml-1m'):
    if data == 'ml-1m':
        data = pd.read_csv(path,sep="::",names=["itemId","title","genre"],encoding='ISO-8859-1')

        mapping = {}
        for index, row in data.iterrows():
            
            mapping[row['itemId']] = row['genre']

        return mapping
    elif data =='books': 
        df = pd.read_csv(path)
        return df.set_index('sid').to_dict()['genres']


  