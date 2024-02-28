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

class DataMatrix(Dataset):
    def __init__(self, train_matrix,encodings,indecis):
        self.train_matrix = naive_sparse2tensor(train_matrix)
        self.indices = indecis


        self.encodings = encodings
    def __len__(self):
        print(f"{ len(self.indices)=}")
        return len(self.indices)

    def __getitem__(self, idx):

        row = self.indices[idx]
        item = self.encodings[row]
        

        label_tensor = self.train_matrix[row]

        item['idx'] = torch.tensor([row])
        item['labels'] = label_tensor

        return  item



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

        rows, cols = tp['uid'], tp['sid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                 (rows, cols)), dtype='float64',
                                 shape=(self.n_users, self.n_items))
        return data
    
    def _load_tr_te_data(self, datatype='test'):
        tr_path = os.path.join(self.pro_dir, '{}_tr.csv'.format(datatype))
        te_path = os.path.join(self.pro_dir, '{}_te.csv'.format(datatype))

        tp_tr = pd.read_csv(tr_path)
        tp_te = pd.read_csv(te_path)
        
        rows_tr, cols_tr = tp_tr['uid'] , tp_tr['sid']
        rows_te, cols_te = tp_te['uid'], tp_te['sid']


        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                    (rows_tr, cols_tr)), dtype='float64', shape=(self.n_users, self.n_items))
        data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                    (rows_te, cols_te)), dtype='float64', shape=(self.n_users, self.n_items))

        return data_tr, data_te

def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())

   

def map_title_to_id(movies_file):
    data = pd.read_csv(movies_file,sep="::",names=["movieId","title","genre"],encoding='ISO-8859-1')
    mapping = {}
    for index, row in data.iterrows():
        mapping[row['title']] = row['movieId']
    return mapping

def map_id_to_title(movies_file):
    data = pd.read_csv(movies_file,sep="::",names=["movieId","title","genre"],encoding='ISO-8859-1')

    mapping = {}
    for index, row in data.iterrows():
        
        mapping[row['movieId']] = row['title']

    return mapping

def map_id_to_genre(movies_file= './data/ml-1m/movies.dat'):
    data = pd.read_csv(movies_file,sep="::",names=["movieId","title","genre"],encoding='ISO-8859-1')

    mapping = {}
    for index, row in data.iterrows():
        
        mapping[row['movieId']] = row['genre']

    return mapping


  