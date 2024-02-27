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


  