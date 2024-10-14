import ast
import pandas as pd
import csv
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from scipy import sparse



class DataMatrix(Dataset):
    def __init__(self, train_matrix, train_matrix_tr, encodings, indecis, user_id_to_row):
        self.train_matrix = naive_sparse2tensor(train_matrix)
        self.train_matrix_tr = naive_sparse2tensor(train_matrix_tr)
        self.indices = indecis
        self.encodings = encodings
        self.user_id_to_row = user_id_to_row
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = self.indices[idx]
        item = self.encodings[row]       
        label_tensor = self.train_matrix[row]
        item['labels_tr']  = self.train_matrix_tr[row]
        if self.user_id_to_row is not None:
            item['idx'] = torch.tensor([self.user_id_to_row[row]])
        else:
            item['idx'] = torch.tensor([row])

        item['labels'] = label_tensor

        return  item


class MatrixDataLoader():
    def __init__(self, path, args):
        self.pro_dir = path
        self.args = args

        assert os.path.exists(
            self.pro_dir), "Preprocessed files does not exist. Run data.py"

        self.n_items = self.load_n_items()

    def load_data(self, datatype='train', head=None):
        if datatype == 'train':
            return self._load_train_data(head=head)
        elif datatype == 'validation':
            return self._load_tr_te_data(datatype, head=head)
        elif datatype == 'test':
            return self._load_tr_te_data(datatype, head=head)
        else:
            raise ValueError("datatype should be in [train, validation, test]")

    def load_n_items(self):

        with open(os.path.join(self.pro_dir, 'show2id.pkl'), 'rb') as f:
            n_items = len(pickle.load(f))

        return n_items

    def _load_train_data(self, head=50):

        path = os.path.join(self.pro_dir, 'train.csv')

        tp = pd.read_csv(path)
        self.n_users = tp['uid'].max() + 1
        tp_tr = tp.groupby('uid').apply(
            lambda x: x.head(head)).reset_index(drop=True)
        rows, cols, rating = tp_tr['uid'], tp_tr['sid'], tp_tr['rating']

        # Ratings as input
        data_tr = sparse.csr_matrix((np.ones_like(rows) if self.args.binarize else rating,
                                     (rows, cols)), dtype='float64',
                                    shape=(self.n_users, self.n_items))
        # Implicif feedback as target
        tp = tp[tp['rating'] > 3]
        rows, cols = tp['uid'], tp['sid']
        data_te = sparse.csr_matrix((np.ones_like(rows),
                                     (rows, cols)), dtype='float64',
                                    shape=(self.n_users, self.n_items))

        return data_tr, data_te

    def _load_tr_te_data(self, datatype='test', head=None):
        tr_path = os.path.join(self.pro_dir, '{}_tr.csv'.format(datatype))
        te_path = os.path.join(self.pro_dir, '{}_te.csv'.format(datatype))
        tp_tr = pd.read_csv(tr_path)
        tp_te = pd.read_csv(te_path)

        rows_tr, cols_tr, rating_tr = tp_tr['uid'], tp_tr['sid'], tp_tr['rating']

        rows_te, cols_te, rating_te = tp_te['uid'], tp_te['sid'], tp_te['rating']

        # use ratings as input
        data_tr = sparse.csr_matrix((np.ones_like(rows_tr) if self.args.binarize else rating_tr,
                                    (rows_tr, cols_tr)), dtype='float64', shape=(self.n_users, self.n_items))
        # test items are simply binary feedback
        data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                    (rows_te, cols_te)), dtype='float64', shape=(self.n_users, self.n_items))

        return data_tr, data_te


def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())


def map_id_to_title(data='ml-1m'):
    if data == 'ml-1m':

        data = pd.read_csv('./data/ml-1m/movies.dat', sep="::",
                           names=["itemId", "title", "genre"], encoding='ISO-8859-1')
        with open('./data_preprocessed/ml-1m/show2id.pkl', 'rb') as f:
            item_id_map = pickle.load(f)
        mapping = {}
        for index, row in data.iterrows():
            if row['itemId'] in item_id_map:
                mapping[item_id_map[row['itemId']]] = row['title']

        return mapping
    elif data == 'netflix':

        with open('./data_preprocessed/netflix/show2id.pkl', 'rb') as f:
            item_id_map = pickle.load(f)

        with open('./data/netflix/movie_titles.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            data = []
            for row in reader:
                new_row = []
                comma_count = 0
                for field in row:
                    new_row.append(field)
                    if ',' in field:
                        comma_count += field.count(',')
                        if comma_count <= 3:
                            new_row.append('entry')  # add a new column with the value 'entry'
                data.append(new_row)

        # Convert the data to a Pandas DataFrame
        df = pd.DataFrame(data)
        df.columns = ['movieId', 'year', 'title1','title2','title3','title4']

        def apply_fun(x):
            cols= ['title1','title2','title3','title4']
            out = ''
            for col in cols:
                if x[col] is not None:
                    out += x[col] + ' '
            return out
        df['title'] = df.apply(apply_fun, axis=1)
        mapping = {}
        for index, row in df.iterrows():

            if row['movieId'] in item_id_map:
                mapping[item_id_map[row['movieId']]] = row['title']
                
    elif data == 'goodbooks':
        df = pd.read_csv('./data/goodbooks/genres.csv')
        df.genres = df.genres.apply(ast.literal_eval)

        mapping = {}
        with open('./data_preprocessed/goodbooks/show2id.pkl', 'rb') as f:
            item_id_map = pickle.load(f)
        for index, row in df.iterrows():
            if row['book_id'] in item_id_map:
                mapping[item_id_map[row['book_id']]] = row['title']

    return mapping


def map_id_to_genre(data='ml-1m'):
    if data == 'ml-1m':
        data = pd.read_csv('./data/ml-1m/movies.dat', sep="::",
                           names=["itemId", "title", "genre"], encoding='ISO-8859-1')
        with open('./data_preprocessed/ml-1m/show2id.pkl', 'rb') as f:
            item_id_map = pickle.load(f)
        mapping = {}
        for index, row in data.iterrows():
            if row['itemId'] in item_id_map:
                genres = row['genre'].lower().replace('-', ' ').split('|')
                mapping[item_id_map[row['itemId']]] = genres

    elif data == 'netflix':
        

        with open('./data_preprocessed/netflix/show2id.pkl', 'rb') as f:
            item_id_map = pickle.load(f)

        df = pd.read_csv('./data/netflix/netflix_genres.csv')

        mapping = {}

        for index, row in df.iterrows():
            if row['movieId'] in item_id_map:
                mapping[item_id_map[row['movieId']]
                        ] = row['genres'].lower().replace('-', ' ').split('|')

    elif data == 'goodbooks':
        df = pd.read_csv('./data/goodbooks/genres.csv')
        df.genres = df.genres.apply(ast.literal_eval)

        mapping = {}
        with open('./data_preprocessed/goodbooks/show2id.pkl', 'rb') as f:
            item_id_map = pickle.load(f)
        for index, row in df.iterrows():
            if row['book_id'] in item_id_map:
                mapping[item_id_map[row['book_id']]] = row['genres']

    return mapping
