
import json
import random 
import os
import pandas as pd
from scipy import sparse
import numpy as np
random.seed(2024)


class DataLoader():
    '''
    Load Movielens-20m dataset
    '''
    def __init__(self, path):
        self.DATA_DIR = os.path.join(path, 'pro_sg_text')
        assert os.path.exists(self.DATA_DIR), "Preprocessed files does not exist. Run data.py"
    
    def load_data(self, datatype='train'):
        if datatype == 'train':
            return self._load_train_data()
        elif datatype == 'validation':
            return self._load_tr_te_data(datatype)
        elif datatype == 'test':
            return self._load_tr_te_data(datatype)
        else:
            raise ValueError("datatype should be in [train, validation, test]")
            
    def _load_train_data(self):
        path = os.path.join(self.DATA_DIR, 'train.csv')

        
        tp = pd.read_csv(path)
        self.n_users = tp['uid'].max() + 1

        rows, cols = tp['uid'], tp['sid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                 (rows, cols)), dtype='float64',
                                 shape=(self.n_users, self.n_items))
        return data
    
    def _load_tr_te_data(self, datatype='test'):
        tr_path = os.path.join(self.DATA_DIR, '{}_tr.csv'.format(datatype))
        te_path = os.path.join(self.DATA_DIR, '{}_te.csv'.format(datatype))
        tp_tr = pd.read_csv(tr_path)
        tp_te = pd.read_csv(te_path)
        test_users = tp_te['uid'].unique()
        train_users = tp_tr['uid'].unique()
        tp_te = tp_te.loc[tp_te['uid'].isin(tp_tr['uid'])]
        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())
        rows_tr, cols_tr = tp_tr['uid'] , tp_tr['sid']
        rows_te, cols_te = tp_te['uid'], tp_te['sid']

        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                    (rows_tr, cols_tr)), dtype='float64', shape=(self.n_users, self.n_items))
        data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                    (rows_te, cols_te)), dtype='float64', shape=(self.n_users, self.n_items))
       
        subset_te = data_te[data_te.nonzero()]

        zero_rows = (data_te.sum(axis=1) == 0).nonzero()[0]
       
        test_users = data_te.nonzero()[0]
        train_users = data_tr.nonzero()[0]
        print(f"{data_te.shape=}")
        print(f"{data_tr.shape=}")
        return data_tr, data_te

    '''
    Load Movielens-20m dataset
    '''
    def __init__(self, path):
        self.DATA_DIR = os.path.join(path, 'pro_sg_text')
        assert os.path.exists(self.DATA_DIR), "Preprocessed files does not exist. Run data.py"

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
        with open(os.path.join(self.DATA_DIR, 'unique_sid.txt'), 'r') as f:
            for line in f:
                unique_sid.append(line.strip())
        n_items = len(unique_sid)
        return n_items
    
    def _load_train_data(self):
        path = os.path.join(self.DATA_DIR, 'train.csv')
        
        tp = pd.read_csv(path)
        n_users = tp['uid'].max() + 1

        rows, cols = tp['uid'], tp['sid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                 (rows, cols)), dtype='float64',
                                 shape=(n_users, self.n_items))
        return data
    
    def _load_tr_te_data(self, datatype='test'):
        tr_path = os.path.join(self.DATA_DIR, '{}_tr.csv'.format(datatype))
        te_path = os.path.join(self.DATA_DIR, '{}_te.csv'.format(datatype))

        tp_tr = pd.read_csv(tr_path)
        tp_te = pd.read_csv(te_path)
        #make sure all the users in tr are in te 
        tp_te = tp_te.loc[tp_te['uid'].isin(tp_tr['uid'])]
        #and viceversa
        tp_tr = tp_tr.loc[tp_tr['uid'].isin(tp_te['uid'])]

        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

        rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
        rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                    (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, self.n_items))
        data_te = sparse.csr_matrix((np.ones_like(rows_te),

        
                                    (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, self.n_items))
        print(f"{data_te.toarray().sum(axis=1)=}")
        #make a map of user_id to row index
        #remove nonzero rows 
        user_id_to_row = {row_idx: user_id for row_idx, user_id in enumerate(tp_tr['uid'].unique())}
        data_te = data_te[data_te.sum(axis = 1).nonzero()[0]]
        data_tr = data_tr[data_tr.sum(axis = 1).nonzero()[0]]
        #assert the have the same dimensions 
        assert data_te.shape == data_tr.shape
        

        return data_tr, data_te,user_id_to_row

def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count



def filter_triplets(tp, min_uc=0, min_sc=0):
    if min_sc > 0:
        itemcount = get_count(tp, 'movieId')
        tp = tp[tp['movieId'].isin(itemcount.index[itemcount['size'] >= min_sc])]
    
    if min_uc > 0:
        usercount = get_count(tp, 'userId')
        tp = tp[tp['userId'].isin(usercount.index[usercount['size'] >= min_uc])]
    
    usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'movieId')
    return tp, usercount, itemcount



def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('userId')
    tr_list, te_list = list(), list()

    np.random.seed(98765)
    promt_dataset = pd.read_csv(f'../data_preprocessed/ml-1m/prompt_set_timestamped.csv')
    promt_dataset = promt_dataset[promt_dataset.rating > 3]
    for _, group in data_grouped_by_user:

        n_items_u = len(group)

        assert len(group.userId.unique()) == 1
        user_prompts_user = promt_dataset[promt_dataset.userId == group.userId.unique()[0  ]]
        group_train = group[group.movieId.isin(user_prompts_user.movieId)]
        group_val = group[~group.movieId.isin(user_prompts_user.movieId)]
        tr_list.append(group_train)
        te_list.append(group_val)
        
        
    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)
    return data_tr, data_te



def numerize(tp, profile2id, show2id):
    
    # uid = tp['userId'].apply(lambda x: profile2id[x])
    # sid = tp['movieId'].apply(lambda x: show2id[x])
    # return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])
    uid = tp['userId'].apply(lambda x: profile2id[x] if x in profile2id else np.nan)
    sid = tp['movieId'].apply(lambda x: show2id[x] if x in show2id else np.nan)
    ratings = tp['rating']
    out_df = pd.DataFrame(data={'uid': uid, 'sid': sid,'rating':ratings}, columns=['uid', 'sid','rating'])
    #filter out nans 
    out_df = out_df[out_df.uid.notna()]
    out_df = out_df[out_df.sid.notna()]


    #assert there are no more nan rows 
    assert out_df.isna().sum().sum() == 0
    return out_df 



if __name__ == '__main__':

    print("Load and Preprocess Movielens-20m dataset")
    # Load Data
    DATA_DIR = '../data_preprocessed/ml-1m/'
    # raw_data = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'), header=0) READ LIKE THIS BUT IT IS A .DAT FILE WITH SEP = ::
    raw_data = pd.read_csv('../data/ml-1m/ratings.dat', sep="::", header=None, encoding='ISO-8859-1')
    raw_data.columns = ['userId', 'movieId', 'rating', 'timestamp']

    raw_data = raw_data[raw_data['rating'] > 3]

    print(f"{len(raw_data.userId.unique())=}")


    # Filter Data
    raw_data, user_activity, item_popularity = filter_triplets(raw_data)


    # Shuffle User Indices
    unique_uid = pd.Series(raw_data.userId.unique())

    np.random.seed(98765)
    idx_perm = np.random.permutation(unique_uid.size)

    unique_uid = unique_uid[idx_perm]

    n_users = unique_uid.size
    print(f"{n_users=}")
    
    n_heldout_users = 250

    # Split Train/Validation/Test User Indices
    train_data = pd.read_csv(f'../data_preprocessed/ml-1m/train_leave_one_out_timestamped.csv')
    
    strong_generalization_set = pd.read_csv(f'../data_preprocessed/ml-1m/strong_generalization_set_timestamped.csv')
    #filter such that the rating is higher than 3 
    strong_generalization_set = strong_generalization_set[strong_generalization_set.rating > 3]
    train_data = train_data[train_data.rating > 3]
    with open(f'../saved_user_summary/ml-1m/user_summary_gpt4_.json','r') as f:
        prompts = json.load(f)
        prompts_keys = set([int(float(k)) for k,v in prompts.items()])

    strong_gen_val_user_set = random.sample(list(strong_generalization_set.userId.unique()),int(len(strong_generalization_set.userId.unique())/2))
    strong_generalization_set_val =strong_generalization_set[strong_generalization_set.userId.isin(strong_gen_val_user_set)]
    strong_generalization_set_test = strong_generalization_set[~strong_generalization_set.userId.isin(strong_gen_val_user_set)]

    unique_uid = unique_uid[unique_uid.isin(prompts_keys) ]

    tr_users = unique_uid[unique_uid.isin(train_data.userId.unique())]
    print(f"{len(tr_users)=}")

    vd_users = unique_uid[unique_uid.isin(strong_generalization_set_val.userId.unique())]
    print(f"{len(vd_users)=}")

    te_users = unique_uid[unique_uid.isin(strong_generalization_set_test.userId.unique())]
    print(f"{len(te_users)=}")

    train_plays = raw_data.loc[raw_data['userId'].isin(tr_users)]

    unique_sid = pd.unique(train_plays['movieId'])

    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(prompts_keys) )

    prompts = {profile2id[int(float(k))]:v for k,v in prompts.items() }
    #save the two dicts abovec to pickle 
    import pickle
    with open(os.path.join(DATA_DIR, 'show2id.pkl'), 'wb') as f:
        pickle.dump(show2id, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(DATA_DIR, 'profile2id.pkl'), 'wb') as f:
        pickle.dump(profile2id, f, pickle.HIGHEST_PROTOCOL)


    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    with open(os.path.join(DATA_DIR, 'unique_sid.txt'), 'w') as f:
        for sid in unique_sid:
            f.write('%s\n' % sid)
    
    vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)]

    vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)]

    vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)
    #save vad_plays_tr to use as validation set for the training of the VAE
    vad_plays_tr.to_csv(os.path.join(DATA_DIR, 'validation_tr_.csv'), index=False)
    vad_plays_te.to_csv(os.path.join(DATA_DIR, 'validation_te_.csv'), index=False)
    #make sure the tr has the same users as the test 
    vad_plays_tr = vad_plays_tr.loc[vad_plays_tr['userId'].isin(vad_plays_te.userId)]

    test_plays = raw_data.loc[raw_data['userId'].isin(te_users)]
    test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sid)]

    test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)
    test_plays_tr = test_plays_tr.loc[test_plays_tr['userId'].isin(test_plays_te.userId)]


    train_data = numerize(train_plays, profile2id, show2id)
    
    train_data.to_csv(os.path.join(DATA_DIR, 'train.csv'), index=False)

    vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)

    vad_data_tr.to_csv(os.path.join(DATA_DIR, 'validation_tr.csv'), index=False)
    
    vad_data_te = numerize(vad_plays_te, profile2id, show2id)

    
    vad_data_te.to_csv(os.path.join(DATA_DIR, 'validation_te.csv'), index=False)

    test_data_tr = numerize(test_plays_tr, profile2id, show2id)
    test_data_tr.to_csv(os.path.join(DATA_DIR, 'test_tr.csv'), index=False)

    test_data_te = numerize(test_plays_te, profile2id, show2id)
    test_data_te.to_csv(os.path.join(DATA_DIR, 'test_te.csv'), index=False)

    print("Done!")