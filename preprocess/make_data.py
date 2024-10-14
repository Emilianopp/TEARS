
import json
import random
import os
import pandas as pd
from scipy import sparse
import numpy as np
import argparse
random.seed(2024)


def parse_args():
    parser = argparse.ArgumentParser(description="Data Preprocessing")
    parser.add_argument('--data_name', type=str, default='ml-1m',
                        help='name of the dataset')
    parser.add_argument('--filtered', action='store_true', default=False)
    parser.add_argument('--llm_backbone', type = str, default="gpt-4-1106-preview")
    
    return parser.parse_args()


args = parse_args()


def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


def get_counts(tp, min_uc=0, min_sc=0):
    usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'itemId')
    return tp, usercount, itemcount


def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('userId')
    tr_list, te_list = list(), list()

    np.random.seed(98765)
    promt_dataset = pd.read_csv(
        f'./data_preprocessed/{args.data_name }/prompt_set_new_.csv')
    if args.data_name == 'ml-1m':
        promt_dataset.rename(columns={'movieId': 'itemId'}, inplace=True)

    for _, group in data_grouped_by_user:
        n_items_u = len(group)
        assert len(group.userId.unique()) == 1
        user_prompts_user = promt_dataset[promt_dataset.userId == group.userId.unique()[
            0]]
        group_train = group[group.itemId.isin(user_prompts_user.itemId)]
        group_val = group[~group.itemId.isin(user_prompts_user.itemId)]
        group_val = group_val[group_val.rating > 3]
        tr_list.append(group_train)
        te_list.append(group_val)

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)
    return data_tr, data_te


def numerize(tp, profile2id, show2id):

    uid = tp['userId'].apply(lambda x: profile2id[x]
                             if x in profile2id else np.nan)
    sid = tp['itemId'].apply(lambda x: show2id[x] if x in show2id else np.nan)

    if 'title' in tp.columns:
        ratings = tp['rating']
        titles = tp['title']
        out_df = pd.DataFrame(data={'uid': uid, 'sid': sid, 'rating': ratings, 'title': titles}, columns=[
                              'uid', 'sid', 'rating', 'title'])
    else:
        ratings = tp['rating']
        out_df = pd.DataFrame(data={'uid': uid, 'sid': sid, 'rating': ratings}, columns=[
                              'uid', 'sid', 'rating'])
    out_df = out_df[out_df.uid.notna()]
    out_df = out_df[out_df.sid.notna()]
    assert out_df.isna().sum().sum() == 0
    return out_df


if __name__ == '__main__':

    print(f"Load and Preprocess {args.data_name} dataset")
    DATA_DIR = f'./data_preprocessed/{args.data_name}/'
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if args.data_name == 'ml-1m':

        raw_data = pd.read_csv(
            f'./data/{args.data_name}/ratings.dat', sep="::", header=None, encoding='ISO-8859-1')
        raw_data.columns = ['userId', 'itemId', 'rating', 'timestamp']

    elif args.data_name == 'books':
        raw_data = pd.read_csv(
            f'./data/{args.data_name}/ratings.csv', header=0)
        raw_data.rename(columns={'book_id': 'itemId', 'review/time': 'timestamp', 'Title': 'title',
                        'review/score': 'rating', 'User_id': 'userId', 'categories': 'genres'}, inplace=True)
    elif args.data_name == 'goodbooks':

        raw_data = pd.read_csv(
            f'./data/{args.data_name}/ratings_filtered.csv', header=0)

        raw_data.rename(columns={
                        'book_id': 'itemId', 'Title': 'title', 'user_id': 'userId'}, inplace=True)
        with open(f'./saved_user_summary/goodbooks/user_summary_{args.llm_backbone}_.json', 'rb') as f:
            prompts = json.load(f)
            prompts_keys = set([int(float(k)) for k, v in prompts.items()])
        raw_data = raw_data[raw_data.userId.isin(prompts_keys)]
    elif args.data_name == 'netflix':
        ratings_file = './data/netflix/ratings_filtered.csv'
        raw_data = pd.read_csv(ratings_file)
        raw_data.rename(columns={'MovieID': 'itemId', 'Title': 'title', 'CustomerID': 'userId',
                        'Name': 'title', 'Rating': 'rating', 'Date': 'timestamp'}, inplace=True)
        valid_item_names = raw_data.title.unique().tolist()
        strong_generalization_set = pd.read_csv(
            f'./data_preprocessed/{args.data_name}/strong_generalization_set_.csv')



    # Filter Data
    raw_data, user_activity, item_popularity = get_counts(raw_data)

    unique_uid = pd.Series(raw_data.userId.unique())
    np.random.seed(98765)
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]
    n_users = unique_uid.size

    # Split Train/Validation/Test User Indices
    train_data = pd.read_csv(
        f'./data_preprocessed/{args.data_name }/train_leave_one_out_.csv')

    strong_generalization_set = pd.read_csv(
        f'./data_preprocessed/{args.data_name }/strong_generalization_set_.csv')
    if args.data_name == 'ml-1m':

        train_data.rename(columns={'movieId': 'itemId'}, inplace=True)
        strong_generalization_set.rename(
            columns={'movieId': 'itemId'}, inplace=True)

    with open(f'./saved_user_summary/{args.data_name }/user_summary_{args.llm_backbone}_.json', 'r') as f:
        prompts = json.load(f)
        prompts_keys = set([int(float(k)) for k, v in prompts.items()])

    random.seed(2024)
    strong_gen_val_user_set = random.sample(list(strong_generalization_set.userId.unique(
    )), int(len(strong_generalization_set.userId.unique())/2))
    strong_generalization_set_val = strong_generalization_set[strong_generalization_set.userId.isin(
        strong_gen_val_user_set)]
    strong_generalization_set_test = strong_generalization_set[~strong_generalization_set.userId.isin(
        strong_gen_val_user_set)]

    unique_uid = unique_uid[unique_uid.isin(prompts_keys)]

    tr_users = unique_uid[unique_uid.isin(train_data.userId.unique())]

    vd_users = unique_uid[unique_uid.isin(
        strong_generalization_set_val.userId.unique())]

    te_users = unique_uid[unique_uid.isin(
        strong_generalization_set_test.userId.unique())]

    train_plays = raw_data.loc[raw_data['userId'].isin(tr_users)]

    if args.data_name == 'ml-1m':
        unique_sid = set(pd.unique(train_plays['itemId'])) & set(pd.unique(
            strong_generalization_set_val['itemId'])) & set(pd.unique(strong_generalization_set_test['itemId']))
        unique_sid = set(raw_data.loc[(raw_data['itemId'].isin(
            unique_sid)) & (raw_data['rating'] > 3)].itemId)
    else:
        unique_sid = set(pd.unique(train_plays['itemId'])) | set(pd.unique(
            strong_generalization_set_val['itemId'])) | set(pd.unique(strong_generalization_set_test['itemId']))

    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(prompts_keys))
    prompts = {profile2id[int(float(k))]: v for k, v in prompts.items()}

    import pickle

    with open(os.path.join(DATA_DIR, 'show2id.pkl'), 'wb') as f:
        pickle.dump(show2id, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(DATA_DIR, 'profile2id.pkl'), 'wb') as f:
        pickle.dump(profile2id, f, pickle.HIGHEST_PROTOCOL)

  

    vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)]

    vad_plays = vad_plays.loc[vad_plays['itemId'].isin(unique_sid)]

    vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)

    vad_plays_tr = vad_plays_tr.loc[vad_plays_tr['userId'].isin(
        vad_plays_te.userId)]

    test_plays = raw_data.loc[raw_data['userId'].isin(te_users)]
    test_plays = test_plays.loc[test_plays['itemId'].isin(unique_sid)]

    test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)
    test_plays_tr = test_plays_tr.loc[test_plays_tr['userId'].isin(
        test_plays_te.userId)]

    train_data = numerize(train_plays, profile2id, show2id)

    train_data.to_csv(os.path.join(DATA_DIR, 'train.csv'), index=False)

    vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)

    vad_data_tr.to_csv(os.path.join(
        DATA_DIR, 'validation_tr.csv'), index=False)

    vad_data_te = numerize(vad_plays_te, profile2id, show2id)

    vad_data_te.to_csv(os.path.join(
        DATA_DIR, 'validation_te.csv'), index=False)

    test_data_tr = numerize(test_plays_tr, profile2id, show2id)
    test_data_tr.to_csv(os.path.join(DATA_DIR, 'test_tr.csv'), index=False)

    test_data_te = numerize(test_plays_te, profile2id, show2id)
    test_data_te.to_csv(os.path.join(DATA_DIR, 'test_te.csv'), index=False)

    dataset_full = pd.concat(
        [train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te])
    dataset_full.to_csv(os.path.join(
        DATA_DIR, 'dataset_full.csv'), index=False)

    print("Done!")
