
import pandas as pd
import json
import random
import os
from collections import OrderedDict
import argparse 
#silence warnings
import warnings
from ast import literal_eval
import numpy as np 

warnings.filterwarnings("ignore")



parser = argparse.ArgumentParser(description='Generate train, validation, and test splits for MovieLens dataset.')
parser.add_argument('--data_name', type=str, choices=['ml-1m', 'ml-100k','books','goodbooks','netflix'], default='ml-1m',
                        help='Name of the MovieLens dataset (ml-1m or ml-100k). Default is ml-1m.')
parser.add_argument('--timestamp', action='store_true')


args = parser.parse_args()
data_name = args.data_name

def preprocess_good_reads():

    path = './data/goodbooks/'
    ratings = pd.read_csv(path + '/ratings_raw.csv')
    books = pd.read_csv(path + '/books.csv')

    genres = pd.read_csv('https://raw.githubusercontent.com/malcolmosh/goodbooks-10k/master/books_enriched.csv', index_col=[0], converters={"genres": literal_eval})[['book_id', 'genres']]
    ratings = ratings.merge(genres, on='book_id', how='left')
    languages =['eng','en-US','en-GB','en-CA']
    book_cols = ['book_id','language_code','title','authors','work_id']
    books = books[book_cols]
    books = books[books.language_code.isin(languages)]



    #preserve the orinal index as data come pre-sorted by timestamp so any shuffling will mess up the order
    ratings['original_index'] = ratings.index
    ratings_merged = ratings.merge(books, left_on='book_id', right_on='book_id', how='inner', sort=False)
    ratings_merged = ratings_merged.sort_values(by='original_index').drop(columns=['original_index'])
    subset_ratings = ratings_merged.groupby('book_id').filter(lambda x: len(x) >= 20)
    subset_ratings = ratings_merged.groupby('user_id').filter(lambda x: len(x) >= 20)


    #set the seed for reproducibility
    np.random.seed(2024)
    #sample 10k users interactioons
    users = set(subset_ratings.user_id.unique())
    sample_users = np.random.choice(list(users),10000,replace=False)
    subset_ratings = subset_ratings[subset_ratings.user_id.isin(sample_users)]

    #delete items that appear less than 20 times 
    items = set(subset_ratings.book_id.unique())
    items = list(items)
    items_count = subset_ratings.groupby('book_id').size()
    items_count = items_count[items_count >= 20]
    items = items_count.index
    subset_ratings = subset_ratings[subset_ratings.book_id.isin(items)]
    #enumerate the user ids and item ids 
    user_ids = subset_ratings.user_id.unique()
    user2id = {user_id: i for i, user_id in enumerate(user_ids)}
    item_ids = subset_ratings.book_id.unique()
    item2id = {item_id: i for i, item_id in enumerate(item_ids)}
    subset_ratings['user_id'] = subset_ratings['user_id'].map(user2id)
    subset_ratings['book_id'] = subset_ratings['book_id'].map(item2id)


    subset_ratings.to_csv(path + '/ratings_filtered.csv', index=False)
    return subset_ratings






    

def split_and_filter_ratings(user_movies, rating_threshold=4):
    # Sort the DataFrame by the 'timestamp' column in descending order and select the first 52 movies
    user_movies = user_movies.sort_values(by='timestamp', ascending=True) if args.data_name != 'goodbooks' else user_movies

    if user_movies.shape[0] < 52:
        #take the last two movies that are rated higher than the threshold
        validation_test_movies = user_movies[user_movies['rating'] >= rating_threshold].tail(2)
        validation_set = validation_test_movies.head(1)
        test_set = validation_test_movies.tail(1)
        #use the rest for train and prompt set
        training_set = prompt_movies= user_movies.drop(validation_test_movies.index)

    else:
        candidate_recent_movies = user_movies.head(52)
        validation_test_movies = candidate_recent_movies[candidate_recent_movies['rating'] >= rating_threshold].tail(2)
        
        # Select the first two movies from the high-rated movies as validation and test sets

        # If there are no high-rated movies or validation_candidatess_recent_movies is empty, set validation and test sets to None

        # Split the selected movies into validation and test sets
        validation_set = validation_test_movies.head(1)
        test_set = validation_test_movies.tail(1)
        training_set = user_movies.drop(validation_test_movies.index)
        prompt_movies = training_set.head(50)
    if validation_test_movies.shape[0] < 2:
        return None,None,None,None
    return prompt_movies, validation_set, test_set,training_set

def generate_train_val_test_splits(ratings, k):
    # Count the number of ratings for each user
    user_counts = ratings['userId'].value_counts()

    # Filter out users with fewer than k ratings
    eligible_users = user_counts[user_counts > k].index
  

    # Initialize empty lists for training, validation, and test data
    train_data = []
    val_data = []
    test_data = []
    prompt_set =[]

    # Iterate through eligible users
    non_users = []
    for user_id in eligible_users:
        # Extract user's ratings
        # user_movies = user_ratings.merge(movie_metadata, left_on='itemId', right_on='movielens_id')
        user_movies = ratings[ratings['userId'] == user_id]
        # user_movies = user_movies.dropna(subset=['summary'])  # Remove movies with 'NaN' summary
        # user_movies = user_movies[user_movies['title'].isin(valid_item_names)]
        movie_set = set(user_movies['itemId'])


        # Randomly choose one movie for validation and remove it from user's ratings
        prompt_movies,validation_movie, test_movie,training_set = split_and_filter_ratings(user_movies)
        if prompt_movies is None:
            non_users.append(user_id)
            continue

        # Append user's training data
        prompt_set.append(prompt_movies)
        train_data.append(training_set)

        # Append user's validation data
        val_data.append(validation_movie)

        # Append user's test data
        test_data.append(test_movie)
        
        
 
    # Concatenate the dataframes to get the final splits
    train_data = pd.concat(train_data)
    val_data = pd.concat(val_data)
    test_data = pd.concat(test_data)
    promp_set = pd.concat(prompt_set)

    return train_data, val_data, test_data,promp_set,non_users



if __name__ == "__main__":
        

    k = 20  # threshold for history length


    if data_name == 'ml-1m':
        k = 10  #ml-1m already has a minimum of 20 interactions
        
        valid_item_names = pd.read_csv(f'./data/ml-1m/movies.dat',encoding='ISO-8859-1',sep='::',header=None).iloc[:,1].tolist()
        ratings_file = './data/ml-1m/ratings.dat'
        separator = "::"
        header = None
        rating_columns = ['userId', 'itemId', 'rating', 'timestamp']
        movie_metadata_file = './data/ml-1m/movies.dat'
        ratings = pd.read_csv(ratings_file, sep=separator, header=header, names=rating_columns)
        
        # Load movie metadata
        movie_metadata = pd.read_csv(movie_metadata_file,encoding='ISO-8859-1',sep='::',header=None)
        movie_metadata.columns = ['movielens_id','title','genre']


        ratings = ratings.merge(movie_metadata, left_on='itemId', right_on='movielens_id', how = 'left')
      
        train_data, val_data, test_data,promp_set,non_users = generate_train_val_test_splits(ratings, k)
 
    elif data_name == 'books':
        ratings_file = './data/books/ratings.csv'
        ratings = pd.read_csv(ratings_file)
        ratings.rename(columns={'book_id':'itemId','review/time':'timestamp','Title':'title','review/score':'rating','User_id':'userId','categories':'genres'}, inplace=True)

        valid_item_names = ratings.title.unique().tolist()
        #rename the bookId column to itemId
        train_data, val_data, test_data,promp_set,non_users = generate_train_val_test_splits(ratings, k)
    elif data_name == 'goodbooks':
        ratings_file = './data/goodbooks/ratings.csv'
        ratings = preprocess_good_reads()

        #remove cold start items
        ratings = ratings[ratings['book_id'].isin(ratings['book_id'].value_counts()[ratings['book_id'].value_counts()>10].index)]
        
        ratings.rename(columns={'book_id':'itemId','Title':'title','user_id':'userId'}, inplace=True)
        valid_item_names = ratings.title.unique().tolist()
        #rename the bookId column to itemId
        train_data, val_data, test_data,promp_set,non_users = generate_train_val_test_splits(ratings, k)
    elif data_name == 'netflix':
        
        ratings_file = './data/netflix/ratings_filtered.csv'
        ratings = pd.read_csv(ratings_file)
        ratings.rename(columns={'MovieID':'itemId','Title':'title','CustomerID':'userId','Name':'title','Rating':'rating','Date':'timestamp'}, inplace=True)
        print(f"{ratings.columns=}")
        valid_item_names = ratings.title.unique().tolist()
        #rename the bookId column to itemId
        train_data, val_data, test_data,promp_set,non_users = generate_train_val_test_splits(ratings, k)
    
        
    valid_item_names = list(valid_item_names)
    train_data = train_data[train_data['title'].isin(valid_item_names)]


    val_data = val_data[val_data['title'].isin(valid_item_names)]
    test_data = test_data[test_data['title'].isin(valid_item_names)]

    # Check for overlapping user-movie pairs again after filtering
    train_user_movie_pairs = set(zip(train_data['userId'], train_data['itemId']))
    val_user_movie_pairs = set(zip(val_data['userId'], val_data['itemId']))
    test_user_movie_pairs = set(zip(test_data['userId'], test_data['itemId']))

    overlap_train_val = train_user_movie_pairs.intersection(val_user_movie_pairs)
    overlap_train_test = train_user_movie_pairs.intersection(test_user_movie_pairs)
    overlap_val_test = val_user_movie_pairs.intersection(test_user_movie_pairs)
    num_users_train = len(set(train_data['userId']))
    num_users_val = len(set(val_data['userId']))
    num_users_test = len(set(test_data['userId']))


    
    ### Error checking 
    assert not overlap_train_val, f"Overlap between train and validation sets in rows:\n{train_data[train_data[['userId', 'itemId']].apply(tuple, axis=1).isin(overlap_train_val)]}"
    assert not overlap_train_test, f"Overlap between train and test sets in rows:\n{train_data[train_data[['userId', 'itemId']].apply(tuple, axis=1).isin(overlap_train_test)]}"
    assert not overlap_val_test, f"Overlap between validation and test sets in rows:\n{val_data[val_data[['userId', 'itemId']].apply(tuple, axis=1).isin(overlap_val_test)]}"

    print("No overlap found after filtering by valid_item_names.")
    assert num_users_test == num_users_train and num_users_test == num_users_val, f'{num_users_val=} {num_users_train=} {num_users_test=}'
    print( 'The total number of users is ', len(set(ratings['userId'])))
    print(f"All sets have the same number of users used = {num_users_train}" , f"The number of non users is {len(non_users)}")
    
    #randomly remove 500 users from the training validation and test split and make a joint set of these called strong generalization set 
    random.seed(42)
    user_set = set(train_data['userId']) | set(val_data['userId']) | set(test_data['userId'])
    user_set = random.sample(list(user_set) ,500) if args.data_name =='ml-1m' else random.sample(list(user_set) ,2000)
    strong_generalization_set = pd.concat([train_data[train_data['userId'].isin(user_set)],val_data[val_data['userId'].isin(user_set)],test_data[test_data['userId'].isin(user_set)]])

    train_data = train_data[~train_data['userId'].isin(user_set)]
    val_data = val_data[~val_data['userId'].isin(user_set)]
    test_data = test_data[~test_data['userId'].isin(user_set)]

    #save data
    '''
    Since all user summaries exclude two movies from them for validation, we holdout two movies always from the training set 
    This procedure is the most fair for the summaries
    '''
    train_data.to_csv(f'./data_preprocessed/{data_name}/train_leave_one_out_.csv', index=False)
    val_data.to_csv(f'./data_preprocessed/{data_name}/validation_leave_one_out_.csv',index=False)
    test_data.to_csv(f'./data_preprocessed/{data_name}/test_leave_one_out_.csv', index=False)
    promp_set.to_csv(f'./data_preprocessed/{data_name}/prompt_set_new_.csv', index=False)
    strong_generalization_set.to_csv(f'./data_preprocessed/{data_name}/strong_generalization_set_.csv', index=False)
    movie_set = set(train_data['itemId']) 
    max_movie_id = max(movie_set)



    # Print a message
    print("data saved to CSVs to data_preprocessed folder")
        
   



