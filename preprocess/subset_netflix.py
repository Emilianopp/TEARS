import pandas as pd
import numpy as np
df = pd.read_csv('./data/netflix/ratings.csv')
#make a random sample of 50k users 
np.random.seed(2024)
sampled_users = np.random.choice(df['CustomerID'].unique(), 50000, replace=False)
print(f"{sampled_users=}")
df = df[df['CustomerID'].isin(sampled_users)]


import pandas as pd

def trim_to_target_users(df, k, m, target_users=10000, delta=100):

    while True:
        initial_shape = df.shape
        movie_counts = df['MovieID'].value_counts()
        movies_filtered = movie_counts[movie_counts >= m].index
        df = df[df['MovieID'].isin(movies_filtered)]
        user_counts = df['CustomerID'].value_counts()
        users_filtered = user_counts[user_counts >= k].index
        df = df[df['CustomerID'].isin(users_filtered)]
        
        if df.shape == initial_shape or df.empty:
            break
    
    # Trim process
    while len(df['CustomerID'].unique()) > target_users + delta:
        print(f"{len(df['CustomerID'].unique())=}")

        user_counts = df['CustomerID'].value_counts()
        movie_counts = df['MovieID'].value_counts()
        

        least_connected_users = user_counts[user_counts > k].index[-1]  # Users with count just above k
        least_connected_movies = movie_counts[movie_counts > m].index[-1]  # Movies with count just above m
        

        df = df[~(df['CustomerID'] == least_connected_users) & ~(df['MovieID'] == least_connected_movies)]
        

        if len(df['CustomerID'].unique()) <= target_users - delta:
            break
    
    return df

filtered_df = trim_to_target_users(df, k=100, m=100)

filtered_df.to_csv('./data/netflix/ratings_filtered.csv', index=False)