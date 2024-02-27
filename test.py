import numpy as np
import pandas as pd 
df = pd.read_csv('/home/mila/e/emiliano.penaloza/LLM4REC/data_preprocessed/ml-1m/prompt_set_timestamped.csv')
print(f"{df=}")
grupped = df.groupby('userId').count().movieId.mean()
print(f"{grupped=}")

df_strong = pd.read_csv('/home/mila/e/emiliano.penaloza/LLM4REC/data_preprocessed/ml-1m/strong_generalization_set_timestamped.csv')
# for each user filter out the movies that are found for that user in df 
num_movues = []
num_movies_in = []
for user in df_strong.userId.unique():
    df_subset = df[df.userId == user]
    df_strong_subset = df_strong[df_strong.userId == user]
    original_l = len(df_strong_subset)
    df_strong_subset = df_strong_subset[~df_strong_subset.movieId.isin(df_subset.movieId)]
    df_strong_subset_in = df_strong_subset[df_strong_subset.movieId.isin(df_subset.movieId)]


    
    num_movues.append(original_l - len(df_strong_subset))
    num_movies_in.append(len(df_strong_subset))    
np.mean(num_movues)
np.mean(num_movies_in)
print(f"{np.mean(num_movies_in)=}")
print(f"{np.mean(num_movues)=}")
