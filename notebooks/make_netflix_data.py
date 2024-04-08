# %%
import pandas as pd
import multiprocessing as mp
from multiprocessing import Manager
from tqdm import tqdm

def process_file(i, data):
    p = f'../data/netflix/combined_data_{i}.txt'
    with open(p, 'r') as file:
        for line in tqdm(file, desc=f"Processing file {i}"):
            # If line ends with ':', it's a movie id
            if line.strip().endswith(':'):
                movie_id = line.strip()[:-1]
            else:
                # Split user interaction into parts
                user, rating, timestamp = line.strip().split(',')

                # Append to list
                data.append({
                    'item': movie_id,
                    'user': user,
                    'rating': int(rating),
                    'timestamp': pd.to_datetime(timestamp)
                })


with Manager() as manager:
    data = manager.list()
    with mp.Pool() as pool:
        pool.starmap(process_file, [(i, data) for i in range(1,5)])

    # Convert list to DataFrame
    df = pd.DataFrame(data)

    # Print the DataFrame
    print(df)

# %%
data.to_csv('../data/netflix/netflix_data.csv', index=False)
print('Data saved to ../data/netflix/netflix_data.csv')

