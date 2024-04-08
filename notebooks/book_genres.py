import pandas as pd 
import numpy as np
import requests
import pandas as pd
from tqdm import tqdm
import logging
import time
path = '/home/mila/e/emiliano.penaloza/LLM4REC/data/goodbooks/goodbooks-10k'
ratings = pd.read_csv(path + '/ratings.csv')
tags = pd.read_csv(path + '/tags.csv')
books = pd.read_csv(path + '/books.csv')

logging.basicConfig(level=logging.INFO)
def get_book_info(isbn):
    response = requests.get(f'https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}&key=AIzaSyCsdHoo53zswipX5dh-J8Decm-3RUr6np4')
    data = response.json()

    # Initialize book info
    book_info = {'genres': None, 'api_title': None}

    # Check if the book was found
    if data['totalItems'] > 0:
        volume_info = data['items'][0]['volumeInfo']

        # Get the book title
        if 'title' in volume_info:
            book_info['api_title'] = volume_info['title']

        # Check if the book has categories
        if 'categories' in volume_info:
            book_info['genres'] = ', '.join(volume_info['categories'])

    return book_info

# Initialize lists to store the genres and titles
genres = []
api_titles = []

# Iterate over the ISBNs in the DataFrame
for i,isbn in tqdm(enumerate(books['isbn'])):
    # Get the book info
    completed = False
    while not completed:
        try:
            if i % 100 == 0 and i != 0:
                time.sleep(60)
            book_info = get_book_info(isbn)

            # Append the genres and title to the lists
            genres.append(book_info['genres'])
            api_titles.append(book_info['api_title'])
            completed = True
        except Exception as e:
            time.sleep(60)
            logging.error(e)


# Add the genres and titles as new columns in the DataFrame
books['genres'] = genres
books['api_title'] = api_titles

# Save the DataFrame to a CSV file
books.to_csv(path + '/books_with_genres.csv', index=False)



