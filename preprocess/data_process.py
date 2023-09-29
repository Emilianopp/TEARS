import pandas as pd
import json
import random


def preprocess_data_summary():
    # Load the Excel data into a pandas dataframe
    reviews_df = pd.read_excel('../data/Reviews2Movielens.xlsx').drop(['asin'], axis=1)
    # Filter rows where 'asin' and 'movielens_id' are not null
    # use 'canon_asin' as 'asin'
    reviews_df = reviews_df[reviews_df['canon_asin'].notnull() & reviews_df['movielens_id'].notnull()].reset_index(
        drop=True)
    reviews_df.rename(columns={'canon_asin': 'asin'}, inplace=True)
    reviews_df.head()

    # Load the CSV data into a pandas dataframe
    movies_df = pd.read_csv('../data/movies_with_summary.csv').drop(['index'], axis=1)
    movies_df.rename(columns={'movieId': 'movielens_id'}, inplace=True)
    movies_df.head()

    merged_movie_df = reviews_df.merge(movies_df, on='movielens_id').sort_values(
        by=['movielens_id']).drop_duplicates().reset_index(drop=True)
    # merged_movie_df = merged_movie_df.drop(['canon_asin', 'imdb_url'])
    merged_movie_df.to_csv('../data/merged_asin_movielens_summary.csv', index=False)


def retrieve_data_reddit(reddit_data, movie_dict):
    # Extend reddit data with movie details
    extended_reddit_data = []
    for item in reddit_data:
        query_full = item['request']['query_full']
        results = []
        for result in item['response']['result']:
            asin = result['asin']
            movie = movie_dict.get(asin, {})

            # Get the summary of the movie
            summary = movie.get('summary')

            # Only include the movie if the summary is not None or 'NaN'
            if summary is not None and not isinstance(summary, float) and summary.lower() != 'nan':
                results.append({
                    'score': result['score'],
                    'asin': asin,
                    'title': movie.get('title'),
                    'genres': movie.get('genres'),
                    'summary': summary
                })

        extended_reddit_data.append({
            'request': query_full,
            'results': results
        })

    # Make each list has at most 3 movies
    # Randomly select elements from extended_reddit_data
    sample_data = random.sample(extended_reddit_data, 4)
    built_context = []

    for data in sample_data:
        context_dict = {}
        context_dict["description"] = "The following is a list of movies."
        context_dict["movies"] = []

        # Select at most three random results
        sample_results = random.sample(data['results'], min(3, len(data['results'])))
        for result in sample_results:
            movie_dict = {}
            movie_dict["Title"] = result['title']
            movie_dict["Genres"] = result['genres']
            movie_dict["Description"] = result['summary']

            context_dict["movies"].append(movie_dict)

        context_dict["User query as summary"] = data['request']
        built_context.append(context_dict)

    # Print in a pretty format
    # print(json.dumps(built_context, indent=4))
    return built_context

