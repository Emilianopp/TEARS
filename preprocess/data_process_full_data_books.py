def save_sorted_json(data, filename):
    sorted_data = OrderedDict(sorted(data.items(), key=lambda t: t[0]))
    with open(filename, 'w') as f:
        json.dump(sorted_data, f, indent=4)


def extract_unique_movies(user_data):
    movies = set()
    for user_movies in user_data.values():
        for genre_movies in user_movies.values():
            if isinstance(genre_movies, list):  # normal train, valid, test set
                for movie in genre_movies:
                    movies.add(movie['title'])
            else:  # leave-one-out valid and test set
                movies.add(genre_movies['title'])
    return movies


def extract_user_genres(data):
    """Extract genres a user has interacted with."""
    user_genres = {}

    for user_id, genres in data.items():
        user_genres[user_id] = list(genres.keys())

    return user_genres

def load_movie_titles_from_dat(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().decode('latin-1')
    movie_titles = set()
    for line in lines:
        tokens = line.split("::")
        title = tokens[1]
        movie_titles.add(title)
    return movie_titles

def filter_movie_titles_by_valid_set(movies, valid_item_names):
    return [movie for movie in movies if movie['title'] in valid_item_names]

def sample_random(user_movies):
    validation_movie = user_movies.sample(n=1)

    user_movies = user_movies.drop(validation_movie.index)
    

    # Randomly choose one movie for test and remove it from user's ratings
    test_movie = user_movies.sample(n=1)
    user_movies = user_movies.drop(test_movie.index)
    return validation_movie,test_movie,user_movies

def sample_most_recent(user_movies):
    # Sort the DataFrame by the 'timestamp' column in descending order
    user_movies = user_movies.sort_values(by='timestamp', ascending=False)

    # Sample the two most recent movies
    most_recent_movies = user_movies.head(2)

    # Drop the two most recent movies from the user's ratings
    user_movies = user_movies.drop(most_recent_movies.index)

    # Split the two most recent movies into validation_movie and test_movie
    validation_movie = most_recent_movies.head(1)
    test_movie = most_recent_movies.tail(1)

    return validation_movie, test_movie, user_movies

    

def split_and_filter_ratings(user_movies, rating_threshold=4):
    # Sort the DataFrame by the 'timestamp' column in descending order and select the first 52 movies
    user_movies = user_movies.sort_values(by='timestamp', ascending=False) 
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
if __name__ == '__main__':
    k = 20  # threshold for history length

    print(f"{data_name=}")
    if data_name == 'ml-1m':
        k = 10  # threshold for history length

        valid_item_names = pd.read_csv(f'../data/ml-1m/movies.dat',encoding='ISO-8859-1',sep='::',header=None).iloc[:,1].tolist()
        ratings_file = '../data/ml-1m/ratings.dat'
        separator = "::"
        header = None
        rating_columns = ['userId', 'itemId', 'rating', 'timestamp']
        movie_metadata_file = '../data/ml-1m/movies.dat'
        ratings = pd.read_csv(ratings_file, sep=separator, header=header, names=rating_columns)
        
        # Load movie metadata
        movie_metadata = pd.read_csv(movie_metadata_file,encoding='ISO-8859-1',sep='::',header=None)
        movie_metadata.columns = ['movielens_id','title','genre']


        ratings = ratings.merge(movie_metadata, left_on='itemId', right_on='movielens_id', how = 'left')
        # display(ratings)
        
        train_data, val_data, test_data,promp_set,non_users = generate_train_val_test_splits(ratings, k)

    elif data_name == 'books':
        ratings_file = '../data/books/ratings.csv'
        ratings = pd.read_csv(ratings_file)
        ratings.rename(columns={'book_id':'itemId','review/time':'timestamp','Title':'title','review/score':'rating','User_id':'userId','categories':'genres'}, inplace=True)

        valid_item_names = ratings.title.unique().tolist()
        #rename the bookId column to itemId
        train_data, val_data, test_data,promp_set,non_users = generate_train_val_test_splits(ratings, k)
    elif data_name == 'goodbooks':
        ratings_file = '../data/goodbooks/ratings.csv'
        ratings = pd.read_csv(ratings_file)
        #remove cold start items
        ratings = ratings[ratings['book_id'].isin(ratings['book_id'].value_counts()[ratings['book_id'].value_counts()>10].index)]
        
        ratings.rename(columns={'book_id':'itemId','Title':'title','user_id':'userId'}, inplace=True)
        valid_item_names = ratings.title.unique().tolist()
        #rename the bookId column to itemId
        train_data, val_data, test_data,promp_set,non_users = generate_train_val_test_splits(ratings, k)
    elif data_name == 'netflix':
        ratings_file = '../data/netflix/ratings.csv'
        ratings = pd.read_csv(ratings_file)
        ratings.rename(columns={'MovieID':'itemId','Title':'title','CustomerID':'userId','Name':'title','Rating':'rating','Date':'timestamp'}, inplace=True)
        print(f"{ratings.columns=}")
        valid_item_names = ratings.title.unique().tolist()
        #rename the bookId column to itemId
        train_data, val_data, test_data,promp_set,non_users = generate_train_val_test_splits(ratings, k)

        
    valid_item_names = list(valid_item_names)
    display(train_data)
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
    user_set = random.sample(list(user_set) ,500)
    strong_generalization_set = pd.concat([train_data[train_data['userId'].isin(user_set)],val_data[val_data['userId'].isin(user_set)],test_data[test_data['userId'].isin(user_set)]])

    train_data = train_data[~train_data['userId'].isin(user_set)]
    val_data = val_data[~val_data['userId'].isin(user_set)]
    test_data = test_data[~test_data['userId'].isin(user_set)]
