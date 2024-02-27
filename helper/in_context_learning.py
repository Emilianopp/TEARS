import json
from helper.text_generation import generate_text


def in_context_user_summary(built_context, user_id, genre, model_name, device):
    # Load the training set JSON file
    with open(
            '/home/haolun/projects/ctb-lcharlin/haolun/LLM4Rec_User_Summary/data_preprocessed/ml-1m/train_set_leave_one.json') as input_file:
        user_data_input = json.load(input_file)

    # Get the movies for the specified user ID and genre
    movies = user_data_input.get(str(user_id), {}).get(genre, [])[-5:]  # only use the last 5 movies due to length limit

    # Prepare the prompt with example pairs
    prompt = "Task: Based on the list of movies, generate a user summary as the user preference. The following are some examples:\n"

    # Add demonstration examples to the prompt
    for demo in built_context:
        # prompt += f"\n The following is an example:"
        for movie in demo['movies']:
            title = movie['Title']
            description = movie['Description'].split(".")[0]  # Use the first sentence as the short description
            prompt += f"\n{title}: {description}"
        prompt += f"\nSummary: {demo['User query as summary']}\n"

    # Add movies for the specified user ID and genre to the prompt
    prompt += f"\nNow, based on the above examples, please generate user summary for the following movies watched by new user id: {user_id}, of genre: {genre}\n: "
    for movie in movies:
        title = movie['title']
        description = movie['summary'].split(".")[0]  # Use the first sentence as the short description
        prompt += f"\n{title}: {description}"
    prompt += "\nGenerate user summary within one sentence: "

    # Print the final prompt\
    print("************************")
    print("Prompt for user summary generation:\n-----\n", prompt)
    print("************************")

    user_summary = generate_text(prompt, model_name, device)

    # Loop over the decoded outputs and store them in a list
    user_summary_list = []
    for i, decoded_output in enumerate(user_summary):
        decoded_output = decoded_output.replace('\n', ' ')
        user_summary_list.append(decoded_output)

    # Print the list of decoded outputs
    print("Generated summaries:")
    for i, decoded_output in enumerate(user_summary_list):
        print(f"\nGenerated summary {i + 1}:\n{decoded_output}")

    return user_summary_list


def in_context_recommendation(built_context, user_id, genre, user_summary_list, model_name, device):
    # Load the training set JSON file
    with open(
            '/home/haolun/projects/ctb-lcharlin/haolun/LLM4Rec_User_Summary/data_preprocessed/ml-1m/train_set_leave_one.json') as input_file:
        user_data_input = json.load(input_file)

    # Get the movies for the specified user ID and genre
    movies = user_data_input.get(str(user_id), {}).get(genre, [])[-5:]  # only use the last 5 movies due to length limit

    # Prepare the prompt with example pairs
    prompt = "Task: Based on the user's preference summary, generate a list of recommended movies. " \
             "The output format should be {movie title}: {movie description}. The following are some examples:\n"

    # Add demonstration examples to the prompt
    for demo in built_context:
        summary = demo['User query as summary']
        prompt += f"\nSummary: {summary}\n"
        for movie in demo['movies']:
            title = movie['Title']
            description = movie['Description'].split(".")[0]  # Use the first sentence as the short description
            prompt += f"{title}: {description}\n"

    # Add user summary to the prompt
    # prompt += f"\nNew Task\nSummary: {user_summary_list[3]}\nRecommended movie titles and descriptions: "
    prompt += f"\nNow, following the above examples, please recommend movies for the new summary: \n\nSummary: {user_summary_list[0]}\n"

    # Print the final prompt
    print("************************")
    print("Prompt for user recommendation generation:\n-----\n", prompt)
    print("************************")

    decoded_outputs = generate_text(prompt, model_name, device)

    # Loop over the decoded outputs and print them
    for i, decoded_output in enumerate(decoded_outputs):
        decoded_output = decoded_output.replace('\n', ', ')
        print(f"\nRecommended movie titles and descriptions:\n{decoded_output}")


def in_context_retrieval(built_context, user_id, genre, user_summary_list, movie_candidates, model_name, device):
    # Load the training set JSON file
    with open(
            '/home/haolun/projects/ctb-lcharlin/haolun/LLM4Rec_User_Summary/data_preprocessed/ml-1m/train_set_leave_one.json') as input_file:
        user_data_input = json.load(input_file)

    # Get the movies for the specified user ID and genre
    watched_movies = user_data_input.get(str(user_id), {}).get(genre, [])[
                     -20:]  # only use the last movie titles due to length limit

    # Prepare the prompt with example pairs
    prompt = "Task: Based on the user's preference summary, retrievel top-10 most relevant movies from the movie candidates in a ranked order, where the top movies should be more relevant. "
    prompt += "The output template should be: \"{movie title 1}\", \"{movie title 2}\", \"{movie title 3}\", ... \n"

    # # Add demonstration examples to the prompt
    # for demo in built_context:
    #     summary = demo['User query as summary']
    #     prompt += f"\nSummary: {summary}\n"
    #     prompt += f"Related movie titles:"
    #     for movie in demo['movies']:
    #         title = movie['Title']
    #         prompt += f"{title},"
    #
    # Format movie candidates for the prompt
    formatted_movie_candidates = ', '.join(f'\"{movie}\"' for movie in movie_candidates)

    # Add user summary and movie candidates to the prompt
    prompt += f"\n\nUser preference summary: {user_summary_list[0]}\n"
    prompt += f"\nMovies watched by the user: \n"
    for movie in watched_movies:
        title = movie['title']
        prompt += f"\"{title}\", "

    prompt += f"\n\nNow, you should only select top-10 most relevant movie titles exisiting in the movie candidates for the given user summary. \nMovie candidates: {formatted_movie_candidates}\n"

    # Print the final prompt
    print("************************")
    print("***Prompt for user retrieval generation:***\n-----\n", prompt)
    print("************************")

    decoded_outputs = generate_text(prompt, model_name, device)

    # Loop over the decoded outputs and print them
    for i, decoded_output in enumerate(decoded_outputs):
        decoded_output = decoded_output.replace('\n', ', ')
        print(f"\nRetrieved movie titles:\n{decoded_output}")
