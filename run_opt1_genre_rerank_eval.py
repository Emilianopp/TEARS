import numpy as np
import json


def compute_reciprocal_rank(ground_truth_path, model_results_path, user_start=None, user_end=None):
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)

    with open(model_results_path, 'r') as f:
        model_results = json.load(f)

    user_genre_rr = {}
    total_pairs = 0
    top_10_count = 0
    top_20_count = 0
    top_50_count = 0

    for user_id, genres in ground_truth.items():
        if user_start and user_end and not user_start <= int(user_id) <= user_end:
            continue

        user_genre_rr[user_id] = {}
        for genre, details in genres.items():
            total_pairs += 1
            truth_movie_id = details["movieId"]
            ranking = model_results.get(user_id, {}).get(genre, [])

            if truth_movie_id in ranking:
                rank = ranking.index(truth_movie_id) + 1  # Add 1 because index is 0-based
                rr = 1.0 / rank
                if rank <= 10:
                    top_10_count += 1
                if rank <= 20:
                    top_20_count += 1
                if rank <= 50:
                    top_50_count += 1
            else:
                rr = 0.0

            user_genre_rr[user_id][genre] = rr

    print(f"Total pairs: {total_pairs}")
    print(f"Top-10 ground truth pairs: {top_10_count}")
    print(f"Top-20 ground truth pairs: {top_20_count}")
    print(f"Top-50 ground truth pairs: {top_50_count}")

    return user_genre_rr




def mean_rr(user_genre_rr):
    total_rr = 0
    total_pairs = 0

    for user_id, genre_rr in user_genre_rr.items():
        for genre, rr in genre_rr.items():
            # Considering only pairs where the ground truth is in the top-20
            if rr > 0 and 1.0 / rr <= 20:
                total_rr += rr
                total_pairs += 1

    mrr = total_rr / total_pairs if total_pairs > 0 else 0.0
    return mrr


user_start, user_end = 1, 943  # Adjust this range as needed

# Compute RR for each (user, genre) pair for both models
bprmf_user_genre_rr = compute_reciprocal_rank("../data_preprocessed/ml-100k/data_split/test_set_leave_one.json",
                                              "../saved_model/ml-100k/BPRMF_user_genre_rankings.json", user_start,
                                              user_end)
gpt_bprmf_user_genre_rr = compute_reciprocal_rank("../data_preprocessed/ml-100k/data_split/test_set_leave_one.json",
                                                  "../saved_model/ml-100k/GPT_BPRMF_user_genre_rankings_all.json",
                                                  user_start, user_end)
# print("gpt_bprmf_user_genre_rr:", gpt_bprmf_user_genre_rr)


# Compute MRR for the two models
bprmf_mrr = mean_rr(bprmf_user_genre_rr)
gpt_bprmf_mrr = mean_rr(gpt_bprmf_user_genre_rr)

print(f"BPRMF MRR: {bprmf_mrr:.4f}")
print(f"GPT + BPRMF MRR: {gpt_bprmf_mrr:.4f}")

# If you want to print RR for each (user, genre) pair:
target_user_genre_pairs = []
for user_id, genre_rr in bprmf_user_genre_rr.items():
    for genre, rr in genre_rr.items():
        # Only print pairs where the ground truth is in the top-20
        if rr > 0 and 1.0 / rr <= 20:
            print(f"User {user_id}, Genre {genre}, BPRMF RR: {rr:.4f}")
            target_user_genre_pairs.append((user_id, genre))
            print(f"User {user_id}, Genre {genre}, GPT + BPRMF RR: {gpt_bprmf_user_genre_rr[user_id][genre]:.4f}")

