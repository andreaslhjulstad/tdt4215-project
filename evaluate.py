import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import ndcg_score
import collaborative as CF


def calculate_precision(recommendations: DataFrame, actual: DataFrame):
    true_positives = 0
    total_recommendations = 0

    for user_id in recommendations.index:
        user_recommendations = recommendations.loc[user_id]["recommended_article_ids"]
        # Get the user's clicked articles from validation data
        ground_truth = actual.loc[user_id]["article_id_fixed"]

        user_true_positives = sum(
            1 for rec in user_recommendations if rec in ground_truth
        )

        true_positives += user_true_positives
        total_recommendations += len(user_recommendations)

    print(f"Total recommendations: {total_recommendations}")
    print(f"TPs: {true_positives}")
    precision = (
        true_positives / total_recommendations if total_recommendations > 0 else 0
    )
    return precision


def dcg(relevances):
    dcg_val = relevances[0]  # First value not discounted
    # Add discounted values (from i = 2)
    for i, rel in enumerate(relevances[1:], start=2):
        dcg_val += rel / np.log2(i + 1)
    return dcg_val


def calculate_ndcg(recommendations: DataFrame, actual: DataFrame):
    ndcg_scores = []  # Initialize a list to store nDCG for all users
    for user in recommendations.index:
        user_recommendations = recommendations.loc[user]["recommended_article_ids"]

        if len(user_recommendations) < 2:
            continue  # Skip users less than 2 recommendations, as it doesn't make sense to calculate nDCG with only 1 or 0 items

        clicked_articles = actual.loc[user]["article_id_fixed"]

        prediction_vector = [
            1 if article in clicked_articles else 0 for article in user_recommendations
        ]
        ideal_ground_truth = sorted(prediction_vector, reverse=True)

        dcg_val = dcg(prediction_vector)
        idcg_val = dcg(ideal_ground_truth)

        user_ndcg = dcg_val / idcg_val if idcg_val > 0 else 0

        ndcg_scores.append(user_ndcg)
        print(f"Calculated nDCG for user: {user} at {user_ndcg:.4f}")
    # Calculate the average nDCG across all users
    system_ndcg = np.mean(ndcg_scores)
    return system_ndcg


def main():
    behaviors = pd.read_parquet("./data/train/behaviors.parquet").set_index("user_id")
    validation_data = pd.read_parquet("./data/validation/history.parquet").set_index(
        "user_id"
    )

    # Only get the users that are in both the training set and validation set
    users = np.array(list(set(behaviors.index) & set(validation_data.index)))

    # Select users at random
    user_sample = np.random.choice(users, size=1000)
    # Limit the impressions to only those the users in the user sample have interacted with
    # train_data = behaviors[behaviors["user_id"].isin(user_sample)]

    recommendations = CF.compute_recommendations_for_users(
        user_sample,
        behaviors,
        n_recommendations=10,
        similarity_threshold=0.2,
        neighborhood_size=10,
    )

    precision = calculate_precision(recommendations, validation_data)

    ndcg = calculate_ndcg(recommendations, validation_data)

    print(f"Precision: {precision:.2f}")
    print(f"nDCG: {ndcg:.4f}")


if __name__ == "__main__":
    main()
