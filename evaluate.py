import numpy as np
import pandas as pd
from pandas import DataFrame
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


def main():
    behaviors = pd.read_parquet("./data/train/behaviors.parquet")
    validation_data = pd.read_parquet("./data/validation/history.parquet").set_index(
        "user_id"
    )

    # Only get the users that are in both the training set and validation set
    users = np.array(list(set(behaviors["user_id"]) & set(validation_data.index)))

    # Select users at random
    user_sample = np.random.choice(users, size=1000)
    # Limit the impressions to only those the users in the user sample have interacted with
    train_data = behaviors[behaviors["user_id"].isin(user_sample)]

    recommendations = CF.compute_recommendations_for_users(
        user_sample,
        train_data,
        n_recommendations=10,
        similarity_threshold=0.2,
        neighborhood_size=10,
    )

    precision = calculate_precision(recommendations, validation_data)
    print(f"Precision: {precision:.2f}")


if __name__ == "__main__":
    main()
