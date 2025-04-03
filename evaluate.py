import datetime
import os
import numpy as np
import pandas as pd
from pandas import DataFrame
import user_based as kNN
import matrixfactorization as MF
import content as CB
import baseline as BL
from codecarbon import EmissionsTracker


def calculate_precision(recommendations: DataFrame, history: DataFrame):
    """
    Calculate the precision of the recommendations.

    Parameters:
        recommendations (pd.DataFrame): DataFrame containing the recommended articles for each user.
        history (pd.DataFrame): DataFrame containing the actual clicked articles for each user.

    Returns:
        float: Precision score.
    """
    true_positives = 0
    total_recommendations = 0

    for user_id in recommendations.index:
        user_recommendations = recommendations.loc[user_id]["recommended_article_ids"]
        # Get the user's clicked articles from validation data
        ground_truth = history.loc[user_id]["article_id_fixed"]

        user_true_positives = sum(
            1 for rec in user_recommendations if rec in ground_truth
        )

        true_positives += user_true_positives
        total_recommendations += len(user_recommendations)

        # Added datatype conversion just in case earlier, probably not necessary anymore as recency was the problem
        user_recommendations = [int(rec) for rec in user_recommendations]
        ground_truth = set(int(article_id) for article_id in ground_truth)

        user_true_positives = sum(
            1 for rec in user_recommendations if rec in ground_truth
        )

        true_positives += user_true_positives
        total_recommendations += len(user_recommendations)

    if "DEBUG" in os.environ: 
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


def calculate_ndcg(recommendations: DataFrame, history: DataFrame):
    """
    Calculate the average nDCG for the recommendations.

    Parameters:
        recommendations (pd.DataFrame): DataFrame containing the recommended articles for each user.
        history (pd.DataFrame): DataFrame containing the actual clicked articles for each user.
    Returns:
        float: Average nDCG score.
    """
    ndcg_scores = []  # Initialize a list to store nDCG for all users
    for user in recommendations.index:
        user_recommendations = recommendations.loc[user]["recommended_article_ids"]

        if len(user_recommendations) < 2:
            continue  # Skip users less than 2 recommendations, as it doesn't make sense to calculate nDCG with only 1 or 0 items

        clicked_articles = history.loc[user]["article_id_fixed"]

        prediction_vector = [
            1 if article in clicked_articles else 0 for article in user_recommendations
        ]
        ideal_ground_truth = sorted(prediction_vector, reverse=True)

        dcg_val = dcg(prediction_vector)
        idcg_val = dcg(ideal_ground_truth)

        user_ndcg = dcg_val / idcg_val if idcg_val > 0 else 0

        ndcg_scores.append(user_ndcg)
        if "DEBUG" in os.environ: print(f"Calculated nDCG for user: {user} at {user_ndcg:.4f}")
    # Calculate the average nDCG across all users
    system_ndcg = np.mean(ndcg_scores)
    return system_ndcg


def main():
    # If DEBUG present, debug printing is enabled.
    # COMMENT OUT next line to disable.
    os.environ["DEBUG"] = "1"

    validation_data = pd.read_parquet("./data/validation/history.parquet").set_index(
        "user_id"
    )
    behaviors = pd.read_parquet("./data/train/behaviors.parquet").set_index("user_id")

    # Only get the users that are in both the training set and validation set
    users = np.array(list(set(behaviors.index) & set(validation_data.index)))

    recommendations_df = None
    current_date = datetime.datetime(2023, 6, 8)  # Chosen date for scenario

    # Determine sampling here!
    sample_users = False

    if sample_users:
        # Select users at random
        user_sample = np.random.choice(users, 100, replace=False)
        users = user_sample

    tracker = EmissionsTracker()
    tracker.start()

    method = ""
    while method != "bow" or method != "lda" or method != "knn" or method != "mat" or method != "bas":
        method = input("Please select a method (bow, lda, knn, mat or bas): ")
        if method == "bow":
            # Compute recommendations for the user sample
            recommendations_df = CB.compute_recommendations_for_users(
                users, 
                n_recommendations=10,
            )
            break
        elif method == "lda":
            # Compute recommendations for the user sample
            recommendations_df = CB.compute_recommendations_for_users(
                users, 
                use_lda=True, 
                n_topics=57, 
                n_recommendations=10,
            )
            break
        elif method == "knn":
            recommendations_df = kNN.compute_recommendations_for_users(
                users,
                behaviors,
                n_recommendations=10,
                similarity_threshold=0.2,
                neighborhood_size=10,
            )
            break
        elif method == "mat":
            time_window = datetime.timedelta(days=30)
            recommendations_df = MF.compute_recommendations_for_users(
                user_sample,
                behaviors,
                n_recommendations=10,
                n_factors=20,
                n_iterations=10,
                learning_rate=0.1,
                regularization=0.2,
                time_window=time_window,
                current_date=current_date,
            )
            break
        elif method == "bas":
            recommendations_df = BL.compute_recommendations_for_users(
                users,
                current_date,
                n_days=30,
            )
            break
        else:
            print("Error selecting recommendation method. See main in evaluate.py")
    emissions = float(tracker.stop())

    if recommendations_df is not None:
        # Calculate precision and ndcg based on the recs
        precision = calculate_precision(recommendations_df, validation_data)
        ndcg = calculate_ndcg(recommendations_df, validation_data)
        print(f"Emissions: {emissions} kg CO2")
        print(f"Precision: {precision:.4f}")
        print(f"nDCG: {ndcg:.4f}")


if __name__ == "__main__":
    main()
