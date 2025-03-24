import datetime
import numpy as np
import pandas as pd
from pandas import DataFrame
import content_bow as CB2
import collaborative as CF
import content_old2 as CB
from test import compute_lda_recommendations, train_lda_model


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

        # Added datatype conversion just in case earlier, probably not necessary anymore as recency was the problem
        user_recommendations = [int(rec) for rec in user_recommendations]
        ground_truth = set(int(article_id) for article_id in ground_truth)

        user_true_positives = sum(1 for rec in user_recommendations if rec in ground_truth)

        true_positives += user_true_positives
        total_recommendations += len(user_recommendations)

    print(f"Total recommendations: {total_recommendations}")
    print(f"TPs: {true_positives}")
    precision = (
        true_positives / total_recommendations if total_recommendations > 0 else 0
    )
    return precision 


def content_lda():
    pass


def collaborative():
    behaviors = pd.read_parquet("./data/train/behaviors.parquet")
    history = pd.read_parquet('./data/train/history.parquet').set_index(
        "user_id"
    )
    articles = pd.read_parquet('./data/articles.parquet')
    validation_data = pd.read_parquet("./data/validation/history.parquet").set_index(
        "user_id"
    )

    # Only get the users that are in both the training set and validation set
    users = np.array(list(set(history.index) & set(validation_data.index)))
    print(users)

    # Select users at random
    user_sample = np.random.choice(users, size=1)
    print(user_sample)
    # Limit the impressions to only those the users in the user sample have interacted with
    train_data = behaviors[behaviors["user_id"].isin(user_sample)]
    #history = history[history.index.isin(user_sample)]

    recommendations = CF.compute_recommendations_for_users(
        user_sample,
        train_data,
        n_recommendations=10,
        similarity_threshold=0.2,
        neighborhood_size=10,
    )

    #recommendations = CB.compute_recommendations_for_users(user_sample, 10, history, articles)

    precision = calculate_precision(recommendations, validation_data)
    print(f"Precision: {precision:.2f}")


def content_bow():
    validation_data = pd.read_parquet("./data/validation/history.parquet").set_index("user_id")
    history = pd.read_parquet('./data/train/history.parquet').set_index("user_id")


    # Only get the users that are in both the training set and validation set
    users = np.array(list(set(history.index) & set(validation_data.index)))

    # Select users at random
    user_sample = np.random.choice(users, size=1000)


    # Limit the impressions to only those the users in the user sample have interacted with
    history = history[history.index.isin(user_sample)]

    # Collect all article IDs read by these users
    all_article_ids = set(history["article_id_fixed"].explode())
    
    # Set initial date and articles
    # Filter articles only on recency here
    curr_date = datetime.date(2023, 6, 8)
    articles = pd.read_parquet('./data/articles.parquet')
    articles_filtered = pd.read_parquet('./data/articles.parquet',  
                            filters=[("published_time", ">", curr_date - datetime.timedelta(days=15)), # 15 days seems best for some reason, have tried several other numbers
                                        ("total_pageviews", ">", 1)])

    # Articles the user have read are combined back into the articles filtered by recency.
    # This is needed as recency filter might remove older articles read by user
    # Might be worth exploring extending this filter to the user's read articles too in the future
    read_articles = articles[articles['article_id'].isin(all_article_ids)]
    combined = pd.concat([articles_filtered, read_articles]).drop_duplicates(subset='article_id').reset_index(drop=True)

    # Create a sparse score vector for the articles in the corpus
    tfidf_matrix = CB2.train_vectorizer(combined)


    # Map article ids to indexes
    id_to_index = dict(zip(combined['article_id'], combined.index))

    recommendations_dict = {}

    # Loop through the users in the sample and generate recommendations
    for user_id in user_sample:
        article_ids = history.loc[user_id]["article_id_fixed"]
        article_ids = [id for id in article_ids if id in id_to_index]

        if not article_ids:
            continue

        # Get recommendations for user and cast to list of ints
        recommended_ids = [int(id) for id in CB2.recommend_for_user(
            article_ids, combined, tfidf_matrix, 10)]

        recommendations_dict[user_id] = recommended_ids
        print(f"Calculated recommendations for user: {user_id}")

    # Make dataframe of recs for the precision method
    recommendations_df = pd.DataFrame(
        {
            "user_id": list(recommendations_dict.keys()),
            "recommended_article_ids": list(recommendations_dict.values()),
        }
    ).set_index("user_id")

    precision = calculate_precision(recommendations_df, validation_data)
    print(f"Precision: {precision:.4f}") # Using 4 digits here as it can be quite low :-(


def main():
    method = ""
    while method != "bow" or method != "lda" or method != "col":
        method = input("Please select a method (bow, lda, col): ")
        if method == "bow":
            content_bow()
            return
        elif method == "lda":
            return
        elif method == "col":
            collaborative()
            return
        else:
            print("Error selecting reccomendation method. See main in evaluate.py")

if __name__ == "__main__":
    main()


# Fra kjøring på alle brukere (content_bow):
# Med pageviews > 100 000, date = 1:    0 TPs, 0.00 Precision
# Med pageviews > 100 000, date = 15:    10401 TPs, 0.09 Precision
# Med pageviews > 1, date = 15:     143 TPs, 0.00 Precision
    


# TODO: globale variabler