import datetime
import numpy as np
import pandas as pd
from pandas import DataFrame
import collaborative as CF
import content as CB


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



def main():
    """ behaviors = pd.read_parquet("./data/train/behaviors.parquet")
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
    # train_data = behaviors[behaviors["user_id"].isin(user_sample)]
    history = history[history.index.isin(user_sample)]

    # recommendations = CF.compute_recommendations_for_users(
    #     user_sample,
    #     train_data,
    #     n_recommendations=10,
    #     similarity_threshold=0.2,
    #     neighborhood_size=10,
    # )

    recommendations = CB.compute_recommendations_for_users(user_sample, 10, history, articles)

    precision = calculate_precision(recommendations, validation_data)
    print(f"Precision: {precision:.2f}") """

    curr_date = datetime.date(2023, 6, 8)
    behaviors = pd.read_parquet("./data/train/behaviors.parquet")
    history = pd.read_parquet('./data/train/history.parquet').set_index("user_id")
    articles = pd.read_parquet('./data/articles.parquet',  
                            filters=[("published_time", ">", curr_date - datetime.timedelta(days=15)), # 15 days seems best for some reason, have tried several other numbers
                                        ("total_pageviews", ">", 100000)]) # Filter by views - see data outputs at the bottom of file
    # If views are set too high (eg. 0.5 million), there will not be enough items (< 1000), resulting in an error as content.py does not handle this atm

    validation_data = pd.read_parquet("./data/validation/history.parquet").set_index("user_id")

    # Only get the users that are in both the training set and validation set
    users = np.array(list(set(history.index) & set(validation_data.index)))
    #print(users)

    # Select users at random
    user_sample = np.random.choice(users, size=1000)
    #print(user_sample)

    history = history[history.index.isin(user_sample)]
    vectorizer = CB.train_vectorizer(articles)

    recommendations = CB.compute_recommendations_for_users(user_sample, 10, history, articles, vectorizer)

    precision = calculate_precision(recommendations, validation_data)
    print(f"Precision: {precision:.2f}")


if __name__ == "__main__":
    main()


# Fra kjøring på alle brukere:
# Med pageviews > 100 000, date = 1:    0 TPs, 0.00 Precision
# Med pageviews > 100 000, date = 15:    10401 TPs, 0.09 Precision
# Med pageviews > 1, date = 15:     143 TPs, 0.00 Precision