import os
import pandas as pd
import numpy as np
import datetime


def baseline(data_path: str, curr_date: datetime):
    """
    Deprecated method. Use compute_recommendations_for_users() instead.
    """
    # Retrieve the articles from the past 3 days (including current date)
    df = pd.read_parquet(data_path,
                         filters=[('total_pageviews', '>', 0), ("published_time", ">", curr_date - datetime.timedelta(days=2))])
    df = df.reset_index().set_index('article_id')

    # Sort by pageviews (published time as tiebreaker) and
    df = df.sort_values(
        by=["total_pageviews", "published_time"], inplace=False, ascending=False)
    df = df.head(20)

    # Create new column that calculates the average read time per view
    df["read_time_per_view"] = df["total_read_time"]/df["total_pageviews"]

    # Normalize scores to be between 0 and 1
    max_read_time = df["read_time_per_view"].max()
    df["score"] = df["read_time_per_view"] / max_read_time

    # Sort by score
    df = df.sort_values(by=["score"], inplace=False, ascending=False)

    # Return dataframe with article IDs as index and scores
    return df[["title", "score"]]


def compute_recommendations_for_users(users: np.ndarray, curr_date: datetime, n_days=3, n_recommendations=10):
    """
    Recommends n_recommendations same articles for any set of users based on popularity metrics.

    Parameters:
        users (np.npdarray): List of users to recommend for, only used to standardize returned dataframe.
        curr_date (datetime): Starting date for recommendations.
        n_recommendations (int): Number of recommendations to compute for each user, default=10.
        n_days (int): Number of days to include (backwards from set date) in recency filter, default=3.

    Returns:
        pd.dataframe: Dataframe of recommended articles.
    """

    # Retrieve the articles from the past n_days days
    df = pd.read_parquet("./data/articles.parquet",
                         filters=[('total_pageviews', '>', 0), ("published_time", ">", curr_date - datetime.timedelta(days=n_days))])

    # Sort by pageviews (published time as tiebreaker) and
    df = df.sort_values(
        by=["total_pageviews", "published_time"], inplace=False, ascending=False)
    df = df.head(2*n_recommendations)

    # Create new column that calculates the average read time per view
    df["read_time_per_view"] = df["total_read_time"]/df["total_pageviews"]

    # Sort by new column to recommend the most "gripping" articles
    df = df.sort_values(by=["read_time_per_view"],
                        inplace=False, ascending=False)

    # Select only best elements
    df = df.head(n_recommendations)

    # Make standarized dataframe, even though it is a bit banal
    recommendations_df = pd.DataFrame(
        {
            "user_id": list(users),
            # Same values for all users
            "recommended_article_ids": [list(df['article_id'])] * len(users),
        }
    ).set_index("user_id")
    if "DEBUG" in os.environ:
        print(recommendations_df)
    return recommendations_df


def main():
    # The date chosen in our scenario
    curr_date = datetime.date(2023, 6, 8)

    filepath = 'data/articles.parquet'

    print(baseline(filepath, curr_date))


if __name__ == "__main__":
    main()
