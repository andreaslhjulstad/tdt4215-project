import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame, Series
from sklearn.metrics.pairwise import cosine_similarity


def create_user_article_matrix(df: DataFrame):
    # The article_ids_clicked column has an array of article_ids as value
    # Explode the array so each article_id in the list gets it's own row
    exploded = df.explode("article_ids_clicked")
    exploded["clicked"] = 1

    # Creates a user-article matrix where the index is the user id and the columns are article IDs
    # The value for each user-article pair is 1 if the user clicked on the article, and 0 otherwise
    user_article_matrix = pd.pivot_table(
        exploded,
        values="clicked",
        index="user_id",
        columns="article_ids_clicked",
    )

    sparse_user_article_matrix = user_article_matrix.astype(
        pd.SparseDtype(float, np.nan)
    )
    return sparse_user_article_matrix


def calculate_user_similarity(user_article_matrix: DataFrame):
    """
    Calculate user similarity using cosine similarity.

    Parameters:
        user_article_matrix (pd.DataFrame): User-article matrix where rows are users and columns are articles.

    Returns:
        pd.DataFrame: DataFrame of user similarity scores.
    """
    # Calculate user similarity for pairs (i, j) using cosine similarity
    user_similarity_values = cosine_similarity(user_article_matrix.fillna(0))
    # Fill diagonal with 0 to ensure that pairs (i, i) are not counted
    np.fill_diagonal(user_similarity_values, 0)

    user_similarity_df = pd.DataFrame(
        user_similarity_values,
        index=user_article_matrix.index,
        columns=user_article_matrix.index,
    )

    return user_similarity_df


def create_neighborhood(
    user_similarity: DataFrame,
    neighborhood_threshold: float,
    neighborhood_size: int,
    picked_userid: int,
):
    """
    Create a neighborhood of similar users for a given user.

    Parameters:
        user_similarity (pd.DataFrame): DataFrame of user similarity scores.
        neighborhood_threshold (float): Threshold for user similarity.
        neighborhood_size (int): Number of similar users to include in the neighborhood.
        picked_userid (int): User ID for whom to create the neighborhood.
    Returns:
        pd.Series: Series of user IDs in the neighborhood sorted by similarity.
    """
    neighborhood = user_similarity[
        user_similarity.loc[picked_userid] > neighborhood_threshold
    ][picked_userid].sort_values(ascending=False)[:neighborhood_size]

    return neighborhood


def calculate_scores(
    similar_users_clicked: pd.DataFrame,
    neighborhood: pd.Series,
    weighted: bool = False,
    base_rating: float = 0.5,
):
    """
    Calculate scores for items based on similar users' interactions.

    Parameters:
        similar_users_clicked (pd.DataFrame): DataFrame of similar users' interactions.
        neighborhood (pd.Series): Series of user similarities.
        weighted (bool): If True, uses weighted deviation algorithm; if False, uses simple averaging.
        base_rating (float): Base rating for items (only used when weighted=True).

    Returns:
        pd.DataFrame: DataFrame of item scores sorted by score in descending order.
    """
    item_scores = {}

    for item in similar_users_clicked.columns:
        # Get the column for this specific item
        item_interactions = similar_users_clicked[item]

        if weighted:
            # Use weighted deviation method
            numerator = 0
            denominator = 0

            for user in similar_users_clicked.index:
                # Check if user interacted with the item
                if pd.notna(item_interactions[user]):
                    # User similarity weight
                    user_weight = neighborhood[user]
                    # Deviation from base rating
                    deviation = item_interactions[user] - base_rating
                    # Weighted deviation
                    numerator += user_weight * deviation
                    # Sum of absolute weights
                    denominator += abs(user_weight)

            # Prevent division by zero
            if denominator == 0:
                item_scores[item] = base_rating
            else:
                # Final score is base rating + weighted deviation
                item_scores[item] = base_rating + (numerator / denominator)
        else:
            # Use simple averaging method
            total = 0
            count = 0

            for user in similar_users_clicked.index:
                user_interaction = item_interactions[user]
                user_similarity = neighborhood[user]

                # Check if user interacted with the item (not NaN)
                if pd.notna(user_interaction):
                    total += user_similarity
                    count += 1

            # Prevent division by zero
            if count == 0:
                item_scores[item] = 0
            else:
                item_scores[item] = total / count

    # Convert to DataFrame and sort
    item_score_df = pd.DataFrame.from_dict(
        item_scores, orient="index", columns=["score"]
    )
    item_score_df.sort_values(by="score", ascending=False, inplace=True)

    return item_score_df


def get_recommendations_for_user(
    userid: int,
    n_recommendations: int,
    user_article_matrix: DataFrame,
    neighborhood: DataFrame,
):
    """
    Get recommendations for a specific user based on their neighborhood.

    Parameters:
        userid (int): User ID for whom to get recommendations.
        n_recommendations (int): Number of recommendations to return.
        user_article_matrix (pd.DataFrame): User-article matrix.
        neighborhood (pd.Series): Series of user similarities.

    Returns:
        pd.DataFrame: DataFrame of recommended articles sorted by score in descending order.
    """

    # Limit dataset to articles that similar users have clicked, and only the ones the selected user has not clicked

    userid_clicked = user_article_matrix[user_article_matrix.index == userid].dropna(
        axis=1, how="all"
    )
    similar_user_clicked = user_article_matrix[
        user_article_matrix.index.isin(neighborhood.index)
    ].dropna(axis=1, how="all")
    similar_user_clicked.drop(userid_clicked.columns, inplace=True, errors="ignore")

    # Calculate the scores
    scores = calculate_scores(similar_user_clicked, neighborhood)

    return scores.head(n_recommendations)


def compute_recommendations_for_users(
    users: ndarray,
    impressions: DataFrame,
    n_recommendations: int,
    similarity_threshold: float,
    neighborhood_size: int,
):
    """
    Compute recommendations for a list of users.

    Parameters:
        users (ndarray): Array of user IDs for whom to compute recommendations.
        impressions (pd.DataFrame): DataFrame of user interactions with articles.
        n_recommendations (int): Number of recommendations to return for each user.
        similarity_threshold (float): Threshold for user similarity.
        neighborhood_size (int): Number of similar users to include in the neighborhood.

    Returns:
        pd.DataFrame: DataFrame of recommended articles for each user.
    """
    user_article_matrix = create_user_article_matrix(impressions)
    user_similarity = calculate_user_similarity(user_article_matrix)
    recommendations_dict = {}
    for user in users:
        neighborhood = create_neighborhood(
            user_similarity, similarity_threshold, neighborhood_size, user
        )
        recommendations = get_recommendations_for_user(
            user, n_recommendations, user_article_matrix, neighborhood
        )
        recommendations_dict[user] = recommendations.index.tolist()
        print(f"Calculated recommendations for user: {user}")
    recommendations_df = pd.DataFrame(
        {
            "user_id": list(recommendations_dict.keys()),
            "recommended_article_ids": list(recommendations_dict.values()),
        }
    ).set_index("user_id")
    return recommendations_df


def main():
    behavior_data = pd.read_parquet("./data/train/behaviors.parquet")

    user_article_matrix = create_user_article_matrix(behavior_data)
    user_similarity = calculate_user_similarity(user_article_matrix)

    neighborhood = create_neighborhood(user_similarity, 0.2, 10, 10701)

    print(get_recommendations_for_user(10701, 10, user_article_matrix, neighborhood))


if __name__ == "__main__":
    main()
