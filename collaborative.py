import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.metrics.pairwise import cosine_similarity


def create_user_article_matrix(df: DataFrame):
    # The article_ids_clicked column has an array of article_ids as value
    # Explode the array so each article_id in the list gets it's own row
    exploded = df.explode("article_ids_clicked")

    # TODO: find better solution
    clicked_df = exploded[["user_id", "article_ids_clicked"]].copy()
    clicked_df["clicked"] = 1

    # Creates a user-article matrix where the index is the user id and the columns are article IDs
    # The value for each user-article pair is 1 if the user clicked on the article, and 0 otherwise
    user_article_matrix = pd.pivot_table(
        clicked_df,
        values="clicked",
        index="user_id",
        columns="article_ids_clicked",
    )

    sparse_user_article_matrix = user_article_matrix.astype(
        pd.SparseDtype(float, np.nan)
    )
    return sparse_user_article_matrix


def calculate_user_similarity(user_article_matrix: DataFrame):
    # Calculate user similarity for pairs (i, j) using cosine similarity
    user_similarity_values = cosine_similarity(user_article_matrix.fillna(0))
    # Fill diagonal with NaN to ensure that pairs (i, i) are not counted
    np.fill_diagonal(user_similarity_values, np.nan)

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
    neighborhood = user_similarity[
        user_similarity[picked_userid] > neighborhood_threshold
    ][picked_userid].sort_values(ascending=False)[:neighborhood_size]

    return neighborhood


def calculate_scores(similar_users_clicked: DataFrame, neighborhood: Series):
    # Calculating weighted scores based on articles similar users have clicked
    item_score = {}

    for i in similar_users_clicked.columns:
        article = similar_users_clicked[i]
        total = 0
        count = 0
        for u in similar_users_clicked.index:
            user_interaction = article[u]  # 1 if user clicked, NaN otherwise
            user_similarity = neighborhood[u]
            article_clicked = pd.isna(user_interaction) == False
            if article_clicked:
                score = (
                    user_similarity * user_interaction
                )  # Score is weighted based on how similar the user is
                total += score
                count += 1
        item_score[article.name] = total / count

    # Convert dictionary to a dataframe
    item_score = pd.DataFrame(
        item_score.values(), columns=["score"], index=item_score.keys()
    )
    item_score.sort_values(by="score", inplace=True, ascending=False)

    return item_score


def get_recommendations_for_user(
    userid: int,
    n_recommendations: int,
    impressions: DataFrame,
    similarity_threshold: float,
    neighborhood_size: int,
):
    user_article_matrix = create_user_article_matrix(impressions)

    user_similarity = calculate_user_similarity(user_article_matrix)

    neighborhood = create_neighborhood(
        user_similarity, similarity_threshold, neighborhood_size, userid
    )

    # Limit dataset to articles that similar users have clicked, and only the ones the selected user has not clicked
    userid_clicked = user_article_matrix[user_article_matrix.index == userid].dropna(
        axis=1, how="all"
    )
    similar_user_clicked = user_article_matrix[
        user_article_matrix.index.isin(neighborhood.index)
    ].dropna(axis=1, how="all")
    similar_user_clicked.drop(userid_clicked.columns, inplace=True, errors="ignore")

    scores = calculate_scores(similar_user_clicked, neighborhood)

    return scores.head(n_recommendations)


def main():
    behavior_data = pd.read_parquet("./data/train/behaviors.parquet")
    print(get_recommendations_for_user(10701, 10, behavior_data, 0.2, 10))


if __name__ == "__main__":
    main()
