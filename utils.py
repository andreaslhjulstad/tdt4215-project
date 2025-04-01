import pandas as pd
import numpy as np


def create_user_article_matrix(behaviors: pd.DataFrame):
    """
    Create a user-article matrix from the given DataFrame.

    Parameters:
        behaviors (pd.DataFrame): DataFrame with user behaviors.
    Returns:
        pd.DataFrame: Returns a sparse DataFrame where the index is user_id and the columns are article IDs.
    """
    # The article_ids_clicked column has an array of article_ids as value
    # Explode the array so each article_id in the list gets it's own row
    exploded = behaviors.explode("article_ids_clicked")
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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
