import numpy as np
import pandas as pd
import datetime

from utils import create_user_article_matrix, sigmoid


def train_matrix_factorization(
    interactions: pd.DataFrame,
    n_factors: int,
    n_iterations: int,
    learning_rate: float,
    regularization: float,
):
    """
    Train a matrix factorization model using stochastic gradient descent.

    Parameters:
        interactions (pd.DataFrame): DataFrame of user-item interactions.
        n_factors (int): Number of latent factors.
        n_iterations (int): Number of iterations for SGD.
        learning_rate (float): Learning rate for SGD.
        regularization (float): Regularization parameter.

    Returns:
        tuple: Tuple containing user and item factors.
    """
    user_article_matrix = create_user_article_matrix(interactions)

    users = user_article_matrix.index
    articles = user_article_matrix.columns

    user_factor_map = {}
    # Initialize user factors with small random values from a normal distribution
    for user in users:
        user_factors = np.random.normal(scale=1.0 / n_factors, size=(n_factors,))
        user_factor_map[user] = user_factors
    # Initialize item factors
    item_factor_map = {}
    for item in articles:
        item_factors = np.random.normal(scale=1.0 / n_factors, size=(n_factors,))
        item_factor_map[item] = item_factors
    # Perform SGD
    for iteration in range(n_iterations):
        for u in users:
            for i in articles:
                user_interaction = user_article_matrix.loc[u, i]
                if not pd.isna(user_interaction):
                    user_factor = user_factor_map[u]
                    item_factor = item_factor_map[i]
                    # Compute prediction and error
                    # Use sigmoid function to scale between 0 and 1
                    pred = sigmoid(
                        np.dot(user_factor, item_factor)
                    )  # Use sigmoid function to scale between 0 and 1
                    error = user_interaction - pred
                    # Update user and item factors, applies regularization to avoid overfitting
                    user_factor_map[u] += learning_rate * (
                        error * item_factor - regularization * user_factor
                    )
                    item_factor_map[i] += learning_rate * (
                        error * user_factor - regularization * item_factor
                    )
        print("Finished iteration", iteration + 1)

    return user_factor_map, item_factor_map


def predict(user_factors, item_factors):
    """
    Predict the interaction between users and items using the trained user and item factors.

    Parameters:
        user_factors (dict): Dictionary of user factors.
        item_factors (dict): Dictionary of item factors.

    Returns:
        pd.DataFrame: DataFrame of predicted interactions.
    """
    predictions = pd.DataFrame(index=user_factors.keys(), columns=item_factors.keys())
    for user, user_factor in user_factors.items():
        for item, item_factor in item_factors.items():
            # Compute prediction using sigmoid function
            predictions.loc[user, item] = sigmoid(np.dot(user_factor, item_factor))
    return predictions


def compute_recommendations_for_users(
    users: np.ndarray,
    behaviors: pd.DataFrame,
    n_recommendations: int,
    n_factors: int,
    n_iterations: int,
    learning_rate: float,
    regularization: float,
    time_window: datetime.timedelta = None,
    current_date: datetime.datetime = None,
):
    """
    Compute recommendations for a list of users using matrix factorization.

    Parameters:
        users (np.ndarray): Array of user IDs for whom to compute recommendations.
        behaviors (pd.DataFrame): DataFrame of user interactions with articles.
        n_recommendations (int): Number of recommendations to return for each user.
        n_factors (int): Number of latent factors for matrix factorization.
        n_iterations (int): Number of iterations for SGD.
        learning_rate (float): Learning rate for SGD.
        regularization (float): Regularization parameter.
        time_window (datetime.timedelta, optional): Time window for filtering articles.
        current_date (datetime.datetime, optional): Current date for filtering articles.

    Returns:
        pd.DataFrame: DataFrame of recommended articles for each user.
    """
    if current_date is not None and time_window is not None:
        # Only load articles if date filtering is needed
        articles = pd.read_parquet("data/articles.parquet").set_index("article_id")
        articles = articles[articles["published_time"] > current_date - time_window]
        # Filter behaviors by users and articles
        filtered_behaviors = (
            behaviors[behaviors.index.isin(users)]
            .explode("article_ids_clicked")
            .loc[lambda df: df["article_ids_clicked"].isin(articles.index)]
        )
    else:
        # Just filter by users if no date filtering needed
        filtered_behaviors = behaviors[behaviors.index.isin(users)].explode(
            "article_ids_clicked"
        )

    user_factors, item_factors = train_matrix_factorization(
        filtered_behaviors,
        n_factors,
        n_iterations,
        learning_rate,
        regularization,
    )

    predictions = predict(user_factors, item_factors)

    recommendations_dict = {}

    article_ids = predictions.columns.to_numpy()
    for u in predictions.index:
        predicted_values = []
        for i in article_ids:
            predicted_values.append(predictions.loc[u, i])
        sorted_article_ids = [x for x, _ in sorted(zip(article_ids, predicted_values))]
        recommendations_dict[u] = sorted_article_ids[:n_recommendations]

    recommendations_df = pd.DataFrame(
        {
            "user_id": list(recommendations_dict.keys()),
            "recommended_article_ids": list(recommendations_dict.values()),
        }
    ).set_index("user_id")
    return recommendations_df


def main():
    behaviors = pd.read_parquet("./data/train/behaviors.parquet")
    n_users = 1000

    user_factors, item_factors = train_matrix_factorization(
        behaviors.head(n_users), 3, 10, 0.01, 0.1
    )

    predictions = predict(user_factors, item_factors)

    print(predictions)


if __name__ == "__main__":
    main()
