import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def create_user_article_matrix(df: pd.DataFrame):
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


class MatrixFactorization:

    def __init__(
        self,
        n_factors: int,
        n_iterations: int,
        learning_rate: float,
        regularization: float,
    ):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.regularization = regularization

    def predict(self, interactions: pd.DataFrame):
        user_article_matrix = create_user_article_matrix(interactions)

        print(user_article_matrix)

        users = user_article_matrix.index
        articles = user_article_matrix.columns

        user_factor_map = {}
        for user in users:
            user_factors = np.random.normal(
                scale=1.0 / self.n_factors, size=(self.n_factors,)
            )
            user_factor_map[user] = user_factors
        item_factor_map = {}
        for item in articles:
            item_factors = np.random.normal(
                scale=1.0 / self.n_factors, size=(self.n_factors,)
            )
            item_factor_map[item] = item_factors
        for _ in range(self.n_iterations):
            for u in users:
                for i in articles:
                    user_interaction = user_article_matrix.loc[u][i]
                    if not pd.isna(user_interaction):
                        user_factor = user_factor_map[u]
                        item_factor = item_factor_map[i]
                        pred = sigmoid(
                            np.dot(user_factor, item_factor)
                        )  # Use sigmoid function to scale between 0 and 1
                        error = user_interaction - pred
                        user_factor_map[u] += self.learning_rate * (
                            error * item_factor - self.regularization * user_factor
                        )
                        item_factor_map[i] += self.learning_rate * (
                            error * user_factor - self.regularization * item_factor
                        )
            print("FINISHED ITERATION", _)

        predictions = pd.DataFrame(
            index=users,
            columns=articles,
            dtype=float,
        )

        # Fill the prediction matrix
        for user_id in users:
            user_factor = user_factor_map[user_id]
            for article_id in articles:
                item_factor = item_factor_map[article_id]
                predictions.loc[user_id, article_id] = sigmoid(
                    np.dot(user_factor, item_factor)
                )

        return predictions


def compute_recommendations_for_users(
    users,
    behaviors: pd.DataFrame,
    n_recommendations,
    n_factors,
    n_iterations,
    learning_rate,
    regularization,
):
    filtered_behaviors = behaviors[behaviors.index.isin(users)]

    matrix_factorization = MatrixFactorization(
        n_factors, n_iterations, learning_rate, regularization
    )

    predictions = matrix_factorization.predict(filtered_behaviors)

    recommendations_dict = {}

    article_ids = predictions.columns.to_numpy()
    for u in predictions.index:
        predicted_values = []
        for i in article_ids:
            predicted_values.append(predictions.loc[u][i])
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
    n_users = 100

    matrix_factorization = MatrixFactorization(3, 10, 0.01, 0.1)

    predictions = matrix_factorization.predict(behaviors.head(n_users))

    print(predictions)


if __name__ == "__main__":
    main()
