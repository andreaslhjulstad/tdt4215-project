import os
import datetime
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from stop_words import get_stop_words
from sklearn.metrics.pairwise import linear_kernel
import scipy.sparse as sp


def train_vectorizer(
    articles: pd.DataFrame,
    n_topics: int = 20,
    use_lda=False,
    n_iterations=10,
    n_features=1000,
):
    """
    Creates a TF-IDF or LDA matrix representing the provided articles.

     Parameters:
         articles (pd.DataFrame): Standard articles DF.
         n_topics (int): Number of topics to use for LDA, if applicable.
         use_lda (bool): Flag for toggling between methods.
         n_iterations (int): Number of max LDA iterations, default=10.
         n_features (int): Number of max features when vectorizing, default=1000.

     Returns:
         sp.csr_matrix: Matrix with vector representations of the articles.
    """

    # Creates a 'words' column that joins together the articles' body and title, etc.
    articles["words"] = (
        (articles["title"].fillna("") + " ") * 1
        + (articles["subtitle"].fillna("") + " ") * 1
        + articles["body"].fillna("")
        + " "
        + (articles["category_str"].fillna("") + " ") * 1
    )
    # Substitues some punctuation and sets everything to lower case
    articles["words"] = (
        articles["words"].str.replace(r"[.,!?-]", "", regex=True).str.lower()
    )

    # Get danish stop words
    stop_words = set(get_stop_words("danish"))

    # Custom list of 20 most common danish surnames and first names, gathered from Danmarks Statistik
    # Used to attempt to mitigate the "Nielsen"-problem - that is, last names like "Nielsen" being
    # overly represented in reccomendations as they are relative rare in the corpus.
    danish_names = [
        # First names
        "anne",
        "mette",
        "kirsten",
        "hanne",
        "anna",
        "helle",
        "maria",
        "susanne",
        "lene",
        "marianne",
        "camilla",
        "lone",
        "louise",
        "charlotte",
        "pia",
        "tina",
        "gitte",
        "ida",
        "emma",
        "julie",
        "peter",
        "michael",
        "lars",
        "thomas",
        "jens",
        "henrik",
        "søren",
        "christian",
        "martin",
        "jan",
        "morten",
        "jesper",
        "anders",
        "mads",
        "niels",
        "rasmus",
        "mikkel",
        "per",
        "kim",
        "hans",
        # Surnames
        "nielsen",
        "jensen",
        "hansen",
        "andersen",
        "pedersen",
        "christensen",
        "larsen",
        "sørensen",
        "rasmussen",
        "jørgensen",
        "petersen",
        "madsen",
        "kristensen",
        "olsen",
        "thomsen",
        "christiansen",
        "poulsen",
        "johansen",
        "møller",
        "mortensen",
        # Some additional semantic phrases used in the body text by Extra Bladet
        "split",
        "element",
    ]

    # The imported stop words and the custom names are combined into one group of stop words
    custom_stop_words = list(stop_words.union(danish_names))

    if not use_lda:
        # Create TfidfVectorizer using these stopwrods
        vectorizer = TfidfVectorizer(
            stop_words=custom_stop_words, max_features=n_features
        )
        matrix = vectorizer.fit_transform(articles["words"])
        return matrix
    else:
        # Convert documents to word count vectors
        vectorizer = CountVectorizer(
            stop_words=custom_stop_words, max_features=n_features
        )
        matrix = vectorizer.fit_transform(articles["words"])

        # Train lda model
        lda = LatentDirichletAllocation(
            n_components=n_topics, max_iter=n_iterations, random_state=1
        )
        matrix = lda.fit_transform(matrix)
        return matrix


def recommend_for_user(
    article_ids: list,
    articles: pd.DataFrame,
    matrix: sp.csr_matrix,
    use_lda: bool,
    k: int,
):
    """
    Recommend k best articles for a given user.

    Parameters:
        article_ids (list): List of article ids read by the user.
        articles (pd.DataFrame): Standard articles DF.
        matrix (sp.csr_matrix): TF-IDF or LDA vectorized representation of the articles.
        use_lda (bool): Flag for selecting either LDA or TF-IDF approach.
        k (int): Number of articles to recommend.

    Returns:
        List[int]: List of recommended article ids.
    """

    # Create tuples of (id, index) pairs
    # Then, create a dict for looking up indexes given an article_id
    a = zip(articles["article_id"], articles.index)
    id_to_index = dict(a)

    # Create list of indexes that the valid article ids map to
    indexes = [id_to_index[id] for id in article_ids if id in id_to_index]
    if len(indexes) == 0:
        raise ValueError("No articles found for user.")

    # Create user feature vector by averaging the scores for all the articles they have read
    user_vector = []
    if not use_lda:
        user_vector = matrix[indexes].mean(axis=0)
        user_vector = sp.csr_matrix(
            user_vector
        )  # Convert from np.matrix to sparse (for the cosine sim func)
    else:
        user_vector = np.mean(matrix[indexes], axis=0).reshape(1, -1)

    # Calculate cosine similarity (here: linear kernel) between user vector and the "universal" matrix
    # Then, exclude already read articles
    scores = linear_kernel(user_vector, matrix).flatten()
    filtered_scores = [i for i in range(len(articles)) if i not in indexes]
    scores = scores[filtered_scores]

    # Get the indexes of the k best scores
    # Then sort them by score, decending
    top_k = np.argpartition(scores, -k)[-k:]
    top_k = top_k[np.argsort(scores[top_k])[::-1]]

    # Map the top_k indexes (filtered) back to their original indexes (similarity scores)
    # This is necessary because 'scores' is a filtered subset of the original similarity array
    top_k = [filtered_scores[i] for i in top_k]

    # Return just the article ids
    return articles.iloc[top_k]["article_id"].tolist()


def compute_recommendations_for_users(
    users: np.ndarray,
    n_recommendations: int,
    n_topics: int = 20,
    n_days=15,
    use_lda=False,
    n_features=1000,
    n_iterations=10,
):
    """
    Recommends n_recommendations articles for a given set of users based on the similarity of articles they have read.

    Parameters:
        n_recommendations (int): Number of recommendations to compute for each user.
        n_topics (int): Number of topics to use for LDA, if applicable.
        n_days (int): Number of days to include (backwards from set date) in recency filter, default=15.
        use_lda (bool): Boolean to decide whether to use LDA or BoW, default=False.
        n_features (int): Number of max features when vectorizing, default=1000.
        n_iterations (int): Number of max LDA iterations, default=10.

    Returns:
        pd.dataframe: Dataframe of recommended articles.
    """

    curr_date = datetime.date(2023, 6, 8)
    history = pd.read_parquet("./data/train/history.parquet").set_index("user_id")
    articles = pd.read_parquet("./data/articles.parquet")
    articles_filtered = pd.read_parquet(
        "./data/articles.parquet",
        filters=[
            ("published_time", ">", curr_date - datetime.timedelta(days=n_days)),
        ],
    )

    # Limit the impressions to only those the users in the user sample have interacted with
    history = history[history.index.isin(users)]

    # Collect all article IDs read by these users
    all_article_ids = set(history["article_id_fixed"].explode())

    # Articles the users have read are combined back into the articles filtered by recency.
    # This is needed as recency filter might remove older articles read by the users.
    # Might be worth exploring extending this filter to the user's read articles too in the future
    read_articles = articles[articles["article_id"].isin(all_article_ids)]
    combined = (
        pd.concat([articles_filtered, read_articles])
        .drop_duplicates(subset="article_id")
        .reset_index(drop=True)
    )

    # Make the vector representation of the articles
    matrix = []
    if use_lda:
        matrix = train_vectorizer(
            combined,
            n_topics,
            use_lda=use_lda,
            n_features=n_features,
            n_iterations=n_iterations,
        )
    else:
        # Create a sparse score vector for the articles in the corpus
        matrix = train_vectorizer(combined, n_features=n_features)

    # Map article ids to indexes
    id_to_index = dict(zip(combined["article_id"], combined.index))

    # Loop through the users in the sample and generate recommendations
    recommendations_dict = {}
    for user_id in users:
        article_ids = history.loc[user_id]["article_id_fixed"]
        article_ids = [id for id in article_ids if id in id_to_index]

        if not article_ids:
            continue

        # Get recommendations for user and cast to list of ints
        recommended_ids = [
            int(id)
            for id in recommend_for_user(
                article_ids, combined, matrix, use_lda, n_recommendations
            )
        ]

        recommendations_dict[user_id] = recommended_ids
        if "DEBUG" in os.environ: print(f"Calculated recommendations for user: {user_id}")

    # Make dataframe of recs to standarize for calculating precision, etc.
    recommendations_df = pd.DataFrame(
        {
            "user_id": list(recommendations_dict.keys()),
            "recommended_article_ids": list(recommendations_dict.values()),
        }
    ).set_index("user_id")
    return recommendations_df


def main():
    """
    Example usage of the content recommendation method.
    """

    user = 13538
    user_ar = []
    user_ar.append(user)

    recommendations_df = compute_recommendations_for_users(user_ar, 10, 40, 15, True)
    print(recommendations_df)


if __name__ == "__main__":
    main()
