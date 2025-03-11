
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from stop_words import get_stop_words
from sklearn.metrics.pairwise import linear_kernel
import scipy.sparse as sp 
import matplotlib.pyplot as plt




def get_bag_of_words_vector(article_id: int, articles: pd.DataFrame, vectorizer: TfidfVectorizer):
    article_row = articles.loc[articles['article_id'] == article_id].copy()
    if article_row.empty:
        return sp.csr_matrix((1, 1000))
    
    article_row['words'] = (
    (article_row['title'].fillna('') + ' ') * 3 +  
    (article_row['subtitle'].fillna('') + ' ') * 2 +  
    article_row['body'].fillna('') + ' ' +
    (article_row['category_str'].fillna('') + ' ') * 2
    )
    #article_row['words'] = article_row['title'].fillna('') + ' ' + article_row['subtitle'].fillna('') + ' ' + article_row['body'].fillna('') + ' ' + article_row['category_str'].fillna('')
    return vectorizer.transform(article_row['words'])



def train_vectorizer(articles: pd.DataFrame):
    #articles['words'] = articles['title'].fillna('') + ' ' + articles['subtitle'].fillna('') + ' ' + articles['body'].fillna('') + ' ' + articles['category_str'].fillna('')
    articles['words'] = (
    (articles['title'].fillna('') + ' ') * 3 +  
    (articles['subtitle'].fillna('') + ' ') * 2 + 
    articles['body'].fillna('') + ' ' +
    (articles['category_str'].fillna('') + ' ') * 2
)
    vectorizer = TfidfVectorizer(stop_words=get_stop_words('danish'), max_features=1000)
    vectorizer.fit(articles['words'])
    return vectorizer


def calculate_user_feature_vector(user_id: int, history: pd.DataFrame, articles: pd.DataFrame, vectorizer: TfidfVectorizer):
    user_row = history.loc[user_id]
    article_ids = user_row["article_id_fixed"]
    
    article_vectors = [get_bag_of_words_vector(article_id, articles, vectorizer) for article_id in article_ids]
    
    if not article_vectors:
        return sp.csr_matrix((1, 1000))  # Returns empty matrix

    all_vectors = sp.vstack(article_vectors) # vstack is supposedly more optimized

    avg_sparse = all_vectors.mean(axis=0)
    return sp.csr_matrix(avg_sparse)

def get_recommendations_from_user_feature_vector(user_feature_vector: sp.csr_matrix, articles: pd.DataFrame, vectorizer: TfidfVectorizer, k: int):
    articles = articles.copy()  # Copy articles to avoid console warning
    article_vectors = vectorizer.transform(articles['words'])
    scores = cosine_similarity(user_feature_vector, article_vectors).flatten()

    # Choose and return top k articles
    top_k_indices = scores.argpartition(-k)[-k:]
    top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]
    top_scores = articles.iloc[top_k_indices]['article_id'].to_list()
    return top_scores


def compute_recommendations_for_users(users: np.ndarray, n_recommendations: int, history: pd.DataFrame, articles: pd.DataFrame, vectorizer: TfidfVectorizer):
    recommendations_dict = {}

    for user in users:
        read_articles = set(history.loc[user]["article_id_fixed"])

        user_feature_vector = calculate_user_feature_vector(user, history, articles, vectorizer)

        # Filter out articles read by user
        filtered_articles = articles[~articles["article_id"].isin(read_articles)]

        reccs = get_recommendations_from_user_feature_vector(user_feature_vector, filtered_articles, vectorizer, n_recommendations)
        print(f"Calculated recommendations for user: {user}")
        
        recommendations_dict[user] = reccs

    recommendations_df = pd.DataFrame({"user_id": list(recommendations_dict.keys()), "recommended_article_ids": list(recommendations_dict.values())}).set_index("user_id")
    return recommendations_df


# 1. History i train set, articles_interacted_with-liste
# 2. For hver artikkel, slå opp i articles og vectorize bag of words
# 3. Lagre dette, regn ut average av disse, dett er user feature vector
# 4. Bruk den for basis til å sammenligne med nye artikler (f.eks. cosine similarity) (bag of words director's cut på alle artikler (som blir item feature vector))
# 5. Anbefale artikler etter sortering på score (fjern artikler som brukeren allerede)

# Eksempel:

def main():
    # pd.set_option('display.max_columns', None)
    user_id = 13538
    k = 10

    history = pd.read_parquet('data/train/history.parquet')
    articles = pd.read_parquet('data/articles.parquet')

    # Tren vectorizer én gang
    vectorizer = train_vectorizer(articles)

    user_feature_vector = calculate_user_feature_vector(user_id, history, articles, vectorizer)
    reccs = get_recommendations_from_user_feature_vector(user_feature_vector, articles, vectorizer, k)

    print(reccs)

if __name__ == "__main__":
    main()



