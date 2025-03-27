
import datetime
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from stop_words import get_stop_words
from sklearn.metrics.pairwise import linear_kernel
import scipy.sparse as sp 



def train_vectorizer(articles: pd.DataFrame, use_lda=False, topic_count=50):
    """
   Creates a TF-IDF vector for each article using a custom column (words). The TfidfVectorizer transforms each article’s bag of words into a sparse vector of scores.

    Parameters:
        articles (pd.DataFrame): Standard articles DF.

    Returns:
        sp.csr_matrix: A sparse score vector (here labeled matrix)
    """

    # Creates a 'words' column that joins together the articles' body and title, etc.
    articles['words'] = (
    (articles['title'].fillna('') + ' ') * 1 +  
    (articles['subtitle'].fillna('') + ' ') * 1 + 
    articles['body'].fillna('') + ' ' +
    (articles['category_str'].fillna('') + ' ') * 1 # categories are weighted tenfold here
)
    # Substitues some punctuation and sets everything to lower case
    articles['words'] = articles['words'].str.replace(r'[.,!?-]', '', regex=True).str.lower()

    # Get danish stop words
    stop_words = set(get_stop_words('danish'))

    # Custom list of 20 most common danish surnames and first names, gathered from Danmarks Statistik
    # Used to attempt to mitigate the "Nielsen"-problem - that is, last names like "Nielsen" being 
    # overly represented in reccomendations as they are relative rare in the corpus.
    danish_names = [
    # First names
    "anne", "mette", "kirsten", "hanne", "anna", "helle", "maria", "susanne", "lene",
    "marianne", "camilla", "lone", "louise", "charlotte", "pia", "tina", "gitte", "ida",
    "emma", "julie", "peter", "michael", "lars", "thomas", "jens", "henrik", "søren", "christian", "martin",
    "jan", "morten", "jesper", "anders", "mads", "niels", "rasmus", "mikkel", "per", "kim", "hans",
    # Surnames
    "nielsen", "jensen", "hansen", "andersen", "pedersen", "christensen", "larsen",
    "sørensen", "rasmussen", "jørgensen", "petersen", "madsen", "kristensen", "olsen",
    "thomsen", "christiansen", "poulsen", "johansen", "møller", "mortensen",

    # Some additional semantic phrases used in the body text by Extra Bladet
    "split", "element"
    ]


    # The imported stop words and the custom names are combined into one group of stop words
    custom_stop_words = list(stop_words.union(danish_names))

    if not use_lda:
        # Create TfidfVectorizer using these stopwrods
        vectorizer = TfidfVectorizer(stop_words=custom_stop_words, max_features=5000)
        matrix = vectorizer.fit_transform(articles['words'])
        return matrix
    else:
        # Convert documents to word count vectors
        vectorizer = CountVectorizer(stop_words=custom_stop_words, max_features=1000)
        matrix = vectorizer.fit_transform(articles['words'])

        # Train lda model
        lda = LatentDirichletAllocation(n_components=topic_count)
        matrix = lda.fit_transform(matrix)
        return matrix


def recommend_for_user(article_ids: list, articles: pd.DataFrame, tfidf_matrix: sp.csr_matrix, k: int):
    """
    Recommend top_k articles for a given user based on cosine similarity (linear kernel).

    Parameters:
        article_ids (list): List of article ids read by the user.
        articles (pd.DataFrame): Standard articles DF.
        tfidf_matrix (sp.csr_matrix): TF-IDF vectorized representation of the articles.
        k (int): # of articles to recommend.

    Returns:
        List[int]: List of recommended article ids.
    """

    # Create tuples of (id, index) pairs
    # Then, create a dict for looking up indexes given an article_id
    a = zip(articles['article_id'], articles.index)
    id_to_index = dict(a)

    # Create list of indexes that the valid article ids map to
    indexes = [id_to_index[id] for id in article_ids if id in id_to_index]
    if len(indexes) == 0:
        raise ValueError("No articles found for user.")
    

    # Create user vector by averaging the tfidf scores for all the articles they have read
    user_vector = tfidf_matrix[indexes].mean(axis=0)
    user_vector = sp.csr_matrix(user_vector)  # Convert from np.matrix to sparse (for the cosine sim func)

    # Calculate cosine similarity (here: linear kernel) between user vector and the "universal" matrix
    # Then, exclude already read articles
    scores = linear_kernel(user_vector, tfidf_matrix).flatten()
    filtered_scores = [i for i in range(len(articles)) if i not in indexes]
    scores = scores[filtered_scores]

    # Get the indexes of the k best scores
    # Then sort them by score, decending
    top_k = np.argpartition(scores, -k)[-k:]
    top_k = top_k[np.argsort(scores[top_k])[::-1]]


    # Map the top_k indexes (filtered) back to their original indexes (cosine scores)
    # This is necessary because 'scores' is a filtered subset of the original similarity array
    top_k = [filtered_scores[i] for i in top_k]

    # Return just the article ids
    return articles.iloc[top_k]['article_id'].tolist()


def main():
    pd.set_option('display.max_rows', None)


    # Example usage given a concrete uid
    # See more complex usage in evaluate.py
    user_id = 13538
    history = pd.read_parquet('./data/train/history.parquet').set_index("user_id")
    user_row = history.loc[user_id]
    article_ids = user_row["article_id_fixed"]
    curr_date = datetime.date(2023, 6, 8)
    articles = pd.read_parquet('./data/articles.parquet')
    articles_filtered = pd.read_parquet('./data/articles.parquet',  
                            filters=[("published_time", ">", curr_date - datetime.timedelta(days=15)),
                                        ])
    read_articles = articles[articles['article_id'].isin(article_ids)]

    # Combines recent articles (filtered) with articles read by the user, as they otherwise would likely be filtered out
    combined = pd.concat([articles_filtered, read_articles]).drop_duplicates(subset='article_id').reset_index(drop=True)

    # Train matrix once based on this combined set of articles
    tfidf_matrix = train_vectorizer(combined)

    recommended = recommend_for_user(article_ids, combined, tfidf_matrix, 10)
    print(recommended)

    for article_id in recommended:
        art = articles[articles['article_id'] == article_id]


        title = art.iloc[0]['title']
        subtitle = art.iloc[0]['subtitle']
        body = art.iloc[0]['body']
        print(body)

        #print(art)



if __name__ == "__main__":
    main()


# Results from running on all users with only categories x10, no view filtering, days=15:
# Total recommendations: 233160
# TPs: 214
# Precision: 0.0009



# Equal weights, topics_# = 40, max_features = 1000
#Precision: 0.0040
#nDCG: 0.0151

# Equal weights, topics_# = 50, max_features = 1000
#Precision: 0.0148, 0.0025
#nDCG: 0.0796, 0.0099
    
# Equal weights, topics_# = 55, max_features = 1000
#Precision: 0.0010, 0.0051
#nDCG: 0.0048, 0.0220
    
# Equal weights, topics_# = 60, max_features = 1000
#Precision: 0.0085
#nDCG: 0.0322
    


# På alle brukere:
# Equal weights, topics_# = 50, max_features = 1000
#Precision: 0.0040
#nDCG: 0.0132
    



# TODO:
    # prøve recency filter på artikler brukeren allerede har lest også - > vil trolig påvirke gjennomsnittet

    # Ta en titt på disse brukerne:
    # Calculated nDCG for user: 2424496 at 1.0000
    # Calculated nDCG for user: 1998523 at 0.2891
    # Calculated nDCG for user: 2227899 at 0.0000