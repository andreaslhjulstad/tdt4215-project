import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from stop_words import get_stop_words
from sklearn.metrics.pairwise import linear_kernel
import scipy.sparse as sp 

def content(data_path: str, article_id: int):
    # Given an article id, find its categories

    
    # Retrieve given article's categories and topics
    df = pd.read_parquet(data_path,  
                        filters=[('article_id', '==', article_id)])
    
    category = df['category'].loc[df.index[0]]
    sub_category = df['subcategory'].loc[df.index[0]]
    topics: list = df['topics'].loc[df.index[0]]

    # Attempt to filter based on subcategories

    df = pd.read_parquet(data_path,  
                        filters=[ ('category', '==', category)])

    if (len(sub_category) != 0):
        # Check if any subcategory 

        #sub_category_df = pd.DataFrame({'subcategory': sub_category})
        print(sub_category)

    

        #n = df['subcategory'].isin(sub_category)
        #print(n)
        
        print(df['subcategory'][0])


        #for  in df['subcategory']:
        #    for cat in sub_category:
        #        if 

        n = df['subcategory'].apply(lambda cats: any(cat in sub_category for cat in cats))

        #n = [cat in sub_category for cat in df['subcategory']]
        
        return df[n]


    # Attempt to filter based on topics

    
    #df = pd.read_parquet(data_path,  
                        #filters=[ ('category', '==', category)])

    return df.dropna()

filepath = 'data/articles.parquet'
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
#print(content(filepath, 9522337))


""" def calculate_user_feature_vector(user_id: int,  history: pd.DataFrame):
    #print(history.head())
    user_row = history.loc[user_id]
    # print(user_row.head())
    article_ids = user_row["article_id_fixed"] # Gets the list of article ids the user has clicked on
    article_vectors = []
    for article_id in article_ids:
        vector = get_bag_of_words_vector(article_id, 'data/articles.parquet')
        article_vectors.append(vector)
    
    all_vectors = sp.lil_matrix((len(article_ids), 1000), dtype=np.float64)

    for i, vec in enumerate(article_vectors):
        vec_array = vec.toarray().flatten()

        actual_length = len(vec_array)

        all_vectors[i, :actual_length] = vec_array

    avg_dense = all_vectors.mean(axis=0)
    avg_sparse = sp.csr_matrix(avg_dense)
    
    return avg_sparse


def get_recommendations_from_user_feature_vector(user_feature_vector: sp.csr_matrix,  articles: pd.DataFrame, k: int):
    # Create new column which combines all relevant info
    # TODO: format topics to str and include in bag of words
    # TODO: fjerne alle artikler som brukeren allerede har lest
    articles['words'] = articles['title'].fillna('') + ' ' + articles['subtitle'].fillna('') + ' ' + articles['body'].fillna('') + ' ' + articles['category_str'].fillna('')
   
    # Create TF-IDF vectors from the 'words' column using danish stop words (removes filler words)
    vectorizer = TfidfVectorizer(stop_words=get_stop_words('danish'), max_features=1000)
    article_vectors = vectorizer.fit_transform(articles['words'])
   
    # Calculate similarity scores for given article
    scores = cosine_similarity(user_feature_vector, article_vectors).flatten()
    
    # Sort scores decending
    scores = scores.argsort()[::-1]
   
    top_scores = articles.loc[scores[:k]]['article_id'].to_list()

    # Return top k scored articles
    return top_scores


def compute_recommendations_for_users(
    users: np.ndarray, # Assume list of user ids (calling method must handle this)
    n_recommendations: int,
    history: pd.DataFrame,
    articles: pd.DataFrame,
):
    recommendations_dict = {}
    for user in users:

        read_articles = history.filter(user_id = user)['article_id_fixed']
        articles.drop(read_articles)


        user_feature_vector = calculate_user_feature_vector(user, history)
        reccs = get_recommendations_from_user_feature_vector(user_feature_vector, articles, n_recommendations)
        print(f"Calculated recommendations for user: {user}")
        

        recommendations_dict[user] = reccs
    
    recommendations_df = pd.DataFrame(
        {
            "user_id": list(recommendations_dict.keys()),
            "recommended_article_ids": list(recommendations_dict.values()),
        }
    ).set_index("user_id")
    print(recommendations_df)
    return recommendations_df
 """
