import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_parquet("./data/train/behaviors.parquet")

# The article_ids_clicked column has an array of article_ids as value
# Explode the array so each article_id in the list gets it's own row
exploded = df.explode("article_ids_clicked")

# TODO: find better solution
clicked_df = exploded[["user_id", "article_ids_clicked"]].copy()
clicked_df["clicked"] = 1
print(clicked_df)

# Creates a user-article matrix where the index is the user id and the columns are article IDs
# The value for each user-article pair is 1 if the user clicked on the article, and 0 otherwise
user_article_matrix = pd.pivot_table(
    clicked_df,
    values="clicked",
    index="user_id",
    columns="article_ids_clicked",
)

sparse_user_article_matrix = user_article_matrix.astype(pd.SparseDtype(float, np.nan))

# Calculate user similarity for pairs (i, j)
user_similarity_values = cosine_similarity(sparse_user_article_matrix.fillna(0))
# Fill diagonal with NaN to ensure that pairs (i, i) are not counted
np.fill_diagonal(user_similarity_values, np.nan)

user_similarity_df = pd.DataFrame(
    user_similarity_values,
    index=sparse_user_article_matrix.index,
    columns=sparse_user_article_matrix.index,
)
# print(user_similarity_df.head())

neighborhood_threshold = 0.3
neighborhood_size = 10

picked_userid = 10201
neighborhood = user_similarity_df[
    user_similarity_df[picked_userid] > neighborhood_threshold
][picked_userid].sort_values(ascending=False)[:neighborhood_size]

# Limit dataset to articles that similar users have clicked, and only the ones the selected user has not clicked

picked_userid_clicked = sparse_user_article_matrix[
    sparse_user_article_matrix.index == picked_userid
].dropna(axis=1, how="all")
similar_user_clicked = sparse_user_article_matrix[
    sparse_user_article_matrix.index.isin(neighborhood.index)
].dropna(axis=1, how="all")
similar_user_clicked.drop(picked_userid_clicked.columns, inplace=True, errors="ignore")

# Calculating weighted scores based on articles similar users have clicked

item_score = {}

for i in similar_user_clicked.columns:
    article = similar_user_clicked[i]
    total = 0
    count = 0
    for u in similar_user_clicked.index:
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

item_score = pd.DataFrame(
    item_score.values(), columns=["score"], index=item_score.keys()
)
item_score.sort_values(by="score", inplace=True, ascending=False)

print(item_score.head(10))
