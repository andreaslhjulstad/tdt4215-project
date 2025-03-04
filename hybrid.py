import datetime
import pandas as pd
import numpy as np
from baseline import baseline
from collaborative import (
    compute_recommendations_for_users,
    create_user_article_matrix,
    calculate_user_similarity,
    create_neighborhood,
    get_recommendations_for_user
)

# Global constants
COLLABORATIVE_WEIGHT = 0.5
BASELINE_WEIGHT = 1 - COLLABORATIVE_WEIGHT


def get_article_titles(article_ids, articles_df):
    """Get titles for a list of article IDs"""
    titles = []
    for aid in article_ids:
        title = articles_df.loc[aid]['title'] if aid in articles_df.index else 'Unknown Title'
        titles.append(title)
    return titles


def hybrid_recommendations(users, curr_date):
    """
    Combines recommendations from baseline and collaborative filtering with weighted scores
    Uses global COLLABORATIVE_WEIGHT and BASELINE_WEIGHT
    """

    # Load articles data
    articles_df = pd.read_parquet("./data/articles.parquet")
    articles_df = articles_df.reset_index().set_index('article_id')

    # Get collaborative recommendations with scores
    behaviors_df = pd.read_parquet("./data/train/behaviors.parquet")
    collab_recommendations = compute_recommendations_for_users(
        users,
        behaviors_df,
        n_recommendations=50,  # Get more recommendations to combine
        similarity_threshold=0.2,
        neighborhood_size=10
    )

    # Get baseline recommendations with scores
    baseline_articles = baseline("./data/articles.parquet", curr_date)

    # Combine recommendations for each user
    final_recommendations = {}
    for user in users:
        # Get all articles from both methods
        user_collab_articles = collab_recommendations.loc[user]["recommended_article_ids"]
        baseline_ids = baseline_articles.index.tolist()
        all_article_ids = list(
            set(user_collab_articles + baseline_ids))  # Remove duplicates

        # Create merged dataframe with all articles
        merged_scores = pd.DataFrame(index=all_article_ids)

        # Add collaborative scores (normalize them to 0-1 range)
        merged_scores['collaborative_score'] = 0.0  # Default score
        collab_scores = np.linspace(1, 0, len(user_collab_articles))
        merged_scores.loc[user_collab_articles,
                          'collaborative_score'] = collab_scores

        # Add baseline scores
        merged_scores['baseline_score'] = 0.0  # Default score
        merged_scores.loc[baseline_ids,
                          'baseline_score'] = baseline_articles['score']

        # Calculate weighted scores
        merged_scores['final_score'] = (
            COLLABORATIVE_WEIGHT * merged_scores['collaborative_score'] +
            BASELINE_WEIGHT * merged_scores['baseline_score']
        )

        # Debug prints
        print("\nSample scores for collaborative articles:")
        print(merged_scores.loc[user_collab_articles[:5], [
              'collaborative_score', 'baseline_score', 'final_score']])
        print("\nSample scores for baseline articles:")
        print(merged_scores.loc[baseline_ids[:5], [
              'collaborative_score', 'baseline_score', 'final_score']])

        # Sort and get top 10
        top_articles = merged_scores.sort_values(
            'final_score', ascending=False).head(10)

        # Get titles
        titles = get_article_titles(top_articles.index, articles_df)

        final_recommendations[user] = {
            'article_ids': top_articles.index.tolist(),
            'titles': titles,
            'scores': top_articles['final_score'].tolist()
        }

    # Create result DataFrame
    result_data = {
        'user_id': [],
        'recommended_article_ids': [],
        'article_titles': [],
        'scores': []
    }

    for user, data in final_recommendations.items():
        result_data['user_id'].append(user)
        result_data['recommended_article_ids'].append(data['article_ids'])
        result_data['article_titles'].append(data['titles'])
        result_data['scores'].append(data['scores'])

    return pd.DataFrame(result_data).set_index('user_id')


def main():
    # Test the hybrid recommender
    curr_date = datetime.date(2023, 6, 8)
    test_users = [10701]  # Example users

    print(f"\nTesting with collaborative weight: {COLLABORATIVE_WEIGHT}")
    recommendations = hybrid_recommendations(
        users=test_users,
        curr_date=curr_date
    )

    print("\nHybrid Recommendations:")
    for user_id, row in recommendations.iterrows():
        print(f"\nRecommendations for user {user_id}:")
        for article_id, title, score in zip(
            row['recommended_article_ids'],
            row['article_titles'],
            row['scores']
        ):
            print(f"- Article {article_id} (score: {score:.3f}): {title}")


if __name__ == "__main__":
    main()
