import datetime
import pandas as pd
import numpy as np
from baseline import baseline
from collaborative import compute_recommendations_for_users

# Constants
COLLABORATIVE_WEIGHT = 0.5
BASELINE_WEIGHT = 1 - COLLABORATIVE_WEIGHT
N_RECOMMENDATIONS = 10
N_CANDIDATES = 50


def get_article_titles(article_ids, articles_df):
    """Get titles for a list of article IDs from articles dataframe"""
    return [articles_df.loc[aid]['title'] if aid in articles_df.index else 'Unknown Title'
            for aid in article_ids]


def combine_and_score_articles(user_collab_articles, baseline_articles):
    """Combine articles from both methods and calculate their scores"""
    # Get all unique articles
    all_article_ids = list(
        set(user_collab_articles + baseline_articles.index.tolist()))

    # Create scores dataframe
    scores_df = pd.DataFrame(index=all_article_ids, columns=[
                             'collaborative_score', 'baseline_score'])
    scores_df.fillna(0.0, inplace=True)

    # Add collaborative scores (normalize them to 0-1 range)
    collab_scores = np.linspace(1, 0, len(user_collab_articles))
    scores_df.loc[user_collab_articles, 'collaborative_score'] = collab_scores

    # Add baseline scores
    scores_df.loc[baseline_articles.index,
                  'baseline_score'] = baseline_articles['score']

    # Calculate final scores
    scores_df['final_score'] = (COLLABORATIVE_WEIGHT * scores_df['collaborative_score'] +
                                BASELINE_WEIGHT * scores_df['baseline_score'])

    return scores_df


def get_top_articles(scores_df, articles_df, n=N_RECOMMENDATIONS):
    """Get top n articles with their titles and scores"""
    top_df = scores_df.sort_values('final_score', ascending=False).head(n)
    return {
        'article_ids': top_df.index.tolist(),
        'titles': get_article_titles(top_df.index, articles_df),
        'scores': top_df['final_score'].tolist()
    }


def hybrid_recommendations(users, curr_date):
    """
    Generate hybrid recommendations combining collaborative filtering and baseline approaches.

    Args:
        users: List of user IDs
        curr_date: Current date for baseline recommendations

    Returns:
        DataFrame with recommendations for each user
    """
    # Load data
    articles_df = pd.read_parquet(
        "./data/articles.parquet").set_index('article_id')
    behaviors_df = pd.read_parquet("./data/train/behaviors.parquet")

    # Get recommendations from both methods
    collab_recommendations = compute_recommendations_for_users(
        users, behaviors_df, N_CANDIDATES, 0.2, 10)
    baseline_articles = baseline("./data/articles.parquet", curr_date)

    # Generate recommendations for each user
    results = []
    for user in users:
        user_collab_articles = collab_recommendations.loc[user]["recommended_article_ids"]
        scores_df = combine_and_score_articles(
            user_collab_articles, baseline_articles)
        recommendations = get_top_articles(scores_df, articles_df)

        results.append({
            'user_id': user,
            'recommended_article_ids': recommendations['article_ids'],
            'article_titles': recommendations['titles'],
            'scores': recommendations['scores']
        })

    return pd.DataFrame(results).set_index('user_id')


def main():
    curr_date = datetime.date(2023, 6, 8)
    test_users = [10701]

    print(f"\nTesting with collaborative weight: {COLLABORATIVE_WEIGHT}")
    recommendations = hybrid_recommendations(test_users, curr_date)

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
