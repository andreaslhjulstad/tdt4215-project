import os
import datetime
import pandas as pd
import numpy as np
from content import compute_recommendations_for_users as content_recommendations
from user_based import compute_recommendations_for_users as user_based_recommendations
from baseline import compute_recommendations_for_users as baseline_recommendations

# Constants - weights should sum to 1
CONTENT_WEIGHT = 0.5
USER_BASED_WEIGHT = 0.5
N_RECOMMENDATIONS = 10
N_CANDIDATES = 50


def get_article_titles(article_ids, articles_df):
    """Get titles for a list of article IDs from articles dataframe"""
    return [articles_df.loc[aid]['title'] if aid in articles_df.index else 'Unknown Title'
            for aid in article_ids]


def combine_and_score_articles(content_articles, user_based_articles):
    """Combine articles from content-based and user-based methods and calculate scores"""
    try:
        # Convert to lists if they're numpy arrays
        content_list = content_articles.tolist() if isinstance(
            content_articles, np.ndarray) else content_articles
        user_based_list = user_based_articles.tolist() if isinstance(
            user_based_articles, np.ndarray) else user_based_articles

        # Get all unique articles
        all_article_ids = list(set(content_list + user_based_list))

        # Create scores dataframe with explicit dtype
        scores_df = pd.DataFrame(index=all_article_ids, columns=[
            'content_score', 'user_based_score'
        ], dtype=float)

        # Fill NA values without downcasting
        scores_df = scores_df.fillna(0.0)

        # Add normalized scores (1 to 0) for each method
        if len(content_list) > 0:
            content_scores = np.linspace(1, 0, len(content_list))
            scores_df.loc[content_list, 'content_score'] = content_scores

        if len(user_based_list) > 0:
            user_based_scores = np.linspace(1, 0, len(user_based_list))
            scores_df.loc[user_based_list,
                          'user_based_score'] = user_based_scores

        # Calculate final scores using weights
        scores_df['final_score'] = (
            CONTENT_WEIGHT * scores_df['content_score'] +
            USER_BASED_WEIGHT * scores_df['user_based_score']
        )

        return scores_df
    except Exception as e:
        print(f"Error in combine_and_score_articles: {str(e)}")
        return pd.DataFrame()


def compute_recommendations_for_users(
    users: np.ndarray,
    curr_date: datetime,
    n_recommendations: int = 10,
    n_candidates: int = 50,
    content_use_lda: bool = True,
    content_n_topics: int = 57,
    similarity_threshold: float = 0.2,
    neighborhood_size: int = 10
):
    """
    Compute hybrid recommendations combining content-based and user-based approaches.

    Parameters:
        users (np.ndarray): Array of user IDs for whom to compute recommendations.
        curr_date (datetime): Current date for recommendations.
        n_recommendations (int): Number of recommendations to return for each user.
        n_candidates (int): Number of candidate articles to consider from each method.
        content_use_lda (bool): Whether to use LDA in content-based.
        content_n_topics (int): Number of topics for content-based LDA.
        similarity_threshold (float): Threshold for user similarity in user-based.
        neighborhood_size (int): Size of the neighborhood in user-based.

    Returns:
        pd.DataFrame: DataFrame of recommended articles for each user.
    """
    try:
        # Load data
        behaviors_df = pd.read_parquet("./data/train/behaviors.parquet")
        if not behaviors_df.index.name == 'user_id':
            behaviors_df = behaviors_df.set_index('user_id')

        # Check for cold start - if user has no behavior data, use baseline
        active_users = behaviors_df.index
        cold_start_users = [user for user in users if user not in active_users]
        warm_users = [user for user in users if user in active_users]

        # Handle cold start users with baseline
        if cold_start_users:
            if "DEBUG" in os.environ:
                print(
                    f"Using baseline for {len(cold_start_users)} cold start users")
            cold_start_recs = baseline_recommendations(
                np.array(cold_start_users),
                curr_date,
                n_days=30,
                n_recommendations=n_recommendations
            )

        # If we only have cold start users, return baseline recommendations
        if not warm_users:
            return cold_start_recs

        # Get recommendations from both methods for warm users
        content_recs = content_recommendations(
            np.array(warm_users),
            n_recommendations=n_candidates,
            curr_date=curr_date,
            use_lda=content_use_lda,
            n_topics=content_n_topics
        )

        user_based_recs = user_based_recommendations(
            np.array(warm_users),
            behaviors_df,
            n_recommendations=n_candidates,
            similarity_threshold=similarity_threshold,
            neighborhood_size=neighborhood_size
        )

        # Generate recommendations for warm users
        results = []
        for user in warm_users:
            if user in content_recs.index and user in user_based_recs.index:
                scores_df = combine_and_score_articles(
                    content_recs.loc[user]["recommended_article_ids"],
                    user_based_recs.loc[user]["recommended_article_ids"]
                )

                if not scores_df.empty:
                    top_articles = scores_df.nlargest(
                        n_recommendations, 'final_score').index.tolist()

                    results.append({
                        'user_id': user,
                        'recommended_article_ids': top_articles
                    })
                    if "DEBUG" in os.environ:
                        print(
                            f"Calculated hybrid recommendations for user: {user}")

        # Combine warm and cold start recommendations
        warm_recs = pd.DataFrame(results).set_index('user_id')
        if cold_start_users:
            return pd.concat([warm_recs, cold_start_recs])
        return warm_recs

    except Exception as e:
        print(f"Error in compute_recommendations_for_users: {str(e)}")
        return pd.DataFrame()


def main():
    curr_date = datetime.datetime(2023, 6, 8)
    test_users = np.array([10701, 5643])

    print(
        f"\nTesting with weights: Content={CONTENT_WEIGHT}, User-based={USER_BASED_WEIGHT}")
    recommendations = compute_recommendations_for_users(test_users, curr_date)
    print(recommendations)


if __name__ == "__main__":
    main()
