import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def create_output_dir():
    """Create output directory for figures if it doesn't exist"""
    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)
    return output_dir


def analyze_articles(articles_df, output_dir):
    """Analyze and visualize article statistics"""
    print("\n=== Article Statistics ===")
    print(f"Total number of articles: {len(articles_df)}")

    articles_df['published_time'] = pd.to_datetime(
        articles_df['published_time'])
    date_range = articles_df['published_time'].agg(['min', 'max'])
    print(
        f"\nDate range: {date_range['min'].date()} to {date_range['max'].date()}")

    pageview_stats = articles_df['total_pageviews'].describe()
    print("\nPageview Statistics:")
    print(f"Mean pageviews: {pageview_stats['mean']:.2f}")
    print(f"Median pageviews: {pageview_stats['50%']:.2f}")
    print(f"Max pageviews: {pageview_stats['max']:.0f}")

    read_time_stats = articles_df['total_read_time'].describe()
    print("\nRead Time Statistics:")
    print(f"Mean read time: {read_time_stats['mean']:.2f}")
    print(f"Median read time: {read_time_stats['50%']:.2f}")

    articles_df['avg_read_time_per_view'] = articles_df['total_read_time'] / \
        articles_df['total_pageviews']
    avg_time_stats = articles_df['avg_read_time_per_view'].describe()
    print(
        f"\nAverage read time per view: {avg_time_stats['mean']:.2f}")

    plt.figure(figsize=(10, 8))
    correlation_matrix = articles_df[[
        'total_pageviews', 'total_read_time', 'avg_read_time_per_view']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation between Article Metrics')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png')
    plt.close()

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    daily_articles = articles_df.groupby(
        articles_df['published_time'].dt.date).size()
    daily_articles.plot(kind='line')
    plt.title('Number of Articles Published Daily')
    plt.xlabel('Date')
    plt.ylabel('Count')

    plt.subplot(2, 1, 2)
    daily_pageviews = articles_df.groupby(articles_df['published_time'].dt.date)[
        'total_pageviews'].sum()
    daily_pageviews.plot(kind='line')
    plt.title('Total Daily Pageviews')
    plt.xlabel('Date')
    plt.ylabel('Pageviews')

    plt.tight_layout()
    plt.savefig(output_dir / 'time_series_analysis.png')
    plt.close()


def analyze_user_behavior(behaviors_df, output_dir):
    """Analyze and visualize user behavior"""
    print("\n=== User Behavior Statistics ===")

    unique_users = behaviors_df['user_id'].nunique()
    print(f"Number of unique users: {unique_users}")

    clicks_per_user = behaviors_df['article_ids_clicked'].apply(len)
    click_stats = clicks_per_user.describe()
    print("\nClicks per user:")
    print(f"Mean: {click_stats['mean']:.2f}")
    print(f"Median: {click_stats['50%']:.2f}")
    print(f"Max: {click_stats['max']:.0f}")

    total_clicks = clicks_per_user.sum()
    print(f"\nTotal number of clicks: {total_clicks}")
    print(f"Average clicks per user: {total_clicks/unique_users:.2f}")

    plt.figure(figsize=(10, 6))
    behaviors_df['session_length'] = behaviors_df['article_ids_clicked'].apply(
        len)
    engagement_dist = behaviors_df['session_length'].value_counts(
    ).sort_index()
    engagement_dist.plot(kind='bar')
    plt.title('Distribution of Session Lengths')
    plt.xlabel('Number of Articles in Session')
    plt.ylabel('Number of Users')
    plt.tight_layout()
    plt.savefig(output_dir / 'user_engagement_distribution.png')
    plt.close()


def plot_engagement_metrics(articles_df, output_dir):
    """Plot detailed engagement metrics"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(articles_df['total_pageviews'],
                articles_df['total_read_time'], alpha=0.5)
    plt.title('Pageviews vs Total Read Time')
    plt.xlabel('Total Pageviews')
    plt.ylabel('Total Read Time (seconds)')

    plt.subplot(1, 2, 2)
    sns.histplot(articles_df['avg_read_time_per_view'].clip(0, 300), bins=50)
    plt.title('Distribution of Average Read Time per View')
    plt.xlabel('Seconds')

    plt.tight_layout()
    plt.savefig(output_dir / 'engagement_metrics.png')
    plt.close()


def analyze_data_quality(articles_df, behaviors_df):
    """Analyze data quality including duplicates and missing values"""
    print("\n=== Data Quality Analysis ===")

    print("\nArticle Dataset:")
    print(f"Total rows: {len(articles_df)}")
    print(f"Unique article IDs: {articles_df.index.nunique()}")

    print("\nMissing values(NaN):")
    print(articles_df.isnull().sum())

    print("\nMissing values(0):")
    print((articles_df.select_dtypes(include=['number']) == 0).sum())

    print("\nDuplicate Analysis for Articles:")

    duplicate_ids = articles_df.index.duplicated()
    if duplicate_ids.any():
        print(f"Found {duplicate_ids.sum()} duplicate article IDs")
        print("Duplicate article IDs:")
        print(articles_df.index[duplicate_ids].tolist())
    else:
        print("No duplicate article IDs found")

    duplicate_titles = articles_df.groupby('title').size()
    duplicate_titles = duplicate_titles[duplicate_titles > 1]
    if not duplicate_titles.empty:
        print(
            f"\nFound {len(duplicate_titles)} titles that appear multiple times:")
        print(duplicate_titles.head())

    print("\nBehavior Dataset Analysis:")
    print(f"Total rows: {len(behaviors_df)}")

    clicks_per_user = behaviors_df['article_ids_clicked'].apply(len)
    suspicious_users = clicks_per_user[clicks_per_user >
                                       clicks_per_user.mean() + 2*clicks_per_user.std()]
    if not suspicious_users.empty:
        print(
            f"\nFound {len(suspicious_users)} users with unusually high activity")
        print("Top 5 most active users:")
        print(suspicious_users.sort_values(ascending=False).head())

    unique_clicked_articles = set([
        article_id
        for clicks in behaviors_df['article_ids_clicked']
        for article_id in clicks
    ])
    print(f"\nUnique articles clicked: {len(unique_clicked_articles)}")
    print(
        f"Percentage of articles with clicks: {(len(unique_clicked_articles)/len(articles_df))*100:.2f}%")

    articles_not_found = [
        article_id
        for article_id in unique_clicked_articles
        if article_id not in articles_df.index
    ]
    print(
        f"\nArticles in behaviors but not in articles dataset: {len(articles_not_found)}")


def analyze_categories(articles_df, output_dir):
    """Analyze and visualize the distribution of articles across categories"""
    print("\n=== Category Analysis ===")

    category_counts = articles_df['category_str'].value_counts()
    print("\nCategory Distribution:")
    print(category_counts)

    plt.figure(figsize=(12, 6))
    category_counts.plot(kind='bar')
    plt.title('Distribution of Articles by Category')
    plt.xlabel('Category')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'category_distribution.png')
    plt.close()


def main():
    print("Loading datasets...")
    articles_df = pd.read_parquet("./data/articles.parquet")
    behaviors_df = pd.read_parquet("./data/train/behaviors.parquet")

    output_dir = create_output_dir()

    # Run analyses
    analyze_articles(articles_df, output_dir)
    analyze_user_behavior(behaviors_df, output_dir)
    plot_engagement_metrics(articles_df, output_dir)
    analyze_data_quality(articles_df, behaviors_df)
    analyze_categories(articles_df, output_dir)

    print(f"\nAnalysis complete. Figures saved in {output_dir}/")


if __name__ == "__main__":
    main()
