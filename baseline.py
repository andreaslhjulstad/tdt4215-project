import datetime
import pandas as pd


def baseline(data_path: str, curr_date: datetime):
    # Load articles with article_id as index
    articles_df = pd.read_parquet(data_path)
    articles_df = articles_df.reset_index().set_index('article_id')

    # Debug print
    print("Sample of original article IDs:", articles_df.index[:5].tolist())
    print("ID type:", type(articles_df.index[0]))

    # Retrieve the articles from the past 3 days (including current date)
    df = pd.read_parquet(data_path,
                         filters=[('total_pageviews', '>', 0), ("published_time", ">", curr_date - datetime.timedelta(days=2))])
    df = df.reset_index().set_index('article_id')

    # Sort by pageviews (published time as tiebreaker) and
    df = df.sort_values(
        by=["total_pageviews", "published_time"], inplace=False, ascending=False)
    df = df.head(20)

    # Create new column that calculates the average read time per view
    df["read_time_per_view"] = df["total_read_time"]/df["total_pageviews"]

    # Normalize scores to be between 0 and 1
    max_read_time = df["read_time_per_view"].max()
    df["score"] = df["read_time_per_view"] / max_read_time

    # Sort by score
    df = df.sort_values(by=["score"], inplace=False, ascending=False)

    # Return dataframe with article IDs as index and scores
    return df[["title", "score"]]


def main():
    # The date chosen in our scenario
    curr_date = datetime.date(2023, 6, 8)

    filepath = 'data/articles.parquet'

    print(baseline(filepath, curr_date))


if __name__ == "__main__":
    main()
