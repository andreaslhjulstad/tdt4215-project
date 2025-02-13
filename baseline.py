import datetime
import pandas as pd 

def baseline(data_path: str, curr_date: datetime):
    # Retrieve the articles from the past 3 days (including current date)
    df = pd.read_parquet(data_path,  
                        filters=[('total_pageviews', '>', 0), ("published_time", ">", curr_date - datetime.timedelta(days=2))])
        
    # Sort by pageviews (published time as tiebreaker) and 
    df = df.sort_values(by=["total_pageviews", "published_time"], inplace=False, ascending=False)
    df = df.head(20)

    # Create new column that calculates the average read time per view
    df["read_time_per_view"] = df["total_read_time"]/df["total_pageviews"]

    # Sort by new column to recommend the most "gripping" articles
    df = df.sort_values(by=["read_time_per_view"], inplace=False, ascending=False)

    # Return 10 'best' elements
    return df.head(10)

def main():
    # The date chosen in our scenario
    curr_date = datetime.date(2023, 6, 8)

    filepath = 'data/articles.parquet'

    print(baseline(filepath, curr_date))

if __name__ == "__main__":
    main()
