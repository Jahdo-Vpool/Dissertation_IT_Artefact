"""
Step 4: Merge Prices and Sentiment (Fixed for prices_processed)
Handles 'return (%)' column name and validates date ranges

Run: python 4_merge_data.py
"""

import pandas as pd

# ============================================
# MAIN CODE
# ============================================

def merge_data(ticker):
    """Merge prices and sentiment for a ticker."""

    print(f"\n{'='*60}")
    print(f"{ticker}")
    print(f"{'='*60}")

    # Read price data (handle different column names)
    prices = pd.read_csv(f'data/{ticker}_prices_processed.csv')

    # Rename 'return (%)' to 'return' for easier handling
    if 'return (%)' in prices.columns:
        prices = prices.rename(columns={'return (%)': 'return'})
        print(f"  Renamed 'return (%)' to 'return'")

    print(f"  ✓ Prices: {len(prices)} trading days")
    print(f"    Date range: {prices['date'].min()} to {prices['date'].max()}")
    print(f"    Columns: {list(prices.columns)}")

    # Read sentiment data
    sentiment = pd.read_csv(f'results/{ticker}_with_sentiment.csv')
    print(f"  ✓ Sentiment: {len(sentiment)} articles")
    print(f"    Date range: {sentiment['date'].min()} to {sentiment['date'].max()}")

    # ============================================
    # CHECK FOR DATE OVERLAP
    # ============================================
    price_start = prices['date'].min()
    price_end = prices['date'].max()
    news_start = sentiment['date'].min()
    news_end = sentiment['date'].max()

    print(f"\n  Date Range Check:")
    print(f"    Prices: {price_start} to {price_end}")
    print(f"    News:   {news_start} to {news_end}")

    # Check if dates overlap
    if price_end < news_start or price_start > news_end:
        print(f"\n  CRITICAL ERROR: NO DATE OVERLAP!")
        print(f"\n  Your price data is from a different time period than your news!")
        print(f"  Price data ends:  {price_end}")
        print(f"  News data starts: {news_start}")
        print(f"\n  SOLUTION: Download price data from {news_start} onwards")
        return None

    # Show sentiment breakdown
    print(f"\n  Sentiment breakdown:")
    label_counts = sentiment['sentiment_label'].value_counts()
    for label in ['positive', 'negative', 'neutral']:
        count = label_counts.get(label, 0)
        pct = count / len(sentiment) * 100
        print(f"    {label.capitalize()}: {count} ({pct:.1f}%)")

    # Group sentiment by date
    daily_sentiment = sentiment.groupby('date').agg({
        'sentiment_score': 'mean',
        'ticker': 'count'
    }).reset_index()

    daily_sentiment = daily_sentiment.rename(columns={'ticker': 'article_count'})
    print(f"  ✓ Daily sentiment: {len(daily_sentiment)} unique dates")

    # ============================================
    # Filter prices to match news dates
    # ============================================
    news_start = daily_sentiment['date'].min()
    news_end = daily_sentiment['date'].max()

    prices_filtered = prices[
        (prices['date'] >= news_start) &
        (prices['date'] <= news_end)
    ].copy()

    print(f"\n  Filtering prices to news coverage period:")
    print(f"    Original prices: {len(prices)} days")
    print(f"    Filtered prices: {len(prices_filtered)} days ({news_start} to {news_end})")
    print(f"    Removed: {len(prices) - len(prices_filtered)} days outside news range")

    if len(prices_filtered) == 0:
        print(f"\n  ERROR: No matching dates after filtering!")
        print(f"\n  This means your price and news data don't overlap.")
        print(f"  You need price data from {news_start} to {news_end}")
        return None

    # Merge
    merged = prices_filtered.merge(
        daily_sentiment,
        on='date',
        how='left'
    )

    # Forward-fill sentiment
    merged['sentiment_score'] = merged['sentiment_score'].ffill()
    merged['article_count'] = merged['article_count'].fillna(0)
    merged['sentiment_score'] = merged['sentiment_score'].fillna(0)

    # Add sentiment label
    def get_label(score):
        if score >= 0.05:
            return 'positive'
        elif score <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    merged['sentiment_label'] = merged['sentiment_score'].apply(get_label)

    # Statistics
    days_with_articles = (merged['article_count'] > 0).sum()

    print(f"\n  ✓ Merged data: {len(merged)} rows")
    print(f"    Dates with articles: {days_with_articles}")
    print(f"    Coverage: {days_with_articles/len(merged)*100:.1f}%")
    print(f"    Average sentiment: {merged['sentiment_score'].mean():.3f}")
    print(f"    Average return: {merged['return'].mean():.3f}%")

    return merged


def main():
    """Main function."""

    print("\n" + "="*60)
    print("MERGE PRICES AND SENTIMENT")
    print("="*60)

    # Merge TAN
    tan_merged = merge_data('TAN')

    if tan_merged is not None:
        output_file = 'results/TAN_merge_prices_news.csv'
        tan_merged.to_csv(output_file, index=False)
        print(f"\n  Saved: {output_file}")
    else:
        print(f"\n  TAN merge failed - check dates!")

    # Merge SPY
    spy_merged = merge_data('SPY')

    if spy_merged is not None:
        output_file = 'results/SPY_merge_prices_news.csv'
        spy_merged.to_csv(output_file, index=False)
        print(f"\n  Saved: {output_file}")
    else:
        print(f"\n SPY merge failed - check dates!")

    # Summary
    print("\n" + "="*60)
    if tan_merged is not None or spy_merged is not None:
        print("MERGE COMPLETE!")
        print("="*60)

        if tan_merged is not None:
            print(f"\nTAN: {len(tan_merged)} trading days")
            print(f"  Date range: {tan_merged['date'].min()} to {tan_merged['date'].max()}")

        if spy_merged is not None:
            print(f"\nSPY: {len(spy_merged)} trading days")
            print(f"  Date range: {spy_merged['date'].min()} to {spy_merged['date'].max()}")

        print("\nNext step: Run 5_generate_signals.py")
    else:
        print("MERGE FAILED - DATE MISMATCH!")
        print("="*60)
        print("\nYour price data is from 2020, but news is from 2024-2025")
        print("Download new price data that matches your news dates!")


if __name__ == "__main__":
    main()
