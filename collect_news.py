"""
Collect NEWS from GDELT - Maximum Coverage
Free, unlimited, historical data!

What this does:
- Collects 6-12 months of news articles
- No API key needed
- Unlimited requests
- Matches your price data period

Run: python collect_news_gdelt_full.py
"""

from gdeltdoc import GdeltDoc, Filters
import pandas as pd
from datetime import datetime, timedelta
import time

# ============================================
# CONFIGURATION - SET YOUR DATE RANGE
# ============================================

# Match these to your EXACT price data range!
START_DATE = '2020-12-01'  # ← Change to your price start date
END_DATE = '2025-12-12'  # ← Change to your price end date

# Calculate period
start_dt = datetime.strptime(START_DATE, '%Y-%m-%d')
end_dt = datetime.strptime(END_DATE, '%Y-%m-%d')
days = (end_dt - start_dt).days
months = days / 30

print(f"\nCollection Period:")
print(f"  Start: {START_DATE}")
print(f"  End: {END_DATE}")
print(f"  Duration: {days} days (~{months:.1f} months)")


# ============================================
# MAIN CODE
# ============================================

def collect_gdelt_news(ticker, keywords, max_articles=500):
    """
    Collect news from GDELT for a ticker.

    ticker: Stock symbol
    keywords: List of search terms
    max_articles: Maximum articles to collect (GDELT can return thousands!)
    """

    print(f"\n{'=' * 60}")
    print(f"Collecting GDELT news for {ticker}")
    print(f"{'=' * 60}")

    # Initialize GDELT
    gd = GdeltDoc()

    all_articles = []

    for keyword in keywords:
        print(f"\n  Searching: '{keyword}'...")

        try:
            # Create filter
            f = Filters(
                keyword=keyword,
                start_date=START_DATE,
                end_date=END_DATE,
                num_records=250  # Max per search term
            )

            # Get articles
            articles = gd.article_search(f)

            if articles is not None and len(articles) > 0:
                print(f"    Found {len(articles)} articles")

                # Process each article
                for idx, row in articles.iterrows():
                    # GDELT gives us the date, title, domain, URL
                    all_articles.append({
                        'ticker': ticker,
                        'date': row['seendate'][:10] if 'seendate' in row else START_DATE,
                        'title': row['title'] if 'title' in row else '',
                        'content': row['title'] if 'title' in row else '',  # Use title as content
                        'source': row['domain'] if 'domain' in row else 'Unknown',
                        'url': row['url'] if 'url' in row else ''
                    })

                # Small delay to be polite
                time.sleep(1)
            else:
                print(f"    No articles found")

        except Exception as e:
            print(f"    Error: {e}")
            continue

    # Convert to DataFrame
    df = pd.DataFrame(all_articles)

    if df.empty:
        print(f"\n  No articles found for {ticker}")
        return None

    # Remove duplicates (same URL)
    original_count = len(df)
    df = df.drop_duplicates(subset=['url'], keep='first')
    print(f"\n  Removed {original_count - len(df)} duplicates")

    # Remove articles with no title
    df = df[df['title'].str.len() > 10]

    # Limit to max_articles
    if len(df) > max_articles:
        print(f"  Limiting to {max_articles} most recent articles")
        df = df.sort_values('date', ascending=False).head(max_articles)

    # Sort by date
    df = df.sort_values('date', ascending=False).reset_index(drop=True)

    # Show statistics
    print(f"\n  FINAL COLLECTION:")
    print(f"    Total articles: {len(df)}")
    print(f"    Unique dates: {df['date'].nunique()}")
    print(f"    Date range: {df['date'].min()} to {df['date'].max()}")

    # Date coverage
    articles_per_day = df.groupby('date').size()
    print(f"    Days with articles: {len(articles_per_day)}/{days}")
    print(f"    Average per day: {articles_per_day.mean():.1f}")
    print(f"    Coverage: {len(articles_per_day) / days * 100:.1f}%")

    # Show top sources
    top_sources = df['source'].value_counts().head(5)
    print(f"\n    Top sources:")
    for source, count in top_sources.items():
        print(f"      {source}: {count} articles")

    return df


def main():
    """Main function."""

    print("\n" + "=" * 60)
    print("GDELT NEWS COLLECTION - MAXIMUM COVERAGE")
    print("=" * 60)
    print("\nFree, unlimited, historical data!")
    print("Perfect for academic research")

    # ============================================
    # COLLECT TAN NEWS
    # ============================================

    tan_articles = collect_gdelt_news(
        ticker='TAN',
        keywords=[
            'solar energy',
            'solar power',
            'renewable energy',
            'clean energy',
            'solar stocks',
            'solar industry',
            'photovoltaic',
            'solar panel',
            'green energy',
            'TAN ETF'
        ],
        max_articles=1000  # Collect up to 1000 articles
    )

    # ============================================
    # COLLECT SPY NEWS
    # ============================================

    spy_articles = collect_gdelt_news(
        ticker='SPY',
        keywords=[
            'stock market',
            'S&P 500',
            'SPY ETF',
            'stocks',
            'Wall Street',
            'Dow Jones',
            'market rally',
            'stock prices',
            'equity market',
            'trading'
        ],
        max_articles=1000  # Collect up to 1000 articles
    )

    # ============================================
    # SAVE TO CSV
    # ============================================

    if tan_articles is not None:
        tan_articles.to_csv('data/TAN_news.csv', index=False)
        print(f"\n✅ Saved: data/TAN_news.csv ({len(tan_articles)} articles)")

    if spy_articles is not None:
        spy_articles.to_csv('data/SPY_news.csv', index=False)
        print(f"✅ Saved: data/SPY_news.csv ({len(spy_articles)} articles)")

    # ============================================
    # SUMMARY
    # ============================================

    print("\n" + "=" * 60)
    print("✅ COLLECTION COMPLETE!")
    print("=" * 60)

    if tan_articles is not None and spy_articles is not None:
        print(f"\nCollected:")
        print(f"  TAN: {len(tan_articles)} articles over {tan_articles['date'].nunique()} days")
        print(f"  SPY: {len(spy_articles)} articles over {spy_articles['date'].nunique()} days")
        print(f"\nPeriod: {START_DATE} to {END_DATE} (~{months:.1f} months)")
        print(f"\nThis gives you {len(tan_articles) + len(spy_articles)} total articles for analysis!")

    print("\nNext step: Run 2_analyze_sentiment.py")


if __name__ == "__main__":
    import os

    os.makedirs('data', exist_ok=True)

    # Run
    main()