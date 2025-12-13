"""
Step 1: Collect News Articles
Simple script to get news from NewsAPI

What this does:
- Searches for TAN and SPY news
- Saves to CSV files
"""

from newsapi import NewsApiClient
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================
# CONFIGURATION
# ============================================

# Get API key from environment variable
API_KEY = os.getenv('NEWS_API_KEY')

if not API_KEY:
    print("\nERROR: NEWS_API_KEY not set!")
    exit(1)

# How many days back to search (from .env or default)
DAYS_BACK = int(os.getenv('DAYS_BACK', '30'))

# How many articles to collect per ETF (from .env or default)
ARTICLES_PER_ETF = int(os.getenv('ARTICLES_PER_ETF', '50'))


# ============================================
# MAIN CODE
# ============================================

def collect_news(ticker, search_terms):
    """
    Collect news for a ticker.

    ticker: Stock symbol (e.g., 'TAN')
    search_terms: List of things to search for
    """

    print(f"\n{'=' * 60}")
    print(f"Collecting news for {ticker}")
    print(f"{'=' * 60}")

    # Set up NewsAPI
    newsapi = NewsApiClient(api_key=API_KEY)

    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=DAYS_BACK)

    print(f"  Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Store all articles
    all_articles = []

    # Search for each term
    for term in search_terms:
        print(f"  Searching: '{term}'...")

        try:
            # Call NewsAPI
            response = newsapi.get_everything(
                q=term,
                language='en',
                sort_by='publishedAt',
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                page_size=20
            )

            # Process articles
            if response['status'] == 'ok':
                for article in response['articles']:
                    # Skip if no content
                    if not article.get('description'):
                        continue

                    # Get text content
                    content = article.get('description', '')
                    if article.get('content'):
                        content = content + ' ' + article['content']

                    # Clean up [+123 chars] that NewsAPI adds
                    if '[+' in content:
                        content = content.split('[+')[0]

                    # Store article
                    all_articles.append({
                        'ticker': ticker,
                        'date': article['publishedAt'][:10],  # Just the date
                        'title': article['title'],
                        'content': content,
                        'source': article['source']['name'],
                        'url': article['url']
                    })

                print(f"    Found {len(response['articles'])} articles")

        except Exception as e:
            print(f"    Error: {e}")
            continue

    # Convert to DataFrame
    df = pd.DataFrame(all_articles)

    if df.empty:
        print(f"No articles found for {ticker}")
        return None

    # Remove duplicates (same URL)
    df = df.drop_duplicates(subset=['url'], keep='first')

    # Keep only articles with enough content
    df = df[df['content'].str.len() > 50]

    # Sort by date
    df = df.sort_values('date', ascending=False)

    # Limit to target number
    df = df.head(ARTICLES_PER_ETF)

    print(f"Collected {len(df)} articles")

    return df

def main():
    """Main function - run everything."""

    print("\n" + "=" * 60)
    print("NEWS COLLECTION")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Days back: {DAYS_BACK}")
    print(f"  Articles per ETF: {ARTICLES_PER_ETF}")

    # Collect TAN news
    tan_articles = collect_news(
        ticker='TAN',
        search_terms=[
            'solar energy stocks',
            'renewable energy ETF',
            'clean energy investment',
            'solar power industry'
        ]
    )

    # Collect SPY news
    spy_articles = collect_news(
        ticker='SPY',
        search_terms=[
            'S&P 500 performance',
            'stock market outlook',
            'market rally',
            'stock market analysis'
        ]
    )

    # Save to CSV
    if tan_articles is not None:
        tan_articles.to_csv('data/TAN_news.csv', index=False)
        print(f"\nSaved: data/TAN_news.csv ({len(tan_articles)} articles)")

    if spy_articles is not None:
        spy_articles.to_csv('data/SPY_news.csv', index=False)
        print(f"Saved: data/SPY_news.csv ({len(spy_articles)} articles)")

    print("\n" + "=" * 60)
    print("✓ COLLECTION COMPLETE!")
    print("=" * 60)
    print("\nNext step: Run 2_analyze_sentiment.py")


if __name__ == "__main__":
    # Create data folder if it doesn't exist
    import os

    os.makedirs('data', exist_ok=True)

    # Run main function
    main()