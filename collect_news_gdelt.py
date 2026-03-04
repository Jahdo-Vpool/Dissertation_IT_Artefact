"""
================================================================================
Script 1: News Data Collection via GDELT
================================================================================
Purpose:
    Collects financial news articles for TAN (Invesco Solar ETF) and SPY
    (S&P 500 ETF) from the GDELT Project database. The collected articles
    form the raw text corpus that feeds into the sentiment analysis pipeline
    (Script 2: analyze_sentiment.py).

Academic Context:
    This script addresses Dissertation Objective 1 — extracting and quantifying
    financial sentiment from news articles. Before sentiment can be measured,
    a sufficiently large and representative dataset of news must be gathered.
    GDELT is selected as the data source because it is freely accessible,
    requires no API key, and provides historically deep coverage, making it
    suitable for academic research with a multi-year date range.

Why GDELT:
    The GDELT (Global Database of Events, Language and Tone) Project monitors
    news media worldwide in real time. Unlike commercial APIs (e.g. NewsAPI,
    Bloomberg), GDELT imposes no rate limits or subscription fees, and offers
    access to articles dating back several years. This is important for
    constructing a dataset that aligns precisely with the ETF price history
    used in later analysis.

Inputs:
    - User-defined START_DATE and END_DATE (must match the price data range)
    - Keyword lists for TAN and SPY (domain-relevant search terms)

Outputs:
    - data/TAN_news.csv  : News articles related to the solar/clean energy sector
    - data/SPY_news.csv  : News articles related to the broad US equity market

Pipeline Position:
    Script 1 of 11. Output feeds directly into Script 2 (analyze_sentiment.py).

Dependencies:
    - gdeltdoc   : Python wrapper for the GDELT Document 2.0 API
    - pandas     : DataFrame construction and CSV export
    - datetime   : Date arithmetic for period calculation
    - time       : Rate-limiting between API requests

Usage:
    python collect_news_gdelt.py
================================================================================
"""

from gdeltdoc import GdeltDoc, Filters
import pandas as pd
from datetime import datetime, timedelta
import time


# ============================================
# SECTION 1: CONFIGURATION
# ============================================
# These dates define the data collection window. They must match the price
# data range used in Script 3 (process_prices_csv.py) so that sentiment
# scores and price returns share a common time axis when merged in Script 4.

START_DATE = '2021-03-01'  # Start of the analysis period
END_DATE = '2026-03-01'    # End of the analysis period

# Calculate the total length of the collection period for reporting purposes.
# This is used later to assess news coverage as a percentage of trading days.
start_dt = datetime.strptime(START_DATE, '%Y-%m-%d')
end_dt = datetime.strptime(END_DATE, '%Y-%m-%d')
days = (end_dt - start_dt).days
months = days / 30

print(f"\nCollection Period:")
print(f"  Start: {START_DATE}")
print(f"  End:   {END_DATE}")
print(f"  Duration: {days} days (~{months:.1f} months)")


# ============================================
# SECTION 2: NEWS COLLECTION FUNCTION
# ============================================

def collect_gdelt_news(ticker, keywords, max_articles=500):
    """
    Query the GDELT API for news articles matching a set of keywords,
    then clean and standardise the results into a single DataFrame.

    The function iterates over each keyword in the provided list, submits
    a separate GDELT query for each, and aggregates the results. This
    multi-keyword approach improves recall — using only a ticker symbol
    (e.g. 'TAN') would miss articles that discuss the underlying sector
    without naming the ETF directly.

    Parameters
    ----------
    ticker : str
        The ETF ticker symbol (e.g. 'TAN' or 'SPY'). Used for labelling
        records in the output DataFrame, not as a search term itself.

    keywords : list of str
        A list of search terms to query in GDELT. Each term is submitted
        as a separate API request. Terms should reflect the language
        commonly used in financial news for the given ETF's sector.

    max_articles : int, optional (default=500)
        Upper limit on the number of articles retained after deduplication.
        If more are found, the most recent articles are kept. This prevents
        the dataset from becoming dominated by high-frequency keywords.

    Returns
    -------
    pandas.DataFrame or None
        A cleaned DataFrame with columns: ticker, date, title, content,
        source, url. Returns None if no articles are found.
    """

    print(f"\n{'=' * 60}")
    print(f"Collecting GDELT news for {ticker}")
    print(f"{'=' * 60}")

    # Initialise the GDELT client. GdeltDoc wraps the GDELT Document 2.0
    # API, which returns structured metadata (title, URL, date, source domain)
    # for each matching news article.
    gd = GdeltDoc()

    all_articles = []  # Accumulator list for results across all keywords

    # --- Keyword Loop ---
    # Each keyword is queried independently. GDELT caps results per query
    # at 250 records, so using multiple targeted keywords increases the
    # total volume and thematic diversity of the corpus.
    for keyword in keywords:
        print(f"\n  Searching: '{keyword}'...")

        try:
            # Build a GDELT filter object specifying the search term,
            # date range, and the maximum number of records to retrieve.
            # num_records=250 is the GDELT API maximum per single request.
            f = Filters(
                keyword=keyword,
                start_date=START_DATE,
                end_date=END_DATE,
                num_records=250
            )

            # Submit the query and receive a DataFrame of matching articles.
            articles = gd.article_search(f)

            if articles is not None and len(articles) > 0:
                print(f"    Found {len(articles)} articles")

                # Standardise each article into a consistent dictionary
                # structure. GDELT returns 'seendate' in a compact datetime
                # format, so we slice the first 10 characters to extract
                # the date portion (YYYY-MM-DD) only.
                for idx, row in articles.iterrows():
                    all_articles.append({
                        'ticker': ticker,
                        'date': row['seendate'][:10] if 'seendate' in row else START_DATE,
                        'title': row['title'] if 'title' in row else '',
                        # GDELT does not provide full article body text via
                        # this API. The title is used as a proxy for content,
                        # which is a known limitation of free-tier news APIs.
                        # VADER and FinBERT can still extract meaningful
                        # sentiment signals from headline text alone.
                        'content': row['title'] if 'title' in row else '',
                        'source': row['domain'] if 'domain' in row else 'Unknown',
                        'url': row['url'] if 'url' in row else ''
                    })

                # Introduce a one-second delay between requests to avoid
                # overwhelming the GDELT server and to comply with
                # responsible API usage conventions.
                time.sleep(1)

            else:
                print(f"    No articles found")

        except Exception as e:
            # Log errors per keyword but continue to the next term rather
            # than terminating the entire collection run. This ensures a
            # partial dataset is still produced if some queries fail.
            print(f"    Error: {e}")
            continue

    # -----------------------------------------------------------------------
    # POST-COLLECTION CLEANING
    # -----------------------------------------------------------------------

    df = pd.DataFrame(all_articles)

    # If no articles were collected across all keywords, exit early.
    if df.empty:
        print(f"\n  No articles found for {ticker}")
        return None

    # Remove duplicate articles based on URL. The same article can appear
    # in results for multiple keywords, so deduplication is essential to
    # avoid inflating the sentiment scores for particular dates.
    original_count = len(df)
    df = df.drop_duplicates(subset=['url'], keep='first')
    print(f"\n  Removed {original_count - len(df)} duplicates")

    # Discard records with very short titles (fewer than 10 characters),
    # as these are unlikely to carry useful sentiment information and may
    # represent malformed or empty API responses.
    df = df[df['title'].str.len() > 10]

    # If the total collection exceeds max_articles, retain only the most
    # recent articles. Recency bias is preferred here because the trading
    # strategy is evaluated on its performance over the full period, and
    # the most recent news is most relevant to price movements near the
    # end of the analysis window.
    if len(df) > max_articles:
        print(f"  Limiting to {max_articles} most recent articles")
        df = df.sort_values('date', ascending=False).head(max_articles)

    # Sort the final DataFrame chronologically for consistency with the
    # time-ordered price data it will be merged with in Script 4.
    df = df.sort_values('date', ascending=False).reset_index(drop=True)

    # -----------------------------------------------------------------------
    # COVERAGE STATISTICS
    # -----------------------------------------------------------------------
    # Report how many calendar days in the analysis period have at least one
    # article. Low coverage would be a methodological concern, as days with
    # no news would receive no sentiment signal and would be excluded from
    # the correlation and regression analyses in Scripts 9 and 10.

    print(f"\n  FINAL COLLECTION:")
    print(f"    Total articles:    {len(df)}")
    print(f"    Unique dates:      {df['date'].nunique()}")
    print(f"    Date range:        {df['date'].min()} to {df['date'].max()}")

    articles_per_day = df.groupby('date').size()
    print(f"    Days with articles:{len(articles_per_day)}/{days}")
    print(f"    Average per day:   {articles_per_day.mean():.1f}")
    print(f"    Coverage:          {len(articles_per_day) / days * 100:.1f}%")

    # Show the most frequent news sources as a basic quality check.
    # A diverse set of sources reduces the risk of source-specific bias
    # in the sentiment corpus.
    top_sources = df['source'].value_counts().head(5)
    print(f"\n    Top sources:")
    for source, count in top_sources.items():
        print(f"      {source}: {count} articles")

    return df


# ============================================
# SECTION 3: MAIN EXECUTION
# ============================================

def main():
    """
    Orchestrate the full data collection run for both ETFs.

    TAN keywords focus on the solar and clean energy sector, reflecting
    the underlying holdings of the Invesco Solar ETF. SPY keywords focus
    on broad US equity market terms, reflecting the diversified nature of
    the S&P 500. The deliberate contrast in keyword sets is central to
    Hypothesis 1 (H1), which proposes that sector-specific sentiment has
    a stronger influence on TAN than general market sentiment has on SPY.
    """

    print("\n" + "=" * 60)
    print("GDELT NEWS COLLECTION - MAXIMUM COVERAGE")
    print("=" * 60)
    print("\nSource: GDELT Project (free, unlimited, historical)")
    print("Purpose: Raw text corpus for sentiment analysis pipeline")

    # -----------------------------------------------------------------------
    # TAN NEWS COLLECTION
    # Solar and clean energy keywords are chosen to capture the full range
    # of news that would plausibly influence investor sentiment toward the
    # solar sector. Both technical terms (e.g. 'photovoltaic') and market
    # terms (e.g. 'solar stocks', 'TAN ETF') are included to maximise recall.
    # -----------------------------------------------------------------------

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
        max_articles=1000
    )

    # -----------------------------------------------------------------------
    # SPY NEWS COLLECTION
    # Broad market keywords are chosen to reflect the general financial
    # discourse that drives sentiment toward the S&P 500. These terms are
    # deliberately general to reflect SPY's diversified composition across
    # all sectors, contrasting with the sector-specific TAN keyword set.
    # -----------------------------------------------------------------------

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
        max_articles=1000
    )

    # -----------------------------------------------------------------------
    # SAVE OUTPUTS
    # Articles are saved to CSV in the /data directory. These files serve
    # as the direct input to Script 2 (analyze_sentiment.py), where VADER
    # and FinBERT sentiment scores are computed for each article title.
    # -----------------------------------------------------------------------

    if tan_articles is not None:
        tan_articles.to_csv('data/TAN_news.csv', index=False)
        print(f"\nSaved: data/TAN_news.csv ({len(tan_articles)} articles)")

    if spy_articles is not None:
        spy_articles.to_csv('data/SPY_news.csv', index=False)
        print(f"Saved: data/SPY_news.csv ({len(spy_articles)} articles)")

    # -----------------------------------------------------------------------
    # SUMMARY REPORT
    # -----------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)

    if tan_articles is not None and spy_articles is not None:
        print(f"\nCollected:")
        print(f"  TAN: {len(tan_articles)} articles over {tan_articles['date'].nunique()} days")
        print(f"  SPY: {len(spy_articles)} articles over {spy_articles['date'].nunique()} days")
        print(f"\nPeriod: {START_DATE} to {END_DATE} (~{months:.1f} months)")
        print(f"\nTotal articles for analysis: {len(tan_articles) + len(spy_articles)}")

    print("\nNext step: Run analyze_sentiment.py")


# ============================================
# SECTION 4: ENTRY POINT
# ============================================

if __name__ == "__main__":
    import os

    # Create the /data directory if it does not already exist.
    # All raw data files produced by this script are stored here.
    os.makedirs('data', exist_ok=True)

    main()