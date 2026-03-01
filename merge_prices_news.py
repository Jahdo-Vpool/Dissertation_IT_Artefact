"""
================================================================================
Script 4: Merging Price Data and Sentiment Scores
================================================================================
Purpose:
    Combines the processed ETF price data (from Script 3) with the daily
    aggregated sentiment scores (from Script 2) into a single unified
    DataFrame for each ETF. The resulting merged dataset is the primary
    analytical input for all downstream scripts — trading signal generation,
    performance evaluation, volatility modelling, and correlation analysis.

Academic Context:
    This script is the integration point of the pipeline. Sentiment analysis
    and price data are collected and processed independently in the earlier
    scripts, but they have no analytical value in isolation. To test any of
    the three hypotheses, both variables must exist on a common time axis.
    This script constructs that shared time axis and handles the practical
    challenges that arise when merging two real-world datasets with different
    temporal structures.

    Specifically:
    - Price data exists only on trading days (Monday to Friday, excluding
      market holidays). Approximately 252 days per year.
    - News data exists on any calendar day, including weekends. Articles
      published on non-trading days carry sentiment that may not be
      reflected in prices until the next trading session.

    The merge strategy accounts for this mismatch using forward-fill
    imputation, which propagates weekend and holiday sentiment into the
    next available trading day.

Inputs:
    - data/{ticker}_prices_processed.csv   : Processed price data (Script 3)
    - results/{ticker}_with_sentiment.csv  : Article-level sentiment (Script 2)

Outputs:
    - results/TAN_merge_prices_news.csv    : Merged dataset for TAN
    - results/SPY_merge_prices_news.csv    : Merged dataset for SPY

Pipeline Position:
    Script 4 of 10. Output feeds into Script 5 (trading_signals.py),
    Script 9 (volitility_calculation.py), and Script 10 (correlation_analysis.py).

Usage:
    python merge_prices_news.py
================================================================================
"""

import pandas as pd


# ============================================
# SECTION 1: MERGE FUNCTION
# ============================================

def merge_data(ticker):
    """
    Load, validate, and merge price and sentiment data for a given ETF.

    The merge is performed as a left join on the price data, meaning every
    trading day in the price dataset is retained regardless of whether a
    matching sentiment score exists for that date. This preserves the full
    price history and avoids introducing gaps in the return series, which
    would cause issues in the volatility and trading signal calculations
    in later scripts.

    Parameters
    ----------
    ticker : str
        The ETF ticker symbol ('TAN' or 'SPY'). Used to construct the
        input file paths and label console output.

    Returns
    -------
    pandas.DataFrame or None
        A merged DataFrame indexed by trading date, containing price
        fields, daily sentiment scores, and article counts. Returns None
        if a critical date mismatch is detected between the two datasets.
    """

    print(f"\n{'='*60}")
    print(f"{ticker}")
    print(f"{'='*60}")

    # -----------------------------------------------------------------------
    # LOAD PRICE DATA
    # The prices_processed file is the output of Script 3. It contains
    # cleaned OHLCV data plus a calculated daily return column. The column
    # may be named 'return (%)' depending on how it was exported, so a
    # rename is applied here for consistency across the pipeline.
    # -----------------------------------------------------------------------

    prices = pd.read_csv(f'data/{ticker}_prices_processed.csv')

    # Standardise the return column name. The trailing '(%)' notation can
    # cause issues when referencing the column in arithmetic operations in
    # later scripts, so it is simplified to 'return' here.
    if 'return (%)' in prices.columns:
        prices = prices.rename(columns={'return (%)': 'return'})
        print(f"  Renamed 'return (%)' to 'return'")

    print(f"  Prices loaded: {len(prices)} trading days")
    print(f"    Date range: {prices['date'].min()} to {prices['date'].max()}")
    print(f"    Columns: {list(prices.columns)}")

    # -----------------------------------------------------------------------
    # LOAD SENTIMENT DATA
    # The sentiment file is the output of Script 2. It contains one row per
    # news article, with a sentiment_score assigned by VADER. Before merging,
    # this article-level data must be aggregated to a single daily score.
    # -----------------------------------------------------------------------

    sentiment = pd.read_csv(f'results/{ticker}_with_sentiment.csv')
    print(f"  Sentiment loaded: {len(sentiment)} articles")
    print(f"    Date range: {sentiment['date'].min()} to {sentiment['date'].max()}")


    # ============================================
    # SECTION 2: DATE OVERLAP VALIDATION
    # ============================================
    # Before attempting to merge, validate that the two datasets share an
    # overlapping date range. A mismatch — for example, price data from 2020
    # and news data from 2024 — would produce a merged dataset with no usable
    # rows and would silently propagate null values through all later scripts.
    # Detecting this early with a clear error message prevents difficult-to-
    # diagnose failures downstream.

    price_start = prices['date'].min()
    price_end   = prices['date'].max()
    news_start  = sentiment['date'].min()
    news_end    = sentiment['date'].max()

    print(f"\n  Date Range Validation:")
    print(f"    Prices:    {price_start} to {price_end}")
    print(f"    News:      {news_start} to {news_end}")

    # If the two date ranges do not overlap at all, abort the merge and
    # return None. The main() function handles this gracefully by skipping
    # the save step and printing a diagnostic message.
    if price_end < news_start or price_start > news_end:
        print(f"\n  ERROR: No date overlap detected between price and news data.")
        print(f"  Price data ends:  {price_end}")
        print(f"  News data starts: {news_start}")
        print(f"\n  Resolution: Re-run Script 3 using a date range that includes {news_start}.")
        return None

    # -----------------------------------------------------------------------
    # SENTIMENT DISTRIBUTION REPORT
    # Report the proportion of positive, negative, and neutral articles
    # before aggregation. This provides a sanity check on the sentiment
    # corpus quality and serves as a useful descriptive statistic for the
    # dissertation results section.
    # -----------------------------------------------------------------------

    print(f"\n  Article-level sentiment distribution:")
    label_counts = sentiment['sentiment_label'].value_counts()
    for label in ['positive', 'negative', 'neutral']:
        count = label_counts.get(label, 0)
        pct = count / len(sentiment) * 100
        print(f"    {label.capitalize()}: {count} ({pct:.1f}%)")


    # ============================================
    # SECTION 3: DAILY SENTIMENT AGGREGATION
    # ============================================
    # The sentiment dataset contains one row per article. Multiple articles
    # may be published on the same day. To align with the price data, which
    # contains one row per trading day, the article-level scores must be
    # collapsed to a single daily average sentiment score.
    #
    # Mean aggregation is chosen over alternatives (e.g. median, weighted
    # average) for simplicity and interpretability. Each article is treated
    # as an equally weighted observation of market sentiment on that date.
    # The article count is also retained as it reflects the volume of news
    # coverage, which could itself be informative in high-volatility periods.

    daily_sentiment = sentiment.groupby('date').agg({
        'sentiment_score': 'mean',   # Average sentiment across all articles on this date
        'ticker': 'count'            # Number of articles published on this date
    }).reset_index()

    # Rename the count column to a more descriptive label
    daily_sentiment = daily_sentiment.rename(columns={'ticker': 'article_count'})

    print(f"  Daily sentiment aggregated: {len(daily_sentiment)} unique dates")


    # ============================================
    # SECTION 4: DATE RANGE ALIGNMENT
    # ============================================
    # The price dataset may cover a longer period than the news dataset.
    # Trading days that fall outside the news coverage window are excluded
    # here to ensure the merged dataset only contains rows for which a
    # sentiment signal is either directly available or can be imputed via
    # forward-fill. Retaining price data beyond the news window would
    # introduce rows with no sentiment signal, which would distort the
    # correlation and regression results in Scripts 10 and 11.

    news_start = daily_sentiment['date'].min()
    news_end   = daily_sentiment['date'].max()

    prices_filtered = prices[
        (prices['date'] >= news_start) &
        (prices['date'] <= news_end)
    ].copy()

    print(f"\n  Aligning price data to news coverage window:")
    print(f"    Original price rows:  {len(prices)} days")
    print(f"    Retained price rows:  {len(prices_filtered)} days ({news_start} to {news_end})")
    print(f"    Excluded:             {len(prices) - len(prices_filtered)} days outside news range")

    if len(prices_filtered) == 0:
        print(f"\n  ERROR: No price rows remain after date filtering.")
        print(f"  The price and news datasets do not share any overlapping dates.")
        return None


    # ============================================
    # SECTION 5: LEFT JOIN MERGE
    # ============================================
    # A left join on the price data ensures every trading day is retained.
    # Trading days with no matching news date will receive NaN sentiment
    # values, which are handled by forward-fill imputation below.
    # This approach is consistent with the assumption that sentiment
    # persists until new information arrives — a principle drawn from
    # the efficient market hypothesis literature on information
    # incorporation into prices.

    merged = prices_filtered.merge(
        daily_sentiment,
        on='date',
        how='left'
    )

    # -----------------------------------------------------------------------
    # MISSING VALUE IMPUTATION
    # Forward-fill propagates the most recent available sentiment score
    # forward into trading days with no news (e.g. the Monday after a
    # quiet weekend). This is preferable to filling with zero, which would
    # imply a neutral sentiment signal and could distort the analysis
    # on days following strongly positive or negative news.
    # Any remaining NaN values at the very start of the series (before the
    # first article) are filled with zero as a neutral baseline.
    # Article count is filled with zero on days with no news coverage.
    # -----------------------------------------------------------------------

    merged['sentiment_score'] = merged['sentiment_score'].ffill()
    merged['sentiment_score'] = merged['sentiment_score'].fillna(0)
    merged['article_count']   = merged['article_count'].fillna(0)

    # -----------------------------------------------------------------------
    # SENTIMENT LABEL ASSIGNMENT
    # A categorical label is derived from the continuous sentiment score
    # using fixed thresholds of +/-0.05. This threshold is consistent with
    # the trading signal thresholds used in Script 5, ensuring that the
    # label here is directly interpretable in the context of the strategy.
    # Scores in the range (-0.05, +0.05) are labelled neutral, reflecting
    # the view that small deviations from zero represent noise rather than
    # a meaningful directional signal.
    # -----------------------------------------------------------------------

    def get_label(score):
        if score >= 0.05:
            return 'positive'
        elif score <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    merged['sentiment_label'] = merged['sentiment_score'].apply(get_label)

    # -----------------------------------------------------------------------
    # COVERAGE REPORT
    # -----------------------------------------------------------------------

    days_with_articles = (merged['article_count'] > 0).sum()

    print(f"\n  Merged dataset summary:")
    print(f"    Total trading days:      {len(merged)}")
    print(f"    Days with direct news:   {days_with_articles}")
    print(f"    News coverage rate:      {days_with_articles / len(merged) * 100:.1f}%")
    print(f"    Mean sentiment score:    {merged['sentiment_score'].mean():.3f}")
    print(f"    Mean daily return:       {merged['return'].mean():.3f}%")

    return merged


# ============================================
# SECTION 6: MAIN EXECUTION
# ============================================

def main():
    """
    Run the merge process for both TAN and SPY and save outputs to CSV.

    Both ETFs are processed independently using the same merge logic,
    which ensures consistency in the structure of their respective output
    files. The resulting CSVs are used by all subsequent analytical scripts.
    """

    print("\n" + "="*60)
    print("MERGE PRICES AND SENTIMENT")
    print("="*60)

    # --- TAN ---
    tan_merged = merge_data('TAN')

    if tan_merged is not None:
        output_file = 'results/TAN_merge_prices_news.csv'
        tan_merged.to_csv(output_file, index=False)
        print(f"\n  Saved: {output_file}")
    else:
        print(f"\n  TAN merge failed — verify date alignment between price and news data.")

    # --- SPY ---
    spy_merged = merge_data('SPY')

    if spy_merged is not None:
        output_file = 'results/SPY_merge_prices_news.csv'
        spy_merged.to_csv(output_file, index=False)
        print(f"\n  Saved: {output_file}")
    else:
        print(f"\n  SPY merge failed — verify date alignment between price and news data.")

    # -----------------------------------------------------------------------
    # SUMMARY REPORT
    # -----------------------------------------------------------------------

    print("\n" + "="*60)

    if tan_merged is not None or spy_merged is not None:
        print("MERGE COMPLETE")
        print("="*60)

        if tan_merged is not None:
            print(f"\nTAN: {len(tan_merged)} trading days")
            print(f"  Date range: {tan_merged['date'].min()} to {tan_merged['date'].max()}")

        if spy_merged is not None:
            print(f"\nSPY: {len(spy_merged)} trading days")
            print(f"  Date range: {spy_merged['date'].min()} to {spy_merged['date'].max()}")

        print("\nNext step: Run trading_signals.py")

    else:
        print("MERGE FAILED — DATE MISMATCH DETECTED")
        print("="*60)
        print("\nBoth ETFs failed to merge. Re-run Script 3 to download price data")
        print("that covers the same date range as the collected news articles.")


# ============================================
# SECTION 7: ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()
