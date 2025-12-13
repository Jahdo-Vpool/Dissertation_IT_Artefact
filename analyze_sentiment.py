"""
Step 2: Analyze Sentiment
Use VADER to score each article

What this does:
- Reads news from CSV
- Analyzes sentiment (positive/negative/neutral)
- Adds sentiment scores to CSV
- Saves results

"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ============================================
# MAIN CODE
# ============================================

def analyze_sentiment(df):
    """
    Analyze sentiment for all articles in a dataframe.

    Adds two new columns:
    - sentiment_score: Number from -1 (negative) to +1 (positive)
    - sentiment_label: "positive", "negative", or "neutral"
    """

    print(f"  Analyzing {len(df)} articles...")

    # Create VADER analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Store scores
    scores = []
    labels = []

    # Analyze each article
    for idx, row in df.iterrows():
        # Combine title and content
        # Title counts more (weight it 2x)
        text = row['title'] + ' ' + row['title'] + ' ' + row['content']

        # Get sentiment
        sentiment = analyzer.polarity_scores(text)

        # The 'compound' score is the overall score (-1 to +1)
        score = sentiment['compound']

        # Classify as positive/negative/neutral
        if score >= 0.05:
            label = 'positive'
        elif score <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'

        scores.append(score)
        labels.append(label)

    # Add to dataframe
    df['sentiment_score'] = scores
    df['sentiment_label'] = labels

    # Show summary
    print(f"    Analysis complete!")
    print(f"    Positive: {sum(df['sentiment_label'] == 'positive')} articles")
    print(f"    Negative: {sum(df['sentiment_label'] == 'negative')} articles")
    print(f"    Neutral:  {sum(df['sentiment_label'] == 'neutral')} articles")
    print(f"    Average sentiment: {df['sentiment_score'].mean():.3f}")

    return df

def main():
    """Main function."""

    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS")
    print("=" * 60)

    # Analyze TAN
    print("\nTAN (Solar Energy):")
    tan_df = pd.read_csv('data/TAN_news.csv')
    tan_df = analyze_sentiment(tan_df)
    tan_df.to_csv('results/TAN_with_sentiment.csv', index=False)
    print(f"  Saved: results/TAN_with_sentiment.csv")

    # Analyze SPY
    print("\nSPY (S&P 500):")
    spy_df = pd.read_csv('data/SPY_news.csv')
    spy_df = analyze_sentiment(spy_df)
    spy_df.to_csv('results/SPY_with_sentiment.csv', index=False)
    print(f"  Saved: results/SPY_with_sentiment.csv")

    print("\n" + "=" * 60)
    print("✓ SENTIMENT ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\nNext step: Run 3_download_prices.py")


if __name__ == "__main__":
    # Create results folder
    import os

    os.makedirs('results', exist_ok=True)

    # Run
    main()