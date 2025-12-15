"""
Step 2: Analyze Sentiment with BOTH VADER and FinBERT
Compare general-purpose vs finance-specific sentiment

What this does:
- Runs VADER (general-purpose sentiment)
- Runs FinBERT (finance-specific sentiment)
- Calculates combined score (average of both)
- Creates multiple output files for comparison

Run: python 2_analyze_sentiment_both.py
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


# ============================================
# VADER ANALYSIS
# ============================================

def fix_date_format(df):
    def clean_date(date_str):
        # Handle: '20251212T235959' or '2025-12-12T23:59:59' or '20251212'

        # Step 1: Remove time (anything after T)
        if 'T' in str(date_str):
            date_str = str(date_str).split('T')[0]
        # Now we have either: '20251212' or '2025-12-12'

        # Step 2: Add dashes if they're missing
        date_str = str(date_str).strip()
        if len(date_str) == 8 and date_str.isdigit():
            # It's '20251212' → make it '2025-12-12'
            date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"


        return date_str

    df['date'] = df['date'].apply(clean_date)
    return df

# ============================================
# VADER ANALYSIS
# ============================================

def analyze_vader(df):
    """Analyze sentiment using VADER."""

    print(f"  Running VADER analysis...")

    analyzer = SentimentIntensityAnalyzer()

    vader_scores = []
    vader_labels = []

    for idx, row in df.iterrows():
        # Combine title and content (weight title 2x)
        text = row['title'] + ' ' + row['title'] + ' ' + row['content']

        # Get VADER sentiment
        sentiment = analyzer.polarity_scores(text)
        score = sentiment['compound']

        # Classify (using 0.05 threshold)
        if score >= 0.05:
            label = 'positive'
        elif score <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'

        vader_scores.append(score)
        vader_labels.append(label)

    df['vader_score'] = vader_scores
    df['vader_label'] = vader_labels

    print(f"    ✓ VADER complete")
    print(f"      Positive: {sum(df['vader_label'] == 'positive')}")
    print(f"      Negative: {sum(df['vader_label'] == 'negative')}")
    print(f"      Neutral:  {sum(df['vader_label'] == 'neutral')}")
    print(f"      Average:  {df['vader_score'].mean():.3f}")

    return df


# ============================================
# FINBERT ANALYSIS
# ============================================

def analyze_finbert(df):
    """Analyze sentiment using FinBERT."""

    print(f"  Running FinBERT analysis...")
    print(f"    (Loading model - may take a minute on first run)")

    # Load FinBERT
    tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')

    finbert_scores = []
    finbert_labels = []

    for idx, row in df.iterrows():
        # Show progress every 50 articles
        if (idx + 1) % 50 == 0:
            print(f"    Processing {idx + 1}/{len(df)}...")

        # Combine title and content (truncate if too long)
        text = row['title'] + ' ' + row['content']
        text = text[:2000]

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt",
                          truncation=True, padding=True, max_length=512)

        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)

        # Get probabilities [positive, negative, neutral]
        probs = softmax(outputs.logits, dim=1)[0]
        positive_prob = probs[0].item()
        negative_prob = probs[1].item()
        neutral_prob = probs[2].item()

        # Score: -1 (negative) to +1 (positive)
        score = positive_prob - negative_prob

        # Classify based on highest probability
        if positive_prob > max(negative_prob, neutral_prob):
            label = 'positive'
        elif negative_prob > max(positive_prob, neutral_prob):
            label = 'negative'
        else:
            label = 'neutral'

        finbert_scores.append(score)
        finbert_labels.append(label)

    df['finbert_score'] = finbert_scores
    df['finbert_label'] = finbert_labels

    print(f"    ✓ FinBERT complete")
    print(f"      Positive: {sum(df['finbert_label'] == 'positive')}")
    print(f"      Negative: {sum(df['finbert_label'] == 'negative')}")
    print(f"      Neutral:  {sum(df['finbert_label'] == 'neutral')}")
    print(f"      Average:  {df['finbert_score'].mean():.3f}")

    return df


# ============================================
# COMBINED ANALYSIS
# ============================================

def create_combined_score(df):
    """Create combined sentiment score (average of VADER and FinBERT)."""

    print(f"  Creating combined score...")

    # Combined score (average)
    df['combined_score'] = (df['vader_score'] + df['finbert_score']) / 2

    # Combined label (based on combined score)
    combined_labels = []
    for score in df['combined_score']:
        if score >= 0.05:
            label = 'positive'
        elif score <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        combined_labels.append(label)

    df['combined_label'] = combined_labels

    # Agreement analysis
    agreement = (df['vader_label'] == df['finbert_label']).sum()
    agreement_pct = agreement / len(df) * 100

    print(f"    ✓ Combined score created")
    print(f"      Positive: {sum(df['combined_label'] == 'positive')}")
    print(f"      Negative: {sum(df['combined_label'] == 'negative')}")
    print(f"      Neutral:  {sum(df['combined_label'] == 'neutral')}")
    print(f"      Average:  {df['combined_score'].mean():.3f}")
    print(f"      VADER-FinBERT agreement: {agreement_pct:.1f}%")

    return df


# ============================================
# MAIN FUNCTION
# ============================================

def analyze_ticker(ticker):
    """Analyze sentiment for a ticker using both methods."""

    print(f"\n{'='*60}")
    print(f"{ticker} - DUAL SENTIMENT ANALYSIS")
    print(f"{'='*60}")

    # Read news data
    df = pd.read_csv(f'data/{ticker}_news.csv')
    print(f"  Loaded {len(df)} articles")

    # Convert dates to proper format
    df = fix_date_format(df)

    # Run VADER
    df = analyze_vader(df)

    # Run FinBERT
    df = analyze_finbert(df)

    # Create combined score
    df = create_combined_score(df)

    # Save results
    # Main file: all scores
    output_all = f'results/{ticker}_with_both_sentiments.csv'
    df.to_csv(output_all, index=False)
    print(f"\n  Saved: {output_all}")

    # VADER only file
    df_vader = df[['ticker', 'date', 'title', 'content', 'source', 'url',
                   'vader_score', 'vader_label']].copy()
    df_vader = df_vader.rename(columns={
        'vader_score': 'sentiment_score',
        'vader_label': 'sentiment_label'
    })
    output_vader = f'results/{ticker}_with_sentiment_vader.csv'
    df_vader.to_csv(output_vader, index=False)
    print(f"  Saved: {output_vader}")

    # FinBERT only file (for comparison)
    df_finbert = df[['ticker', 'date', 'title', 'content', 'source', 'url',
                     'finbert_score', 'finbert_label']].copy()
    df_finbert = df_finbert.rename(columns={
        'finbert_score': 'sentiment_score',
        'finbert_label': 'sentiment_label'
    })
    output_finbert = f'results/{ticker}_with_sentiment_finbert.csv'
    df_finbert.to_csv(output_finbert, index=False)
    print(f"  Saved: {output_finbert}")

    # Combined score file (recommended for analysis)
    df_combined = df[['ticker', 'date', 'title', 'content', 'source', 'url',
                      'combined_score', 'combined_label']].copy()
    df_combined = df_combined.rename(columns={
        'combined_score': 'sentiment_score',
        'combined_label': 'sentiment_label'
    })
    output_combined = f'results/{ticker}_with_sentiment.csv'
    df_combined.to_csv(output_combined, index=False)
    print(f"  Saved: {output_combined} (combined)")

    return df


def main():
    """Main function."""

    print("\n" + "="*60)

    # Analyze TAN
    tan_df = analyze_ticker('TAN')

    # Analyze SPY
    spy_df = analyze_ticker('SPY')

    # Summary
    print("\n" + "="*60)
    print("DUAL SENTIMENT ANALYSIS COMPLETE!")
    print("="*60)

    print("\nFiles created for each ticker:")
    print("  1. {ticker}_with_both_sentiments.csv    - All scores (VADER, FinBERT, Combined)")
    print("  2. {ticker}_with_sentiment_vader.csv    - VADER only")
    print("  3. {ticker}_with_sentiment_finbert.csv  - FinBERT only")
    print("  4. {ticker}_with_sentiment.csv          - Combined (recommended)")

    print("\nComparison summary:")
    print(f"\nTAN:")
    print(f"  VADER neutrals:   {sum(tan_df['vader_label'] == 'neutral')} ({sum(tan_df['vader_label'] == 'neutral')/len(tan_df)*100:.1f}%)")
    print(f"  FinBERT neutrals: {sum(tan_df['finbert_label'] == 'neutral')} ({sum(tan_df['finbert_label'] == 'neutral')/len(tan_df)*100:.1f}%)")
    print(f"  Combined neutrals: {sum(tan_df['combined_label'] == 'neutral')} ({sum(tan_df['combined_label'] == 'neutral')/len(tan_df)*100:.1f}%)")

    print(f"\nSPY:")
    print(f"  VADER neutrals:   {sum(spy_df['vader_label'] == 'neutral')} ({sum(spy_df['vader_label'] == 'neutral')/len(spy_df)*100:.1f}%)")
    print(f"  FinBERT neutrals: {sum(spy_df['finbert_label'] == 'neutral')} ({sum(spy_df['finbert_label'] == 'neutral')/len(spy_df)*100:.1f}%)")
    print(f"  Combined neutrals: {sum(spy_df['combined_label'] == 'neutral')} ({sum(spy_df['combined_label'] == 'neutral')/len(spy_df)*100:.1f}%)")

if __name__ == "__main__":
    # Create results folder
    os.makedirs('results', exist_ok=True)

    # Run
    main()