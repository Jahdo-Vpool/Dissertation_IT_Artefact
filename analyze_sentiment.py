"""
================================================================================
Script 2: Sentiment Analysis using VADER and FinBERT
================================================================================
Purpose:
    Processes the raw news articles collected in Script 1 and assigns a
    quantitative sentiment score to each article. Two models are applied
    independently — VADER and FinBERT — and their scores are averaged into
    a single combined score. This combined score becomes the primary sentiment
    variable used throughout the rest of the pipeline.

Academic Context:
    This script directly addresses Dissertation Objective 1 — extracting and
    quantifying financial sentiment from news articles. Sentiment analysis
    converts unstructured text into a numerical variable that can be correlated
    with price volatility (Objective 3) and used to generate trading signals
    (Objective 4). The use of two complementary models is a methodological
    strength: VADER provides a fast, interpretable baseline while FinBERT
    applies domain-specific knowledge of financial language, reducing the
    risk that results are an artefact of any single model's characteristics.

    The dual-model approach is also directly relevant to Hypothesis 1 (H1),
    which tests whether sentiment has a stronger effect on TAN than SPY. Using
    a combined score that draws from both general and finance-specific sentiment
    produces a more robust sentiment signal than either model alone.

Analysis Period:
    START_DATE : 2021-03-01
    END_DATE: 2026-03-01

Inputs:
    - data/TAN_news.csv  : Raw TAN news articles from Script 1
    - data/SPY_news.csv  : Raw SPY news articles from Script 1

Outputs:
    - results/{ticker}_with_both_sentiments.csv  : Full article-level data with
                                                   VADER, FinBERT, and combined scores
    - results/{ticker}_with_sentiment_vader.csv  : VADER scores only
    - results/{ticker}_with_sentiment_finbert.csv: FinBERT scores only
    - results/{ticker}_with_sentiment.csv        : Combined score (primary output,
                                                   used by all downstream scripts)

Pipeline Position:
    Script 2 of 10. Receives output from Script 1 (collect_news_gdelt.py).
    Output feeds into Script 4 (merge_prices_news.py).

Dependencies:
    - pandas              : Data loading and CSV export
    - vaderSentiment      : VADER lexicon-based sentiment analyser
    - transformers        : Hugging Face library for loading FinBERT
    - torch               : PyTorch backend required by FinBERT
    - warnings            : Suppresses non-critical FutureWarning messages

Usage:
    python analyze_sentiment.py
================================================================================
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax
import os
import warnings

# Suppress FutureWarning messages from the transformers library.
# These warnings relate to internal deprecations that do not affect
# the correctness of the analysis.
warnings.filterwarnings('ignore', category=FutureWarning)


# ============================================
# SECTION 1: DATE CLEANING UTILITY
# ============================================

def fix_date_format(df):
    """
    Standardise all date strings in the DataFrame to the format YYYY-MM-DD.

    GDELT returns dates in a compact timestamp format (e.g. '20251212T235959'),
    which is not directly compatible with the date format used in the price
    data from Yahoo Finance (e.g. '2025-12-12'). If these formats are not
    reconciled before the merge in Script 4, records will fail to join and
    the merged dataset will be empty or incomplete.

    This function handles three known GDELT date formats:
        - '20251212T235959'      : compact datetime with time component
        - '2025-12-12T23:59:59' : ISO datetime with separators
        - '20251212'             : compact date with no time component

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a 'date' column with raw GDELT date strings.

    Returns
    -------
    pandas.DataFrame
        The same DataFrame with the 'date' column standardised to YYYY-MM-DD.
    """

    def clean_date(date_str):
        # Strip the time component if present. GDELT separates date and time
        # with 'T', so splitting on 'T' and taking the first element isolates
        # the date portion only.
        if 'T' in str(date_str):
            date_str = str(date_str).split('T')[0]

        # Insert dashes if the date is in compact YYYYMMDD format.
        # A string of exactly 8 digits with no dashes indicates this format.
        date_str = str(date_str).strip()
        if len(date_str) == 8 and date_str.isdigit():
            date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

        return date_str

    df['date'] = df['date'].apply(clean_date)
    return df


# ============================================
# SECTION 2: VADER SENTIMENT ANALYSIS
# ============================================

def analyze_vader(df):
    """
    Score each article using the VADER (Valence Aware Dictionary and sEntiment
    Reasoner) sentiment analyser.

    VADER is a rule-based model developed by Hutto and Gilbert (2014) that
    assigns sentiment scores using a hand-crafted lexicon of words with
    pre-assigned valence scores. It is particularly well-suited to short,
    informal text and produces a compound score in the range [-1, +1], where
    values approaching +1 indicate strong positive sentiment and values
    approaching -1 indicate strong negative sentiment.

    Title Weighting:
        The article title is concatenated twice before the content field,
        effectively doubling the influence of the headline on the final score.
        This design choice reflects the fact that in GDELT data the title is
        the primary unique text field available. Weighting the title also
        aligns with established practice in financial NLP, where headlines are
        recognised as the primary carrier of market-relevant sentiment signals.

    Classification Threshold:
        Scores are classified as positive (>= +0.05), negative (<= -0.05),
        or neutral (between -0.05 and +0.05). The 0.05 threshold is the
        standard convention recommended in the original VADER documentation
        and is widely adopted in the sentiment analysis literature.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame of news articles with 'title' and 'content' columns.

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with two new columns: 'vader_score' (float, -1 to +1)
        and 'vader_label' (str: 'positive', 'negative', or 'neutral').
    """

    print(f"  Running VADER analysis...")

    # Initialise the VADER sentiment intensity analyser.
    # No model download is required as VADER uses a static lexicon.
    analyzer = SentimentIntensityAnalyzer()

    vader_scores = []
    vader_labels = []

    for idx, row in df.iterrows():
        # Construct the input text with title weighted double.
        text = row['title'] + ' ' + row['title'] + ' ' + row['content']

        # polarity_scores() returns a dictionary with four keys:
        # 'pos', 'neg', 'neu' (proportions), and 'compound' (overall score).
        # The compound score is used as it provides a single normalised value
        # suitable for comparison and aggregation across articles.
        sentiment = analyzer.polarity_scores(text)
        score = sentiment['compound']

        # Apply the standard 0.05 classification threshold.
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

    print(f"    VADER complete")
    print(f"      Positive: {sum(df['vader_label'] == 'positive')}")
    print(f"      Negative: {sum(df['vader_label'] == 'negative')}")
    print(f"      Neutral:  {sum(df['vader_label'] == 'neutral')}")
    print(f"      Average:  {df['vader_score'].mean():.3f}")

    return df


# ============================================
# SECTION 3: FINBERT SENTIMENT ANALYSIS
# ============================================

def analyze_finbert(df):
    """
    Score each article using FinBERT, a BERT-based transformer model
    fine-tuned on financial text by Araci (2019) via ProsusAI.

    Unlike VADER, which applies a general-purpose lexicon, FinBERT was
    trained on financial news, earnings call transcripts, and analyst reports.
    It therefore has a learned understanding of domain-specific vocabulary
    and context — for example, recognising that terms like 'downside risk'
    or 'margin compression' carry negative sentiment even without explicit
    negative words in the VADER lexicon.

    FinBERT Architecture:
        FinBERT is a sequence classification model built on the BERT base
        architecture. It outputs logits for three classes: positive, negative,
        and neutral. A softmax function converts these logits to probabilities
        that sum to 1. The sentiment score is derived as:
            score = positive_probability - negative_probability
        This yields a value in the range [-1, +1], directly comparable with
        the VADER compound score and suitable for equal-weight averaging.

    Input Truncation:
        BERT-based models have a maximum input length of 512 tokens. Text
        inputs are pre-truncated to 2,000 characters before tokenisation to
        reduce computational overhead. The tokeniser enforces the 512-token
        hard limit as an additional safeguard.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame of news articles with 'title' and 'content' columns.

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with two new columns: 'finbert_score' (float, -1 to +1)
        and 'finbert_label' (str: 'positive', 'negative', or 'neutral').
    """

    print(f"  Running FinBERT analysis...")
    print(f"    (Loading model - may take a minute on first run)")

    # Load the pre-trained FinBERT tokeniser and classification model from
    # the Hugging Face model hub. On the first run, approximately 500MB of
    # model weights are downloaded to a local cache directory.
    tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')

    finbert_scores = []
    finbert_labels = []

    for idx, row in df.iterrows():
        # Log progress every 50 articles. FinBERT is significantly slower
        # than VADER due to the neural inference step at each article.
        if (idx + 1) % 50 == 0:
            print(f"    Processing {idx + 1}/{len(df)}...")

        # Truncate input to 2,000 characters to reduce tokenisation time
        # while retaining the most informative portion of the text.
        text = (row['title'] + ' ' + row['content'])[:2000]

        # Tokenise the input. return_tensors="pt" returns PyTorch tensors.
        # truncation=True and max_length=512 enforce the BERT input limit.
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        # Run inference without computing gradients.
        # torch.no_grad() reduces memory usage and speeds up inference
        # as gradient tracking is only required during model training.
        with torch.no_grad():
            outputs = model(**inputs)

        # Convert raw logits to probabilities via softmax.
        # FinBERT output class order: [positive, negative, neutral]
        probs = softmax(outputs.logits, dim=1)[0]
        positive_prob = probs[0].item()
        negative_prob = probs[1].item()
        neutral_prob  = probs[2].item()

        # Derive a continuous score in [-1, +1] analogous to VADER compound.
        score = positive_prob - negative_prob

        # Assign the label of the class with the highest probability.
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

    print(f"    FinBERT complete")
    print(f"      Positive: {sum(df['finbert_label'] == 'positive')}")
    print(f"      Negative: {sum(df['finbert_label'] == 'negative')}")
    print(f"      Neutral:  {sum(df['finbert_label'] == 'neutral')}")
    print(f"      Average:  {df['finbert_score'].mean():.3f}")

    return df


# ============================================
# SECTION 4: COMBINED SCORE CALCULATION
# ============================================

def create_combined_score(df):
    """
    Average the VADER and FinBERT scores to produce a single combined
    sentiment score for each article.

    Rationale for Equal Weighting:
        Both models are assigned equal weight (0.5 each). This reflects the
        absence of a strong prior reason to prefer one model over the other
        at the article level. Equal weighting avoids introducing an additional
        tunable parameter that would require empirical justification. The
        model agreement rate (percentage of articles where both models assign
        the same label) is reported as a quality indicator: high agreement
        provides confidence in the combined labels; divergence highlights
        articles where models interpret the text differently.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'vader_score', 'vader_label', 'finbert_score',
        and 'finbert_label' columns.

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with two new columns: 'combined_score' (float, -1 to +1)
        and 'combined_label' (str: 'positive', 'negative', or 'neutral').
    """

    print(f"  Creating combined score...")

    # Simple arithmetic mean of the two model scores.
    df['combined_score'] = (df['vader_score'] + df['finbert_score']) / 2

    # Apply the same 0.05 classification threshold used for VADER.
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

    # Report inter-model agreement as a data quality metric.
    agreement = (df['vader_label'] == df['finbert_label']).sum()
    agreement_pct = agreement / len(df) * 100

    print(f"    Combined score created")
    print(f"      Positive: {sum(df['combined_label'] == 'positive')}")
    print(f"      Negative: {sum(df['combined_label'] == 'negative')}")
    print(f"      Neutral:  {sum(df['combined_label'] == 'neutral')}")
    print(f"      Average:  {df['combined_score'].mean():.3f}")
    print(f"      VADER-FinBERT agreement: {agreement_pct:.1f}%")

    return df


# ============================================
# SECTION 5: PER-TICKER ORCHESTRATION
# ============================================

def analyze_ticker(ticker):
    """
    Run the full dual-model sentiment analysis pipeline for a single ETF ticker
    and save all output files to the results directory.

    Four output files are produced per ticker. The primary file,
    {ticker}_with_sentiment.csv, uses standardised column names
    ('sentiment_score', 'sentiment_label') so that all downstream scripts
    can consume it without modification, regardless of which model's scores
    they were derived from. The three additional files support model
    comparison and academic reporting.

    Parameters
    ----------
    ticker : str
        ETF ticker symbol — either 'TAN' or 'SPY'.

    Returns
    -------
    pandas.DataFrame
        The fully scored article-level DataFrame, retained in memory for
        the cross-ticker summary printed in main().
    """

    print(f"\n{'='*60}")
    print(f"{ticker} - DUAL SENTIMENT ANALYSIS")
    print(f"{'='*60}")

    # Load the raw news CSV produced by Script 1.
    df = pd.read_csv(f'data/{ticker}_news.csv')
    print(f"  Loaded {len(df)} articles")

    # Standardise date format before any further processing.
    df = fix_date_format(df)

    # Apply both sentiment models sequentially.
    df = analyze_vader(df)
    df = analyze_finbert(df)

    # Compute the combined score from both model outputs.
    df = create_combined_score(df)

    # Save full results with all three score sets.
    output_all = f'results/{ticker}_with_both_sentiments.csv'
    df.to_csv(output_all, index=False)
    print(f"\n  Saved: {output_all}")

    # Save VADER-only results with standardised column names.
    df_vader = df[['ticker', 'date', 'title', 'content', 'source', 'url',
                   'vader_score', 'vader_label']].copy()
    df_vader = df_vader.rename(columns={
        'vader_score': 'sentiment_score',
        'vader_label': 'sentiment_label'
    })
    output_vader = f'results/{ticker}_with_sentiment_vader.csv'
    df_vader.to_csv(output_vader, index=False)
    print(f"  Saved: {output_vader}")

    # Save FinBERT-only results with standardised column names.
    df_finbert = df[['ticker', 'date', 'title', 'content', 'source', 'url',
                     'finbert_score', 'finbert_label']].copy()
    df_finbert = df_finbert.rename(columns={
        'finbert_score': 'sentiment_score',
        'finbert_label': 'sentiment_label'
    })
    output_finbert = f'results/{ticker}_with_sentiment_finbert.csv'
    df_finbert.to_csv(output_finbert, index=False)
    print(f"  Saved: {output_finbert}")

    # Save combined score results. This is the primary file used by Script 4.
    df_combined = df[['ticker', 'date', 'title', 'content', 'source', 'url',
                      'combined_score', 'combined_label']].copy()
    df_combined = df_combined.rename(columns={
        'combined_score': 'sentiment_score',
        'combined_label': 'sentiment_label'
    })
    output_combined = f'results/{ticker}_with_sentiment.csv'
    df_combined.to_csv(output_combined, index=False)
    print(f"  Saved: {output_combined} (primary - used by Script 4)")

    return df


# ============================================
# SECTION 6: MAIN EXECUTION
# ============================================

def main():
    """
    Run dual sentiment analysis for both TAN and SPY, then print a
    cross-ticker comparison of neutral article rates across both models.

    The neutral article rate is a useful diagnostic: VADER tends to classify
    a higher proportion of financial headlines as neutral because its general
    lexicon is less sensitive to finance-specific language. FinBERT typically
    produces a lower neutral rate for the same corpus. A large divergence
    between the two models is expected and reinforces the rationale for
    combining both rather than relying on either alone.
    """

    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS: VADER + FINBERT")
    print("Period: 2021-03-01 to 2026-03-01")
    print("="*60)

    tan_df = analyze_ticker('TAN')
    spy_df = analyze_ticker('SPY')

    # Cross-ticker neutral rate comparison. Systematic differences in
    # sentiment distribution between TAN and SPY provide early directional
    # evidence for Hypothesis 1 (H1).
    print("\n" + "="*60)
    print("DUAL SENTIMENT ANALYSIS COMPLETE")
    print("="*60)

    print("\nOutput files created per ticker:")
    print("  1. {ticker}_with_both_sentiments.csv   - All scores")
    print("  2. {ticker}_with_sentiment_vader.csv   - VADER only")
    print("  3. {ticker}_with_sentiment_finbert.csv - FinBERT only")
    print("  4. {ticker}_with_sentiment.csv          - Combined (primary)")

    print("\nCross-ticker neutral rate comparison:")
    for ticker, df in [('TAN', tan_df), ('SPY', spy_df)]:
        print(f"\n{ticker}:")
        print(f"  VADER neutrals:    {sum(df['vader_label'] == 'neutral')} "
              f"({sum(df['vader_label'] == 'neutral')/len(df)*100:.1f}%)")
        print(f"  FinBERT neutrals:  {sum(df['finbert_label'] == 'neutral')} "
              f"({sum(df['finbert_label'] == 'neutral')/len(df)*100:.1f}%)")
        print(f"  Combined neutrals: {sum(df['combined_label'] == 'neutral')} "
              f"({sum(df['combined_label'] == 'neutral')/len(df)*100:.1f}%)")

    print("\nNext step: Run process_prices_csv.py")


# ============================================
# SECTION 7: ENTRY POINT
# ============================================

if __name__ == "__main__":
    # Ensure the results directory exists before writing output files.
    os.makedirs('results', exist_ok=True)

    main()