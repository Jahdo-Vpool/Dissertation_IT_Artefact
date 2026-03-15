"""
================================================================================
Script 7: Visualisation of Results (plots.py)
================================================================================
Purpose:
    Generates publication-quality figures for both TAN and SPY that visually
    communicate the findings of the sentiment-driven trading analysis. Charts
    support the Results and Discussion chapters of the dissertation by
    providing clear graphical evidence for each hypothesis test.

Academic Context:
    Visualisation is a critical component of quantitative research. Raw
    numerical outputs from performance metrics and correlation analyses are
    difficult to interpret in isolation. These figures contextualise the
    results by showing patterns across time, distributions of key variables,
    and direct comparisons between the two ETFs. Each chart is mapped
    explicitly to one or more dissertation hypotheses (H1, H2, H3).

Figures Produced Per Ticker (TAN and SPY):
    1. {TICKER}_cumulative_returns.png        - Tests H3 (strategy vs buy-and-hold)
    2. {TICKER}_sentiment_returns_scatter.png - Tests H1 and H2 (sentiment-return link)
    3. {TICKER}_trading_signals.png           - Illustrates the trading methodology
    4. {TICKER}_drawdown_comparison.png       - Risk analysis supporting H3
    5. {TICKER}_sentiment_distribution.png    - Data quality and sentiment overview

Comparative Figure:
    6. TAN_vs_SPY_comparison.png              - Tests H1 (differential sentiment effects)

Inputs:
    - results/TAN_trading_signals.csv
    - results/SPY_trading_signals.csv
    - results/TAN_merge_prices_news.csv
    - results/SPY_merge_prices_news.csv
    - results/performance_metrics_detailed.csv

Outputs:
    - plots/  (directory containing all eleven .png figures at 300 DPI)

Pipeline Position:
    Script 7 of 10. Runs after calculate_metrics.py. All input files must
    exist before this script is executed.

Dependencies:
    - pandas     : Data loading and manipulation
    - matplotlib : Core plotting library
    - seaborn    : Style and colour palette management
    - numpy      : Numerical operations (drawdown, trend lines)
    - os         : Directory creation

Usage:
    python plots.py
================================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ============================================
# SECTION 1: GLOBAL STYLE CONFIGURATION
# ============================================

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

os.makedirs('plots', exist_ok=True)

# Tickers to generate plots for
TICKERS = ['TAN', 'SPY']


# ============================================
# SECTION 2: INDIVIDUAL PLOTTING FUNCTIONS
# ============================================

def plot_cumulative_returns(ticker):
    """
    Generate a line chart comparing cumulative returns of the sentiment-based
    trading strategy against a passive buy-and-hold benchmark.

    Dissertation relevance:
        Directly tests Hypothesis 3 (H3) — whether a sentiment-informed
        strategy outperforms buy-and-hold. Produced for both TAN and SPY
        to support the comparative analysis.

    Parameters
    ----------
    ticker : str
        ETF ticker symbol ('TAN' or 'SPY').
    """

    print(f"    Creating cumulative returns comparison for {ticker}...")

    df = pd.read_csv(f'results/{ticker}_trading_signals.csv')

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df.index, df['cumulative_return'] * 100,
            linewidth=2.5, label='Buy & Hold', color='tab:blue')
    ax.plot(df.index, df['cumulative_strategy'] * 100,
            linewidth=2.5, label='Sentiment Strategy', color='tab:green')

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    ax.set_xlabel('Trading Days', fontsize=12)
    ax.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax.set_title(f'{ticker}: Cumulative Returns Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    filename = f'plots/{ticker}_cumulative_returns.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"      Saved: {filename}")


def plot_sentiment_distribution(ticker):
    """
    Generate a two-panel figure showing the distribution of daily sentiment
    scores and their breakdown by sentiment classification label.

    Dissertation relevance:
        Serves as a data quality check and provides descriptive context for
        the sentiment corpus for each ETF. Comparing TAN and SPY distributions
        supports the discussion of why strategy behaviour differs between them.

    Parameters
    ----------
    ticker : str
        ETF ticker symbol ('TAN' or 'SPY').
    """

    print(f"    Creating sentiment distribution for {ticker}...")

    df = pd.read_csv(f'results/{ticker}_merge_prices_news.csv')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel 1: Histogram of sentiment scores ---
    ax1.hist(df['sentiment_score'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Neutral')
    ax1.axvline(x=0.05, color='green', linestyle='--', linewidth=2, label='Buy Threshold')
    ax1.axvline(x=-0.05, color='orange', linestyle='--', linewidth=2, label='Sell Threshold')
    ax1.set_xlabel('Sentiment Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Sentiment Score Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Box plot by classification label ---
    sentiment_by_label = []
    labels = []
    for label in ['negative', 'neutral', 'positive']:
        scores = df[df['sentiment_label'] == label]['sentiment_score']
        if len(scores) > 0:
            sentiment_by_label.append(scores)
            labels.append(label.capitalize())

    if sentiment_by_label:
        bp = ax2.boxplot(sentiment_by_label, tick_labels=labels, patch_artist=True)
        colors = ['#ff9999', 'lightgray', '#90ee90']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)

    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_ylabel('Sentiment Score', fontsize=12)
    ax2.set_title('Sentiment by Classification', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'{ticker}: Sentiment Analysis', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()

    filename = f'plots/{ticker}_sentiment_distribution.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"      Saved: {filename}")


def plot_trading_signals(ticker):
    """
    Plot ETF closing price with buy/sell signal markers overlaid and shaded
    regions indicating periods when the strategy holds a long position.

    Dissertation relevance:
        Illustrates the trading strategy methodology for both ETFs, showing
        how sentiment signals translate into market positions. Comparing TAN
        and SPY signal charts highlights differences in trading frequency
        driven by their respective sentiment distributions.

    Parameters
    ----------
    ticker : str
        ETF ticker symbol ('TAN' or 'SPY').
    """

    print(f"    Creating trading signals chart for {ticker}...")

    df = pd.read_csv(f'results/{ticker}_trading_signals.csv')

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(df.index, df['close'], linewidth=2, label='Close Price',
            color='black', alpha=0.7)

    buys = df[df['signal'] == 'BUY']
    ax.scatter(buys.index, buys['close'], color='green', marker='^',
               s=100, label='BUY Signal', zorder=5, alpha=0.8)

    sells = df[df['signal'] == 'SELL']
    ax.scatter(sells.index, sells['close'], color='red', marker='v',
               s=100, label='SELL Signal', zorder=5, alpha=0.8)

    in_market = df['position'] == 1
    ax.fill_between(df.index, df['close'].min() * 0.95, df['close'].max() * 1.05,
                    where=in_market, alpha=0.1, color='green', label='In Market')

    ax.set_xlabel('Trading Days', fontsize=12)
    ax.set_ylabel('Close Price ($)', fontsize=12)
    ax.set_title(f'{ticker}: Trading Signals Based on Sentiment', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    filename = f'plots/{ticker}_trading_signals.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"      Saved: {filename}")


def plot_drawdown_comparison(ticker):
    """
    Plot the drawdown profile of both the sentiment strategy and the
    buy-and-hold benchmark over the full analysis period.

    Dissertation relevance:
        Supports Hypothesis 3 (H3) by showing whether the sentiment strategy
        achieves a shallower drawdown than buy-and-hold. Produced for both
        TAN and SPY to enable direct risk comparison across ETF types.

    Parameters
    ----------
    ticker : str
        ETF ticker symbol ('TAN' or 'SPY').
    """

    print(f"    Creating drawdown comparison for {ticker}...")

    df = pd.read_csv(f'results/{ticker}_trading_signals.csv')

    def calculate_drawdown(cumulative_returns):
        """
        Compute the percentage drawdown at each point relative to the
        highest cumulative return achieved up to that point.
        """
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / (running_max + 1)
        return drawdown * 100

    buy_hold_dd = calculate_drawdown(df['cumulative_return'].values)
    strategy_dd = calculate_drawdown(df['cumulative_strategy'].values)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.fill_between(df.index, 0, buy_hold_dd, alpha=0.3,
                    color='tab:blue', label='Buy & Hold')
    ax.fill_between(df.index, 0, strategy_dd, alpha=0.3,
                    color='tab:green', label='Strategy')
    ax.plot(df.index, buy_hold_dd, linewidth=1.5, color='tab:blue')
    ax.plot(df.index, strategy_dd, linewidth=1.5, color='tab:green')

    ax.set_xlabel('Trading Days', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_title(f'{ticker}: Drawdown Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    max_bh_dd = buy_hold_dd.min()
    max_st_dd = strategy_dd.min()
    ax.text(0.02, 0.98,
            f'Buy & Hold Max DD: {max_bh_dd:.2f}%\nStrategy Max DD: {max_st_dd:.2f}%',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    filename = f'plots/{ticker}_drawdown_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"      Saved: {filename}")


def plot_sentiment_returns_scatter(ticker):
    """
    Generate a scatter plot of daily sentiment scores against the following
    day's ETF return, with a linear trend line and Pearson correlation.

    Dissertation relevance:
        Provides visual evidence for Hypotheses 1 and 2 (H1, H2). Produced
        for both TAN and SPY so the strength of the sentiment-return
        relationship can be compared directly across ETF types.

    Parameters
    ----------
    ticker : str
        ETF ticker symbol ('TAN' or 'SPY').
    """

    print(f"    Creating sentiment-returns scatter for {ticker}...")

    df = pd.read_csv(f'results/{ticker}_merge_prices_news.csv')

    df['next_return'] = df['return'].shift(-1)
    df = df[:-1]

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(df['sentiment_score'], df['next_return'],
                         alpha=0.6, s=50, c=df['sentiment_score'],
                         cmap='RdYlGn', edgecolors='black', linewidth=0.5)

    z = np.polyfit(df['sentiment_score'], df['next_return'], 1)
    p = np.poly1d(z)
    ax.plot(df['sentiment_score'], p(df['sentiment_score']),
            "r--", alpha=0.8, linewidth=2,
            label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

    corr = df['sentiment_score'].corr(df['next_return'])
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    ax.set_xlabel('Sentiment Score (Today)', fontsize=12)
    ax.set_ylabel('Return (Next Day) %', fontsize=12)
    ax.set_title(f'{ticker}: Sentiment vs Next-Day Returns', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Sentiment Score')

    filename = f'plots/{ticker}_sentiment_returns_scatter.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"      Saved: {filename}")


def create_comparison_dashboard():
    """
    Generate a four-panel dashboard comparing TAN and SPY across strategy
    performance, rolling volatility, sentiment scores, and key metrics.

    Dissertation relevance:
        Primary visual for testing Hypothesis 1 (H1). Places both ETFs
        side-by-side across multiple dimensions to assess whether TAN
        consistently exhibits stronger sentiment-driven behaviour than SPY.
    """

    print(f"    Creating TAN vs SPY comparison dashboard...")

    tan_df = pd.read_csv('results/TAN_trading_signals.csv')
    spy_df = pd.read_csv('results/SPY_trading_signals.csv')

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # --- Panel 1: Cumulative strategy returns ---
    ax = axes[0, 0]
    ax.plot(tan_df.index, tan_df['cumulative_strategy'] * 100,
            linewidth=2, label='TAN Strategy', color='tab:orange')
    ax.plot(spy_df.index, spy_df['cumulative_strategy'] * 100,
            linewidth=2, label='SPY Strategy', color='tab:blue')
    ax.set_title('Strategy Performance Comparison', fontsize=12, fontweight='bold')
    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Cumulative Return (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # --- Panel 2: Rolling volatility ---
    ax = axes[0, 1]
    tan_vol = tan_df['return'].rolling(window=10).std()
    spy_vol = spy_df['return'].rolling(window=10).std()
    ax.plot(tan_df.index, tan_vol, linewidth=2, label='TAN', color='tab:orange')
    ax.plot(spy_df.index, spy_vol, linewidth=2, label='SPY', color='tab:blue')
    ax.set_title('Rolling Volatility (10-day)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Volatility (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Sentiment score comparison ---
    ax = axes[1, 0]
    tan_sentiment = pd.read_csv('results/TAN_merge_prices_news.csv')
    spy_sentiment = pd.read_csv('results/SPY_merge_prices_news.csv')
    ax.plot(tan_sentiment.index, tan_sentiment['sentiment_score'],
            linewidth=2, alpha=0.7, label='TAN', color='tab:orange')
    ax.plot(spy_sentiment.index, spy_sentiment['sentiment_score'],
            linewidth=2, alpha=0.7, label='SPY', color='tab:blue')
    ax.set_title('Sentiment Score Comparison', fontsize=12, fontweight='bold')
    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Sentiment Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

    # --- Panel 4: Key performance metrics bar chart ---
    ax = axes[1, 1]
    metrics_df = pd.read_csv('results/performance_metrics_detailed.csv')
    tan_metrics = metrics_df[metrics_df['ticker'] == 'TAN'].iloc[0]
    spy_metrics = metrics_df[metrics_df['ticker'] == 'SPY'].iloc[0]

    categories = ['Return\n(%)', 'Sharpe\n(x10)', 'Max DD\n(%)', 'Win Rate\n(%)']
    tan_values = [
        tan_metrics['strategy_total_return'],
        tan_metrics['strategy_sharpe'] * 10,
        abs(tan_metrics['strategy_max_drawdown']),
        tan_metrics['win_rate']
    ]
    spy_values = [
        spy_metrics['strategy_total_return'],
        spy_metrics['strategy_sharpe'] * 10,
        abs(spy_metrics['strategy_max_drawdown']),
        spy_metrics['win_rate']
    ]

    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width / 2, tan_values, width, label='TAN', color='tab:orange', alpha=0.8)
    ax.bar(x + width / 2, spy_values, width, label='SPY', color='tab:blue', alpha=0.8)
    ax.set_title('Key Performance Metrics', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('TAN vs SPY: Comprehensive Comparison', fontsize=16,
                 fontweight='bold', y=0.995)
    fig.tight_layout()

    filename = 'plots/TAN_vs_SPY_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"      Saved: {filename}")


# ============================================
# SECTION 3: MAIN EXECUTION
# ============================================

def main():
    """
    Execute all visualisation functions for both TAN and SPY, followed by
    the comparative dashboard. Ticker-specific charts are grouped by ticker
    so output is easy to follow in the terminal.
    """

    print("\n" + "=" * 60)
    print("CREATING DISSERTATION VISUALISATIONS")
    print("=" * 60)

    # --- Per-ticker charts (TAN then SPY) ---
    for ticker in TICKERS:
        print(f"\n{ticker} Charts:")
        print(f"  1/5: Cumulative Returns         (H3 - Strategy Performance)")
        plot_cumulative_returns(ticker)
        print(f"  2/5: Sentiment-Returns Scatter  (H1, H2 - Predictive Relationship)")
        plot_sentiment_returns_scatter(ticker)
        print(f"  3/5: Trading Signals            (Methodology)")
        plot_trading_signals(ticker)
        print(f"  4/5: Drawdown Comparison        (H3 - Risk Analysis)")
        plot_drawdown_comparison(ticker)
        print(f"  5/5: Sentiment Distribution     (Data Quality)")
        plot_sentiment_distribution(ticker)

    # --- Comparative dashboard ---
    print("\nComparative Analysis:")
    print("  Creating TAN vs SPY Dashboard   (H1 - Differential Effects)")
    create_comparison_dashboard()

    # -----------------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("VISUALISATIONS COMPLETE")
    print("=" * 60)

    print("\nOutput location: plots/")
    print("\nFigures produced (11 total):")

    for ticker in TICKERS:
        print(f"\n  {ticker}:")
        print(f"    {ticker}_cumulative_returns.png          — H3 Testing")
        print(f"    {ticker}_sentiment_returns_scatter.png   — H1, H2 Testing")
        print(f"    {ticker}_trading_signals.png             — Methodology")
        print(f"    {ticker}_drawdown_comparison.png         — Risk Analysis")
        print(f"    {ticker}_sentiment_distribution.png      — Data Quality")

    print("\n  Comparative:")
    print("    TAN_vs_SPY_comparison.png                 — H1 Testing")

    print("\nDissertation chapter usage:")
    print("  Chapter 3 (Methodology):")
    print("    - TAN_trading_signals.png")
    print("    - SPY_trading_signals.png")
    print("    - TAN_sentiment_distribution.png")
    print("    - SPY_sentiment_distribution.png")
    print("  Chapter 5 (Results and Evaluation):")
    print("    - TAN_vs_SPY_comparison.png")
    print("    - TAN_sentiment_distribution.png")
    print("    - SPY_sentiment_distribution.png")
    print("    - TAN_sentiment_returns_scatter.png")
    print("    - SPY_sentiment_returns_scatter.png")
    print("    - TAN_trading_signals.png")
    print("    - TAN_cumulative_returns.png")
    print("    - TAN_drawdown_comparison.png")

    print("\nAll figures saved at 300 DPI — ready for dissertation insertion.")
    print("\nNext step: Run volatility_calculation.py")


# ============================================
# SECTION 4: ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()