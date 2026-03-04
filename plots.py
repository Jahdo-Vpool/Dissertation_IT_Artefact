"""
================================================================================
Script 7: Visualisation of Results (plots.py)
================================================================================
Purpose:
    Generates six publication-quality figures that visually communicate the
    findings of the sentiment-driven trading analysis. The charts are designed
    to support the Results and Discussion chapters of the dissertation by
    providing clear graphical evidence for each hypothesis test.

Academic Context:
    Visualisation is a critical component of quantitative research. Raw
    numerical outputs from performance metrics and correlation analyses are
    difficult to interpret in isolation. These figures contextualise the
    results by showing patterns across time, distributions of key variables,
    and direct comparisons between the two ETFs. Each chart is mapped
    explicitly to one or more dissertation hypotheses (H1, H2, H3) so that
    figures can be referenced directly in the written analysis.

Figures Produced and Their Dissertation Purpose:
    1. TAN_cumulative_returns.png       - Tests H3 (strategy vs buy-and-hold)
    2. TAN_sentiment_returns_scatter.png - Tests H1 and H2 (sentiment-return link)
    3. TAN_trading_signals.png          - Illustrates the trading methodology
    4. TAN_drawdown_comparison.png      - Risk analysis supporting H3
    5. TAN_sentiment_distribution.png   - Data quality and sentiment overview
    6. TAN_vs_SPY_comparison.png        - Tests H1 (differential sentiment effects)

Inputs:
    - results/TAN_trading_signals.csv
    - results/SPY_trading_signals.csv
    - results/TAN_merge_prices_news.csv
    - results/SPY_merge_prices_news.csv
    - results/performance_metrics_detailed.csv

Outputs:
    - plots/  (directory containing all six .png figures at 300 DPI)

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
# A consistent visual style is applied across all figures to ensure the
# dissertation charts appear cohesive and professional. The darkgrid style
# provides clear gridlines against a neutral background, which aids
# readability when figures are printed in greyscale as well as colour.
# The 'husl' palette is perceptually uniform, meaning colour differences
# are proportional across the spectrum — important for accessibility.

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create the output directory if it does not already exist.
# All figures are saved here for direct use in the dissertation.
os.makedirs('plots', exist_ok=True)


# ============================================
# SECTION 2: INDIVIDUAL PLOTTING FUNCTIONS
# ============================================

def plot_cumulative_returns(ticker):
    """
    Generate a line chart comparing cumulative returns of the sentiment-based
    trading strategy against a passive buy-and-hold benchmark.

    Dissertation relevance:
        This figure directly tests Hypothesis 3 (H3), which posits that a
        sentiment-informed trading strategy outperforms a simple buy-and-hold
        approach. The chart allows a visual assessment of whether the strategy
        generates superior returns, and at what points in time it diverges
        from or converges with the benchmark.

    Parameters
    ----------
    ticker : str
        ETF ticker symbol ('TAN' or 'SPY'). Determines which results file
        is loaded and which filename is used for the saved figure.
    """

    print(f"\n  Creating cumulative returns comparison for {ticker}...")

    df = pd.read_csv(f'results/{ticker}_trading_signals.csv')

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot cumulative returns for both strategies.
    # Values are multiplied by 100 to convert from decimal to percentage,
    # which is the conventional format for return charts in finance.
    ax.plot(df.index, df['cumulative_return'] * 100,
            linewidth=2.5, label='Buy & Hold', color='tab:blue')
    ax.plot(df.index, df['cumulative_strategy'] * 100,
            linewidth=2.5, label='Sentiment Strategy', color='tab:green')

    # A horizontal reference line at zero distinguishes periods of positive
    # and negative cumulative return, aiding interpretation.
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    ax.set_xlabel('Trading Days', fontsize=12)
    ax.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax.set_title(f'{ticker}: Cumulative Returns Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    filename = f'plots/{ticker}_cumulative_returns.png'
    # dpi=300 ensures the figure meets the resolution requirements for
    # academic publication and dissertation printing standards.
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    Saved: {filename}")


def plot_sentiment_distribution(ticker):
    """
    Generate a two-panel figure showing the distribution of daily sentiment
    scores and their breakdown by sentiment classification label.

    Dissertation relevance:
        This figure serves as a data quality check and provides descriptive
        context for the sentiment corpus. It demonstrates that the VADER
        scores are spread across the full range and that the classification
        thresholds (buy at +0.05, sell at -0.05) are positioned meaningfully
        relative to the observed distribution. This supports the methodology
        discussion in the dissertation.

    Parameters
    ----------
    ticker : str
        ETF ticker symbol ('TAN' or 'SPY').
    """

    print(f"\n  Creating sentiment distribution for {ticker}...")

    df = pd.read_csv(f'results/{ticker}_merge_prices_news.csv')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel 1: Histogram of sentiment scores ---
    # The histogram shows the full distribution of VADER compound scores.
    # Vertical reference lines mark the neutral point (0.0) and the
    # trading signal thresholds (+/- 0.05) used in Script 5.
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
    # Grouping scores by their classification (negative, neutral, positive)
    # verifies that the VADER labels are internally consistent — positive
    # labels should correspond to higher scores than negative labels.
    sentiment_by_label = []
    labels = []
    for label in ['negative', 'neutral', 'positive']:
        scores = df[df['sentiment_label'] == label]['sentiment_score']
        if len(scores) > 0:
            sentiment_by_label.append(scores)
            labels.append(label.capitalize())

    if sentiment_by_label:
        bp = ax2.boxplot(sentiment_by_label, tick_labels=labels, patch_artist=True)
        # Colour coding: red for negative, grey for neutral, green for positive.
        # This convention is consistent with standard financial charting practice.
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

    print(f"    Saved: {filename}")


def plot_trading_signals(ticker):
    """
    Plot ETF closing price as a time series with buy and sell signal markers
    overlaid, and shaded regions indicating periods when the strategy holds
    a long position.

    Dissertation relevance:
        This figure illustrates the trading strategy methodology, showing
        how and when sentiment signals translate into market positions. It
        is most appropriate for the Methodology chapter to demonstrate that
        the signal generation logic produces a plausible and interpretable
        pattern of trades rather than random noise.

    Parameters
    ----------
    ticker : str
        ETF ticker symbol ('TAN' or 'SPY').
    """

    print(f"\n  Creating trading signals chart for {ticker}...")

    df = pd.read_csv(f'results/{ticker}_trading_signals.csv')

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot the closing price series as the primary time series.
    ax.plot(df.index, df['close'], linewidth=2, label='Close Price',
            color='black', alpha=0.7)

    # Buy signals: upward-pointing triangles (conventional financial charting
    # convention for long entry points).
    buys = df[df['signal'] == 'BUY']
    ax.scatter(buys.index, buys['close'], color='green', marker='^',
               s=100, label='BUY Signal', zorder=5, alpha=0.8)

    # Sell signals: downward-pointing triangles (conventional for exit points).
    sells = df[df['signal'] == 'SELL']
    ax.scatter(sells.index, sells['close'], color='red', marker='v',
               s=100, label='SELL Signal', zorder=5, alpha=0.8)

    # Shaded regions highlight days when the strategy holds an active long
    # position (position == 1), providing a clear visual of market exposure
    # relative to the price series.
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

    print(f"    Saved: {filename}")


def plot_drawdown_comparison(ticker):
    """
    Plot the drawdown profile of both the sentiment strategy and the
    buy-and-hold benchmark over the full analysis period.

    Dissertation relevance:
        Drawdown measures the peak-to-trough decline in portfolio value and
        is one of the most important risk metrics in quantitative finance.
        This figure supports Hypothesis 3 (H3) by showing whether the
        sentiment strategy achieves a shallower drawdown than buy-and-hold,
        which would indicate superior downside risk management even if
        absolute returns are comparable.

    Parameters
    ----------
    ticker : str
        ETF ticker symbol ('TAN' or 'SPY').
    """

    print(f"\n  Creating drawdown comparison for {ticker}...")

    df = pd.read_csv(f'results/{ticker}_trading_signals.csv')

    def calculate_drawdown(cumulative_returns):
        """
        Compute the percentage drawdown at each point in time relative to
        the highest cumulative return achieved up to that point.

        The running maximum captures the portfolio's historical peak value.
        Dividing the difference between current value and peak by the peak
        produces a normalised drawdown measure that is comparable across
        different asset classes and time periods.
        """
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / (running_max + 1)
        return drawdown * 100  # Convert to percentage

    buy_hold_dd = calculate_drawdown(df['cumulative_return'].values)
    strategy_dd = calculate_drawdown(df['cumulative_strategy'].values)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Filled area charts emphasise the magnitude of drawdown periods more
    # effectively than line charts alone, making peak drawdown events
    # immediately apparent to the reader.
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

    # Annotate the maximum drawdown values directly on the chart so the
    # key risk figures are visible without requiring the reader to consult
    # a separate table.
    max_bh_dd = buy_hold_dd.min()
    max_st_dd = strategy_dd.min()
    ax.text(0.02, 0.98,
            f'Buy & Hold Max DD: {max_bh_dd:.2f}%\nStrategy Max DD: {max_st_dd:.2f}%',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    filename = f'plots/{ticker}_drawdown_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    Saved: {filename}")


def plot_sentiment_returns_scatter(ticker):
    """
    Generate a scatter plot of daily sentiment scores against the following
    day's ETF return, with a linear trend line and Pearson correlation
    coefficient displayed.

    Dissertation relevance:
        This figure provides visual evidence for Hypotheses 1 and 2 (H1, H2).
        A positive slope and statistically meaningful correlation would suggest
        that higher sentiment scores are associated with higher next-day returns,
        supporting the notion that sentiment contains predictive information
        about short-term price movements. The one-day lag (using today's
        sentiment to predict tomorrow's return) is a standard approach in
        financial sentiment research that avoids look-ahead bias.

    Parameters
    ----------
    ticker : str
        ETF ticker symbol ('TAN' or 'SPY').
    """

    print(f"\n  Creating sentiment-returns correlation for {ticker}...")

    df = pd.read_csv(f'results/{ticker}_merge_prices_news.csv')

    # Shift returns forward by one day so that each sentiment score is
    # aligned with the return it is hypothesised to predict, not the return
    # from the same day (which would introduce look-ahead bias).
    df['next_return'] = df['return'].shift(-1)
    df = df[:-1]  # Remove the final row, which has no next-day return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Colour the scatter points by sentiment score using a red-yellow-green
    # diverging palette, which reinforces the negative/neutral/positive
    # classification visually.
    scatter = ax.scatter(df['sentiment_score'], df['next_return'],
                         alpha=0.6, s=50, c=df['sentiment_score'],
                         cmap='RdYlGn', edgecolors='black', linewidth=0.5)

    # Fit a first-degree polynomial (linear trend line) to the data.
    # The slope of this line indicates the direction and magnitude of the
    # relationship between sentiment and next-day returns.
    z = np.polyfit(df['sentiment_score'], df['next_return'], 1)
    p = np.poly1d(z)
    ax.plot(df['sentiment_score'], p(df['sentiment_score']),
            "r--", alpha=0.8, linewidth=2,
            label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')

    # Reference lines at zero for both axes divide the plot into four
    # quadrants: high sentiment/positive return, high sentiment/negative return,
    # low sentiment/positive return, and low sentiment/negative return.
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

    # Display the Pearson correlation coefficient directly on the chart.
    # This is the same statistic reported in the hypothesis testing results,
    # providing a direct link between the figure and the statistical analysis.
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

    print(f"    Saved: {filename}")


def create_comparison_dashboard():
    """
    Generate a four-panel dashboard providing a side-by-side comparison
    of TAN and SPY across strategy performance, rolling volatility,
    sentiment scores, and key performance metrics.

    Dissertation relevance:
        This is the primary visual for testing Hypothesis 1 (H1), which
        proposes that news sentiment has a stronger and more direct effect on
        the sector-specific TAN ETF than on the diversified SPY ETF. Placing
        both ETFs side-by-side across multiple dimensions allows the reader
        to assess whether TAN consistently exhibits stronger sentiment-driven
        behaviour without needing to cross-reference separate figures.
    """

    print(f"\n  Creating TAN vs SPY comparison dashboard...")

    tan_df = pd.read_csv('results/TAN_trading_signals.csv')
    spy_df = pd.read_csv('results/SPY_trading_signals.csv')

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # --- Panel 1: Cumulative strategy returns ---
    # Compares the total return generated by the sentiment strategy for each
    # ETF over the full analysis period. Divergence between the two lines
    # indicates differential performance attributable to the strategy.
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
    # A 10-day rolling standard deviation of daily returns is used as a
    # simple measure of short-term price volatility. TAN is expected to show
    # higher and more variable volatility than SPY due to its sector
    # concentration, which is central to the H1 argument.
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
    # Overlaying TAN and SPY sentiment scores shows whether the two series
    # move together (correlated market-wide sentiment) or independently
    # (sector-specific sentiment dynamics). Greater independence in TAN's
    # sentiment would support the H1 hypothesis.
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
    # A grouped bar chart summarising four headline metrics enables rapid
    # comparison without requiring the reader to search through result tables.
    # Sharpe ratio is scaled by a factor of 10 solely for visual legibility
    # alongside percentage-scale metrics; this is noted in the chart title.
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

    print(f"    Saved: {filename}")


# ============================================
# SECTION 3: MAIN EXECUTION
# ============================================

def main():
    """
    Execute all six visualisation functions in sequence.

    The charts are ordered so that TAN-specific figures are produced first,
    followed by the comparative TAN vs SPY dashboard. This order mirrors the
    structure of the Results chapter, where TAN analysis is presented in
    detail before the cross-ETF comparison is introduced.
    """

    print("\n" + "=" * 60)
    print("CREATING DISSERTATION VISUALISATIONS")
    print("=" * 60)

    # --- TAN-specific charts ---
    print("\nTAN Charts:")

    # Chart 1: Tests H3 — does the strategy outperform buy-and-hold?
    print("  Chart 1/6: Cumulative Returns (H3 - Strategy Performance)")
    plot_cumulative_returns('TAN')

    # Chart 2: Tests H1 and H2 — does sentiment predict next-day returns?
    print("  Chart 2/6: Sentiment-Returns Scatter (H1, H2 - Predictive Relationship)")
    plot_sentiment_returns_scatter('TAN')

    # Chart 3: Methodology illustration — how signals are generated and applied
    print("  Chart 3/6: Trading Signals (Methodology)")
    plot_trading_signals('TAN')

    # Chart 4: Tests H3 risk dimension — does the strategy reduce drawdown?
    print("  Chart 4/6: Drawdown Comparison (H3 - Risk Analysis)")
    plot_drawdown_comparison('TAN')

    # Chart 5: Data quality — distribution of the sentiment input variable
    print("  Chart 5/6: Sentiment Distribution (Data Quality)")
    plot_sentiment_distribution('TAN')

    # --- Comparative dashboard ---
    # Chart 6: Tests H1 — are sentiment effects stronger for TAN than SPY?
    print("\nComparative Analysis:")
    print("  Chart 6/6: TAN vs SPY Dashboard (H1 - Differential Effects)")
    create_comparison_dashboard()

    # -----------------------------------------------------------------------
    # SUMMARY AND DISSERTATION USAGE GUIDE
    # -----------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("VISUALISATIONS COMPLETE")
    print("=" * 60)

    print("\nOutput location: plots/")
    print("\nFigures produced:")
    print("  1. TAN_cumulative_returns.png         — H3 Testing")
    print("  2. TAN_sentiment_returns_scatter.png  — H1, H2 Testing")
    print("  3. TAN_trading_signals.png            — Methodology")
    print("  4. TAN_drawdown_comparison.png        — Risk Analysis")
    print("  5. TAN_sentiment_distribution.png     — Data Quality")
    print("  6. TAN_vs_SPY_comparison.png          — H1 Testing")

    print("\nDissertation chapter usage:")
    print("  Chapter 3 (Methodology):")
    print("    - TAN_trading_signals.png")
    print("    - TAN_sentiment_distribution.png")
    print("  Chapter 4 (Results):")
    print("    - TAN_cumulative_returns.png")
    print("    - TAN_sentiment_returns_scatter.png")
    print("    - TAN_drawdown_comparison.png")
    print("    - TAN_vs_SPY_comparison.png")

    print("\nAll figures saved at 300 DPI — ready for dissertation insertion.")
    print("\nNext step: Run volitility_calculation.py")


# ============================================
# SECTION 4: ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()