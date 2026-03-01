"""
Step 7: Create Visualizations
Generate professional charts for dissertation

What this does:
- Creates time series plots (price, sentiment, volatility)
- Compares strategy vs buy-and-hold performance
- Shows sentiment distribution
- Overlays trading signals on price charts
- Generates publication-quality figures (PNG)

Run: python 7_create_plots.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style for professional-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Ensure plots directory exists
os.makedirs('plots', exist_ok=True)


# ============================================
# PLOTTING FUNCTIONS
# ============================================

def plot_price_sentiment_timeseries(ticker):
    """Plot price and sentiment over time (dual axis)."""

    print(f"\n  Creating price & sentiment time series for {ticker}...")

    df = pd.read_csv(f'results/{ticker}_merge_prices_news.csv')

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Price on left axis
    color = 'tab:blue'
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Close Price ($)', color=color, fontsize=12)
    ax1.plot(df.index, df['close'], color=color, linewidth=2, label='Close Price')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Sentiment on right axis
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Sentiment Score', color=color, fontsize=12)
    ax2.plot(df.index, df['sentiment_score'], color=color, linewidth=2,
             alpha=0.7, label='Sentiment Score')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and formatting
    plt.title(f'{ticker}: Price and Sentiment Over Time', fontsize=14, fontweight='bold')
    fig.tight_layout()

    # Save
    filename = f'plots/{ticker}_price_sentiment_timeseries.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ✅ Saved: {filename}")


def plot_cumulative_returns(ticker):
    """Plot cumulative returns: Strategy vs Buy-and-Hold."""

    print(f"\n  Creating cumulative returns comparison for {ticker}...")

    df = pd.read_csv(f'results/{ticker}_trading_signals.csv')

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot both strategies
    ax.plot(df.index, df['cumulative_return'] * 100,
            linewidth=2.5, label='Buy & Hold', color='tab:blue')
    ax.plot(df.index, df['cumulative_strategy'] * 100,
            linewidth=2.5, label='Sentiment Strategy', color='tab:green')

    # Zero line
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Formatting
    ax.set_xlabel('Trading Days', fontsize=12)
    ax.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax.set_title(f'{ticker}: Cumulative Returns Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Save
    filename = f'plots/{ticker}_cumulative_returns.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ✅ Saved: {filename}")


def plot_sentiment_distribution(ticker):
    """Plot sentiment score distribution."""

    print(f"\n  Creating sentiment distribution for {ticker}...")

    df = pd.read_csv(f'results/{ticker}_merge_prices_news.csv')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(df['sentiment_score'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Neutral')
    ax1.axvline(x=0.05, color='green', linestyle='--', linewidth=2, label='Buy Threshold')
    ax1.axvline(x=-0.05, color='orange', linestyle='--', linewidth=2, label='Sell Threshold')
    ax1.set_xlabel('Sentiment Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Sentiment Score Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot by label
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

    # Save
    filename = f'plots/{ticker}_sentiment_distribution.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ✅ Saved: {filename}")


def plot_trading_signals(ticker):
    """Plot price with trading signals overlay."""

    print(f"\n  Creating trading signals chart for {ticker}...")

    df = pd.read_csv(f'results/{ticker}_trading_signals.csv')

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot price
    ax.plot(df.index, df['close'], linewidth=2, label='Close Price', color='black', alpha=0.7)

    # Mark buy signals
    buys = df[df['signal'] == 'BUY']
    ax.scatter(buys.index, buys['close'], color='green', marker='^',
               s=100, label='BUY Signal', zorder=5, alpha=0.8)

    # Mark sell signals
    sells = df[df['signal'] == 'SELL']
    ax.scatter(sells.index, sells['close'], color='red', marker='v',
               s=100, label='SELL Signal', zorder=5, alpha=0.8)

    # Shade periods when in market
    in_market = df['position'] == 1
    ax.fill_between(df.index, df['close'].min() * 0.95, df['close'].max() * 1.05,
                    where=in_market, alpha=0.1, color='green', label='In Market')

    # Formatting
    ax.set_xlabel('Trading Days', fontsize=12)
    ax.set_ylabel('Close Price ($)', fontsize=12)
    ax.set_title(f'{ticker}: Trading Signals Based on Sentiment', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Save
    filename = f'plots/{ticker}_trading_signals.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ✅ Saved: {filename}")


def plot_drawdown_comparison(ticker):
    """Plot drawdown comparison."""

    print(f"\n  Creating drawdown comparison for {ticker}...")

    df = pd.read_csv(f'results/{ticker}_trading_signals.csv')

    # Calculate drawdowns
    def calculate_drawdown(cumulative_returns):
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / (running_max + 1)
        return drawdown * 100

    buy_hold_dd = calculate_drawdown(df['cumulative_return'].values)
    strategy_dd = calculate_drawdown(df['cumulative_strategy'].values)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot drawdowns
    ax.fill_between(df.index, 0, buy_hold_dd, alpha=0.3, color='tab:blue', label='Buy & Hold')
    ax.fill_between(df.index, 0, strategy_dd, alpha=0.3, color='tab:green', label='Strategy')
    ax.plot(df.index, buy_hold_dd, linewidth=1.5, color='tab:blue')
    ax.plot(df.index, strategy_dd, linewidth=1.5, color='tab:green')

    # Formatting
    ax.set_xlabel('Trading Days', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_title(f'{ticker}: Drawdown Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Annotate max drawdowns
    max_bh_dd = buy_hold_dd.min()
    max_st_dd = strategy_dd.min()
    ax.text(0.02, 0.98, f'Buy & Hold Max DD: {max_bh_dd:.2f}%\nStrategy Max DD: {max_st_dd:.2f}%',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Save
    filename = f'plots/{ticker}_drawdown_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ✅ Saved: {filename}")


def plot_returns_distribution(ticker):
    """Plot distribution of daily returns."""

    print(f"\n  Creating returns distribution for {ticker}...")

    df = pd.read_csv(f'results/{ticker}_trading_signals.csv')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Buy & Hold returns
    ax1.hist(df['return'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=df['return'].mean(), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {df["return"].mean():.2f}%')
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Daily Return (%)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Buy & Hold Returns Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Strategy returns (only when in market)
    strategy_returns = df[df['position'] == 1]['strategy_return']
    ax2.hist(strategy_returns, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    ax2.axvline(x=strategy_returns.mean(), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {strategy_returns.mean():.2f}%')
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Daily Return (%)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Strategy Returns Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'{ticker}: Returns Distribution Comparison', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()

    # Save
    filename = f'plots/{ticker}_returns_distribution.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ✅ Saved: {filename}")


def plot_sentiment_returns_scatter(ticker):
    """Scatter plot: Sentiment vs next-day returns."""

    print(f"\n  Creating sentiment-returns correlation for {ticker}...")

    df = pd.read_csv(f'results/{ticker}_merge_prices_news.csv')

    # Shift returns to align with previous day's sentiment
    df['next_return'] = df['return'].shift(-1)

    # Remove last row (no next return)
    df = df[:-1]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot
    scatter = ax.scatter(df['sentiment_score'], df['next_return'],
                         alpha=0.6, s=50, c=df['sentiment_score'],
                         cmap='RdYlGn', edgecolors='black', linewidth=0.5)

    # Trend line
    z = np.polyfit(df['sentiment_score'], df['next_return'], 1)
    p = np.poly1d(z)
    ax.plot(df['sentiment_score'], p(df['sentiment_score']),
            "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')

    # Zero lines
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

    # Correlation
    corr = df['sentiment_score'].corr(df['next_return'])
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Formatting
    ax.set_xlabel('Sentiment Score (Today)', fontsize=12)
    ax.set_ylabel('Return (Next Day) %', fontsize=12)
    ax.set_title(f'{ticker}: Sentiment vs Next-Day Returns', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Sentiment Score')

    # Save
    filename = f'plots/{ticker}_sentiment_returns_scatter.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ✅ Saved: {filename}")


def create_comparison_dashboard():
    """Create side-by-side comparison: TAN vs SPY."""

    print(f"\n  Creating TAN vs SPY comparison dashboard...")

    tan_df = pd.read_csv('results/TAN_trading_signals.csv')
    spy_df = pd.read_csv('results/SPY_trading_signals.csv')

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plot 1: Cumulative returns comparison
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

    # Plot 2: Volatility comparison (rolling)
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

    # Plot 3: Sentiment comparison
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

    # Plot 4: Performance metrics bar chart
    ax = axes[1, 1]
    metrics_df = pd.read_csv('results/performance_metrics_detailed.csv')

    tan_metrics = metrics_df[metrics_df['ticker'] == 'TAN'].iloc[0]
    spy_metrics = metrics_df[metrics_df['ticker'] == 'SPY'].iloc[0]

    categories = ['Return\n(%)', 'Sharpe\nRatio', 'Max DD\n(%)', 'Win Rate\n(%)']
    tan_values = [
        tan_metrics['strategy_total_return'],
        tan_metrics['strategy_sharpe'] * 10,  # Scale for visibility
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

    plt.suptitle('TAN vs SPY: Comprehensive Comparison', fontsize=16, fontweight='bold', y=0.995)
    fig.tight_layout()

    # Save
    filename = 'plots/TAN_vs_SPY_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ✅ Saved: {filename}")


# ============================================
# MAIN FUNCTION
# ============================================

def main():
    """Generate all visualizations."""

    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    print("\nGenerating publication-quality charts for dissertation...")

    # Generate charts for TAN
    print("\n📊 TAN Charts:")
    plot_price_sentiment_timeseries('TAN')
    plot_cumulative_returns('TAN')
    plot_sentiment_distribution('TAN')
    plot_trading_signals('TAN')
    plot_drawdown_comparison('TAN')
    plot_returns_distribution('TAN')
    plot_sentiment_returns_scatter('TAN')

    # Generate charts for SPY
    print("\n📊 SPY Charts:")
    plot_price_sentiment_timeseries('SPY')
    plot_cumulative_returns('SPY')
    plot_sentiment_distribution('SPY')
    plot_trading_signals('SPY')
    plot_drawdown_comparison('SPY')
    plot_returns_distribution('SPY')
    plot_sentiment_returns_scatter('SPY')

    # Comparison dashboard
    print("\n📊 Comparison Charts:")
    create_comparison_dashboard()

    # Summary
    print("\n" + "=" * 60)
    print("✅ ALL VISUALIZATIONS CREATED!")
    print("=" * 60)

    print("\n📁 Charts saved in: plots/")
    print("\nChart types generated (for each ticker):")
    print("  1. Price & Sentiment Time Series")
    print("  2. Cumulative Returns (Strategy vs Buy-Hold)")
    print("  3. Sentiment Distribution")
    print("  4. Trading Signals Overlay")
    print("  5. Drawdown Comparison")
    print("  6. Returns Distribution")
    print("  7. Sentiment-Returns Correlation")
    print("\nPlus:")
    print("  8. TAN vs SPY Comparison Dashboard")

    print("\n📊 Total charts created: 15")
    print("\n✅ All charts are high-resolution (300 DPI)")
    print("✅ Ready for insertion into dissertation")

    print("\n" + "=" * 60)
    print("Next step: Run 9_calculate_volatility.py (GARCH)")
    print("=" * 60)


if __name__ == "__main__":
    main()