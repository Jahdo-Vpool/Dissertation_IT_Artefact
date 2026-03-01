"""
================================================================================
Script 5: Sentiment-Based Trading Signal Generation
================================================================================
Purpose:
    Reads the merged price and sentiment dataset produced by Script 4
    (merge_prices_news.py) and applies a rules-based decision framework to
    generate a trading signal for each day. The signal determines whether the
    strategy holds a long position (BUY/HOLD) or exits the market (SELL).
    Strategy returns are then calculated and compared against a passive
    buy-and-hold benchmark.

Academic Context:
    This script is the core implementation of Dissertation Objective 4 —
    designing and backtesting a rules-based trading strategy informed by
    sentiment signals. It also directly supports Hypothesis 3 (H3), which
    proposes that a sentiment-driven strategy can generate superior
    risk-adjusted returns compared to a buy-and-hold approach.

    The strategy is intentionally kept simple and rule-based. This is a
    deliberate design choice: a transparent, interpretable strategy is easier
    to evaluate academically, avoids the risk of overfitting that more complex
    models carry, and allows the role of sentiment — rather than model
    complexity — to be the primary driver of any performance difference.

Signal Logic:
    Three signal states are defined based on the daily sentiment score:
        - BUY  : sentiment score >= +0.05  (positive market sentiment)
        - SELL : sentiment score <= -0.05  (negative market sentiment)
        - HOLD : sentiment score between -0.05 and +0.05 (neutral, no action)

    The thresholds of +/-0.05 were chosen to filter out noise in near-neutral
    sentiment readings, ensuring only meaningfully positive or negative days
    trigger a position change. This is consistent with threshold-based signal
    generation approaches used in the computational finance literature.

Inputs:
    - results/TAN_merge_prices_news.csv  (from Script 4)
    - results/SPY_merge_prices_news.csv  (from Script 4)

Outputs:
    - results/TAN_trading_signals.csv  : Signal and return data for TAN
    - results/SPY_trading_signals.csv  : Signal and return data for SPY

Pipeline Position:
    Script 5 of 10. Receives merged data from Script 4. Output feeds into
    Script 6 (calculate_metrics.py) for full performance evaluation.

Dependencies:
    - pandas  : Data loading, manipulation, and CSV export
    - numpy   : Volatility and Sharpe ratio calculations

Usage:
    python trading_signals.py
================================================================================
"""

import pandas as pd
import numpy as np


# ============================================
# SECTION 1: STRATEGY CONFIGURATION
# ============================================
# These thresholds define the sensitivity of the trading strategy to sentiment.
# A score above BUY_THRESHOLD is interpreted as a sufficiently positive signal
# to enter or maintain a long position. A score below SELL_THRESHOLD is
# interpreted as a sufficiently negative signal to exit the market entirely.
#
# The neutral zone between -0.05 and +0.05 acts as a buffer to prevent
# excessive trading on days where sentiment is ambiguous. In a live trading
# context, frequent position switching would incur transaction costs; the
# neutral buffer helps mitigate this even in a simulated backtest.

BUY_THRESHOLD = 0.05    # Enter long if sentiment exceeds this value
SELL_THRESHOLD = -0.05  # Exit market if sentiment falls below this value
# Sentiment between -0.05 and +0.05 → HOLD (no change to current position)


# ============================================
# SECTION 2: SIGNAL GENERATION FUNCTION
# ============================================

def generate_signals(ticker):
    """
    Apply the sentiment-based trading rules to a single ETF's merged dataset,
    calculate daily and cumulative returns for both the strategy and a
    buy-and-hold benchmark, and report summary performance statistics.

    The function processes each trading day sequentially. This is necessary
    because the HOLD logic is stateful — whether a HOLD day results in a long
    position (position = 1) or no position (position = 0) depends on what
    the position was on the previous day. A purely vectorised approach cannot
    capture this path-dependency without additional logic.

    Parameters
    ----------
    ticker : str
        The ETF ticker symbol ('TAN' or 'SPY'). Used to locate the correct
        input file and label outputs.

    Returns
    -------
    pandas.DataFrame or None
        The input DataFrame extended with the following columns:
            signal              - String label: 'BUY', 'SELL', or 'HOLD'
            position            - Integer: 1 (in market) or 0 (out of market)
            strategy_return     - Daily return earned by the strategy
            cumulative_return   - Compounded buy-and-hold return to date
            cumulative_strategy - Compounded strategy return to date
        Returns None if the input file cannot be found or is empty.
    """

    print(f"\n{'=' * 60}")
    print(f"{ticker} - TRADING SIGNALS")
    print(f"{'=' * 60}")

    # Load the merged dataset produced by Script 4. This file contains one
    # row per trading day, with columns for closing price, daily return,
    # and the aggregated sentiment score for that date.
    df = pd.read_csv(f'results/{ticker}_merge_prices_news.csv')
    print(f"  Loaded {len(df)} trading days")
    print(f"    Date range:  {df['date'].min()} to {df['date'].max()}")
    print(f"    Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")


    # -----------------------------------------------------------------------
    # SIGNAL GENERATION LOOP
    # -----------------------------------------------------------------------
    # Each day is evaluated independently against the sentiment thresholds.
    # The HOLD case introduces path-dependency: if the previous position was
    # long (1) and today's sentiment is neutral, the position is maintained
    # at 1. If the strategy was already out of the market (0), a neutral
    # sentiment day does not trigger re-entry — only a BUY signal does.
    # This reflects a conservative approach where the default state in
    # uncertain conditions is to preserve the existing position.

    signals = []
    positions = []  # Tracks daily position: 1 = in market, 0 = out of market

    for idx, row in df.iterrows():
        sentiment = row['sentiment_score']

        if sentiment >= BUY_THRESHOLD:
            # Positive sentiment: enter or remain in a long position.
            signal = 'BUY'
            position = 1

        elif sentiment <= SELL_THRESHOLD:
            # Negative sentiment: exit the market to avoid anticipated losses.
            # Note: this implementation does not take short positions (position = -1).
            # The strategy is long-only, which reflects a realistic constraint
            # for retail investors who may not have access to short-selling.
            signal = 'SELL'
            position = 0

        else:
            # Neutral sentiment: maintain the previous day's position.
            # On the very first day (idx == 0), there is no previous position,
            # so the default is to remain out of the market (position = 0).
            signal = 'HOLD'
            position = 1 if idx > 0 and positions[idx - 1] == 1 else 0

        signals.append(signal)
        positions.append(position)

    df['signal'] = signals
    df['position'] = positions


    # -----------------------------------------------------------------------
    # RETURN CALCULATION
    # -----------------------------------------------------------------------
    # Strategy return on any given day is the market return multiplied by
    # the position flag. When position = 1, the strategy earns the full
    # market return. When position = 0, the strategy earns nothing (cash).
    # This assumes no transaction costs or slippage, which is a standard
    # simplifying assumption in academic backtesting studies.

    df['strategy_return'] = df['return'] * df['position']

    # Cumulative returns are computed using compounding (chain multiplication
    # of gross returns). Dividing by 100 converts percentage returns to
    # decimals before compounding, then subtracting 1 returns the net gain
    # as a decimal (e.g. 0.15 = 15% total return).
    df['cumulative_return'] = (1 + df['return'] / 100).cumprod() - 1
    df['cumulative_strategy'] = (1 + df['strategy_return'] / 100).cumprod() - 1


    # -----------------------------------------------------------------------
    # SUMMARY STATISTICS
    # -----------------------------------------------------------------------

    # Signal distribution: shows how frequently each signal type was triggered.
    # A heavily skewed distribution (e.g. mostly BUY) might suggest the
    # thresholds need adjustment or that the sentiment data is biased.
    buy_count = sum(df['signal'] == 'BUY')
    sell_count = sum(df['signal'] == 'SELL')
    hold_count = sum(df['signal'] == 'HOLD')

    # Market exposure: proportion of days the strategy held a long position.
    # Lower exposure typically implies lower returns but also lower risk.
    days_in_market = sum(df['position'] == 1)
    days_out = sum(df['position'] == 0)

    # Final cumulative returns as percentages for display.
    total_return = df['cumulative_return'].iloc[-1] * 100
    strategy_return = df['cumulative_strategy'].iloc[-1] * 100

    # Win rate: the percentage of days in market where the strategy earned
    # a positive return. This is a simple measure of signal quality —
    # a win rate significantly above 50% suggests the sentiment signal has
    # genuine predictive value for next-day price direction.
    in_market_returns = df[df['position'] == 1]['strategy_return']
    win_rate = (in_market_returns > 0).sum() / len(in_market_returns) * 100 \
        if len(in_market_returns) > 0 else 0

    print(f"\n  Signal Distribution:")
    print(f"    BUY:  {buy_count} days ({buy_count / len(df) * 100:.1f}%)")
    print(f"    SELL: {sell_count} days ({sell_count / len(df) * 100:.1f}%)")
    print(f"    HOLD: {hold_count} days ({hold_count / len(df) * 100:.1f}%)")

    print(f"\n  Market Exposure:")
    print(f"    Days in market: {days_in_market} ({days_in_market / len(df) * 100:.1f}%)")
    print(f"    Days in cash:   {days_out} ({days_out / len(df) * 100:.1f}%)")

    print(f"\n  Return Comparison:")
    print(f"    Buy and Hold:  {total_return:+.2f}%")
    print(f"    Strategy:      {strategy_return:+.2f}%")
    print(f"    Difference:    {strategy_return - total_return:+.2f}%")
    print(f"    Win rate:      {win_rate:.1f}%")


    # -----------------------------------------------------------------------
    # RISK METRICS
    # -----------------------------------------------------------------------
    # Annualised volatility is the standard deviation of daily returns scaled
    # by sqrt(252), where 252 is the conventional number of trading days in
    # a calendar year. This allows volatility to be expressed on a comparable
    # annual basis regardless of the length of the analysis period.
    #
    # The Sharpe ratio measures risk-adjusted return: how much return is earned
    # per unit of volatility. A higher Sharpe ratio indicates more efficient
    # use of risk. A risk-free rate of 0% is assumed here for simplicity,
    # which is a common convention in short-horizon academic backtesting.
    # This is noted as a limitation — using a non-zero risk-free rate
    # (e.g. US Treasury rate) would produce more conservative Sharpe values.

    buy_hold_vol = df['return'].std() * np.sqrt(252)
    strategy_vol = df['strategy_return'].std() * np.sqrt(252)

    buy_hold_sharpe = (df['return'].mean() * 252) / buy_hold_vol \
        if buy_hold_vol > 0 else 0
    strategy_sharpe = (df['strategy_return'].mean() * 252) / strategy_vol \
        if strategy_vol > 0 else 0

    print(f"\n  Risk-Adjusted Metrics:")
    print(f"    Buy and Hold volatility: {buy_hold_vol:.2f}%")
    print(f"    Strategy volatility:     {strategy_vol:.2f}%")
    print(f"    Buy and Hold Sharpe:     {buy_hold_sharpe:.3f}")
    print(f"    Strategy Sharpe:         {strategy_sharpe:.3f}")

    return df


# ============================================
# SECTION 3: MAIN EXECUTION
# ============================================

def main():
    """
    Run signal generation for both TAN and SPY, save results to CSV,
    and print a side-by-side performance comparison.

    Running both ETFs through the same strategy logic is essential for
    the comparative analysis in Hypothesis 1 (H1) and Hypothesis 3 (H3).
    H1 examines whether sentiment has a stronger effect on TAN than SPY.
    H3 evaluates whether the strategy outperforms buy-and-hold for either
    or both ETFs. Applying identical thresholds to both ensures the
    comparison is fair and any performance difference is attributable to
    the underlying sentiment-return relationship rather than different
    strategy configurations.
    """

    print("\n" + "=" * 60)
    print("SENTIMENT-BASED TRADING STRATEGY")
    print("=" * 60)
    print(f"\nStrategy Rules:")
    print(f"  Sentiment > {BUY_THRESHOLD:+.2f}  : BUY  (enter long position)")
    print(f"  Sentiment < {SELL_THRESHOLD:+.2f} : SELL (exit to cash)")
    print(f"  Otherwise        : HOLD (maintain current position)")

    # -----------------------------------------------------------------------
    # PROCESS TAN
    # -----------------------------------------------------------------------
    tan_signals = generate_signals('TAN')

    if tan_signals is not None:
        output_file = 'results/TAN_trading_signals.csv'
        tan_signals.to_csv(output_file, index=False)
        print(f"\n  Saved: {output_file}")

    # -----------------------------------------------------------------------
    # PROCESS SPY
    # -----------------------------------------------------------------------
    spy_signals = generate_signals('SPY')

    if spy_signals is not None:
        output_file = 'results/SPY_trading_signals.csv'
        spy_signals.to_csv(output_file, index=False)
        print(f"\n  Saved: {output_file}")


    # -----------------------------------------------------------------------
    # COMPARATIVE SUMMARY
    # -----------------------------------------------------------------------
    # Direct comparison of strategy returns across both ETFs provides the
    # first indication of whether sentiment-based signals perform differently
    # for a sector ETF (TAN) versus a broad market ETF (SPY), which is the
    # central question in H1 and H3.

    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON: TAN vs SPY")
    print("=" * 60)

    if tan_signals is not None and spy_signals is not None:
        tan_final = tan_signals['cumulative_strategy'].iloc[-1] * 100
        spy_final = spy_signals['cumulative_strategy'].iloc[-1] * 100

        print(f"\nTAN (Invesco Solar ETF):")
        print(f"  Strategy Return: {tan_final:+.2f}%")
        print(f"  Days analysed:   {len(tan_signals)}")

        print(f"\nSPY (S&P 500 ETF):")
        print(f"  Strategy Return: {spy_final:+.2f}%")
        print(f"  Days analysed:   {len(spy_signals)}")

        if tan_final > spy_final:
            print(f"\nTAN strategy outperformed SPY by {tan_final - spy_final:.2f}%")
        else:
            print(f"\nSPY strategy outperformed TAN by {spy_final - tan_final:.2f}%")

    print("\n" + "=" * 60)
    print("OUTPUT FILES:")
    print("  results/TAN_trading_signals.csv")
    print("  results/SPY_trading_signals.csv")
    print("\nColumns added:")
    print("  signal              - BUY / SELL / HOLD")
    print("  position            - 1 (in market) or 0 (out of market)")
    print("  strategy_return     - Daily return earned by the strategy")
    print("  cumulative_return   - Compounded buy-and-hold return")
    print("  cumulative_strategy - Compounded strategy return")
    print("\nNext step: Run calculate_metrics.py")
    print("=" * 60)


# ============================================
# SECTION 4: ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()