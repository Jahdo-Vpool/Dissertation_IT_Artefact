"""
Step 5: Generate Trading Signals
Create buy/sell/hold signals based on sentiment

What this does:
- Reads merged price and sentiment data
- Creates trading signals (buy/sell/hold)
- Calculates strategy returns
- Saves results

Run: python 5_generate_signals.py
"""

import pandas as pd
import numpy as np

# ============================================
# TRADING STRATEGY CONFIGURATION
# ============================================

# Sentiment thresholds for signals
BUY_THRESHOLD = 0.05  # Buy if sentiment > 0.05
SELL_THRESHOLD = -0.05  # Sell if sentiment < -0.05


# Between -0.05 and 0.05 = HOLD (neutral)

# ============================================
# MAIN CODE
# ============================================

def generate_signals(ticker):
    """Generate trading signals based on sentiment."""

    print(f"\n{'=' * 60}")
    print(f"{ticker} - TRADING SIGNALS")
    print(f"{'=' * 60}")

    # Read merged data
    df = pd.read_csv(f'results/{ticker}_merge_prices_news.csv')
    print(f"  ✓ Loaded {len(df)} trading days")
    print(f"    Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"    Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")

    # ============================================
    # GENERATE SIGNALS
    # ============================================

    signals = []
    positions = []  # 1 = long, 0 = out of market, -1 = short (if allowed)

    for idx, row in df.iterrows():
        sentiment = row['sentiment_score']

        # Generate signal based on sentiment
        if sentiment >= BUY_THRESHOLD:
            signal = 'BUY'
            position = 1  # Long position
        elif sentiment <= SELL_THRESHOLD:
            signal = 'SELL'
            position = 0  # Out of market (or -1 for short)
        else:
            signal = 'HOLD'
            position = 1 if idx > 0 and positions[idx - 1] == 1 else 0  # Maintain previous

        signals.append(signal)
        positions.append(position)

    df['signal'] = signals
    df['position'] = positions

    # ============================================
    # CALCULATE STRATEGY RETURNS
    # ============================================

    # Strategy return = return * position (only earn return when in market)
    df['strategy_return'] = df['return'] * df['position']

    # Cumulative returns
    df['cumulative_return'] = (1 + df['return'] / 100).cumprod() - 1
    df['cumulative_strategy'] = (1 + df['strategy_return'] / 100).cumprod() - 1

    # ============================================
    # STATISTICS
    # ============================================

    # Signal counts
    buy_count = sum(df['signal'] == 'BUY')
    sell_count = sum(df['signal'] == 'SELL')
    hold_count = sum(df['signal'] == 'HOLD')

    # Time in market
    days_in_market = sum(df['position'] == 1)
    days_out = sum(df['position'] == 0)

    # Performance
    total_return = df['cumulative_return'].iloc[-1] * 100
    strategy_return = df['cumulative_strategy'].iloc[-1] * 100

    # Win rate (days when strategy return > 0 while in market)
    in_market_returns = df[df['position'] == 1]['strategy_return']
    win_rate = (in_market_returns > 0).sum() / len(in_market_returns) * 100 if len(in_market_returns) > 0 else 0

    print(f"\n  Trading Signals:")
    print(f"    BUY:  {buy_count} ({buy_count / len(df) * 100:.1f}%)")
    print(f"    SELL: {sell_count} ({sell_count / len(df) * 100:.1f}%)")
    print(f"    HOLD: {hold_count} ({hold_count / len(df) * 100:.1f}%)")

    print(f"\n  Position:")
    print(f"    Days in market: {days_in_market} ({days_in_market / len(df) * 100:.1f}%)")
    print(f"    Days out: {days_out} ({days_out / len(df) * 100:.1f}%)")

    print(f"\n  Performance:")
    print(f"    Buy & Hold return: {total_return:+.2f}%")
    print(f"    Strategy return:   {strategy_return:+.2f}%")
    print(f"    Difference:        {strategy_return - total_return:+.2f}%")
    print(f"    Win rate:          {win_rate:.1f}%")

    # ============================================
    # VOLATILITY METRICS
    # ============================================

    # Annualized volatility (assuming 252 trading days/year)
    buy_hold_vol = df['return'].std() * np.sqrt(252)
    strategy_vol = df['strategy_return'].std() * np.sqrt(252)

    # Sharpe ratio (assuming 0% risk-free rate for simplicity)
    buy_hold_sharpe = (df['return'].mean() * 252) / buy_hold_vol if buy_hold_vol > 0 else 0
    strategy_sharpe = (df['strategy_return'].mean() * 252) / strategy_vol if strategy_vol > 0 else 0

    print(f"\n  Risk Metrics:")
    print(f"    Buy & Hold volatility: {buy_hold_vol:.2f}%")
    print(f"    Strategy volatility:   {strategy_vol:.2f}%")
    print(f"    Buy & Hold Sharpe:     {buy_hold_sharpe:.3f}")
    print(f"    Strategy Sharpe:       {strategy_sharpe:.3f}")

    return df


def main():
    """Main function."""

    print("\n" + "=" * 60)
    print("SENTIMENT-BASED TRADING STRATEGY")
    print("=" * 60)
    print(f"\nStrategy Rules:")
    print(f"  • Sentiment > {BUY_THRESHOLD:+.2f}  → BUY  (go long)")
    print(f"  • Sentiment < {SELL_THRESHOLD:+.2f} → SELL (exit position)")
    print(f"  • Otherwise        → HOLD (maintain position)")

    # Generate signals for TAN
    tan_signals = generate_signals('TAN')

    if tan_signals is not None:
        output_file = 'results/TAN_trading_signals.csv'
        tan_signals.to_csv(output_file, index=False)
        print(f"\n  Saved: {output_file}")

    # Generate signals for SPY
    spy_signals = generate_signals('SPY')

    if spy_signals is not None:
        output_file = 'results/SPY_trading_signals.csv'
        spy_signals.to_csv(output_file, index=False)
        print(f"\n Saved: {output_file}")

    # ============================================
    # COMPARISON SUMMARY
    # ============================================

    print("\n" + "=" * 60)
    print("TRADING SIGNALS GENERATED!")
    print("=" * 60)

    if tan_signals is not None and spy_signals is not None:
        # Compare TAN vs SPY
        tan_final = tan_signals['cumulative_strategy'].iloc[-1] * 100
        spy_final = spy_signals['cumulative_strategy'].iloc[-1] * 100

        print(f"\nStrategy Performance Comparison:")
        print(f"\nTAN (Solar Energy ETF):")
        print(f"  Strategy Return: {tan_final:+.2f}%")
        print(f"  Days analyzed:   {len(tan_signals)}")

        print(f"\nSPY (S&P 500 ETF):")
        print(f"  Strategy Return: {spy_final:+.2f}%")
        print(f"  Days analyzed:   {len(spy_signals)}")

        if tan_final > spy_final:
            print(f"\nTAN outperformed SPY by {tan_final - spy_final:+.2f}%")
        else:
            print(f"\nSPY outperformed TAN by {spy_final - tan_final:+.2f}%")

    print("\n" + "=" * 60)
    print("OUTPUT FILES:")
    print("=" * 60)
    print("  TAN_trading_signals.csv")
    print("  SPY_trading_signals.csv")

    print("\nColumns added:")
    print("  • signal              - BUY/SELL/HOLD")
    print("  • position            - 1 (in market) or 0 (out)")
    print("  • strategy_return     - Daily strategy return")
    print("  • cumulative_return   - Buy & hold cumulative return")
    print("  • cumulative_strategy - Strategy cumulative return")

    print("\n" + "=" * 60)
    print("Next step: Run 6_calculate_performance.py")
    print("=" * 60)


if __name__ == "__main__":
    main()