"""
================================================================================
Script 6: Trading Strategy Performance Evaluation (calculate_metrics.py)
================================================================================
Purpose:
    Evaluates the performance of the sentiment-based trading strategy developed
    in Script 5 (trading_signals.py) against a passive buy-and-hold benchmark.
    The script computes a comprehensive set of return, risk, and risk-adjusted
    metrics for both TAN and SPY, then produces a comparative summary table
    suitable for direct inclusion in the dissertation results chapter.

Academic Context:
    This script is the primary vehicle for testing Hypothesis 3 (H3):
    "A sentiment-informed trading strategy generates superior risk-adjusted
    returns compared to a passive buy-and-hold approach."

    Evaluating a trading strategy purely on total return is insufficient in
    academic finance research. A strategy may achieve higher returns while
    simultaneously exposing the investor to significantly greater risk. For
    this reason, the evaluation framework here includes multiple risk-adjusted
    metrics (Sharpe, Sortino, Calmar), each of which captures a different
    dimension of the risk-return relationship. Together they provide a more
    complete and academically defensible assessment of strategy performance.

    The comparative structure — strategy vs. buy-and-hold, TAN vs. SPY —
    also contributes to the evaluation of H1, by revealing whether the
    sentiment signal adds more value in the sector-specific ETF (TAN) than
    in the broad market ETF (SPY).

Inputs:
    - results/TAN_trading_signals.csv  : Output of Script 5 for TAN
    - results/SPY_trading_signals.csv  : Output of Script 5 for SPY

Outputs:
    - results/performance_metrics_detailed.csv : Full metric set for both ETFs
    - results/performance_summary.csv          : Formatted comparison table

Pipeline Position:
    Script 6 of 10. Depends on Script 5 (trading_signals.py).
    Output feeds into Script 7 (charts.py) for visualisation and
    Script 11 (hypothesis_results.py) for formal hypothesis reporting.

Dependencies:
    - pandas : Data loading and DataFrame construction
    - numpy  : Numerical calculations (volatility, drawdown, ratios)

Usage:
    python calculate_metrics.py
================================================================================
"""

import pandas as pd
import numpy as np


# ============================================
# SECTION 1: METRIC CALCULATION FUNCTIONS
# ============================================
# Each function below calculates one performance metric. Separating these
# into standalone functions rather than embedding them in the main analysis
# loop serves two purposes: it makes each formula transparent and testable
# in isolation, and it allows the same functions to be reused for both
# TAN and SPY without code duplication.

def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Calculate the annualised Sharpe ratio for a return series.

    The Sharpe ratio measures the average excess return earned per unit of
    total risk (standard deviation). It is the most widely used metric for
    comparing risk-adjusted performance in both academic and practitioner
    finance. A higher Sharpe ratio indicates better risk-adjusted performance.

    Formula: Sharpe = (mean(excess returns) * T) / (std(returns) * sqrt(T))
    Where T = number of trading periods per year (252 for daily data).

    The risk-free rate is set to 0.0 as a simplifying assumption. This is
    standard practice in academic strategy backtests where the focus is on
    the relative comparison between strategies rather than absolute excess
    return over Treasury yields.

    Parameters
    ----------
    returns       : array-like, daily return series (as decimals, not percentages)
    risk_free_rate: float, daily risk-free rate (default 0.0)
    periods_per_year: int, trading days per year for annualisation (default 252)

    Returns
    -------
    float : Annualised Sharpe ratio. Returns 0.0 if input is empty or has
            zero standard deviation (i.e. constant returns).
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    excess_returns = returns - risk_free_rate
    sharpe = (excess_returns.mean() * periods_per_year) / (returns.std() * np.sqrt(periods_per_year))
    return sharpe


def calculate_max_drawdown(cumulative_returns):
    """
    Calculate the maximum drawdown from a cumulative return series.

    Maximum drawdown (MDD) measures the largest peak-to-trough decline in
    portfolio value over the entire evaluation period. It is a critical risk
    metric because it reflects the worst-case loss an investor would have
    experienced if they had entered at the peak and exited at the trough.
    Unlike standard deviation, MDD captures the sequential, path-dependent
    nature of losses, making it particularly relevant for evaluating
    trading strategies.

    The calculation proceeds in two steps:
    1. Compute the running maximum of cumulative returns at each point in time.
    2. Compute the drawdown at each point as the percentage decline from
       the running maximum.

    The +1 adjustment in the denominator handles the case where cumulative
    returns start at 0 (i.e. no initial gain), preventing a division-by-zero
    error at the start of the series.

    Parameters
    ----------
    cumulative_returns : array-like, cumulative return series (starting from 0)

    Returns
    -------
    float : Maximum drawdown as a negative decimal (e.g. -0.25 = -25% drawdown).
            Returns 0.0 if input is empty.
    """
    if len(cumulative_returns) == 0:
        return 0.0

    # Step 1: Running maximum — the highest cumulative return seen so far
    running_max = np.maximum.accumulate(cumulative_returns)

    # Step 2: Drawdown at each point relative to the prior peak
    drawdown = (cumulative_returns - running_max) / (running_max + 1)

    max_dd = drawdown.min()
    return max_dd


def calculate_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Calculate the annualised Sortino ratio for a return series.

    The Sortino ratio is a refinement of the Sharpe ratio. Where the Sharpe
    ratio penalises both upside and downside volatility equally, the Sortino
    ratio penalises only downside volatility (returns below zero). This makes
    it a more appropriate metric when the return distribution is asymmetric,
    as is common in sentiment-driven strategies that may have periods of
    inactivity with zero returns.

    A strategy that avoids large losses but occasionally misses upside moves
    will score better on the Sortino than on the Sharpe ratio, which is
    consistent with the risk management objective of the sentiment strategy
    tested here.

    Parameters
    ----------
    returns       : array-like, daily return series (as decimals)
    risk_free_rate: float, daily risk-free rate (default 0.0)
    periods_per_year: int, trading days per year (default 252)

    Returns
    -------
    float : Annualised Sortino ratio. Returns 0.0 if no negative returns exist
            or if the downside deviation is zero.
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate

    # Isolate days where returns were negative (the downside)
    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0

    # Annualise the downside standard deviation
    downside_std = downside_returns.std() * np.sqrt(periods_per_year)
    sortino = (excess_returns.mean() * periods_per_year) / downside_std
    return sortino


def calculate_calmar_ratio(total_return, max_drawdown, years):
    """
    Calculate the Calmar ratio (annualised return divided by maximum drawdown).

    The Calmar ratio relates the annualised return of a strategy directly to
    its worst observed loss. It is particularly useful for comparing strategies
    over different time periods because it normalises return by the most extreme
    risk event rather than average volatility. A higher Calmar ratio indicates
    that the strategy generates more return per unit of worst-case drawdown.

    Parameters
    ----------
    total_return  : float, total return over the full period (as a decimal)
    max_drawdown  : float, maximum drawdown (as a negative decimal)
    years         : float, length of the evaluation period in years

    Returns
    -------
    float : Calmar ratio. Returns 0.0 if max drawdown is zero.
    """
    if max_drawdown == 0:
        return 0.0

    annualized_return = total_return / years
    calmar = annualized_return / abs(max_drawdown)
    return calmar


# ============================================
# SECTION 2: MAIN ANALYSIS FUNCTION
# ============================================

def analyze_performance(ticker):
    """
    Run the full performance evaluation for a single ETF ticker.

    This function loads the trading signal data produced by Script 5,
    computes all return, risk, and risk-adjusted metrics for both the
    sentiment-based strategy and the buy-and-hold benchmark, and prints
    a structured summary to the console.

    The function is designed to be called independently for TAN and SPY
    so that results can be directly compared in the summary table produced
    by create_summary_table().

    Parameters
    ----------
    ticker : str
        ETF ticker symbol ('TAN' or 'SPY'). Used to locate the correct
        input file and label the output metrics.

    Returns
    -------
    dict : A dictionary of all computed metrics, keyed by descriptive names.
           This dictionary is passed to create_summary_table() and saved
           to CSV for use in the dissertation results chapter.
    """

    print(f"\n{'=' * 60}")
    print(f"{ticker} - PERFORMANCE ANALYSIS")
    print(f"{'=' * 60}")

    # Load the trading signals file produced by Script 5.
    # This file contains daily returns, cumulative returns, position flags,
    # and strategy returns for the full evaluation period.
    df = pd.read_csv(f'results/{ticker}_trading_signals.csv')
    print(f"  Loaded {len(df)} trading days")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    # Convert the number of trading days to years for annualisation.
    # 252 is the standard assumption for the number of trading days in a
    # calendar year on US equity markets.
    days = len(df)
    years = days / 252

    # -----------------------------------------------------------------------
    # RETURN METRICS
    # The final value of the cumulative return series represents the total
    # percentage gain or loss over the full period. Multiplying by 100
    # converts from decimal to percentage for reporting purposes.
    # -----------------------------------------------------------------------

    buy_hold_total = df['cumulative_return'].iloc[-1] * 100
    strategy_total = df['cumulative_strategy'].iloc[-1] * 100

    # Annualised return is calculated using the compound annual growth rate
    # (CAGR) formula: (1 + total_return)^(1/years) - 1. This allows returns
    # over periods of different lengths to be compared on a consistent basis.
    buy_hold_annualized = ((1 + df['cumulative_return'].iloc[-1]) ** (1 / years) - 1) * 100
    strategy_annualized = ((1 + df['cumulative_strategy'].iloc[-1]) ** (1 / years) - 1) * 100

    buy_hold_avg_daily = df['return'].mean()
    strategy_avg_daily = df['strategy_return'].mean()

    print(f"\n  RETURN METRICS:")
    print(f"    Period: {years:.2f} years ({days} trading days)")
    print(f"\n    Total Returns:")
    print(f"      Buy & Hold:  {buy_hold_total:+.2f}%")
    print(f"      Strategy:    {strategy_total:+.2f}%")
    print(f"      Difference:  {strategy_total - buy_hold_total:+.2f}%")
    print(f"\n    Annualised Returns:")
    print(f"      Buy & Hold:  {buy_hold_annualized:+.2f}%")
    print(f"      Strategy:    {strategy_annualized:+.2f}%")

    # -----------------------------------------------------------------------
    # RISK METRICS
    # Annualised volatility is calculated by multiplying the daily standard
    # deviation by sqrt(252). This scaling factor converts daily volatility
    # to an annual figure under the assumption that daily returns are
    # independently and identically distributed — a standard assumption in
    # financial modelling despite its known limitations.
    # -----------------------------------------------------------------------

    buy_hold_vol = df['return'].std() * np.sqrt(252)
    strategy_vol = df['strategy_return'].std() * np.sqrt(252)

    buy_hold_dd = calculate_max_drawdown(df['cumulative_return'].values) * 100
    strategy_dd = calculate_max_drawdown(df['cumulative_strategy'].values) * 100

    # Risk-adjusted metrics — each uses the daily return series divided by
    # 100 to convert back from percentage to decimal form, as the metric
    # functions expect decimal inputs.
    buy_hold_sharpe = calculate_sharpe_ratio(df['return'].values / 100)
    strategy_sharpe = calculate_sharpe_ratio(df['strategy_return'].values / 100)

    buy_hold_sortino = calculate_sortino_ratio(df['return'].values / 100)
    strategy_sortino = calculate_sortino_ratio(df['strategy_return'].values / 100)

    buy_hold_calmar = calculate_calmar_ratio(buy_hold_total / 100, buy_hold_dd / 100, years)
    strategy_calmar = calculate_calmar_ratio(strategy_total / 100, strategy_dd / 100, years)

    print(f"\n  RISK METRICS:")
    print(f"    Volatility (annualised):")
    print(f"      Buy & Hold:  {buy_hold_vol:.2f}%")
    print(f"      Strategy:    {strategy_vol:.2f}%")
    print(f"      Reduction:   {buy_hold_vol - strategy_vol:+.2f}%")
    print(f"\n    Maximum Drawdown:")
    print(f"      Buy & Hold:  {buy_hold_dd:.2f}%")
    print(f"      Strategy:    {strategy_dd:.2f}%")
    print(f"      Improvement: {buy_hold_dd - strategy_dd:+.2f}%")

    print(f"\n  RISK-ADJUSTED RETURNS:")
    print(f"    Sharpe Ratio:  Buy & Hold {buy_hold_sharpe:.3f} | Strategy {strategy_sharpe:.3f} | Diff {strategy_sharpe - buy_hold_sharpe:+.3f}")
    print(f"    Sortino Ratio: Buy & Hold {buy_hold_sortino:.3f} | Strategy {strategy_sortino:.3f}")
    print(f"    Calmar Ratio:  Buy & Hold {buy_hold_calmar:.3f} | Strategy {strategy_calmar:.3f}")

    # -----------------------------------------------------------------------
    # TRADING ACTIVITY
    # These metrics describe how actively the strategy trades and how often
    # it is invested in the market. A strategy that is rarely in the market
    # will naturally have lower volatility, so these figures provide important
    # context for interpreting the risk metrics above.
    # -----------------------------------------------------------------------

    # Count the number of position changes (entries and exits).
    # diff() identifies days where the position changed; subtracting 1
    # removes the initial position assignment on day one.
    position_changes = (df['position'].diff() != 0).sum() - 1

    days_in = (df['position'] == 1).sum()   # Days the strategy holds a position
    days_out = (df['position'] == 0).sum()  # Days the strategy is in cash

    # Win rate: percentage of days in market where the ETF return was positive.
    # This measures how often the strategy is invested on profitable days.
    in_market = df[df['position'] == 1]
    if len(in_market) > 0:
        winning_days = (in_market['return'] > 0).sum()
        win_rate = winning_days / len(in_market) * 100
    else:
        win_rate = 0.0

    winning_returns = in_market[in_market['return'] > 0]['return']
    losing_returns = in_market[in_market['return'] < 0]['return']
    avg_win = winning_returns.mean() if len(winning_returns) > 0 else 0
    avg_loss = losing_returns.mean() if len(losing_returns) > 0 else 0

    print(f"\n  TRADING ACTIVITY:")
    print(f"    Position changes: {position_changes}")
    print(f"    Days in market:   {days_in} ({days_in / days * 100:.1f}%)")
    print(f"    Days out:         {days_out} ({days_out / days * 100:.1f}%)")
    print(f"    Win rate:         {win_rate:.1f}%")
    print(f"    Average win:      {avg_win:+.2f}%")
    print(f"    Average loss:     {avg_loss:+.2f}%")
    if avg_loss != 0:
        print(f"    Win/Loss ratio:   {abs(avg_win / avg_loss):.2f}")

    # -----------------------------------------------------------------------
    # COMPILE METRICS DICTIONARY
    # All computed values are collected into a single dictionary so they
    # can be returned to main() and used to build the comparison table.
    # -----------------------------------------------------------------------

    metrics = {
        'ticker': ticker,
        'days': days,
        'years': years,
        'buy_hold_total_return': buy_hold_total,
        'strategy_total_return': strategy_total,
        'buy_hold_annualized_return': buy_hold_annualized,
        'strategy_annualized_return': strategy_annualized,
        'outperformance': strategy_total - buy_hold_total,
        'buy_hold_volatility': buy_hold_vol,
        'strategy_volatility': strategy_vol,
        'volatility_reduction': buy_hold_vol - strategy_vol,
        'buy_hold_max_drawdown': buy_hold_dd,
        'strategy_max_drawdown': strategy_dd,
        'drawdown_improvement': buy_hold_dd - strategy_dd,
        'buy_hold_sharpe': buy_hold_sharpe,
        'strategy_sharpe': strategy_sharpe,
        'sharpe_improvement': strategy_sharpe - buy_hold_sharpe,
        'buy_hold_sortino': buy_hold_sortino,
        'strategy_sortino': strategy_sortino,
        'buy_hold_calmar': buy_hold_calmar,
        'strategy_calmar': strategy_calmar,
        'trades': position_changes,
        'days_in_market': days_in,
        'days_out_market': days_out,
        'time_in_market_pct': days_in / days * 100,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': abs(avg_win / avg_loss) if avg_loss != 0 else 0
    }

    return metrics


# ============================================
# SECTION 3: COMPARISON TABLE FUNCTION
# ============================================

def create_summary_table(tan_metrics, spy_metrics):
    """
    Produce a formatted comparison table across all key metrics for both ETFs.

    This table is structured for direct inclusion in the dissertation results
    chapter. Each row represents one performance metric; columns represent
    the four conditions being compared: TAN strategy, TAN buy-and-hold,
    SPY strategy, and SPY buy-and-hold.

    The side-by-side format is designed to make the H3 evaluation immediately
    readable — the examiner can compare strategy vs. benchmark within each ETF,
    and then compare the differential effects across ETFs to assess H1.

    Parameters
    ----------
    tan_metrics : dict, output of analyze_performance('TAN')
    spy_metrics : dict, output of analyze_performance('SPY')

    Returns
    -------
    pandas.DataFrame : Formatted comparison table.
    """

    print("\n" + "=" * 60)
    print("COMPARATIVE PERFORMANCE SUMMARY")
    print("=" * 60)

    comparison = pd.DataFrame({
        'Metric': [
            'Total Return (%)',
            'Annualised Return (%)',
            'Volatility (%)',
            'Max Drawdown (%)',
            'Sharpe Ratio',
            'Sortino Ratio',
            'Calmar Ratio',
            'Time in Market (%)',
            'Win Rate (%)',
            'Number of Trades'
        ],
        'TAN Strategy': [
            f"{tan_metrics['strategy_total_return']:.2f}",
            f"{tan_metrics['strategy_annualized_return']:.2f}",
            f"{tan_metrics['strategy_volatility']:.2f}",
            f"{tan_metrics['strategy_max_drawdown']:.2f}",
            f"{tan_metrics['strategy_sharpe']:.3f}",
            f"{tan_metrics['strategy_sortino']:.3f}",
            f"{tan_metrics['strategy_calmar']:.3f}",
            f"{tan_metrics['time_in_market_pct']:.1f}",
            f"{tan_metrics['win_rate']:.1f}",
            f"{tan_metrics['trades']}"
        ],
        'TAN Buy & Hold': [
            f"{tan_metrics['buy_hold_total_return']:.2f}",
            f"{tan_metrics['buy_hold_annualized_return']:.2f}",
            f"{tan_metrics['buy_hold_volatility']:.2f}",
            f"{tan_metrics['buy_hold_max_drawdown']:.2f}",
            f"{tan_metrics['buy_hold_sharpe']:.3f}",
            f"{tan_metrics['buy_hold_sortino']:.3f}",
            f"{tan_metrics['buy_hold_calmar']:.3f}",
            "100.0", "-", "0"
        ],
        'SPY Strategy': [
            f"{spy_metrics['strategy_total_return']:.2f}",
            f"{spy_metrics['strategy_annualized_return']:.2f}",
            f"{spy_metrics['strategy_volatility']:.2f}",
            f"{spy_metrics['strategy_max_drawdown']:.2f}",
            f"{spy_metrics['strategy_sharpe']:.3f}",
            f"{spy_metrics['strategy_sortino']:.3f}",
            f"{spy_metrics['strategy_calmar']:.3f}",
            f"{spy_metrics['time_in_market_pct']:.1f}",
            f"{spy_metrics['win_rate']:.1f}",
            f"{spy_metrics['trades']}"
        ],
        'SPY Buy & Hold': [
            f"{spy_metrics['buy_hold_total_return']:.2f}",
            f"{spy_metrics['buy_hold_annualized_return']:.2f}",
            f"{spy_metrics['buy_hold_volatility']:.2f}",
            f"{spy_metrics['buy_hold_max_drawdown']:.2f}",
            f"{spy_metrics['buy_hold_sharpe']:.3f}",
            f"{spy_metrics['buy_hold_sortino']:.3f}",
            f"{spy_metrics['buy_hold_calmar']:.3f}",
            "100.0", "-", "0"
        ]
    })

    print("\n")
    print(comparison.to_string(index=False))
    return comparison


# ============================================
# SECTION 4: MAIN EXECUTION
# ============================================

def main():
    """
    Orchestrate the full performance evaluation for both TAN and SPY.

    Calls analyze_performance() for each ETF, constructs the comparison
    table, saves both outputs to CSV, and prints a structured interpretation
    of the results in relation to H3 (and secondarily H1).
    """

    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS CALCULATION")
    print("=" * 60)
    print("\nThis script addresses H3: Can sentiment signals outperform")
    print("a passive buy-and-hold strategy on a risk-adjusted basis?")

    # Run the full metric suite for each ETF independently
    tan_metrics = analyze_performance('TAN')
    spy_metrics = analyze_performance('SPY')

    # Build and display the side-by-side comparison table
    comparison = create_summary_table(tan_metrics, spy_metrics)

    # -----------------------------------------------------------------------
    # SAVE OUTPUTS
    # Two files are produced: a detailed file with every computed metric
    # (for reproducibility and supplementary material), and a formatted
    # summary table intended for the dissertation results chapter.
    # -----------------------------------------------------------------------

    metrics_df = pd.DataFrame([tan_metrics, spy_metrics])
    metrics_df.to_csv('results/performance_metrics_detailed.csv', index=False)
    print(f"\nSaved: results/performance_metrics_detailed.csv")

    comparison.to_csv('results/performance_summary.csv', index=False)
    print(f"Saved: results/performance_summary.csv")

    # -----------------------------------------------------------------------
    # H3 INTERPRETATION
    # The script auto-interprets the results relative to H3 to provide a
    # preliminary finding. This does not replace the formal hypothesis
    # discussion in the dissertation, but serves as an immediate sense-check.
    # -----------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("KEY FINDINGS - H3 EVALUATION")
    print("=" * 60)

    for metrics, label in [(tan_metrics, 'TAN'), (spy_metrics, 'SPY')]:
        print(f"\n{label}:")
        direction = "OUTPERFORMED" if metrics['outperformance'] > 0 else "UNDERPERFORMED"
        print(f"  Strategy {direction} buy-and-hold by {abs(metrics['outperformance']):.2f}%")
        sharpe_dir = "improved" if metrics['sharpe_improvement'] > 0 else "lower"
        print(f"  Sharpe ratio {sharpe_dir} by {abs(metrics['sharpe_improvement']):.3f}")
        vol_dir = "reduced" if metrics['volatility_reduction'] > 0 else "increased"
        print(f"  Volatility {vol_dir} by {abs(metrics['volatility_reduction']):.2f}%")

    # Compare the strategy benefit across the two ETFs — relevant to H1
    print(f"\n  Comparative insight (relevant to H1):")
    if tan_metrics['outperformance'] > spy_metrics['outperformance']:
        print(f"  Sentiment strategy shows greater benefit for TAN than SPY,")
        print(f"  consistent with H1 — sector-specific ETFs are more sensitive")
        print(f"  to targeted sentiment signals than broad market ETFs.")
    else:
        print(f"  Sentiment strategy shows greater benefit for SPY than TAN.")
        print(f"  This warrants further discussion in relation to H1.")

    print("\n" + "=" * 60)
    print("Next step: Run charts.py")
    print("=" * 60)


# ============================================
# SECTION 5: ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()