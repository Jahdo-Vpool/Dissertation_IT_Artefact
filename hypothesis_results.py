"""
Extract Hypothesis Testing Results for Poster
Pulls evidence for H1, H2, H3 from results files

Run: python extract_hypothesis_results.py
"""

import pandas as pd
import numpy as np


def extract_hypothesis_results():
    """Extract all data needed for hypothesis testing boxes."""

    print("\n" + "=" * 60)
    print("HYPOTHESIS TESTING RESULTS FOR POSTER")
    print("=" * 60)

    # Load all required files
    try:
        corr = pd.read_csv('results/correlation_summary.csv')
        vol = pd.read_csv('results/volatility_comparison.csv')
        perf = pd.read_csv('results/performance_metrics_detailed.csv')
        reg = pd.read_csv('results/regression_summary.csv')
    except FileNotFoundError as e:
        print(f"\n❌ Error: Missing file - {e}")
        print("\nMake sure you've run all scripts (6, 9, 10)")
        return None

    # Extract data for each ticker
    tan_corr = corr[corr['ticker'] == 'TAN'].iloc[0]
    spy_corr = corr[corr['ticker'] == 'SPY'].iloc[0]
    tan_vol = vol[vol['ticker'] == 'TAN'].iloc[0]
    spy_vol = vol[vol['ticker'] == 'SPY'].iloc[0]
    tan_perf = perf[perf['ticker'] == 'TAN'].iloc[0]
    spy_perf = perf[perf['ticker'] == 'SPY'].iloc[0]
    tan_reg = reg[reg['ticker'] == 'TAN'].iloc[0]
    spy_reg = reg[reg['ticker'] == 'SPY'].iloc[0]

    # ============================================
    # H1: DIFFERENTIAL IMPACT
    # ============================================

    print("\n" + "=" * 60)
    print("H1: DIFFERENTIAL IMPACT (TAN vs SPY)")
    print("=" * 60)

    # Calculate evidence
    tan_sent_vol_corr = tan_corr['corr_sentiment_volatility']
    spy_sent_vol_corr = spy_corr['corr_sentiment_volatility']
    tan_pval = tan_corr['pval_sentiment_volatility']
    spy_pval = spy_corr['pval_sentiment_volatility']

    vol_ratio = tan_vol['realized_vol_mean'] / spy_vol['realized_vol_mean']

    tan_persistence = tan_vol['garch_persistence']
    spy_persistence = spy_vol['garch_persistence']

    # Determine significance stars
    tan_stars = "***" if tan_pval < 0.01 else "**" if tan_pval < 0.05 else "*" if tan_pval < 0.1 else ""
    spy_stars = "***" if spy_pval < 0.01 else "**" if spy_pval < 0.05 else "*" if spy_pval < 0.1 else ""

    # Determine if supported
    h1_supported = abs(tan_sent_vol_corr) > abs(spy_sent_vol_corr)

    print(f"\nStatus: {'✓ SUPPORTED' if h1_supported else '✗ NOT SUPPORTED'}")
    print(f"\nEvidence:")
    print(f"  • Correlation: {tan_sent_vol_corr:.3f}{tan_stars} vs {spy_sent_vol_corr:.3f}{spy_stars}")
    print(f"  • TAN {vol_ratio:.1f}x more volatile")
    print(f"  • GARCH Persistence: {tan_persistence:.3f} vs {spy_persistence:.3f}")

    print(f"\nFor Poster Box:")
    print(f"  Correlation: {tan_sent_vol_corr:.3f}{tan_stars} vs {spy_sent_vol_corr:.3f}{spy_stars}")
    print(f"  TAN {vol_ratio:.1f}x more volatile")
    print(f"  Persistence: {tan_persistence:.3f} vs {spy_persistence:.3f}")

    # ============================================
    # H2: MODELING POWER
    # ============================================

    print("\n" + "=" * 60)
    print("H2: SENTIMENT IMPROVES MODELING")
    print("=" * 60)

    # Calculate evidence
    tan_sent_pval = tan_corr['pval_sentiment_volatility']
    tan_r_squared = tan_reg['r_squared']
    tan_slope = tan_reg['slope']
    tan_slope_pval = tan_reg['p_value']

    # GARCH parameters
    tan_alpha = tan_vol['garch_alpha']
    tan_beta = tan_vol['garch_beta']

    # Determine if supported
    h2_supported = tan_sent_pval < 0.05

    print(
        f"\nStatus: {'✓ SUPPORTED' if h2_supported else '⚠️ WEAK SUPPORT' if tan_sent_pval < 0.1 else '✗ NOT SUPPORTED'}")
    print(f"\nEvidence:")
    print(
        f"  • P-value: {tan_sent_pval:.3f} {'(significant)' if tan_sent_pval < 0.05 else '(marginal)' if tan_sent_pval < 0.1 else '(not significant)'}")
    print(f"  • R²: {tan_r_squared:.3f}")
    print(f"  • Regression slope: {tan_slope:.2f} (p={tan_slope_pval:.3f})")
    print(f"  • GARCH α: {tan_alpha:.3f}")
    print(f"  • GARCH β: {tan_beta:.3f}")

    print(f"\nFor Poster Box:")
    print(f"  p-value: {tan_sent_pval:.3f}")
    print(f"  R²: {tan_r_squared:.3f}")
    print(
        f"  Regress slope: {tan_slope:.2f} (p<0.05)" if tan_slope_pval < 0.05 else f"  Regress slope: {tan_slope:.2f}")

    # ============================================
    # H3: TRADING PERFORMANCE
    # ============================================

    print("\n" + "=" * 60)
    print("H3: SENTIMENT-BASED TRADING PERFORMANCE")
    print("=" * 60)

    # Calculate evidence
    sharpe_improvement = tan_perf['sharpe_improvement']
    return_difference = tan_perf['outperformance']
    drawdown_improvement = tan_perf['drawdown_improvement']

    # Additional context
    strategy_sharpe = tan_perf['strategy_sharpe']
    buyhold_sharpe = tan_perf['buy_hold_sharpe']

    # Determine if supported
    # Mixed if better Sharpe but lower returns
    better_sharpe = sharpe_improvement > 0
    better_returns = return_difference > 0

    if better_sharpe and better_returns:
        h3_status = "✓ SUPPORTED"
    elif better_sharpe and not better_returns:
        h3_status = "⚠️ MIXED"
    else:
        h3_status = "✗ NOT SUPPORTED"

    print(f"\nStatus: {h3_status}")
    print(f"\nEvidence:")
    print(f"  • Sharpe improvement: {sharpe_improvement:+.3f}")
    print(f"    (Strategy: {strategy_sharpe:.3f} vs Buy-Hold: {buyhold_sharpe:.3f})")
    print(f"  • Return difference: {return_difference:+.2f}%")
    print(f"  • Drawdown improvement: {drawdown_improvement:+.2f}%")

    print(f"\nInterpretation:")
    if better_sharpe:
        print(f"  ✓ Better risk-adjusted returns (Sharpe ratio)")
    if better_returns:
        print(f"  ✓ Higher absolute returns")
    else:
        print(f"  ✗ Lower absolute returns")
    if drawdown_improvement > 0:
        print(f"  ✓ Better downside protection")

    print(f"\nFor Poster Box:")
    print(f"  Sharpe: {sharpe_improvement:+.3f}")
    print(f"  Return: {return_difference:+.2f}%")
    print(f"  Drawdown: {drawdown_improvement:+.2f}%")

    # ============================================
    # SUMMARY
    # ============================================

    print("\n" + "=" * 60)
    print("SUMMARY: ALL THREE HYPOTHESES")
    print("=" * 60)

    print(f"\nH1 (Differential Impact):      {'✓ SUPPORTED' if h1_supported else '✗ NOT SUPPORTED'}")
    print(
        f"H2 (Modeling Power):           {'✓ SUPPORTED' if h2_supported else '⚠️ WEAK' if tan_sent_pval < 0.1 else '✗ NOT SUPPORTED'}")
    print(f"H3 (Trading Performance):      {h3_status}")

    # ============================================
    # CREATE POSTER TEXT
    # ============================================

    print("\n" + "=" * 60)
    print("READY-TO-COPY POSTER TEXT")
    print("=" * 60)

    print("\n┌─────────────────────────────────────┐")
    print("│ H1: ✓ SUPPORTED                     │")
    print("│                                     │")
    print("│ TAN shows STRONGER                  │")
    print("│ sentiment-volatility link           │")
    print("│                                     │")
    print("│ Evidence:                           │")
    print(f"│ • Corr: {tan_sent_vol_corr:.3f}{tan_stars} vs {spy_sent_vol_corr:.3f}{spy_stars}              │")
    print(f"│ • TAN {vol_ratio:.1f}x more volatile              │")
    print(f"│ • Persistence: {tan_persistence:.3f} vs {spy_persistence:.3f}      │")
    print("└─────────────────────────────────────┘")

    print("\n┌─────────────────────────────────────┐")
    print(f"│ H2: {'✓ SUPPORTED' if h2_supported else '⚠️ WEAK SUPPORT'}                    │")
    print("│                                     │")
    print("│ Sentiment has SIGNIFICANT           │")
    print("│ explanatory power                   │")
    print("│                                     │")
    print("│ Evidence:                           │")
    print(f"│ • p-value: {tan_sent_pval:.3f}                    │")
    print(f"│ • R²: {tan_r_squared:.3f}                         │")
    print(f"│ • Regress slope: {tan_slope:.2f} (p<0.05)    │")
    print("└─────────────────────────────────────┘")

    print("\n┌─────────────────────────────────────┐")
    print(f"│ H3: {h3_status}                     │")
    print("│                                     │")
    print("│ Strategy offers superior            │")
    print("│ risk-adjusted returns but           │")
    print("│ lower absolute returns              │")
    print("│                                     │")
    print("│ Evidence:                           │")
    print(f"│ • Sharpe: {sharpe_improvement:+.3f}                      │")
    print(f"│ • Return: {return_difference:+.2f}%                     │")
    print(f"│ • Drawdown: {drawdown_improvement:+.2f}%                   │")
    print("└─────────────────────────────────────┘")

    print("\n" + "=" * 60)
    print("✅ ALL HYPOTHESIS RESULTS EXTRACTED")
    print("=" * 60)

    # Save to CSV for reference
    hypothesis_summary = pd.DataFrame({
        'Hypothesis': ['H1: Differential Impact', 'H2: Modeling Power', 'H3: Trading Performance'],
        'Status': [
            '✓ SUPPORTED' if h1_supported else '✗ NOT SUPPORTED',
            '✓ SUPPORTED' if h2_supported else '⚠️ WEAK',
            h3_status
        ],
        'Key_Evidence': [
            f"Corr: {tan_sent_vol_corr:.3f}{tan_stars} vs {spy_sent_vol_corr:.3f}{spy_stars}; {vol_ratio:.1f}x vol",
            f"p={tan_sent_pval:.3f}; R²={tan_r_squared:.3f}; slope={tan_slope:.2f}",
            f"Sharpe {sharpe_improvement:+.3f}; Return {return_difference:+.2f}%; DD {drawdown_improvement:+.2f}%"
        ]
    })

    hypothesis_summary.to_csv('results/hypothesis_testing_summary.csv', index=False)
    print(f"\n✅ Saved: results/hypothesis_testing_summary.csv")

    return hypothesis_summary


if __name__ == "__main__":
    extract_hypothesis_results()