"""
================================================================================
Script 10: Hypothesis Testing Results Extraction
================================================================================
Purpose:
    Consolidates statistical evidence from across the pipeline into a single,
    structured evaluation of the three research hypotheses. Rather than
    computing new statistics, this script reads the outputs already produced
    by Scripts 6, 8, and 9, extracts the relevant values, applies decision
    rules to determine each hypothesis outcome, and presents the findings in
    a format suitable for academic reporting and poster presentation.

Academic Context:
    The three hypotheses under examination are:

    H1 - Differential Impact:
        Sentiment has a stronger influence on TAN (Invesco Solar ETF) than
        on SPY (S&P 500 ETF), due to TAN's concentration in a single
        sentiment-sensitive sector.

    H2 - Modelling Power:
        News sentiment carries statistically significant explanatory power
        for ETF return volatility, as evidenced by regression analysis and
        GARCH model parameters.

    H3 - Trading Performance:
        A sentiment-based rules trading strategy produces superior
        risk-adjusted returns compared to a passive buy-and-hold strategy.

    Evaluating these hypotheses is the culmination of the entire analytical
    pipeline, and directly addresses the dissertation's central research
    question.

Inputs:
    - results/correlation_summary.csv         : From Script 9
    - results/volatility_comparison.csv       : From Script 8
    - results/performance_metrics_detailed.csv: From Script 6
    - results/regression_summary.csv          : From Script 10

Outputs:
    - results/hypothesis_testing_summary.csv  : Structured hypothesis outcomes
    - Console output: Formatted evidence tables and poster-ready text

Pipeline Position:
    Script 10 of 10. Depends on Scripts 6, 8, and 9 having been run
    successfully.

Dependencies:
    - pandas : Data loading and CSV export
    - numpy  : Numerical operations

Usage:
    python hypothesis_results.py
================================================================================
"""

import pandas as pd
import numpy as np


# ============================================
# SECTION 1: DATA LOADING AND EXTRACTION
# ============================================

def extract_hypothesis_results():
    """
    Load all required results files, extract hypothesis-relevant statistics,
    apply evidence-based decision rules, and report findings.

    This function is structured in four parts, one per hypothesis plus a
    final summary. Each hypothesis section follows the same pattern:
    extract the key statistics, apply a decision rule, print the evidence,
    and format output for the academic poster.

    Returns
    -------
    pandas.DataFrame or None
        A summary DataFrame with one row per hypothesis, containing the
        outcome status and key supporting evidence. Returns None if any
        required input file is missing.
    """

    print("\n" + "=" * 60)
    print("HYPOTHESIS TESTING RESULTS FOR POSTER")
    print("=" * 60)

    # Load the four results files produced by earlier scripts.
    # All four are required; if any is missing, the function exits early
    # with an informative error message rather than producing partial output.
    try:
        corr = pd.read_csv('results/correlation_summary.csv')       # From Script 10
        vol  = pd.read_csv('results/volatility_comparison.csv')      # From Script 9
        perf = pd.read_csv('results/performance_metrics_detailed.csv') # From Script 6
        reg  = pd.read_csv('results/regression_summary.csv')         # From Script 10
    except FileNotFoundError as e:
        print(f"\nError: Missing file - {e}")
        print("\nEnsure Scripts 6, 9, and 10 have been run before this script.")
        return None

    # Filter each DataFrame to isolate the TAN and SPY rows.
    # Using iloc[0] retrieves the first matching row as a Series,
    # allowing individual fields to be accessed by column name.
    tan_corr = corr[corr['ticker'] == 'TAN'].iloc[0]
    spy_corr = corr[corr['ticker'] == 'SPY'].iloc[0]
    tan_vol  = vol[vol['ticker'] == 'TAN'].iloc[0]
    spy_vol  = vol[vol['ticker'] == 'SPY'].iloc[0]
    tan_perf = perf[perf['ticker'] == 'TAN'].iloc[0]
    spy_perf = perf[perf['ticker'] == 'SPY'].iloc[0]
    tan_reg  = reg[reg['ticker'] == 'TAN'].iloc[0]
    spy_reg  = reg[reg['ticker'] == 'SPY'].iloc[0]


    # ============================================
    # SECTION 2: HYPOTHESIS 1 — DIFFERENTIAL IMPACT
    # ============================================
    # H1 proposes that sentiment has a stronger and more direct influence
    # on TAN than on SPY. This is evaluated using three pieces of evidence:
    #
    #   1. The Pearson correlation between sentiment and realised volatility
    #      for each ETF. A larger absolute correlation for TAN supports H1.
    #
    #   2. The volatility ratio (TAN / SPY), which quantifies how much more
    #      volatile TAN is relative to SPY. Higher volatility in a sector ETF
    #      is consistent with greater sensitivity to sentiment shocks.
    #
    #   3. GARCH persistence (alpha + beta), which measures how long a
    #      volatility shock takes to decay. Higher persistence in TAN would
    #      indicate that sentiment-driven shocks have a more lasting effect.

    print("\n" + "=" * 60)
    print("H1: DIFFERENTIAL IMPACT (TAN vs SPY)")
    print("=" * 60)

    tan_sent_vol_corr = tan_corr['corr_sentiment_volatility']
    spy_sent_vol_corr = spy_corr['corr_sentiment_volatility']
    tan_pval          = tan_corr['pval_sentiment_volatility']
    spy_pval          = spy_corr['pval_sentiment_volatility']

    # Volatility ratio: how many times more volatile is TAN than SPY on average.
    vol_ratio = tan_vol['realized_vol_mean'] / spy_vol['realized_vol_mean']

    tan_persistence = tan_vol['garch_persistence']
    spy_persistence = spy_vol['garch_persistence']

    # Assign statistical significance stars following standard academic
    # convention: *** p<0.01, ** p<0.05, * p<0.1. These are appended to
    # correlation values in the output for immediate readability.
    tan_stars = "***" if tan_pval < 0.01 else "**" if tan_pval < 0.05 else "*" if tan_pval < 0.1 else ""
    spy_stars = "***" if spy_pval < 0.01 else "**" if spy_pval < 0.05 else "*" if spy_pval < 0.1 else ""

    # Decision rule for H1: supported if TAN's sentiment-volatility
    # correlation is larger in absolute magnitude than SPY's. This is the
    # primary quantitative criterion for differential impact.
    h1_supported = abs(tan_sent_vol_corr) > abs(spy_sent_vol_corr)

    print(f"\nStatus: {'SUPPORTED' if h1_supported else 'NOT SUPPORTED'}")
    print(f"\nEvidence:")
    print(f"  Correlation: {tan_sent_vol_corr:.3f}{tan_stars} vs {spy_sent_vol_corr:.3f}{spy_stars}")
    print(f"  TAN {vol_ratio:.1f}x more volatile")
    print(f"  GARCH Persistence: {tan_persistence:.3f} vs {spy_persistence:.3f}")

    print(f"\nFor Poster Box:")
    print(f"  Correlation: {tan_sent_vol_corr:.3f}{tan_stars} vs {spy_sent_vol_corr:.3f}{spy_stars}")
    print(f"  TAN {vol_ratio:.1f}x more volatile")
    print(f"  Persistence: {tan_persistence:.3f} vs {spy_persistence:.3f}")


    # ============================================
    # SECTION 3: HYPOTHESIS 2 — MODELLING POWER
    # ============================================
    # H2 proposes that news sentiment carries statistically significant
    # explanatory power for ETF volatility. This is evaluated using:
    #
    #   1. The p-value from the sentiment-volatility regression. A p-value
    #      below 0.05 indicates that the sentiment coefficient is statistically
    #      different from zero, i.e. sentiment is a significant predictor.
    #
    #   2. R-squared, which quantifies the proportion of variance in volatility
    #      explained by the sentiment variable. A low R² is expected given
    #      the complexity of financial markets; even a small but significant
    #      R² is academically meaningful.
    #
    #   3. The regression slope, which indicates the direction and magnitude
    #      of the relationship. A positive slope means that higher sentiment
    #      is associated with higher volatility; a negative slope the reverse.
    #
    #   4. GARCH alpha and beta parameters, which describe the ARCH effect
    #      (sensitivity to recent shocks) and volatility persistence
    #      respectively. These contextualise how the GARCH model captures
    #      volatility dynamics independently of sentiment.

    print("\n" + "=" * 60)
    print("H2: SENTIMENT IMPROVES MODELLING")
    print("=" * 60)

    tan_sent_pval  = tan_corr['pval_sentiment_volatility']
    tan_r_squared  = tan_reg['r_squared']
    tan_slope      = tan_reg['slope']
    tan_slope_pval = tan_reg['p_value']
    tan_alpha      = tan_vol['garch_alpha']
    tan_beta       = tan_vol['garch_beta']

    # Decision rule for H2: supported if the regression p-value falls below
    # the conventional 0.05 significance threshold. A marginal result
    # (p < 0.10) is reported separately to avoid a binary pass/fail framing
    # that could obscure borderline findings.
    h2_supported = tan_sent_pval < 0.05

    print(f"\nStatus: {'SUPPORTED' if h2_supported else 'WEAK SUPPORT' if tan_sent_pval < 0.1 else 'NOT SUPPORTED'}")
    print(f"\nEvidence:")
    print(f"  P-value:          {tan_sent_pval:.3f} {'(significant)' if tan_sent_pval < 0.05 else '(marginal)' if tan_sent_pval < 0.1 else '(not significant)'}")
    print(f"  R-squared:        {tan_r_squared:.3f}")
    print(f"  Regression slope: {tan_slope:.2f} (p={tan_slope_pval:.3f})")
    print(f"  GARCH alpha:      {tan_alpha:.3f}")
    print(f"  GARCH beta:       {tan_beta:.3f}")

    print(f"\nFor Poster Box:")
    print(f"  p-value: {tan_sent_pval:.3f}")
    print(f"  R-squared: {tan_r_squared:.3f}")
    print(f"  Regression slope: {tan_slope:.2f} (p<0.05)" if tan_slope_pval < 0.05 else f"  Regression slope: {tan_slope:.2f}")


    # ============================================
    # SECTION 4: HYPOTHESIS 3 — TRADING PERFORMANCE
    # ============================================
    # H3 proposes that a sentiment-based trading strategy outperforms a
    # passive buy-and-hold strategy. Three metrics are used to evaluate this:
    #
    #   1. Sharpe ratio improvement: the difference in risk-adjusted return
    #      between the sentiment strategy and buy-and-hold. The Sharpe ratio
    #      is the preferred metric here because it penalises excessive risk
    #      taking, making it a more robust performance measure than raw return.
    #
    #   2. Return difference: the difference in absolute total return between
    #      the two strategies. A strategy can improve the Sharpe ratio while
    #      still delivering lower absolute returns if it significantly reduces
    #      volatility — this nuance is captured by reporting both metrics.
    #
    #   3. Drawdown improvement: the difference in maximum drawdown. A smaller
    #      maximum drawdown indicates better downside protection, which is
    #      particularly relevant for risk-averse investors.
    #
    # H3 is classified as Mixed if the strategy improves the Sharpe ratio
    # but not absolute returns, reflecting the academic convention of
    # acknowledging partial support rather than forcing a binary outcome.

    print("\n" + "=" * 60)
    print("H3: SENTIMENT-BASED TRADING PERFORMANCE")
    print("=" * 60)

    sharpe_improvement  = tan_perf['sharpe_improvement']
    return_difference   = tan_perf['outperformance']
    drawdown_improvement = tan_perf['drawdown_improvement']
    strategy_sharpe     = tan_perf['strategy_sharpe']
    buyhold_sharpe      = tan_perf['buy_hold_sharpe']

    better_sharpe  = sharpe_improvement > 0
    better_returns = return_difference > 0

    # Three-way classification for H3:
    # Supported — better Sharpe AND better absolute returns
    # Mixed     — better Sharpe but lower absolute returns
    # Not Supported — worse Sharpe ratio
    if better_sharpe and better_returns:
        h3_status = "SUPPORTED"
    elif better_sharpe and not better_returns:
        h3_status = "MIXED"
    else:
        h3_status = "NOT SUPPORTED"

    print(f"\nStatus: {h3_status}")
    print(f"\nEvidence:")
    print(f"  Sharpe improvement: {sharpe_improvement:+.3f}")
    print(f"    (Strategy: {strategy_sharpe:.3f} vs Buy-Hold: {buyhold_sharpe:.3f})")
    print(f"  Return difference:  {return_difference:+.2f}%")
    print(f"  Drawdown improvement: {drawdown_improvement:+.2f}%")

    print(f"\nInterpretation:")
    if better_sharpe:
        print(f"  Better risk-adjusted returns (Sharpe ratio)")
    if better_returns:
        print(f"  Higher absolute returns")
    else:
        print(f"  Lower absolute returns")
    if drawdown_improvement > 0:
        print(f"  Better downside protection")

    print(f"\nFor Poster Box:")
    print(f"  Sharpe: {sharpe_improvement:+.3f}")
    print(f"  Return: {return_difference:+.2f}%")
    print(f"  Drawdown: {drawdown_improvement:+.2f}%")


    # ============================================
    # SECTION 5: CONSOLIDATED SUMMARY
    # ============================================

    print("\n" + "=" * 60)
    print("SUMMARY: ALL THREE HYPOTHESES")
    print("=" * 60)

    print(f"\nH1 (Differential Impact):  {'SUPPORTED' if h1_supported else 'NOT SUPPORTED'}")
    print(f"H2 (Modelling Power):      {'SUPPORTED' if h2_supported else 'WEAK SUPPORT' if tan_sent_pval < 0.1 else 'NOT SUPPORTED'}")
    print(f"H3 (Trading Performance):  {h3_status}")


    # ============================================
    # SECTION 6: POSTER-READY FORMATTED OUTPUT
    # ============================================
    # The boxed text below is formatted to be copied directly into the
    # academic poster (Script 12). Each box corresponds to one hypothesis
    # and contains the status label, interpretive statement, and
    # key numerical evidence with significance indicators.

    print("\n" + "=" * 60)
    print("READY-TO-COPY POSTER TEXT")
    print("=" * 60)

    print("\n+-------------------------------------+")
    print(f"| H1: {'SUPPORTED' if h1_supported else 'NOT SUPPORTED':<32}|")
    print("|                                     |")
    print("| TAN shows STRONGER                  |")
    print("| sentiment-volatility link           |")
    print("|                                     |")
    print("| Evidence:                           |")
    print(f"| Corr: {tan_sent_vol_corr:.3f}{tan_stars} vs {spy_sent_vol_corr:.3f}{spy_stars}              |")
    print(f"| TAN {vol_ratio:.1f}x more volatile              |")
    print(f"| Persistence: {tan_persistence:.3f} vs {spy_persistence:.3f}      |")
    print("+-------------------------------------+")

    print("\n+-------------------------------------+")
    print(f"| H2: {'SUPPORTED' if h2_supported else 'WEAK SUPPORT':<32}|")
    print("|                                     |")
    print("| Sentiment has SIGNIFICANT           |")
    print("| explanatory power                   |")
    print("|                                     |")
    print("| Evidence:                           |")
    print(f"| p-value: {tan_sent_pval:.3f}                    |")
    print(f"| R-squared: {tan_r_squared:.3f}                  |")
    print(f"| Regression slope: {tan_slope:.2f}             |")
    print("+-------------------------------------+")

    print("\n+-------------------------------------+")
    print(f"| H3: {h3_status:<33}|")
    print("|                                     |")
    print("| Strategy offers superior            |")
    print("| risk-adjusted returns               |")
    print("|                                     |")
    print("| Evidence:                           |")
    print(f"| Sharpe: {sharpe_improvement:+.3f}                      |")
    print(f"| Return: {return_difference:+.2f}%                     |")
    print(f"| Drawdown: {drawdown_improvement:+.2f}%                   |")
    print("+-------------------------------------+")

    print("\n" + "=" * 60)
    print("ALL HYPOTHESIS RESULTS EXTRACTED")
    print("=" * 60)


    # ============================================
    # SECTION 7: SAVE SUMMARY TO CSV
    # ============================================
    # The hypothesis outcomes are saved as a structured CSV so they can be
    # referenced programmatically in Script 12 (create_poster_table.py)
    # without needing to re-run the full extraction logic.

    hypothesis_summary = pd.DataFrame({
        'Hypothesis': [
            'H1: Differential Impact',
            'H2: Modelling Power',
            'H3: Trading Performance'
        ],
        'Status': [
            'SUPPORTED' if h1_supported else 'NOT SUPPORTED',
            'SUPPORTED' if h2_supported else 'WEAK SUPPORT',
            h3_status
        ],
        'Key_Evidence': [
            f"Corr: {tan_sent_vol_corr:.3f}{tan_stars} vs {spy_sent_vol_corr:.3f}{spy_stars}; {vol_ratio:.1f}x vol",
            f"p={tan_sent_pval:.3f}; R-squared={tan_r_squared:.3f}; slope={tan_slope:.2f}",
            f"Sharpe {sharpe_improvement:+.3f}; Return {return_difference:+.2f}%; Drawdown {drawdown_improvement:+.2f}%"
        ]
    })

    hypothesis_summary.to_csv('results/hypothesis_testing_summary.csv', index=False)
    print(f"\nSaved: results/hypothesis_testing_summary.csv")

    return hypothesis_summary


# ============================================
# SECTION 8: ENTRY POINT
# ============================================

if __name__ == "__main__":
    extract_hypothesis_results()