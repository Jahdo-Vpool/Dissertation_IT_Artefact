"""
================================================================================
Script 9: Sentiment-Volatility Correlation Analysis
================================================================================
Purpose:
    Performs comprehensive statistical analysis to evaluate the relationship
    between news sentiment scores and ETF price behaviour (returns and
    volatility) for both TAN and SPY. This script produces the primary
    quantitative evidence used to accept or reject all three research
    hypotheses.

Academic Context:
    This script directly addresses Dissertation Objective 3 — evaluating
    the correlation and predictive relationship between sentiment and
    volatility. It also consolidates evidence for:

        H1: Sentiment has a stronger influence on TAN than on SPY, due to
            TAN's sector-specific nature and narrower investor base.

        H2: Incorporating sentiment improves the explanatory power of
            volatility models beyond price-based inputs alone.

        H3: Sentiment-informed trading signals can outperform a passive
            buy-and-hold strategy on a risk-adjusted basis.

    Four analytical approaches are applied: Pearson correlation (parametric,
    assumes normality), Spearman rank correlation (non-parametric, robust to
    outliers), lead-lag analysis (tests whether sentiment predicts future
    returns), and OLS regression (quantifies the magnitude of sentiment's
    effect on returns and provides R-squared as an explanatory power metric).

Inputs:
    - results/TAN_merge_prices_news.csv  : Merged TAN sentiment and price data
    - results/SPY_merge_prices_news.csv  : Merged SPY sentiment and price data
    - results/TAN_volatility_analysis.csv: TAN GARCH and realised volatility
    - results/SPY_volatility_analysis.csv: SPY GARCH and realised volatility

Outputs:
    - results/correlation_summary.csv    : All Pearson/Spearman correlations
    - results/regression_summary.csv     : OLS regression coefficients and fit
    - results/TAN_correlation_matrix.csv : Full variable correlation matrix
    - results/SPY_correlation_matrix.csv : Full variable correlation matrix

Pipeline Position:
    Script 9 of 10. Final analytical step. Results feed directly into
    the dissertation Results and Discussion chapters.

Dependencies:
    - pandas  : Data loading and manipulation
    - numpy   : Numerical operations
    - scipy   : Statistical tests (Pearson, Spearman, OLS regression)

Usage:
    python correlation_analysis.py
================================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings

warnings.filterwarnings('ignore')


# ============================================
# SECTION 1: CORRELATION ANALYSIS
# ============================================

def calculate_correlations(df, ticker):
    """
    Calculate Pearson and Spearman correlations between sentiment scores
    and key price-based variables: same-day returns, next-day returns
    (lead-lag), realised volatility, and absolute returns as a volatility
    proxy.

    Two correlation methods are used deliberately:
    - Pearson: Assumes a linear relationship and normally distributed
      variables. Standard in financial econometrics and directly comparable
      across ETFs.
    - Spearman: Rank-based and makes no distributional assumptions. Used as
      a robustness check given that financial returns are known to exhibit
      fat tails and non-normality (Cont, 2001).

    Parameters
    ----------
    df : pandas.DataFrame
        Merged dataset containing at minimum: sentiment_score, return,
        and date columns.
    ticker : str
        ETF identifier ('TAN' or 'SPY'). Used for labelling output.

    Returns
    -------
    dict
        Dictionary of correlation coefficients and p-values for all
        variable pairs tested.
    """

    print(f"\n{'=' * 60}")
    print(f"{ticker} - CORRELATION ANALYSIS")
    print(f"{'=' * 60}")

    # Drop rows with missing values. NaN values can arise from the rolling
    # window calculations in Script 9 (first N rows have no volatility
    # estimate) and from days where no news articles were collected.
    df_clean = df.dropna()
    print(f"  Analyzing {len(df_clean)} complete observations")

    # -----------------------------------------------------------------------
    # 1A. SENTIMENT-RETURN CORRELATIONS (SAME DAY)
    # Tests whether sentiment on a given day is associated with the return
    # on that same day. A positive correlation would suggest that more
    # positive news coverage coincides with price appreciation, consistent
    # with the efficient market hypothesis in its semi-strong form.
    # -----------------------------------------------------------------------

    print(f"\n  Sentiment-Return Relationships:")

    corr_same_day, p_same_day = pearsonr(
        df_clean['sentiment_score'],
        df_clean['return']
    )
    print(f"    Sentiment vs Same-day return:")
    print(f"      Correlation: {corr_same_day:+.3f}")
    print(f"      P-value:     {p_same_day:.4f} "
          f"{'***' if p_same_day < 0.01 else '**' if p_same_day < 0.05 else '*' if p_same_day < 0.1 else 'ns'}")

    # -----------------------------------------------------------------------
    # 1B. LEAD-LAG ANALYSIS (SENTIMENT TODAY vs RETURN TOMORROW)
    # Shifts the return series forward by one day to test whether today's
    # sentiment score has predictive power for tomorrow's return. This is
    # the most practically relevant test for H3, as a trading strategy must
    # act on current signals to generate future returns.
    # -----------------------------------------------------------------------

    if 'next_return' not in df_clean.columns:
        df_clean['next_return'] = df_clean['return'].shift(-1)

    df_lead = df_clean.dropna()
    if len(df_lead) > 2:
        corr_next_day, p_next_day = pearsonr(
            df_lead['sentiment_score'],
            df_lead['next_return']
        )
        print(f"    Sentiment vs Next-day return (lead-lag):")
        print(f"      Correlation: {corr_next_day:+.3f}")
        print(f"      P-value:     {p_next_day:.4f} "
              f"{'***' if p_next_day < 0.01 else '**' if p_next_day < 0.05 else '*' if p_next_day < 0.1 else 'ns'}")
    else:
        corr_next_day, p_next_day = np.nan, np.nan

    # -----------------------------------------------------------------------
    # 1C. SENTIMENT-VOLATILITY CORRELATIONS
    # Merges realised volatility estimates from Script 8 and tests whether
    # sentiment is associated with the level of price variability. This
    # addresses H2 directly. The absolute sentiment score is also tested
    # to capture the hypothesis that extreme sentiment in either direction
    # (strongly positive or strongly negative) coincides with higher
    # volatility, regardless of sentiment direction.
    # -----------------------------------------------------------------------

    print(f"\n  Sentiment-Volatility Relationships:")

    try:
        vol_df = pd.read_csv(f'results/{ticker}_volatility_analysis.csv')
        merged = df_clean.merge(
            vol_df[['date', 'volatility_10d', 'garch_volatility']],
            on='date',
            how='left'
        )
        merged = merged.dropna()

        if len(merged) > 2:
            # Directional sentiment vs realised volatility
            corr_vol, p_vol = pearsonr(
                merged['sentiment_score'],
                merged['volatility_10d']
            )
            print(f"    Sentiment vs Realised volatility:")
            print(f"      Correlation: {corr_vol:+.3f}")
            print(f"      P-value:     {p_vol:.4f} "
                  f"{'***' if p_vol < 0.01 else '**' if p_vol < 0.05 else '*' if p_vol < 0.1 else 'ns'}")

            # Sentiment extremity (absolute value) vs realised volatility.
            # A negative directional correlation but a positive extremity
            # correlation would suggest that uncertainty (not just negative
            # sentiment) drives volatility — a nuanced finding worth discussing.
            merged['sentiment_abs'] = merged['sentiment_score'].abs()
            corr_ext_vol, p_ext_vol = pearsonr(
                merged['sentiment_abs'],
                merged['volatility_10d']
            )
            print(f"    |Sentiment| vs Volatility (extremity test):")
            print(f"      Correlation: {corr_ext_vol:+.3f}")
            print(f"      P-value:     {p_ext_vol:.4f} "
                  f"{'***' if p_ext_vol < 0.01 else '**' if p_ext_vol < 0.05 else '*' if p_ext_vol < 0.1 else 'ns'}")
        else:
            corr_vol, p_vol = np.nan, np.nan
            corr_ext_vol, p_ext_vol = np.nan, np.nan

    except Exception:
        # Volatility file may not exist if Script 9 was not run.
        # The script continues with NaN placeholders rather than aborting.
        print(f"    Volatility data not available - run volitility_calculation.py first")
        corr_vol, p_vol = np.nan, np.nan
        corr_ext_vol, p_ext_vol = np.nan, np.nan

    # -----------------------------------------------------------------------
    # 1D. ABSOLUTE RETURNS AS A VOLATILITY PROXY
    # When GARCH volatility estimates are unavailable, the absolute value
    # of daily returns serves as a simple, model-free volatility proxy.
    # Large absolute returns indicate high price movement regardless of
    # direction, and are widely used in the empirical finance literature
    # as a measure of realised market activity.
    # -----------------------------------------------------------------------

    print(f"\n  Sentiment vs Absolute Returns (volatility proxy):")

    df_clean['abs_return'] = df_clean['return'].abs()
    corr_abs, p_abs = pearsonr(
        df_clean['sentiment_score'],
        df_clean['abs_return']
    )
    print(f"    Correlation: {corr_abs:+.3f}")
    print(f"    P-value:     {p_abs:.4f} "
          f"{'***' if p_abs < 0.01 else '**' if p_abs < 0.05 else '*' if p_abs < 0.1 else 'ns'}")

    # -----------------------------------------------------------------------
    # 1E. SPEARMAN RANK CORRELATIONS (ROBUSTNESS CHECK)
    # These are computed for both return and absolute return to verify that
    # Pearson results are not driven by distributional outliers. If Pearson
    # and Spearman coefficients point in the same direction with similar
    # magnitudes, the finding can be considered robust to distributional
    # assumptions.
    # -----------------------------------------------------------------------

    print(f"\n  Spearman Rank Correlations (robustness check):")

    spearman_return, p_spear_ret = spearmanr(
        df_clean['sentiment_score'],
        df_clean['return']
    )
    print(f"    Sentiment vs Return:   {spearman_return:+.3f} (p={p_spear_ret:.4f})")

    spearman_abs, p_spear_abs = spearmanr(
        df_clean['sentiment_score'],
        df_clean['abs_return']
    )
    print(f"    Sentiment vs |Return|: {spearman_abs:+.3f} (p={p_spear_abs:.4f})")

    # Compile all results into a structured dictionary for export
    correlations = {
        'ticker': ticker,
        'observations': len(df_clean),
        'corr_sentiment_return_same': corr_same_day,
        'pval_sentiment_return_same': p_same_day,
        'corr_sentiment_return_next': corr_next_day,
        'pval_sentiment_return_next': p_next_day,
        'corr_sentiment_volatility': corr_vol,
        'pval_sentiment_volatility': p_vol,
        'corr_sentiment_abs_volatility': corr_ext_vol,
        'pval_sentiment_abs_volatility': p_ext_vol,
        'corr_sentiment_abs_return': corr_abs,
        'pval_sentiment_abs_return': p_abs,
        'spearman_sentiment_return': spearman_return,
        'spearman_sentiment_abs_return': spearman_abs,
    }

    return correlations


# ============================================
# SECTION 2: REGRESSION ANALYSIS
# ============================================

def regression_analysis(df, ticker):
    """
    Perform Ordinary Least Squares (OLS) regression of daily returns on
    sentiment scores to quantify the magnitude and statistical significance
    of the sentiment effect.

    The regression takes the form:
        Return_t = alpha + beta * Sentiment_t + epsilon_t

    Where:
        alpha  : Intercept (baseline return independent of sentiment)
        beta   : Regression coefficient (change in return per unit of
                 sentiment score). A positive beta would indicate that a
                 one-unit increase in sentiment is associated with a
                 beta-sized increase in daily returns.
        R²     : Proportion of return variance explained by sentiment alone.
                 Addresses H2 directly — a statistically significant R²
                 suggests sentiment has meaningful explanatory power.

    Parameters
    ----------
    df : pandas.DataFrame
        Merged dataset with sentiment_score and return columns.
    ticker : str
        ETF identifier for labelling.

    Returns
    -------
    dict
        Regression slope, intercept, R-squared, p-value, and standard error.
    """

    print(f"\n  OLS Regression: Return ~ Sentiment")

    df_clean = df.dropna()

    from scipy.stats import linregress

    slope, intercept, r_value, p_value, std_err = linregress(
        df_clean['sentiment_score'],
        df_clean['return']
    )

    print(f"    Equation:  Return = {intercept:.4f} + {slope:.4f} x Sentiment")
    print(f"    R-squared: {r_value ** 2:.4f}")
    print(f"    P-value:   {p_value:.4f} "
          f"{'***' if p_value < 0.01 else '**' if p_value < 0.05 else '*' if p_value < 0.1 else 'ns'}")
    print(f"    Std Error: {std_err:.4f}")

    return {
        'ticker': ticker,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'std_error': std_err
    }


# ============================================
# SECTION 3: COMPARATIVE ANALYSIS (TAN vs SPY)
# ============================================

def compare_correlations(tan_corr, spy_corr):
    """
    Directly compare the correlation magnitudes between TAN and SPY to
    evaluate Hypothesis 1.

    H1 predicts that TAN, as a narrow sector ETF tracking solar energy
    companies, will show a stronger sentiment-volatility relationship than
    SPY, which tracks 500 diversified companies across all sectors. The
    rationale is that sector-specific ETFs attract a more concentrated
    investor base whose sentiment is more directly shaped by domain-specific
    news coverage, whereas broad market ETFs are subject to a wider and
    more diffuse set of information inputs.

    Parameters
    ----------
    tan_corr : dict
        Correlation results dictionary for TAN (from calculate_correlations).
    spy_corr : dict
        Correlation results dictionary for SPY (from calculate_correlations).
    """

    print("\n" + "=" * 60)
    print("COMPARATIVE ANALYSIS: TAN vs SPY")
    print("=" * 60)

    # Sentiment-Return comparison
    print(f"\n  Sentiment vs Same-day Return:")
    print(f"    TAN: {tan_corr['corr_sentiment_return_same']:+.3f} (p={tan_corr['pval_sentiment_return_same']:.4f})")
    print(f"    SPY: {spy_corr['corr_sentiment_return_same']:+.3f} (p={spy_corr['pval_sentiment_return_same']:.4f})")
    if abs(tan_corr['corr_sentiment_return_same']) > abs(spy_corr['corr_sentiment_return_same']):
        print(f"    TAN shows stronger sentiment-return relationship")
    else:
        print(f"    SPY shows stronger sentiment-return relationship")

    # Sentiment-Volatility comparison (core H1 test)
    print(f"\n  Sentiment vs Realised Volatility (H1 test):")
    print(f"    TAN: {tan_corr['corr_sentiment_volatility']:+.3f} (p={tan_corr['pval_sentiment_volatility']:.4f})")
    print(f"    SPY: {spy_corr['corr_sentiment_volatility']:+.3f} (p={spy_corr['pval_sentiment_volatility']:.4f})")
    if abs(tan_corr['corr_sentiment_volatility']) > abs(spy_corr['corr_sentiment_volatility']):
        print(f"    TAN shows stronger sentiment-volatility relationship")
        print(f"    Result: Evidence supports H1")
    else:
        print(f"    SPY shows stronger sentiment-volatility relationship")
        print(f"    Result: H1 not supported by correlation evidence")

    # Lead-lag comparison (predictive power for H3)
    print(f"\n  Sentiment vs Next-day Return (lead-lag, relevant to H3):")
    print(f"    TAN: {tan_corr['corr_sentiment_return_next']:+.3f} (p={tan_corr['pval_sentiment_return_next']:.4f})")
    print(f"    SPY: {spy_corr['corr_sentiment_return_next']:+.3f} (p={spy_corr['pval_sentiment_return_next']:.4f})")
    if (tan_corr['pval_sentiment_return_next'] < 0.05
            or spy_corr['pval_sentiment_return_next'] < 0.05):
        print(f"    Statistically significant predictive relationship detected")
    else:
        print(f"    Predictive relationship is weak or statistically insignificant")


# ============================================
# SECTION 4: CORRELATION MATRIX
# ============================================

def create_correlation_matrix(ticker):
    """
    Construct a full pairwise correlation matrix across all key variables
    for a given ETF: sentiment score, daily return, trading volume, and
    realised volatility where available.

    The correlation matrix serves two purposes in the dissertation:
    1. It provides a complete overview of inter-variable relationships,
       allowing the examiner to assess multicollinearity and confounding.
    2. It is a standard table in empirical finance papers and demonstrates
       methodological rigour.

    Parameters
    ----------
    ticker : str
        ETF identifier ('TAN' or 'SPY').

    Returns
    -------
    pandas.DataFrame or None
        Square correlation matrix, also saved to CSV.
    """

    print(f"\n  Building correlation matrix for {ticker}...")

    try:
        df = pd.read_csv(f'results/{ticker}_merge_prices_news.csv')
        variables = ['sentiment_score', 'return', 'volume']

        # Append volatility column if the volatility file exists
        try:
            vol_df = pd.read_csv(f'results/{ticker}_volatility_analysis.csv')
            df = df.merge(vol_df[['date', 'volatility_10d']], on='date', how='left')
            variables.append('volatility_10d')
        except Exception:
            pass  # Proceed without volatility if unavailable

        corr_matrix = df[variables].corr()

        corr_matrix.to_csv(f'results/{ticker}_correlation_matrix.csv')
        print(f"    Saved: results/{ticker}_correlation_matrix.csv")

        return corr_matrix

    except Exception as e:
        print(f"    Could not create correlation matrix: {e}")
        return None


# ============================================
# SECTION 5: MAIN EXECUTION
# ============================================

def main():
    """
    Orchestrate the full correlation analysis pipeline for both ETFs,
    then produce a consolidated hypothesis testing summary.

    This function runs the four analytical stages in sequence:
    (1) correlation analysis, (2) regression analysis, (3) comparative
    analysis, and (4) correlation matrix construction. Results are saved
    to CSV and a printed summary maps each finding to the relevant
    hypothesis, providing a direct reference for the Results chapter.
    """

    print("\n" + "=" * 60)
    print("SENTIMENT-VOLATILITY CORRELATION ANALYSIS")
    print("=" * 60)
    print("\nFinal analytical step in the pipeline.")
    print("Addresses: H1 (differential impact), H2 (explanatory power),")
    print("           H3 (trading effectiveness), Objective 3 (correlation).")

    # -----------------------------------------------------------------------
    # TAN ANALYSIS
    # -----------------------------------------------------------------------
    tan_df = pd.read_csv('results/TAN_merge_prices_news.csv')
    tan_corr = calculate_correlations(tan_df, 'TAN')
    tan_reg = regression_analysis(tan_df, 'TAN')
    tan_matrix = create_correlation_matrix('TAN')

    # -----------------------------------------------------------------------
    # SPY ANALYSIS
    # -----------------------------------------------------------------------
    spy_df = pd.read_csv('results/SPY_merge_prices_news.csv')
    spy_corr = calculate_correlations(spy_df, 'SPY')
    spy_reg = regression_analysis(spy_df, 'SPY')
    spy_matrix = create_correlation_matrix('SPY')

    # -----------------------------------------------------------------------
    # COMPARATIVE ANALYSIS (H1)
    # -----------------------------------------------------------------------
    compare_correlations(tan_corr, spy_corr)

    # -----------------------------------------------------------------------
    # SAVE OUTPUTS
    # -----------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    pd.DataFrame([tan_corr, spy_corr]).to_csv(
        'results/correlation_summary.csv', index=False
    )
    print("  Saved: results/correlation_summary.csv")

    pd.DataFrame([tan_reg, spy_reg]).to_csv(
        'results/regression_summary.csv', index=False
    )
    print("  Saved: results/regression_summary.csv")

    # -----------------------------------------------------------------------
    # HYPOTHESIS TESTING SUMMARY
    # A structured summary mapping statistical evidence to each hypothesis.
    # These printed results can be directly referenced when writing the
    # Results chapter.
    # -----------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("HYPOTHESIS TESTING SUMMARY")
    print("=" * 60)

    # H1: TAN more sentiment-sensitive than SPY
    print("\n  H1: Sentiment impacts TAN volatility more than SPY")
    tan_sv = abs(tan_corr['corr_sentiment_volatility'])
    spy_sv = abs(spy_corr['corr_sentiment_volatility'])
    if tan_sv > spy_sv:
        print(f"    Result:    SUPPORTED")
        print(f"    TAN corr:  {tan_corr['corr_sentiment_volatility']:+.3f}")
        print(f"    SPY corr:  {spy_corr['corr_sentiment_volatility']:+.3f}")
        print(f"    Magnitude difference: {tan_sv - spy_sv:.3f}")
    else:
        print(f"    Result:    NOT SUPPORTED by correlation evidence")

    # H2: Sentiment improves volatility modelling
    print("\n  H2: Sentiment improves volatility modelling")
    if (tan_corr['pval_sentiment_volatility'] < 0.05
            or spy_corr['pval_sentiment_volatility'] < 0.05):
        print(f"    Result:    SUPPORTED")
        print(f"    TAN p-value: {tan_corr['pval_sentiment_volatility']:.4f}")
        print(f"    SPY p-value: {spy_corr['pval_sentiment_volatility']:.4f}")
        print(f"    Statistically significant sentiment-volatility link detected")
    else:
        print(f"    Result:    WEAK SUPPORT")
        print(f"    Correlations present but below the 5% significance threshold")

    # H3: Sentiment strategy vs buy-and-hold
    print("\n  H3: Sentiment signals can outperform passive strategies")
    print(f"    Result: See performance_metrics from calculate_metrics.py")
    print(f"    Key metrics: Sharpe ratio, cumulative return, max drawdown")

    # -----------------------------------------------------------------------
    # STATISTICAL SIGNIFICANCE TABLE
    # Significance codes follow the standard academic convention:
    # *** p < 0.01, ** p < 0.05, * p < 0.1, ns = not significant
    # -----------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("SIGNIFICANCE SUMMARY (*** p<0.01, ** p<0.05, * p<0.1, ns)")
    print("=" * 60)

    for ticker, corr in [('TAN', tan_corr), ('SPY', spy_corr)]:
        print(f"\n  {ticker}:")
        print(f"    Sentiment vs Return:     {corr['corr_sentiment_return_same']:+.3f} "
              f"{'***' if corr['pval_sentiment_return_same'] < 0.01 else '**' if corr['pval_sentiment_return_same'] < 0.05 else '*' if corr['pval_sentiment_return_same'] < 0.1 else 'ns'}")
        print(f"    Sentiment vs Volatility: {corr['corr_sentiment_volatility']:+.3f} "
              f"{'***' if corr['pval_sentiment_volatility'] < 0.01 else '**' if corr['pval_sentiment_volatility'] < 0.05 else '*' if corr['pval_sentiment_volatility'] < 0.1 else 'ns'}")

    print("\n" + "=" * 60)
    print("ANALYSIS PIPELINE COMPLETE")
    print("=" * 60)
    print("\nAll objectives addressed. Results ready for dissertation chapters.")


# ============================================
# SECTION 6: ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()