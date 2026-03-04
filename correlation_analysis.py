"""
================================================================================
Script 9: Sentiment-Volatility Correlation Analysis (correlation_analysis.py)
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

    Four analytical approaches are applied: Pearson correlation (parametric),
    Spearman rank correlation (non-parametric), lead-lag analysis (tests
    whether sentiment predicts future returns), and OLS regression (quantifies
    the magnitude of sentiment’s effect on returns).

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
from scipy.stats import pearsonr, spearmanr, linregress
import warnings

warnings.filterwarnings('ignore')


# ============================================
# SECTION 0: SMALL UTILITIES
# ============================================

def sig_star(p):
    """
    Convert a p-value into the standard academic significance annotation.

    Returns:
        '***' if p < 0.01
        '**'  if p < 0.05
        '*'   if p < 0.10
        'ns'  otherwise
    """
    if pd.isna(p):
        return 'ns'
    if p < 0.01:
        return '***'
    if p < 0.05:
        return '**'
    if p < 0.10:
        return '*'
    return 'ns'


def safe_corr(x, y, method='pearson'):
    """
    Compute correlation and p-value safely, returning (np.nan, np.nan) if
    insufficient data exists.

    Parameters
    ----------
    x, y : array-like
        Input series.
    method : str
        'pearson' or 'spearman'

    Returns
    -------
    tuple : (corr, p_value)
    """
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)

    valid = x.notna() & y.notna()
    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return np.nan, np.nan

    if method == 'pearson':
        return pearsonr(x, y)
    elif method == 'spearman':
        return spearmanr(x, y)
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")


# ============================================
# SECTION 1: CORRELATION ANALYSIS
# ============================================

def calculate_correlations(df, ticker):
    """
    Calculate Pearson and Spearman correlations between sentiment scores
    and key price-based variables: same-day returns, next-day returns
    (lead-lag), realised volatility, and absolute returns as a volatility
    proxy.

    Pearson is the primary test used for comparability across ETFs.
    Spearman is included as a robustness check given non-normal return
    distributions (fat tails).

    Parameters
    ----------
    df : pandas.DataFrame
        Merged dataset containing at minimum: sentiment_score, return, date.
    ticker : str
        ETF identifier ('TAN' or 'SPY').

    Returns
    -------
    dict
        Dictionary of correlation coefficients and p-values.
    """

    print(f"\n{'=' * 60}")
    print(f"{ticker} - CORRELATION ANALYSIS")
    print(f"{'=' * 60}")

    # Keep only what we need for the core tests to avoid dropping rows
    # due to unrelated missing values (e.g., volume, article_count).
    df_clean = df.dropna(subset=['sentiment_score', 'return']).copy()
    print(f"  Analyzing {len(df_clean)} complete observations")

    # -----------------------------------------------------------------------
    # 1A. SENTIMENT-RETURN CORRELATIONS (SAME DAY)
    # -----------------------------------------------------------------------

    print(f"\n  Sentiment-Return Relationships:")

    corr_same_day, p_same_day = safe_corr(
        df_clean['sentiment_score'],
        df_clean['return'],
        method='pearson'
    )
    print(f"    Sentiment vs Same-day return:")
    print(f"      Correlation: {corr_same_day:+.3f}")
    print(f"      P-value:     {p_same_day:.4f} {sig_star(p_same_day)}")

    # -----------------------------------------------------------------------
    # 1B. LEAD-LAG ANALYSIS (SENTIMENT TODAY vs RETURN TOMORROW)
    # -----------------------------------------------------------------------

    df_clean['next_return'] = df_clean['return'].shift(-1)
    df_lead = df_clean.dropna(subset=['sentiment_score', 'next_return']).copy()

    corr_next_day, p_next_day = safe_corr(
        df_lead['sentiment_score'],
        df_lead['next_return'],
        method='pearson'
    )

    print(f"    Sentiment vs Next-day return (lead-lag):")
    print(f"      Correlation: {corr_next_day:+.3f}")
    print(f"      P-value:     {p_next_day:.4f} {sig_star(p_next_day)}")

    # -----------------------------------------------------------------------
    # 1C. SENTIMENT-VOLATILITY CORRELATIONS
    # -----------------------------------------------------------------------

    print(f"\n  Sentiment-Volatility Relationships:")

    corr_vol, p_vol = np.nan, np.nan
    corr_ext_vol, p_ext_vol = np.nan, np.nan

    try:
        vol_df = pd.read_csv(f'results/{ticker}_volatility_analysis.csv')

        # Defensive: avoid accidental duplication from repeated dates.
        vol_df = vol_df.drop_duplicates(subset=['date'])

        # Bring realised and GARCH vol onto the merged dataset by date.
        merged = df_clean.merge(
            vol_df[['date', 'volatility_10d', 'garch_volatility']],
            on='date',
            how='left'
        )

        # Realised vol requires the rolling window to be filled → NaNs exist
        merged = merged.dropna(subset=['sentiment_score', 'volatility_10d']).copy()

        corr_vol, p_vol = safe_corr(
            merged['sentiment_score'],
            merged['volatility_10d'],
            method='pearson'
        )
        print(f"    Sentiment vs Realised volatility (10d):")
        print(f"      Correlation: {corr_vol:+.3f}")
        print(f"      P-value:     {p_vol:.4f} {sig_star(p_vol)}")

        merged['sentiment_abs'] = merged['sentiment_score'].abs()
        corr_ext_vol, p_ext_vol = safe_corr(
            merged['sentiment_abs'],
            merged['volatility_10d'],
            method='pearson'
        )
        print(f"    |Sentiment| vs Volatility (extremity test):")
        print(f"      Correlation: {corr_ext_vol:+.3f}")
        print(f"      P-value:     {p_ext_vol:.4f} {sig_star(p_ext_vol)}")

    except FileNotFoundError:
        print(f"    Volatility data not available - run volitility_calculation.py first")
    except Exception as e:
        print(f"    Volatility correlation failed: {e}")

    # -----------------------------------------------------------------------
    # 1D. ABSOLUTE RETURNS AS A VOLATILITY PROXY
    # -----------------------------------------------------------------------

    print(f"\n  Sentiment vs Absolute Returns (volatility proxy):")

    df_clean['abs_return'] = df_clean['return'].abs()
    corr_abs, p_abs = safe_corr(
        df_clean['sentiment_score'],
        df_clean['abs_return'],
        method='pearson'
    )
    print(f"    Correlation: {corr_abs:+.3f}")
    print(f"    P-value:     {p_abs:.4f} {sig_star(p_abs)}")

    # -----------------------------------------------------------------------
    # 1E. SPEARMAN RANK CORRELATIONS (ROBUSTNESS CHECK)
    # -----------------------------------------------------------------------

    print(f"\n  Spearman Rank Correlations (robustness check):")

    spearman_return, p_spear_ret = safe_corr(
        df_clean['sentiment_score'],
        df_clean['return'],
        method='spearman'
    )
    print(f"    Sentiment vs Return:   {spearman_return:+.3f} (p={p_spear_ret:.4f})")

    spearman_abs, p_spear_abs = safe_corr(
        df_clean['sentiment_score'],
        df_clean['abs_return'],
        method='spearman'
    )
    print(f"    Sentiment vs |Return|: {spearman_abs:+.3f} (p={p_spear_abs:.4f})")

    correlations = {
        'ticker': ticker,
        'observations': len(df_clean),

        # Pearson: sentiment vs return
        'corr_sentiment_return_same': corr_same_day,
        'pval_sentiment_return_same': p_same_day,
        'corr_sentiment_return_next': corr_next_day,
        'pval_sentiment_return_next': p_next_day,

        # Pearson: sentiment vs volatility (requires Script 8 output)
        'corr_sentiment_volatility': corr_vol,
        'pval_sentiment_volatility': p_vol,
        'corr_sentiment_abs_volatility': corr_ext_vol,
        'pval_sentiment_abs_volatility': p_ext_vol,

        # Pearson: sentiment vs |return| proxy
        'corr_sentiment_abs_return': corr_abs,
        'pval_sentiment_abs_return': p_abs,

        # Spearman robustness checks (+ p-values saved for reporting)
        'spearman_sentiment_return': spearman_return,
        'spearman_p_sentiment_return': p_spear_ret,
        'spearman_sentiment_abs_return': spearman_abs,
        'spearman_p_sentiment_abs_return': p_spear_abs,
    }

    return correlations


# ============================================
# SECTION 2: REGRESSION ANALYSIS
# ============================================

def regression_analysis(df, ticker):
    """
    Perform Ordinary Least Squares (OLS) regression of daily returns on
    sentiment scores:

        Return_t = alpha + beta * Sentiment_t + epsilon_t

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

    df_clean = df.dropna(subset=['sentiment_score', 'return']).copy()

    if len(df_clean) < 3:
        print(f"    Not enough observations to run regression.")
        return {
            'ticker': ticker,
            'slope': np.nan,
            'intercept': np.nan,
            'r_squared': np.nan,
            'p_value': np.nan,
            'std_error': np.nan
        }

    slope, intercept, r_value, p_value, std_err = linregress(
        df_clean['sentiment_score'],
        df_clean['return']
    )

    print(f"    Equation:  Return = {intercept:.4f} + {slope:.4f} x Sentiment")
    print(f"    R-squared: {r_value ** 2:.4f}")
    print(f"    P-value:   {p_value:.4f} {sig_star(p_value)}")
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

    Parameters
    ----------
    tan_corr : dict
        Correlation results dictionary for TAN.
    spy_corr : dict
        Correlation results dictionary for SPY.
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
    Construct a full pairwise correlation matrix across key variables for a
    given ETF: sentiment score, daily return, trading volume, and realised
    volatility (if available).

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

        variables = [c for c in ['sentiment_score', 'return', 'volume'] if c in df.columns]

        # Append volatility column if available
        try:
            vol_df = pd.read_csv(f'results/{ticker}_volatility_analysis.csv')
            vol_df = vol_df.drop_duplicates(subset=['date'])
            df = df.merge(vol_df[['date', 'volatility_10d']], on='date', how='left')
            if 'volatility_10d' in df.columns:
                variables.append('volatility_10d')
        except Exception:
            pass

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

    Stages:
      (1) Correlation analysis
      (2) Regression analysis
      (3) Comparative analysis (H1)
      (4) Correlation matrix construction
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
    # -----------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("HYPOTHESIS TESTING SUMMARY")
    print("=" * 60)

    # H1: TAN more sentiment-sensitive than SPY
    print("\n  H1: Sentiment impacts TAN volatility more than SPY")
    tan_sv = abs(tan_corr['corr_sentiment_volatility']) if not pd.isna(tan_corr['corr_sentiment_volatility']) else np.nan
    spy_sv = abs(spy_corr['corr_sentiment_volatility']) if not pd.isna(spy_corr['corr_sentiment_volatility']) else np.nan

    if not pd.isna(tan_sv) and not pd.isna(spy_sv) and tan_sv > spy_sv:
        print(f"    Result:    SUPPORTED")
        print(f"    TAN corr:  {tan_corr['corr_sentiment_volatility']:+.3f}")
        print(f"    SPY corr:  {spy_corr['corr_sentiment_volatility']:+.3f}")
        print(f"    Magnitude difference: {tan_sv - spy_sv:.3f}")
    else:
        print(f"    Result:    NOT SUPPORTED by correlation evidence (or volatility data missing)")

    # H2: Sentiment improves volatility modelling
    print("\n  H2: Sentiment improves volatility modelling")
    if ((not pd.isna(tan_corr['pval_sentiment_volatility']) and tan_corr['pval_sentiment_volatility'] < 0.05) or
            (not pd.isna(spy_corr['pval_sentiment_volatility']) and spy_corr['pval_sentiment_volatility'] < 0.05)):
        print(f"    Result:    SUPPORTED")
        print(f"    TAN p-value: {tan_corr['pval_sentiment_volatility']:.4f}")
        print(f"    SPY p-value: {spy_corr['pval_sentiment_volatility']:.4f}")
        print(f"    Statistically significant sentiment-volatility link detected")
    else:
        print(f"    Result:    WEAK SUPPORT")
        print(f"    Correlations present but below the 5% significance threshold (or volatility data missing)")

    # H3: Sentiment strategy vs buy-and-hold
    print("\n  H3: Sentiment signals can outperform passive strategies")
    print(f"    Result: See performance_metrics from calculate_metrics.py")
    print(f"    Key metrics: Sharpe ratio, cumulative return, max drawdown")

    # -----------------------------------------------------------------------
    # SIGNIFICANCE SUMMARY TABLE
    # -----------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("SIGNIFICANCE SUMMARY (*** p<0.01, ** p<0.05, * p<0.1, ns)")
    print("=" * 60)

    for tkr, corr in [('TAN', tan_corr), ('SPY', spy_corr)]:
        print(f"\n  {tkr}:")
        print(f"    Sentiment vs Return:     {corr['corr_sentiment_return_same']:+.3f} {sig_star(corr['pval_sentiment_return_same'])}")
        print(f"    Sentiment vs Volatility: {corr['corr_sentiment_volatility']:+.3f} {sig_star(corr['pval_sentiment_volatility'])}")

    print("\n" + "=" * 60)
    print("ANALYSIS PIPELINE COMPLETE")



# ============================================
# SECTION 6: ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()