"""
================================================================================
Script 8: Volatility Modelling with GARCH
================================================================================
Purpose:
    Quantifies and models the volatility dynamics of TAN and SPY returns using
    two complementary approaches: realized volatility (rolling standard
    deviation) and GARCH(1,1) conditional volatility. The script also measures
    the correlation between sentiment scores and volatility, providing the
    empirical basis for evaluating Hypotheses 1 and 2.

Academic Context:
    This script directly addresses Dissertation Objective 2 — modelling and
    comparing ETF volatility using realized volatility and GARCH(1,1). It also
    contributes evidence toward:

    H1 (Differential Sentiment Impact): If TAN exhibits a stronger
    sentiment-volatility correlation than SPY, this supports the proposition
    that sector-specific ETFs are more sensitive to news sentiment than
    diversified market indices.

    H2 (Sentiment Improves Volatility Forecasting): A statistically meaningful
    correlation between sentiment scores and realized volatility suggests that
    sentiment carries explanatory power beyond what price history alone provides.

Why Two Volatility Measures:
    Realized volatility (rolling standard deviation) is simple, transparent,
    and widely used in empirical finance as a benchmark measure. However, it
    treats all days within the rolling window equally and cannot capture the
    well-documented tendency for high-volatility periods to cluster together.

    GARCH(1,1) addresses this limitation by modelling conditional
    heteroskedasticity — the phenomenon where volatility today is partially
    determined by volatility yesterday. This is expressed as:

        sigma^2_t = omega + alpha * epsilon^2_{t-1} + beta * sigma^2_{t-1}

    Where omega is the long-run variance, alpha captures the immediate impact
    of past shocks (the ARCH effect), and beta captures the persistence of
    prior volatility (the GARCH effect). The sum alpha + beta, known as the
    persistence parameter, indicates how quickly volatility reverts to its
    long-run mean after a shock. A value close to 1.0 signals slow mean
    reversion; a value above 1.0 indicates non-stationary volatility.

Inputs:
    - results/TAN_merge_prices_news.csv  : Merged price and sentiment data
    - results/SPY_merge_prices_news.csv  : Merged price and sentiment data

Outputs:
    - results/TAN_volatility_analysis.csv  : TAN volatility data with GARCH
    - results/SPY_volatility_analysis.csv  : SPY volatility data with GARCH
    - results/volatility_comparison.csv    : Side-by-side comparative summary

Pipeline Position:
    Script 8 of 10. Receives merged data from Script 4 (merge_prices_news.py).
    Output feeds into Script 9 (correlation_analysis.py) and Script 10
    (hypothesis_results.py).

Dependencies:
    - pandas   : Data manipulation and CSV handling
    - numpy    : Numerical operations and annualisation scaling
    - arch     : GARCH model estimation (arch_model)
    - warnings : Suppresses convergence warnings from the arch library

Usage:
    python volitility_calculation.py
================================================================================
"""

import pandas as pd
import numpy as np
from arch import arch_model
import warnings

# Suppress convergence and optimisation warnings from the arch library.
# These do not affect the validity of results but clutter the console output.
warnings.filterwarnings('ignore')


# ============================================
# SECTION 1: REALIZED VOLATILITY
# ============================================

def calculate_realized_volatility(df, window=10):
    """
    Calculate annualised realized volatility using a rolling standard deviation.

    Realized volatility is the most straightforward measure of historical
    price variability. It computes the standard deviation of daily log returns
    over a rolling window, then annualises the result by scaling by the
    square root of 252 (the approximate number of trading days in a year).
    This scaling converts the daily standard deviation into an annualised
    percentage figure, which is the standard convention in financial reporting.

    A 10-day window is used as the primary measure, capturing short-term
    volatility dynamics relevant to the 5-day trading signals generated in
    Script 5. A 20-day window is also computed as a secondary measure to
    assess whether findings are robust across different window lengths.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a 'return' column of daily percentage returns.

    window : int, optional (default=10)
        The number of trading days over which to compute the rolling
        standard deviation.

    Returns
    -------
    pandas.Series
        Annualised rolling volatility as a percentage series.
    """

    # Multiply by sqrt(252) to annualise the daily standard deviation.
    # This is the standard convention in financial volatility reporting.
    rolling_vol = df['return'].rolling(window=window).std() * np.sqrt(252)

    return rolling_vol


# ============================================
# SECTION 2: GARCH MODEL FITTING
# ============================================

def fit_garch_model(returns, model_type='GARCH', p=1, q=1):
    """
    Fit a GARCH(p,q) model to a series of daily returns.

    GARCH (Generalised Autoregressive Conditional Heteroskedasticity) models
    are the standard tool in quantitative finance for modelling time-varying
    volatility. Unlike realized volatility, which looks backward over a fixed
    window, GARCH produces a conditional volatility estimate for each period
    that accounts for both recent shocks and the persistence of prior
    volatility states.

    The GARCH(1,1) specification used here is the most widely adopted in the
    academic literature due to its parsimony and strong empirical performance
    across a broad range of asset classes. It requires only three parameters
    (omega, alpha, beta) and typically fits financial return series well.

    Returns are divided by 100 before fitting because the arch library expects
    returns in decimal form rather than percentage form.

    Parameters
    ----------
    returns : pandas.Series
        Series of daily percentage returns.

    model_type : str, optional (default='GARCH')
        The volatility model specification. 'GARCH' is used here. Alternative
        specifications such as 'EGARCH' or 'GJR-GARCH' could capture
        asymmetric effects (leverage effects) but are not required for the
        baseline analysis.

    p : int, optional (default=1)
        The GARCH lag order. p=1 means one lag of conditional variance.

    q : int, optional (default=1)
        The ARCH lag order. q=1 means one lag of squared residuals.

    Returns
    -------
    arch model result object or None
        The fitted model result containing parameters, conditional volatility,
        and information criteria. Returns None if fitting fails.
    """

    try:
        # Convert from percentage returns to decimal form as required by the
        # arch library. Failure to do this would produce inflated parameter
        # estimates and an invalid conditional volatility series.
        returns_clean = returns.dropna() / 100

        # Initialise and fit the GARCH(1,1) model. disp='off' suppresses
        # the optimisation iteration output for cleaner console reporting.
        model = arch_model(returns_clean, vol=model_type, p=p, q=q)
        result = model.fit(disp='off', show_warning=False)

        return result

    except Exception as e:
        print(f"    GARCH fitting failed: {e}")
        return None


# ============================================
# SECTION 3: FULL VOLATILITY ANALYSIS PER TICKER
# ============================================

def analyze_volatility(ticker):
    """
    Run the complete volatility analysis pipeline for a single ETF.

    This function orchestrates four analytical stages: realized volatility
    calculation, GARCH(1,1) model fitting, sentiment-volatility correlation
    measurement, and volatility clustering assessment. Together these stages
    provide the empirical evidence needed to evaluate H1 and H2.

    Parameters
    ----------
    ticker : str
        The ETF ticker symbol ('TAN' or 'SPY'). Used to locate the correct
        input file and label all output columns and files.

    Returns
    -------
    tuple : (dict, pandas.DataFrame)
        A summary dictionary of key metrics and the full augmented DataFrame
        with volatility columns appended.
    """

    print(f"\n{'=' * 60}")
    print(f"{ticker} - VOLATILITY ANALYSIS")
    print(f"{'=' * 60}")

    # Load the merged price and sentiment data produced by Script 4.
    # This file contains daily returns and daily sentiment scores aligned
    # on a common date index.
    df = pd.read_csv(f'results/{ticker}_merge_prices_news.csv')
    print(f"  Loaded {len(df)} trading days")

    # ------------------------------------------------------------------
    # STAGE 1: REALIZED VOLATILITY
    # Two window lengths are computed to assess robustness. The 10-day
    # window aligns with the short-term trading horizon of the signals
    # in Script 5. The 20-day window (~1 calendar month) captures a
    # broader view of the volatility environment.
    # ------------------------------------------------------------------

    print(f"\n  Calculating Realized Volatility...")

    df['volatility_10d'] = calculate_realized_volatility(df, window=10)
    df['volatility_20d'] = calculate_realized_volatility(df, window=20)

    # Drop rows where the rolling window has not yet been filled.
    # These NaN rows cannot contribute to correlation analysis.
    vol_stats = df.dropna()

    print(f"    10-day volatility:")
    print(f"      Mean:   {vol_stats['volatility_10d'].mean():.2f}%")
    print(f"      Min:    {vol_stats['volatility_10d'].min():.2f}%")
    print(f"      Max:    {vol_stats['volatility_10d'].max():.2f}%")
    print(f"      Median: {vol_stats['volatility_10d'].median():.2f}%")

    print(f"    20-day volatility:")
    print(f"      Mean:   {vol_stats['volatility_20d'].mean():.2f}%")
    print(f"      Std:    {vol_stats['volatility_20d'].std():.2f}%")

    # ------------------------------------------------------------------
    # STAGE 2: GARCH(1,1) MODEL
    # The GARCH model is fitted to the full return series to estimate
    # conditional volatility for every observation. The model parameters
    # are reported for dissertation transparency. Persistence (alpha +
    # beta) is the key diagnostic: values near 1.0 indicate that
    # volatility shocks are long-lasting, a finding common in equity
    # markets and consistent with the stylised facts of financial returns.
    # ------------------------------------------------------------------

    print(f"\n  Fitting GARCH(1,1) Model (Baseline)...")

    garch_result = fit_garch_model(df['return'])

    if garch_result is not None:
        print(f"    GARCH(1,1) fitted successfully")
        print(f"    Log-Likelihood: {garch_result.loglikelihood:.2f}")
        print(f"    AIC: {garch_result.aic:.2f}")
        print(f"    BIC: {garch_result.bic:.2f}")

        # Align the conditional volatility series back onto the original
        # DataFrame index. The arch library returns volatility in decimal
        # form, so it is rescaled to percentage and annualised.
        df['garch_volatility'] = np.nan
        df.loc[df['return'].notna(), 'garch_volatility'] = (
            garch_result.conditional_volatility * 100 * np.sqrt(252)
        )

        print(f"\n    GARCH Parameters:")
        print(f"      omega (long-run variance):  {garch_result.params['omega']:.6f}")
        print(f"      alpha[1] (ARCH effect):     {garch_result.params['alpha[1]']:.4f}")
        print(f"      beta[1]  (GARCH effect):    {garch_result.params['beta[1]']:.4f}")

        # Persistence = alpha + beta. A value below 1.0 confirms stationarity.
        # Values close to 1.0 suggest integrated GARCH (IGARCH) behaviour,
        # where the impact of past shocks decays very slowly.
        persistence = garch_result.params['alpha[1]'] + garch_result.params['beta[1]']
        print(f"      Persistence (alpha+beta):   {persistence:.4f}")

        if persistence >= 1.0:
            print(f"      Warning: Persistence >= 1.0 — non-stationary volatility process")
        else:
            print(f"      Persistence < 1.0 — stationary volatility process confirmed")

    # ------------------------------------------------------------------
    # STAGE 3: SENTIMENT-VOLATILITY CORRELATION
    # Pearson correlations are computed between the daily sentiment score
    # and three volatility proxies: 10-day realized vol, 20-day realized
    # vol, and absolute daily returns. Absolute returns serve as the
    # simplest possible proxy for daily volatility and act as a
    # robustness check. A fourth correlation measures whether the
    # magnitude of sentiment (regardless of direction) predicts vol,
    # testing the idea that extreme news — positive or negative — tends
    # to coincide with elevated market uncertainty.
    # ------------------------------------------------------------------

    print(f"\n  Sentiment-Volatility Relationship...")

    corr_10d = df['sentiment_score'].corr(df['volatility_10d'])
    corr_20d = df['sentiment_score'].corr(df['volatility_20d'])

    print(f"    Correlation (sentiment vs 10-day vol): {corr_10d:.3f}")
    print(f"    Correlation (sentiment vs 20-day vol): {corr_20d:.3f}")

    # Absolute returns are a simple same-day volatility proxy that does
    # not require a rolling window, preserving more observations.
    df['abs_return'] = df['return'].abs()
    corr_abs = df['sentiment_score'].corr(df['abs_return'])
    print(f"    Correlation (sentiment vs |returns|):  {corr_abs:.3f}")

    # Sentiment extremity captures the idea that both strongly positive
    # and strongly negative news events tend to increase market uncertainty,
    # regardless of the sign of the signal.
    df['sentiment_extremity'] = df['sentiment_score'].abs()
    corr_extremity = df['sentiment_extremity'].corr(df['volatility_10d'])
    print(f"    Correlation (|sentiment| vs vol):      {corr_extremity:.3f}")

    # ------------------------------------------------------------------
    # STAGE 4: VOLATILITY CLUSTERING TEST
    # Volatility clustering — the tendency for large price movements to
    # be followed by further large movements — is one of the most robust
    # stylised facts in financial econometrics (Mandelbrot, 1963; Engle,
    # 1982). It is tested here using the autocorrelation of squared
    # returns at lags 1 and 2. A positive autocorrelation confirms that
    # squared returns today predict squared returns tomorrow, validating
    # the use of GARCH modelling over simpler homoskedastic approaches.
    # ------------------------------------------------------------------

    print(f"\n  Volatility Clustering Analysis...")

    squared_returns = (df['return'] ** 2).dropna()

    if len(squared_returns) > 1:
        lag1_corr = squared_returns.autocorr(lag=1)
        lag2_corr = squared_returns.autocorr(lag=2)
        print(f"    Autocorrelation of squared returns (lag 1): {lag1_corr:.3f}")
        print(f"    Autocorrelation of squared returns (lag 2): {lag2_corr:.3f}")

        if lag1_corr > 0.1:
            print(f"    Evidence of volatility clustering detected — GARCH is appropriate")
        else:
            print(f"    Weak volatility clustering — GARCH assumption may be marginal")

    # ------------------------------------------------------------------
    # SUMMARY DICTIONARY
    # Collects all key metrics into a single dictionary for use in the
    # comparative analysis function and for export to CSV.
    # ------------------------------------------------------------------

    summary = {
        'ticker': ticker,
        'observations': len(df),

        # Realized volatility statistics
        'realized_vol_mean': vol_stats['volatility_10d'].mean(),
        'realized_vol_std': vol_stats['volatility_10d'].std(),
        'realized_vol_min': vol_stats['volatility_10d'].min(),
        'realized_vol_max': vol_stats['volatility_10d'].max(),

        # GARCH model parameters (NaN if model failed to converge)
        'garch_omega': garch_result.params['omega'] if garch_result else np.nan,
        'garch_alpha': garch_result.params['alpha[1]'] if garch_result else np.nan,
        'garch_beta': garch_result.params['beta[1]'] if garch_result else np.nan,
        'garch_persistence': persistence if garch_result else np.nan,
        'garch_loglik': garch_result.loglikelihood if garch_result else np.nan,
        'garch_aic': garch_result.aic if garch_result else np.nan,

        # Sentiment-volatility correlation measures
        'corr_sentiment_vol': corr_10d,
        'corr_sentiment_abs_return': corr_abs,
        'corr_extremity_vol': corr_extremity,

        # Volatility clustering diagnostic
        'squared_returns_autocorr_lag1': lag1_corr if len(squared_returns) > 1 else np.nan,
    }

    # Save the full augmented DataFrame with all new volatility columns.
    df.to_csv(f'results/{ticker}_volatility_analysis.csv', index=False)
    print(f"\n  Saved: results/{ticker}_volatility_analysis.csv")

    return summary, df


# ============================================
# SECTION 4: COMPARATIVE ANALYSIS (TAN vs SPY)
# ============================================

def compare_volatility(tan_summary, spy_summary):
    """
    Produce a side-by-side comparison of volatility characteristics for
    TAN and SPY, providing the evidence base for H1.

    The comparison examines three dimensions: the level of realized
    volatility (is TAN more volatile than SPY as expected for a niche
    sector ETF?), GARCH persistence (does TAN's volatility take longer
    to mean-revert after a shock?), and the sentiment-volatility
    correlation (is TAN's volatility more responsive to news sentiment?).
    A stronger sentiment-volatility relationship for TAN than SPY would
    constitute direct support for H1.

    Parameters
    ----------
    tan_summary : dict
        Summary metrics dictionary returned by analyze_volatility('TAN').

    spy_summary : dict
        Summary metrics dictionary returned by analyze_volatility('SPY').

    Returns
    -------
    pandas.DataFrame
        A two-row DataFrame containing the summary statistics for both
        ETFs, saved to results/volatility_comparison.csv.
    """

    print("\n" + "=" * 60)
    print("COMPARATIVE VOLATILITY ANALYSIS: TAN vs SPY")
    print("=" * 60)

    # --- Realized Volatility Level ---
    # TAN is expected to be more volatile than SPY given that it is
    # concentrated in a single sector (solar energy), which is subject
    # to regulatory, policy, and commodity price risks that affect the
    # whole fund simultaneously. SPY's diversification across 500 stocks
    # naturally dampens these idiosyncratic shocks.
    print(f"\n  Realized Volatility:")
    print(f"    TAN mean volatility: {tan_summary['realized_vol_mean']:.2f}%")
    print(f"    SPY mean volatility: {spy_summary['realized_vol_mean']:.2f}%")
    print(f"    Difference:          {tan_summary['realized_vol_mean'] - spy_summary['realized_vol_mean']:+.2f}%")

    if tan_summary['realized_vol_mean'] > spy_summary['realized_vol_mean']:
        ratio = tan_summary['realized_vol_mean'] / spy_summary['realized_vol_mean']
        print(f"    TAN is {ratio:.2f}x more volatile than SPY")

    # --- GARCH Persistence ---
    # Higher persistence in TAN would indicate that volatility shocks
    # in the solar sector take longer to dissipate, consistent with the
    # sector's sensitivity to sustained news cycles around energy policy.
    print(f"\n  GARCH Persistence (alpha + beta):")
    print(f"    TAN: {tan_summary['garch_persistence']:.4f}")
    print(f"    SPY: {spy_summary['garch_persistence']:.4f}")

    if tan_summary['garch_persistence'] > spy_summary['garch_persistence']:
        print(f"    TAN shows higher volatility persistence (slower mean reversion)")
    else:
        print(f"    SPY shows higher volatility persistence")

    # --- Sentiment-Volatility Correlation (Key H1 Test) ---
    # This is the most direct evidence for H1. A larger absolute correlation
    # for TAN indicates that its volatility is more tightly coupled to the
    # sentiment of news headlines — as expected for a concentrated ETF
    # where a small number of sector-specific news themes dominate.
    print(f"\n  Sentiment-Volatility Correlation (H1 Evidence):")
    print(f"    TAN: {tan_summary['corr_sentiment_vol']:.3f}")
    print(f"    SPY: {spy_summary['corr_sentiment_vol']:.3f}")

    if abs(tan_summary['corr_sentiment_vol']) > abs(spy_summary['corr_sentiment_vol']):
        print(f"    TAN shows stronger sentiment-volatility relationship")
        print(f"    Supports H1: Sector ETFs are more sensitive to news sentiment")
    else:
        print(f"    SPY shows stronger correlation — mixed support for H1")

    # Export the comparison table for use in dissertation results tables.
    comparison = pd.DataFrame([tan_summary, spy_summary])
    comparison.to_csv('results/volatility_comparison.csv', index=False)
    print(f"\n  Saved: results/volatility_comparison.csv")

    return comparison


# ============================================
# SECTION 5: MAIN EXECUTION
# ============================================

def main():
    """
    Execute the full volatility analysis for both ETFs and report key
    findings mapped to the dissertation hypotheses.
    """

    print("\n" + "=" * 60)
    print("VOLATILITY ANALYSIS WITH GARCH MODELLING")
    print("=" * 60)
    print("\nThis script addresses:")
    print("  H1: Does sentiment impact volatility differently for TAN vs SPY?")
    print("  H2: Does sentiment improve volatility modelling?")
    print("  Objective 2: Realized volatility and GARCH(1,1) modelling")

    # Run the full volatility analysis for each ETF independently.
    tan_summary, tan_df = analyze_volatility('TAN')
    spy_summary, spy_df = analyze_volatility('SPY')

    # Run the comparative analysis to generate H1 evidence.
    comparison = compare_volatility(tan_summary, spy_summary)

    # ------------------------------------------------------------------
    # KEY FINDINGS SUMMARY
    # Printed results are organised by hypothesis to make it
    # straightforward to transfer findings into the dissertation
    # results and discussion chapters.
    # ------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    print("\nVolatility Modelling Complete:")
    print("  Realized volatility calculated (10-day and 20-day windows)")
    print("  GARCH(1,1) models fitted for both ETFs")
    print("  Sentiment-volatility correlations computed")
    print("  Volatility clustering tested")

    # H1 findings
    print("\nFor H1 (Differential Sentiment Impact):")
    if abs(tan_summary['corr_sentiment_vol']) > abs(spy_summary['corr_sentiment_vol']):
        print(f"  TAN shows stronger sentiment-volatility correlation")
        print(f"    TAN: {tan_summary['corr_sentiment_vol']:.3f}")
        print(f"    SPY: {spy_summary['corr_sentiment_vol']:.3f}")
        print(f"  Supports H1: Niche ETFs more sentiment-sensitive")
    else:
        print(f"  SPY shows stronger correlation — mixed support for H1")

    # H2 findings
    print("\nFor H2 (Sentiment Improves Modelling):")
    print(f"  Sentiment-volatility correlations:")
    print(f"    TAN: {tan_summary['corr_sentiment_vol']:.3f}")
    print(f"    SPY: {spy_summary['corr_sentiment_vol']:.3f}")

    if abs(tan_summary['corr_sentiment_vol']) > 0.1:
        print(f"  Moderate correlation suggests sentiment has explanatory power")
        print(f"  Supports H2: Sentiment improves volatility modelling")
    else:
        print(f"  Weak correlation — limited support for H2")

    print("\nVolatility Characteristics:")
    print(f"  TAN average volatility: {tan_summary['realized_vol_mean']:.2f}%")
    print(f"  SPY average volatility: {spy_summary['realized_vol_mean']:.2f}%")

    if tan_summary['realized_vol_mean'] > spy_summary['realized_vol_mean'] * 1.2:
        print(f"  TAN is notably more volatile — consistent with niche ETF theory")

    print("\n" + "=" * 60)
    print("OUTPUT FILES")
    print("=" * 60)
    print("  results/TAN_volatility_analysis.csv  - TAN volatility data")
    print("  results/SPY_volatility_analysis.csv  - SPY volatility data")
    print("  results/volatility_comparison.csv    - Comparative summary")

    print("\nKey metrics for dissertation results section:")
    print(f"  GARCH persistence (TAN): {tan_summary['garch_persistence']:.4f}")
    print(f"  GARCH persistence (SPY): {spy_summary['garch_persistence']:.4f}")
    print(f"  Sentiment-vol correlation (TAN): {tan_summary['corr_sentiment_vol']:.3f}")
    print(f"  Sentiment-vol correlation (SPY): {spy_summary['corr_sentiment_vol']:.3f}")

    print("\nNext step: Run correlation_analysis.py")


# ============================================
# SECTION 6: ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()