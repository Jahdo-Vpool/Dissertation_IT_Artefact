# Computer Science Capstone Project Artefact

**Title**: A Comparative Analysis of News Sentiment's Impact on Market Volatility and Its Implications for Trading Strategy

**Author**: Jahdo A. Vanterpool  
**Programme**: MSc Data Science & Artificial Intelligence  
**University**: University of Liverpool  
**Supervisor**: Dr. Andrea Corrandini  
**Date**: August 2025

---

## Project Overview

This dissertation investigates the relationship between financial news sentiment
and the daily price volatility of two exchange-traded funds (ETFs): the Invesco
Solar ETF (TAN), representing a niche renewable energy sector, and the SPDR
S&P 500 ETF (SPY), representing the broader market.

The artefact demonstrates how Natural Language Processing (NLP), econometric
modelling, and quantitative backtesting can be combined into a **reproducible,
end-to-end pipeline** to assess sentiment-volatility interactions and evaluate
sentiment-driven trading strategies.

---

## Dissertation Objectives

1. **Extract and quantify** financial sentiment from news articles using VADER and FinBERT models
2. **Model and compare** ETF volatility using realized volatility and GARCH(1,1)
3. **Evaluate** the correlation and predictive relationship between sentiment and volatility
4. **Design and backtest** rules-based trading strategies informed by sentiment signals
5. **Compare outcomes** between TAN (sector-specific) and SPY (broad market)

---

## Research Hypotheses

| # | Hypothesis | Description |
|---|-----------|-------------|
| H1 | Differential Impact | Sentiment has a stronger influence on TAN than SPY due to TAN's sector concentration |
| H2 | Modelling Power | News sentiment carries statistically significant explanatory power for ETF volatility |
| H3 | Trading Performance | A sentiment-based strategy generates superior risk-adjusted returns vs buy-and-hold |

---

## Pipeline Architecture

The artefact is structured as a sequential 10-script pipeline. Each script
produces output files that feed into the next stage. Scripts must be run
in the order listed below.

```
Script 1  →  Script 2  →  Script 3  →       Script 4  →     Script 5
  ↓              ↓              ↓              ↓              ↓
Collect        Score          Process        Merge          Generate
News           Sentiment      Prices         Datasets       Signals
  
Script 5  →  Script 6  →  Script 7  →       Script 8  →      Script 9  →     Script 10
  ↓              ↓              ↓              ↓                ↓                 ↓
Generate       Evaluate       Visualise      Model          Correlation       Hypothesis
Signals        Performance    Results        Volatility     Analysis          Results
```

| Script | File                        | Purpose |
|--------|-----------------------------|---------|
| 1 | `collect_news_gdelt.py`     | Collect financial news articles via GDELT API |
| 2 | `analyze_sentiment.py`      | Score articles with VADER and FinBERT |
| 3 | `process_prices_csv.py`     | Clean and process Yahoo Finance price data |
| 4 | `merge_prices_news.py`      | Align sentiment and price data on trading calendar |
| 5 | `trading_signals.py`        | Generate BUY/SELL/HOLD signals from sentiment |
| 6 | `calculate_metrics.py`      | Evaluate strategy performance vs buy-and-hold |
| 7 | `plots.py`                  | Produce all dissertation figures at 300 DPI |
| 8 | `volatility_calculation.py` | Fit GARCH(1,1) and compute realized volatility |
| 9 | `correlation_analysis.py`   | Statistical correlation and regression analysis |
| 10 | `hypothesis_results.py`     | Consolidate evidence and report hypothesis outcomes |

---

## Project Structure

```
project/
│
├── data/                          # Raw input files (not modified by any script)
│   ├── TAN.csv                    # Yahoo Finance price data for TAN (manual download)
│   └── SPY.csv                    # Yahoo Finance price data for SPY (manual download)
│
├── results/                       # All generated output CSV files
│   ├── TAN_news.csv               # Raw collected news articles (Script 1)
│   ├── SPY_news.csv               # Raw collected news articles (Script 1)
│   ├── TAN_with_sentiment.csv     # Article-level sentiment scores (Script 2)
│   ├── SPY_with_sentiment.csv     # Article-level sentiment scores (Script 2)
│   ├── TAN_prices_processed.csv   # Cleaned price data with daily returns (Script 3)
│   ├── SPY_prices_processed.csv   # Cleaned price data with daily returns (Script 3)
│   ├── TAN_merge_prices_news.csv  # Merged sentiment and price data (Script 4)
│   ├── SPY_merge_prices_news.csv  # Merged sentiment and price data (Script 4)
│   ├── TAN_trading_signals.csv    # Signals and strategy returns (Script 5)
│   ├── SPY_trading_signals.csv    # Signals and strategy returns (Script 5)
│   ├── performance_metrics_detailed.csv  # Full metric set (Script 6)
│   ├── performance_summary.csv    # Formatted comparison table (Script 6)
│   ├── TAN_volatility_analysis.csv # GARCH and realized volatility (Script 8)
│   ├── SPY_volatility_analysis.csv # GARCH and realized volatility (Script 8)
│   ├── volatility_comparison.csv  # TAN vs SPY volatility summary (Script 8)
│   ├── correlation_summary.csv    # All Pearson/Spearman results (Script 9)
│   ├── regression_summary.csv     # OLS regression coefficients (Script 9)
│   ├── TAN_correlation_matrix.csv # Full variable correlation matrix (Script 9)
│   ├── SPY_correlation_matrix.csv # Full variable correlation matrix (Script 9)
│   └── hypothesis_testing_summary.csv  # Final hypothesis outcomes (Script 10)
│
├── plots/                         # All dissertation figures (Script 7)
│   ├── SPY_cumulative_returns.png
│   ├── SPY_sentiment_returns_scatter.png
│   ├── SPY_trading_signals.png
│   ├── SPY_drawdown_comparison.png
│   ├── SPY_sentiment_distribution.png
│   ├── TAN_cumulative_returns.png
│   ├── TAN_sentiment_returns_scatter.png
│   ├── TAN_trading_signals.png
│   ├── TAN_drawdown_comparison.png
│   ├── TAN_sentiment_distribution.png
│   └── TAN_vs_SPY_comparison.png
│
├── collect_news_gdelt.py
├── analyze_sentiment.py
├── process_prices_csv.py
├── merge_prices_news.py
├── trading_signals.py
├── calculate_metrics.py
├── plots.py
├── volatility_calculation.py
├── correlation_analysis.py
├── hypothesis_results.py
└── README.md
```

---

## Installation

### Requirements

- Python 3.9 or higher
- pip

### Install Dependencies

```bash
pip install pandas numpy scipy matplotlib seaborn transformers torch
pip install arch gdeltdoc requests warnings
```

Or install from a requirements file if provided:

```bash
pip install -r requirements.txt
```

### Price Data Setup (Manual Step Required Before Running)

Scripts 3 and 4 require manually downloaded Yahoo Finance CSV files. This
is intentional — manual download ensures the dataset is fixed and auditable,
as programmatic downloads can return retroactively adjusted values.

1. Go to [finance.yahoo.com](https://finance.yahoo.com)
2. Search for **TAN** and download historical data covering your analysis period
3. Search for **SPY** and download historical data covering the same period
4. Save both files to the `data/` directory as `TAN.csv` and `SPY.csv`

---

## Usage

Run each script in order from the project root directory. Each script prints
a structured summary to the console on completion and saves its outputs to
the `results/` or `plots/` directories.

```bash
# Step 1 — Collect news articles from GDELT
python collect_news_gdelt.py

# Step 2 — Score articles with VADER and FinBERT
python analyze_sentiment.py

# Step 3 — Process downloaded Yahoo Finance price data
python process_prices_csv.py

# Step 4 — Merge sentiment scores with price data
python merge_prices_news.py

# Step 5 — Generate trading signals from sentiment
python trading_signals.py

# Step 6 — Evaluate strategy performance
python calculate_metrics.py

# Step 7 — Produce all dissertation figures
python plots.py

# Step 8 — Fit GARCH model and compute volatility
python volatility_calculation.py

# Step 9 — Run correlation and regression analysis
python correlation_analysis.py

# Step 10 — Extract and report hypothesis outcomes
python hypothesis_results.py
```

> **Note:** Scripts 7, 8, and 9 can be run independently of each other
> once Scripts 1–6 have completed, as they each draw from the same merged
> dataset. Script 10 requires all preceding scripts to have completed.

---

## Sentiment Models

Two independent sentiment models are applied in Script 2 and their scores
are averaged to produce the combined sentiment signal used throughout the
pipeline.

**VADER** (Valence Aware Dictionary and sEntiment Reasoner)  
A rule-based lexicon model developed by Hutto and Gilbert (2014). Produces
a compound score in the range [-1, +1]. Computationally efficient and
highly interpretable. Best suited for general text but less sensitive to
domain-specific financial language.

**FinBERT** (ProsusAI/finbert)  
A BERT-based transformer model fine-tuned on financial news, earnings call
transcripts, and analyst reports (Araci, 2019). Available via Hugging Face.
Produces class probabilities for positive, negative, and neutral labels.
Scores are derived as (positive probability − negative probability) to
produce a [-1, +1] comparable scale.

---

## Trading Strategy

The strategy implemented in Script 5 is **rules-based and long-only**.

| Condition | Signal | Position |
|-----------|--------|----------|
| Sentiment ≥ +0.05 | BUY | Enter / maintain long position |
| Sentiment ≤ −0.05 | SELL | Exit to cash |
| −0.05 < Sentiment < +0.05 | HOLD | Maintain current position |

The ±0.05 neutral buffer filters out near-zero sentiment readings that are
more likely to represent noise than a genuine directional signal. The HOLD
signal is stateful — whether a neutral day results in a long or cash position
depends on the prior day's position.

---

## Key Design Decisions

- **Dual sentiment models**: Reduces risk of findings being an artefact of one model's characteristics
- **Forward-fill imputation**: Weekend and holiday sentiment propagated to next trading day (Script 4)
- **Non-destructive processing**: Raw data files are never modified; all outputs written to new files (Script 3)
- **Manual price download**: Ensures reproducibility by fixing the exact dataset used (Script 3)
- **GARCH(1,1) specification**: Chosen for parsimony, empirical performance, and interpretability (Script 8)
- **Four performance metrics**: Sharpe, Sortino, Calmar, and maximum drawdown to avoid single-metric bias (Script 6)
- **Three-way H3 classification**: Captures Mixed outcomes where risk-adjusted returns improve but absolute returns do not (Script 10)

---

## Acknowledged Limitations

- **GDELT article volume**: The GDELT free API returns a maximum of 250 articles per query call and has a rolling 3-month window. TAN's niche coverage results in a smaller corpus than SPY. Periods with low article coverage rely on forward-fill imputation.
- **Headline-only text**: GDELT does not provide full article body text at the free tier. Sentiment is scored from headlines, which are a recognised but incomplete proxy for full article sentiment.
- **No transaction costs**: The backtest assumes zero transaction costs and no slippage, which is standard in academic backtesting but would reduce real-world returns.
- **Long-only strategy**: Short-selling is excluded to reflect retail investor constraints. A long-short variant would likely produce different absolute return profiles.
- **Univariate regression**: OLS regression uses sentiment as the sole predictor. Returns are influenced by many factors; low R² values are expected and should be interpreted accordingly.
- **Zero risk-free rate**: Sharpe and Sortino ratios assume a 0% risk-free rate, which is a standard academic simplification.

---

## References

**Behavioural Finance and Sentiment**

- Tetlock, P.C. (2007) 'Giving Content to Investor Sentiment: The Role of Media in the Stock Market', *The Journal of Finance*, 62(3), pp. 1139–1168.
- Tetlock, P.C., Saar-Tsechansky, M. and Macskassy, S. (2008) 'More than words: Quantifying language to measure firms' fundamentals', *The Journal of Finance*, 63(3), pp. 1437–1467.
- Baker, M. and Wurgler, J. (2007) 'Investor Sentiment in the Stock Market', *Journal of Economic Perspectives*, 21(2), pp. 129–151.
- Engelberg, J. and Parsons, C.A. (2011) 'The Causal Impact of Media in Financial Markets', *The Journal of Finance*, 66(1), pp. 67–97.
- Bollen, J., Mao, H. and Zeng, X. (2011) 'Twitter mood predicts the stock market', *Journal of Computational Science*, 2(1), pp. 1–8.

**Sentiment Analysis and NLP**

- Loughran, T. and McDonald, B. (2011) 'When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks', *The Journal of Finance*, 66(1), pp. 35–65.
- Araci, D. (2019) 'FinBERT: Financial Sentiment Analysis with Pre-trained Language Models', *arXiv preprint arXiv:1908.10063*.

**Volatility Modelling**

- Engle, R.F. (2001) 'GARCH 101: The Use of ARCH/GARCH Models in Applied Econometrics', *Journal of Economic Perspectives*, 15(4), pp. 157–168.
- Ozkan, O., Abosedra, S., Sharif, A. and Alola, A.A. (2024) 'Dynamic Volatility Among Fossil Energy, Clean Energy and Major Assets: Evidence from DCC-GARCH', *Economic Change and Restructuring*, 57(3), pp. 1123–1148.

**Trading Frameworks**

- Chan, E.P. (2013) *Algorithmic Trading: Winning Strategies and Their Rationale*. Hoboken: Wiley.
- López de Prado, M. (2018) *Advances in Financial Machine Learning*. Hoboken: Wiley.
- Lo, A.W. (2004) 'The Adaptive Markets Hypothesis: Market Efficiency from an Evolutionary Perspective', *The Journal of Portfolio Management*, 30(5), pp. 15–29.

---

## Academic Context

This artefact was produced as part of the MSc Data Science & Artificial
Intelligence dissertation at the University of Liverpool. All code, comments,
and write-ups were developed to satisfy the IT artefact assessment criteria,
including documentation standards, coding practices, and alignment with the
research methods described in the dissertation.