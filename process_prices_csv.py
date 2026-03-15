"""
================================================================================
Script 3: ETF Price Data Processing (process_prices_csv.py)
================================================================================
Purpose:
    Reads manually downloaded Yahoo Finance CSV files for TAN and SPY,
    standardizes their format, computes daily percentage returns, and saves
    a clean processed version for use in the downstream pipeline. The original
    source files are never modified.

Academic Context:
    This script addresses Dissertation Objective 5 — comparing the behaviour
    of TAN (Invesco Solar ETF) and SPY (S&P 500 ETF). Accurate and consistently
    formatted price data is a prerequisite for computing returns, fitting the
    GARCH(1,1) volatility model in Script 9, and evaluating trading strategy
    performance in Script 6. Any inconsistency in date formatting, column
    naming, or numeric encoding between the two ETF files would introduce
    errors when the price data is merged with sentiment scores in Script 4.

Why Manual CSV Download:
    Yahoo Finance CSV files were downloaded manually rather than fetched
    programmatically via yfinance. This approach was chosen to ensure the
    exact data used in the analysis is fixed and reproducible — programmatic
    downloads can return slightly different results depending on the date and
    time of execution due to retroactive adjustments. Storing the raw CSVs
    in the /data directory alongside the processed outputs provides a clear
    audit trail for the dataset.

Non-Destructive Design:
    The script deliberately writes output to a new file (*_prices_processed.csv)
    rather than overwriting the original. This ensures the raw source data
    remains intact and the processing step can be re-run safely at any point
    without risk of data loss.

Inputs:
    - data/TAN_prices.csv  : Manually downloaded Yahoo Finance data for TAN
    - data/SPY_prices.csv  : Manually downloaded Yahoo Finance data for SPY

Outputs:
    - data/TAN_prices_processed.csv : Cleaned and standardised TAN price data
    - data/SPY_prices_processed.csv : Cleaned and standardised SPY price data

Pipeline Position:
    Script 3 of 10. Output feeds into Script 4 (merge_prices_news.py),
    where processed prices are joined with daily sentiment scores.

Dependencies:
    - pandas : Data loading, cleaning, transformation, and CSV export

Usage:
    python process_prices_csv.py
================================================================================
"""

import pandas as pd


# ============================================
# SECTION 1: CONVERSION FUNCTION
# ============================================

def convert_yahoo_csv(input_filename: str, ticker: str) -> pd.DataFrame | None:
    """
    Read a raw Yahoo Finance CSV file, standardise its structure, compute
    daily returns, and save the result as a new processed CSV.

    Yahoo Finance CSVs do not follow a guaranteed schema — column names
    can vary (e.g. 'Close' vs 'Adj Close' vs 'close'), date formats can
    differ, and volume fields sometimes contain comma-formatted strings.
    This function handles all known variants defensively so that the
    pipeline does not break when column names differ between downloads.

    Parameters
    ----------
    input_filename : str
        The filename of the raw Yahoo Finance CSV within the /data directory
        (e.g. 'TAN_prices.csv'). The /data/ prefix is prepended internally.

    ticker : str
        The ETF ticker symbol (e.g. 'TAN' or 'SPY'). Used to construct
        the output filename and for console reporting.

    Returns
    -------
    pandas.DataFrame or None
        A cleaned DataFrame with columns: date, close, volume, return (%).
        Returns None if the input file is not found or cannot be parsed.
    """

    print(f"\n{ticker}:")
    print(f"  Reading data/{input_filename}...")

    try:
        # Load the raw CSV into a DataFrame. No parsing assumptions are made
        # at this stage — all columns are read as raw strings or default types
        # and normalised in the steps below.
        df = pd.read_csv(f"data/{input_filename}")
        print(f"  Columns found: {list(df.columns)}")

        # Initialise a clean output DataFrame. Building into a separate
        # DataFrame rather than modifying df in place ensures the original
        # data remains unaltered in memory throughout the function.
        clean_df = pd.DataFrame()

        # -------------------------------------------------------------------
        # STEP 1: DATE PARSING
        # -------------------------------------------------------------------
        # Yahoo Finance exports dates under either 'Date' or 'date' depending
        # on the download method and regional settings. Both variants are
        # handled to ensure robustness across different export formats.
        # errors='coerce' converts any unparseable date strings to NaT rather
        # than raising an exception, allowing bad rows to be identified and
        # removed cleanly in the next step.

        if "Date" in df.columns:
            date_raw = df["Date"]
        elif "date" in df.columns:
            date_raw = df["date"]
        else:
            print("  Error: No 'Date' column found!")
            return None

        clean_df["date"] = pd.to_datetime(
            date_raw,
            format="%Y-%m-%d",
            errors="coerce"
        )

        # Remove rows where the date could not be parsed. These rows would
        # produce NaT values that would cause alignment errors in Script 4.
        bad_dates = clean_df["date"].isna().sum()
        if bad_dates > 0:
            print(f"  Warning: {bad_dates} rows with invalid dates removed.")
            clean_df = clean_df.dropna(subset=["date"]).copy()

        # -------------------------------------------------------------------
        # STEP 2: CLOSE PRICE
        # -------------------------------------------------------------------
        # Three column name variants are checked in priority order.
        # 'Adj Close' (adjusted close) accounts for dividends and stock splits,
        # making it the preferred column for return calculations as it reflects
        # the true economic return to an investor holding the ETF. Where only
        # 'Close' or 'close' is available, that is used as a fallback.

        if "Adj Close" in df.columns:
            close_raw = df["Adj Close"]
            close_source = "Adj Close"
        elif "Close" in df.columns:
            close_raw = df["Close"]
            close_source = "Close"
        elif "close" in df.columns:
            close_raw = df["close"]
            close_source = "close"
        else:
            print("  Error: No close price column found!")
            return None

        # Convert to numeric, coercing any non-numeric entries (e.g. 'N/A')
        # to NaN so they can be removed without crashing the pipeline.
        clean_df["close"] = pd.to_numeric(close_raw, errors="coerce")

        bad_close = clean_df["close"].isna().sum()
        if bad_close > 0:
            print(f"  Warning: {bad_close} rows with invalid prices removed.")
            clean_df = clean_df.dropna(subset=["close"]).copy()

        # -------------------------------------------------------------------
        # STEP 3: VOLUME
        # -------------------------------------------------------------------
        # Volume is retained as a supplementary field. Yahoo Finance sometimes
        # exports volume with comma separators (e.g. '1,234,567'), which
        # prevents direct numeric conversion. The string replacement step
        # strips these commas before conversion. Missing volume fields are
        # filled with zero rather than dropping the row, as volume is not
        # a required input for any downstream calculation.

        if "Volume" in df.columns:
            clean_df["volume"] = (
                df["Volume"]
                .astype(str)
                .str.replace(",", "", regex=False)  # Remove comma a thousand separators
            )
            clean_df["volume"] = (
                pd.to_numeric(clean_df["volume"], errors="coerce")
                .fillna(0)
                .astype(int)
            )
        elif "volume" in df.columns:
            clean_df["volume"] = pd.to_numeric(
                df["volume"], errors="coerce"
            ).fillna(0).astype(int)
        else:
            print("  Warning: No volume column found — setting to 0.")
            clean_df["volume"] = 0

        # -------------------------------------------------------------------
        # STEP 4: CHRONOLOGICAL SORTING
        # -------------------------------------------------------------------
        # Yahoo Finance exports data in reverse chronological order (newest
        # first). The data must be sorted oldest-to-newest before computing
        # returns, because pct_change() calculates the change relative to the
        # immediately preceding row. Sorting incorrectly would produce
        # returns with the wrong sign.

        clean_df = clean_df.sort_values("date").reset_index(drop=True)

        # -------------------------------------------------------------------
        # STEP 5: DAILY RETURNS
        # -------------------------------------------------------------------
        # Daily percentage return is computed as:
        #   r_t = ((P_t - P_{t-1}) / P_{t-1}) * 100
        #
        # pct_change() handles this calculation directly. The first row has
        # no predecessor, so its return is set to 0.0 to avoid a NaN value
        # that would propagate through later calculations.
        #
        # Daily return is the primary financial variable used in:
        #   - GARCH(1,1) volatility modelling (Script 9)
        #   - Trading strategy backtesting (Script 6)
        #   - Correlation analysis with sentiment (Script 10)

        clean_df["return (%)"] = clean_df["close"].pct_change() * 100
        clean_df.loc[0, "return (%)"] = 0.0  # Set first row return to zero

        # -------------------------------------------------------------------
        # STEP 6: DATE FORMATTING AND SAVE
        # -------------------------------------------------------------------
        # Dates are converted to ISO 8601 string format (YYYY-MM-DD) to
        # ensure consistent formatting when this file is read by Script 4.
        # A unified date format is essential for the merge join to work
        # correctly, as mismatched formats would prevent records from matching.

        clean_df["date"] = clean_df["date"].dt.strftime("%Y-%m-%d")

        output_filename = f"data/{ticker}_prices_processed.csv"
        clean_df.to_csv(output_filename, index=False)

        # Confirmation report for quality assurance
        print(f"    Converted {len(clean_df)} rows")
        print(f"    Close source used: {close_source}")
        print(f"    Date range: {clean_df['date'].min()} to {clean_df['date'].max()}")
        print(f"    Latest price: ${clean_df['close'].iloc[-1]:.2f}")
        print(f"    Saved: {output_filename}")

        return clean_df

    except FileNotFoundError:
        # Raised if the specified CSV does not exist in /data.
        # A clear error message is printed rather than a traceback to make
        # the issue immediately actionable for the user.
        print(f"  File not found: data/{input_filename}")
        return None

    except Exception as e:
        # Catch-all for any unexpected errors during parsing or processing.
        print(f"  Error: {e}")
        return None


# ============================================
# SECTION 2: MAIN EXECUTION
# ============================================

def main():
    """
    Run the conversion pipeline for both TAN and SPY.

    The input filenames are declared as named variables at the top of main()
    rather than being hard-coded inside the function call. This makes it
    straightforward to update the filenames if the downloaded CSVs are
    named differently, without needing to search through the function body.
    """

    print("\n" + "=" * 60)
    print("CONVERT MANUAL PRICE DATA (SAFE OUTPUT)")
    print("=" * 60)

    # Declare input filenames — update these if your downloaded files
    # have different names.
    TAN_INPUT_FILE = "TAN_prices.csv"
    SPY_INPUT_FILE = "SPY_prices.csv"

    print("\nOriginal files (unchanged):")
    print(f"  data/{TAN_INPUT_FILE}")
    print(f"  data/{SPY_INPUT_FILE}")

    # Process each ETF independently. If one conversion fails, the other
    # still runs rather than the entire pipeline halting.
    tan_df = convert_yahoo_csv(TAN_INPUT_FILE, "TAN")
    spy_df = convert_yahoo_csv(SPY_INPUT_FILE, "SPY")

    # -----------------------------------------------------------------------
    # COMPLETION REPORT
    # -----------------------------------------------------------------------

    print("\n" + "=" * 60)
    if tan_df is not None and spy_df is not None:
        print("CONVERSION COMPLETE - ORIGINAL FILES PRESERVED")
        print("\nNext step: Run merge_prices_news.py")
    else:
        print("CONVERSION INCOMPLETE")
        print("Check the error messages above and verify your CSV files")
        print("are present in the /data directory.")
    print("=" * 60)


# ============================================
# SECTION 3: ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()
