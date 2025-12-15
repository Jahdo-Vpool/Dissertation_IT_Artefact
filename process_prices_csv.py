"""
Convert manually downloaded Yahoo Finance CSV to a dissertation-safe format
(NON-DESTRUCTIVE: original CSVs are never overwritten)

What this does:
- Reads Yahoo Finance CSV format
- Parses dates correctly
- Sorts chronologically (oldest → newest)
- Keeps only required columns
- Computes DAILY returns (%)
- Saves as *_prices_processed.csv

"""

import pandas as pd


def convert_yahoo_csv(input_filename: str, ticker: str) -> pd.DataFrame | None:
    """
    Convert Yahoo Finance CSV to a standardized research format.

    input_filename: Original Yahoo CSV (e.g., 'TAN.csv') in data/
    ticker: Ticker symbol (e.g., 'TAN')
    """

    print(f"\n{ticker}:")
    print(f"  Reading data/{input_filename}...")

    try:
        # Read raw CSV
        df = pd.read_csv(f"data/{input_filename}")
        print(f"  Columns found: {list(df.columns)}")

        clean_df = pd.DataFrame()

        # ----------------------------
        # 1) Date: parse to datetime
        # ----------------------------
        if "Date" in df.columns:
            date_raw = df["Date"]
        elif "date" in df.columns:
            date_raw = df["date"]
        else:
            print(" Error: No 'Date' column found!")
            return None

        clean_df["date"] = pd.to_datetime(
            date_raw,
            errors="coerce",
            dayfirst=True
        )

        # Drop rows with bad dates
        bad_dates = clean_df["date"].isna().sum()
        if bad_dates > 0:
            print(f"  Warning: {bad_dates} rows with invalid dates removed.")
            clean_df = clean_df.dropna(subset=["date"]).copy()

        # ----------------------------
        # 2) Close price
        # ----------------------------
        if "Close" in df.columns:
            close_raw = df["Close"]
            close_source = "Close"
        elif "Adj Close" in df.columns:
            close_raw = df["Adj Close"]
            close_source = "Adj Close"
        elif "close" in df.columns:
            close_raw = df["close"]
            close_source = "close"
        else:
            print("  Error: No close price column found!")
            return None

        clean_df["close"] = pd.to_numeric(close_raw, errors="coerce")

        bad_close = clean_df["close"].isna().sum()
        if bad_close > 0:
            print(f"  Warning: {bad_close} rows with invalid prices removed.")
            clean_df = clean_df.dropna(subset=["close"]).copy()

        # ----------------------------
        # 3) Volume
        # ----------------------------
        if "Volume" in df.columns:
            clean_df["volume"] = (
                df["Volume"]
                .astype(str)
                .str.replace(",", "", regex=False)
            )
            clean_df["volume"] = (
                pd.to_numeric(clean_df["volume"], errors="coerce")
                .fillna(0)
                .astype(int)
            )
        elif "volume" in df.columns:
            clean_df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
        else:
            print("  Warning: No volume column found — setting to 0.")
            clean_df["volume"] = 0

        # ----------------------------
        # 4) Sort chronologically
        # ----------------------------
        clean_df = clean_df.sort_values("date").reset_index(drop=True)

        # ----------------------------
        # 5) Daily returns (%)
        # ----------------------------
        clean_df["return (%)"] = clean_df["close"].pct_change() * 100
        clean_df.loc[0, "return (%)"] = 0.0

        # ISO date format for consistency
        clean_df["date"] = clean_df["date"].dt.strftime("%Y-%m-%d")

        # ----------------------------
        # 6) Save with SAFE name
        # ----------------------------
        output_filename = f"data/{ticker}_prices_processed.csv"
        clean_df.to_csv(output_filename, index=False)

        print(f"    Converted {len(clean_df)} rows")
        print(f"    Close source used: {close_source}")
        print(f"    Date range: {clean_df['date'].min()} → {clean_df['date'].max()}")
        print(f"    Latest price: ${clean_df['close'].iloc[-1]:.2f}")
        print(f"    Saved: {output_filename}")

        return clean_df

    except FileNotFoundError:
        print(f"  File not found: data/{input_filename}")
        return None
    except Exception as e:
        print(f" Error: {e}")
        return None


def main():
    print("\n" + "=" * 60)
    print("CONVERT MANUAL PRICE DATA (SAFE OUTPUT)")
    print("=" * 60)

    TAN_INPUT_FILE = "TAN_prices.csv"
    SPY_INPUT_FILE = "SPY_prices.csv"

    print("\nOriginal files (unchanged):")
    print(f"  data/{TAN_INPUT_FILE}")
    print(f"  data/{SPY_INPUT_FILE}")

    tan_df = convert_yahoo_csv(TAN_INPUT_FILE, "TAN")
    spy_df = convert_yahoo_csv(SPY_INPUT_FILE, "SPY")

    print("\n" + "=" * 60)
    if tan_df is not None and spy_df is not None:
        print("✓ CONVERSION COMPLETE (ORIGINAL FILES PRESERVED)")

    else:
        print("CONVERSION INCOMPLETE")
        print("=" * 60)

if __name__ == "__main__":
    main()
