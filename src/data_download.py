"""Download historical price data for S&P 500 constituents."""
import time
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

# Configuration
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM", "V", "UNH", "XOM", "JNJ", "WMT", "MA", "PG", "HD", "CVX",
    "LLY", "ABBV", "PEP", "KO", "AVGO", "COST", "MRK", "ADBE", "MCD",
    "WFC", "CRM", "TMO", "ACN", "ABT", "LIN", "CSCO", "DHR", "AMD",
    "TXN", "NFLX", "PM", "ORCL", "QCOM", "DIS", "INTU", "AMGN", "CAT",
    "IBM", "GE", "BA", "GS", "BLK",
]

START_DATE = "2010-01-01"
END_DATE = "2023-12-31"
RAW_DIR = Path("data/raw")

def download_ticker(ticker: str) -> pd.DataFrame | None:
    """Download daily OHLCV data for a single ticker."""
    try:
        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            return None
            
        # Flatten multi-level columns if they exist (yfinance v1.7 behavior)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df["ticker"] = ticker
        df.reset_index(inplace=True)
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        return df
    except Exception as e:
        print(f"Failed for {ticker}: {e}")
        return None

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_frames = []
    for ticker in tqdm(TICKERS, desc="Downloading"):
        df = download_ticker(ticker)
        if df is not None:
            all_frames.append(df)
        time.sleep(0.3)  # Be polite to Yahoo Finance

    if not all_frames:
        print("No data downloaded. Check your internet connection.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    out_path = RAW_DIR / "prices.parquet"
    combined.to_parquet(out_path)
    print(f"Saved {len(combined):,} rows to {out_path}")

if __name__ == "__main__":
    main()