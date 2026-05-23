"""One-shot data fetch via Yahoo Finance.

Pulls daily OHLCV for each coin and writes a CSV per coin in `data/raw/`,
matching the file layout the rest of the project expects (`coin_<Name>.csv`).

Originally targeted the Kaggle `cryptocurrencypricehistory` dataset, but
Kaggle's Python library still requires the legacy `kaggle.json` format
while their website now issues new-style access tokens, so the auth path
is broken in practice. Yahoo Finance is no-auth and stays current, which
works out better for a portfolio piece anyway.
"""

from pathlib import Path
from datetime import date
import sys

import pandas as pd
import yfinance as yf


RAW = Path(__file__).parent / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)


# Name we use in the rest of the project  ->  Yahoo ticker
COINS = {
    "Bitcoin":      "BTC-USD",
    "Ethereum":     "ETH-USD",
    "BinanceCoin":  "BNB-USD",
    "Cardano":      "ADA-USD",
    "Solana":       "SOL-USD",
    "XRP":          "XRP-USD",
    "Polkadot":     "DOT-USD",
    "Dogecoin":     "DOGE-USD",
    "Litecoin":     "LTC-USD",
    "ChainLink":    "LINK-USD",
    "Tether":       "USDT-USD",
}

START = "2014-01-01"
END = date.today().isoformat()


def fetch(name: str, ticker: str) -> pd.DataFrame:
    """Download one coin and reshape to the project's expected schema."""
    df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=False)
    if df.empty:
        return df

    # yfinance can return a MultiIndex column header; flatten it.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df["Name"] = name
    df["Symbol"] = ticker.split("-")[0]
    df["Marketcap"] = pd.NA  # yfinance doesn't expose historical market cap

    # Match the original Kaggle layout (same column names and order).
    cols = ["Date", "Name", "Symbol", "High", "Low", "Open", "Close", "Volume", "Marketcap"]
    return df[cols].sort_values("Date").reset_index(drop=True)


def main():
    print(f"fetching {len(COINS)} coins from Yahoo Finance ({START} → {END})\n")
    written = 0
    for name, ticker in COINS.items():
        try:
            df = fetch(name, ticker)
            if df.empty:
                print(f"  {name:14s} ({ticker:9s})  empty response, skipping")
                continue
            out = RAW / f"coin_{name}.csv"
            df.to_csv(out, index=False)
            print(f"  {name:14s} ({ticker:9s})  {len(df):>5} rows  {df.Date.min().date()} → {df.Date.max().date()}")
            written += 1
        except Exception as e:
            print(f"  {name:14s} ({ticker:9s})  failed: {e}")
    print(f"\nwrote {written} files to {RAW}")
    if written == 0:
        sys.exit("nothing was downloaded - check your internet connection")


if __name__ == "__main__":
    main()
