"""Shared helpers for the crypto contagion analysis.

Most of the heavy lifting lives in the notebooks. This file is just the bits
that get reused across more than one of them: data loading, returns, regime
labels, risk metrics. I try to keep one-off code in the notebook where it's
used and only promote things here when they're called from 2+ places.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# Coins we focus on. Picked for long history, top market cap, and some variety
# in story (BTC = digital gold, ETH = smart contracts, USDT = stable, DOGE = meme,
# SOL = newer L1, etc.). Tether is here mostly so the EDA can show what "stable"
# actually looks like next to the rest.
MAJOR_COINS = [
    "Bitcoin", "Ethereum", "BinanceCoin", "Cardano", "Solana",
    "XRP", "Polkadot", "Dogecoin", "Litecoin", "ChainLink", "Tether",
]

# Known stress windows used in the crash-anatomy notebook.
EVENTS = {
    "COVID":  ("2020-02-15", "2020-04-30"),
    "May-21": ("2021-05-01", "2021-07-31"),   # China mining ban / Elon flip
    "Luna":   ("2022-05-01", "2022-06-30"),   # UST de-peg, Luna -> 0
    "FTX":    ("2022-11-01", "2022-12-15"),   # FTX implosion
}


# ---------- loading ---------------------------------------------------------

def load_coin(name: str) -> pd.DataFrame:
    """Load one coin's daily OHLCV CSV from data/raw and normalize the index."""
    path = RAW / f"coin_{name}.csv"
    df = pd.read_csv(path)
    # Some sources (the legacy Kaggle CSVs) include an SNo column. Drop if present.
    df = df.drop(columns=[c for c in ("SNo",) if c in df.columns])
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None).dt.normalize()
    return df.set_index("Date").sort_index()


def load_panel(coins=None, field="Close") -> pd.DataFrame:
    """Wide panel: one column per coin, rows = days, values = `field`."""
    coins = coins or MAJOR_COINS
    out = {}
    for c in coins:
        try:
            out[c] = load_coin(c)[field]
        except FileNotFoundError:
            # Don't blow up the whole panel if one coin is missing from the dump
            print(f"skip {c}: file not found")
    return pd.concat(out, axis=1).sort_index()


# ---------- returns & risk --------------------------------------------------

def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna(how="all")


def rolling_vol(returns: pd.DataFrame, window: int = 30, annualize: bool = True):
    vol = returns.rolling(window).std()
    return vol * np.sqrt(365) if annualize else vol


def drawdown(prices: pd.DataFrame) -> pd.DataFrame:
    """Percentage drop from the running peak. Always <= 0."""
    return prices / prices.cummax() - 1


def max_drawdown(prices) -> float | pd.Series:
    return drawdown(prices).min()


def sharpe(returns, rf: float = 0.0, periods: int = 365) -> float | pd.Series:
    excess = returns - rf / periods
    return excess.mean() / excess.std() * np.sqrt(periods)


def sortino(returns, rf: float = 0.0, periods: int = 365):
    excess = returns - rf / periods
    downside = excess.where(excess < 0).std()
    return excess.mean() / downside * np.sqrt(periods)


def calmar(prices, periods: int = 365):
    rets = log_returns(prices)
    annual = rets.mean() * periods
    return annual / -max_drawdown(prices)


# ---------- market structure -----------------------------------------------

def avg_pairwise_corr(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    """Average pairwise correlation across all coins on a rolling window.

    With fewer than 2 columns the metric is undefined, so we just return an
    all-NaN series rather than warn on empty np.nanmean calls.
    """
    idx = returns.index
    out = pd.Series(index=idx, dtype=float)
    arr = returns.values
    n_obs, n_assets = arr.shape
    if n_assets < 2:
        return out
    # Doing this row-wise on a rolling object is slow. Build it manually.
    iu = np.triu_indices(n_assets, k=1)
    for i in range(window, n_obs):
        block = arr[i - window:i]
        c = pd.DataFrame(block).corr().values
        out.iloc[i] = np.nanmean(c[iu])
    return out


# ---------- regime tagging --------------------------------------------------

def label_regimes(prices: pd.Series,
                  vol_window: int = 30,
                  dd_bear: float = -0.20,
                  dd_crisis: float = -0.40) -> pd.Series:
    """Simple regime classifier.

    bull   : drawdown shallower than -20%
    bear   : drawdown between -20% and -40%
    crisis : drawdown beyond -40% AND vol in top decile

    Not an HMM, just a heuristic. The point is something interpretable I can
    defend in an interview, not a black box. Worth comparing against an HMM
    in a notebook later if curious.
    """
    dd = drawdown(prices)
    rets = log_returns(prices)
    vol = rolling_vol(rets, window=vol_window)
    vol_hi = vol > vol.quantile(0.90)

    regime = pd.Series("bull", index=prices.index)
    regime[dd <= dd_bear] = "bear"
    regime[(dd <= dd_crisis) & vol_hi] = "crisis"
    return regime
