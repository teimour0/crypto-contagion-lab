# Crypto Contagion Lab

How crypto market shocks travel, looked at through correlation structure and crash anatomy rather than next-day price prediction.

Notebooks plus a Streamlit dashboard. Built as a portfolio piece while figuring out why most crypto "ML" projects are misleading.

## Why this angle

Most portfolio projects in this space try to forecast tomorrow's price. Daily crypto forecasting from OHLCV barely works (notebook 04 actually backtests this and shows it), so the interesting questions are the structural ones. How correlated does the market get under stress? How do different kinds of crashes propagate? What does diversification across coins actually buy you?

That's what's here.

## Layout

```
notebooks/
  01_explore.ipynb              returns, fat tails, vol clustering
  02_market_structure.ipynb     correlations, regime tagging
  03_crashes.ipynb              COVID, Luna and FTX side by side
  04_portfolio_forecasting.ipynb  Sharpe/Sortino/Calmar + ARIMA backtest
utils.py                        loading + risk metrics
app.py                          Streamlit dashboard
download_data.py                fetch the data
smoke_test.py                   30-second sanity check
```

## What I found

A bunch of stuff, but the things I'd actually talk about in an interview:

- Crypto returns are absurdly fat-tailed. Kurtosis is in the 10 to 40+ range vs ~5-10 for equities. So a normal-distribution VaR is fiction.
- Volatility clusters strongly. Squared-return ACF stays positive out to ~30 days. Calm follows calm, panic follows panic.
- Cross-coin correlations collapse upward in crises. Average pairwise ~0.45 in calm markets, ~0.75+ during stress. Diversification fails right when you need it.
- The three crashes I dug into all looked different. COVID was every coin dumping on the same day. Luna was a contagion radiating out from one protocol's design failure. FTX was step-function drops at news triggers with magnitude scaling with FTX exposure.
- Daily ARIMA on BTC returns hit 51% directional accuracy in walk-forward backtest. Naive 7-day mean was 47.5%. Neither is enough to trade on after fees.

## Setup

Python 3.10 or newer.

```bash
pip install -r requirements.txt
python download_data.py
```

`download_data.py` pulls daily OHLCV from Yahoo Finance for 11 coins and dumps CSVs in `data/raw/`. No auth, no API keys. Takes about 10 seconds.

The original plan was to use the `cryptocurrencypricehistory` Kaggle dataset, but Kaggle's Python library still wants the legacy `kaggle.json` while their website now only issues new-format access tokens, so the auth is broken in practice. yfinance is no-setup and stays current, which works out better for a portfolio piece anyway.

## Run

```bash
# notebooks
jupyter lab

# dashboard
streamlit run app.py

# quick sanity check
python smoke_test.py
```

## Methodology notes

A few choices worth flagging because someone could ask about them:

- Log returns, not simple returns. Additive, slightly better-behaved.
- For cross-sectional work I restrict to dates where every coin has data, so correlations aren't computed over a moving universe.
- Annualization uses 365 days, not 252. Crypto trades on weekends.
- Regime labels are heuristic (drawdown plus vol percentile), not an HMM. They are interpretable rather than optimal. Easier to defend in conversation.
- Tether is excluded from the correlation and portfolio work because its near-zero variance breaks the math. It's included in the crash analysis as an anchor.

## Things I deliberately didn't do

There are some shortcuts that come up a lot in crypto portfolio projects, and I didn't take them:

- No LSTM-on-prices "predictive model" with a 0.99 R² that's just learning the trend.
- No Sharpe ratio without the max drawdown next to it.
- No backtest on the same window the model was fit on.
- No 252 trading days for an asset that trades 7 days a week.

Each of those buries the real result.

## Data caveats

yfinance coverage starts in 2014 for BTC and LTC, late 2017 for the next batch, mid-2020 for Solana and Polkadot. So some of the cross-coin analysis only has about five years of overlap. The notebooks call this out where it matters. If you want to swap data sources later (CoinGecko, Binance REST, whatever), only `download_data.py` needs to change. The rest just reads `data/raw/coin_<Name>.csv`.
