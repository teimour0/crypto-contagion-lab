"""Interactive companion to the notebooks.

Run with:
    streamlit run app.py

The notebooks tell the story end-to-end. The dashboard lets you poke at the
data: pick coins, slide the date range, watch correlations shift.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import utils as U


st.set_page_config(page_title="Crypto Contagion Lab", layout="wide", page_icon=None)

# ---------- data ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data():
    panel = U.load_panel()
    rets = U.log_returns(panel)
    return panel, rets


try:
    prices, rets = load_data()
except FileNotFoundError:
    st.error(
        "No data found in `data/raw/`. Run `python download_data.py` first "
        "(see README for the Kaggle token setup)."
    )
    st.stop()


# ---------- sidebar controls -----------------------------------------------

st.sidebar.markdown("### Filters")

all_coins = list(prices.columns)
default_coins = [c for c in ["Bitcoin", "Ethereum", "Solana", "BinanceCoin", "XRP", "Dogecoin"]
                 if c in all_coins]
coins = st.sidebar.multiselect("Coins", all_coins, default=default_coins)

min_d, max_d = prices.index.min().date(), prices.index.max().date()
start, end = st.sidebar.slider(
    "Date range",
    min_value=min_d, max_value=max_d,
    value=(min_d, max_d),
    format="YYYY-MM",
)

vol_window = st.sidebar.slider("Volatility window (days)", 7, 90, 30)
corr_window = st.sidebar.slider("Rolling correlation window (days)", 30, 180, 60)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Companion notebooks under `notebooks/`. "
    "Methodology in the README."
)


# ---------- filter to selection --------------------------------------------

if not coins:
    st.warning("Pick at least one coin in the sidebar.")
    st.stop()

p = prices.loc[str(start):str(end), coins].dropna(how="all")
r = U.log_returns(p)


# ---------- header ----------------------------------------------------------

st.title("Crypto Contagion Lab")
st.caption(
    "How shocks propagate across the crypto market. Interactive companion to "
    "the analysis notebooks."
)


# ---------- top row: KPIs ---------------------------------------------------

c1, c2, c3, c4 = st.columns(4)
with c1:
    days = (p.index[-1] - p.index[0]).days
    st.metric("Sample window", f"{days/365:.1f} yrs", f"{len(p)} days")
with c2:
    st.metric("Coins in view", f"{len(coins)}")
with c3:
    mdds = U.max_drawdown(p)
    worst = mdds.idxmin()
    st.metric("Worst drawdown", f"{mdds.min()*100:.0f}%", f"{worst}")
with c4:
    avg_corr = U.avg_pairwise_corr(r.dropna(), window=min(corr_window, max(len(r)-1, 10)))
    st.metric("Mean pairwise corr", f"{avg_corr.mean():.2f}",
              f"max {avg_corr.max():.2f}" if not avg_corr.dropna().empty else "")


st.markdown("---")


# ---------- price chart -----------------------------------------------------

st.subheader("Prices (log scale, rebased)")

rebase_base = p.bfill().iloc[0]
norm = p.div(rebase_base).mul(100)

fig = go.Figure()
for c in coins:
    fig.add_trace(go.Scatter(x=norm.index, y=norm[c], name=c, mode="lines",
                             line=dict(width=1.4)))
fig.update_layout(
    yaxis_type="log",
    yaxis_title="rebased to 100",
    hovermode="x unified",
    height=420,
    legend=dict(orientation="h", y=-0.2),
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig, use_container_width=True)


# ---------- vol + drawdown two-up ------------------------------------------

st.subheader("Volatility and drawdowns")

vc1, vc2 = st.columns(2)

with vc1:
    vol = U.rolling_vol(r, window=vol_window)
    fig = go.Figure()
    for c in coins:
        if c in vol.columns:
            fig.add_trace(go.Scatter(x=vol.index, y=vol[c], name=c, mode="lines",
                                     line=dict(width=1)))
    fig.update_layout(
        title=f"{vol_window}-day annualized vol",
        height=360, hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

with vc2:
    dd = U.drawdown(p)
    fig = go.Figure()
    for c in coins:
        if c in dd.columns:
            fig.add_trace(go.Scatter(x=dd.index, y=dd[c]*100, name=c, mode="lines",
                                     line=dict(width=1), fill="tozeroy", opacity=0.4))
    fig.update_layout(
        title="Drawdown from running peak (%)",
        height=360, hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------- correlation heatmap + rolling ----------------------------------

st.subheader("Market structure")

cc1, cc2 = st.columns([1, 1.4])

with cc1:
    C = r.dropna().corr()
    fig = px.imshow(
        C.values,
        x=C.columns, y=C.index,
        color_continuous_scale="RdYlBu_r", zmin=-0.2, zmax=1, aspect="auto",
        text_auto=".2f",
    )
    fig.update_layout(
        title="Correlation of daily returns",
        height=420, margin=dict(l=10, r=10, t=40, b=10),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)

with cc2:
    if len(coins) >= 2 and len(r.dropna()) > corr_window:
        apc = U.avg_pairwise_corr(r.dropna(), window=corr_window).dropna()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=apc.index, y=apc.values, mode="lines",
                                 line=dict(color="black", width=1.5)))
        # event shading
        ymin, ymax = apc.min(), apc.max()
        for label, (s, e) in U.EVENTS.items():
            fig.add_vrect(x0=s, x1=e, fillcolor="red", opacity=0.08, line_width=0,
                          annotation_text=label, annotation_position="top left",
                          annotation_font_size=10)
        fig.update_layout(
            title=f"Average pairwise correlation ({corr_window}-day window)",
            yaxis_title="mean ρ",
            height=420, hovermode="x",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least 2 coins and enough history for the rolling window.")


# ---------- regime + risk metrics ------------------------------------------

st.subheader("Regimes and risk metrics")

rc1, rc2 = st.columns([1.6, 1])

with rc1:
    if "Bitcoin" in p.columns:
        btc = p["Bitcoin"].dropna()
        regime = U.label_regimes(btc)
        regime_num = regime.map({"bull": 0, "bear": 1, "crisis": 2})

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.75, 0.25], vertical_spacing=0.04)
        fig.add_trace(go.Scatter(x=btc.index, y=btc.values, mode="lines",
                                 name="BTC", line=dict(color="black", width=1)),
                      row=1, col=1)
        fig.update_yaxes(type="log", row=1, col=1)

        colors = {"bull": "#7bc97b", "bear": "#e8a05c", "crisis": "#c0392b"}
        for r_name, col in colors.items():
            mask = regime == r_name
            fig.add_trace(go.Scatter(
                x=regime.index[mask], y=[1]*mask.sum(),
                mode="markers", marker=dict(color=col, symbol="square", size=4),
                name=r_name,
            ), row=2, col=1)
        fig.update_yaxes(visible=False, row=2, col=1)
        fig.update_layout(
            title="BTC price with regime overlay",
            height=420, hovermode="x unified",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Add Bitcoin to the selection to see the regime overlay.")

with rc2:
    metrics = pd.DataFrame({
        "CAGR":     ((p.iloc[-1] / p.iloc[0]) ** (365 / max(days, 1)) - 1) * 100,
        "Vol":      (r.std() * np.sqrt(365) * 100),
        "Sharpe":   U.sharpe(r),
        "Sortino":  U.sortino(r),
        "Max DD":   (U.max_drawdown(p) * 100),
    }).round(2)
    metrics = metrics.dropna()
    st.dataframe(
        metrics.style.format({"CAGR": "{:.1f}%", "Vol": "{:.1f}%",
                              "Sharpe": "{:.2f}", "Sortino": "{:.2f}",
                              "Max DD": "{:.1f}%"}),
        use_container_width=True,
        height=420,
    )


# ---------- footer ----------------------------------------------------------

st.markdown("---")
st.caption(
    "Built with pandas, Plotly and Streamlit. Data from Kaggle "
    "(`sudalairajkumar/cryptocurrencypricehistory`). "
    "Not investment advice."
)
