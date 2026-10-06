"""
S&P 500 Sharpe-Ratio demo (textbook Markowitz, long-only).
Educational use only. Contains no proprietary methodology.
"""
import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize

# ---------------------------------------------------------------
# Step 0: all tunable parameters live here (nothing hard-coded below)
# ---------------------------------------------------------------
CONFIG = {
    "page_title": "S&P 500 Sharpe Optimizer",
    "default_tickers": [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "BRK-B", "JPM",
        "JNJ", "XOM", "PG", "KO", "PEP", "WMT", "HD", "UNH", "V", "MA",
        "CVX", "MRK", "ABBV", "COST", "MCD", "CSCO", "ORCL",
    ],
    "default_start_years_back": 5,
    "min_assets": 2,
    "trading_days": 252,
    "default_rf_pct": 2.0,
    "default_max_weight_pct": 40,
    "frontier_points": 40,
}

st.set_page_config(page_title=CONFIG["page_title"], layout="wide")


# Step 1: download adjusted close prices (cached so repeated clicks are fast)
@st.cache_data(show_spinner=False)
def load_prices(tickers, start, end):
    data = yf.download(list(tickers), start=start, end=end,
                       auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=list(tickers)[0])
    # drop tickers with no data, then rows with gaps
    return data.dropna(axis=1, how="all").dropna()


# Step 2: portfolio statistics from weights
def portfolio_stats(w, mu, cov, rf):
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    sharpe = (ret - rf) / vol if vol > 0 else np.nan
    return ret, vol, sharpe


# Step 3: maximize the Sharpe ratio (long-only, weights sum to 1, weight cap)
def max_sharpe(mu, cov, rf, max_w):
    n = len(mu)
    x0 = np.full(n, 1.0 / n)
    bounds = [(0.0, max_w)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    res = minimize(lambda w: -portfolio_stats(w, mu, cov, rf)[2], x0,
                   method="SLSQP", bounds=bounds, constraints=cons)
    return res.x


# Step 4: efficient frontier = minimum variance for each target return
def efficient_frontier(mu, cov, max_w, n_points):
    n = len(mu)
    bounds = [(0.0, max_w)] * n
    targets = np.linspace(mu.min(), mu.max(), n_points)
    pts = []
    for t in targets:
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "eq", "fun": lambda w, t=t: w @ mu - t}]
        r = minimize(lambda w: w @ cov @ w, np.full(n, 1.0 / n),
                     method="SLSQP", bounds=bounds, constraints=cons)
        if r.success:
            pts.append((float(np.sqrt(r.x @ cov @ r.x)), t))
    return pts


# ---------------------------------------------------------------
# Home page: inputs
# ---------------------------------------------------------------
st.title("S&P 500 Sharpe Ratio Optimizer")
st.caption("Teaching demo: classic mean-variance optimization on historical "
           "data. Not investment advice.")

col1, col2 = st.columns(2)
with col1:
    chosen = st.multiselect("Select assets", CONFIG["default_tickers"],
                            default=CONFIG["default_tickers"][:8])
    extra = st.text_input("Add other tickers (comma-separated, optional)")
with col2:
    today = dt.date.today()
    start = st.date_input(
        "Start date",
        today - dt.timedelta(days=365 * CONFIG["default_start_years_back"]))
    end = st.date_input("End date", today)
    rf_pct = st.number_input("Risk-free rate (% p.a.)", 0.0, 20.0,
                             CONFIG["default_rf_pct"], 0.1)
    cap_pct = st.slider("Maximum weight per asset (%)", 5, 100,
                        CONFIG["default_max_weight_pct"])

tickers = list(dict.fromkeys(
    chosen + [t.strip().upper() for t in extra.split(",") if t.strip()]))

# Step 5: run only when the button is pressed
if st.button("Optimize", type="primary"):
    if len(tickers) < CONFIG["min_assets"]:
        st.error(f"Select at least {CONFIG['min_assets']} assets.")
        st.stop()
    if start >= end:
        st.error("Start date must be before end date.")
        st.stop()

    with st.spinner("Downloading data and optimizing..."):
        prices = load_prices(tuple(tickers), start, end)
        if prices.shape[1] < CONFIG["min_assets"] or len(prices) < 30:
            st.error("Not enough data for this selection and period.")
            st.stop()

        # annualized mean returns and covariance from daily returns
        rets = prices.pct_change().dropna()
        mu = rets.mean().values * CONFIG["trading_days"]
        cov = rets.cov().values * CONFIG["trading_days"]
        names = list(rets.columns)
        rf, max_w = rf_pct / 100.0, cap_pct / 100.0

        # the cap must allow weights to sum to 1
        if max_w * len(names) < 1.0:
            st.error("Weight cap too low for this number of assets. "
                     "Raise the cap or add assets.")
            st.stop()

        w = max_sharpe(mu, cov, rf, max_w)
        ret, vol, sharpe = portfolio_stats(w, mu, cov, rf)
        frontier = efficient_frontier(mu, cov, max_w, CONFIG["frontier_points"])

    # Step 6: results
    m1, m2, m3 = st.columns(3)
    m1.metric("Expected return (p.a.)", f"{ret:.1%}")
    m2.metric("Volatility (p.a.)", f"{vol:.1%}")
    m3.metric("Sharpe ratio", f"{sharpe:.2f}")

    left, right = st.columns(2)
    table = (pd.DataFrame({"Asset": names, "Weight": w})
             .query("Weight > 0.0005").sort_values("Weight", ascending=False))
    with left:
        st.subheader("Optimal weights")
        st.dataframe(table.style.format({"Weight": "{:.1%}"}),
                     hide_index=True, use_container_width=True)
    with right:
        st.subheader("Efficient frontier")
        fig, ax = plt.subplots()
        if frontier:
            ax.plot([p[0] for p in frontier], [p[1] for p in frontier],
                    label="Efficient frontier")
        ax.scatter(np.sqrt(np.diag(cov)), mu, s=15, color="grey",
                   label="Single assets")
        ax.scatter([vol], [ret], color="red", zorder=5, label="Max Sharpe")
        ax.set_xlabel("Volatility")
        ax.set_ylabel("Expected return")
        ax.legend()
        st.pyplot(fig)

    st.info("Notes: results are based on historical data and are in-sample, "
            "so they say nothing reliable about future performance. "
            "Using today's index constituents introduces survivorship bias.")

# ---------------------------------------------------------------
# Version log
# v1.0 - Initial version: asset and date selection, Optimize button,
#        max-Sharpe weights (long-only, weight cap), efficient frontier,
#        all parameters in CONFIG.
# ---------------------------------------------------------------
