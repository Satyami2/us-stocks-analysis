"""
S&P 500 Rolling Returns Dashboard — Streamlit
==============================================
Interactive dashboard for exploring 10 years of S&P 500 price data and
3-year daily rolling returns.

Setup:
    pip install streamlit yfinance pandas numpy plotly lxml requests

Run:
    streamlit run sp500_streamlit_app.py

First load will download ~10 years of data for all ~500 tickers (10-20 min).
Subsequent loads use cached parquet files and are near-instant.
"""

import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="S&P 500 Rolling Returns Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
YEARS_BACK     = 10
ROLLING_WINDOW = 756          # ~3 trading years
CACHE_DIR      = "sp500_cache"
PRICES_FILE    = os.path.join(CACHE_DIR, "prices.parquet")
META_FILE      = os.path.join(CACHE_DIR, "meta.parquet")
BATCH_SIZE     = 50
SLEEP_BETWEEN  = 1.0

os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# DATA LOADING (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_sp500_tickers() -> pd.DataFrame:
    """Scrape S&P 500 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers, timeout=30).text
    tables = pd.read_html(html)
    df = tables[0][["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].copy()
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    df = df.rename(columns={
        "Symbol":   "Ticker",
        "Security": "Company",
        "GICS Sector":       "Sector",
        "GICS Sub-Industry": "Industry",
    })
    return df.reset_index(drop=True)


def download_batch(tickers, start, end):
    """Download one batch of tickers."""
    try:
        data = yf.download(
            tickers=" ".join(tickers),
            start=start, end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="ticker",
        )
        closes = {}
        if len(tickers) == 1:
            t = tickers[0]
            if "Close" in data.columns:
                closes[t] = data["Close"]
        else:
            for t in tickers:
                if t in data.columns.get_level_values(0):
                    closes[t] = data[t]["Close"]
        return pd.DataFrame(closes)
    except Exception:
        return pd.DataFrame()


def download_prices(tickers, years_back=YEARS_BACK, progress_bar=None, status_text=None):
    """Download adjusted close prices for all tickers in batches."""
    end_date   = datetime.today()
    start_date = end_date.replace(year=end_date.year - years_back)
    start_str  = start_date.strftime("%Y-%m-%d")
    end_str    = end_date.strftime("%Y-%m-%d")

    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    frames = []
    for i, batch in enumerate(batches):
        if status_text is not None:
            status_text.text(f"Downloading batch {i + 1}/{len(batches)} "
                             f"({batch[0]}…{batch[-1]})")
        df = download_batch(batch, start_str, end_str)
        if not df.empty:
            frames.append(df)
        if progress_bar is not None:
            progress_bar.progress((i + 1) / len(batches))
        time.sleep(SLEEP_BETWEEN)

    if not frames:
        raise RuntimeError("No data downloaded.")
    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    return prices.sort_index().dropna(how="all")


@st.cache_data(show_spinner=False)
def load_cached_data():
    """Load prices + meta from parquet cache if both exist."""
    if os.path.exists(PRICES_FILE) and os.path.exists(META_FILE):
        prices = pd.read_parquet(PRICES_FILE)
        meta   = pd.read_parquet(META_FILE)
        return prices, meta
    return None, None


def fetch_and_cache_data():
    """Full fetch pipeline with Streamlit progress UI."""
    status = st.empty()
    pbar   = st.progress(0.0)

    status.text("Fetching S&P 500 constituents from Wikipedia…")
    meta = get_sp500_tickers()
    status.text(f"Found {len(meta)} constituents. Starting download…")

    prices = download_prices(
        meta["Ticker"].tolist(),
        years_back=YEARS_BACK,
        progress_bar=pbar,
        status_text=status,
    )

    prices.to_parquet(PRICES_FILE)
    meta.to_parquet(META_FILE)
    pbar.empty(); status.empty()
    st.cache_data.clear()
    return prices, meta


# ---------------------------------------------------------------------------
# CALCULATIONS (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")


@st.cache_data(show_spinner=False)
def compute_rolling_3yr(prices: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Annualised 3-year rolling CAGR, computed daily."""
    ratio = prices / prices.shift(window)
    return (ratio.pow(1.0 / 3.0) - 1.0).dropna(how="all")


@st.cache_data(show_spinner=False)
def build_summary(prices: pd.DataFrame, rolling: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    latest     = rolling.iloc[-1]
    mean_3yr   = rolling.mean()
    median_3yr = rolling.median()
    std_3yr    = rolling.std()
    min_3yr    = rolling.min()
    max_3yr    = rolling.max()

    first_idx = prices.apply(lambda s: s.first_valid_index())
    last_idx  = prices.apply(lambda s: s.last_valid_index())
    first_px  = pd.Series({t: prices[t].loc[first_idx[t]] if pd.notna(first_idx[t]) else np.nan
                           for t in prices.columns})
    last_px   = pd.Series({t: prices[t].loc[last_idx[t]] if pd.notna(last_idx[t]) else np.nan
                           for t in prices.columns})
    total_ret = last_px / first_px - 1.0

    df = pd.DataFrame({
        "First Date":         first_idx,
        "Last Date":          last_idx,
        "First Price":        first_px.round(2),
        "Last Price":         last_px.round(2),
        "Total Return (10y)": total_ret.round(4),
        "Latest 3Y CAGR":     latest.round(4),
        "Mean 3Y CAGR":       mean_3yr.round(4),
        "Median 3Y CAGR":     median_3yr.round(4),
        "Std 3Y CAGR":        std_3yr.round(4),
        "Min 3Y CAGR":        min_3yr.round(4),
        "Max 3Y CAGR":        max_3yr.round(4),
    })
    df.index.name = "Ticker"
    df = df.reset_index().merge(meta, on="Ticker", how="left")
    cols = ["Ticker", "Company", "Sector", "Industry"] + \
           [c for c in df.columns if c not in ("Ticker", "Company", "Sector", "Industry")]
    return df[cols]


# ---------------------------------------------------------------------------
# SIDEBAR — DATA CONTROL
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Controls")

prices, meta = load_cached_data()

if prices is None:
    st.sidebar.warning("No cached data found.")
    if st.sidebar.button("⬇️ Download S&P 500 data (~10-20 min)", type="primary"):
        with st.spinner("Downloading from Yahoo Finance…"):
            prices, meta = fetch_and_cache_data()
        st.sidebar.success("Download complete!")
        st.rerun()
    else:
        st.title("📈 S&P 500 Rolling Returns Dashboard")
        st.info("👈 Click **Download S&P 500 data** in the sidebar to begin. "
                "This runs once and caches ~10 years of daily prices locally.")
        st.stop()
else:
    last_date = prices.index.max().strftime("%Y-%m-%d")
    st.sidebar.success(f"✓ Cache loaded\n\n"
                       f"**{prices.shape[1]}** tickers\n\n"
                       f"**{prices.shape[0]:,}** days\n\n"
                       f"Last date: **{last_date}**")
    if st.sidebar.button("🔄 Refresh data"):
        with st.spinner("Re-downloading…"):
            prices, meta = fetch_and_cache_data()
        st.rerun()

# Pre-compute
daily_ret = compute_daily_returns(prices)
rolling   = compute_rolling_3yr(prices)
summary   = build_summary(prices, rolling, meta)

# ---------------------------------------------------------------------------
# FILTERS
# ---------------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.subheader("🎯 Filters")

sectors = sorted(meta["Sector"].dropna().unique().tolist())
selected_sectors = st.sidebar.multiselect("Sector", sectors, default=sectors)

min_date = prices.index.min().date()
max_date = prices.index.max().date()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    d0, d1 = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    d0, d1 = pd.Timestamp(min_date), pd.Timestamp(max_date)

filt_meta    = meta[meta["Sector"].isin(selected_sectors)]
filt_tickers = filt_meta["Ticker"].tolist()
filt_summary = summary[summary["Ticker"].isin(filt_tickers)]

# ---------------------------------------------------------------------------
# MAIN — TABS
# ---------------------------------------------------------------------------
st.title("📈 S&P 500 Rolling Returns Dashboard")
st.caption(f"Daily prices from Yahoo Finance · 3-year rolling CAGR · "
           f"Data through {prices.index.max().strftime('%B %d, %Y')}")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview", "📊 Ticker Explorer", "🔥 Sectors",
    "🏆 Rankings", "📥 Download"
])

# ---------- TAB 1: OVERVIEW ----------
with tab1:
    st.subheader("Market snapshot")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tickers covered",      f"{len(filt_summary):,}")
    c2.metric("Median latest 3Y CAGR", f"{filt_summary['Latest 3Y CAGR'].median() * 100:.2f}%")
    c3.metric("Median 10Y total return", f"{filt_summary['Total Return (10y)'].median() * 100:.1f}%")
    c4.metric("Positive 3Y CAGR",
              f"{(filt_summary['Latest 3Y CAGR'] > 0).mean() * 100:.1f}%")

    st.divider()

    colL, colR = st.columns(2)
    with colL:
        st.markdown("**Distribution — Latest 3Y CAGR**")
        fig = px.histogram(
            filt_summary.dropna(subset=["Latest 3Y CAGR"]),
            x="Latest 3Y CAGR", nbins=50, color="Sector",
        )
        fig.update_layout(xaxis_tickformat=".0%", height=400,
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with colR:
        st.markdown("**Distribution — 10-year Total Return**")
        fig = px.histogram(
            filt_summary.dropna(subset=["Total Return (10y)"]),
            x="Total Return (10y)", nbins=50, color="Sector",
        )
        fig.update_layout(xaxis_tickformat=".0%", height=400,
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("**Risk vs. Return — Std dev of 3Y CAGR vs. Mean 3Y CAGR**")
    scat = filt_summary.dropna(subset=["Mean 3Y CAGR", "Std 3Y CAGR"])
    fig = px.scatter(
        scat, x="Std 3Y CAGR", y="Mean 3Y CAGR",
        color="Sector", hover_name="Ticker",
        hover_data={"Company": True, "Latest 3Y CAGR": ":.2%"},
        opacity=0.75,
    )
    fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                      height=550, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ---------- TAB 2: TICKER EXPLORER ----------
with tab2:
    st.subheader("Individual ticker analysis")

    col1, col2 = st.columns([2, 1])
    with col1:
        default_pick = "AAPL" if "AAPL" in filt_tickers else filt_tickers[0]
        ticker = st.selectbox(
            "Choose a ticker",
            options=filt_tickers,
            index=filt_tickers.index(default_pick) if default_pick in filt_tickers else 0,
        )
    with col2:
        compare = st.multiselect(
            "Compare against",
            options=[t for t in filt_tickers if t != ticker],
            default=["SPY"] if "SPY" in filt_tickers and ticker != "SPY" else [],
        )

    info = meta[meta["Ticker"] == ticker].iloc[0]
    st.markdown(f"### {info['Company']} ({ticker})")
    st.caption(f"{info['Sector']} · {info['Industry']}")

    row = summary[summary["Ticker"] == ticker].iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latest Price",       f"${row['Last Price']:.2f}")
    m2.metric("10Y Total Return",   f"{row['Total Return (10y)'] * 100:.1f}%")
    m3.metric("Latest 3Y CAGR",     f"{row['Latest 3Y CAGR'] * 100:.2f}%")
    m4.metric("3Y CAGR Volatility", f"{row['Std 3Y CAGR'] * 100:.2f}%")

    tickers_to_plot = [ticker] + compare

    # Price chart (normalised to 100)
    st.markdown("**Normalised price (first value = 100)**")
    px_window = prices.loc[d0:d1, tickers_to_plot].dropna(how="all")
    norm = px_window.apply(lambda s: s / s.dropna().iloc[0] * 100 if s.dropna().size else s)
    fig = px.line(norm, labels={"value": "Indexed", "variable": "Ticker"})
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Rolling 3Y CAGR chart
    st.markdown("**3-Year Rolling CAGR (annualised)**")
    roll_window = rolling.loc[d0:d1, tickers_to_plot].dropna(how="all")
    fig = px.line(roll_window, labels={"value": "3Y CAGR", "variable": "Ticker"})
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_tickformat=".0%", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Daily returns distribution
    st.markdown("**Daily return distribution**")
    dr = daily_ret.loc[d0:d1, ticker].dropna()
    fig = px.histogram(dr, nbins=80)
    fig.update_layout(height=320, showlegend=False, xaxis_tickformat=".1%",
                      margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="Daily return", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

# ---------- TAB 3: SECTORS ----------
with tab3:
    st.subheader("Sector-level view")

    sec = (filt_summary.groupby("Sector")
           .agg(Tickers=("Ticker", "count"),
                **{"Mean 10Y Total Return": ("Total Return (10y)", "mean"),
                   "Median 10Y Total Return": ("Total Return (10y)", "median"),
                   "Mean Latest 3Y CAGR":    ("Latest 3Y CAGR", "mean"),
                   "Median Latest 3Y CAGR":  ("Latest 3Y CAGR", "median"),
                   "Mean 3Y Volatility":     ("Std 3Y CAGR", "mean")})
           .sort_values("Mean Latest 3Y CAGR", ascending=False))

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(sec.reset_index(), x="Sector", y="Mean Latest 3Y CAGR",
                     color="Mean Latest 3Y CAGR", color_continuous_scale="RdYlGn")
        fig.update_layout(yaxis_tickformat=".1%", height=420,
                          margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(sec.reset_index(), x="Sector", y="Mean 10Y Total Return",
                     color="Mean 10Y Total Return", color_continuous_scale="RdYlGn")
        fig.update_layout(yaxis_tickformat=".0%", height=420,
                          margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Mean 3Y rolling CAGR over time — by sector**")
    sector_map = meta.set_index("Ticker")["Sector"]
    rolling_filt = rolling[filt_tickers].copy()
    sector_ts = rolling_filt.T.groupby(sector_map).mean().T
    sector_ts = sector_ts.loc[d0:d1]
    fig = px.line(sector_ts)
    fig.update_layout(yaxis_tickformat=".0%", height=500, hovermode="x unified",
                      margin=dict(l=10, r=10, t=30, b=10),
                      legend_title_text="Sector")
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.4)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Sector summary**")
    st.dataframe(
        sec.style.format({
            "Mean 10Y Total Return":   "{:.1%}",
            "Median 10Y Total Return": "{:.1%}",
            "Mean Latest 3Y CAGR":     "{:.2%}",
            "Median Latest 3Y CAGR":   "{:.2%}",
            "Mean 3Y Volatility":      "{:.2%}",
        }),
        use_container_width=True,
    )

# ---------- TAB 4: RANKINGS ----------
with tab4:
    st.subheader("Top / bottom performers")

    metric_choice = st.radio(
        "Rank by",
        ["Latest 3Y CAGR", "Mean 3Y CAGR", "Total Return (10y)", "Std 3Y CAGR"],
        horizontal=True,
    )
    top_n = st.slider("How many", 5, 50, 20)

    ranked = filt_summary.dropna(subset=[metric_choice]).sort_values(metric_choice, ascending=False)
    top    = ranked.head(top_n)
    bottom = ranked.tail(top_n).iloc[::-1]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**🏆 Top {top_n}**")
        fig = px.bar(top, x=metric_choice, y="Ticker", color="Sector",
                     orientation="h",
                     hover_data={"Company": True, metric_choice: ":.2%"})
        fig.update_layout(height=max(400, 25 * top_n),
                          yaxis={'categoryorder': 'total ascending'},
                          xaxis_tickformat=".1%",
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown(f"**📉 Bottom {top_n}**")
        fig = px.bar(bottom, x=metric_choice, y="Ticker", color="Sector",
                     orientation="h",
                     hover_data={"Company": True, metric_choice: ":.2%"})
        fig.update_layout(height=max(400, 25 * top_n),
                          yaxis={'categoryorder': 'total descending'},
                          xaxis_tickformat=".1%",
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("**Full ranked table**")
    display_df = ranked.copy()
    for col in ["Total Return (10y)", "Latest 3Y CAGR", "Mean 3Y CAGR",
                "Median 3Y CAGR", "Std 3Y CAGR", "Min 3Y CAGR", "Max 3Y CAGR"]:
        display_df[col] = (display_df[col] * 100).round(2)
    st.dataframe(display_df, use_container_width=True, height=500)

# ---------- TAB 5: DOWNLOAD ----------
with tab5:
    st.subheader("Download data")

    st.markdown("Export the underlying and derived datasets as CSV.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "📄 Summary (per ticker)",
            data=summary.to_csv(index=False).encode(),
            file_name="sp500_summary.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "📄 3Y Rolling CAGR (daily)",
            data=rolling.to_csv().encode(),
            file_name="sp500_3yr_rolling.csv",
            mime="text/csv",
        )
    with c3:
        st.download_button(
            "📄 Adjusted close prices",
            data=prices.to_csv().encode(),
            file_name="sp500_prices.csv",
            mime="text/csv",
        )

    st.divider()
    st.markdown("**Summary preview**")
    st.dataframe(summary.head(20), use_container_width=True)
