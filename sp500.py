"""
US Stocks Multibagger Dashboard
================================
For every US stock (full Russell 3000-style universe):
  - Multibagger flag (10x+ total return)
  - Median 1-year rolling return
  - Median 3-year rolling CAGR
  - Per-stock rolling returns chart on demand

Setup:
    pip install streamlit yfinance pandas numpy plotly requests pyarrow

Run:
    streamlit run app.py

NOTE: First-time download for the full universe (~5000-6000 tickers) takes
30-60 minutes and may hit Yahoo Finance rate limits, especially on Streamlit
Cloud's free tier. Run locally for the full universe; commit the resulting
cache/*.parquet files to your repo for cloud deploys.
"""

import os
import io
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.express as px

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(page_title="US Stocks — Multibaggers",
                   page_icon="🚀", layout="wide")

YEARS_BACK    = 10
W_1Y          = 252
W_3Y          = 756
MULTIBAGGER_X = 10.0

CACHE_DIR     = "cache"
PRICES_FILE   = os.path.join(CACHE_DIR, "prices.parquet")
META_FILE     = os.path.join(CACHE_DIR, "meta.parquet")

BATCH_SIZE    = 50
SLEEP_BETWEEN = 0.4

os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# UNIVERSE
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=24 * 3600)
def get_us_tickers() -> pd.DataFrame:
    """All NYSE + NASDAQ common stocks (excludes ETFs, warrants, units)."""
    headers = {"User-Agent": "Mozilla/5.0"}

    def _fetch(url):
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        return r.text

    frames = []
    try:
        nas = pd.read_csv(io.StringIO(_fetch(
            "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt")),
            sep="|")
        nas = nas[(nas["Test Issue"] == "N") & (nas["ETF"] == "N")]
        nas = nas.rename(columns={"Symbol": "Ticker",
                                  "Security Name": "Company"})
        frames.append(nas[["Ticker", "Company"]])
    except Exception:
        pass

    try:
        oth = pd.read_csv(io.StringIO(_fetch(
            "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt")),
            sep="|")
        oth = oth[(oth["Test Issue"] == "N") & (oth["ETF"] == "N")]
        oth = oth.rename(columns={"ACT Symbol": "Ticker",
                                  "Security Name": "Company"})
        frames.append(oth[["Ticker", "Company"]])
    except Exception:
        pass

    if not frames:
        raise RuntimeError("Couldn't fetch ticker lists.")

    df = pd.concat(frames, ignore_index=True).dropna(subset=["Ticker"])
    df = df[df["Ticker"].astype(str).str.match(r"^[A-Z][A-Z0-9.\-]*$")]
    df["Ticker"] = df["Ticker"].str.replace(".", "-", regex=False)
    df = df[~df["Ticker"].str.endswith(("W", "WS", "U", "R", "RT", "P"))]
    bad = "WARRANT|UNIT|PREFERRED|RIGHT|DEPOSITARY|ETF|TRUST|FUND"
    df = df[~df["Company"].str.upper().str.contains(bad, na=False)]
    return df.drop_duplicates(subset=["Ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# DOWNLOAD
# ---------------------------------------------------------------------------
def download_batch(tickers, start, end):
    try:
        data = yf.download(tickers=" ".join(tickers), start=start, end=end,
                           interval="1d", auto_adjust=True, progress=False,
                           threads=True, group_by="ticker")
        if data is None or data.empty:
            return pd.DataFrame()
        closes = {}
        if len(tickers) == 1:
            t = tickers[0]
            if "Close" in data.columns:
                closes[t] = data["Close"]
        else:
            top = data.columns.get_level_values(0)
            for t in tickers:
                if t in top:
                    try:
                        closes[t] = data[t]["Close"]
                    except Exception:
                        continue
        return pd.DataFrame(closes)
    except Exception:
        return pd.DataFrame()


def download_prices(tickers, pbar, status):
    end_date   = datetime.today()
    start_date = end_date.replace(year=end_date.year - YEARS_BACK)
    start = start_date.strftime("%Y-%m-%d")
    end   = end_date.strftime("%Y-%m-%d")

    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    frames = []
    n_ok = 0
    for i, batch in enumerate(batches):
        status.text(f"Batch {i + 1}/{len(batches)} · {batch[0]}…{batch[-1]} · "
                    f"{n_ok} batches loaded")
        df = download_batch(batch, start, end)
        if not df.empty:
            frames.append(df); n_ok += 1
        pbar.progress((i + 1) / len(batches))
        time.sleep(SLEEP_BETWEEN)

    if not frames:
        raise RuntimeError("No data downloaded.")
    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    return prices.sort_index().dropna(how="all")


def load_cached():
    try:
        if os.path.exists(PRICES_FILE) and os.path.exists(META_FILE):
            return pd.read_parquet(PRICES_FILE), pd.read_parquet(META_FILE)
    except Exception:
        pass
    return None, None


def fetch_all():
    """Download the full universe."""
    status = st.empty()
    pbar   = st.progress(0.0)
    status.text("Fetching ticker list…")
    universe = get_us_tickers()
    status.text(f"Universe size: {len(universe)} tickers · downloading…")
    prices = download_prices(universe["Ticker"].tolist(), pbar, status)
    meta = universe[universe["Ticker"].isin(prices.columns)].reset_index(drop=True)
    try:
        prices.to_parquet(PRICES_FILE)
        meta.to_parquet(META_FILE)
    except Exception:
        pass
    pbar.empty(); status.empty()
    return prices, meta


# ---------------------------------------------------------------------------
# CALCULATIONS
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_rolling(prices: pd.DataFrame, window: int, years: float) -> pd.DataFrame:
    """Daily annualised rolling CAGR over `window` trading days."""
    return ((prices / prices.shift(window)).pow(1.0 / years) - 1.0)


@st.cache_data(show_spinner=False)
def build_summary(prices: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    r1 = compute_rolling(prices, W_1Y, 1.0)
    r3 = compute_rolling(prices, W_3Y, 3.0)

    first_idx = prices.apply(lambda s: s.first_valid_index())
    last_idx  = prices.apply(lambda s: s.last_valid_index())
    first_px  = pd.Series({t: prices[t].loc[first_idx[t]]
                           if pd.notna(first_idx[t]) else np.nan
                           for t in prices.columns})
    last_px   = pd.Series({t: prices[t].loc[last_idx[t]]
                           if pd.notna(last_idx[t]) else np.nan
                           for t in prices.columns})
    multiple  = last_px / first_px
    years     = (last_idx - first_idx).dt.days / 365.25

    df = pd.DataFrame({
        "First Date":       first_idx,
        "Last Date":        last_idx,
        "Years of Data":    years.round(1),
        "First Price":      first_px.round(2),
        "Last Price":       last_px.round(2),
        "Multiple (x)":     multiple.round(2),
        "Median 1Y Return": r1.median().round(4),
        "Median 3Y CAGR":   r3.median().round(4),
    })
    df.index.name = "Ticker"
    df = df.reset_index().merge(meta, on="Ticker", how="left")
    df["Multibagger"] = df["Multiple (x)"] >= MULTIBAGGER_X
    return df[["Ticker", "Company", "Multibagger", "Multiple (x)",
               "Median 1Y Return", "Median 3Y CAGR",
               "Years of Data", "First Date", "Last Date",
               "First Price", "Last Price"]]


# ---------------------------------------------------------------------------
# SIDEBAR — DATA LOAD
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Controls")

if "prices" not in st.session_state:
    p, m = load_cached()
    st.session_state.prices = p
    st.session_state.meta   = m

prices = st.session_state.prices
meta   = st.session_state.meta

if prices is None:
    st.title("🚀 US Stocks — Multibaggers")
    st.warning(
        "**Heads-up:** downloading the full US-listed universe "
        "(~5000-6000 stocks) takes 30-60 minutes and may hit Yahoo Finance "
        "rate limits — especially on Streamlit Cloud free tier. "
        "Recommended: run locally and copy `cache/prices.parquet` and "
        "`cache/meta.parquet` to your deployed repo."
    )
    if st.sidebar.button("⬇️ Download ALL US stocks", type="primary"):
        try:
            with st.spinner("Downloading…"):
                p, m = fetch_all()
            st.session_state.prices = p
            st.session_state.meta   = m
            st.rerun()
        except Exception as e:
            st.error(f"Failed: {e}")
            st.stop()
    st.stop()

st.sidebar.success(
    f"✓ {prices.shape[1]} tickers · {prices.shape[0]:,} days\n\n"
    f"Last: {prices.index.max().strftime('%Y-%m-%d')}"
)
if st.sidebar.button("🔄 Re-download"):
    st.session_state.prices = None
    st.session_state.meta   = None
    st.cache_data.clear()
    st.rerun()

# ---------------------------------------------------------------------------
# SUMMARY + FILTERS
# ---------------------------------------------------------------------------
summary = build_summary(prices, meta)

st.sidebar.divider()
multibagger_only = st.sidebar.checkbox(f"🚀 Multibaggers only (≥{MULTIBAGGER_X:.0f}×)")
min_years = st.sidebar.slider("Min years of history", 1, 10, 3)

view = summary[summary["Years of Data"].fillna(0) >= min_years]
if multibagger_only:
    view = view[view["Multibagger"]]

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
st.title("🚀 US Stocks — Multibaggers")
st.caption(f"{prices.shape[1]} stocks · 10y daily prices from Yahoo Finance · "
           f"Multibagger = {MULTIBAGGER_X:.0f}× total return")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Stocks in view", f"{len(view):,}")
c2.metric("Multibaggers (10×+)", f"{int(view['Multibagger'].sum()):,}",
          f"{view['Multibagger'].mean() * 100:.1f}%")
c3.metric("Median 1Y return",
          f"{view['Median 1Y Return'].median() * 100:.2f}%"
          if view['Median 1Y Return'].notna().any() else "—")
c4.metric("Median 3Y CAGR",
          f"{view['Median 3Y CAGR'].median() * 100:.2f}%"
          if view['Median 3Y CAGR'].notna().any() else "—")

st.divider()

# ---------------------------------------------------------------------------
# PER-STOCK ROLLING RETURNS CHART
# ---------------------------------------------------------------------------
st.subheader("📈 Per-stock rolling returns")

avail = view["Ticker"].sort_values().tolist()
if not avail:
    st.info("No stocks match current filters.")
else:
    default_pick = "AAPL" if "AAPL" in avail else avail[0]
    picked = st.selectbox(
        "Choose a stock",
        options=avail,
        index=avail.index(default_pick),
        format_func=lambda t: (
            f"{t} — {summary.loc[summary['Ticker'] == t, 'Company'].iloc[0]}"
            f"{' 🚀' if summary.loc[summary['Ticker'] == t, 'Multibagger'].iloc[0] else ''}"
        ),
    )

    row = summary[summary["Ticker"] == picked].iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Multiple",
              f"{row['Multiple (x)']:.2f}×" if pd.notna(row['Multiple (x)']) else "—",
              "🚀 Multibagger" if row["Multibagger"] else None)
    m2.metric("Median 1Y Return",
              f"{row['Median 1Y Return'] * 100:.2f}%"
              if pd.notna(row['Median 1Y Return']) else "—")
    m3.metric("Median 3Y CAGR",
              f"{row['Median 3Y CAGR'] * 100:.2f}%"
              if pd.notna(row['Median 3Y CAGR']) else "—")
    m4.metric("Years of data",
              f"{row['Years of Data']:.1f}"
              if pd.notna(row['Years of Data']) else "—")

    # Compute rolling series for this stock only (fast)
    s = prices[picked].dropna()
    r1_series = ((s / s.shift(W_1Y)) - 1.0).dropna()
    r3_series = ((s / s.shift(W_3Y)).pow(1.0 / 3.0) - 1.0).dropna()

    chart_df = pd.DataFrame({
        "1Y rolling return": r1_series,
        "3Y rolling CAGR":   r3_series,
    }).dropna(how="all")

    if chart_df.empty:
        st.info("Not enough history to compute rolling returns for this stock.")
    else:
        fig = px.line(chart_df, labels={"value": "Return", "variable": ""})
        fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.4)
        # Median reference lines
        if pd.notna(row["Median 1Y Return"]):
            fig.add_hline(y=row["Median 1Y Return"], line_dash="dot",
                          line_color="#1f77b4", opacity=0.5,
                          annotation_text=f"Median 1Y "
                                          f"{row['Median 1Y Return']*100:.1f}%",
                          annotation_position="top left")
        if pd.notna(row["Median 3Y CAGR"]):
            fig.add_hline(y=row["Median 3Y CAGR"], line_dash="dot",
                          line_color="#ff7f0e", opacity=0.5,
                          annotation_text=f"Median 3Y "
                                          f"{row['Median 3Y CAGR']*100:.1f}%",
                          annotation_position="bottom left")
        fig.update_layout(height=480, yaxis_tickformat=".0%",
                          hovermode="x unified",
                          margin=dict(l=10, r=10, t=30, b=10),
                          legend=dict(orientation="h", yanchor="bottom",
                                      y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# TOP MULTIBAGGERS
# ---------------------------------------------------------------------------
mb = view[view["Multibagger"]].sort_values("Multiple (x)", ascending=False)
if not mb.empty:
    st.subheader(f"🚀 Top multibaggers ({len(mb)} found)")
    top_n = min(30, len(mb))
    fig = px.bar(mb.head(top_n), x="Multiple (x)", y="Ticker",
                 orientation="h",
                 hover_data={"Company": True, "Multiple (x)": ":.2f"},
                 color="Multiple (x)", color_continuous_scale="Greens")
    fig.update_layout(height=max(400, 22 * top_n),
                      yaxis={'categoryorder': 'total ascending'},
                      margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# FULL TABLE
# ---------------------------------------------------------------------------
st.subheader("📊 All stocks — sortable")

display = view.copy().sort_values("Multiple (x)", ascending=False)
display["Median 1Y Return"] = (display["Median 1Y Return"] * 100).round(2)
display["Median 3Y CAGR"]   = (display["Median 3Y CAGR"] * 100).round(2)
display["Multibagger"]      = display["Multibagger"].map({True: "🚀", False: ""})

st.dataframe(
    display.rename(columns={
        "Median 1Y Return": "Median 1Y Return (%)",
        "Median 3Y CAGR":   "Median 3Y CAGR (%)",
    }),
    use_container_width=True,
    height=600,
    column_config={
        "Multiple (x)": st.column_config.NumberColumn(format="%.2f×"),
        "First Price":  st.column_config.NumberColumn(format="$%.2f"),
        "Last Price":   st.column_config.NumberColumn(format="$%.2f"),
    },
)

st.download_button(
    "📥 Download as CSV",
    data=view.to_csv(index=False).encode(),
    file_name="us_stocks_multibaggers.csv",
    mime="text/csv",
)
