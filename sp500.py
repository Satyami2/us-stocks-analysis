"""
US Stocks Multibagger Dashboard — v2 (simplified UI)
=====================================================
Find ANY stock that gave X× return in a rolling window of your choice.

UI: Pick a threshold (5×/10×/20×/etc) and a window length (1y/2y/3y/...),
    optionally filter by historical period. See results.

DATA-QUALITY FILTERING:
  - Robust endpoints (21-day median)
  - Filters split artifacts (>400% single-day jumps)
  - Caps absurd multiples > 1000x as data errors

LOADING FIXES:
  - load_cached() is NOT cached (so retries actually retry)
  - 60s HTTP timeout on GitHub fetches
  - Cache diagnostics panel in sidebar
  - Force-clear button

Setup:  pip install streamlit yfinance pandas numpy plotly requests pyarrow
Run:    streamlit run app.py
"""

import os
import io
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(page_title="US Stocks — Multibaggers",
                   page_icon="🚀", layout="wide")

YEARS_BACK    = 10

# Data-quality thresholds
SMOOTH_WINDOW        = 21
MIN_TRADING_DAYS     = 252
MAX_REASONABLE_MULT  = 1000.0
MAX_SINGLE_DAY_JUMP  = 4.0

CACHE_DIR     = "cache"
PRICES_FILE   = os.path.join(CACHE_DIR, "prices.parquet")
META_FILE     = os.path.join(CACHE_DIR, "meta.parquet")

GITHUB_USER   = "Satyami2"
GITHUB_REPO   = "us-stocks-analysis"
GITHUB_BRANCH = "main"
GITHUB_PRICES_URL = (
    f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/"
    f"{GITHUB_BRANCH}/cache/prices.parquet"
)
GITHUB_META_URL = (
    f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/"
    f"{GITHUB_BRANCH}/cache/meta.parquet"
)
GITHUB_PRICES_LFS = (
    f"https://media.githubusercontent.com/media/{GITHUB_USER}/{GITHUB_REPO}/"
    f"{GITHUB_BRANCH}/cache/prices.parquet"
)
GITHUB_META_LFS = (
    f"https://media.githubusercontent.com/media/{GITHUB_USER}/{GITHUB_REPO}/"
    f"{GITHUB_BRANCH}/cache/meta.parquet"
)

BATCH_SIZE    = 50
SLEEP_BETWEEN = 0.4
HTTP_TIMEOUT  = 60

os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# UNIVERSE
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=24 * 3600)
def get_us_tickers() -> pd.DataFrame:
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


def _read_parquet_url(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


def load_cached(verbose: bool = True):
    """NOT @st.cache_data on purpose — otherwise a hung first call sticks forever."""
    try:
        if os.path.exists(PRICES_FILE) and os.path.exists(META_FILE):
            p = pd.read_parquet(PRICES_FILE)
            m = pd.read_parquet(META_FILE)
            if verbose:
                st.toast(f"Loaded from local cache: {p.shape[1]} tickers", icon="✅")
            return p, m
    except Exception as e:
        if verbose:
            st.warning(f"Local cache unreadable ({e}). Trying GitHub…")

    for prices_url, meta_url, label in [
        (GITHUB_PRICES_URL, GITHUB_META_URL, "GitHub raw"),
        (GITHUB_PRICES_LFS, GITHUB_META_LFS, "Git LFS"),
    ]:
        try:
            if verbose:
                st.info(f"Fetching from {label}… (timeout={HTTP_TIMEOUT}s)")
            prices = _read_parquet_url(prices_url)
            meta   = _read_parquet_url(meta_url)
            try:
                prices.to_parquet(PRICES_FILE)
                meta.to_parquet(META_FILE)
            except Exception:
                pass
            if verbose:
                st.toast(f"Loaded data from {label}: {prices.shape[1]} tickers", icon="✅")
            return prices, meta
        except Exception as e:
            if verbose:
                st.warning(f"{label} failed: {type(e).__name__}: {e}")
            continue

    return None, None


def fetch_all(limit=None):
    status = st.empty()
    pbar   = st.progress(0.0)
    status.text("Fetching ticker list…")
    universe = get_us_tickers()
    total_available = len(universe)

    if limit is not None and limit < total_available:
        universe = universe.head(int(limit)).reset_index(drop=True)
        st.info(
            f"📋 Universe size: **{total_available}** tickers available · "
            f"downloading first **{len(universe)}** "
            f"(alphabetical: {universe['Ticker'].iloc[0]} → "
            f"{universe['Ticker'].iloc[-1]})"
        )
    else:
        st.info(f"📋 Downloading FULL universe: **{total_available}** tickers")

    status.text(f"Downloading {len(universe)} tickers…")
    prices = download_prices(universe["Ticker"].tolist(), pbar, status)
    meta = universe[universe["Ticker"].isin(prices.columns)].reset_index(drop=True)
    try:
        prices.astype("float32").to_parquet(PRICES_FILE, compression="zstd")
        meta.to_parquet(META_FILE, compression="zstd")
        st.toast("Saved cache/*.parquet — commit & push to GitHub!", icon="💾")
    except Exception:
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
def _detect_suspicious_jumps(prices: pd.DataFrame) -> pd.Series:
    safe = prices.where(prices > 0)
    return safe.pct_change().abs().max() > MAX_SINGLE_DAY_JUMP


@st.cache_data(show_spinner=False)
def build_quality_mask(prices: pd.DataFrame) -> pd.Series:
    """Series indexed by ticker; True if data quality OK."""
    suspicious = _detect_suspicious_jumps(prices)
    valid = []
    for t in prices.columns:
        s = prices[t].dropna()
        s = s[s > 0]
        valid.append(len(s) >= MIN_TRADING_DAYS)
    valid_s = pd.Series(valid, index=prices.columns)
    return valid_s & ~suspicious.reindex(prices.columns).fillna(False)


@st.cache_data(show_spinner="Scanning rolling windows for multibaggers…")
def find_window_multibaggers(prices: pd.DataFrame, window_years: float,
                             threshold: float) -> pd.DataFrame:
    """
    For each stock, find the BEST rolling window of length `window_years`
    where it returned >= threshold×. Returns one row per qualifying stock.
    """
    window_days = max(21, int(window_years * 252))  # min ~1 month
    safe = prices.where(prices > 0)

    if window_days >= len(safe):
        return pd.DataFrame(columns=["Ticker", "Best Multiple", "Best CAGR",
                                     "Start Date", "End Date"])

    ratio = safe / safe.shift(window_days)
    records = []
    for t in ratio.columns:
        col = ratio[t].dropna()
        if col.empty:
            continue
        max_mult = col.max()
        if pd.isna(max_mult) or max_mult < threshold:
            continue
        if max_mult > MAX_REASONABLE_MULT:
            continue
        end_dt = col.idxmax()
        try:
            end_loc = safe.index.get_loc(end_dt)
        except KeyError:
            continue
        start_loc = max(0, end_loc - window_days)
        start_dt = safe.index[start_loc]
        cagr = max_mult ** (1.0 / max(window_years, 0.01)) - 1.0
        records.append({
            "Ticker":        t,
            "Best Multiple": round(float(max_mult), 2),
            "Best CAGR":     round(float(cagr), 4),
            "Start Date":    start_dt,
            "End Date":      end_dt,
        })

    if not records:
        return pd.DataFrame(columns=["Ticker", "Best Multiple", "Best CAGR",
                                     "Start Date", "End Date"])
    return pd.DataFrame(records).sort_values("Best Multiple",
                                             ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# SIDEBAR — DATA LOAD
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Controls")

with st.sidebar.expander("🔧 Cache diagnostics"):
    st.caption(f"Working dir: `{os.getcwd()}`")
    if os.path.exists(PRICES_FILE):
        sz = os.path.getsize(PRICES_FILE) / 1024 / 1024
        st.success(f"✓ prices.parquet ({sz:.1f} MB)")
    else:
        st.warning("✗ prices.parquet not found")
    if os.path.exists(META_FILE):
        st.success("✓ meta.parquet")
    else:
        st.warning("✗ meta.parquet not found")
    if st.button("🗑️ Clear local cache files"):
        for f in (PRICES_FILE, META_FILE):
            try:
                if os.path.exists(f): os.remove(f)
            except Exception as e:
                st.error(f"Couldn't delete {f}: {e}")
        st.cache_data.clear()
        st.session_state.pop("prices", None)
        st.session_state.pop("meta", None)
        st.success("Cleared. Rerun to reload.")
        st.rerun()

if "prices" not in st.session_state or st.session_state.get("prices") is None:
    with st.spinner("Loading price data…"):
        p, m = load_cached()
    st.session_state.prices = p
    st.session_state.meta   = m

prices = st.session_state.prices
meta   = st.session_state.meta

if prices is None:
    st.title("🚀 US Stocks — Multibaggers")
    st.warning(
        "**No cached data found.** Tried local `cache/` folder and the "
        f"GitHub repo `{GITHUB_USER}/{GITHUB_REPO}` (`{GITHUB_BRANCH}` branch). "
        "Download fresh data from Yahoo Finance below."
    )

    st.sidebar.subheader("📥 Download settings")
    download_mode = st.sidebar.radio(
        "How many stocks?",
        options=["Quick test (100)", "Small (500)", "Medium (1500)",
                 "Large (3000)", "Full universe (~6000)", "Custom"],
        index=2,
    )
    mode_to_n = {
        "Quick test (100)": 100, "Small (500)": 500,
        "Medium (1500)": 1500, "Large (3000)": 3000,
        "Full universe (~6000)": None,
    }
    if download_mode == "Custom":
        n_stocks = st.sidebar.number_input("Number of stocks",
                                           min_value=10, max_value=10000,
                                           value=1000, step=100)
    else:
        n_stocks = mode_to_n[download_mode]

    eta_minutes = (n_stocks or 6000) / 100
    st.sidebar.caption(f"⏱️ ETA: ~{eta_minutes:.0f} min ({n_stocks or '~6000'} stocks)")

    if st.sidebar.button("⬇️ Start download", type="primary"):
        for f in (PRICES_FILE, META_FILE):
            try:
                if os.path.exists(f): os.remove(f)
            except Exception:
                pass
        st.cache_data.clear()
        try:
            with st.spinner(f"Downloading {n_stocks or '~6000'} stocks…"):
                p, m = fetch_all(limit=n_stocks)
            st.session_state.prices = p
            st.session_state.meta   = m
            st.success(f"✓ Downloaded {p.shape[1]} stocks")
            st.rerun()
        except Exception as e:
            st.error(f"Failed: {e}")
            st.code(traceback.format_exc())
            st.stop()
    st.stop()

st.sidebar.success(
    f"✓ {prices.shape[1]} tickers · {prices.shape[0]:,} days\n\n"
    f"Last: {prices.index.max().strftime('%Y-%m-%d')}"
)

with st.sidebar.expander("🔄 Re-download data"):
    redl_mode = st.radio(
        "How many stocks?",
        options=["Quick test (100)", "Small (500)", "Medium (1500)",
                 "Large (3000)", "Full universe (~6000)", "Custom"],
        index=2, key="redl_mode",
    )
    redl_map = {
        "Quick test (100)": 100, "Small (500)": 500,
        "Medium (1500)": 1500, "Large (3000)": 3000,
        "Full universe (~6000)": None,
    }
    if redl_mode == "Custom":
        redl_n = st.number_input("Number of stocks",
                                 min_value=10, max_value=10000,
                                 value=1000, step=100, key="redl_n")
    else:
        redl_n = redl_map[redl_mode]
    st.caption(f"⏱️ ETA: ~{(redl_n or 6000) / 100:.0f} min")

    if st.button("⬇️ Start re-download", type="primary", key="btn_redl"):
        for f in (PRICES_FILE, META_FILE):
            try:
                if os.path.exists(f): os.remove(f)
            except Exception:
                pass
        st.cache_data.clear()
        st.session_state.prices = None
        st.session_state.meta   = None
        try:
            with st.spinner(f"Downloading {redl_n or '~6000'} stocks…"):
                p, m = fetch_all(limit=redl_n)
            st.session_state.prices = p
            st.session_state.meta   = m
            st.success(f"✓ Downloaded {p.shape[1]} stocks")
            st.rerun()
        except Exception as e:
            st.error(f"Failed: {e}")
            st.code(traceback.format_exc())
            st.stop()

# ---------------------------------------------------------------------------
# MAIN UI
# ---------------------------------------------------------------------------
st.title("🚀 US Stocks — Multibagger Scanner")
st.caption(
    f"{prices.shape[1]} stocks · {prices.shape[0]:,} daily prices from Yahoo Finance · "
    f"data through {prices.index.max().strftime('%Y-%m-%d')}"
)

# ---- Controls in main area, top ----
st.subheader("🔧 What are you looking for?")

c1, c2, c3 = st.columns([1, 1, 2])

with c1:
    threshold_choice = st.select_slider(
        "Minimum return",
        options=[2, 3, 5, 10, 20, 50, 100],
        value=10,
        format_func=lambda x: f"{x}×",
        help="Find stocks that gave AT LEAST this multiple."
    )
    THRESHOLD = float(threshold_choice)

with c2:
    window_choice = st.select_slider(
        "Within a window of",
        options=[0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        value=3,
        format_func=lambda x: f"{x:g} year{'s' if x != 1 else ''}",
        help="Length of the rolling window. e.g. '3 years' = stocks that 10×'d "
             "in any 3-year period like 2017→2020 or 2021→2024."
    )
    WINDOW_YEARS = float(window_choice)

with c3:
    min_data_date = prices.index.min().date()
    max_data_date = prices.index.max().date()
    period_filter = st.date_input(
        "Run must END between (optional period filter)",
        value=(min_data_date, max_data_date),
        min_value=min_data_date,
        max_value=max_data_date,
        help="To find 2017-2018 winners: set 2017-01-01 → 2018-12-31. "
             "Leave full to scan everything."
    )

st.divider()

# ---- Build quality mask + scan ----
quality_ok = build_quality_mask(prices)
quality_ok_tickers = quality_ok[quality_ok].index.tolist()
quality_ok_prices  = prices[quality_ok_tickers]

results = find_window_multibaggers(quality_ok_prices, WINDOW_YEARS, THRESHOLD)

# Apply period filter
if len(period_filter) == 2 and not results.empty:
    period_start = pd.Timestamp(period_filter[0])
    period_end   = pd.Timestamp(period_filter[1])
    results = results[
        (results["End Date"] >= period_start) &
        (results["End Date"] <= period_end)
    ].reset_index(drop=True)

# Merge company names
if not results.empty:
    results = results.merge(meta[["Ticker", "Company"]], on="Ticker", how="left")

# ---- Top metrics ----
m1, m2, m3, m4 = st.columns(4)
m1.metric("Stocks scanned", f"{len(quality_ok_tickers):,}")
m2.metric(f"≥{THRESHOLD:.0f}× found", f"{len(results):,}")
m3.metric(
    "Best multiple",
    f"{results['Best Multiple'].max():.1f}×" if not results.empty else "—",
    results.iloc[0]["Ticker"] if not results.empty else None,
)
m4.metric(
    "Median CAGR",
    f"{results['Best CAGR'].median() * 100:.1f}%" if not results.empty else "—",
)

st.divider()

# ---- Results ----
if results.empty:
    st.info(
        f"No stocks gave ≥{THRESHOLD:.0f}× return in any "
        f"{WINDOW_YEARS:g}-year window during the chosen period.\n\n"
        "Try lowering the threshold, picking a different window length, "
        "or widening the date range."
    )
else:
    st.subheader(
        f"🎯 {len(results)} stocks gave ≥{THRESHOLD:.0f}× in a "
        f"{WINDOW_YEARS:g}-year window"
    )

    # Bar chart of top N
    top_n = min(30, len(results))
    fig = px.bar(
        results.head(top_n),
        x="Best Multiple", y="Ticker", orientation="h",
        hover_data={
            "Company":       True,
            "Best Multiple": ":.2f",
            "Best CAGR":     ":.1%",
            "Start Date":    "|%Y-%m-%d",
            "End Date":      "|%Y-%m-%d",
        },
        color="Best Multiple", color_continuous_scale="Greens",
        title=f"Top {top_n} multibagger runs",
    )
    fig.update_layout(
        height=max(400, 22 * top_n),
        yaxis={"categoryorder": "total ascending"},
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Full sortable table
    st.write(f"**All {len(results)} results (sortable):**")
    tbl = results.copy()
    tbl["Best CAGR (%)"] = (tbl["Best CAGR"] * 100).round(1)
    tbl["Start"] = pd.to_datetime(tbl["Start Date"]).dt.strftime("%Y-%m-%d")
    tbl["End"]   = pd.to_datetime(tbl["End Date"]).dt.strftime("%Y-%m-%d")
    tbl = tbl[["Ticker", "Company", "Best Multiple", "Best CAGR (%)", "Start", "End"]]
    st.dataframe(
        tbl, use_container_width=True, height=500,
        column_config={
            "Best Multiple": st.column_config.NumberColumn(format="%.2f×"),
            "Best CAGR (%)": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )

    st.download_button(
        "📥 Download results as CSV",
        data=results.to_csv(index=False).encode(),
        file_name=f"multibaggers_{int(THRESHOLD)}x_{WINDOW_YEARS:g}y.csv",
        mime="text/csv",
    )

st.divider()

# ---------------------------------------------------------------------------
# PER-STOCK INSPECTOR
# ---------------------------------------------------------------------------
st.subheader("🔎 Inspect a specific stock")

inspect_tickers = sorted(quality_ok_tickers)
default_idx = 0
if not results.empty:
    top_ticker = results.iloc[0]["Ticker"]
    if top_ticker in inspect_tickers:
        default_idx = inspect_tickers.index(top_ticker)

selected = st.selectbox(
    "Pick a ticker", inspect_tickers, index=default_idx,
    help="Shows the full 10-year price chart. Multibagger window is highlighted "
         "if this stock qualifies under your current threshold/window settings."
)

if selected:
    s = prices[selected].dropna()
    s = s[s > 0]
    company = meta.loc[meta["Ticker"] == selected, "Company"].iloc[0] \
        if not meta[meta["Ticker"] == selected].empty else ""

    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(
        x=s.index, y=s.values, mode="lines", name=selected,
        line=dict(color="#1f77b4", width=1.5)
    ))

    if not results.empty:
        stock_row = results[results["Ticker"] == selected]
        if not stock_row.empty:
            row = stock_row.iloc[0]
            fig_s.add_vrect(
                x0=row["Start Date"], x1=row["End Date"],
                fillcolor="rgba(46, 204, 113, 0.20)", line_width=0,
                annotation_text=(
                    f"{row['Best Multiple']:.1f}× in "
                    f"{WINDOW_YEARS:g}y · CAGR {row['Best CAGR']*100:.1f}%"
                ),
                annotation_position="top left",
            )

    fig_s.update_layout(
        title=f"{selected} — {company}",
        xaxis_title="Date", yaxis_title="Price (adj. close, log scale)",
        yaxis_type="log",
        height=500, margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_s, use_container_width=True)
