"""
US Stocks Multibagger Dashboard — Streamlit
============================================
~3000 Russell-3000 stocks · 10 years of daily prices · 1Y/3Y/5Y rolling CAGR
+ multibagger flagging (10x+) and market-cap tier highlighting.

Setup:
    pip install streamlit yfinance pandas numpy plotly lxml requests pyarrow

Run:
    streamlit run us_stocks_multibagger_app.py

Streamlit Cloud (requirements.txt):
    streamlit>=1.30
    yfinance>=0.2.40
    pandas>=2.0
    numpy>=1.24
    plotly>=5.18
    lxml>=4.9
    requests>=2.31
    pyarrow>=14.0

IMPORTANT: First-time download for ~3000 tickers takes 30-60 minutes and may
hit Yahoo rate limits. Use the universe-size selector to start smaller.
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
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="US Stocks Multibagger Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
YEARS_BACK    = 10
W_1Y          = 252           # ~1 trading year
W_3Y          = 756           # ~3 trading years
W_5Y          = 1260          # ~5 trading years
MULTIBAGGER_X = 10.0          # 10x+ over 10 years

CACHE_DIR     = "us_cache"
PRICES_FILE   = os.path.join(CACHE_DIR, "prices.parquet")
META_FILE     = os.path.join(CACHE_DIR, "meta.parquet")
MCAP_FILE     = os.path.join(CACHE_DIR, "mcap.parquet")

BATCH_SIZE    = 50
SLEEP_BETWEEN = 0.4

os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# UNIVERSE — Russell 3000 sources
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=24 * 3600)
def get_russell3000_tickers() -> pd.DataFrame:
    """
    Build a Russell 3000-style universe.
    Strategy:
      1. NASDAQ Trader's official nasdaqlisted.txt + otherlisted.txt (NYSE/AMEX)
      2. Filter out test issues, ETFs, warrants, units, preferred shares
    Result: ~5500-6500 common stocks across NYSE/NASDAQ/AMEX.
    We then take the top ~3000 by liquidity proxy (priced > $1).
    """
    headers = {"User-Agent": "Mozilla/5.0"}

    def _fetch(url):
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        return r.text

    # NASDAQ-listed
    try:
        nas_txt = _fetch("https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt")
        nas = pd.read_csv(io.StringIO(nas_txt), sep="|")
        nas = nas[nas["Test Issue"] == "N"]
        nas = nas[nas["ETF"] == "N"]
        nas = nas.rename(columns={"Symbol": "Ticker", "Security Name": "Company"})
        nas["Exchange"] = "NASDAQ"
        nas = nas[["Ticker", "Company", "Exchange"]]
    except Exception:
        nas = pd.DataFrame(columns=["Ticker", "Company", "Exchange"])

    # NYSE / AMEX-listed
    try:
        oth_txt = _fetch("https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt")
        oth = pd.read_csv(io.StringIO(oth_txt), sep="|")
        oth = oth[oth["Test Issue"] == "N"]
        oth = oth[oth["ETF"] == "N"]
        oth = oth.rename(columns={"ACT Symbol": "Ticker",
                                  "Security Name": "Company",
                                  "Exchange": "ExCode"})
        ex_map = {"N": "NYSE", "A": "NYSE American", "P": "NYSE Arca",
                  "Z": "BATS", "V": "IEX"}
        oth["Exchange"] = oth["ExCode"].map(ex_map).fillna("Other")
        oth = oth[["Ticker", "Company", "Exchange"]]
    except Exception:
        oth = pd.DataFrame(columns=["Ticker", "Company", "Exchange"])

    df = pd.concat([nas, oth], ignore_index=True).dropna(subset=["Ticker"])
    df = df[df["Ticker"].astype(str).str.match(r"^[A-Z][A-Z0-9.\-]*$")]
    # Yahoo uses '-' in place of '.'
    df["Ticker"] = df["Ticker"].str.replace(".", "-", regex=False)
    # Drop suspicious suffixes (warrants, units, rights)
    bad_suffix = ("W", "WS", "U", "R", "RT", "P")
    df = df[~df["Ticker"].str.endswith(bad_suffix)]
    # Strip prefixes like 'WARRANT', 'UNIT', 'PREF' in name
    bad_words = ["WARRANT", "UNIT", "PREFERRED", "RIGHT", "DEPOSITARY",
                 "ETF", "TRUST", "FUND"]
    pat = "|".join(bad_words)
    df = df[~df["Company"].str.upper().str.contains(pat, na=False)]
    df = df.drop_duplicates(subset=["Ticker"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# PRICE DOWNLOAD
# ---------------------------------------------------------------------------
def download_batch(tickers, start, end):
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


def download_prices(tickers, years_back=YEARS_BACK,
                    progress_bar=None, status_text=None):
    end_date   = datetime.today()
    start_date = end_date.replace(year=end_date.year - years_back)
    start_str  = start_date.strftime("%Y-%m-%d")
    end_str    = end_date.strftime("%Y-%m-%d")

    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    frames = []
    n_ok = 0
    for i, batch in enumerate(batches):
        if status_text is not None:
            status_text.text(
                f"Batch {i + 1}/{len(batches)} ({batch[0]}…{batch[-1]}) · "
                f"loaded {n_ok} batches"
            )
        df = download_batch(batch, start_str, end_str)
        if not df.empty:
            frames.append(df); n_ok += 1
        if progress_bar is not None:
            progress_bar.progress((i + 1) / len(batches))
        time.sleep(SLEEP_BETWEEN)

    if not frames:
        raise RuntimeError("No data downloaded. Yahoo may be rate-limiting.")
    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    return prices.sort_index().dropna(how="all")


# ---------------------------------------------------------------------------
# MARKET CAP & SECTOR (yfinance .info — slow, batched)
# ---------------------------------------------------------------------------
def fetch_metadata(tickers, progress_bar=None, status_text=None):
    """Pull market cap, sector, industry via yf.Ticker.info — best effort."""
    rows = []
    for i, t in enumerate(tickers):
        if status_text is not None and i % 25 == 0:
            status_text.text(f"Fetching metadata {i}/{len(tickers)}…")
        if progress_bar is not None and len(tickers) > 0:
            progress_bar.progress((i + 1) / len(tickers))
        try:
            info = yf.Ticker(t).info or {}
            rows.append({
                "Ticker":   t,
                "MarketCap": info.get("marketCap", np.nan),
                "Sector":   info.get("sector", "Unknown"),
                "Industry": info.get("industry", "Unknown"),
            })
        except Exception:
            rows.append({"Ticker": t, "MarketCap": np.nan,
                         "Sector": "Unknown", "Industry": "Unknown"})
        time.sleep(0.05)
    return pd.DataFrame(rows)


def cap_tier(mcap):
    if pd.isna(mcap): return "Unknown"
    if mcap >= 200e9: return "Mega ($200B+)"
    if mcap >= 10e9:  return "Large ($10B-200B)"
    if mcap >= 2e9:   return "Mid ($2B-10B)"
    if mcap >= 300e6: return "Small ($300M-2B)"
    if mcap >= 50e6:  return "Micro ($50M-300M)"
    return "Nano (<$50M)"


# ---------------------------------------------------------------------------
# DISK CACHE
# ---------------------------------------------------------------------------
def load_from_disk():
    try:
        if (os.path.exists(PRICES_FILE) and os.path.exists(META_FILE)
                and os.path.exists(MCAP_FILE)):
            prices = pd.read_parquet(PRICES_FILE)
            meta   = pd.read_parquet(META_FILE)
            mcap   = pd.read_parquet(MCAP_FILE)
            return prices, meta, mcap
    except Exception:
        pass
    return None, None, None


def save_to_disk(prices, meta, mcap):
    try:
        prices.to_parquet(PRICES_FILE)
        meta.to_parquet(META_FILE)
        mcap.to_parquet(MCAP_FILE)
    except Exception:
        pass


def fetch_full_pipeline(n_tickers=None, fetch_mcap=True):
    status = st.empty()
    pbar   = st.progress(0.0)

    status.text("Fetching US listed-stocks universe…")
    universe = get_russell3000_tickers()
    if n_tickers is not None and n_tickers < len(universe):
        universe = universe.head(n_tickers).reset_index(drop=True)
        status.text(f"Universe limited to {n_tickers} tickers · "
                    f"downloading prices…")
    else:
        status.text(f"Universe: {len(universe)} tickers · downloading prices…")

    prices = download_prices(
        universe["Ticker"].tolist(),
        years_back=YEARS_BACK,
        progress_bar=pbar, status_text=status,
    )

    meta = universe[universe["Ticker"].isin(prices.columns)].reset_index(drop=True)

    if fetch_mcap:
        status.text("Fetching market caps & sectors (this is the slow part)…")
        pbar.progress(0.0)
        mcap = fetch_metadata(meta["Ticker"].tolist(),
                              progress_bar=pbar, status_text=status)
    else:
        mcap = pd.DataFrame({"Ticker": meta["Ticker"],
                             "MarketCap": np.nan,
                             "Sector": "Unknown",
                             "Industry": "Unknown"})

    save_to_disk(prices, meta, mcap)
    pbar.empty(); status.empty()
    return prices, meta, mcap


# ---------------------------------------------------------------------------
# CALCULATIONS
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_rolling_cagr(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Daily rolling annualised CAGR for a given window in trading days."""
    years = window / 252.0
    ratio = prices / prices.shift(window)
    return (ratio.pow(1.0 / years) - 1.0).dropna(how="all")


@st.cache_data(show_spinner=False)
def build_summary(prices: pd.DataFrame,
                  meta: pd.DataFrame,
                  mcap: pd.DataFrame,
                  r1: pd.DataFrame,
                  r3: pd.DataFrame,
                  r5: pd.DataFrame) -> pd.DataFrame:
    first_idx = prices.apply(lambda s: s.first_valid_index())
    last_idx  = prices.apply(lambda s: s.last_valid_index())
    first_px  = pd.Series({t: prices[t].loc[first_idx[t]] if pd.notna(first_idx[t]) else np.nan
                           for t in prices.columns})
    last_px   = pd.Series({t: prices[t].loc[last_idx[t]] if pd.notna(last_idx[t]) else np.nan
                           for t in prices.columns})
    total_x   = last_px / first_px            # multiple (e.g. 12.5 = 12.5x)
    total_ret = total_x - 1.0

    # Years of history
    yrs = ((last_idx - first_idx).dt.days / 365.25).rename("Years")

    # Rolling stats
    def stats(df, label):
        return pd.DataFrame({
            f"Latest {label} CAGR": df.iloc[-1] if not df.empty else np.nan,
            f"Median {label} CAGR": df.median(),
            f"Mean {label} CAGR":   df.mean(),
        })

    s1 = stats(r1, "1Y")
    s3 = stats(r3, "3Y")
    s5 = stats(r5, "5Y")

    summ = pd.DataFrame({
        "First Date":  first_idx,
        "Last Date":   last_idx,
        "First Price": first_px.round(2),
        "Last Price":  last_px.round(2),
        "Years":       yrs.round(1),
        "Multiple (x)":   total_x.round(2),
        "Total Return":   total_ret.round(4),
    }).join([s1, s3, s5])

    summ.index.name = "Ticker"
    summ = summ.reset_index().merge(meta, on="Ticker", how="left")
    summ = summ.merge(mcap, on="Ticker", how="left")

    summ["Cap Tier"]    = summ["MarketCap"].apply(cap_tier)
    summ["Multibagger"] = summ["Multiple (x)"] >= MULTIBAGGER_X

    front = ["Ticker", "Company", "Exchange", "Sector", "Industry",
             "Cap Tier", "MarketCap", "Multibagger",
             "Multiple (x)", "Total Return", "Years"]
    rest  = [c for c in summ.columns if c not in front]
    return summ[front + rest]


# ---------------------------------------------------------------------------
# SIDEBAR — DATA CONTROL
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Controls")

if "prices" not in st.session_state:
    p, m, mc = load_from_disk()
    st.session_state.prices = p
    st.session_state.meta   = m
    st.session_state.mcap   = mc

prices = st.session_state.prices
meta   = st.session_state.meta
mcap   = st.session_state.mcap

if prices is None:
    st.title("🚀 US Stocks Multibagger Dashboard")
    st.info(
        "👇 Pick a universe size and download. **Heads up:** the full ~3000-ticker "
        "Russell 3000 download takes 30-60 minutes and may hit Yahoo rate limits "
        "on Streamlit Cloud's free tier. Start small to verify the pipeline, then "
        "scale up."
    )

    universe_choice = st.sidebar.radio(
        "Universe size",
        options=["Top 200 (test run)", "Top 500", "Top 1000",
                 "Top 2000", "Full Russell 3000"],
        index=0,
    )
    size_map = {
        "Top 200 (test run)": 200,
        "Top 500": 500,
        "Top 1000": 1000,
        "Top 2000": 2000,
        "Full Russell 3000": None,
    }
    n = size_map[universe_choice]

    fetch_mcap = st.sidebar.checkbox(
        "Fetch market caps & sectors (slow but enables tier filtering)",
        value=True,
    )

    if st.sidebar.button("⬇️ Download data", type="primary"):
        try:
            with st.spinner("Downloading…"):
                p, m, mc = fetch_full_pipeline(n_tickers=n, fetch_mcap=fetch_mcap)
            st.session_state.prices = p
            st.session_state.meta   = m
            st.session_state.mcap   = mc
            st.sidebar.success(f"Loaded {p.shape[1]} tickers × {p.shape[0]} days")
            st.rerun()
        except Exception as e:
            st.error(f"Download failed: {e}")
            st.stop()
    else:
        st.stop()
else:
    last_date = prices.index.max().strftime("%Y-%m-%d")
    st.sidebar.success(
        f"✓ Data loaded\n\n"
        f"**{prices.shape[1]}** tickers\n\n"
        f"**{prices.shape[0]:,}** days\n\n"
        f"Last date: **{last_date}**"
    )
    if st.sidebar.button("🔄 Re-download"):
        st.session_state.prices = None
        st.session_state.meta   = None
        st.session_state.mcap   = None
        st.cache_data.clear()
        st.rerun()

# Pre-compute derived data
with st.spinner("Computing rolling returns…"):
    r1 = compute_rolling_cagr(prices, W_1Y)
    r3 = compute_rolling_cagr(prices, W_3Y)
    r5 = compute_rolling_cagr(prices, W_5Y)
    summary = build_summary(prices, meta, mcap, r1, r3, r5)

# ---------------------------------------------------------------------------
# SIDEBAR — FILTERS
# ---------------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.subheader("🎯 Filters")

cap_tiers = ["Mega ($200B+)", "Large ($10B-200B)", "Mid ($2B-10B)",
             "Small ($300M-2B)", "Micro ($50M-300M)", "Nano (<$50M)", "Unknown"]
present_tiers = [t for t in cap_tiers if t in summary["Cap Tier"].unique()]
selected_tiers = st.sidebar.multiselect("Market-cap tier", present_tiers,
                                        default=present_tiers)

sectors = sorted([s for s in summary["Sector"].dropna().unique() if s])
selected_sectors = st.sidebar.multiselect("Sector", sectors, default=sectors)

multibagger_only = st.sidebar.checkbox(f"🚀 Multibaggers only (≥{MULTIBAGGER_X:.0f}x)")

min_years = st.sidebar.slider("Min years of history", 1, 10, 3)

filt = summary[
    summary["Cap Tier"].isin(selected_tiers)
    & summary["Sector"].isin(selected_sectors + [None])
    & (summary["Years"].fillna(0) >= min_years)
]
if multibagger_only:
    filt = filt[filt["Multibagger"]]

if filt.empty:
    st.warning("No tickers match the current filters.")
    st.stop()

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
st.title("🚀 US Stocks Multibagger Dashboard")
st.caption(
    f"Russell-3000 universe · daily prices from Yahoo Finance · "
    f"1Y/3Y/5Y rolling CAGR · multibagger ≥ {MULTIBAGGER_X:.0f}× · "
    f"data through {prices.index.max().strftime('%B %d, %Y')}"
)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Overview", "🚀 Multibaggers", "📊 Ticker Explorer",
    "🔥 Sectors & Caps", "🏆 Rankings", "📥 Download"
])

# ---------- TAB 1: OVERVIEW ----------
with tab1:
    st.subheader("Market snapshot")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tickers in view",     f"{len(filt):,}")
    c2.metric("Multibaggers (10x+)", f"{int(filt['Multibagger'].sum()):,}",
              f"{filt['Multibagger'].mean() * 100:.1f}% of view")
    c3.metric("Median 1Y CAGR",
              f"{filt['Median 1Y CAGR'].median() * 100:.2f}%"
              if filt['Median 1Y CAGR'].notna().any() else "—")
    c4.metric("Median 3Y CAGR",
              f"{filt['Median 3Y CAGR'].median() * 100:.2f}%"
              if filt['Median 3Y CAGR'].notna().any() else "—")
    c5.metric("Median 5Y CAGR",
              f"{filt['Median 5Y CAGR'].median() * 100:.2f}%"
              if filt['Median 5Y CAGR'].notna().any() else "—")

    st.divider()

    st.markdown("**Total return multiple — distribution (log scale)**")
    plot_df = filt.dropna(subset=["Multiple (x)"])
    plot_df = plot_df[plot_df["Multiple (x)"] > 0]
    if not plot_df.empty:
        fig = px.histogram(plot_df, x="Multiple (x)", nbins=60,
                           color="Cap Tier", log_x=True)
        fig.add_vline(x=MULTIBAGGER_X, line_dash="dash", line_color="red",
                      annotation_text=f"{MULTIBAGGER_X:.0f}x cutoff")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                          xaxis_title="Total return multiple (log)")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("**Rolling CAGR distributions**")
    c1, c2, c3 = st.columns(3)
    for col, label, container in [("Latest 1Y CAGR", "1Y", c1),
                                  ("Latest 3Y CAGR", "3Y", c2),
                                  ("Latest 5Y CAGR", "5Y", c3)]:
        with container:
            sub = filt.dropna(subset=[col])
            if not sub.empty:
                fig = px.histogram(sub, x=col, nbins=50, color="Cap Tier")
                fig.add_vline(x=0, line_dash="dash", line_color="grey")
                fig.update_layout(height=320, xaxis_tickformat=".0%",
                                  margin=dict(l=10, r=10, t=30, b=10),
                                  showlegend=False, title=f"{label} CAGR")
                st.plotly_chart(fig, use_container_width=True)

# ---------- TAB 2: MULTIBAGGERS ----------
with tab2:
    st.subheader(f"🚀 Multibaggers (≥{MULTIBAGGER_X:.0f}× over their life)")

    mb = filt[filt["Multibagger"]].sort_values("Multiple (x)", ascending=False)

    if mb.empty:
        st.info("No multibaggers in the current filter view.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Multibaggers", f"{len(mb):,}")
        c2.metric("Best multiple", f"{mb['Multiple (x)'].max():.1f}×",
                  delta=mb.iloc[0]["Ticker"])
        c3.metric("Median multiple", f"{mb['Multiple (x)'].median():.1f}×")
        c4.metric("Median 5Y CAGR",
                  f"{mb['Median 5Y CAGR'].median() * 100:.1f}%"
                  if mb['Median 5Y CAGR'].notna().any() else "—")

        st.divider()
        st.markdown("**Top 30 multibaggers**")
        top30 = mb.head(30)
        fig = px.bar(top30, x="Multiple (x)", y="Ticker",
                     color="Cap Tier", orientation="h",
                     hover_data={"Company": True, "Sector": True,
                                 "Multiple (x)": ":.1f"})
        fig.update_layout(height=720,
                          yaxis={'categoryorder': 'total ascending'},
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("**Sector mix of multibaggers**")
        sec_mix = (mb.groupby(["Sector", "Cap Tier"]).size()
                   .reset_index(name="Count"))
        if not sec_mix.empty:
            fig = px.bar(sec_mix, x="Sector", y="Count", color="Cap Tier",
                         barmode="stack")
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("**Full multibagger list**")
        show_cols = ["Ticker", "Company", "Sector", "Cap Tier", "MarketCap",
                     "Multiple (x)", "Years",
                     "Median 1Y CAGR", "Median 3Y CAGR", "Median 5Y CAGR"]
        disp = mb[show_cols].copy()
        for c in ["Median 1Y CAGR", "Median 3Y CAGR", "Median 5Y CAGR"]:
            disp[c] = (disp[c] * 100).round(2)
        disp["MarketCap"] = (disp["MarketCap"] / 1e9).round(2)
        disp = disp.rename(columns={"MarketCap": "MarketCap ($B)"})
        st.dataframe(disp, use_container_width=True, height=520)

# ---------- TAB 3: TICKER EXPLORER ----------
with tab3:
    st.subheader("Individual ticker analysis")

    avail = filt["Ticker"].tolist()
    default_pick = "AAPL" if "AAPL" in avail else avail[0]
    col1, col2 = st.columns([2, 1])
    with col1:
        ticker = st.selectbox("Choose a ticker", options=avail,
                              index=avail.index(default_pick))
    with col2:
        compare_opts = [t for t in avail if t != ticker]
        compare = st.multiselect("Compare against", options=compare_opts, default=[])

    row = summary[summary["Ticker"] == ticker].iloc[0]
    badge = "🚀 Multibagger" if row["Multibagger"] else ""
    st.markdown(f"### {row['Company']} ({ticker}) {badge}")
    st.caption(f"{row['Exchange']} · {row['Sector']} · {row['Industry']} · "
               f"{row['Cap Tier']}")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Last Price",
              f"${row['Last Price']:.2f}" if pd.notna(row['Last Price']) else "—")
    m2.metric("Total Multiple",
              f"{row['Multiple (x)']:.2f}×" if pd.notna(row['Multiple (x)']) else "—")
    m3.metric("Latest 1Y CAGR",
              f"{row['Latest 1Y CAGR'] * 100:.1f}%"
              if pd.notna(row['Latest 1Y CAGR']) else "—")
    m4.metric("Latest 3Y CAGR",
              f"{row['Latest 3Y CAGR'] * 100:.1f}%"
              if pd.notna(row['Latest 3Y CAGR']) else "—")
    m5.metric("Latest 5Y CAGR",
              f"{row['Latest 5Y CAGR'] * 100:.1f}%"
              if pd.notna(row['Latest 5Y CAGR']) else "—")

    tickers_to_plot = [ticker] + compare

    st.markdown("**Normalised price (start = 100, log scale)**")
    px_win = prices[tickers_to_plot].dropna(how="all")
    if not px_win.empty:
        norm = px_win.apply(lambda s: s / s.dropna().iloc[0] * 100
                            if s.dropna().size else s)
        fig = px.line(norm, log_y=True,
                      labels={"value": "Indexed (log)", "variable": "Ticker"})
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                          hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**1Y / 3Y / 5Y rolling CAGR (annualised)**")
    available_cols = [t for t in tickers_to_plot if t in r1.columns]
    if available_cols:
        for window_df, lbl in [(r1, "1Y"), (r3, "3Y"), (r5, "5Y")]:
            sub = window_df[available_cols].dropna(how="all")
            if not sub.empty:
                fig = px.line(sub, labels={"value": f"{lbl} CAGR",
                                           "variable": "Ticker"})
                fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.4)
                fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10),
                                  yaxis_tickformat=".0%", hovermode="x unified",
                                  title=f"{lbl} rolling CAGR")
                st.plotly_chart(fig, use_container_width=True)

# ---------- TAB 4: SECTORS & CAPS ----------
with tab4:
    st.subheader("Sector & market-cap breakdown")

    st.markdown("**Multibagger rate by sector**")
    sec_stats = (filt.groupby("Sector")
                 .agg(Tickers=("Ticker", "count"),
                      Multibaggers=("Multibagger", "sum"),
                      MultibaggerRate=("Multibagger", "mean"),
                      MedianMultiple=("Multiple (x)", "median"),
                      Median1Y=("Median 1Y CAGR", "median"),
                      Median3Y=("Median 3Y CAGR", "median"),
                      Median5Y=("Median 5Y CAGR", "median"))
                 .sort_values("MultibaggerRate", ascending=False))

    plot_df = sec_stats.reset_index()
    if not plot_df.empty:
        fig = px.bar(plot_df, x="Sector", y="MultibaggerRate",
                     color="Multibaggers",
                     hover_data={"Tickers": True, "Multibaggers": True,
                                 "MultibaggerRate": ":.1%"},
                     color_continuous_scale="Greens")
        fig.update_layout(yaxis_tickformat=".1%", height=420,
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Multibagger rate by market-cap tier**")
    cap_stats = (filt.groupby("Cap Tier")
                 .agg(Tickers=("Ticker", "count"),
                      Multibaggers=("Multibagger", "sum"),
                      MultibaggerRate=("Multibagger", "mean"),
                      MedianMultiple=("Multiple (x)", "median"))
                 .reindex(cap_tiers).dropna(subset=["Tickers"]))
    plot_df = cap_stats.reset_index()
    if not plot_df.empty:
        fig = px.bar(plot_df, x="Cap Tier", y="MultibaggerRate",
                     color="Multibaggers",
                     hover_data={"Tickers": True, "MedianMultiple": ":.2f"},
                     color_continuous_scale="Blues")
        fig.update_layout(yaxis_tickformat=".1%", height=380,
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Sector summary table**")
    disp = sec_stats.copy()
    for c in ["Median1Y", "Median3Y", "Median5Y", "MultibaggerRate"]:
        disp[c] = (disp[c] * 100).round(2)
    disp["MedianMultiple"] = disp["MedianMultiple"].round(2)
    st.dataframe(disp, use_container_width=True)

# ---------- TAB 5: RANKINGS ----------
with tab5:
    st.subheader("Top performers")

    metric_choice = st.radio(
        "Rank by",
        ["Multiple (x)", "Median 1Y CAGR", "Median 3Y CAGR",
         "Median 5Y CAGR", "Latest 1Y CAGR", "Latest 3Y CAGR",
         "Latest 5Y CAGR"],
        horizontal=True,
    )
    top_n = st.slider("How many", 10, 100, 30)

    ranked = filt.dropna(subset=[metric_choice]).sort_values(
        metric_choice, ascending=False)
    if ranked.empty:
        st.info("No data for ranking.")
    else:
        top = ranked.head(top_n)
        fig = px.bar(top, x=metric_choice, y="Ticker", color="Cap Tier",
                     orientation="h",
                     hover_data={"Company": True, "Sector": True,
                                 "Multiple (x)": ":.2f"})
        is_pct = metric_choice != "Multiple (x)"
        fig.update_layout(
            height=max(500, 22 * top_n),
            yaxis={'categoryorder': 'total ascending'},
            xaxis_tickformat=".1%" if is_pct else None,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Full ranked table**")
        disp = ranked.copy()
        for c in ["Latest 1Y CAGR", "Median 1Y CAGR", "Mean 1Y CAGR",
                  "Latest 3Y CAGR", "Median 3Y CAGR", "Mean 3Y CAGR",
                  "Latest 5Y CAGR", "Median 5Y CAGR", "Mean 5Y CAGR",
                  "Total Return"]:
            if c in disp.columns:
                disp[c] = (disp[c] * 100).round(2)
        if "MarketCap" in disp.columns:
            disp["MarketCap ($B)"] = (disp["MarketCap"] / 1e9).round(2)
            disp = disp.drop(columns=["MarketCap"])
        st.dataframe(disp, use_container_width=True, height=520)

# ---------- TAB 6: DOWNLOAD ----------
with tab6:
    st.subheader("Download data")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button("📄 Summary (filtered)",
                           data=filt.to_csv(index=False).encode(),
                           file_name="us_stocks_summary_filtered.csv",
                           mime="text/csv")
    with c2:
        st.download_button("📄 Summary (all)",
                           data=summary.to_csv(index=False).encode(),
                           file_name="us_stocks_summary_all.csv",
                           mime="text/csv")
    with c3:
        st.download_button("📄 Adjusted close prices",
                           data=prices.to_csv().encode(),
                           file_name="us_stocks_prices.csv",
                           mime="text/csv")
    with c4:
        mb = summary[summary["Multibagger"]]
        st.download_button("🚀 Multibaggers only",
                           data=mb.to_csv(index=False).encode(),
                           file_name="us_stocks_multibaggers.csv",
                           mime="text/csv")

    st.divider()
    st.markdown("**Filtered summary preview**")
    st.dataframe(filt.head(50), use_container_width=True)
