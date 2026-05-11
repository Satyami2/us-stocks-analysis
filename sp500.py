"""
US Stocks Multibagger Dashboard
================================
For every US stock (full Russell 3000-style universe):
  - Multibagger flag (10x+ total return) WITH data-quality filtering
  - Median 1-year rolling return
  - Median 3-year rolling CAGR
  - Per-stock rolling returns chart on demand

DATA-QUALITY FIXES (prevents false multibaggers):
  - Robust endpoints: uses 21-day median, not single-day prints
  - Filters sub-$1 starting prices (reverse-split / penny-stock artifacts)
  - Detects suspicious single-day jumps > 400% (split artifacts)
  - Caps absurd multiples > 1000x as data errors
  - Requires minimum trading-day history

Setup:
    pip install streamlit yfinance pandas numpy plotly requests pyarrow

Run:
    streamlit run app.py
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

# Data-quality thresholds
MIN_START_PRICE      = 1.00   # ignore sub-$1 starting prices (penny/split artifacts)
SMOOTH_WINDOW        = 21     # use 21-day median for "first" and "last" prices
MIN_TRADING_DAYS     = 252    # need at least 1 year of real data
MAX_REASONABLE_MULT  = 1000.0 # cap absurd multiples (data error sentinel)
MAX_SINGLE_DAY_JUMP  = 4.0    # 400% single-day = almost certainly bad data

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


def _read_parquet_url(url: str) -> pd.DataFrame:
    """Download a parquet file from a URL and read into a DataFrame."""
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


@st.cache_data(show_spinner="Loading price data…", ttl=24 * 3600)
def load_cached():
    """Try local cache first, then GitHub raw, then Git LFS."""
    try:
        if os.path.exists(PRICES_FILE) and os.path.exists(META_FILE):
            return pd.read_parquet(PRICES_FILE), pd.read_parquet(META_FILE)
    except Exception as e:
        st.warning(f"Local cache unreadable ({e}). Trying GitHub…")

    for prices_url, meta_url, label in [
        (GITHUB_PRICES_URL, GITHUB_META_URL, "GitHub raw"),
        (GITHUB_PRICES_LFS, GITHUB_META_LFS, "Git LFS"),
    ]:
        try:
            prices = _read_parquet_url(prices_url)
            meta   = _read_parquet_url(meta_url)
            try:
                prices.to_parquet(PRICES_FILE)
                meta.to_parquet(META_FILE)
            except Exception:
                pass
            st.toast(f"Loaded data from {label}", icon="✅")
            return prices, meta
        except Exception:
            continue

    return None, None


def fetch_all(limit=None):
    """Download from Yahoo Finance. limit=None means full universe."""
    status = st.empty()
    pbar   = st.progress(0.0)
    status.text("Fetching ticker list…")
    universe = get_us_tickers()
    if limit is not None and limit < len(universe):
        universe = universe.head(limit).reset_index(drop=True)
        status.text(f"Universe limited to {len(universe)} tickers · downloading…")
    else:
        status.text(f"Universe size: {len(universe)} tickers · downloading…")
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
# CALCULATIONS (with data-quality fixes)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_rolling(prices: pd.DataFrame, window: int, years: float) -> pd.DataFrame:
    """Daily annualised rolling CAGR over `window` trading days."""
    # Mask non-positive prices to avoid garbage rolling returns
    safe = prices.where(prices > 0)
    return ((safe / safe.shift(window)).pow(1.0 / years) - 1.0)


def _robust_endpoints(s: pd.Series, window: int = SMOOTH_WINDOW):
    """
    Return (first_date, first_price, last_date, last_price) using a
    rolling-median smoothing on both ends to kill single-print artifacts.
    """
    s = s.dropna()
    s = s[s > 0]  # drop zeros / negatives
    if len(s) < max(window * 2, MIN_TRADING_DAYS):
        return (pd.NaT, np.nan, pd.NaT, np.nan)

    first_window = s.iloc[:window]
    last_window  = s.iloc[-window:]
    first_px = float(first_window.median())
    last_px  = float(last_window.median())
    first_dt = first_window.index[0]
    last_dt  = last_window.index[-1]
    return (first_dt, first_px, last_dt, last_px)


def _detect_suspicious_jumps(prices: pd.DataFrame) -> pd.Series:
    """
    Detect tickers with single-day moves > MAX_SINGLE_DAY_JUMP (400%).
    These are almost always reverse-split artifacts or bad data, not real returns.
    """
    safe = prices.where(prices > 0)
    daily_ret = safe.pct_change()
    max_jump  = daily_ret.abs().max()
    return max_jump > MAX_SINGLE_DAY_JUMP


@st.cache_data(show_spinner=False)
def build_summary(prices: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    r1 = compute_rolling(prices, W_1Y, 1.0)
    r3 = compute_rolling(prices, W_3Y, 3.0)

    # Robust endpoints (21-day median, not single prints)
    records = []
    for t in prices.columns:
        first_dt, first_px, last_dt, last_px = _robust_endpoints(prices[t])
        records.append((t, first_dt, first_px, last_dt, last_px))

    eps = pd.DataFrame(records, columns=["Ticker", "First Date", "First Price",
                                         "Last Date", "Last Price"])
    eps = eps.set_index("Ticker")

    multiple = eps["Last Price"] / eps["First Price"]
    years    = (eps["Last Date"] - eps["First Date"]).dt.days / 365.25

    # ---- DATA-QUALITY FLAGS ----
    too_cheap_start = eps["First Price"].fillna(0) < MIN_START_PRICE
    too_short       = years.fillna(0) < (MIN_TRADING_DAYS / 252)
    absurd_multiple = multiple.fillna(0) > MAX_REASONABLE_MULT
    no_data         = eps["First Price"].isna() | eps["Last Price"].isna()

    suspicious_jump = _detect_suspicious_jumps(prices)
    suspicious_jump = suspicious_jump.reindex(eps.index).fillna(False)

    quality_bad = (too_cheap_start | too_short | absurd_multiple |
                   no_data | suspicious_jump)

    # Per-flag reasons for transparency
    reasons = []
    for t in eps.index:
        rs = []
        if no_data.loc[t]:                  rs.append("no data")
        if too_short.loc[t]:                rs.append("<1y history")
        if too_cheap_start.loc[t]:          rs.append(f"start <${MIN_START_PRICE:.0f}")
        if absurd_multiple.loc[t]:          rs.append(f">{int(MAX_REASONABLE_MULT)}x mult")
        if suspicious_jump.loc[t]:          rs.append("split artifact")
        reasons.append(", ".join(rs) if rs else "")

    df = pd.DataFrame({
        "First Date":       eps["First Date"],
        "Last Date":        eps["Last Date"],
        "Years of Data":    years.round(1),
        "First Price":      eps["First Price"].round(2),
        "Last Price":       eps["Last Price"].round(2),
        "Multiple (x)":     multiple.round(2),
        "Median 1Y Return": r1.median().round(4),
        "Median 3Y CAGR":   r3.median().round(4),
        "Data Quality OK":  ~quality_bad,
        "Quality Issues":   reasons,
    })
    df.index.name = "Ticker"
    df = df.reset_index().merge(meta, on="Ticker", how="left")

    # Multibagger ONLY if data quality is OK
    df["Multibagger"] = (df["Multiple (x)"] >= MULTIBAGGER_X) & df["Data Quality OK"]

    return df[["Ticker", "Company", "Multibagger", "Multiple (x)",
               "Median 1Y Return", "Median 3Y CAGR",
               "Years of Data", "First Date", "Last Date",
               "First Price", "Last Price",
               "Data Quality OK", "Quality Issues"]]


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
        "**No cached data found.** Tried local `cache/` folder and the "
        f"GitHub repo `{GITHUB_USER}/{GITHUB_REPO}` (`{GITHUB_BRANCH}` branch) "
        "— neither had the parquet files.\n\n"
        "Download fresh data from Yahoo Finance below. "
        "You can limit how many stocks to fetch to control download time."
    )

    st.sidebar.subheader("📥 Download settings")
    download_mode = st.sidebar.radio(
        "How many stocks?",
        options=["Quick test (100)", "Small (500)", "Medium (1500)",
                 "Large (3000)", "Full universe (~6000)", "Custom"],
        index=2,
        help="Smaller = faster. Full universe takes 30-60 minutes."
    )
    mode_to_n = {
        "Quick test (100)":         100,
        "Small (500)":              500,
        "Medium (1500)":            1500,
        "Large (3000)":              3000,
        "Full universe (~6000)":   None,  # None = all
    }
    if download_mode == "Custom":
        n_stocks = st.sidebar.number_input(
            "Number of stocks",
            min_value=10, max_value=10000, value=1000, step=100,
            help="Tickers are sampled in alphabetical order."
        )
    else:
        n_stocks = mode_to_n[download_mode]

    eta_minutes = (n_stocks or 6000) / 100  # ~rough estimate: 100 stocks/min
    st.sidebar.caption(
        f"⏱️ Estimated time: **~{eta_minutes:.0f} min** "
        f"({n_stocks or '~6000'} stocks)"
    )

    if st.sidebar.button("⬇️ Start download", type="primary"):
        try:
            with st.spinner("Downloading…"):
                p, m = fetch_all(limit=n_stocks)
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

with st.sidebar.expander("🔄 Re-download data"):
    redl_mode = st.radio(
        "How many stocks?",
        options=["Quick test (100)", "Small (500)", "Medium (1500)",
                 "Large (3000)", "Full universe (~6000)", "Custom"],
        index=2,
        key="redl_mode",
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

    if st.button("⬇️ Start re-download", type="primary"):
        st.session_state.prices = None
        st.session_state.meta   = None
        st.cache_data.clear()
        try:
            with st.spinner("Downloading…"):
                p, m = fetch_all(limit=redl_n)
            st.session_state.prices = p
            st.session_state.meta   = m
            st.rerun()
        except Exception as e:
            st.error(f"Failed: {e}")
            st.stop()

with st.sidebar.expander("ℹ️ Data source"):
    st.markdown(
        f"- **Repo:** [{GITHUB_USER}/{GITHUB_REPO}]"
        f"(https://github.com/{GITHUB_USER}/{GITHUB_REPO})\n"
        f"- **Branch:** `{GITHUB_BRANCH}`\n"
        f"- Load order: local → GitHub raw → Git LFS → manual download"
    )

# ---------------------------------------------------------------------------
# SUMMARY + FILTERS
# ---------------------------------------------------------------------------
summary = build_summary(prices, meta)

st.sidebar.divider()
st.sidebar.subheader("🧹 Data quality")
hide_bad_quality = st.sidebar.checkbox(
    "Hide bad-quality stocks",
    value=True,
    help="Hides stocks with sub-$1 starting prices, reverse-split artifacts, "
         "insufficient history, or absurd multiples."
)

with st.sidebar.expander("Advanced quality settings"):
    st.caption(
        f"**Filters applied:**\n"
        f"- Start price ≥ ${MIN_START_PRICE:.2f}\n"
        f"- ≥ {MIN_TRADING_DAYS} trading days\n"
        f"- Multiple ≤ {int(MAX_REASONABLE_MULT)}×\n"
        f"- No single-day jump > {int(MAX_SINGLE_DAY_JUMP*100)}%\n"
        f"- Uses 21-day median for first/last price"
    )

st.sidebar.divider()
multibagger_only = st.sidebar.checkbox(f"🚀 Multibaggers only (≥{MULTIBAGGER_X:.0f}×)")
min_years = st.sidebar.slider("Min years of history", 1, 10, 3)

view = summary.copy()
if hide_bad_quality:
    view = view[view["Data Quality OK"]]
view = view[view["Years of Data"].fillna(0) >= min_years]
if multibagger_only:
    view = view[view["Multibagger"]]

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
st.title("🚀 US Stocks — Multibaggers")
st.caption(f"{prices.shape[1]} stocks · 10y daily prices from Yahoo Finance · "
           f"Multibagger = {MULTIBAGGER_X:.0f}× total return · "
           "Data-quality filtered")

# Top summary metrics
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Stocks in view", f"{len(view):,}")
c2.metric("Multibaggers (10×+)", f"{int(view['Multibagger'].sum()):,}",
          f"{view['Multibagger'].mean() * 100:.1f}%" if len(view) else "—")
c3.metric("Median 1Y return",
          f"{view['Median 1Y Return'].median() * 100:.2f}%"
          if view['Median 1Y Return'].notna().any() else "—")
c4.metric("Median 3Y CAGR",
          f"{view['Median 3Y CAGR'].median() * 100:.2f}%"
          if view['Median 3Y CAGR'].notna().any() else "—")
n_filtered_out = int((~summary["Data Quality OK"]).sum())
c5.metric("Filtered out (bad data)", f"{n_filtered_out:,}",
          f"{n_filtered_out / len(summary) * 100:.1f}%" if len(summary) else "—",
          delta_color="off")

st.divider()

# ---------------------------------------------------------------------------
# TOP MULTIBAGGERS
# ---------------------------------------------------------------------------
mb = view[view["Multibagger"]].sort_values("Multiple (x)", ascending=False)
if not mb.empty:
    st.subheader(f"🚀 Top multibaggers ({len(mb)} found, data-quality verified)")
    top_n = min(30, len(mb))
    fig = px.bar(mb.head(top_n), x="Multiple (x)", y="Ticker",
                 orientation="h",
                 hover_data={"Company": True, "Multiple (x)": ":.2f",
                             "Years of Data": True},
                 color="Multiple (x)", color_continuous_scale="Greens")
    fig.update_layout(height=max(400, 22 * top_n),
                      yaxis={'categoryorder': 'total ascending'},
                      margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# FILTERED-OUT INSPECTION (transparency)
# ---------------------------------------------------------------------------
with st.expander(f"🔍 Inspect filtered-out stocks ({n_filtered_out:,} hidden)"):
    st.caption("These stocks were excluded due to data-quality issues. "
               "Common reasons: reverse-split artifacts, sub-$1 starting prices, "
               "insufficient history, or absurd multiples (likely bad data).")
    bad = summary[~summary["Data Quality OK"]].copy()
    bad = bad.sort_values("Multiple (x)", ascending=False)
    bad_display = bad[["Ticker", "Company", "Quality Issues", "Multiple (x)",
                       "First Price", "Last Price", "Years of Data"]]
    st.dataframe(bad_display, use_container_width=True, height=400,
                 column_config={
                     "Multiple (x)": st.column_config.NumberColumn(format="%.2f×"),
                     "First Price":  st.column_config.NumberColumn(format="$%.2f"),
                     "Last Price":   st.column_config.NumberColumn(format="$%.2f"),
                 })

st.divider()

# ---------------------------------------------------------------------------
# FULL TABLE
# ---------------------------------------------------------------------------
st.subheader("📊 All stocks — sortable")

display = view.copy().sort_values("Multiple (x)", ascending=False)
display["Median 1Y Return"] = (display["Median 1Y Return"] * 100).round(2)
display["Median 3Y CAGR"]   = (display["Median 3Y CAGR"] * 100).round(2)
display["Multibagger"]      = display["Multibagger"].map({True: "🚀", False: ""})

# Drop internal columns from view
display = display.drop(columns=["Data Quality OK", "Quality Issues"], errors="ignore")

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
