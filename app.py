"""
Indian Stock Rebound Scanner — Streamlit App
============================================
Scans NSE equity universe for stocks that have rebounded the most from
their recent low over a user-selected lookback window.

Deploy on Streamlit Cloud:
    1. Push these files to a GitHub repo:
         - app.py
         - requirements.txt
         - EQUITY_L.csv  (optional; users can also upload)
    2. Go to https://share.streamlit.io
    3. "New app" -> select repo + branch + app.py
    4. Deploy.
"""

import io
import time
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import yfinance as yf

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Indian Stock Rebound Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- STYLING ----------------
st.markdown(
    """
    <style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@400;600;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'JetBrains Mono', monospace;
    }
    h1, h2, h3 {
        font-family: 'Fraunces', serif !important;
        letter-spacing: -0.02em;
    }
    .main-title {
        font-family: 'Fraunces', serif;
        font-weight: 800;
        font-size: 2.8rem;
        line-height: 1;
        margin: 0;
    }
    .subtitle {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-top: 0.4rem;
    }
    .metric-box {
        border: 1px solid #e5e7eb;
        padding: 1rem 1.2rem;
        border-radius: 2px;
        background: #fafaf9;
    }
    .metric-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #6b7280;
    }
    .metric-value {
        font-family: 'Fraunces', serif;
        font-size: 1.8rem;
        font-weight: 600;
        color: #111827;
    }
    .divider {
        height: 1px;
        background: linear-gradient(to right, #111, transparent);
        margin: 1.5rem 0;
    }
    [data-testid="stSidebar"] {
        background: #0f172a;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- HEADER ----------------
col_a, col_b = st.columns([3, 1])
with col_a:
    st.markdown('<div class="main-title">Rebound Scanner</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">NSE Equity · Low-to-Latest Momentum Analysis</div>',
        unsafe_allow_html=True,
    )
with col_b:
    st.markdown(
        f'<div style="text-align:right; padding-top:1rem; font-size:0.8rem; color:#6b7280;">'
        f'Run date<br><b style="color:#111; font-size:1rem;">{datetime.now().strftime("%d %b %Y")}</b>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ---------------- SIDEBAR CONTROLS ----------------
st.sidebar.header("Scan Parameters")

uploaded_file = st.sidebar.file_uploader(
    "EQUITY_L.csv (optional)",
    type=["csv"],
    help="NSE symbol list. If not uploaded, app looks for EQUITY_L.csv in repo root.",
)

lookback_days = st.sidebar.slider(
    "Lookback window (days)",
    min_value=20,
    max_value=180,
    value=60,
    step=5,
    help="Calendar days of history to pull. 60 ≈ 2 months.",
)

min_rebound_days = st.sidebar.slider(
    "Minimum days since low",
    min_value=1,
    max_value=30,
    value=3,
    help="Excludes stocks whose low is too recent to be considered a rebound.",
)

max_stocks = st.sidebar.number_input(
    "Max stocks to scan",
    min_value=50,
    max_value=3000,
    value=500,
    step=50,
    help="Start small (500) for a quick scan. Full universe ~2,300 stocks takes 15-25 min.",
)

batch_size = st.sidebar.slider(
    "Batch size",
    min_value=20,
    max_value=100,
    value=50,
    step=10,
    help="Stocks downloaded per yfinance call. Lower = gentler on API.",
)

min_fall_filter = st.sidebar.number_input(
    "Require fall ≥ (%)",
    min_value=0.0,
    max_value=50.0,
    value=5.0,
    step=1.0,
    help="Only show stocks that fell at least this much before rebounding.",
)

st.sidebar.markdown("---")
run_button = st.sidebar.button("▶ Run Scan", type="primary", use_container_width=True)

# ---------------- HELPERS ----------------
@st.cache_data(show_spinner=False)
def load_symbols(file_bytes: bytes | None) -> list[str]:
    if file_bytes is not None:
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = pd.read_csv("EQUITY_L.csv")
    df.columns = [c.strip() for c in df.columns]
    df = df[df["SERIES"].str.strip() == "EQ"]
    return df["SYMBOL"].str.strip().tolist()


def download_batch(tickers, start, end):
    """Download one batch; return dict {ticker: close_series}."""
    try:
        data = yf.download(
            tickers,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
            group_by="ticker",
            threads=True,
        )
    except Exception:
        return {}

    out = {}
    if data.empty:
        return out

    if isinstance(data.columns, pd.MultiIndex):
        for t in data.columns.get_level_values(0).unique():
            try:
                s = data[t]["Close"].dropna()
                if len(s) > 0:
                    out[t] = s
            except KeyError:
                continue
    else:
        # single ticker case
        s = data["Close"].dropna()
        if len(s) > 0:
            out[tickers[0]] = s
    return out


def compute_rebound_table(closes: dict, min_days: int) -> pd.DataFrame:
    rows = []
    for ticker, s in closes.items():
        if len(s) < min_days + 2:
            continue
        latest_price = s.iloc[-1]
        latest_date = s.index[-1]
        low_price = s.min()
        low_date = s.idxmin()
        days_since_low = (latest_date - low_date).days
        if days_since_low < min_days:
            continue

        pre_low = s.loc[:low_date]
        peak_before_low = pre_low.max()
        peak_date = pre_low.idxmax()
        fall_pct = (low_price - peak_before_low) / peak_before_low * 100
        rebound_pct = (latest_price - low_price) / low_price * 100
        recovery_of_fall = (
            (latest_price - low_price) / (peak_before_low - low_price) * 100
            if peak_before_low > low_price
            else 0.0
        )

        rows.append({
            "Symbol": ticker.replace(".NS", ""),
            "Peak ₹": round(peak_before_low, 2),
            "Peak Date": peak_date.strftime("%d-%b"),
            "Low ₹": round(low_price, 2),
            "Low Date": low_date.strftime("%d-%b"),
            "Latest ₹": round(latest_price, 2),
            "Fall %": round(fall_pct, 2),
            "Rebound %": round(rebound_pct, 2),
            "Recovery of Fall %": round(recovery_of_fall, 2),
            "Days Since Low": days_since_low,
        })
    return pd.DataFrame(rows)


# ---------------- MAIN BODY ----------------
if not run_button:
    st.info(
        "**How this works** — The scanner pulls daily closing prices from Yahoo Finance for "
        "every NSE equity symbol in your `EQUITY_L.csv`, finds each stock's lowest close in the "
        "lookback window, and calculates the % rebound from that low to the latest close. "
        "Stocks that are still making new lows are excluded (controlled by *Minimum days since low*). "
        "Configure the parameters in the sidebar and click **Run Scan**."
    )

    with st.expander("Understanding the output"):
        st.markdown(
            """
            - **Fall %** — how far the stock fell from its pre-low peak to its low (negative number).
            - **Rebound %** — how much it has risen off the low since.
            - **Recovery of Fall %** — what fraction of the drawdown has been recovered.
              *100% = back to the peak. <50% = still weak bounce.*
            - **Days Since Low** — how long the bounce has been going.

            A large Rebound % with a large Fall % and low Recovery of Fall % is often a **dead-cat bounce**.
            A large Rebound % with high Recovery of Fall % is usually a **real recovery**.
            """
        )
    st.stop()

# ---------------- RUN THE SCAN ----------------
try:
    file_bytes = uploaded_file.getvalue() if uploaded_file else None
    symbols = load_symbols(file_bytes)
except FileNotFoundError:
    st.error("EQUITY_L.csv not found. Please upload it via the sidebar.")
    st.stop()
except Exception as e:
    st.error(f"Could not read symbol file: {e}")
    st.stop()

symbols = symbols[: int(max_stocks)]
tickers = [s + ".NS" for s in symbols]

end = datetime.now()
start = end - timedelta(days=lookback_days + 10)

st.markdown(f"**Scanning {len(tickers)} symbols** · {start.date()} → {end.date()}")

progress = st.progress(0.0, text="Starting download...")
status = st.empty()

all_closes = {}
total_batches = (len(tickers) - 1) // batch_size + 1

t0 = time.time()
for i in range(0, len(tickers), batch_size):
    batch = tickers[i : i + batch_size]
    batch_num = i // batch_size + 1
    status.markdown(
        f"Batch **{batch_num}/{total_batches}** — "
        f"tickers {i + 1}–{min(i + batch_size, len(tickers))}"
    )
    batch_data = download_batch(batch, start, end)
    all_closes.update(batch_data)
    progress.progress(batch_num / total_batches, text=f"Downloaded {len(all_closes)} stocks so far")
    time.sleep(0.5)  # polite delay

elapsed = time.time() - t0
status.success(f"Download complete · {len(all_closes)} stocks · {elapsed:.0f}s elapsed")

if not all_closes:
    st.error("No data retrieved. Yahoo Finance may be rate-limiting. Try a smaller batch size.")
    st.stop()

# ---------------- COMPUTE & DISPLAY ----------------
ranking = compute_rebound_table(all_closes, min_rebound_days)

if ranking.empty:
    st.warning("No stocks matched the criteria. Try loosening the filters.")
    st.stop()

# Apply fall filter
ranking_filtered = ranking[ranking["Fall %"] <= -min_fall_filter].copy()
ranking_filtered = ranking_filtered.sort_values("Rebound %", ascending=False).reset_index(drop=True)

# Summary metrics
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(
        f'<div class="metric-box"><div class="metric-label">Stocks Scanned</div>'
        f'<div class="metric-value">{len(all_closes):,}</div></div>',
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        f'<div class="metric-box"><div class="metric-label">Passing Filter</div>'
        f'<div class="metric-value">{len(ranking_filtered):,}</div></div>',
        unsafe_allow_html=True,
    )
with m3:
    top_val = ranking_filtered["Rebound %"].iloc[0] if len(ranking_filtered) else 0
    st.markdown(
        f'<div class="metric-box"><div class="metric-label">Top Rebound</div>'
        f'<div class="metric-value">{top_val:.1f}%</div></div>',
        unsafe_allow_html=True,
    )
with m4:
    median_val = ranking_filtered["Rebound %"].median() if len(ranking_filtered) else 0
    st.markdown(
        f'<div class="metric-box"><div class="metric-label">Median Rebound</div>'
        f'<div class="metric-value">{median_val:.1f}%</div></div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ----- Top rebounders table -----
st.subheader("Top Rebounders")
top_n = st.slider("Show top N", 10, 100, 25, key="topn")

def style_row(row):
    """Color the Recovery cell only — keeps the rest of the row clean and readable."""
    styles = [""] * len(row)
    rec_idx = row.index.get_loc("Recovery of Fall %")
    rebound_idx = row.index.get_loc("Rebound %")
    fall_idx = row.index.get_loc("Fall %")

    rec = row["Recovery of Fall %"]
    if rec >= 75:
        styles[rec_idx] = "background-color: #166534; color: #ffffff; font-weight: 600;"
    elif rec >= 50:
        styles[rec_idx] = "background-color: #65a30d; color: #ffffff; font-weight: 600;"
    elif rec >= 30:
        styles[rec_idx] = "background-color: #ca8a04; color: #ffffff; font-weight: 600;"
    else:
        styles[rec_idx] = "background-color: #b91c1c; color: #ffffff; font-weight: 600;"

    # Color the Rebound % green (it's always positive in this table)
    styles[rebound_idx] = "color: #047857; font-weight: 600;"
    # Color the Fall % red (it's always negative)
    styles[fall_idx] = "color: #b91c1c; font-weight: 600;"
    return styles

display_df = ranking_filtered.head(top_n)
styled = (
    display_df.style
    .apply(style_row, axis=1)
    .format({
        "Peak ₹": "{:,.2f}",
        "Low ₹": "{:,.2f}",
        "Latest ₹": "{:,.2f}",
        "Fall %": "{:+.2f}",
        "Rebound %": "{:+.2f}",
        "Recovery of Fall %": "{:.1f}",
    })
    .set_properties(**{
        "color": "#111827",
        "background-color": "#ffffff",
        "font-family": "JetBrains Mono, monospace",
        "font-size": "0.85rem",
    })
    .set_table_styles([
        {"selector": "thead th", "props": [
            ("background-color", "#0f172a"),
            ("color", "#ffffff"),
            ("font-weight", "600"),
            ("text-transform", "uppercase"),
            ("letter-spacing", "0.05em"),
            ("font-size", "0.7rem"),
            ("padding", "10px 8px"),
        ]},
        {"selector": "tbody td", "props": [
            ("padding", "8px"),
            ("border-bottom", "1px solid #e5e7eb"),
        ]},
        {"selector": "tbody tr:hover", "props": [
            ("background-color", "#f9fafb"),
        ]},
    ])
)

st.dataframe(styled, use_container_width=True, height=600)

st.caption(
    "**Recovery of Fall %** color key: "
    "🟩 ≥75% (strong recovery)  ·  "
    "🟨 50–75% (solid bounce)  ·  "
    "🟧 30–50% (partial)  ·  "
    "🟥 <30% (weak — possible dead-cat bounce)"
)

# ----- Drill-down chart -----
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.subheader("Inspect a Stock")

chosen = st.selectbox(
    "Pick a symbol to chart",
    ranking_filtered["Symbol"].tolist(),
)
if chosen:
    ticker = chosen + ".NS"
    if ticker in all_closes:
        series = all_closes[ticker]
        chart_df = pd.DataFrame({"Close": series})
        st.line_chart(chart_df, height=300)

        row = ranking_filtered[ranking_filtered["Symbol"] == chosen].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Fall from Peak", f"{row['Fall %']:+.2f}%")
        c2.metric("Rebound from Low", f"{row['Rebound %']:+.2f}%")
        c3.metric("Recovery of Fall", f"{row['Recovery of Fall %']:.1f}%")

# ----- Download buttons -----
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.subheader("Export Results")

csv_ranking = ranking_filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇ Download rebound ranking (CSV)",
    csv_ranking,
    file_name=f"rebound_ranking_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv",
)

# Build wide close-price CSV
wide_closes = pd.DataFrame(all_closes)
wide_closes.columns = [c.replace(".NS", "") for c in wide_closes.columns]
csv_raw = wide_closes.to_csv().encode("utf-8")
st.download_button(
    "⬇ Download raw close prices (CSV)",
    csv_raw,
    file_name=f"close_prices_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv",
)

st.caption(
    "Data via Yahoo Finance (yfinance). Not investment advice. "
    "Analyze at your own discretion."
)
