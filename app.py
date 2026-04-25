"""
Indian Stock Rebound Scanner — Streamlit App
============================================
Ranks NSE stocks by how much they have rebounded from their
LAST 2-MONTH LOW (rolling 60-calendar-day window from today).

Deploy on Streamlit Cloud:
    1. Push these files to a GitHub repo:
         - app.py
         - requirements.txt
         - EQUITY_L.csv
    2. Go to https://share.streamlit.io
    3. "New app" -> select repo + branch + app.py
    4. Deploy.
"""

import time
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import yfinance as yf

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Indian Stock 2-Month Rebound Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- STYLING ----------------
st.markdown(
    """
    <style>
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
    st.markdown('<div class="main-title">2-Month Rebound Scanner</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">NSE Equity · Recovery from 60-Day Low</div>',
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

# Fixed 2-month window
LOOKBACK_DAYS = 60
DOWNLOAD_BUFFER_DAYS = 15

st.sidebar.markdown(
    f"<div style='font-size:0.8rem; color:#94a3b8; margin-bottom:1rem;'>"
    f"Window: <b style='color:#fbbf24;'>Last {LOOKBACK_DAYS} days</b> (fixed)"
    f"</div>",
    unsafe_allow_html=True,
)

min_rebound_days = st.sidebar.slider(
    "Minimum days since low",
    min_value=1,
    max_value=20,
    value=2,
    help="Excludes stocks whose 2-month low is too recent (still falling).",
)

max_stocks = st.sidebar.number_input(
    "Max stocks to scan",
    min_value=50,
    max_value=3000,
    value=2000,
    step=50,
    help="2,000+ scans the near-full NSE universe. Smaller = faster.",
)

batch_size = st.sidebar.slider(
    "Batch size",
    min_value=20,
    max_value=100,
    value=50,
    step=10,
    help="Stocks per yfinance call. Lower if you see download failures.",
)

min_rebound_filter = st.sidebar.number_input(
    "Minimum rebound to show (%)",
    min_value=0.0,
    max_value=100.0,
    value=5.0,
    step=1.0,
    help="Hide stocks that have rebounded less than this from the 2-month low.",
)

st.sidebar.markdown("---")
run_button = st.sidebar.button("▶ Run Scan", type="primary", use_container_width=True)

# ---------------- HELPERS ----------------
@st.cache_data(show_spinner=False)
def load_symbols() -> list[str]:
    """Load NSE equity symbols from EQUITY_L.csv in the repo root."""
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
        s = data["Close"].dropna()
        if len(s) > 0:
            out[tickers[0]] = s
    return out


def compute_one_month_rebound(closes: dict, min_days: int) -> pd.DataFrame:
    """
    For each stock, look at ONLY the last 60 calendar days of trading data.
    Find the lowest close in that 60-day window, then compute the rebound
    from that low to the latest close.
    """
    rows = []
    cutoff = pd.Timestamp(datetime.now()) - pd.Timedelta(days=LOOKBACK_DAYS)

    for ticker, full_series in closes.items():
        # Restrict to the last 60 calendar days
        s = full_series[full_series.index >= cutoff]
        if len(s) < min_days + 2:
            continue

        latest_price = float(s.iloc[-1])
        latest_date = s.index[-1]
        low_price = float(s.min())
        low_date = s.idxmin()
        high_price = float(s.max())
        high_date = s.idxmax()

        days_since_low = (latest_date - low_date).days
        if days_since_low < min_days:
            continue
        if low_price <= 0:
            continue

        # Core metric: rebound from the 2-month low
        rebound_pct = (latest_price - low_price) / low_price * 100

        # Drop from 2-month high (drawdown context)
        drop_from_high_pct = (latest_price - high_price) / high_price * 100

        # Where in the 2M range is the price now? 100% = at high, 0% = at low
        if high_price > low_price:
            position_in_range = (latest_price - low_price) / (high_price - low_price) * 100
        else:
            position_in_range = 50.0

        new_30d_high = latest_price >= high_price

        rows.append({
            "Symbol": ticker.replace(".NS", ""),
            "2M Low ₹": round(low_price, 2),
            "Low Date": low_date.strftime("%d-%b"),
            "2M High ₹": round(high_price, 2),
            "High Date": high_date.strftime("%d-%b"),
            "Latest ₹": round(latest_price, 2),
            "Rebound from 2M Low %": round(rebound_pct, 2),
            "Drop from 2M High %": round(drop_from_high_pct, 2),
            "Position in 2M Range %": round(position_in_range, 1),
            "New 60D High?": "✅" if new_30d_high else "",
            "Days Since Low": days_since_low,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Rebound from 2M Low %", ascending=False).reset_index(drop=True)
    return df


# ---------------- INTRO PAGE ----------------
if not run_button and "scan_results" not in st.session_state:
    st.info(
        f"**How this works** — The scanner pulls daily closing prices for every NSE equity symbol "
        f"in `EQUITY_L.csv`, finds each stock's lowest close in the **last {LOOKBACK_DAYS} days**, "
        f"and ranks stocks by % rebound from that low to today's close. "
        f"Configure parameters in the sidebar and click **Run Scan**."
    )

    with st.expander("Understanding the output"):
        st.markdown(
            """
            All metrics are computed over the **last 60 calendar days only**:

            - **Rebound from 2M Low %** — how much the stock has risen from its lowest
              close in the past 60 days. *This is the primary ranking metric.*
            - **Drop from 2M High %** — how far the latest price is below the 2-month high.
            - **Position in 2M Range %** — where the latest price sits between the 2M low (0%)
              and 2M high (100%). 100% = trading at the 60-day high.
            - **New 60D High?** — ✅ if the latest price is at or above the 2-month high.
            - **Days Since Low** — how many days ago the 2-month low was hit.

            **Strong setups:** large Rebound %, Position in Range close to 100%, Days Since Low ≥ 5.
            **Weak/risky:** large Rebound % but Position in Range < 50% (still in lower half).
            """
        )
    st.stop()

# ---------------- RUN THE SCAN ----------------
if run_button:
    try:
        symbols = load_symbols()
    except FileNotFoundError:
        st.error("`EQUITY_L.csv` not found in the repo. Make sure it's committed alongside `app.py`.")
        st.stop()
    except Exception as e:
        st.error(f"Could not read symbol file: {e}")
        st.stop()

    symbols = symbols[: int(max_stocks)]
    tickers = [s + ".NS" for s in symbols]

    end = datetime.now()
    start = end - timedelta(days=LOOKBACK_DAYS + DOWNLOAD_BUFFER_DAYS)

    st.markdown(
        f"**Scanning {len(tickers)} symbols** · "
        f"Window: last {LOOKBACK_DAYS} days ({(end - timedelta(days=LOOKBACK_DAYS)).date()} → {end.date()})"
    )

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
        progress.progress(
            batch_num / total_batches,
            text=f"Downloaded {len(all_closes)} stocks so far",
        )
        time.sleep(0.5)

    elapsed = time.time() - t0
    status.success(f"Download complete · {len(all_closes)} stocks · {elapsed:.0f}s elapsed")

    if not all_closes:
        st.error("No data retrieved. Yahoo Finance may be rate-limiting. Try a smaller batch size.")
        st.stop()

    st.session_state["scan_results"] = {
        "all_closes": all_closes,
        "min_rebound_days": min_rebound_days,
        "min_rebound_filter": min_rebound_filter,
        "scan_time": datetime.now(),
        "scan_duration": elapsed,
    }

# ---------------- LOAD FROM SESSION STATE ----------------
results = st.session_state["scan_results"]
all_closes = results["all_closes"]
scan_min_rebound_days = results["min_rebound_days"]
scan_min_rebound_filter = results["min_rebound_filter"]

st.caption(
    f"Showing results from scan at {results['scan_time'].strftime('%H:%M:%S')} "
    f"({len(all_closes)} stocks, took {results['scan_duration']:.0f}s). "
    f"Click **Run Scan** in the sidebar to refresh."
)

# ---------------- COMPUTE & DISPLAY ----------------
ranking = compute_one_month_rebound(all_closes, scan_min_rebound_days)

if ranking.empty:
    st.warning("No stocks matched the criteria. Try loosening the filters and re-scanning.")
    st.stop()

ranking_filtered = ranking[ranking["Rebound from 2M Low %"] >= scan_min_rebound_filter].copy()
ranking_filtered = ranking_filtered.reset_index(drop=True)

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
    top_val = ranking_filtered["Rebound from 2M Low %"].iloc[0] if len(ranking_filtered) else 0
    st.markdown(
        f'<div class="metric-box"><div class="metric-label">Top Rebound</div>'
        f'<div class="metric-value">{top_val:.1f}%</div></div>',
        unsafe_allow_html=True,
    )
with m4:
    median_val = ranking_filtered["Rebound from 2M Low %"].median() if len(ranking_filtered) else 0
    st.markdown(
        f'<div class="metric-box"><div class="metric-label">Median Rebound</div>'
        f'<div class="metric-value">{median_val:.1f}%</div></div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ---------------- TABBED LAYOUT ----------------
tab1, tab2, tab3 = st.tabs([
    "🏆 Top Rebounders",
    "🔍 Stock Inspector",
    "⬇ Export",
])

# ============= TAB 1: TOP REBOUNDERS =============
with tab1:
    st.subheader("Stocks Ranked by 2-Month Low Rebound")
    st.markdown(
        "Stocks are sorted by **% gain from their lowest close in the last 60 days**. "
        "Cross-reference with **Position in 2M Range %** — high values mean the stock has "
        "fully recovered toward the period's high, low values mean it's still in the lower "
        "half of its monthly range (potentially weak)."
    )

    # ----- HERO CARDS for top 5 -----
    if len(ranking_filtered) >= 1:
        st.markdown("### 🎯 Top 5 Highlights")
        top5 = ranking_filtered.head(5).reset_index(drop=True)

        cols = st.columns(len(top5))
        for col, (_, r) in zip(cols, top5.iterrows()):
            new_high_badge = (
                '<div style="display:inline-block; background:#16a34a; color:white; '
                'padding:2px 6px; font-size:0.6rem; font-weight:600; border-radius:2px; '
                'margin-top:4px;">NEW 60D HIGH</div>'
                if r.get("New 60D High?") == "✅" else ""
            )
            position = r["Position in 2M Range %"]
            pos_color = "#047857" if position >= 75 else ("#ca8a04" if position >= 50 else "#b91c1c")

            card = (
                f'<div style="border: 2px solid #d97706; '
                f'background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%); '
                f'padding: 1.2rem; border-radius: 4px; position: relative; min-height: 200px;">'
                f'<div style="position: absolute; top: 8px; right: 12px; font-size: 0.7rem; '
                f'color: #92400e; letter-spacing: 0.1em;">★</div>'
                f'<div style="font-family: \'Fraunces\', serif; font-weight: 800; font-size: 1.3rem; '
                f'color: #111; margin-bottom: 0.3rem;">{r["Symbol"]}</div>'
                f'<div style="font-size: 0.7rem; color: #6b7280; text-transform: uppercase; '
                f'letter-spacing: 0.1em;">Rebound from 2M Low</div>'
                f'<div style="font-family: \'Fraunces\', serif; font-weight: 600; font-size: 1.6rem; '
                f'color: #047857;">+{r["Rebound from 2M Low %"]:.1f}%</div>'
                f'<div style="margin-top: 0.6rem; font-size: 0.75rem; color: #4b5563;">'
                f'In range: <b style="color:{pos_color};">{position:.0f}%</b><br>'
                f'Low: ₹{r["2M Low ₹"]:.2f} ({r["Low Date"]})<br>'
                f'Now: ₹{r["Latest ₹"]:.2f} · {r["Days Since Low"]}d ago'
                f'</div>{new_high_badge}</div>'
            )
            with col:
                st.markdown(card, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ----- Full ranked table -----
    st.markdown("### Full Ranking")
    top_n = st.slider("Show top N", 10, 300, 100, key="topn_main")

    def style_rebound_row(row):
        styles = [""] * len(row)
        rebound_idx = row.index.get_loc("Rebound from 2M Low %")
        position_idx = row.index.get_loc("Position in 2M Range %")
        drop_idx = row.index.get_loc("Drop from 2M High %")

        styles[rebound_idx] = "background-color: #047857; color: #ffffff; font-weight: 700;"

        pos = row["Position in 2M Range %"]
        if pos >= 90:
            styles[position_idx] = "background-color: #166534; color: #ffffff; font-weight: 600;"
        elif pos >= 70:
            styles[position_idx] = "background-color: #65a30d; color: #ffffff; font-weight: 600;"
        elif pos >= 40:
            styles[position_idx] = "background-color: #ca8a04; color: #ffffff; font-weight: 600;"
        else:
            styles[position_idx] = "background-color: #b91c1c; color: #ffffff; font-weight: 600;"

        styles[drop_idx] = "color: #b91c1c; font-weight: 600;"
        return styles

    display_df = ranking_filtered.head(top_n)
    styled = (
        display_df.style
        .apply(style_rebound_row, axis=1)
        .format({
            "2M Low ₹": "{:,.2f}",
            "2M High ₹": "{:,.2f}",
            "Latest ₹": "{:,.2f}",
            "Rebound from 2M Low %": "+{:.2f}",
            "Drop from 2M High %": "{:+.2f}",
            "Position in 2M Range %": "{:.0f}",
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
        "**Position in 2M Range %** color key: "
        "🟩 ≥90% (at/near 2M high)  ·  "
        "🟨 70–90% (upper range)  ·  "
        "🟧 40–70% (mid)  ·  "
        "🟥 <40% (still in lower half — risky)"
    )

# ============= TAB 2: INSPECTOR =============
with tab2:
    st.subheader("Inspect a Stock")
    chosen = st.selectbox(
        "Pick a symbol to chart",
        ranking_filtered["Symbol"].tolist(),
    )
    if chosen:
        ticker = chosen + ".NS"
        if ticker in all_closes:
            full_series = all_closes[ticker]
            cutoff = pd.Timestamp(datetime.now()) - pd.Timedelta(days=LOOKBACK_DAYS)
            series = full_series[full_series.index >= cutoff]

            chart_df = pd.DataFrame({"Close": series})
            st.line_chart(chart_df, height=350)

            row = ranking_filtered[ranking_filtered["Symbol"] == chosen].iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("2M Low → Now", f"+{row['Rebound from 2M Low %']:.2f}%")
            c2.metric("Drop from 2M High", f"{row['Drop from 2M High %']:+.2f}%")
            c3.metric("Position in Range", f"{row['Position in 2M Range %']:.0f}%")
            c4.metric("Days Since Low", f"{row['Days Since Low']}")

            st.markdown(
                f"**2M Low:** ₹{row['2M Low ₹']:.2f} on {row['Low Date']}  ·  "
                f"**2M High:** ₹{row['2M High ₹']:.2f} on {row['High Date']}  ·  "
                f"**Latest:** ₹{row['Latest ₹']:.2f}"
            )

# ============= TAB 3: EXPORT =============
with tab3:
    st.subheader("Export Results")
    csv_ranking = ranking_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download 2-month rebound ranking (CSV)",
        csv_ranking,
        file_name=f"rebound_2m_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

    wide_closes = pd.DataFrame(all_closes)
    wide_closes.columns = [c.replace(".NS", "") for c in wide_closes.columns]
    csv_raw = wide_closes.to_csv().encode("utf-8")
    st.download_button(
        "⬇ Download raw close prices (CSV)",
        csv_raw,
        file_name=f"close_prices_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.caption(
    "Data via Yahoo Finance (yfinance). Window: rolling 60 calendar days. "
    "Not investment advice. Analyze at your own discretion."
)
