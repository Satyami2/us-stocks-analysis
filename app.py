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

        # Skip stocks where there was no real fall before the low
        # (low == peak means the stock never actually dropped before bottoming —
        # often happens with new listings or stocks in a steady uptrend).
        fall_amount = peak_before_low - low_price
        if fall_amount <= 0 or peak_before_low <= 0:
            continue

        fall_pct = (low_price - peak_before_low) / peak_before_low * 100
        rebound_pct = (latest_price - low_price) / low_price * 100

        # Recovery of Fall: what fraction of the drawdown is reclaimed.
        # Cap at 100% — anything above means the stock is at NEW HIGHS, which
        # is no longer "recovery" but breakout territory. We flag those separately.
        raw_recovery = (latest_price - low_price) / fall_amount * 100
        broke_past_peak = latest_price > peak_before_low
        recovery_of_fall = min(raw_recovery, 100.0)

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
            "New Highs?": "✅" if broke_past_peak else "",
            "Days Since Low": days_since_low,
        })
    return pd.DataFrame(rows)


# ---------------- MAIN BODY ----------------
# ---------------- INTRO PAGE (shown until first scan completes) ----------------
if not run_button and "scan_results" not in st.session_state:
    st.info(
        "**How this works** — The scanner pulls daily closing prices from Yahoo Finance for "
        "every NSE equity symbol in `EQUITY_L.csv` (loaded automatically from the repo), "
        "finds each stock's lowest close in the lookback window, and calculates the % rebound "
        "from that low to the latest close. Stocks that are still making new lows are excluded "
        "(controlled by *Minimum days since low*). Configure parameters in the sidebar and click **Run Scan**."
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

# ---------------- RUN THE SCAN (only when button clicked) ----------------
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

    # Persist results so subsequent reruns (e.g. changing selectbox) don't redownload
    st.session_state["scan_results"] = {
        "all_closes": all_closes,
        "min_rebound_days": min_rebound_days,
        "min_fall_filter": min_fall_filter,
        "scan_time": datetime.now(),
        "scan_duration": elapsed,
    }

# ---------------- LOAD FROM SESSION STATE ----------------
# At this point we either just ran a scan (results in session_state) or
# we're rerunning because the user changed a tab/selectbox/slider.
results = st.session_state["scan_results"]
all_closes = results["all_closes"]

# Use the filter values that were active at scan time (so changing sidebar
# values without re-scanning doesn't silently change the displayed numbers).
scan_min_rebound_days = results["min_rebound_days"]
scan_min_fall_filter = results["min_fall_filter"]

st.caption(
    f"Showing results from scan at {results['scan_time'].strftime('%H:%M:%S')} "
    f"({len(all_closes)} stocks, took {results['scan_duration']:.0f}s). "
    f"Click **Run Scan** in the sidebar to refresh."
)

# ---------------- COMPUTE & DISPLAY ----------------
ranking = compute_rebound_table(all_closes, scan_min_rebound_days)

if ranking.empty:
    st.warning("No stocks matched the criteria. Try loosening the filters and re-scanning.")
    st.stop()

# Apply fall filter
ranking_filtered = ranking[ranking["Fall %"] <= -scan_min_fall_filter].copy()
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

# ---------------- COMPUTE A COMPOSITE QUALITY SCORE ----------------
# Rationale: a "great rebounder" isn't just whichever stock bounced the most
# percentage-wise. It's one that:
#   1) Bounced strongly off the low (Rebound %)
#   2) Recovered a meaningful chunk of the prior fall (Recovery of Fall %)
#   3) Has had time for the bounce to develop (Days Since Low)
#   4) Actually fell first (Fall % magnitude — not just a stock that drifted up)
# We z-score each metric across the universe and combine them.

def add_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if len(out) < 2:
        out["Score"] = 0.0
        out["Tier"] = "—"
        return out

    def z(series):
        s = series.astype(float)
        std = s.std()
        return (s - s.mean()) / std if std > 0 else s * 0

    z_rebound = z(out["Rebound %"])
    z_recovery = z(out["Recovery of Fall %"])
    z_fall_mag = z(out["Fall %"].abs())  # bigger fall = more impressive comeback
    z_days = z(out["Days Since Low"]).clip(-1.5, 1.5)  # cap; we don't want stale lows dominating

    # Weighted blend
    out["Score"] = (
        0.35 * z_rebound
        + 0.35 * z_recovery
        + 0.20 * z_fall_mag
        + 0.10 * z_days
    ).round(2)

    # Assign tiers based on score percentile
    def tier(score):
        if score >= 1.5:
            return "★★★ Exceptional"
        elif score >= 0.8:
            return "★★ Strong"
        elif score >= 0.0:
            return "★ Above Avg"
        else:
            return "—"

    out["Tier"] = out["Score"].apply(tier)
    return out.sort_values("Score", ascending=False).reset_index(drop=True)


ranking_scored = add_quality_score(ranking_filtered)

# ---------------- TABBED LAYOUT ----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Star Performers",
    "📊 All Rebounders",
    "🔍 Stock Inspector",
    "⬇ Export",
])

# ============= TAB 1: STAR PERFORMERS =============
with tab1:
    st.subheader("Exceptional Rebound Candidates")
    st.markdown(
        "These are stocks that score highly on a **composite quality score** combining: "
        "rebound size, recovery of fall, drawdown magnitude, and time since low. "
        "Higher scores = more impressive, more durable comebacks — not just the biggest "
        "one-day pop off a bottom."
    )

    stars = ranking_scored[ranking_scored["Tier"] != "—"].copy()
    exceptional = ranking_scored[ranking_scored["Tier"] == "★★★ Exceptional"]
    strong = ranking_scored[ranking_scored["Tier"] == "★★ Strong"]
    above_avg = ranking_scored[ranking_scored["Tier"] == "★ Above Avg"]

    # Tier counts
    sm1, sm2, sm3 = st.columns(3)
    with sm1:
        st.markdown(
            f'<div class="metric-box" style="border-left: 4px solid #d97706;">'
            f'<div class="metric-label">★★★ Exceptional</div>'
            f'<div class="metric-value">{len(exceptional)}</div></div>',
            unsafe_allow_html=True,
        )
    with sm2:
        st.markdown(
            f'<div class="metric-box" style="border-left: 4px solid #65a30d;">'
            f'<div class="metric-label">★★ Strong</div>'
            f'<div class="metric-value">{len(strong)}</div></div>',
            unsafe_allow_html=True,
        )
    with sm3:
        st.markdown(
            f'<div class="metric-box" style="border-left: 4px solid #0891b2;">'
            f'<div class="metric-label">★ Above Average</div>'
            f'<div class="metric-value">{len(above_avg)}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ----- HERO CARDS for the top 5 exceptional stocks -----
    if len(exceptional) > 0:
        st.markdown("### 🎯 Top 5 Highlights")
        top5 = exceptional.head(5).reset_index(drop=True)

        # Render each card in its own column with its own st.markdown call.
        # Concatenating all 5 into one giant HTML string causes Streamlit's
        # sanitizer to bail out and show raw HTML on screen.
        cols = st.columns(len(top5))
        for col, (_, r) in zip(cols, top5.iterrows()):
            new_high_badge = (
                '<div style="display:inline-block; background:#16a34a; color:white; '
                'padding:2px 6px; font-size:0.6rem; font-weight:600; border-radius:2px; '
                'margin-top:4px;">NEW HIGH</div>'
                if r.get("New Highs?") == "✅" else ""
            )
            card = f"""<div style="border: 2px solid #d97706; background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%); padding: 1.2rem; border-radius: 4px; position: relative; min-height: 180px;"><div style="position: absolute; top: 8px; right: 12px; font-size: 0.7rem; color: #92400e; letter-spacing: 0.1em;">★★★</div><div style="font-family: 'Fraunces', serif; font-weight: 800; font-size: 1.3rem; color: #111; margin-bottom: 0.3rem;">{r['Symbol']}</div><div style="font-size: 0.7rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.1em;">Rebound</div><div style="font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.6rem; color: #047857;">+{r['Rebound %']:.1f}%</div><div style="margin-top: 0.6rem; font-size: 0.75rem; color: #4b5563;">Recovered <b>{r['Recovery of Fall %']:.0f}%</b> of fall<br>Score: <b>{r['Score']:.2f}σ</b> · {r['Days Since Low']}d ago</div>{new_high_badge}</div>"""
            with col:
                st.markdown(card, unsafe_allow_html=True)
    else:
        st.info(
            "No stocks crossed the *Exceptional* threshold (composite score ≥ 1.5σ) in this scan. "
            "Try widening the lookback window or scanning more stocks. The strongest performers below "
            "still merit attention."
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ----- Star performers full table -----
    st.markdown("### Ranked Star Performers")

    if len(stars) == 0:
        st.warning("No star performers found. Loosen filters and re-scan.")
    else:
        def style_star_row(row):
            styles = [""] * len(row)
            tier_idx = row.index.get_loc("Tier")
            score_idx = row.index.get_loc("Score")
            rebound_idx = row.index.get_loc("Rebound %")
            recovery_idx = row.index.get_loc("Recovery of Fall %")
            fall_idx = row.index.get_loc("Fall %")

            t = row["Tier"]
            if t == "★★★ Exceptional":
                styles[tier_idx] = "background-color: #d97706; color: #ffffff; font-weight: 700;"
            elif t == "★★ Strong":
                styles[tier_idx] = "background-color: #65a30d; color: #ffffff; font-weight: 600;"
            elif t == "★ Above Avg":
                styles[tier_idx] = "background-color: #0891b2; color: #ffffff; font-weight: 600;"

            styles[score_idx] = "background-color: #1e293b; color: #fbbf24; font-weight: 700;"
            styles[rebound_idx] = "color: #047857; font-weight: 600;"
            styles[recovery_idx] = "color: #1e40af; font-weight: 600;"
            styles[fall_idx] = "color: #b91c1c; font-weight: 600;"
            return styles

        star_cols = [
            "Symbol", "Tier", "Score",
            "Rebound %", "Recovery of Fall %", "New Highs?", "Fall %",
            "Latest ₹", "Low ₹", "Peak ₹",
            "Days Since Low", "Low Date",
        ]
        star_table = stars[star_cols]

        styled_stars = (
            star_table.style
            .apply(style_star_row, axis=1)
            .format({
                "Score": "{:+.2f}σ",
                "Rebound %": "{:+.2f}",
                "Recovery of Fall %": "{:.1f}",
                "Fall %": "{:+.2f}",
                "Latest ₹": "{:,.2f}",
                "Low ₹": "{:,.2f}",
                "Peak ₹": "{:,.2f}",
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
        st.dataframe(styled_stars, use_container_width=True, height=600)

    with st.expander("How is the score computed?"):
        st.markdown(
            """
            For each stock, four metrics are converted to **z-scores** (standard deviations
            above/below the mean of the scanned universe), then combined:

            | Metric | Weight | Why |
            |---|---|---|
            | Rebound % | 35% | Raw bounce size off the low |
            | Recovery of Fall % | 35% | How much of the drawdown is reclaimed |
            | Fall % magnitude | 20% | Rewards stocks that fell hard *and* came back |
            | Days Since Low (capped) | 10% | Slight bonus for sustained recovery |

            **Tiers:**
            - ★★★ Exceptional — Score ≥ 1.5σ above mean
            - ★★ Strong — Score ≥ 0.8σ above mean
            - ★ Above Average — Score ≥ 0σ
            """
        )

# ============= TAB 2: ALL REBOUNDERS =============
with tab2:
    st.subheader("All Rebounders — Full Ranked List")
    top_n = st.slider("Show top N", 10, 200, 50, key="topn_all")

    def style_row(row):
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

        styles[rebound_idx] = "color: #047857; font-weight: 600;"
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

# ============= TAB 3: INSPECTOR =============
with tab3:
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
            st.line_chart(chart_df, height=350)

            row = ranking_filtered[ranking_filtered["Symbol"] == chosen].iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Fall from Peak", f"{row['Fall %']:+.2f}%")
            c2.metric("Rebound from Low", f"{row['Rebound %']:+.2f}%")
            c3.metric("Recovery of Fall", f"{row['Recovery of Fall %']:.1f}%")
            c4.metric("Days Since Low", f"{row['Days Since Low']}")

            score_row = ranking_scored[ranking_scored["Symbol"] == chosen]
            if len(score_row) > 0:
                sr = score_row.iloc[0]
                st.markdown(
                    f"**Quality Score:** `{sr['Score']:+.2f}σ`  ·  **Tier:** **{sr['Tier']}**"
                )

# ============= TAB 4: EXPORT =============
with tab4:
    st.subheader("Export Results")
    csv_ranking = ranking_scored.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download full ranking with scores (CSV)",
        csv_ranking,
        file_name=f"rebound_ranking_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

    stars_only = ranking_scored[ranking_scored["Tier"] != "—"]
    csv_stars = stars_only.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download star performers only (CSV)",
        csv_stars,
        file_name=f"star_performers_{datetime.now().strftime('%Y%m%d')}.csv",
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
    "Data via Yahoo Finance (yfinance). Not investment advice. "
    "Analyze at your own discretion."
)
