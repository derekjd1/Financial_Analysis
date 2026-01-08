# app.py — Streamlit Finance Analyzer (TICKER-ONLY input, no company-name search, no .TO auto-append)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta

import yfinance as yf

from database import get_conn, add_favorite, remove_favorite, list_favorites
from data import compute_date_range, get_history, get_prices_for
from analytics import compute_metrics, annual_returns, pct
from charts import make_chart, make_compare_chart, make_returns_heatmap, rolling_vol_chart
from reporting import generate_pdf_report
from investment_calc import render_investment_calculator_tab


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Finance Analyzer", page_icon="📈", layout="wide")
st.title("📈 Finance Analyzer — Stocks & ETFs")

conn = get_conn()


def safe_rerun() -> None:
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass


# ----------------------------
# Session defaults
# ----------------------------
if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = ""
if "selected_name" not in st.session_state:
    st.session_state.selected_name = ""
if "query_input" not in st.session_state:
    st.session_state.query_input = ""


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def get_company_name_from_yf(symbol: str) -> str:
    """Cached yfinance name lookup (for chart title only)."""
    try:
        t = yf.Ticker(symbol)
        info = getattr(t, "info", {}) or {}
        return (
            info.get("shortName")
            or info.get("longName")
            or info.get("displayName")
            or info.get("name")
            or ""
        ).strip()
    except Exception:
        return ""


def is_valid_ticker(user_input: str) -> bool:
    """
    Simple validation:
    - ticker only (no spaces)
    - allow letters/numbers and: . - ^
    """
    s = (user_input or "").strip()
    if not s:
        return False
    if any(ch.isspace() for ch in s):
        return False
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-^")
    s_up = s.upper()
    return all(ch in allowed for ch in s_up)


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.subheader("Search (Ticker only)")

    st.info("Enter a **ticker symbol**, not a company name.")
    st.info("🇨🇦 For Canadian stocks/ETFs, include **.TO** (e.g., `BNS.TO`, `XEQT.TO`).")

    with st.form("ticker_form", clear_on_submit=False):
        query = st.text_input(
            "Ticker",
            value=st.session_state.query_input,
            placeholder="Enter Here",
        )
        st.divider()
        st.subheader("Range & View")
        range_choice = st.radio("Date range", ["YTD", "1Y", "5Y", "MAX", "Custom"], horizontal=True)

        custom_start = custom_end = None
        if range_choice == "Custom":
            c1, c2 = st.columns(2)
            with c1:
                custom_start = st.date_input("Start", value=date.today() - timedelta(days=365))
            with c2:
                custom_end = st.date_input("End", value=date.today())
            if custom_start > custom_end:
                st.error("Start date must be before end date.")

        chart_type = st.selectbox("Chart type", ["Line (Adj Close)", "Candlestick (OHLC)"])
        show_sma = st.checkbox("Show 50/200 SMA", value=True)
        auto_adjust = st.checkbox("Use adjusted prices (recommended)", value=True)
        interval = "1d"

        load_clicked = st.form_submit_button("🔎 Load")

    if load_clicked:
        raw = (query or "").strip()
        if not is_valid_ticker(raw):
            st.error("Please enter a valid ticker symbol (no spaces). Example: `AAPL` or `BNS.TO`.")
        else:
            st.session_state.query_input = raw
            st.session_state.selected_symbol = raw.upper()
            st.session_state.selected_name = ""  # will be filled after data loads
            safe_rerun()

    selected_symbol = st.session_state.selected_symbol

    st.divider()
    st.subheader("⭐ Favorites")
    fav_label = st.text_input("Label (optional)", key="fav_label", placeholder="e.g., Core S&P 500")

    if st.button("Add current symbol to favorites", use_container_width=True):
        if selected_symbol:
            add_favorite(conn, selected_symbol, fav_label)
            st.success(f"Added {selected_symbol.upper()} to favorites.")
            safe_rerun()

    favs = list_favorites(conn)
    if favs.empty:
        st.caption("No favorites yet. Add one above!")
    else:
        display = [f"{row.symbol}" + (f" — {row.label}" if row.label else "") for _, row in favs.iterrows()]
        pick = st.selectbox("Go to favorite", ["—"] + display, index=0)
        if pick != "—":
            st.session_state.selected_symbol = pick.split(" —")[0].strip().upper()
            st.session_state.selected_name = ""
            safe_rerun()

        st.markdown("**Manage favorites**")
        for _, row in favs.iterrows():
            c1, c2 = st.columns([4, 1])
            with c1:
                st.caption(f"{row.symbol}" + (f" — {row.label}" if row.label else ""))
            with c2:
                if st.button("🗑️", key=f"del_{row.id}", help=f"Remove {row.symbol}"):
                    remove_favorite(conn, row.symbol)
                    safe_rerun()


st.markdown(
    "Enter a **ticker symbol** in the sidebar and click **Load**. "
    "For Canadian tickers, include **.TO** (example: `BNS.TO`, `XEQT.TO`)."
)

# ----------------------------
# Resolve date range
# ----------------------------
if range_choice == "Custom":
    start, end, period = custom_start, custom_end, None
else:
    start, end, period = compute_date_range(range_choice)

# ----------------------------
# Fetch history
# ----------------------------
if not selected_symbol:
    st.warning("Please enter a ticker symbol to load data.")
    hist = pd.DataFrame()
else:
    with st.spinner(f"Loading {selected_symbol}…"):
        hist = get_history(selected_symbol, start, end, period, interval, auto_adjust)

# Name for chart title (optional polish)
if selected_symbol and not st.session_state.selected_name:
    st.session_state.selected_name = get_company_name_from_yf(selected_symbol)
selected_name = st.session_state.selected_name

left, right = st.columns((7, 5))

with left:
    if hist.empty:
        st.warning("No price data returned. Double-check the ticker (and include .TO for Canadian symbols).")
        fig_main = go.Figure()
    else:
        title_left = selected_symbol.upper()
        if selected_name:
            title_left = f"{title_left} — {selected_name}"
        fig_main = make_chart(hist, f"{title_left} ({range_choice})", chart_type, show_sma)
        st.plotly_chart(fig_main, use_container_width=True)

with right:
    st.subheader("Quick Stats")
    fig_returns = None
    fig_rolling = None

    if hist.empty:
        st.info("Stats will appear here once data loads.")
        metrics = {"return_pct": None, "cagr": None, "volatility": None, "max_drawdown": None}
    else:
        metrics = compute_metrics(hist)

        if "Adj Close" in hist:
            last_price = float(hist["Adj Close"].iloc[-1])
        elif "Close" in hist:
            last_price = float(hist["Close"].iloc[-1])
        else:
            last_price = float("nan")

        c1, c2, c3 = st.columns(3)
        c1.metric("Last Price", f"{last_price:,.2f}")
        c2.metric("Total Return", pct(metrics["return_pct"]))
        c3.metric("CAGR", pct(metrics["cagr"]))

        c4, c5 = st.columns(2)
        wk_low = wk_high = None
        try:
            last_252 = hist.tail(252)
            wk_low = float(last_252["Adj Close"].min())
            wk_high = float(last_252["Adj Close"].max())
        except Exception:
            pass

        c4.metric("52-Week Low", f"{wk_low:,.2f}" if wk_low is not None else "—")
        c5.metric("52-Week High", f"{wk_high:,.2f}" if wk_high is not None else "—")

        st.caption("Volatility is annualized from daily returns; max drawdown uses running peaks in the selected window.")
        c6, c7 = st.columns(2)
        c6.metric("Volatility (ann.)", pct(metrics["volatility"]))
        c7.metric("Max Drawdown", pct(metrics["max_drawdown"]))

        csv = hist.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download history (CSV)",
            data=csv,
            file_name=f"{selected_symbol}_{range_choice}.csv",
            mime="text/csv",
            use_container_width=True
        )

# ----------------------------
# Tabs
# ----------------------------
st.divider()
st.subheader("📊 Analytics & Tools")
tab1, tab2, tab3, tab4 = st.tabs([
    "🔀 Compare",
    "📅 Annual Returns",
    "📈 Rolling Volatility",
    "💰 Investment Calculator",
])

with tab1:
    st.caption("Compare multiple tickers by normalizing each to 100 at the start date.")
    fav_options = favs.symbol.tolist() if (favs is not None and not favs.empty) else []
    compare_favs = st.multiselect("Pick favorites to compare", options=fav_options, default=[])
    manual = st.text_input("Or enter tickers (comma-separated)", placeholder="e.g., VFV.TO, XEQT.TO, AAPL")

    manual_syms = [s.strip().upper() for s in manual.split(",") if s.strip()] if manual else []
    base_sym = [selected_symbol.upper()] if selected_symbol else []
    symbols = base_sym + [s.upper() for s in compare_favs] + manual_syms

    # de-dupe, keep order
    seen = set()
    ordered = []
    for s in symbols:
        if s and s not in seen:
            seen.add(s)
            ordered.append(s)

    if len(ordered) <= 1:
        st.info("Add at least one more symbol to compare.")
    else:
        with st.spinner(f"Comparing: {', '.join(ordered)}"):
            prices = get_prices_for(ordered, start, end, period, interval, auto_adjust)
        st.plotly_chart(make_compare_chart(prices, "Relative Performance"), use_container_width=True)

with tab2:
    st.caption("Year-by-year total returns calculated from adjusted close.")
    rets = annual_returns(hist) if not hist.empty else pd.Series(dtype="float64")
    if rets.empty:
        st.info("Not enough data to compute annual returns for this selection.")
    else:
        fig_returns = make_returns_heatmap(rets, selected_symbol)
        st.plotly_chart(fig_returns, use_container_width=True)
        df_rets = pd.DataFrame({"Year": rets.index, "Return %": (rets.values * 100).round(2)})
        st.dataframe(df_rets, use_container_width=True, hide_index=True)

with tab3:
    st.caption("Rolling annualized volatility from daily returns.")
    rolling_win = st.slider("Window (days)", min_value=20, max_value=250, value=60, step=5, key="rolling_win")
    fig_rolling = rolling_vol_chart(hist, rolling_win, selected_symbol if selected_symbol else "")
    st.plotly_chart(fig_rolling, use_container_width=True)

with tab4:
    render_investment_calculator_tab()

# ----------------------------
# PDF
# ----------------------------
st.divider()
if hist.empty:
    st.info("Load a symbol to enable report generation.")
else:
    if st.button("📄 Generate PDF Report", use_container_width=True):
        try:
            pdf_bytes = generate_pdf_report(
                symbol=selected_symbol,
                range_choice=range_choice,
                metrics=metrics,
                fig_main=fig_main,
                fig_returns=fig_returns,
                fig_rolling=fig_rolling,
                pct_fmt=pct,
            )
            st.download_button(
                "⬇️ Download Report PDF",
                data=pdf_bytes,
                file_name=f"{selected_symbol}_{range_choice}_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.error("Could not generate the PDF report.")
            st.exception(e)

st.divider()
with st.expander("Notes & Tips"):
    st.markdown(
        "- Input must be a **ticker symbol** (no company names).\n"
        "- For Canadian symbols, include **.TO**.\n"
        "- If you get no data, the ticker is likely wrong or missing the exchange suffix."
    )
