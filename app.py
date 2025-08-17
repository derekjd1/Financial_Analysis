# app.py — Streamlit Finance Analyzer (with Favorites, Compare, Heatmap, Rolling Vol, PDF export)
# Run: streamlit run app.py

from __future__ import annotations

import math
from io import BytesIO
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

# DB helpers
from database import get_conn, add_favorite, remove_favorite, list_favorites

# Yahoo search
try:
    import requests  # noqa: F401
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False

# Search Functions 

@st.cache_data(show_spinner=False, ttl=300)
def search_symbols(query: str) -> List[Dict]:
    """Use Yahoo Finance search to find symbols by company/ETF name or ticker."""
    if not HAS_REQUESTS or not query.strip():
        return []
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {"q": query.strip(), "quotesCount": 10, "newsCount": 0}
        r = requests.get(url, params=params, timeout=6)
        r.raise_for_status()
        js = r.json()
        quotes = js.get("quotes", []) or []
        results = [
            q for q in quotes
            if q.get("symbol") and q.get("exchange") and q.get("quoteType") in {"EQUITY", "ETF"}
        ]
        return results[:10]
    except Exception:
        return []


# Date Ranges
def _ytd_range(today: date) -> Tuple[date, date]:
    return date(today.year, 1, 1), today

def _years_ago(today: date, years: int) -> Tuple[date, date]:
    start = today - timedelta(days=int(round(365.25 * years)))
    return start, today

def compute_date_range(label: str, today: Optional[date] = None) -> Tuple[Optional[date], Optional[date], Optional[str]]:
    """Return (start, end, period) for yfinance."""
    if today is None:
        today = date.today()
    label = label.upper()
    if label == "YTD":
        s, e = _ytd_range(today)
        return s, e, None
    if label == "1Y":
        s, e = _years_ago(today, 1)
        return s, e, None
    if label == "5Y":
        s, e = _years_ago(today, 5)
        return s, e, None
    if label == "MAX":
        return None, None, "max"
    return None, None, None


# Fetch the data
@st.cache_data(show_spinner=True)
def get_history(symbol: str,
                start: Optional[date],
                end: Optional[date],
                period: Optional[str],
                interval: str,
                auto_adjust: bool) -> pd.DataFrame:
    """Download price history for a symbol with yfinance; normalize columns."""
    dl_kwargs = dict(interval=interval, auto_adjust=auto_adjust, progress=False, group_by="column")
    if period:
        df = yf.download(symbol, period=period, **dl_kwargs)
    else:
        sdt = pd.to_datetime(start) if start else None
        edt = pd.to_datetime(end) if end else None
        df = yf.download(symbol, start=sdt, end=edt, **dl_kwargs)

    if df is None or df.empty:
        return pd.DataFrame()

    # Handle MultiIndex vs single-level columns
    if isinstance(df.columns, pd.MultiIndex):
        try:
            if symbol in df.columns.get_level_values(-1):
                df = df.xs(symbol, axis=1, level=-1)
            else:
                df.columns = df.columns.get_level_values(0)
        except Exception:
            df.columns = [" ".join([str(p) for p in tup if p is not None]).strip() for tup in df.columns]

    df.columns = [str(c).strip().title() for c in df.columns]
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    return df


# Base chart and information / metrics
def pct(x: Optional[float]) -> str:
    return f"{x*100:.2f}%" if x is not None else "—"

def compute_metrics(df: pd.DataFrame):
    if df is None or df.empty:
        return {"return_pct": None, "cagr": None, "volatility": None, "max_drawdown": None}
    prices = df["Adj Close"].dropna()
    if len(prices) < 2:
        return {"return_pct": None, "cagr": None, "volatility": None, "max_drawdown": None}

    total_return = (prices.iloc[-1] / prices.iloc[0]) - 1.0
    n_days = (prices.index[-1] - prices.index[0]).days
    yrs = max(n_days / 365.25, 1e-9)
    cagr = (prices.iloc[-1] / prices.iloc[0]) ** (1 / yrs) - 1.0 if yrs > 0 else None

    daily_ret = prices.pct_change().dropna()
    vol = float(daily_ret.std() * np.sqrt(252)) if not daily_ret.empty else None

    cummax = prices.cummax()
    dd = (prices / cummax) - 1.0
    max_dd = float(dd.min()) if not dd.empty else None

    return {"return_pct": float(total_return), "cagr": float(cagr) if cagr is not None else None,
            "volatility": vol, "max_drawdown": max_dd}

def make_chart(df: pd.DataFrame, title: str, chart_type: str, log_scale: bool, show_sma: bool) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title="No data")
        return fig

    if chart_type == "Candlestick (OHLC)" and all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="OHLC"))
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode="lines", name="Adj Close"))

    if show_sma:
        for win in (50, 200):
            if df.shape[0] >= win:
                sma = df["Adj Close"].rolling(win).mean()
                fig.add_trace(go.Scatter(x=df.index, y=sma, mode="lines", name=f"SMA {win}"))

    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Price",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=30, t=60, b=40),
    )
    fig.update_yaxes(type="log" if log_scale else "linear")
    return fig


# Function to compare Analytics
def get_prices_for(symbols, start, end, period, interval, auto_adjust):
    """Fetch Adj Close for many symbols and align into one DataFrame."""
    frames = {}
    for s in symbols:
        df = get_history(s, start, end, period, interval, auto_adjust)
        if not df.empty and "Adj Close" in df:
            frames[s.upper()] = df["Adj Close"].rename(s.upper())
    if not frames:
        return pd.DataFrame()
    prices = pd.concat(frames.values(), axis=1).dropna(how="all")
    return prices

def make_compare_chart(prices: pd.DataFrame, title: str) -> go.Figure:
    """Normalize series to 100 at the first common date and plot."""
    fig = go.Figure()
    if prices.empty:
        fig.update_layout(title="No data for comparison")
        return fig
    prices = prices.dropna()
    base = prices.iloc[0]
    norm = (prices / base) * 100.0
    for col in norm.columns:
        fig.add_trace(go.Scatter(x=norm.index, y=norm[col], mode="lines", name=col))
    fig.update_layout(
        title=title + " (Indexed to 100)",
        xaxis_title="Date",
        yaxis_title="Index (Start = 100)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=30, t=60, b=40),
    )
    return fig

def annual_returns(df: pd.DataFrame) -> pd.Series:
    """Compute annual total returns from Adj Close for one symbol history."""
    if df.empty or "Adj Close" not in df:
        return pd.Series(dtype="float64")
    yearly_last = df["Adj Close"].resample("Y").last()
    returns = yearly_last.pct_change().dropna()
    returns.index = returns.index.year
    return returns

def make_returns_heatmap(returns: pd.Series, symbol: str) -> go.Figure:
    """Single-row heatmap of annual returns."""
    if returns.empty:
        fig = go.Figure()
        fig.update_layout(title="No annual returns to display")
        return fig
    years = list(returns.index.astype(str))
    z = [list(returns.values * 100.0)]  # percent
    fig = go.Figure(
        data=go.Heatmap(
            z=z, x=years, y=[symbol.upper()],
            colorscale="RdYlGn", zmid=0, colorbar=dict(title="Return %")
        )
    )
    fig.update_layout(
        title=f"{symbol.upper()} — Annual Returns",
        xaxis_title="Year", yaxis_title="",
        margin=dict(l=40, r=30, t=60, b=40),
    )
    return fig

def rolling_vol_chart(df: pd.DataFrame, window: int, symbol: str) -> go.Figure:
    """Annualized rolling volatility from daily returns."""
    fig = go.Figure()
    if df.empty or "Adj Close" not in df:
        fig.update_layout(title="No data for rolling stats")
        return fig
    daily = df["Adj Close"].pct_change().dropna()
    roll_vol = daily.rolling(window).std() * np.sqrt(252)
    fig.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol, mode="lines", name=f"Rolling vol ({window}d)"))
    fig.update_layout(
        title=f"{symbol.upper()} — Rolling Volatility ({window} day)",
        xaxis_title="Date", yaxis_title="Annualized Volatility",
        hovermode="x unified",
        margin=dict(l=40, r=30, t=60, b=40),
    )
    return fig


# Function to generate / download a PDF report
def _fig_to_imgreader(fig: go.Figure) -> Optional[ImageReader]:
    """Convert Plotly figure to an ImageReader using kaleido."""
    try:
        png = fig.to_image(format="png", scale=2)  # requires 'kaleido'
        return ImageReader(BytesIO(png))
    except Exception as e:
        st.error("Failed to render chart image. Ensure `kaleido` is installed.")
        st.exception(e)
        return None

def generate_pdf_report(symbol: str,
                        range_choice: str,
                        metrics: Dict[str, Optional[float]],
                        fig_main: go.Figure,
                        fig_returns: Optional[go.Figure],
                        fig_rolling: Optional[go.Figure]) -> bytes:
    """Build a multi-page PDF and return bytes."""
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    W, H = letter
    M = 0.6 * inch

    # Header
    y = H - M
    c.setFont("Helvetica-Bold", 16)
    c.drawString(M, y, f"Finance Report: {symbol.upper()}")
    c.setFont("Helvetica", 10)
    c.drawString(M, y - 14, f"Range: {range_choice}")
    c.drawString(M, y - 28, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 44

    # Metrics block
    lines = [
        f"Last Price: see chart tooltip for most recent close",
        f"Total Return: {pct(metrics.get('return_pct'))}",
        f"CAGR: {pct(metrics.get('cagr'))}",
        f"Volatility (ann.): {pct(metrics.get('volatility'))}",
        f"Max Drawdown: {pct(metrics.get('max_drawdown'))}",
    ]
    c.setFont("Helvetica", 11)
    for ln in lines:
        c.drawString(M, y, ln)
        y -= 14
    y -= 6

    # Helper to place a figure
    def place_fig(figure: go.Figure, title: str, y_pos: float) -> float:
        imgR = _fig_to_imgreader(figure)
        if imgR is None:
            return y_pos
        iw, ih = imgR.getSize()
        max_w = W - 2 * M
        tgt_h = (max_w * ih) / iw
        if tgt_h > (y_pos - M - 32):
            # new page if it won't fit
            c.showPage()
            c.setFont("Helvetica-Bold", 12)
            c.drawString(M, H - M, title)
            y_loc = H - M - 18
        else:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(M, y_pos - 14, title)
            y_loc = y_pos - 18
        c.drawImage(imgR, M, y_loc - tgt_h, width=max_w, height=tgt_h, preserveAspectRatio=True, mask='auto')
        return y_loc - tgt_h - 16

    # Add charts
    y = place_fig(fig_main, "Price Chart", y)
    if fig_returns is not None:
        y = place_fig(fig_returns, "Annual Returns", y)
    if fig_rolling is not None:
        y = place_fig(fig_rolling, "Rolling Volatility", y)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()


# UI section
st.set_page_config(page_title="Finance Analyzer", page_icon="📈", layout="wide")
st.title("📈 Finance Analyzer — Stocks & ETFs")

conn = get_conn()  # SQLite connection (cached)

with st.sidebar:
    st.subheader("Search")
    query = st.text_input("Enter a ticker or name (e.g., 'AAPL' or 'Apple')", value="VFV.TO")

    results = search_symbols(query) if query else []
    help_msg = "Select a match below or type a known symbol directly above."

    values_map: Dict[str, Dict] = {}
    if results:
        labels = []
        for q in results:
            sym = q.get("symbol", "")
            nm = q.get("shortname") or q.get("longname") or ""
            exch = q.get("exchangeDisplay") or q.get("exchange") or ""
            typ = q.get("typeDisp") or q.get("quoteType") or ""
            label = f"{sym} — {nm} ({exch} · {typ})" if nm else f"{sym} ({exch} · {typ})"
            labels.append(label)
            values_map[label] = q
        choice = st.selectbox("Matches", labels, help=help_msg)
        selected_symbol = values_map[choice]["symbol"] if choice else query.strip()
    else:
        st.caption("No matches (or offline search). Using your input as the ticker symbol.")
        selected_symbol = query.strip()

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
    log_scale = st.checkbox("Log scale", value=False)
    show_sma = st.checkbox("Show 50/200 SMA", value=True)
    auto_adjust = st.checkbox("Use adjusted prices (recommended)", value=True)
    interval = "1d"

    # Favorites
    st.divider()
    st.subheader("⭐ Favorites")
    fav_label = st.text_input("Label (optional)", key="fav_label", placeholder="e.g., Core S&P 500")
    if st.button("Add current symbol to favorites", use_container_width=True):
        if selected_symbol:
            add_favorite(conn, selected_symbol, fav_label)
            st.success(f"Added {selected_symbol.upper()} to favorites.")
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

    favs = list_favorites(conn)
    if favs.empty:
        st.caption("No favorites yet. Add one above!")
    else:
        display = [f"{row.symbol}" + (f" — {row.label}" if row.label else "") for _, row in favs.iterrows()]
        pick = st.selectbox("Go to favorite", ["—"] + display, index=0)
        if pick != "—":
            selected_symbol = pick.split(" —")[0].strip()
            st.info(f"Loaded favorite: {selected_symbol}")

        st.markdown("**Manage favorites**")
        for _, row in favs.iterrows():
            c1, c2 = st.columns([4, 1])
            with c1:
                st.caption(f"{row.symbol}" + (f" — {row.label}" if row.label else ""))
            with c2:
                if st.button("🗑️", key=f"del_{row.id}", help=f"Remove {row.symbol}"):
                    remove_favorite(conn, row.symbol)
                    try:
                        st.rerun()
                    except Exception:
                        st.experimental_rerun()

st.markdown(
    "Use the sidebar to search by name or ticker, choose a date range, and switch chart types. "
    "Data via `yfinance`. Favorites are saved locally."
)

# Resolve range
if range_choice == "Custom":
    start, end, period = custom_start, custom_end, None
else:
    start, end, period = compute_date_range(range_choice)

# Fetch & display
with st.spinner(f"Loading {selected_symbol}…"):
    hist = get_history(selected_symbol, start, end, period, interval, auto_adjust)

left, right = st.columns((7, 5))

with left:
    if hist.empty:
        st.warning("No price data returned. Double-check the symbol and range.")
        fig_main = go.Figure()
    else:
        fig_main = make_chart(hist, f"{selected_symbol} — {range_choice}", chart_type, log_scale, show_sma)
        st.plotly_chart(fig_main, use_container_width=True)

with right:
    st.subheader("Quick Stats")
    if hist.empty:
        st.info("Stats will appear here once data loads.")
        metrics = {"return_pct": None, "cagr": None, "volatility": None, "max_drawdown": None}
    else:
        metrics = compute_metrics(hist)
        last_price = float(hist["Adj Close"].iloc[-1]) if "Adj Close" in hist else float(hist["Close"].iloc[-1])

        c1, c2, c3 = st.columns(3)
        c1.metric("Last Price", f"{last_price:,.2f}")
        c2.metric("Total Return", pct(metrics["return_pct"]))
        c3.metric("CAGR", pct(metrics["cagr"]))

        c4, c5 = st.columns(2)
        try:
            last_252 = hist.tail(252)
            wk_low = float(last_252["Adj Close"].min())
            wk_high = float(last_252["Adj Close"].max())
        except Exception:
            wk_low = wk_high = None
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

# Analytics tabs
st.divider()
tab1, tab2, tab3 = st.tabs(["🔀 Compare", "📅 Annual Returns", "📈 Rolling Vol"])

with tab1:
    st.caption("Compare multiple tickers by normalizing each to 100 at the start date.")
    fav_options = favs.symbol.tolist() if 'favs' in globals() and not favs.empty else []
    compare_favs = st.multiselect("Pick favorites to compare", options=fav_options, default=[])
    manual = st.text_input("Or enter tickers (comma-separated)", placeholder="e.g., VOO, VTI, QQQ")

    manual_syms = [s.strip().upper() for s in manual.split(",") if s.strip()] if manual else []
    symbols = [selected_symbol.upper()] + [s.upper() for s in compare_favs] + manual_syms
    seen = set()
    symbols = [s for s in symbols if not (s in seen or seen.add(s))]

    if len(symbols) <= 1:
        st.info("Add at least one more symbol to compare.")
        fig_compare = None
    else:
        with st.spinner(f"Comparing: {', '.join(symbols)}"):
            prices = get_prices_for(symbols, start, end, period, interval, auto_adjust)
        fig_compare = make_compare_chart(prices, "Relative Performance")
        st.plotly_chart(fig_compare, use_container_width=True)

with tab2:
    st.caption("Year-by-year total returns calculated from adjusted close.")
    rets = annual_returns(hist)
    if rets.empty:
        st.info("Not enough data to compute annual returns for this selection.")
        fig_returns = None
    else:
        fig_returns = make_returns_heatmap(rets, selected_symbol)
        st.plotly_chart(fig_returns, use_container_width=True)
        df_rets = pd.DataFrame({"Year": rets.index, "Return %": (rets.values * 100).round(2)})
        st.dataframe(df_rets, use_container_width=True, hide_index=True)

with tab3:
    st.caption("Rolling annualized volatility from daily returns.")
    rolling_win = st.slider("Window (days)", min_value=20, max_value=250, value=60, step=5, key="rolling_win")
    fig_rolling = rolling_vol_chart(hist, rolling_win, selected_symbol)
    st.plotly_chart(fig_rolling, use_container_width=True)

# PDF report button
st.divider()
if hist.empty:
    st.disabled = True
else:
    if st.button("📄 Generate PDF Report", use_container_width=True):
        # Use current figures where available; if any are None, they’ll be skipped
        try:
            pdf_bytes = generate_pdf_report(
                symbol=selected_symbol,
                range_choice=range_choice,
                metrics=metrics,
                fig_main=fig_main,
                fig_returns=fig_returns if 'fig_returns' in locals() else None,
                fig_rolling=fig_rolling if 'fig_rolling' in locals() else None,
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
        "- **Adjusted prices** include splits/dividends.\n"
        "- If Yahoo search is flaky, type the ticker directly.\n"
        "- PDF export uses `kaleido` (for Plotly images) and `reportlab`."
    )
