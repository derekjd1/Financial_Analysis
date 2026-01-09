from __future__ import annotations
from datetime import date
from typing import List, Optional, Tuple
import pandas as pd
import streamlit as st
import yfinance as yf

# data.py — fetching price data via yfinance (ticker-only)

# Map a UI date-range label (YTD, 1Y, 5Y, MAX) to yfinance-compatible
def compute_date_range(label: str, today: Optional[date] = None) -> Tuple[Optional[date], Optional[date], Optional[str]]:
    if today is None:
        today = date.today()

    label = label.upper()
    if label == "YTD":
        return date(today.year, 1, 1), today, None
    if label == "1Y":
        return today.replace(year=today.year - 1), today, None
    if label == "5Y":
        return today.replace(year=today.year - 5), today, None
    if label == "MAX":
        return None, None, "max"
    return None, None, None


# Download and normalize historical price data for a single ticker using yfinance,
# handling date ranges, column formats, and index cleanup.
@st.cache_data(show_spinner=True, ttl=300)
def get_history(
    symbol: str,
    start: Optional[date],
    end: Optional[date],
    period: Optional[str],
    interval: str,
    auto_adjust: bool,
) -> pd.DataFrame:
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return pd.DataFrame()

    dl_kwargs = dict(interval=interval, auto_adjust=auto_adjust, progress=False, group_by="column")

    if period:
        df = yf.download(symbol, period=period, **dl_kwargs)
    else:
        sdt = pd.to_datetime(start) if start else None
        edt = (pd.to_datetime(end) + pd.Timedelta(days=1)) if end else None
        df = yf.download(symbol, start=sdt, end=edt, **dl_kwargs)

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        try:
            if symbol in df.columns.get_level_values(-1):
                df = df.xs(symbol, axis=1, level=-1)
            else:
                df.columns = df.columns.get_level_values(0)
        except Exception:
            df.columns = [
                " ".join([str(p) for p in tup if p is not None]).strip()  # type: ignore
                for tup in df.columns
            ]

    df.columns = [str(c).strip().title() for c in df.columns]
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]

    try:
        df = df[~df.index.duplicated(keep="last")]
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
        df = df.sort_index()
    except Exception:
        pass

    return df


# Fetch adjusted close prices for multiple tickers and align them
# into a single DataFrame for comparison charts.
def get_prices_for(
    symbols: List[str],
    start: Optional[date],
    end: Optional[date],
    period: Optional[str],
    interval: str,
    auto_adjust: bool,
) -> pd.DataFrame:
    frames = {}
    for s in symbols:
        sym = (s or "").strip().upper()
        if not sym:
            continue
        df = get_history(sym, start, end, period, interval, auto_adjust)
        if not df.empty and "Adj Close" in df:
            frames[sym] = df["Adj Close"].rename(sym)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames.values(), axis=1).dropna(how="all")
