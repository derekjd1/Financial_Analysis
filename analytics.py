from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import pandas as pd

# analytics.py — calculations/metrics

def pct(x: Optional[float]) -> str:
    return f"{x*100:.2f}%" if x is not None else "—"


def _normalize_time_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()
    return df[~df.index.isna()]


def compute_metrics(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    if df is None or df.empty:
        return {"return_pct": None, "cagr": None, "volatility": None, "max_drawdown": None}

    prices = df["Adj Close"].dropna() if "Adj Close" in df else pd.Series(dtype="float64")
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

    return {
        "return_pct": float(total_return),
        "cagr": float(cagr) if cagr is not None else None,
        "volatility": vol,
        "max_drawdown": max_dd,
    }


def annual_returns(df: pd.DataFrame) -> pd.Series:
    """Compute annual total returns from Adj Close for one symbol history."""
    if df.empty or "Adj Close" not in df:
        return pd.Series(dtype="float64")

    yearly_last = df["Adj Close"].resample("YE").last()
    returns = yearly_last.pct_change().dropna()
    returns.index = returns.index.year
    return returns

