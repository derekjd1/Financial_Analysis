# charts.py — Plotly chart builders

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def make_chart(df: pd.DataFrame, title: str, chart_type: str, show_sma: bool) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title="No data")
        return fig

    if chart_type == "Candlestick (OHLC)" and all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="OHLC"
        ))
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
    return fig


def make_compare_chart(prices: pd.DataFrame, title: str) -> go.Figure:
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


def make_returns_heatmap(returns: pd.Series, symbol: str) -> go.Figure:
    if returns.empty:
        fig = go.Figure()
        fig.update_layout(title="No annual returns to display")
        return fig

    years = list(returns.index.astype(str))
    z = [list(returns.values * 100.0)]
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
