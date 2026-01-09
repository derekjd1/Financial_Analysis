from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# charts.py — Plotly chart builders

# Select the appropriate price column from a DataFrame
def _price_col(df: pd.DataFrame) -> str | None:
    for c in ("Adj Close", "Close"):
        if c in df.columns:
            return c
    return None


# Build the main price chart (line or candlestick) with optional moving averages.
def make_chart(df: pd.DataFrame, title: str, chart_type: str, show_sma: bool) -> go.Figure:
    fig = go.Figure()
    if df is None or df.empty:
        return fig.update_layout(title="No data")

    pcol = _price_col(df)

    if chart_type == "Candlestick (OHLC)" and all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
        fig.add_trace(
            go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="OHLC"
            )
        )
    elif pcol is not None:
        fig.add_trace(go.Scatter(x=df.index, y=df[pcol], mode="lines", name="Price"))
    else:
        return fig.update_layout(title="No price column found")

    if show_sma and pcol is not None:
        for win in (50, 200):
            if df.shape[0] >= win:
                sma = df[pcol].rolling(win).mean()
                fig.add_trace(go.Scatter(x=df.index, y=sma, mode="lines", name=f"SMA {win}"))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=30, t=60, b=40),
        xaxis=dict(rangeslider=dict(visible=False)),
    )
    return fig


# Build a normalized comparison chart (indexed to 100) for multiple symbols.
def make_compare_chart(prices: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    if prices is None or prices.empty:
        return fig.update_layout(title="No data for comparison")

    prices = prices.sort_index().copy()
    prices = prices.ffill() 

    base = prices.iloc[0].replace(0, np.nan)
    norm = (prices / base) * 100.0

    for col in norm.columns:
        series = norm[col].dropna()
        if not series.empty:
            fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name=str(col)))

    fig.update_layout(
        title=f"{title} (Indexed to 100)",
        xaxis_title="Date",
        yaxis_title="Index (Start = 100)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=30, t=60, b=40),
        xaxis=dict(rangeslider=dict(visible=False)),
    )
    return fig


# Build a heatmap showing year-by-year total returns for a single symbol.
def make_returns_heatmap(returns: pd.Series, symbol: str) -> go.Figure:
    fig = go.Figure()
    if returns is None or returns.empty:
        return fig.update_layout(title="No annual returns to display")

    idx = returns.index
    if isinstance(idx, pd.DatetimeIndex):
        years = [str(y) for y in idx.year]
    else:
        years = [str(y) for y in idx]

    z = [list(returns.values * 100.0)]
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=years,
            y=[symbol.upper()],
            colorscale="RdYlGn",
            zmid=0,
            colorbar=dict(title="Return %"),
        )
    )
    fig.update_layout(
        title=f"{symbol.upper()} — Annual Returns",
        xaxis_title="Year",
        yaxis_title="",
        margin=dict(l=40, r=30, t=60, b=40),
    )
    return fig



# Build a rolling annualized volatility chart from daily returns.
def rolling_vol_chart(df: pd.DataFrame, window: int, symbol: str) -> go.Figure:
    fig = go.Figure()
    if df is None or df.empty:
        return fig.update_layout(title="No data for rolling stats")

    pcol = _price_col(df)
    if pcol is None:
        return fig.update_layout(title="No price column found")

    daily = df[pcol].pct_change().dropna()
    if daily.empty:
        return fig.update_layout(title="Not enough data for rolling stats")

    roll_vol = daily.rolling(window).std() * np.sqrt(252)
    fig.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol, mode="lines", name=f"Rolling vol ({window}d)"))

    fig.update_layout(
        title=f"{symbol.upper()} — Rolling Volatility ({window} day)",
        xaxis_title="Date",
        yaxis_title="Annualized Volatility",
        hovermode="x unified",
        margin=dict(l=40, r=30, t=60, b=40),
        xaxis=dict(rangeslider=dict(visible=False)),
    )
    return fig
