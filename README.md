# 📈 Financial Analysis (Streamlit)

A lightweight Streamlit app for exploring stocks & ETFs.  
Search by name or ticker (Yahoo Finance), pick a date range (YTD, 1Y, 5Y, MAX, Custom), view interactive Plotly charts, and save favorites in a local SQLite database.

---

## Features

- **Smart search**: type a company/ETF name or ticker (Yahoo Finance autocomplete).
- **Flexible ranges**: YTD, 1Y, 5Y, MAX, or custom start/end dates.
- **Views**: Line (Adj Close) or Candlestick (OHLC), optional 50/200-day SMAs, optional log scale.
- **Quick stats**: last price, total return, CAGR, annualized volatility, max drawdown, 52-week high/low.
- **Favorites (SQLite)**: add, label, load, and remove favorites. Persists between runs.
- **Compare multiple tickers**: normalize each series to 100 and overlay performance.
- **Annual returns heatmap**: year-by-year returns + a compact table.
- **Rolling volatility**: adjustable window (e.g., 60d) with annualized vol.
- **Generate PDF report**: exports the main chart + (when available) heatmap & rolling vol.
- **Export**: download the displayed price history as CSV.


> Data via [`yfinance`](https://github.com/ranaroussi/yfinance).

---

## Quick Start

### 1) Clone & enter the project folder
```bash
git clone <YOUR_REPO_URL> Financial_Analysis
cd Financial_Analysis
