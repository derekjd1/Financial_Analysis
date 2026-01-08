# 📈 Financial Analysis (Streamlit)
# App URL(copy into browser): financialanalysisapplink.streamlit.app

A lightweight Streamlit app for exploring stocks & ETFs.  
Search by ticker symbol (Yahoo Finance), pick a date range (YTD, 1Y, 5Y, MAX, Custom), view interactive Plotly charts, and save favorites in a local SQLite database.

---

## Features

- **Ticker-only search:** enter a ticker symbol (e.g., `AAPL`, `MSFT`, `BNS TO`, `XEQT.TO`).
- **Flexible ranges:** YTD, 1Y, 5Y, MAX, or custom start/end dates.
- **Views:** Line (Adj Close) or Candlestick (OHLC), optional 50/200-day SMAs.
- **Quick stats:** last price, total return, CAGR, annualized volatility, max drawdown, 52-week high low.
- **Compare tickers:** normalize each series to 100 and overlay relative performance.
- **Annual returns heatmap:** year-by-year total returns + compact table.
- **Rolling volatility:** adjustable window (e.g., 60d) with annualized volatility.
- **Favorites (SQLite):** add, label, load, and remove favorites.
- **Investment calculator:** project growth with contributions and expected return + charts.
- **PDF export:** generate a multi-page report with charts (requires `kaleido`).
- **CSV export:** download the displayed price history.


> Data via [`yfinance`](https://github.com/ranaroussi/yfinance).

## Favorites & Persistence

Favorites are stored in a local SQLite database (`favorites.db`).

- **Local runs:** favorites persist between sessions on your machine.
---

## Quick Start

### Clone & enter the project folder
```bash
git clone <YOUR_REPO_URL> Financial_Analysis
cd Financial_Analysis
# Python 3.10-3.12 + virtual env
pip install -r requirements.txt 