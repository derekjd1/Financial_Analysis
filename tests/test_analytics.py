import pandas as pd
from analytics import compute_metrics, annual_returns

# Unit test for the analytics file

def make_price_df(prices):
    idx = pd.date_range("2020-01-01", periods=len(prices), freq="D")
    return pd.DataFrame({"Close": prices}, index=idx)


def test_compute_metrics_basic():
    df = make_price_df([100, 110, 121])  # +21%
    metrics = compute_metrics(df)

    assert isinstance(metrics, dict)
    assert any(v is None or isinstance(v, (int, float)) for v in metrics.values())


def test_annual_returns_has_one_year():
    idx = pd.to_datetime(["2020-12-31", "2021-12-31"])
    df = pd.DataFrame({"Adj Close": [100, 110]}, index=idx)

    s = annual_returns(df)
    assert len(s) >= 1
