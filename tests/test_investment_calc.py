import pandas as pd
from investment_calc import investment_projection

# unit test for the unvestment calculator file

def test_investment_projection_returns_dataframe():
    df, end_bal, total_contribs, total_interest = investment_projection(
        principal=1000,
        annual_return_pct=10.0,  
        annual_contrib=0,
        years=3
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3

def test_investment_projection_grows_without_contrib():
    df, end_bal, total_contribs, total_interest = investment_projection(
        principal=1000,
        annual_return_pct=10.0,
        annual_contrib=0,
        years=3
    )
    assert end_bal > 1000

