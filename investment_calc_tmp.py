# investment_calc.py — Self-contained Investment Calculator tab for the Streamlit app

from __future__ import annotations

from typing import Tuple, List, Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def money(x: float | None) -> str:
    """Format a float as currency."""
    return f"${x:,.2f}" if x is not None else "—"


def investment_projection(
    principal: float,
    annual_return_pct: float,
    annual_contrib: float,
    years: int,
    contrib_timing: str = "End",  # "End" (ordinary annuity) or "Beginning" (annuity due)
) -> tuple[pd.DataFrame, float, float, float]:
    """
    Compute a yearly projection given a constant annual rate and annual contributions.

    Returns:
        df (pd.DataFrame): Yearly rows with Balance, cumulative contributions & interest.
        end_balance (float)
        total_contribs (float)
        total_interest (float)
    """
    r = max(annual_return_pct / 100.0, 0.0)
    years = max(int(years), 0)

    balance = float(principal)
    contribs_cum = 0.0
    interest_cum = 0.0

    rows: List[Dict] = []

    for yr in range(1, years + 1):
        # Contribution at beginning (annuity due)
        if contrib_timing == "Beginning" and annual_contrib > 0:
            balance += annual_contrib
            contribs_cum += annual_contrib

        # Annual growth
        interest = balance * r
        balance += interest
        interest_cum += interest

        # Contribution at end (ordinary annuity)
        if contrib_timing == "End" and annual_contrib > 0:
            balance += annual_contrib
            contribs_cum += annual_contrib

        rows.append(
            {
                "Year": yr,
                "Balance": balance,
                "Total Contributions": contribs_cum,
                "Total Interest": interest_cum,
            }
        )

    df = pd.DataFrame(rows)
    total_interest = max(balance - principal - contribs_cum, 0.0)
    return df, balance, contribs_cum, total_interest


def _growth_line_chart(df_proj: pd.DataFrame) -> go.Figure:
    """Plot projected balance over time."""
    fig = go.Figure()
    if df_proj.empty:
        fig.update_layout(title="No projection to display")
        return fig
    fig.add_trace(
        go.Scatter(
            x=df_proj["Year"],
            y=df_proj["Balance"],
            mode="lines",
            name="Balance",
        )
    )
    fig.update_layout(
        title="Projected Balance Over Time",
        xaxis_title="Year",
        yaxis_title="Balance",
        hovermode="x unified",
        margin=dict(l=40, r=30, t=60, b=40),
    )
    return fig


def _ending_pie_chart(start_amt: float, total_contribs: float, total_interest: float) -> go.Figure:
    """Plot ending composition pie (principal vs. contributions vs. interest)."""
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Starting Amount", "Total Contributions", "Total Interest"],
                values=[start_amt, total_contribs, total_interest],
                hole=0.4,
                sort=False,
            )
        ]
    )
    fig.update_layout(
        title="Ending Balance Composition",
        margin=dict(l=40, r=30, t=60, b=40),
    )
    return fig


def render_investment_calculator_tab() -> None:
    """
    Render the full calculator UI inside the current Streamlit container/tab.
    """
    st.caption(
        "Project your investment growth with annual contributions and a constant expected return."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        start_amt = st.number_input(
            "Starting amount", min_value=0.0, value=10_000.0, step=500.0, format="%.2f"
        )
        years_invest = st.number_input("Years", min_value=0, value=20, step=1)
    with c2:
        exp_return = st.number_input(
            "Expected yearly return (%)", min_value=0.0, value=7.0, step=0.1, format="%.2f"
        )
        contrib = st.number_input(
            "Annual contribution", min_value=0.0, value=6_000.0, step=500.0, format="%.2f"
        )
    with c3:
        timing = st.selectbox(
            "Contribution timing",
            ["End", "Beginning"],
            help="End = ordinary annuity (deposit at year end); Beginning = annuity due (deposit before growth).",
        )
        show_table = st.checkbox("Show yearly table", value=False)

    if years_invest == 0:
        st.info("Enter at least 1 year to project results.")
        return

    df_proj, end_bal, total_contribs, total_interest = investment_projection(
        principal=start_amt,
        annual_return_pct=exp_return,
        annual_contrib=contrib,
        years=years_invest,
        contrib_timing=timing,
    )

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("End Balance", money(end_bal))
    k2.metric("Starting Amount", money(start_amt))
    k3.metric("Total Contributions", money(total_contribs))
    k4.metric("Total Interest", money(total_interest))

    # Charts
    st.plotly_chart(_growth_line_chart(df_proj), use_container_width=True)
    st.plotly_chart(_ending_pie_chart(start_amt, total_contribs, total_interest), use_container_width=True)

    if show_table:
        st.dataframe(
            df_proj.rename(
                columns={
                    "Total Contributions": "Contributions (cum)",
                    "Total Interest": "Interest (cum)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
