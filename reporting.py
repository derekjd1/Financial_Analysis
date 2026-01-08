# reporting.py — PDF generation for the app

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Dict, Optional

import streamlit as st
import plotly.graph_objects as go

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def _fig_to_imgreader(fig: go.Figure) -> Optional[ImageReader]:
    """Convert Plotly figure to an ImageReader using kaleido."""
    try:
        png = fig.to_image(format="png", scale=2)  # requires 'kaleido'
        return ImageReader(BytesIO(png))
    except Exception as e:
        st.error("Failed to render chart image. Ensure `kaleido` is installed.")
        st.exception(e)
        return None


def generate_pdf_report(
    symbol: str,
    range_choice: str,
    metrics: Dict[str, Optional[float]],
    fig_main: go.Figure,
    fig_returns: Optional[go.Figure],
    fig_rolling: Optional[go.Figure],
    pct_fmt,
) -> bytes:
    """Build a multi-page PDF and return bytes."""
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    W, H = letter
    M = 0.6 * inch

    y = H - M
    c.setFont("Helvetica-Bold", 16)
    c.drawString(M, y, f"Finance Report: {symbol.upper()}")
    c.setFont("Helvetica", 10)
    c.drawString(M, y - 14, f"Range: {range_choice}")
    c.drawString(M, y - 28, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 44

    lines = [
        "Last Price: see chart tooltip for most recent close",
        f"Total Return: {pct_fmt(metrics.get('return_pct'))}",
        f"CAGR: {pct_fmt(metrics.get('cagr'))}",
        f"Volatility (ann.): {pct_fmt(metrics.get('volatility'))}",
        f"Max Drawdown: {pct_fmt(metrics.get('max_drawdown'))}",
    ]
    c.setFont("Helvetica", 11)
    for ln in lines:
        c.drawString(M, y, ln)
        y -= 14
    y -= 6

    def place_fig(figure: go.Figure, title: str, y_pos: float) -> float:
        imgR = _fig_to_imgreader(figure)
        if imgR is None:
            return y_pos

        iw, ih = imgR.getSize()
        max_w = W - 2 * M
        tgt_h = (max_w * ih) / iw

        if tgt_h > (y_pos - M - 32):
            c.showPage()
            c.setFont("Helvetica-Bold", 12)
            c.drawString(M, H - M, title)
            y_loc = H - M - 18
        else:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(M, y_pos - 14, title)
            y_loc = y_pos - 18

        c.drawImage(imgR, M, y_loc - tgt_h, width=max_w, height=tgt_h, preserveAspectRatio=True, mask="auto")
        return y_loc - tgt_h - 16

    y = place_fig(fig_main, "Price Chart", y)
    if fig_returns is not None:
        y = place_fig(fig_returns, "Annual Returns", y)
    if fig_rolling is not None:
        y = place_fig(fig_rolling, "Rolling Volatility", y)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()
