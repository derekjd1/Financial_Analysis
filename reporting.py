from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Dict, Optional, Callable

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
    except Exception:
        return None


def generate_pdf_report(
    symbol: str,
    range_choice: str,
    metrics: Dict[str, Optional[float]],
    fig_main: go.Figure,
    fig_returns: Optional[go.Figure],
    fig_rolling: Optional[go.Figure],
    pct_fmt: Callable[[Optional[float]], str],
) -> bytes:
    """Build a multi-page PDF and return bytes."""
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.setTitle(f"Finance Report: {symbol.upper()}")

    W, H = letter
    M = 0.6 * inch
    page_num = 1

    def header():
        nonlocal page_num
        y0 = H - M
        c.setFont("Helvetica-Bold", 16)
        c.drawString(M, y0, f"Finance Report: {symbol.upper()}")
        c.setFont("Helvetica", 10)
        c.drawString(M, y0 - 14, f"Range: {range_choice}")
        c.drawString(M, y0 - 28, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        c.setFont("Helvetica", 9)
        c.drawRightString(W - M, M / 2, f"Page {page_num}")
        return y0 - 44

    y = header()

    # Summary metrics
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
    y -= 8

    def new_page(title: str) -> float:
        nonlocal page_num
        c.showPage()
        page_num += 1
        y0 = header()
        c.setFont("Helvetica-Bold", 12)
        c.drawString(M, y0, title)
        return y0 - 18

    def place_fig(figure: go.Figure, title: str, y_pos: float) -> float:
        imgR = _fig_to_imgreader(figure)
        if imgR is None:
            # If image export fails, skip gracefully
            return y_pos

        iw, ih = imgR.getSize()
        max_w = W - 2 * M
        tgt_h = (max_w * ih) / iw

        # Clamp overly tall images to page height
        max_h = H - 2 * M - 60
        if tgt_h > max_h:
            tgt_h = max_h

        needed = tgt_h + 28  # title + padding
        if y_pos - needed < M:
            y_loc = new_page(title)
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

    c.save()
    buf.seek(0)
    return buf.getvalue()
