"""
utils/chart_builder.py — Plotly chart generation
Candlestick + Volume | RSI | MACD subplots with SMAs, BB, Fibonacci overlays.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
from config import CHART_THEME, CHART_HEIGHT_MAIN, CHART_HEIGHT_RSI, CHART_HEIGHT_MACD, SMA_PERIODS
from utils.indicators import compute_all_indicators, compute_fibonacci, compute_support_resistance


SMA_COLORS = {20: "#38bdf8", 50: "#fb923c", 200: "#a78bfa"}
BULL_COLOR  = "#22c55e"
BEAR_COLOR  = "#ef4444"
BB_COLOR    = "#facc15"
FIB_COLORS  = ["#94a3b8", "#64748b", "#475569", "#334155", "#1e293b", "#0f172a", "#020617"]


def build_stock_chart(
    df: pd.DataFrame,
    symbol: str,
    fib_levels: Optional[Dict] = None,
    supports: Optional[List] = None,
    resistances: Optional[List] = None,
) -> go.Figure:
    """
    Three-panel chart:
      Panel 1 (tall):   Candlestick + Volume bars + SMAs + BB + Fibonacci + S/R
      Panel 2 (medium): RSI with overbought/oversold bands
      Panel 3 (medium): MACD line + signal + histogram
    """
    df = compute_all_indicators(df.copy())

    row_heights = [0.55, 0.22, 0.23]
    total_height = CHART_HEIGHT_MAIN + CHART_HEIGHT_RSI + CHART_HEIGHT_MACD

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.03,
        subplot_titles=[f"{symbol} Price & Volume", "RSI (14)", "MACD (12,26,9)"],
    )

    # ── Candlestick ───────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color=BULL_COLOR, decreasing_line_color=BEAR_COLOR,
        name="Price",
        showlegend=False,
    ), row=1, col=1)

    # ── Volume (secondary y-axis effect via bar opacity) ─────────────────────
    colors = [BULL_COLOR if c >= o else BEAR_COLOR
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=colors, marker_opacity=0.25,
        name="Volume", yaxis="y2",
        showlegend=False,
    ), row=1, col=1)

    # ── SMAs ──────────────────────────────────────────────────────────────────
    for p in SMA_PERIODS:
        col_name = f"SMA_{p}"
        if col_name in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col_name],
                mode="lines", name=f"SMA {p}",
                line=dict(color=SMA_COLORS.get(p, "#ffffff"), width=1.2, dash="solid"),
            ), row=1, col=1)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    for band_col, band_name, dash in [("BB_Upper", "BB Upper", "dot"),
                                       ("BB_Lower", "BB Lower", "dot"),
                                       ("BB_Mid",   "BB Mid",   "dash")]:
        if band_col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[band_col],
                mode="lines", name=band_name,
                line=dict(color=BB_COLOR, width=0.8, dash=dash),
                opacity=0.6,
            ), row=1, col=1)

    # ── Fibonacci Levels ──────────────────────────────────────────────────────
    if fib_levels:
        for i, (label, val) in enumerate(fib_levels.items()):
            fig.add_hline(
                y=val, line_dash="dashdot",
                line_color=FIB_COLORS[i % len(FIB_COLORS)],
                line_width=0.8, opacity=0.7,
                annotation_text=f"Fib {label}: ${val:.2f}",
                annotation_position="left",
                annotation_font_size=9,
                row=1, col=1,
            )

    # ── Support / Resistance ──────────────────────────────────────────────────
    if supports:
        for s in supports:
            fig.add_hline(
                y=s, line_dash="dot", line_color="#22c55e",
                line_width=1.0, opacity=0.55,
                annotation_text=f"S ${s:.2f}",
                annotation_position="right",
                annotation_font_size=9,
                row=1, col=1,
            )
    if resistances:
        for r in resistances:
            fig.add_hline(
                y=r, line_dash="dot", line_color="#ef4444",
                line_width=1.0, opacity=0.55,
                annotation_text=f"R ${r:.2f}",
                annotation_position="right",
                annotation_font_size=9,
                row=1, col=1,
            )

    # ── RSI ───────────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"],
        mode="lines", name="RSI",
        line=dict(color="#818cf8", width=1.5),
    ), row=2, col=1)

    fig.add_hrect(y0=70, y1=100, row=2, col=1,
                  fillcolor="#ef4444", opacity=0.08, line_width=0)
    fig.add_hrect(y0=0,  y1=30,  row=2, col=1,
                  fillcolor="#22c55e", opacity=0.08, line_width=0)
    for level in [30, 70]:
        fig.add_hline(y=level, line_dash="dash", line_color="#475569",
                      line_width=0.8, row=2, col=1)

    # ── MACD ──────────────────────────────────────────────────────────────────
    macd_hist_colors = [BULL_COLOR if v >= 0 else BEAR_COLOR for v in df["MACD_Hist"]]
    fig.add_trace(go.Bar(
        x=df.index, y=df["MACD_Hist"],
        marker_color=macd_hist_colors, marker_opacity=0.6,
        name="MACD Hist",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD"],
        mode="lines", name="MACD",
        line=dict(color="#38bdf8", width=1.5),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD_Signal"],
        mode="lines", name="Signal",
        line=dict(color="#fb923c", width=1.2),
    ), row=3, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#475569",
                  line_width=0.8, row=3, col=1)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        template=CHART_THEME,
        height=total_height,
        margin=dict(l=50, r=60, t=40, b=20),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0, bgcolor="rgba(0,0,0,0.3)",
            font=dict(size=10),
        ),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#94a3b8", family="IBM Plex Mono, monospace"),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )

    fig.update_yaxes(gridcolor="#1e293b", zerolinecolor="#1e293b")
    fig.update_xaxes(gridcolor="#1e293b", zerolinecolor="#1e293b")

    return fig
