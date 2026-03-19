"""utils/chart_builder.py — Plotly chart: candlestick+extended hours+volume+SMA+BB+Fib+S/R | RSI | MACD"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional

from config import SMA_PERIODS
from utils.indicators import compute_all_indicators

SMA_COLORS  = {20: "#38bdf8", 50: "#fb923c", 200: "#a78bfa"}
BULL_COLOR  = "#22c55e"
BEAR_COLOR  = "#ef4444"
BB_COLOR    = "#facc15"
EXT_BULL    = "#86efac"   # extended hours bull (lighter green)
EXT_BEAR    = "#fca5a5"   # extended hours bear (lighter red)
FIB_COLORS  = ["#e2e8f0","#94a3b8","#64748b","#475569","#334155","#1e293b","#0f172a"]

TOTAL_H     = 740
ROW_HEIGHTS = [0.55, 0.22, 0.23]


def build_stock_chart(
    df: pd.DataFrame,
    symbol: str,
    fib_levels: Optional[Dict] = None,
    supports: Optional[List]   = None,
    resistances: Optional[List] = None,
    df_ext: Optional[pd.DataFrame] = None,   # extended-hours OHLCV (optional)
) -> go.Figure:

    df = compute_all_indicators(df.copy())

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=ROW_HEIGHTS,
        vertical_spacing=0.025,
        subplot_titles=[f"{symbol} — Price & Volume", "RSI (14)", "MACD (12,26,9)"],
    )

    # ── Extended-hours candlestick (behind regular) ───────────────────────────
    if df_ext is not None and not df_ext.empty:
        ext_colors = [EXT_BULL if c >= o else EXT_BEAR
                      for c, o in zip(df_ext["Close"], df_ext["Open"])]
        fig.add_trace(go.Candlestick(
            x=df_ext.index, open=df_ext["Open"], high=df_ext["High"],
            low=df_ext["Low"], close=df_ext["Close"],
            increasing_line_color=EXT_BULL, decreasing_line_color=EXT_BEAR,
            name="Extended", opacity=0.5,
            showlegend=True,
        ), row=1, col=1)

    # ── Regular-hours candlestick ─────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color=BULL_COLOR, decreasing_line_color=BEAR_COLOR,
        name="Price", showlegend=True,
    ), row=1, col=1)

    # ── Volume bars ───────────────────────────────────────────────────────────
    vol_colors = [BULL_COLOR if c >= o else BEAR_COLOR
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=vol_colors, marker_opacity=0.20,
        name="Volume", showlegend=False,
    ), row=1, col=1)

    # ── SMAs ─────────────────────────────────────────────────────────────────
    for p in SMA_PERIODS:
        col = f"SMA_{p}"
        if col in df.columns and df[col].notna().any():
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col],
                mode="lines", name=f"SMA {p}",
                line=dict(color=SMA_COLORS.get(p, "#fff"), width=1.5),
            ), row=1, col=1)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    for col, name, dash in [("BB_Upper","BB+2σ","dot"),
                             ("BB_Lower","BB-2σ","dot"),
                             ("BB_Mid",  "BB Mid","dash")]:
        if col in df.columns and df[col].notna().any():
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], mode="lines", name=name,
                line=dict(color=BB_COLOR, width=0.9, dash=dash), opacity=0.65,
            ), row=1, col=1)

    # ── Fibonacci horizontal lines ────────────────────────────────────────────
    if fib_levels:
        for i, (label, val) in enumerate(fib_levels.items()):
            color = FIB_COLORS[min(i, len(FIB_COLORS)-1)]
            fig.add_trace(go.Scatter(
                x=[df.index[0], df.index[-1]], y=[val, val],
                mode="lines", name=f"Fib {label}",
                line=dict(color="#a78bfa", width=0.8, dash="dashdot"),
                showlegend=True, opacity=0.7,
            ), row=1, col=1)
            fig.add_annotation(
                x=df.index[0], y=val, xref="x", yref="y",
                text=f"  Fib {label}: ${val:.2f}",
                showarrow=False, font=dict(size=8, color="#a78bfa"),
                xanchor="left", row=1, col=1,
            )

    # ── Support zones ─────────────────────────────────────────────────────────
    if supports:
        for i, s in enumerate(supports):
            fig.add_trace(go.Scatter(
                x=[df.index[0], df.index[-1]], y=[s, s],
                mode="lines", name="Support" if i == 0 else None,
                showlegend=(i == 0),
                line=dict(color=BULL_COLOR, width=1.2, dash="dot"), opacity=0.7,
            ), row=1, col=1)
            fig.add_annotation(
                x=df.index[-1], y=s, xref="x", yref="y",
                text=f"S ${s:.2f}", showarrow=False,
                font=dict(size=8, color=BULL_COLOR), xanchor="right",
                row=1, col=1,
            )

    # ── Resistance zones ──────────────────────────────────────────────────────
    if resistances:
        for i, r in enumerate(resistances):
            fig.add_trace(go.Scatter(
                x=[df.index[0], df.index[-1]], y=[r, r],
                mode="lines", name="Resistance" if i == 0 else None,
                showlegend=(i == 0),
                line=dict(color=BEAR_COLOR, width=1.2, dash="dot"), opacity=0.7,
            ), row=1, col=1)
            fig.add_annotation(
                x=df.index[-1], y=r, xref="x", yref="y",
                text=f"R ${r:.2f}", showarrow=False,
                font=dict(size=8, color=BEAR_COLOR), xanchor="right",
                row=1, col=1,
            )

    # ── RSI ───────────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"], mode="lines", name="RSI",
        line=dict(color="#818cf8", width=1.5),
    ), row=2, col=1)
    fig.add_hrect(y0=70, y1=100, row=2, col=1, fillcolor="#ef4444", opacity=0.07, line_width=0)
    fig.add_hrect(y0=0,  y1=30,  row=2, col=1, fillcolor="#22c55e", opacity=0.07, line_width=0)
    for lvl in [30, 70]:
        fig.add_hline(y=lvl, line_dash="dash", line_color="#475569", line_width=0.8, row=2, col=1)

    # ── MACD ──────────────────────────────────────────────────────────────────
    hist_colors = [BULL_COLOR if v >= 0 else BEAR_COLOR for v in df["MACD_Hist"]]
    fig.add_trace(go.Bar(
        x=df.index, y=df["MACD_Hist"],
        marker_color=hist_colors, marker_opacity=0.6, name="Histogram",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD"], mode="lines", name="MACD",
        line=dict(color="#38bdf8", width=1.5),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD_Signal"], mode="lines", name="Signal",
        line=dict(color="#fb923c", width=1.2),
    ), row=3, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#475569", line_width=0.8, row=3, col=1)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        height=TOTAL_H,
        margin=dict(l=55, r=70, t=45, b=20),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#94a3b8", size=10),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, bgcolor="rgba(0,0,0,0.4)",
            font=dict(size=9), itemsizing="constant",
        ),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    fig.update_yaxes(gridcolor="#1e293b", zerolinecolor="#1e293b", showgrid=True)
    fig.update_xaxes(gridcolor="#1e293b", showgrid=True, rangebreaks=[
        dict(bounds=["sat","mon"]),   # hide weekends
    ])
    return fig
