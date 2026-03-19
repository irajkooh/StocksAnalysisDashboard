"""utils/chart_builder.py — mplfinance candlestick chart with signal overlays."""

import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from PIL import Image
from typing import Dict, List, Optional

from config import SMA_PERIODS
from utils.indicators import compute_all_indicators

# ── Dark style ────────────────────────────────────────────────────────────────
_MC = mpf.make_marketcolors(
    up="#22c55e", down="#ef4444",
    edge="inherit",
    wick={"up": "#22c55e", "down": "#ef4444"},
    volume={"up": "#22c55e", "down": "#ef4444"},
)
_STYLE = mpf.make_mpf_style(
    marketcolors=_MC,
    facecolor="#0f172a",
    edgecolor="#334155",
    figcolor="#0f172a",
    gridcolor="#1e293b",
    gridstyle="--",
    y_on_right=True,
    rc={
        "font.size": 9,
        "text.color": "#94a3b8",
        "axes.labelcolor": "#94a3b8",
        "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8",
        "axes.titlecolor": "#cbd5e1",
    },
)

SMA_CLR = {20: "#38bdf8", 50: "#fb923c", 200: "#a78bfa"}


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def build_stock_chart(
    df: pd.DataFrame,
    symbol: str,
    fib_levels:  Optional[Dict] = None,
    supports:    Optional[List] = None,
    resistances: Optional[List] = None,
) -> Image.Image:

    df = _flatten(df.copy())
    df = compute_all_indicators(df)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    ap = []

    # ── SMAs ──────────────────────────────────────────────────────────────────
    for p in SMA_PERIODS:
        col = f"SMA_{p}"
        if col in df.columns and df[col].notna().any():
            ap.append(mpf.make_addplot(
                df[col], panel=0,
                color=SMA_CLR.get(p, "#94a3b8"), width=1.4,
                label=f"SMA{p}",
            ))

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    for col, clr, ls in [
        ("BB_Upper", "#facc15", "--"),
        ("BB_Lower", "#facc15", "--"),
        ("BB_Mid",   "#facc1580", ":"),
    ]:
        if col in df.columns and df[col].notna().any():
            ap.append(mpf.make_addplot(
                df[col], panel=0, color=clr, width=0.8,
                linestyle=ls, alpha=0.6,
            ))

    # ── RSI ───────────────────────────────────────────────────────────────────
    if "RSI" in df.columns and df["RSI"].notna().any():
        ap.append(mpf.make_addplot(
            df["RSI"], panel=1, color="#818cf8", width=1.4,
            ylabel="RSI", ylim=(0, 100),
        ))
        ap.append(mpf.make_addplot(
            pd.Series(70, index=df.index), panel=1,
            color="#ef4444", width=0.7, linestyle="--", alpha=0.5,
        ))
        ap.append(mpf.make_addplot(
            pd.Series(30, index=df.index), panel=1,
            color="#22c55e", width=0.7, linestyle="--", alpha=0.5,
        ))

    # ── MACD ──────────────────────────────────────────────────────────────────
    if all(c in df.columns for c in ["MACD", "MACD_Signal", "MACD_Hist"]):
        macd_ok = df["MACD"].notna().any()
        if macd_ok:
            hc = ["#22c55e" if v >= 0 else "#ef4444"
                  for v in df["MACD_Hist"].fillna(0)]
            ap.append(mpf.make_addplot(
                df["MACD_Hist"], panel=2, type="bar",
                color=hc, alpha=0.65, ylabel="MACD",
            ))
            ap.append(mpf.make_addplot(
                df["MACD"], panel=2, color="#38bdf8", width=1.4,
            ))
            ap.append(mpf.make_addplot(
                df["MACD_Signal"], panel=2, color="#fb923c", width=1.1,
            ))

    # ── Horizontal lines (support, resistance, fibonacci) ─────────────────────
    hline_vals, hline_colors = [], []

    for s in (supports or []):
        hline_vals.append(s)
        hline_colors.append("#22c55e")
    for r in (resistances or []):
        hline_vals.append(r)
        hline_colors.append("#ef4444")
    for val in (fib_levels or {}).values():
        hline_vals.append(val)
        hline_colors.append("#c084fc")

    hlines_kwargs = {}
    if hline_vals:
        hlines_kwargs["hlines"] = dict(
            hlines=hline_vals,
            colors=hline_colors,
            linestyle="--",
            linewidths=1.0,
            alpha=0.7,
        )

    # ── Plot ──────────────────────────────────────────────────────────────────
    num_panels = 1 + (1 if any(p.get("panel") == 1 for p in (
        [a._make_addplot_dict() if hasattr(a, "_make_addplot_dict") else a
         for a in ap]
    )) else 0)

    # Simpler panel count: always 3 (price, RSI, MACD)
    panel_ratios = (4, 1.2, 1.2)

    try:
        fig, axes = mpf.plot(
            df,
            type="candle",
            style=_STYLE,
            volume=True,
            addplot=ap if ap else None,
            panel_ratios=panel_ratios,
            figsize=(14, 9),
            title=f"\n{symbol}",
            returnfig=True,
            warn_too_much_data=10000,
            **hlines_kwargs,
        )
    except Exception:
        # Fallback: minimal chart without addplots
        fig, axes = mpf.plot(
            df, type="candle", style=_STYLE, volume=True,
            figsize=(14, 9), title=f"\n{symbol}", returnfig=True,
            warn_too_much_data=10000, **hlines_kwargs,
        )

    fig.patch.set_facecolor("#0f172a")
    plt.tight_layout(pad=1.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="#0f172a")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()
