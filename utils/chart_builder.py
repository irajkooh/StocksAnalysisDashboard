"""utils/chart_builder.py — candlestick charts (mplfinance static + Plotly interactive)."""

import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from PIL import Image
from typing import Dict, List, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.config import SMA_PERIODS
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


# ── Plotly interactive chart (hover-enabled) ──────────────────────────────────

_PLOTLY_SMA_COLORS = {20: "#22c55e", 50: "#fb923c", 100: "#38bdf8", 200: "#a78bfa"}

_AXIS_STYLE = dict(
    gridcolor="#1e293b", gridwidth=1,
    showgrid=True, zeroline=False,
    tickfont=dict(color="#94a3b8", size=10),
    linecolor="#334155",
)


def build_plotly_chart(df: pd.DataFrame, symbol: str) -> str:
    """Return a self-contained HTML fragment with an interactive Plotly chart.

    Hover shows OHLCV + SMA 20/50/100/200 + RSI + MACD for the crosshair date.
    """
    df = _flatten(df.copy())
    df = compute_all_indicators(df)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    has_rsi  = "RSI" in df.columns and df["RSI"].notna().any()
    has_macd = (
        all(c in df.columns for c in ["MACD", "MACD_Signal", "MACD_Hist"])
        and df["MACD"].notna().any()
    )

    n_rows = 1 + int(has_rsi) + int(has_macd)
    row_heights = [0.60] + ([0.20] if has_rsi else []) + ([0.20] if has_macd else [])
    specs = [[{"secondary_y": True}]] + [[{"secondary_y": False}]] * (n_rows - 1)

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        specs=specs,
    )

    # ── Candlestick ───────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="OHLC",
        increasing=dict(line=dict(color="#22c55e"), fillcolor="#22c55e"),
        decreasing=dict(line=dict(color="#ef4444"), fillcolor="#ef4444"),
        hovertext=[
            f"O {o:.2f}  H {h:.2f}  L {l:.2f}  C {c:.2f}"
            for o, h, l, c in zip(df["Open"], df["High"], df["Low"], df["Close"])
        ],
        hoverinfo="x+text",
        xhoverformat="%b %d %Y",
    ), row=1, col=1, secondary_y=False)

    # ── Volume bars ───────────────────────────────────────────────────────────
    vol_colors = [
        "#22c55e" if float(c) >= float(o) else "#ef4444"
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        name="Volume",
        marker_color=vol_colors,
        opacity=0.35,
        hovertemplate="Vol: %{y:,.0f}<extra></extra>",
        showlegend=False,
    ), row=1, col=1, secondary_y=True)

    # ── SMA lines ─────────────────────────────────────────────────────────────
    for period, color in _PLOTLY_SMA_COLORS.items():
        col_name = f"SMA_{period}"
        if col_name in df.columns and df[col_name].notna().any():
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col_name],
                name=f"SMA {period}",
                line=dict(color=color, width=1.6),
                hovertemplate=f"SMA {period}: %{{y:.2f}}<extra></extra>",
            ), row=1, col=1, secondary_y=False)

    rsi_row  = 2 if has_rsi else None
    macd_row = (3 if has_rsi else 2) if has_macd else None

    # ── RSI ───────────────────────────────────────────────────────────────────
    if has_rsi:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"],
            name="RSI", line=dict(color="#818cf8", width=1.4),
            hovertemplate="RSI: %{y:.1f}<extra></extra>",
        ), row=rsi_row, col=1)
        for level, color in [(70, "#ef4444"), (30, "#22c55e")]:
            fig.add_hline(
                y=level, line_dash="dash", line_color=color,
                line_width=0.7, opacity=0.5, row=rsi_row, col=1,
            )

    # ── MACD ──────────────────────────────────────────────────────────────────
    if has_macd:
        hist_colors = [
            "#22c55e" if v >= 0 else "#ef4444"
            for v in df["MACD_Hist"].fillna(0)
        ]
        fig.add_trace(go.Bar(
            x=df.index, y=df["MACD_Hist"],
            name="MACD Hist", marker_color=hist_colors, opacity=0.65,
            hovertemplate="Hist: %{y:.3f}<extra></extra>",
        ), row=macd_row, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD"],
            name="MACD", line=dict(color="#38bdf8", width=1.4),
            hovertemplate="MACD: %{y:.3f}<extra></extra>",
        ), row=macd_row, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD_Signal"],
            name="Signal", line=dict(color="#fb923c", width=1.1),
            hovertemplate="Signal: %{y:.3f}<extra></extra>",
        ), row=macd_row, col=1)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        height=560,
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#94a3b8", size=11),
        title=dict(text=f"<b>{symbol}</b>", font=dict(color="#cbd5e1", size=14), x=0.01),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#1e293b", font_color="#e2e8f0",
            bordercolor="#334155", font_size=12,
        ),
        legend=dict(
            orientation="h", x=0, y=1.04, xanchor="left",
            font=dict(size=11, color="#94a3b8"),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis_rangeslider_visible=False,
        margin=dict(l=8, r=50, t=44, b=8),
        dragmode="pan",
    )
    fig.update_yaxes(**_AXIS_STYLE)
    fig.update_xaxes(**_AXIS_STYLE, showticklabels=False)
    # Show x-axis labels only on the bottom subplot
    fig.update_xaxes(showticklabels=True, row=n_rows, col=1)
    # Hide volume secondary y-axis ticks (volume scale not useful to show)
    fig.update_yaxes(showticklabels=False, row=1, col=1, secondary_y=True)

    html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=True,
        config={
            "responsive": True,
            "displayModeBar": True,
            "scrollZoom": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "toImageButtonOptions": {
                "format": "png",
                "filename": "chart",
                "height": 800,
                "width": 1400,
                "scale": 2,
            },
        },
    )
    # Inject html2canvas-based PNG download that works inside a sandboxed iframe.
    # Plotly's built-in toImage needs Kaleido server-side; this runs fully in-browser.
    _dl_script = f"""
<script>
(function() {{
  function waitForPlotly(cb, n) {{
    n = n || 0;
    var gd = document.querySelector('.js-plotly-plot');
    if (gd && gd._context) {{ cb(gd); }}
    else if (n < 50) {{ setTimeout(function(){{ waitForPlotly(cb, n+1); }}, 200); }}
  }}
  waitForPlotly(function(gd) {{
    var now = new Date();
    var date = now.getFullYear() + '-' +
      String(now.getMonth()+1).padStart(2,'0') + '-' +
      String(now.getDate()).padStart(2,'0');
    var time = String(now.getHours()).padStart(2,'0') + '-' +
      String(now.getMinutes()).padStart(2,'0') + '-' +
      String(now.getSeconds()).padStart(2,'0');
    gd._context.toImageButtonOptions.filename = '{symbol}_' + date + '_' + time;
  }});
}})();
</script>
"""
    return html.replace("</body>", _dl_script + "</body>")
