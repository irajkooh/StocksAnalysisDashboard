"""
frontend.py — Stocks Analysis Dashboard
Target: Gradio 3.50.2 (pinned in requirements.txt)

Gradio 3.50.2 API used here:
- gr.Blocks(), gr.Tabs(), gr.Tab(visible=)
- gr.Chatbot() — tuples [user, bot], no type= param
- gr.State() — works correctly in inputs/outputs
- tab.select(fn, outputs) — fires when tab clicked
- NO js=, NO gr.Timer, NO gr.render, NO type="messages"
- gr.update(visible=, label=) to show/hide tabs dynamically

Add/Delete: instant via gr.update on pre-built tab slots
"""

import logging
import tempfile
import requests as req_lib
import gradio as gr

_GRADIO_MAJOR = int(gr.__version__.split(".")[0])

from config import (
    BACKEND_URL, IS_HF_SPACE, LLM_PROVIDER, OLLAMA_MODEL, GROQ_MODEL,
    DEFAULT_WATCHLIST, REFRESH_OPTIONS, MAX_CHATBOT_MEMORY,
)
from utils.device import get_device_label
from utils.session_manager import load_session, save_session
from utils.tts_engine import text_to_speech_file

logger = logging.getLogger(__name__)

MAX_SLOTS = 12  # pre-built tab slots

_owned_map:      dict = {}
_analysis_cache: dict = {}
_chat_history:   dict = {}
_chatbot_ctx:    dict = {}
_watchlist:      list = []

SAMPLE_QUESTIONS = [
    "Good entry point?",
    "Key support levels?",
    "What does RSI say?",
    "Over or undervalued?",
    "What is the risk?",
    "MACD signal?",
]

# ── Backend ───────────────────────────────────────────────────────────────────

def _api(path, method="GET", payload=None, timeout=180):
    try:
        url = f"{BACKEND_URL}{path}"
        r = req_lib.get(url, timeout=timeout) if method == "GET" \
            else req_lib.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"API error: {e}")
        return {}

def _analyze_api(ticker, owns):
    return _api("/analyze", "POST", {"ticker": ticker, "owns_stock": owns})

def _chat_api(ticker, question):
    return _api("/chat", "POST", {
        "ticker": ticker,
        "question": question,
        "chatbot_ctx": _chatbot_ctx.get(ticker, ""),
        "history": _chat_history.get(ticker, []),
    }).get("response", "No response.")

# ── HTML helpers ──────────────────────────────────────────────────────────────

def _status_bar():
    env = "HuggingFace" if IS_HF_SPACE else "Local"
    dev = get_device_label()
    mdl = OLLAMA_MODEL if LLM_PROVIDER == "ollama" else GROQ_MODEL
    return (
        '<div style="background:linear-gradient(135deg,#0d1b2a,#1a2744);'
        'border-bottom:2px solid #1e40af;padding:10px 20px;'
        'display:flex;align-items:center;gap:16px;font-family:monospace;flex-wrap:wrap">'
        '<b style="color:#38bdf8;font-size:16px;letter-spacing:2px">'
        '&#128202; STOCKS ANALYSIS DASHBOARD</b>'
        f'<span style="color:#60a5fa;font-size:12px">| {env} | {dev} | {LLM_PROVIDER}/{mdl}</span>'
        '</div>'
    )

def _c(label, value, color="#38bdf8", sub=None):
    s = f'<div style="color:#64748b;font-size:10px;margin-top:2px">{sub}</div>' if sub else ""
    return (
        f'<div style="background:#1e293b;border:1px solid #334155;border-radius:8px;'
        f'padding:8px 12px;text-align:center;min-width:80px;display:inline-block;margin:2px">'
        f'<div style="color:#64748b;font-size:9px;text-transform:uppercase;'
        f'letter-spacing:1px;margin-bottom:3px">{label}</div>'
        f'<div style="color:{color};font-size:14px;font-weight:700">{value}</div>{s}</div>'
    )

def _session_pills(si: dict) -> str:
    """Inline pre-market / regular / after-hours pills for the hero box."""
    if not si:
        return ""

    def _pill(label, price, change, pct, lc):
        if price is not None:
            sign  = "+" if (change or 0) >= 0 else ""
            color = "#22c55e" if (change or 0) >= 0 else "#ef4444"
            arrow = "▲" if (change or 0) >= 0 else "▼"
            ch    = f"{sign}{change:.2f}" if change is not None else "—"
            pc    = f"{sign}{pct:.2f}%" if pct is not None else "—"
            body  = (
                f'<div style="color:#f1f5f9;font-size:13px;font-weight:700">${price:.2f}</div>'
                f'<div style="color:{color};font-size:11px;font-weight:600">'
                f'{arrow} {ch}&nbsp;&nbsp;{pc}</div>'
            )
        else:
            body = '<div style="color:#475569;font-size:10px;margin-top:4px">—</div>'

        return (
            f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;'
            f'padding:6px 14px;text-align:center;min-width:100px">'
            f'<div style="color:{lc};font-size:9px;text-transform:uppercase;'
            f'letter-spacing:1px;font-weight:700;margin-bottom:2px">{label}</div>'
            f'{body}</div>'
        )

    pre  = _pill("Pre-Market",  si.get("pre_price"),     si.get("pre_change"),     si.get("pre_pct"),     "#60a5fa")
    reg  = _pill("Regular",     si.get("regular_price"), si.get("regular_change"), si.get("regular_pct"), "#38bdf8")
    post = _pill("After-Hours", si.get("post_price"),    si.get("post_change"),    si.get("post_pct"),    "#a78bfa")
    return f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px">{pre}{reg}{post}</div>'


def _hero_html(ticker, action, price, dcf, owns, session_info=None):
    ac   = "#22c55e" if "BUY" in str(action) else "#ef4444" if "SELL" in str(action) else "#facc15"
    iv   = dcf.get("intrinsic_value")
    prem = dcf.get("premium_discount")
    iv_s = f"${iv:.2f}" if iv is not None else "N/A"
    pr_s = f"{prem:+.1f}%" if prem is not None else "N/A"
    pc   = "#22c55e" if (prem or 0) < 0 else "#ef4444"
    badge = ('<b style="background:#7c3aed;color:#fff;padding:2px 8px;border-radius:12px;'
             'font-size:11px;margin-left:8px">I OWN</b>') if owns else ""
    sess = _session_pills(session_info or {})
    header = (
        f'<div style="color:#38bdf8;font-size:13px;font-weight:700;font-family:monospace;'
        f'text-transform:uppercase;letter-spacing:3px;margin-bottom:8px;'
        f'border-bottom:1px solid #1e3a5f;padding-bottom:6px">'
        f'&#128202; {ticker} &mdash; Stock Analysis</div>'
    )
    return (
        header +
        '<div style="background:linear-gradient(135deg,#0f172a,#1e293b);'
        'border:2px solid #1e40af;border-radius:12px;padding:16px 24px;margin-bottom:10px;'
        'display:flex;align-items:center;gap:20px;flex-wrap:wrap">'
        f'<div>'
        f'<div style="color:#94a3b8;font-size:10px;font-family:monospace;'
        f'text-transform:uppercase;letter-spacing:2px">{ticker}</div>'
        f'<div style="color:#f8fafc;font-size:30px;font-weight:800;margin:3px 0">'
        f'${price:.2f}{badge}</div>'
        f'<div style="color:{ac};font-size:20px;font-weight:700">{action}</div>'
        f'{sess}'
        f'</div>'
        f'<div style="display:flex;gap:6px;flex-wrap:wrap">'
        f'{_c("Intrinsic", iv_s, "#a78bfa")}'
        f'{_c("Premium", pr_s, pc, "vs DCF")}'
        f'</div></div>'
    )

def _signals_html(decision, indicators, risk):
    reasons = decision.get("reasons", [])
    pp   = decision.get("probability_profit", 50)
    pl   = decision.get("probability_loss", 50)
    conf = decision.get("confidence", 0)
    rsi  = indicators.get("rsi", 50)
    rc   = "#22c55e" if rsi < 35 else "#ef4444" if rsi > 65 else "#facc15"
    def dc(r):
        t = r.lower()
        if any(w in t for w in ["bull","buy","above","under","oversold"]): return "#22c55e"
        if any(w in t for w in ["bear","sell","below","over","risk"]): return "#ef4444"
        return "#facc15"
    rows = "".join(
        f'<div style="color:#cbd5e1;font-size:12px;padding:3px 0;border-bottom:1px solid #1e293b">'
        f'<span style="color:{dc(r)}">&#9679;</span> {r}</div>' for r in reasons)
    cards = (
        _c("RSI",    f"{rsi:.1f}",                              rc, indicators.get("rsi_state",""))
        + _c("ATR",  f'${indicators.get("atr",0):.2f}',         "#fb923c")
        + _c("StochK",f'{indicators.get("stoch_k",50):.1f}',    "#a78bfa")
        + _c("VolAnn",f'{risk.get("annual_volatility",0):.1f}%', "#f43f5e")
        + _c("Sharpe",f'{risk.get("sharpe_ratio",0):.2f}',       "#22c55e")
        + _c("Risk",  f'{risk.get("risk_score",5)}/10',          "#fb923c", risk.get("risk_label",""))
    )
    return (
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px">'
        '<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px">'
        '<div style="color:#38bdf8;font-weight:700;font-size:12px;margin-bottom:6px">KEY SIGNALS</div>'
        + rows +
        '</div><div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px">'
        '<div style="color:#38bdf8;font-weight:700;font-size:12px;margin-bottom:6px">INDICATORS</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:3px">{cards}</div>'
        '</div></div>'
        '<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;'
        'padding:12px;margin-bottom:10px">'
        '<div style="color:#38bdf8;font-weight:700;font-size:12px;margin-bottom:8px">TRADE PROBABILITY</div>'
        '<div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">'
        '<div style="flex:1">'
        '<div style="color:#94a3b8;font-size:11px;margin-bottom:3px">Profit</div>'
        '<div style="background:#0f172a;border-radius:6px;height:18px;overflow:hidden">'
        f'<div style="background:#22c55e;width:{pp}%;height:100%;display:flex;align-items:center;'
        f'justify-content:center;color:#fff;font-size:10px;font-weight:700">{pp}%</div></div></div>'
        '<div style="flex:1">'
        '<div style="color:#94a3b8;font-size:11px;margin-bottom:3px">Loss</div>'
        '<div style="background:#0f172a;border-radius:6px;height:18px;overflow:hidden">'
        f'<div style="background:#ef4444;width:{pl}%;height:100%;display:flex;align-items:center;'
        f'justify-content:center;color:#fff;font-size:10px;font-weight:700">{pl}%</div></div></div>'
        f'<div style="color:#60a5fa;font-size:12px;font-weight:700">Conf: {conf}%</div>'
        '</div></div>'
    )

def _levels_html(supports, resistances, pivots, fibonacci):
    def fmt(lst, color):
        if not lst: return "<span style='color:#475569'>N/A</span>"
        return " | ".join(
            f'<span style="color:{color};font-weight:700">${v:.2f}</span>' for v in lst)
    fib_rows = "".join(
        f'<div style="display:flex;justify-content:space-between;padding:2px 0;'
        f'border-bottom:1px solid #1e293b">'
        f'<span style="color:#94a3b8;font-size:10px">{k}</span>'
        f'<span style="color:#a78bfa;font-size:10px;font-weight:600">${v:.2f}</span></div>'
        for k, v in list(fibonacci.items())[:7])
    piv_rows = "".join(
        f'<div style="display:flex;justify-content:space-between;padding:2px 0;'
        f'border-bottom:1px solid #1e293b">'
        f'<span style="color:#94a3b8;font-size:10px">{k}</span>'
        f'<span style="color:#{"22c55e" if k.startswith("R") else "ef4444" if k.startswith("S") else "facc15"};'
        f'font-size:10px;font-weight:700">${v:.2f}</span></div>'
        for k, v in pivots.items())
    return (
        '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:10px">'
        '<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px">'
        f'<div style="color:#22c55e;font-weight:700;font-size:12px;margin-bottom:5px">SUPPORT</div>'
        f'<div style="margin-bottom:8px">{fmt(supports, "#22c55e")}</div>'
        f'<div style="color:#ef4444;font-weight:700;font-size:12px;margin-bottom:5px">RESISTANCE</div>'
        f'<div>{fmt(resistances, "#ef4444")}</div></div>'
        '<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px">'
        f'<div style="color:#a78bfa;font-weight:700;font-size:12px;margin-bottom:5px">FIBONACCI</div>'
        + fib_rows + '</div>'
        '<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px">'
        f'<div style="color:#facc15;font-weight:700;font-size:12px;margin-bottom:5px">PIVOT POINTS</div>'
        + piv_rows + '</div></div>'
    )

def _fundamentals_html(fund, dcf):
    def row(label, val, color="#f1f5f9"):
        v = str(val) if val is not None else "N/A"
        return (
            f'<div style="display:flex;justify-content:space-between;padding:3px 0;'
            f'border-bottom:1px solid #1e293b">'
            f'<span style="color:#94a3b8;font-size:11px">{label}</span>'
            f'<span style="color:{color};font-size:11px;font-weight:600">{v}</span></div>'
        )
    dc  = dcf.get("label", "N/A")
    dcc = "#22c55e" if "Under" in dc else "#ef4444" if "Over" in dc else "#facc15"
    dy  = fund.get("dividend_yield")
    roe = fund.get("return_on_equity")
    return (
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px">'
        '<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px">'
        '<div style="color:#38bdf8;font-weight:700;font-size:12px;margin-bottom:6px">FUNDAMENTALS</div>'
        + row("Sector",   fund.get("sector", "N/A"))
        + row("P/E",      fund.get("pe_ratio"))
        + row("Fwd P/E",  fund.get("forward_pe"))
        + row("P/B",      fund.get("price_to_book"))
        + row("Beta",     fund.get("beta"))
        + row("Div Yield",f"{dy*100:.2f}%" if dy else "N/A")
        + '</div>'
        '<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px">'
        '<div style="color:#a78bfa;font-weight:700;font-size:12px;margin-bottom:6px">DCF VALUATION</div>'
        + row("Intrinsic Value",  f'${dcf.get("intrinsic_value","N/A")}', "#a78bfa")
        + row("Current Price",    f'${dcf.get("current_price","N/A")}')
        + row("Premium/Disc.",    f'{dcf.get("premium_discount","N/A")}%', dcc)
        + row("Margin of Safety", f'{dcf.get("margin_of_safety","N/A")}%', "#22c55e")
        + row("Valuation",        dc, dcc)
        + row("ROE",              f"{roe*100:.1f}%" if roe else "N/A")
        + '</div></div>'
    )

def _sentiment_html(sentiment):
    agg = sentiment.get("aggregate_score", 0)
    lbl = sentiment.get("aggregate_label", "N/A")
    ac  = "#22c55e" if agg > 0.1 else "#ef4444" if agg < -0.1 else "#facc15"
    legend = (
        '<div style="display:flex;gap:10px;font-size:10px;margin-bottom:7px;'
        'padding:4px 8px;background:#0f172a;border-radius:4px;flex-wrap:wrap">'
        '<span style="color:#94a3b8">-1=Bearish | 0=Neutral | +1=Bullish</span>'
        '<span style="color:#22c55e">&#9632; Pos</span>'
        '<span style="color:#facc15">&#9632; Neutral</span>'
        '<span style="color:#ef4444">&#9632; Neg</span></div>'
    )
    def bar(icon, name, score, count, src_color):
        raw = float(score)
        pct = int((raw + 1) / 2 * 100)
        bc  = "#22c55e" if raw > 0.05 else "#ef4444" if raw < -0.05 else "#facc15"
        sgn = "+" if raw > 0 else ""
        return (
            f'<div style="margin-bottom:7px">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:2px">'
            f'<span style="color:{src_color};font-size:11px;font-weight:600">{icon} {name}</span>'
            f'<span style="color:#94a3b8;font-size:10px">{sgn}{raw:.2f} ({count})</span></div>'
            f'<div style="background:#0f172a;border-radius:3px;height:8px;position:relative;overflow:hidden">'
            f'<div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:#334155;z-index:1"></div>'
            f'<div style="background:{bc};width:{pct}%;height:100%;border-radius:3px"></div></div></div>'
        )
    tw = sentiment.get("twitter", {})
    rd = sentiment.get("reddit",  {})
    nw = sentiment.get("news",    {})
    sc = sentiment.get("sec",     {})
    return (
        '<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;'
        'padding:12px;margin-bottom:10px">'
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">'
        '<b style="color:#38bdf8;font-size:12px">SENTIMENT</b>'
        f'<b style="color:{ac};font-size:14px">{lbl} ({agg:+.2f})</b></div>'
        + legend
        + bar("𝕏", "X.com",  tw.get("score",0), tw.get("count",0), "#1d9bf0")
        + bar("R", "Reddit", rd.get("score",0), rd.get("count",0), "#ff4500")
        + bar("N", "News",   nw.get("score",0), nw.get("count",0), "#38bdf8")
        + bar("S", "SEC",    sc.get("score",0), sc.get("count",0), "#a78bfa")
        + '</div>'
    )

def _wl_html(wl, active_syms, selected=""):
    if not wl:
        return '<span style="color:#475569;font-size:11px">Empty</span>'
    items = "".join(
        f'<div onclick="document.getElementById(\'wlb_{s}\').querySelector(\'button\').click();document.getElementById(\'wls_{s}\').querySelector(\'button\').click()" '
        f'style="background:{"#1e3a5f" if s == selected else "#0f2a1a" if s in active_syms else "#1e293b"};'
        f'border:1px solid {"#38bdf8" if s == selected else "#334155"};border-radius:6px;padding:4px 8px;'
        f'color:{"#22c55e" if s in active_syms else "#38bdf8"};'
        f'font-size:12px;font-weight:600;font-family:monospace;cursor:pointer">'
        f'{"&#10003; " if s in active_syms else ""}{"&#9998; " if s == selected else ""}{s}</div>'
        for s in wl)
    return f'<div style="display:flex;flex-direction:column;gap:4px">{items}</div>'

# ── Analysis runner ───────────────────────────────────────────────────────────

def _session_bar(si: dict) -> str:
    """Render pre-market / regular / after-hours bar above the chart."""
    if not si:
        return ""

    def _pill(label, price, change, pct, label_color):
        if price is None:
            return ""
        sign   = "+" if (change or 0) >= 0 else ""
        color  = "#22c55e" if (change or 0) >= 0 else "#ef4444"
        arrow  = "▲" if (change or 0) >= 0 else "▼"
        ch_str = f"{sign}{change:.2f}" if change is not None else "—"
        pt_str = f"{sign}{pct:.2f}%" if pct is not None else "—"
        return (
            f'<div style="display:flex;flex-direction:column;align-items:center;'
            f'padding:6px 14px;background:#1e293b;border-radius:8px;min-width:160px">'
            f'<span style="color:{label_color};font-size:9px;text-transform:uppercase;'
            f'letter-spacing:1px;margin-bottom:3px;font-weight:600">{label}</span>'
            f'<span style="color:#f1f5f9;font-size:15px;font-weight:700">${price:.2f}</span>'
            f'<span style="color:{color};font-size:11px;font-weight:600">'
            f'{arrow} {ch_str} ({pt_str})</span>'
            f'</div>'
        )

    pre  = _pill("Pre-Market",  si.get("pre_price"),     si.get("pre_change"),     si.get("pre_pct"),     "#60a5fa")
    reg  = _pill("Regular",     si.get("regular_price"), si.get("regular_change"), si.get("regular_pct"), "#38bdf8")
    post = _pill("After-Hours", si.get("post_price"),    si.get("post_change"),    si.get("post_pct"),    "#a78bfa")

    pills = "".join(p for p in [pre, reg, post] if p)
    if not pills:
        return ""
    return (
        f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px;'
        f'padding:8px 12px;background:#0f172a;border:1px solid #1e293b;border-radius:10px">'
        f'{pills}</div>'
    )


def _tv_chart(symbol: str, session_info: dict = None) -> str:
    """TradingView Advanced Chart widget embedded in an iframe."""
    uid = f"tv_{symbol}"
    return f"""
<div style="height:560px;width:100%;border-radius:8px;overflow:hidden;border:1px solid #1e293b">
<iframe srcdoc='<!DOCTYPE html><html><head>
<meta charset="utf-8">
<style>html,body{{margin:0;padding:0;height:100%;background:#0f172a}}</style>
</head><body>
<div class="tradingview-widget-container" style="height:100%;width:100%">
  <div id="{uid}" style="height:100%;width:100%"></div>
  <script src="https://s3.tradingview.com/tv.js"></script>
  <script>
    new TradingView.widget({{
      container_id: "{uid}",
      autosize: true,
      symbol: "{symbol}",
      interval: "D",
      timezone: "America/New_York",
      theme: "dark",
      style: "1",
      locale: "en",
      withdateranges: true,
      hide_side_toolbar: false,
      allow_symbol_change: false,
      studies: ["RSI@tv-basicstudies","MACD@tv-basicstudies","BB@tv-basicstudies"]
    }});
  </script>
</div>
</body></html>'
width="100%" height="560" frameborder="0"
style="width:100%;height:560px;border:none"
sandbox="allow-scripts allow-same-origin allow-popups"
></iframe></div>"""


def _run(ticker):
    owns = _owned_map.get(ticker, False)
    data = _analyze_api(ticker, owns)
    if not data or (data.get("errors") and not data.get("indicators")):
        err = "; ".join(data.get("errors", ["Unknown error"]))
        return (
            f'<div style="color:#ef4444;padding:14px"><b>Error ({ticker}):</b> {err}</div>',
            _tv_chart(ticker), "", "", "", "", f"**Failed:** {err}", f"**Failed:** {err}"
        )
    ind  = data.get("indicators", {})
    dcf  = data.get("dcf", {})
    dec  = data.get("decision", {})
    fund = data.get("fundamentals", {})
    sent = data.get("sentiment", {})
    risk = data.get("risk", {})
    _chatbot_ctx[ticker]    = data.get("llm_chatbot_ctx", "")
    _analysis_cache[ticker] = data
    si = data.get("session_info", {})
    return (
        _hero_html(ticker, dec.get("action","N/A"), ind.get("price",0), dcf, owns, si),
        _tv_chart(ticker, si),
        _signals_html(dec, ind, risk),
        _levels_html(data.get("supports",[]), data.get("resistances",[]),
                     data.get("pivots",{}), data.get("fibonacci",{})),
        _fundamentals_html(fund, dcf),
        _sentiment_html(sent),
        f"### 📊 {ticker} — AI Analysis\n\n" + data.get("llm_summary",""),
        f"### 📊 {ticker} — AI Analysis\n\n" + data.get("llm_summary",""),
    )

def _render_from_data(ticker, data):
    ind  = data.get("indicators", {})
    dcf  = data.get("dcf", {})
    dec  = data.get("decision", {})
    fund = data.get("fundamentals", {})
    sent = data.get("sentiment", {})
    risk = data.get("risk", {})
    owns   = _owned_map.get(ticker, False)
    report = f"### 📊 {ticker} — AI Analysis\n\n" + data.get("llm_summary", "")
    si     = data.get("session_info", {})
    return (
        _hero_html(ticker, dec.get("action","N/A"), ind.get("price",0), dcf, owns, si),
        _tv_chart(ticker, si),
        _signals_html(dec, ind, risk),
        _levels_html(data.get("supports",[]), data.get("resistances",[]),
                     data.get("pivots",{}), data.get("fibonacci",{})),
        _fundamentals_html(fund, dcf),
        _sentiment_html(sent),
        report,
        report,  # report_state
    )


# ── Tab slot helpers ──────────────────────────────────────────────────────────

def _tab_updates(syms):
    """Return gr.update() for each of MAX_SLOTS tabs based on current syms list."""
    updates = []
    for i in range(MAX_SLOTS):
        if i < len(syms) and syms[i]:
            updates.append(gr.update(visible=True, label=syms[i]))
        else:
            updates.append(gr.update(visible=False))
    return updates

# ── App ────────────────────────────────────────────────────────────────────────

def build_app():
    session     = load_session()
    init_syms   = list(session.get("symbols", []))
    _owned_map.update(session.get("owned", {}))
    _watchlist.clear()
    _watchlist.extend(session.get("watchlist", list(DEFAULT_WATCHLIST)))
    saved_ref   = session.get("refresh_interval", "Off")

    _blocks_kwargs = {} if _GRADIO_MAJOR >= 6 else {"css": CSS, "theme": THEME}
    demo = gr.Blocks(title="Stocks Analysis Dashboard", **_blocks_kwargs)

    with demo:
        gr.HTML(_status_bar())

        # ── Toolbar row 1 ──────────────────────────────────────────────────
        with gr.Row():
            sym_in  = gr.Textbox(placeholder="Enter symbol e.g. AAPL",
                                 show_label=False, scale=3, max_lines=1)
            add_btn = gr.Button("Add Symbol",    variant="primary",   scale=1)
            wf_btn  = gr.Button("Show Workflow", variant="secondary", scale=1)

        # ── Toolbar row 2 ──────────────────────────────────────────────────
        with gr.Row():
            ref_btn  = gr.Button("Analyze Stock", variant="primary",   scale=2, elem_id="analyze_btn")
            ref_all  = gr.Button("Analyze All",   variant="primary",   scale=2, elem_id="ar_ref_all")
            save_btn = gr.Button("💾 Save Dashboard",        variant="secondary", scale=1, elem_id="save_btn")
            ref_dd   = gr.Dropdown(choices=list(REFRESH_OPTIONS.keys()),
                                   value=saved_ref, label="Auto-Refresh", scale=1,
                                   elem_id="ar_dd", min_width=120)

        status_msg = gr.HTML("")
        wf_panel   = gr.HTML(value="", visible=False)
        wf_vis     = gr.State(False)
        wf_cache   = gr.State("")
        # visible=False hides the Gradio wrapper (no border line rendered).
        # In Gradio 6 the element is still mounted in the DOM, so JS can read textarea.value.
        ar_secs    = gr.Textbox(value=str(REFRESH_OPTIONS.get(saved_ref, 0)),
                                elem_id="ar_secs_input", visible=False, show_label=False)

        # syms_state: live list of active symbols (drives tab visibility)
        syms_state = gr.State(value=list(init_syms))

        # cur_sym: which symbol the shared panel shows (updated by tab.select)
        cur_sym      = gr.State(value=init_syms[0] if init_syms else "")
        report_state = gr.State("")
        rep_reading  = gr.State(False)
        rep_tts_text  = gr.Textbox(value="", elem_id="rep_tts_buf",  visible=False, show_label=False)
        chat_reading  = gr.State(False)
        chat_tts_text = gr.Textbox(value="", elem_id="chat_tts_buf", visible=False, show_label=False)


        with gr.Row(elem_id="main_row"):
            # Watchlist sidebar
            with gr.Column(scale=1, min_width=130):
                gr.HTML(
                    '<b style="color:#38bdf8;font-size:12px;text-transform:uppercase;'
                    'letter-spacing:1px">Watchlist</b><br>'
                    '<span style="color:#475569;font-size:10px">Select to open as tab</span>'
                )
                wl_radio = gr.Radio(
                    choices=list(_watchlist),
                    value=None,
                    show_label=False,
                    interactive=True,
                    elem_id="watchlist_radio",
                )
                wl_in  = gr.Textbox(placeholder="Add symbol...",
                                    show_label=False, max_lines=1)
                wl_add = gr.Button("+ Add", size="sm")
                wl_del = gr.Button("- Delete", size="sm", variant="stop")

            with gr.Column(scale=6, elem_id="main_col"):
                # ── Pre-built tab slots ────────────────────────────────────
                # MAX_SLOTS tabs exist at all times; unused ones are hidden.
                # Adding a symbol makes the next hidden slot visible with the new label.
                # Deleting hides the slot.
                tab_objs    = []   # gr.Tab objects
                del_tab_btns = []  # per-tab delete buttons

                with gr.Tabs():
                    for i in range(MAX_SLOTS):
                        sym     = init_syms[i] if i < len(init_syms) else ""
                        visible = bool(sym)
                        label   = sym if sym else f"__slot{i}__"
                        with gr.Tab(label=label, visible=visible) as t:
                            with gr.Row():
                                own_chk = gr.Checkbox(
                                    label=f"I OWN {sym}" if sym else "I OWN",
                                    value=_owned_map.get(sym, False) if sym else False,
                                    scale=4,
                                )
                                dtab_btn = gr.Button(
                                    "Delete", size="sm", variant="stop",
                                    min_width=0, scale=1,
                                )
                            # CRITICAL: read current symbol from syms_state at slot index i
                            # NOT from the closure variable sym (which is frozen at startup)
                            # This ensures that after add/delete, clicking a tab returns
                            # the symbol currently assigned to that slot.
                            t.select(
                                fn=lambda syms, idx=i: list(syms)[idx] if idx < len(list(syms)) else "",
                                inputs=[syms_state],
                                outputs=[cur_sym],
                            )
                            own_chk.change(
                                fn=lambda v, syms, idx=i: _owned_map.update(
                                    {list(syms)[idx]: v}
                                ) if idx < len(list(syms)) and list(syms)[idx] else None,
                                inputs=[own_chk, syms_state],
                            )
                        tab_objs.append(t)
                        del_tab_btns.append(dtab_btn)


                # ── Shared analysis panel ──────────────────────────────────
                hero_out    = gr.HTML(
                    '<div style="color:#475569;padding:12px">'
                    'Select a tab above, then click <b>Analyze Stock</b> to analyze.</div>'
                )
                chart_out   = gr.HTML("")
                signals_out = gr.HTML()
                levels_out  = gr.HTML()
                fund_out    = gr.HTML()
                sent_out    = gr.HTML()

                with gr.Accordion("AI Analysis Report", open=False):
                    report_out = gr.Markdown("*Run analysis to see AI report.*")
                    rd_rep     = gr.Button("▶ READ", size="sm", variant="secondary")

                with gr.Accordion("Ask About This Stock", open=False):
                    _chat_kwargs = {} if _GRADIO_MAJOR >= 6 else {"type": "messages"}
                    chatbot  = gr.Chatbot(height=240, show_label=False, **_chat_kwargs)
                    with gr.Row():
                        chat_in = gr.Textbox(
                            placeholder="Ask a question...",
                            show_label=False, scale=5,
                        )
                        rd_chat = gr.Button("▶ READ", size="sm", scale=1)
                        snd_btn = gr.Button("Send", variant="primary", size="sm", scale=1)
                    gr.HTML('<div style="color:#64748b;font-size:10px;margin:3px 0">'
                            'Quick questions (auto-submit):</div>')
                    with gr.Row():
                        qbtns = [gr.Button(q, size="sm", min_width=0)
                                 for q in SAMPLE_QUESTIONS]
                    with gr.Row():
                        cpy_btn = gr.Button("Copy Chat", size="sm", variant="secondary")
                        clr_btn = gr.Button("Clear Chat", size="sm", variant="stop")
                    cpy_out    = gr.HTML("")
                    copy_buf   = gr.Textbox(value="", elem_id="chat_copy_buf", visible=True, show_label=False)

        # ── Output lists ───────────────────────────────────────────────────
        PANEL = [hero_out, chart_out, signals_out, levels_out,
                 fund_out, sent_out, report_out, report_state]

        # ── Add symbol ─────────────────────────────────────────────────────
        def do_add(raw_sym, syms):
            sym  = raw_sym.strip().upper()
            syms = list(syms)
            if not sym:
                return ("", syms,
                        '<div style="color:#ef4444;font-size:12px">Enter a symbol.</div>',
                        *_tab_updates(syms))
            if sym in syms:
                return ("", syms,
                        f'<div style="color:#facc15;font-size:12px">{sym} already open.</div>',
                        *_tab_updates(syms))
            if len(syms) >= MAX_SLOTS:
                return ("", syms,
                        f'<div style="color:#ef4444;font-size:12px">Max {MAX_SLOTS} tabs reached.</div>',
                        *_tab_updates(syms))
            syms.append(sym)
            _owned_map.setdefault(sym, False)
            save_session(syms, _owned_map, _watchlist)
            return ("", syms,
                    f'<div style="color:#22c55e;font-size:12px"><b>{sym}</b> added.</div>',
                    *_tab_updates(syms))

        def _sync_tabs(syms):
            """Re-apply tab updates from current syms_state (fixes Gradio 5.x render lag)."""
            return _tab_updates(list(syms))

        _ADD_OUTPUTS = [sym_in, syms_state, status_msg] + tab_objs
        (add_btn.click(fn=do_add, inputs=[sym_in, syms_state], outputs=_ADD_OUTPUTS)
                .then(fn=_sync_tabs, inputs=[syms_state], outputs=tab_objs))
        (sym_in.submit(fn=do_add, inputs=[sym_in, syms_state], outputs=_ADD_OUTPUTS)
                .then(fn=_sync_tabs, inputs=[syms_state], outputs=tab_objs))

        # ── Per-tab Delete buttons (reliable: no cur_sym race condition) ───
        def do_delete_slot(syms, idx):
            syms = list(syms)
            if idx >= len(syms) or not syms[idx]:
                return (syms,
                        '<div style="color:#ef4444;font-size:12px">No stock at this slot.</div>',
                        *_tab_updates(syms))
            sym = syms[idx]
            syms.pop(idx)
            _owned_map.pop(sym, None)
            _analysis_cache.pop(sym, None)
            save_session(syms, _owned_map, _watchlist)
            return (syms,
                    f'<div style="color:#22c55e;font-size:12px"><b>{sym}</b> deleted.</div>',
                    *_tab_updates(syms))

        for i, btn in enumerate(del_tab_btns):
            (btn.click(
                fn=lambda syms, idx=i: do_delete_slot(syms, idx),
                inputs=[syms_state],
                outputs=[syms_state, status_msg] + tab_objs,
            ).then(fn=_sync_tabs, inputs=[syms_state], outputs=tab_objs))

        # ── Save ───────────────────────────────────────────────────────────
        def _on_ref_dd_change(syms, rv):
            save_session(list(syms), _owned_map, _watchlist, rv)
            return str(REFRESH_OPTIONS.get(rv, 0))

        ref_dd.change(fn=_on_ref_dd_change, inputs=[syms_state, ref_dd], outputs=[ar_secs])

        # ── Analysis ───────────────────────────────────────────────────────
        def do_analyze(sym):
            sym = (sym or "").strip().upper()
            if not sym:
                return [gr.update()] * 8
            return list(_run(sym))

        def do_refresh(sym):
            sym = (sym or "").strip().upper()
            if not sym:
                return [gr.update()] * 8
            _analysis_cache.pop(sym, None)
            return list(_run(sym))

        def do_refresh_all(syms, active_sym):
            """Clear all caches, re-analyze all symbols."""
            syms = [s for s in list(syms) if s]
            if not syms:
                return [gr.update()] * 8
            for s in syms:
                _analysis_cache.pop(s, None)
                _run(s)
            show = (active_sym or "").strip().upper()
            if show not in syms:
                show = syms[0]
            data = _analysis_cache.get(show)
            if not data:
                return [gr.update()] * 8
            return list(_render_from_data(show, data))

        def on_sym_change(sym):
            sym = (sym or "").strip().upper()
            if not sym:
                return [
                    '<div style="color:#475569;padding:12px">Select a tab above, then click <b>Analyze Stock</b> to analyze.</div>',
                    "", "", "", "", "", "*Run analysis to see AI report.*", "",
                ]
            data = _analysis_cache.get(sym)
            if not data:
                return [
                    f'<div style="color:#475569;padding:12px"><b>{sym}</b> — Click <b>Analyze Stock</b> to analyze.</div>',
                    _tv_chart(sym, {}), "", "", "", "",
                    "*Run analysis to see AI report.*", "",
                ]
            return list(_render_from_data(sym, data))

        cur_sym.change(fn=on_sym_change, inputs=[cur_sym], outputs=PANEL)

        def _clear_chat(sym):
            sym = (sym or "").strip().upper()
            _chat_history.pop(sym, None)
            return []

        def _clear_chat_all(syms):
            for s in list(syms):
                _chat_history.pop(s, None)
            return []

        (ref_btn.click(fn=do_refresh, inputs=[cur_sym], outputs=PANEL)
                .then(fn=_clear_chat, inputs=[cur_sym], outputs=[chatbot]))
        (ref_all.click(fn=do_refresh_all, inputs=[syms_state, cur_sym], outputs=PANEL)
                .then(fn=_clear_chat_all, inputs=[syms_state], outputs=[chatbot]))

        def do_save(syms, ref):
            ok = save_session(list(syms), _owned_map, _watchlist, ref)
            if ok:
                return '<div style="color:#22c55e;font-size:12px">&#10003; Dashboard saved.</div>'
            return '<div style="color:#ef4444;font-size:12px">Save failed.</div>'

        save_btn.click(fn=do_save, inputs=[syms_state, ref_dd], outputs=[status_msg])


        # ── Auto-Refresh (JS polling loop injected at page load) ───────────
        _JS_AUTO_REFRESH = """() => {
            var lastRefresh = Date.now();
            function getSecs() {
                var el = document.querySelector('#ar_secs_input textarea');
                if (!el) el = document.querySelector('#ar_secs_input input');
                return el ? (parseFloat(el.value) || 0) : 0;
            }
            function clickRefreshAll() {
                var btn = document.querySelector('#ar_ref_all button');
                if (!btn) btn = document.querySelector('#ar_ref_all');
                if (btn) btn.click();
            }
            function tick() {
                var secs = getSecs();
                if (secs > 0 && (Date.now() - lastRefresh) >= secs * 1000) {
                    lastRefresh = Date.now();
                    clickRefreshAll();
                }
            }
            setTimeout(function() { setInterval(tick, 2000); }, 3000);
        }"""
        demo.load(fn=None, js=_JS_AUTO_REFRESH)

        # ── Watchlist ──────────────────────────────────────────────────────
        def do_wl_add(sym):
            sym = sym.strip().upper()
            if sym and sym not in _watchlist:
                _watchlist.append(sym)
                save_session([], _owned_map, _watchlist)
            return gr.update(choices=list(_watchlist), value=None), ""

        wl_add.click(fn=do_wl_add, inputs=[wl_in], outputs=[wl_radio, wl_in])
        wl_in.submit(fn=do_wl_add, inputs=[wl_in], outputs=[wl_radio, wl_in])

        def do_wl_delete(sym, syms):
            sym = (sym or "").strip().upper()
            syms = list(syms)
            if sym and sym in _watchlist:
                _watchlist.remove(sym)
                save_session(syms, _owned_map, _watchlist)
            return gr.update(choices=list(_watchlist), value=None)

        wl_del.click(fn=do_wl_delete, inputs=[wl_radio, syms_state], outputs=[wl_radio])

        def do_wl_select(sym, syms):
            """Add selected watchlist stock as a tab."""
            if not sym:
                return (list(syms), gr.update(), *_tab_updates(list(syms)))
            syms = list(syms)
            if sym in syms:
                return (syms,
                        f'<div style="color:#facc15;font-size:12px">{sym} already open.</div>',
                        *_tab_updates(syms))
            if len(syms) >= MAX_SLOTS:
                return (syms,
                        f'<div style="color:#ef4444;font-size:12px">Max {MAX_SLOTS} tabs reached.</div>',
                        *_tab_updates(syms))
            syms.append(sym)
            _owned_map.setdefault(sym, False)
            save_session(syms, _owned_map, _watchlist)
            return (syms,
                    f'<div style="color:#22c55e;font-size:12px"><b>{sym}</b> added.</div>',
                    *_tab_updates(syms))

        (wl_radio.change(fn=do_wl_select, inputs=[wl_radio, syms_state],
                         outputs=[syms_state, status_msg] + tab_objs)
                 .then(fn=_sync_tabs, inputs=[syms_state], outputs=tab_objs))

        # ── Workflow ───────────────────────────────────────────────────────
        def do_wf(vis, cached):
            new = not vis
            if not new:
                return gr.update(visible=False, value=""), False, cached
            src   = cached or _api("/workflow").get("mermaid", "")
            nodes = [
                ("#0f766e","Data","yfinance"),
                ("#7c3aed","Technical","RSI·MACD·SMA"),
                ("#b45309","Sentiment","News·Reddit"),
                ("#0369a1","Valuation","DCF"),
                ("#b91c1c","Risk","Vol·ATR"),
                ("#065f46","Decision","Buy/Sell"),
            ]
            arrow = '<span style="color:#38bdf8;font-size:13px;margin:0 2px">&#8594;</span>'
            boxes = arrow.join(
                f'<span style="background:{c};border-radius:5px;padding:3px 7px;'
                f'display:inline-block;text-align:center">'
                f'<span style="color:#fff;font-weight:700;font-size:9px">{t}</span><br>'
                f'<span style="color:rgba(255,255,255,.65);font-size:8px">{s}</span></span>'
                for c, t, s in nodes)
            html = (
                '<div style="background:#0f172a;border:1px solid #1e40af;border-radius:8px;'
                'padding:8px 14px;margin-bottom:8px">'
                '<b style="color:#38bdf8;font-size:11px;font-family:monospace">Pipeline: </b>'
                f'<span style="line-height:2.2">{boxes}</span></div>'
            )
            return gr.update(visible=True, value=html), True, src

        wf_btn.click(fn=do_wf, inputs=[wf_vis, wf_cache],
                     outputs=[wf_panel, wf_vis, wf_cache])

        # ── Report TTS (toggle) ─────────────────────────────────────────────
        _JS_SPEAK_REP  = ("() => { var el = document.querySelector('#rep_tts_buf textarea');"
                          " var t = el ? el.value : '';"
                          " window.speechSynthesis.cancel();"
                          " if (t) { var u = new SpeechSynthesisUtterance(t);"
                          " window.speechSynthesis.speak(u); } }")
        _JS_SPEAK_CHAT = ("() => { var el = document.querySelector('#chat_tts_buf textarea');"
                          " var t = el ? el.value : '';"
                          " window.speechSynthesis.cancel();"
                          " if (t) { var u = new SpeechSynthesisUtterance(t);"
                          " window.speechSynthesis.speak(u); } }")

        def toggle_rep_tts(sym, is_reading):
            if is_reading:
                return "", False, gr.update(value="▶ READ")
            ticker = (sym or "").strip().upper()
            summary = _analysis_cache.get(ticker, {}).get("llm_summary", "")
            text = f"{ticker} AI Analysis. {summary}" if summary else ""
            if not text:
                return "", False, gr.update(value="▶ READ")
            return text, True, gr.update(value="⏹ STOP")

        (rd_rep.click(fn=toggle_rep_tts, inputs=[cur_sym, rep_reading],
                      outputs=[rep_tts_text, rep_reading, rd_rep])
               .then(fn=None, js=_JS_SPEAK_REP))

        # ── Chat ───────────────────────────────────────────────────────────
        def do_chat(question, history, ticker):
            q = (question or "").strip()
            if not q or not ticker:
                return history or [], ""
            answer = _chat_api(ticker, q)
            labeled = f"**[{ticker.strip().upper()}]** {answer}"
            new_h = list(history or []) + [
                {"role": "user",      "content": q},
                {"role": "assistant", "content": labeled},
            ]
            _chat_history.setdefault(ticker, [])
            _chat_history[ticker].append([q, labeled])
            if len(_chat_history[ticker]) > MAX_CHATBOT_MEMORY:
                _chat_history[ticker] = _chat_history[ticker][-MAX_CHATBOT_MEMORY:]
            return new_h, ""

        snd_btn.click(fn=do_chat, inputs=[chat_in, chatbot, cur_sym],
                      outputs=[chatbot, chat_in])
        chat_in.submit(fn=do_chat, inputs=[chat_in, chatbot, cur_sym],
                       outputs=[chatbot, chat_in])

        # Sample questions — auto-submit on click
        for q, btn in zip(SAMPLE_QUESTIONS, qbtns):
            def _qfn(history, ticker, question=q):
                return do_chat(question, history, ticker)
            btn.click(fn=_qfn, inputs=[chatbot, cur_sym], outputs=[chatbot, chat_in])

        # Chat READ (toggle)
        def toggle_chat_tts(history, is_reading):
            if is_reading:
                return "", False, gr.update(value="▶ READ")
            text = ""
            for msg in reversed(history or []):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    text = _extract_msg_text(msg.get("content", ""))
                    break
                elif isinstance(msg, (list, tuple)) and len(msg) > 1:
                    text = _extract_msg_text(msg[1] or "")
                    break
            if not text:
                return "", False, gr.update(value="▶ READ")
            return text, True, gr.update(value="⏹ STOP")

        (rd_chat.click(fn=toggle_chat_tts, inputs=[chatbot, chat_reading],
                       outputs=[chat_tts_text, chat_reading, rd_chat])
                .then(fn=None, js=_JS_SPEAK_CHAT))

        # Copy
        def _extract_msg_text(content):
            """Handle Gradio 5.x content: plain str or list of {text, type} dicts."""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in content
                )
            return str(content)

        def do_copy(history):
            if not history:
                return "", '<span style="color:#facc15;font-size:11px">Nothing to copy.</span>'
            lines = []
            for msg in history:
                if isinstance(msg, (list, tuple)) and len(msg) == 2:
                    lines.append(f"You: {_extract_msg_text(msg[0])}\nAI:  {_extract_msg_text(msg[1])}")
                elif isinstance(msg, dict):
                    role = "You" if msg.get("role") == "user" else "AI"
                    lines.append(f"{role}: {_extract_msg_text(msg.get('content', ''))}")
            text = "\n\n".join(lines)
            return text, '<span style="color:#22c55e;font-size:11px">&#10003; Copied!</span>'

        _JS_COPY = ("() => { var el = document.querySelector('#chat_copy_buf textarea');"
                    " var t = el ? el.value : '';"
                    " if (t) navigator.clipboard.writeText(t).catch(()=>{}); }")

        (cpy_btn.click(fn=do_copy, inputs=[chatbot], outputs=[copy_buf, cpy_out])
                .then(fn=None, js=_JS_COPY))

        # Clear
        def do_clear(ticker):
            _chat_history.pop(ticker, None)
            return [], "", ""

        clr_btn.click(fn=do_clear, inputs=[cur_sym], outputs=[chatbot, chat_in, cpy_out])

    demo.queue()
    return demo


CSS = """
body,.gradio-container{background:#0a0f1e !important;font-family:'Segoe UI',system-ui,sans-serif !important;}
button[role="tab"],.tab-nav button{background:#1e293b !important;color:#ffffff !important;border:1px solid #334155 !important;font-weight:600 !important;border-radius:6px 6px 0 0 !important;}
button[role="tab"][aria-selected="true"],.tab-nav button.selected{background:#1e40af !important;color:#ffffff !important;border-color:#3b82f6 !important;}
button[role="tab"]:hover:not([aria-selected="true"]){background:#334155 !important;color:#ffffff !important;}
textarea,input[type=text]{background:#1e293b !important;border:1px solid #334155 !important;color:#f1f5f9 !important;font-family:monospace !important;}
.progress-bar-wrap{background:#1e293b !important;border-radius:8px !important;}
.progress-bar{background:linear-gradient(90deg,#3b82f6,#60a5fa) !important;border-radius:8px !important;}
#chat_copy_buf{display:none !important;}
#ar_dd{font-size:11px !important;min-width:100px !important;}
#ar_dd label{font-size:10px !important;margin-bottom:1px !important;}
#ar_dd .wrap-inner,#ar_dd .wrap{padding:2px 6px !important;min-height:unset !important;}
#ar_dd input,#ar_dd select{font-size:11px !important;padding:1px 4px !important;}
#analyze_btn button{background:linear-gradient(135deg,#0369a1,#0ea5e9) !important;border:none !important;color:#fff !important;font-weight:700 !important;letter-spacing:0.5px !important;}
#analyze_btn button:hover{background:linear-gradient(135deg,#0284c7,#38bdf8) !important;}
#ar_ref_all button{background:linear-gradient(135deg,#065f46,#10b981) !important;border:none !important;color:#fff !important;font-weight:700 !important;letter-spacing:0.5px !important;}
#ar_ref_all button:hover{background:linear-gradient(135deg,#047857,#34d399) !important;}
#save_btn button{background:linear-gradient(135deg,#7c3aed,#a855f7) !important;border:none !important;color:#fff !important;font-weight:700 !important;letter-spacing:0.5px !important;}
#save_btn button:hover{background:linear-gradient(135deg,#6d28d9,#c084fc) !important;}
#watchlist_radio .wrap{gap:3px !important;flex-direction:column !important;}
#watchlist_radio label{font-size:10px !important;padding:3px 8px !important;border-radius:12px !important;background:#1e293b !important;border:1px solid #334155 !important;color:#60a5fa !important;font-weight:600 !important;letter-spacing:0.5px !important;cursor:pointer !important;transition:all .15s !important;}
#watchlist_radio label:hover{background:#1e3a5f !important;border-color:#3b82f6 !important;color:#93c5fd !important;}
#watchlist_radio input[type=radio]:checked~span,#watchlist_radio label:has(input:checked){background:linear-gradient(135deg,#0369a1,#0ea5e9) !important;border-color:#38bdf8 !important;color:#fff !important;}
"""
THEME = gr.themes.Base(primary_hue="blue", secondary_hue="slate", neutral_hue="slate")


if __name__ == "__main__":
    from config import FRONTEND_PORT
    build_app().launch(
        server_name="0.0.0.0",
        server_port=FRONTEND_PORT,
        share=IS_HF_SPACE,
        css=CSS,
        theme=THEME,
        allowed_paths=[tempfile.gettempdir()],
    )
