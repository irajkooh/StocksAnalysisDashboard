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

import asyncio
import asyncio
import logging
import re
import tempfile
import requests as req_lib
import gradio as gr
import rich

_GRADIO_MAJOR = int(gr.__version__.split(".")[0])

from utils.config import (
    BACKEND_URL, IS_HF_SPACE, LLM_PROVIDER, OLLAMA_MODEL, GROQ_MODEL,
    DEFAULT_WATCHLIST, DEFAULT_TABS, REFRESH_OPTIONS, MAX_CHATBOT_MEMORY,
)
from utils.device import get_device_label
from utils.session_manager import load_session, save_session, list_users, create_user, delete_user, rename_user, DEFAULT_USER
from utils.tts_engine import text_to_speech_file

logger = logging.getLogger(__name__)

MAX_SLOTS = 12  # pre-built tab slots

_owned_map:      dict = {}
_analysis_cache: dict = {}
_chat_history:   dict = {}
_chatbot_ctx:    dict = {}
_watchlist:      list = []
_current_user:   str  = ""   # set when a user is selected; gates session saves

SAMPLE_QUESTIONS = [
    "Good entry point?",
    "Key support levels?",
    "What does RSI say?",
    "Over or undervalued?",
    "What is the risk?",
    "MACD signal?",
    "Bullish or bearish?",
    "Where is strong resistance?",
    "What is the trend?",
    "Short-term price target?",
    "Stop-loss suggestion?",
    "Volume analysis?",
]

CT_SAMPLE_QUESTIONS = [
    "Latest Congress trades?",
    "Top politician stocks?",
    "Recent senator trades?",
    "Who bought NVDA?",
    "Top issuers this month?",
    "Any TSLA Congress trades?",
    "Which lawmakers sold recently?",
    "Biggest Congress trade sizes?",
    "Any AI stock trades by Congress?",
    "Who traded energy stocks?",
    "Most active Congress traders?",
    "Any AAPL trades by politicians?",
]

# Keywords that route a question to Capitol Trades instead of per-stock chat
_CT_KEYWORDS = frozenset({
    "politician", "congress", "senator", "representative", "lawmaker",
    "pelosi", "capitol trades", "stock act", "capitol hill",
    "insider", "elected", "who traded", "lawmakers", "house member",
    "who bought", "who sold", "issuer", "trade size", "elected official",
})

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

def _price_refresh_api(ticker):
    return _api(f"/price/{ticker}")

def _chat_api(ticker, question):
    return _api("/chat", "POST", {
        "ticker": ticker,
        "question": question,
        "chatbot_ctx": _chatbot_ctx.get(ticker, ""),
        "history": _chat_history.get(ticker, []),
    }).get("response", "No response.")

def _is_ct_question(q: str) -> bool:
    """Return True if the question is about Capitol Trades / politician trading."""
    q_l = q.lower()
    return any(k in q_l for k in _CT_KEYWORDS)

def _ct_chat_api(question: str) -> str:
    return _api("/chat/capitol_trades", "POST", {
        "question": question,
        "history": _chat_history.get("__ct__", []),
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

def _c(label, value, color="#38bdf8", sub=None, full_name=None, fill=False, sub_color="#64748b"):
    s = f'<div style="color:{sub_color};font-size:10px;margin-top:2px">{sub}</div>' if sub else ""
    if full_name:
        lbl_html = (
            f'<span style="color:#38bdf8;font-size:9px;font-weight:700;letter-spacing:1px">{label}</span>'
            f'<br><span style="color:#38bdf8;font-size:8px;letter-spacing:0">{full_name}</span>'
        )
    else:
        lbl_html = f'<span style="color:#64748b;font-size:9px;text-transform:uppercase;letter-spacing:1px">{label}</span>'
    box_style = "width:100%;box-sizing:border-box" if fill else "min-width:80px;display:inline-block;margin:2px"
    return (
        f'<div style="background:#1e293b;border:1px solid #334155;border-radius:8px;'
        f'padding:8px 12px;text-align:center;{box_style}">'
        f'<div style="margin-bottom:3px">{lbl_html}</div>'
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
            body = '<div style="color:#475569;font-size:13px;font-weight:700">—</div>'
        return (
            f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;'
            f'padding:6px 14px;text-align:center;min-width:100px">'
            f'<div style="color:{lc};font-size:9px;text-transform:uppercase;'
            f'letter-spacing:1px;font-weight:700;margin-bottom:2px">{label}</div>'
            f'{body}</div>'
        )

    cs = si.get("current_session", "regular")
    # During pre-market no regular-session bars exist today — suppress the pill
    # so yesterday's OHLCV close isn't mistaken for a live regular-session price.
    reg_p  = si.get("regular_price")  if cs != "pre" else None
    reg_ch = si.get("regular_change") if cs != "pre" else None
    reg_pc = si.get("regular_pct")    if cs != "pre" else None

    pre  = _pill("Pre-Market",  si.get("pre_price"),       si.get("pre_change"),    si.get("pre_pct"),    "#60a5fa")
    reg  = _pill("Regular",     reg_p,                     reg_ch,                  reg_pc,               "#38bdf8")
    post = _pill("After-Hours", si.get("post_price"),      si.get("post_change"),   si.get("post_pct"),   "#a78bfa")
    ovn  = _pill("Overnight",   si.get("overnight_price"), si.get("overnight_change"), si.get("overnight_pct"), "#f472b6")
    return f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px">{pre}{reg}{post}{ovn}</div>'


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

    # Pick the most current available price based on the active session.
    # current_session reflects the session of the most recently traded bar,
    # preventing stale pre-market prices from showing during regular hours.
    si = session_info or {}
    cs = si.get("current_session", "regular")
    if si.get("post_price") is not None and cs in ("post", "overnight"):
        live_price  = si["post_price"]
        live_change = si.get("post_change")
        live_pct    = si.get("post_pct")
        sess_label  = "After-Hours"
        sess_color  = "#a78bfa"
    elif si.get("pre_price") is not None and cs == "pre":
        live_price  = si["pre_price"]
        live_change = si.get("pre_change")
        live_pct    = si.get("pre_pct")
        sess_label  = "Pre-Market"
        sess_color  = "#60a5fa"
    else:
        live_price  = si.get("regular_price") or price
        live_change = si.get("regular_change")
        live_pct    = si.get("regular_pct")
        sess_label  = "Regular"
        sess_color  = "#38bdf8"

    # Change/pct line
    if live_change is not None and live_pct is not None:
        sign   = "+" if live_change >= 0 else ""
        chg_cl = "#22c55e" if live_change >= 0 else "#ef4444"
        arrow  = "▲" if live_change >= 0 else "▼"
        chg_html = (
            f'<span style="color:{chg_cl};font-size:14px;font-weight:600">'
            f'{arrow} {sign}{live_change:.2f} ({sign}{live_pct:.2f}%)</span>'
        )
    else:
        chg_html = ""

    sess_badge = (
        f'<span style="background:{sess_color}22;border:1px solid {sess_color};'
        f'color:{sess_color};font-size:9px;font-weight:700;padding:1px 7px;'
        f'border-radius:10px;text-transform:uppercase;letter-spacing:1px;'
        f'margin-left:8px;vertical-align:middle">{sess_label}</span>'
    )

    price_time = si.get("price_time")
    time_html  = (
        f'<div style="color:#475569;font-size:10px;font-family:monospace;margin-top:1px">'
        f'last trade: {price_time}</div>'
    ) if price_time else ""

    refreshed_at   = si.get("_refreshed_at")
    refreshed_html = (
        f'<div style="color:#22c55e;font-size:10px;font-family:monospace;margin-top:2px">'
        f'&#128259; prices refreshed at {refreshed_at}</div>'
    ) if refreshed_at else ""

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
        f'${live_price:.2f}{sess_badge}{badge}</div>'
        f'{time_html}'
        f'{refreshed_html}'
        f'<div style="margin-bottom:2px">{chg_html}</div>'
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
    rsi     = indicators.get("rsi", 50)
    stoch_k = indicators.get("stoch_k", 50)
    vol_ann = risk.get("annual_volatility", 0)
    sharpe  = risk.get("sharpe_ratio", 0)
    risk_score = risk.get("risk_score", 5)
    rc   = "#22c55e" if rsi < 35 else "#ef4444" if rsi > 65 else "#facc15"
    rsi_state  = indicators.get("rsi_state", "")
    rsi_sc     = "#22c55e" if rsi < 35 else "#ef4444" if rsi > 65 else "#facc15"
    stoch_state = "Oversold" if stoch_k < 20 else "Overbought" if stoch_k > 80 else "Neutral"
    stoch_sc    = "#22c55e" if stoch_k < 20 else "#ef4444" if stoch_k > 80 else "#facc15"
    vol_state   = "Low" if vol_ann < 20 else "Moderate" if vol_ann < 40 else "High" if vol_ann < 70 else "Very High"
    vol_sc      = "#22c55e" if vol_ann < 20 else "#facc15" if vol_ann < 40 else "#ef4444"
    sharpe_state = "Good" if sharpe > 1 else "Fair" if sharpe >= 0 else "Poor"
    sharpe_sc    = "#22c55e" if sharpe > 1 else "#facc15" if sharpe >= 0 else "#ef4444"
    risk_label   = risk.get("risk_label", "")
    risk_sc      = "#22c55e" if risk_score <= 3 else "#facc15" if risk_score <= 6 else "#ef4444"
    atr      = indicators.get("atr", 0)
    price_ref = indicators.get("price") or 1
    atr_pct  = atr / price_ref * 100
    atr_state = "Low" if atr_pct < 2 else "Moderate" if atr_pct < 5 else "High"
    atr_sc    = "#22c55e" if atr_pct < 2 else "#facc15" if atr_pct < 5 else "#ef4444"
    def dc(r):
        t = r.lower()
        if any(w in t for w in ["bull","buy","above","under","oversold"]): return "#22c55e"
        if any(w in t for w in ["bear","sell","below","over","risk"]): return "#ef4444"
        return "#facc15"
    rows = "".join(
        f'<div style="color:#cbd5e1;font-size:12px;padding:3px 0;border-bottom:1px solid #1e293b">'
        f'<span style="color:{dc(r)}">&#9679;</span> {r}</div>' for r in reasons)
    cards = (
        _c("RSI",    f"{rsi:.1f}",           rsi_sc,    rsi_state,    "Relative Strength Index", fill=True, sub_color=rsi_sc)
        + _c("ATR",  f'${atr:.2f}',  atr_sc, atr_state, "Average True Range", fill=True, sub_color=atr_sc)
        + _c("StochK",f"{stoch_k:.1f}",      stoch_sc,  stoch_state,  "Stochastic %K",          fill=True, sub_color=stoch_sc)
        + _c("VolAnn",f"{vol_ann:.1f}%",     vol_sc,    vol_state,    "Annualized Volatility",   fill=True, sub_color=vol_sc)
        + _c("Sharpe",f"{sharpe:.2f}",       sharpe_sc, sharpe_state, "Sharpe Ratio",            fill=True, sub_color=sharpe_sc)
        + _c("Risk",  f"{risk_score}/10",    risk_sc,   risk_label,   "Risk Score",              fill=True, sub_color=risk_sc)
    )
    return (
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px">'
        '<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px">'
        '<div style="color:#38bdf8;font-weight:700;font-size:12px;margin-bottom:6px">KEY SIGNALS</div>'
        + rows +
        '</div><div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px">'
        '<div style="color:#38bdf8;font-weight:700;font-size:12px;margin-bottom:6px">INDICATORS</div>'
        f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px">{cards}</div>'
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
        + bar("🌐", "Google News Headlines",         tw.get("score",0), tw.get("count",0), "#1d9bf0")
        + bar("💬", "Reddit (WSB + Investing)",       rd.get("score",0), rd.get("count",0), "#ff4500")
        + bar("📰", "News Articles (NewsAPI/Yahoo)",  nw.get("score",0), nw.get("count",0), "#38bdf8")
        + bar("📋", "SEC Filings (8-K/10-K/10-Q)",   sc.get("score",0), sc.get("count",0), "#a78bfa")
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

    cs   = si.get("current_session", "regular")
    pre  = _pill("Pre-Market",  si.get("pre_price"),                              si.get("pre_change"),    si.get("pre_pct"),    "#60a5fa")
    reg  = _pill("Regular",     si.get("regular_price")  if cs != "pre" else None, si.get("regular_change") if cs != "pre" else None, si.get("regular_pct") if cs != "pre" else None, "#38bdf8")
    post = _pill("After-Hours", si.get("post_price"),                             si.get("post_change"),   si.get("post_pct"),   "#a78bfa")

    pills = "".join(p for p in [pre, reg, post] if p)
    if not pills:
        return ""
    return (
        f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px;'
        f'padding:8px 12px;background:#0f172a;border:1px solid #1e293b;border-radius:10px">'
        f'{pills}</div>'
    )


# SMA colour palette — kept in sync with the legend strip below the chart
_SMA_COLORS = {20: "#22c55e", 50: "#fb923c", 100: "#38bdf8", 200: "#a78bfa"}

def _tv_chart(symbol: str, session_info: dict = None) -> str:
    """TradingView Advanced Chart widget embedded in an iframe + SMA legend."""
    uid = f"tv_{symbol}"
    # SMA legend strip
    def _swatch(period, color):
        return (
            f'<span style="display:inline-flex;align-items:center;gap:5px;'
            f'margin-right:14px">'
            f'<span style="display:inline-block;width:28px;height:3px;'
            f'background:{color};border-radius:2px"></span>'
            f'<span style="color:{color};font-size:11px;font-weight:700;'
            f'font-family:monospace">SMA {period}</span></span>'
        )
    c20  = _SMA_COLORS[20]
    c50  = _SMA_COLORS[50]
    c100 = _SMA_COLORS[100]
    c200 = _SMA_COLORS[200]
    legend = (
        '<div style="display:flex;align-items:center;flex-wrap:wrap;'
        'padding:6px 12px;background:#0f172a;border:1px solid #1e293b;'
        'border-top:none;border-radius:0 0 8px 8px">'
        + _swatch(20,  c20)
        + _swatch(50,  c50)
        + _swatch(100, c100)
        + _swatch(200, c200)
        + '</div>'
    )
    return f"""
<div style="border-radius:8px;overflow:hidden;border:1px solid #1e293b">
<div style="height:560px;width:100%">
<iframe srcdoc='<!DOCTYPE html><html><head>
<meta charset="utf-8">
<style>html,body{{margin:0;padding:0;height:100%;background:#0f172a}}</style>
</head><body>
<div class="tradingview-widget-container" style="height:100%;width:100%">
  <div id="{uid}" style="height:100%;width:100%"></div>
  <script src="https://s3.tradingview.com/tv.js"></script>
  <script>
    var _w = new TradingView.widget({{
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
      studies: ["RSI@tv-basicstudies", "MACD@tv-basicstudies", "BB@tv-basicstudies"]
    }});
    _w.onChartReady(function() {{
      var c = _w.chart();
      c.createStudy("Moving Average", false, false, {{length: 20}},  {{"Plot.color": "{c20}",  "Plot.linewidth": 2}});
      c.createStudy("Moving Average", false, false, {{length: 50}},  {{"Plot.color": "{c50}",  "Plot.linewidth": 2}});
      c.createStudy("Moving Average", false, false, {{length: 100}}, {{"Plot.color": "{c100}", "Plot.linewidth": 2}});
      c.createStudy("Moving Average", false, false, {{length: 200}}, {{"Plot.color": "{c200}", "Plot.linewidth": 2}});
    }});
  </script>
</div>
</body></html>'
width="100%" height="560" frameborder="0"
style="width:100%;height:560px;border:none"
sandbox="allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox allow-downloads allow-forms"
></iframe></div>
{legend}
</div>"""


def _wrap_plotly(html: str) -> str:
    if not html:
        return ""
    import html as _html_mod
    escaped = _html_mod.escape(html, quote=True)
    return (
        '<div style="border-radius:8px;overflow:hidden;border:1px solid #1e293b;background:#0f172a">'
        f'<iframe srcdoc="{escaped}" '
        'style="width:100%;height:580px;border:none;background:#0f172a" '
        'sandbox="allow-scripts allow-same-origin allow-downloads"></iframe>'
        '</div>'
    )


def _hero_placeholder(sym=""):
    """Styled hero placeholder before analysis."""
    if not sym:
        return (
            '<div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;'
            'padding:32px 20px;text-align:center;margin-bottom:8px">'
            '<div style="color:#64748b;font-size:14px;margin-bottom:6px">'
            '📊 Select a tab above and click <b style="color:#38bdf8">Analyze Stock</b></div>'
            '<div style="color:#475569;font-size:11px">'
            'Chart, signals, fundamentals, sentiment, and AI report will appear here</div></div>'
        )
    return (
        '<div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;'
        'padding:32px 20px;text-align:center;margin-bottom:8px">'
        f'<div style="color:#38bdf8;font-size:18px;font-weight:700;'
        f'font-family:monospace;margin-bottom:8px">{sym}</div>'
        '<div style="color:#64748b;font-size:13px">'
        'Click <b style="color:#38bdf8">Analyze Stock</b> to load chart, signals, '
        'fundamentals, sentiment &amp; AI report</div></div>'
    )


def _run(ticker):
    owns = _owned_map.get(ticker, False)
    data = _analyze_api(ticker, owns)
    if not data or (data.get("errors") and not data.get("indicators")):
        err = "; ".join(data.get("errors", ["Unknown error"]))
        err_panel = (
            '<div style="min-height:60px;background:#1a0000;border:1px solid #7f1d1d;'
            'border-radius:8px;display:flex;align-items:center;justify-content:center;'
            'margin-bottom:6px">'
            f'<span style="color:#fca5a5;font-size:12px">Analysis failed: {err}</span></div>'
        )
        return (
            f'<div style="color:#ef4444;padding:14px"><b>Error ({ticker}):</b> {err}</div>',
            _tv_chart(ticker), err_panel, err_panel, err_panel, err_panel,
            f"**Failed:** {err}", f"**Failed:** {err}",
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
    chart_html = data.get("chart_json") or _tv_chart(ticker, si)
    return (
        _hero_html(ticker, dec.get("action","N/A"), ind.get("price",0), dcf, owns, si),
        _wrap_plotly(chart_html),
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
    chart_html = data.get("chart_json") or _tv_chart(ticker, si)
    return (
        _hero_html(ticker, dec.get("action","N/A"), ind.get("price",0), dcf, owns, si),
        _wrap_plotly(chart_html),
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
    # No session loaded at startup — wait for user selection
    init_syms = []
    _watchlist.clear()
    _watchlist.extend(list(DEFAULT_WATCHLIST))
    saved_ref = "Off"

    if _GRADIO_MAJOR >= 6:
        _blocks_kwargs = {"fill_width": True}
    else:
        _blocks_kwargs = {"css": CSS, "theme": THEME}
    demo = gr.Blocks(title="Stocks Analysis Dashboard", **_blocks_kwargs)

    with demo:
        gr.HTML(_status_bar(), elem_id="app-header")

        # ── User Management ────────────────────────────────────────────────
        with gr.Row(elem_id="user_mgmt_row"):
            user_dd         = gr.Dropdown(
                choices=list_users(), value=None,
                label="Select User", scale=3, interactive=True,
            )
            load_user_btn   = gr.Button("Load",    variant="primary",   scale=1)
            delete_user_btn = gr.Button("Delete",  variant="stop",      scale=1)
            new_user_in     = gr.Textbox(
                placeholder="New username…", show_label=False,
                scale=2, max_lines=1,
            )
            create_user_btn = gr.Button("Create",  variant="secondary", scale=1)
        with gr.Row(elem_id="user_rename_row"):
            rename_in       = gr.Textbox(
                placeholder="Rename selected user to…", show_label=False,
                scale=4, max_lines=1,
            )
            rename_user_btn = gr.Button("Rename",  variant="secondary", scale=1)
        user_status = gr.HTML("")

        # ── Toolbar row 1 ──────────────────────────────────────────────────
        with gr.Row(elem_id="toolbar"):
            sym_in  = gr.Textbox(placeholder="Enter symbol e.g. AAPL",
                                 show_label=False, scale=3, max_lines=1)
            add_btn = gr.Button("Add Symbol",    variant="primary",   scale=1, elem_id="add-btn")
            wf_btn  = gr.Button("Show Workflow", variant="secondary", scale=1)

        # ── Toolbar row 2 ──────────────────────────────────────────────────
        with gr.Row():
            ref_btn   = gr.Button("Analyze Stock",  variant="primary",   scale=2, elem_id="refresh-sel-btn")
            ref_all   = gr.Button("Analyze All",    variant="primary",   scale=2, elem_id="refresh-all-btn")
            price_btn = gr.Button("🔄 Live Prices", variant="secondary", scale=1, elem_id="price-refresh-btn")
            save_btn  = gr.Button("💾 Save Dashboard", variant="secondary", scale=1, elem_id="save-btn")
            ref_dd   = gr.Dropdown(choices=list(REFRESH_OPTIONS.keys()),
                                   value=saved_ref, label="Auto-Refresh", scale=1,
                                   elem_id="ar_dd", min_width=120)
            # Inside the row so it doesn't create a top-level bordered wrapper in Gradio 6.
            # Must be visible=True so Gradio 6 keeps it in the DOM for JS to read.
            ar_secs  = gr.Textbox(value=str(REFRESH_OPTIONS.get(saved_ref, 0)),
                                  elem_id="ar_secs_input", visible=True, show_label=False, scale=0)

        status_msg = gr.HTML("")
        wf_panel   = gr.HTML(value="", visible=False)
        wf_vis     = gr.State(False)
        wf_cache   = gr.State("")

        # syms_state: live list of active symbols (drives tab visibility)
        syms_state = gr.State(value=list(init_syms))

        # cur_sym: which symbol the shared panel shows (updated by tab.select)
        # Intentionally starts empty so demo.load can set it to syms[0],
        # which fires cur_sym.change and renders the first tab's panel on startup.
        cur_sym      = gr.State(value="")
        report_state = gr.State("")
        rep_reading  = gr.State(False)
        chat_reading = gr.State(False)
        ct_mode_state = gr.State(False)   # True = Capitol Trades mode, False = Stock mode


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
                tab_objs         = []  # gr.Tab objects
                del_tab_btns     = []  # per-tab delete buttons
                own_chk_list     = []  # per-tab "I OWN" checkboxes
                mv_left_btns     = []  # per-tab ◀ move-left buttons
                mv_right_btns    = []  # per-tab ▶ move-right buttons

                with gr.Tabs() as tabs_grp:
                    for i in range(MAX_SLOTS):
                        sym     = init_syms[i] if i < len(init_syms) else ""
                        visible = bool(sym)
                        label   = sym if sym else f"__slot{i}__"
                        with gr.Tab(label=label, visible=visible, id=i) as t:
                            with gr.Row():
                                own_chk = gr.Checkbox(
                                    label=f"I OWN {sym}" if sym else "I OWN",
                                    value=_owned_map.get(sym, False) if sym else False,
                                    scale=3,
                                )
                                mv_l_btn = gr.Button(
                                    "◀", size="sm", variant="secondary",
                                    min_width=0, scale=1,
                                )
                                mv_r_btn = gr.Button(
                                    "▶", size="sm", variant="secondary",
                                    min_width=0, scale=1,
                                )
                                dtab_btn = gr.Button(
                                    "Delete", size="sm", variant="stop",
                                    min_width=0, scale=1,
                                )
                            own_chk.change(
                                fn=lambda v, syms, idx=i: _owned_map.update(
                                    {list(syms)[idx]: v}
                                ) if idx < len(list(syms)) and list(syms)[idx] else None,
                                inputs=[own_chk, syms_state],
                            )
                        tab_objs.append(t)
                        del_tab_btns.append(dtab_btn)
                        own_chk_list.append(own_chk)
                        mv_left_btns.append(mv_l_btn)
                        mv_right_btns.append(mv_r_btn)


                # ── Shared analysis panel ──────────────────────────────────
                hero_out    = gr.HTML(_hero_placeholder())
                chart_out   = gr.HTML()
                signals_out = gr.HTML()
                levels_out  = gr.HTML()
                fund_out    = gr.HTML()
                sent_out    = gr.HTML()

                with gr.Accordion("AI Analysis Report", open=False) as report_acc:
                    report_out   = gr.Markdown("*Run analysis to see AI report.*")
                    rd_rep       = gr.Button("▶ READ", size="sm", variant="secondary", elem_id="rd-rep-btn")
                    rep_tts_text = gr.Textbox(value="", elem_id="rep_tts_buf", show_label=False, visible=True)

                with gr.Accordion("Ask a Question", open=False):
                    _chat_kwargs = {} if _GRADIO_MAJOR >= 6 else {"type": "messages"}
                    chatbot  = gr.Chatbot(height=650, show_label=False, elem_id="chatbot-box", **_chat_kwargs)
                    with gr.Row():
                        cpy_btn = gr.Button("📋 Copy Chat",  size="sm", scale=1, elem_id="cpy-btn")
                        clr_btn = gr.Button("🗑 Clear Chat", size="sm", scale=1, elem_id="clr-btn")
                    with gr.Row():
                        chat_in = gr.Textbox(
                            placeholder="Ask about this stock or Congress trades (e.g. 'Latest senator trades?')",
                            show_label=False, scale=5,
                        )
                        rd_chat = gr.Button("▶ READ", size="sm", scale=1, elem_id="rd-chat-btn")
                        snd_btn = gr.Button("Send",    variant="primary", size="sm", scale=1, elem_id="snd-btn")
                    cpy_out       = gr.HTML("")
                    copy_buf      = gr.Textbox(value="", elem_id="chat_copy_buf",  visible=True, show_label=False)
                    chat_tts_text = gr.Textbox(value="", elem_id="chat_tts_buf",   visible=True, show_label=False)
                    with gr.Row():
                        mode_btn = gr.Button(
                            "📈 Analyzed Stock Questions", size="sm", variant="secondary",
                            min_width=0, elem_id="mode-btn",
                        )
                    qbtns = []
                    for row_start in range(0, len(SAMPLE_QUESTIONS), 4):
                        with gr.Row():
                            for i, q in enumerate(SAMPLE_QUESTIONS[row_start:row_start+4], start=row_start):
                                b = gr.Button(q, size="sm", scale=1, min_width=0, elem_id=f"qbtn-{i}")
                                qbtns.append(b)
                    ct_qbtns = []
                    for row_start in range(0, len(CT_SAMPLE_QUESTIONS), 4):
                        with gr.Row():
                            for i, q in enumerate(CT_SAMPLE_QUESTIONS[row_start:row_start+4], start=row_start):
                                b = gr.Button(q, size="sm", scale=1, min_width=0, elem_id=f"ctbtn-{i}")
                                ct_qbtns.append(b)

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
                        *_tab_updates(syms), *_own_chk_updates(syms))
            if sym in syms:
                return ("", syms,
                        f'<div style="color:#facc15;font-size:12px">{sym} already open.</div>',
                        *_tab_updates(syms), *_own_chk_updates(syms))
            if len(syms) >= MAX_SLOTS:
                return ("", syms,
                        f'<div style="color:#ef4444;font-size:12px">Max {MAX_SLOTS} tabs reached.</div>',
                        *_tab_updates(syms), *_own_chk_updates(syms))
            syms.append(sym)
            _owned_map.setdefault(sym, False)
            save_session(syms, _owned_map, _watchlist, snapshots=_analysis_cache, username=_current_user)
            return ("", syms,
                    f'<div style="color:#22c55e;font-size:12px"><b>{sym}</b> added.</div>',
                    *_tab_updates(syms), *_own_chk_updates(syms))

        def _own_chk_updates(syms):
            """Return gr.update() for each I OWN checkbox based on current syms list."""
            updates = []
            for i in range(MAX_SLOTS):
                if i < len(syms) and syms[i]:
                    s = syms[i]
                    updates.append(gr.update(label=f"I OWN {s}", value=_owned_map.get(s, False)))
                else:
                    updates.append(gr.update(label="I OWN", value=False))
            return updates

        def _sync_tabs(syms):
            """Re-apply tab and checkbox updates from current syms_state."""
            return _tab_updates(list(syms)) + _own_chk_updates(list(syms))

        _ADD_OUTPUTS = [sym_in, syms_state, status_msg] + tab_objs + own_chk_list
        (add_btn.click(fn=do_add, inputs=[sym_in, syms_state], outputs=_ADD_OUTPUTS)
                .then(fn=_sync_tabs, inputs=[syms_state], outputs=tab_objs + own_chk_list)
                .then(fn=lambda sym, syms: _startup_prices(sym, syms), inputs=[cur_sym, syms_state], outputs=PANEL))
        (sym_in.submit(fn=do_add, inputs=[sym_in, syms_state], outputs=_ADD_OUTPUTS)
                .then(fn=_sync_tabs, inputs=[syms_state], outputs=tab_objs + own_chk_list)
                .then(fn=lambda sym, syms: _startup_prices(sym, syms), inputs=[cur_sym, syms_state], outputs=PANEL))

        # ── Per-tab Delete buttons (reliable: no cur_sym race condition) ───
        def do_delete_slot(syms, idx, cur):
            syms = list(syms)
            if idx >= len(syms) or not syms[idx]:
                return (syms, cur,
                        '<div style="color:#ef4444;font-size:12px">No stock at this slot.</div>',
                        *_tab_updates(syms), *_own_chk_updates(syms))
            sym = syms[idx]
            syms.pop(idx)
            _owned_map.pop(sym, None)
            _analysis_cache.pop(sym, None)
            save_session(syms, _owned_map, _watchlist, snapshots=_analysis_cache, username=_current_user)
            # Pick a new active sym if the deleted one was active
            new_cur = cur
            if cur == sym:
                new_cur = syms[0] if syms else ""
            return (syms, new_cur,
                    f'<div style="color:#22c55e;font-size:12px"><b>{sym}</b> deleted.</div>',
                    *_tab_updates(syms), *_own_chk_updates(syms))

        _EMPTY_PANEL = [
            _hero_placeholder(),
            "", "", "", "", "", "*Run analysis to see AI report.*", "",
        ]

        def _panel_after_delete(syms, cur):
            """After deletion, show the new active symbol's data or clear the panel."""
            cur = (cur or "").strip().upper()
            if not cur:
                return _EMPTY_PANEL
            data = _analysis_cache.get(cur)
            if data and data.get("indicators"):
                return list(_render_from_data(cur, data))
            return [
                _hero_placeholder(cur),
                _tv_chart(cur), "", "", "", "", "*Run analysis to see AI report.*", "",
            ]

        for i, btn in enumerate(del_tab_btns):
            (btn.click(
                fn=lambda syms, cur, idx=i: do_delete_slot(syms, idx, cur),
                inputs=[syms_state, cur_sym],
                outputs=[syms_state, cur_sym, status_msg] + tab_objs + own_chk_list,
            ).then(fn=_sync_tabs, inputs=[syms_state], outputs=tab_objs + own_chk_list)
             .then(fn=_panel_after_delete, inputs=[syms_state, cur_sym], outputs=PANEL))

        # ── Per-tab Move Left / Move Right buttons ─────────────────────────
        def do_move_slot(syms, idx, direction):
            syms = list(syms)
            n = len(syms)
            swap = idx + direction   # direction: -1 = left, +1 = right
            if idx < 0 or idx >= n or swap < 0 or swap >= n:
                return (syms, '', *_tab_updates(syms), *_own_chk_updates(syms))
            syms[idx], syms[swap] = syms[swap], syms[idx]
            save_session(syms, _owned_map, _watchlist, snapshots=_analysis_cache, username=_current_user)
            return (syms, '', *_tab_updates(syms), *_own_chk_updates(syms))

        _MOVE_OUTPUTS = [syms_state, status_msg] + tab_objs + own_chk_list
        for i, (lb, rb) in enumerate(zip(mv_left_btns, mv_right_btns)):
            (lb.click(
                fn=lambda syms, idx=i: do_move_slot(syms, idx, -1),
                inputs=[syms_state],
                outputs=_MOVE_OUTPUTS,
            ).then(fn=_sync_tabs, inputs=[syms_state], outputs=tab_objs + own_chk_list))
            (rb.click(
                fn=lambda syms, idx=i: do_move_slot(syms, idx, +1),
                inputs=[syms_state],
                outputs=_MOVE_OUTPUTS,
            ).then(fn=_sync_tabs, inputs=[syms_state], outputs=tab_objs + own_chk_list))

        # ── Save ───────────────────────────────────────────────────────────
        def _on_ref_dd_change(syms, rv):
            save_session(list(syms), _owned_map, _watchlist, rv, snapshots=_analysis_cache, username=_current_user)
            return str(REFRESH_OPTIONS.get(rv, 0))

        ref_dd.change(fn=_on_ref_dd_change, inputs=[syms_state, ref_dd], outputs=[ar_secs])

        # ── Analysis ───────────────────────────────────────────────────────
        def do_refresh(sym):
            sym = (sym or "").strip().upper()
            if not sym:
                return [gr.update()] * 8
            _analysis_cache.pop(sym, None)
            result = list(_run(sym))
            save_session(list(_analysis_cache.keys()), _owned_map, _watchlist, snapshots=_analysis_cache, username=_current_user)
            return result

        def do_price_refresh(cur_sym_val, syms):
            """Fetch fresh prices for ALL open tabs directly via yfinance — no HTTP."""
            import time as _time
            import yfinance as yf
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout
            from agents.technical_agent import _session_info as _si_fn

            syms = [s for s in list(syms) if s]
            cur  = (cur_sym_val or "").strip().upper()
            if not syms:
                return gr.update(), ""

            t       = _time.localtime()
            h12     = t.tm_hour % 12 or 12
            ampm    = "AM" if t.tm_hour < 12 else "PM"
            tz      = _time.strftime("%Z")
            now_str = f"{h12}:{t.tm_min:02d}:{t.tm_sec:02d} {ampm} {tz}"

            def _fetch_si(s):
                """Return a fresh session_info dict for symbol s."""
                try:
                    stock = yf.Ticker(s)
                    ext_last = ext_time = pre_last = reg_last = post_last = ovn_last = None
                    try:
                        df_1m = stock.history(period="1d", interval="1m", prepost=True)
                        if df_1m is not None and not df_1m.empty:
                            ext_last = float(df_1m["Close"].iloc[-1])
                            ts = df_1m.index[-1]
                            h, m = ts.hour, ts.minute
                            ap = "AM" if h < 12 else "PM"
                            ext_time = f"{h % 12 or 12}:{m:02d} {ap} ET"
                            # Walk every bar — assign each to its session window (ET hours)
                            for bar_ts in df_1m.index:
                                bh, bm = bar_ts.hour, bar_ts.minute
                                price  = float(df_1m.loc[bar_ts, "Close"])
                                if bh < 9 or (bh == 9 and bm < 30):
                                    pre_last  = price   # Pre-market:  4:00–9:30 AM ET
                                elif (bh == 9 and bm >= 30) or (10 <= bh < 16):
                                    reg_last  = price   # Regular:     9:30 AM–4:00 PM ET
                                elif 16 <= bh < 20:
                                    post_last = price   # After-hours: 4:00–8:00 PM ET
                                elif bh >= 20:
                                    ovn_last  = price   # Overnight:   8:00 PM+ ET
                    except Exception:
                        pass
                    df_1d = None
                    try:
                        df_1d = stock.history(period="2d", interval="1d", auto_adjust=True)
                        if df_1d is not None and not df_1d.empty:
                            df_1d.index = df_1d.index.tz_localize(None)
                        else:
                            df_1d = None
                    except Exception:
                        pass
                    # "_reg_last_price" always present — signals price-refresh context
                    # Each session's price is None until that session's bars appear → pills show "—"
                    info = {"_reg_last_price": reg_last}
                    if pre_last  is not None: info["_pre_last_price"]  = pre_last
                    if post_last is not None: info["_post_last_price"] = post_last
                    if ovn_last  is not None: info["_ovn_last_price"]  = ovn_last
                    if ext_last  is not None: info["_ext_last_price"]  = ext_last
                    if ext_time  is not None: info["_ext_last_time"]   = ext_time
                    si = _si_fn(info, df_1d)
                    si["_refreshed_at"] = now_str   # persist through tab switches
                    return si
                except Exception as e:
                    logger.error(f"do_price_refresh [{s}]: {e}")
                    return None

            # Refresh every open tab in parallel with a per-symbol timeout
            _TIMEOUT = 20  # seconds per symbol
            refreshed, failed = 0, 0
            with ThreadPoolExecutor(max_workers=len(syms)) as pool:
                futures = {pool.submit(_fetch_si, s): s for s in syms}
                for fut, s in futures.items():
                    try:
                        si = fut.result(timeout=_TIMEOUT)
                        if si:
                            data = _analysis_cache.get(s, {})
                            data["session_info"] = si
                            _analysis_cache[s]   = data
                            refreshed += 1
                        else:
                            failed += 1
                    except _FuturesTimeout:
                        logger.warning(f"do_price_refresh [{s}]: timed out after {_TIMEOUT}s")
                        failed += 1
                    except Exception as e:
                        logger.error(f"do_price_refresh [{s}]: {e}")
                        failed += 1

            # Render hero for the currently visible tab
            data  = _analysis_cache.get(cur, {})
            si    = data.get("session_info", {}) if data else {}
            ind   = data.get("indicators", {})
            dcf   = data.get("dcf", {})
            dec   = data.get("decision", {})
            owns  = _owned_map.get(cur, False)
            hero  = _hero_html(cur, dec.get("action", "N/A"), ind.get("price", 0), dcf, owns, si)

            parts = [f"{refreshed} symbol{'s' if refreshed != 1 else ''} refreshed"]
            if failed:
                parts.append(f"{failed} failed")
            msg = (
                f'<div style="color:#22c55e;font-size:12px">'
                f'&#128259; {" | ".join(parts)} at {now_str}</div>'
            )
            return hero, msg

        def do_refresh_all(syms, active_sym):
            """Clear all caches, re-analyze all symbols."""
            syms = [s for s in list(syms) if s]
            if not syms:
                return [gr.update()] * 8
            for s in syms:
                _analysis_cache.pop(s, None)
                _run(s)
            save_session(syms, _owned_map, _watchlist, snapshots=_analysis_cache, username=_current_user)
            show = (active_sym or "").strip().upper()
            if show not in syms:
                show = syms[0]
            data = _analysis_cache.get(show)
            if not data:
                return [gr.update()] * 8
            return list(_render_from_data(show, data))

        def on_sym_change(sym):
            sym = (sym or "").strip().upper()
            # logger.debug(f"[on_sym_change] sym={sym!r} cache_keys={list(_analysis_cache.keys())}")
            if not sym:
                return [
                    _hero_placeholder(),
                    "", "", "", "", "",
                    "*Run analysis to see AI report.*", "",
                ]
            data = _analysis_cache.get(sym)
            if not data or not data.get("indicators"):
                # Price-only cache (from Live Prices) — show hero + chart but no analysis panels
                si   = data.get("session_info", {}) if data else {}
                owns = _owned_map.get(sym, False)
                hero = _hero_html(sym, "N/A", si.get("_reg_last_price") or 0, {}, owns, si) if si else _hero_placeholder(sym)
                return [
                    hero,
                    _tv_chart(sym, si), "", "", "", "",
                    "*Run analysis to see AI report.*", "",
                ]
            return list(_render_from_data(sym, data))

        def _clear_chat(sym):
            sym = (sym or "").strip().upper()
            _chat_history.pop(sym, None)
            return []

        def _clear_chat_all(syms):
            for s in list(syms):
                _chat_history.pop(s, None)
            return []

        # Single tabs_grp.select fires exactly once per click (avoids per-tab
        # race condition where all 12 t.select events fire simultaneously).
        def on_tab_click(syms, evt: gr.SelectData):
            idx  = evt.index
            syms = list(syms)
            return syms[idx] if idx < len(syms) else ""

        (tabs_grp.select(fn=on_tab_click, inputs=[syms_state], outputs=[cur_sym])
                 .then(fn=on_sym_change, inputs=[cur_sym], outputs=PANEL)
                 .then(fn=_clear_chat,   inputs=[cur_sym], outputs=[chatbot]))

        (ref_btn.click(fn=do_refresh, inputs=[cur_sym], outputs=PANEL)
                .then(fn=_clear_chat, inputs=[cur_sym], outputs=[chatbot]))
        (ref_all.click(fn=do_refresh_all, inputs=[syms_state, cur_sym], outputs=PANEL)
                .then(fn=_clear_chat_all, inputs=[syms_state], outputs=[chatbot]))
        price_btn.click(fn=do_price_refresh, inputs=[cur_sym, syms_state], outputs=[hero_out, status_msg])

        def do_save(syms, ref):
            try:
                syms = list(syms) if syms else []
                ok, err = save_session(syms, _owned_map, _watchlist, ref, snapshots=_analysis_cache, username=_current_user)
                if ok:
                    return '<div style="color:#22c55e;font-size:12px">&#10003; Dashboard saved.</div>'
                return f'<div style="color:#ef4444;font-size:12px">Save failed: {err}</div>'
            except Exception as e:
                logger.error(f"do_save error: {e}")
                return f'<div style="color:#ef4444;font-size:12px">Save error: {e}</div>'

        save_btn.click(fn=do_save, inputs=[syms_state, ref_dd], outputs=[status_msg])

        # ── User Management callbacks ───────────────────────────────────────
        _USER_LOAD_OUTPUTS = (
            [tabs_grp, user_status, syms_state, ref_dd, ar_secs, cur_sym, wl_radio]
            + tab_objs + own_chk_list
            + PANEL
            + [add_btn, sym_in]
            + del_tab_btns + mv_left_btns + mv_right_btns
        )

        def do_load_user(username):
            global _current_user
            if not username:
                yield (
                    [gr.update(), '<div style="color:#facc15;font-size:12px">Select a user first.</div>']
                    + [gr.update()] * (len(_USER_LOAD_OUTPUTS) - 2)
                )
                return
            _current_user = username
            _owned_map.clear()
            _analysis_cache.clear()
            _chat_history.clear()
            _chatbot_ctx.clear()
            _watchlist.clear()

            sess = load_session(username)
            syms = list(sess.get("symbols", []))
            _owned_map.update(sess.get("owned", {}))
            _watchlist.extend(sess.get("watchlist") or list(DEFAULT_WATCHLIST))
            ref       = sess.get("refresh_interval", "Off")
            secs      = str(REFRESH_OPTIONS.get(ref, 0))

            # Default User is always read-only with a single fixed stock
            if username == DEFAULT_USER:
                syms = ["SPY"]
                _owned_map.clear()
                _owned_map["SPY"] = False

            first_sym = syms[0] if syms else ""
            wl_update = gr.update(choices=list(_watchlist), value=None)

            # Visibility of edit controls (hidden for Default User)
            _editable = username != DEFAULT_USER
            _ev = gr.update(visible=_editable)
            _edit_updates = [_ev, _ev] + [_ev] * (MAX_SLOTS * 3)

            # Yield 1: instant — tabs appear with loading placeholder
            loading_status = f'<div style="color:#38bdf8;font-size:12px">&#8987; Loading {username}…</div>'
            loading_panel  = [_hero_placeholder(first_sym), "", "", "", "", "", "*Fetching prices…*", ""]
            yield (
                [gr.update(selected=0), loading_status, syms, ref, secs, first_sym, wl_update]
                + list(_tab_updates(syms))
                + list(_own_chk_updates(syms))
                + loading_panel
                + _edit_updates
            )

            # Yield 2: after prices fetched — show real hero + chart
            # Safe to call _startup_prices here because cur_sym.change → on_sym_change
            # is no longer wired, so there is no concurrent PANEL update race.
            panel_data  = _startup_prices(first_sym, syms)
            done_status = f'<div style="color:#22c55e;font-size:12px">&#10003; User <b>{username}</b> loaded.</div>'
            yield (
                [gr.update(selected=0), done_status, syms, ref, secs, first_sym, wl_update]
                + list(_tab_updates(syms))
                + list(_own_chk_updates(syms))
                + panel_data
                + _edit_updates
            )

        def do_create_user(new_name):
            name = new_name.strip()
            ok, err = create_user(name)
            if not ok:
                return new_name, gr.update(), f'<div style="color:#ef4444;font-size:12px">{err}</div>'
            users = list_users()
            return "", gr.update(choices=users, value=name), f'<div style="color:#22c55e;font-size:12px">User <b>{name}</b> created — click Load.</div>'

        def do_delete_user(username):
            global _current_user
            if not username:
                return gr.update(), '<div style="color:#facc15;font-size:12px">Select a user first.</div>'
            ok, err = delete_user(username)
            if not ok:
                return gr.update(), f'<div style="color:#ef4444;font-size:12px">{err}</div>'
            if _current_user == username:
                _current_user = ""
                _owned_map.clear()
                _analysis_cache.clear()
                _watchlist.clear()
                _watchlist.extend(list(DEFAULT_WATCHLIST))
            users = list_users()
            return gr.update(choices=users, value=None), f'<div style="color:#22c55e;font-size:12px">User <b>{username}</b> deleted.</div>'

        def do_rename_user(username, new_name):
            global _current_user
            if not username:
                return gr.update(), new_name, '<div style="color:#facc15;font-size:12px">Select a user first.</div>'
            ok, err = rename_user(username, new_name)
            if not ok:
                return gr.update(), new_name, f'<div style="color:#ef4444;font-size:12px">{err}</div>'
            if _current_user == username:
                _current_user = new_name.strip()
            users = list_users()
            return gr.update(choices=users, value=new_name.strip()), "", f'<div style="color:#22c55e;font-size:12px">Renamed <b>{username}</b> → <b>{new_name.strip()}</b>.</div>'

        user_dd.select(fn=do_load_user, inputs=[user_dd], outputs=_USER_LOAD_OUTPUTS)

        create_user_btn.click(
            fn=do_create_user,
            inputs=[new_user_in],
            outputs=[new_user_in, user_dd, user_status],
        )

        delete_user_btn.click(
            fn=do_delete_user,
            inputs=[user_dd],
            outputs=[user_dd, user_status],
        )

        rename_user_btn.click(
            fn=do_rename_user,
            inputs=[user_dd, rename_in],
            outputs=[user_dd, rename_in, user_status],
        )
        rename_in.submit(
            fn=do_rename_user,
            inputs=[user_dd, rename_in],
            outputs=[user_dd, rename_in, user_status],
        )


        # ── Auto-Refresh (JS polling loop injected at page load) ───────────
        _JS_AUTO_REFRESH = """() => {
            var lastRefresh      = Date.now();
            var lastPriceRefresh = Date.now();
            function getSecs() {
                var el = document.querySelector('#ar_secs_input textarea');
                if (!el) el = document.querySelector('#ar_secs_input input');
                return el ? (parseFloat(el.value) || 0) : 0;
            }
            function clickRefreshAll() {
                var btn = document.querySelector('#refresh-all-btn button');
                if (!btn) btn = document.querySelector('#refresh-all-btn');
                if (btn) btn.click();
            }
            function clickPriceRefresh() {
                var btn = document.querySelector('#price-refresh-btn button');
                if (!btn) btn = document.querySelector('#price-refresh-btn');
                if (btn) btn.click();
            }
            function tick() {
                var secs = getSecs();
                if (secs > 0 && (Date.now() - lastRefresh) >= secs * 1000) {
                    lastRefresh = Date.now();
                    clickRefreshAll();
                }
                if ((Date.now() - lastPriceRefresh) >= 120000) {
                    lastPriceRefresh = Date.now();
                    clickPriceRefresh();
                }
            }
            setTimeout(function() { setInterval(tick, 2000); }, 3000);
        }"""
        # ── JS: full-width fix + persistent button styling ──
        # KEY BUGS FIXED vs prior version:
        #   1. MutationObserver was watching the <button> element directly — but Gradio 6
        #      *replaces* the entire <button> node on each update, so the observer was
        #      watching a detached dead node.  Now we watch the wrapper div (elem_id holder).
        #   2. Buttons inside closed accordions (open=False) are not in the DOM at page-load,
        #      so one-shot init() found nothing.  Now a persistent 500 ms poll + debounced
        #      body MutationObserver handles late-rendered elements.
        _JS_FIX_HEIGHT = """() => {
            /* ── colour palette ── */
            var BLUE    = 'linear-gradient(135deg,#1d4ed8,#3b82f6)';
            var BLUE_H  = 'linear-gradient(135deg,#2563eb,#60a5fa)';
            var BLUE_B  = '2px solid #60a5fa';
            var ORA     = 'linear-gradient(135deg,#c2410c,#ea580c)';
            var ORA_H   = 'linear-gradient(135deg,#9a3412,#f97316)';
            var ORA_B   = '2px solid #fb923c';
            var GREEN   = 'linear-gradient(135deg,#065f46,#10b981)';
            var GREEN_H = 'linear-gradient(135deg,#047857,#34d399)';
            var TEAL    = 'linear-gradient(135deg,#0e7490,#06b6d4)';
            var TEAL_H  = 'linear-gradient(135deg,#0891b2,#22d3ee)';
            var RED     = 'linear-gradient(135deg,#991b1b,#ef4444)';
            var RED_H   = 'linear-gradient(135deg,#b91c1c,#f87171)';
            var IND     = 'linear-gradient(135deg,#4338ca,#818cf8)';
            var IND_H   = 'linear-gradient(135deg,#6366f1,#a5b4fc)';
            var AMB     = 'linear-gradient(135deg,#b45309,#f59e0b)';
            var AMB_H   = 'linear-gradient(135deg,#d97706,#fbbf24)';

            /* ── helpers ── */
            function btnOf(id) {
                var w = document.getElementById(id);
                if (!w) return null;
                return w.tagName === 'BUTTON' ? w : w.querySelector('button');
            }
            function s(btn, bg, border, hBg) {
                if (!btn) return;
                btn.style.background   = bg;
                btn.style.border       = border;
                btn.style.color        = '#fff';
                btn.style.fontWeight   = '700';
                btn.style.borderRadius = '6px';
                btn.onmouseenter = function() { this.style.background = hBg; };
                btn.onmouseleave = function() { this.style.background = bg;  };
            }

            /* ── apply all button colours based on current DOM text ── */
            function applyAll() {
                /* READ buttons: blue while idle, orange while STOP */
                ['rd-rep-btn','rd-chat-btn'].forEach(function(id) {
                    var b = btnOf(id);
                    if (!b) return;
                    var stop = b.textContent.indexOf('STOP') !== -1 ||
                               b.textContent.indexOf('\u23f9') !== -1;
                    s(b, stop ? ORA : BLUE, stop ? ORA_B : BLUE_B, stop ? ORA_H : BLUE_H);
                });
                /* Mode button: blue = Stock, orange = CT Mode ON */
                var mb = btnOf('mode-btn');
                if (mb) {
                    var ct = mb.textContent.indexOf('Politicians') !== -1;
                    s(mb, ct ? AMB : IND, ct ? '1px solid #fbbf24' : '1px solid #6366f1', ct ? AMB_H : IND_H);
                }
                /* Live Prices */
                s(btnOf('price-refresh-btn'), TEAL, '2px solid #22d3ee', TEAL_H);
                /* Send / Copy / Clear */
                s(btnOf('snd-btn'), GREEN, '2px solid #34d399', GREEN_H);
                s(btnOf('cpy-btn'), TEAL,  'none',              TEAL_H);
                s(btnOf('clr-btn'), RED,   'none',              RED_H);
                /* Stock sample-question buttons — indigo */
                for (var si = 0; si < 20; si++) {
                    s(btnOf('qbtn-' + si), IND, '1px solid #6366f1', IND_H);
                }
                /* Capitol Trades question buttons — amber */
                for (var ci = 0; ci < 20; ci++) {
                    s(btnOf('ctbtn-' + ci), AMB, '1px solid #fbbf24', AMB_H);
                }
            }

            /* ── full-width layout fix ── */
            function applyWidth() {
                ['gradio-app','.gradio-container','.app','.main','body','html']
                    .forEach(function(sel) {
                        var el = document.querySelector(sel);
                        if (!el) return;
                        el.style.minHeight = '100vh';
                        el.style.width     = '100%';
                        el.style.maxWidth  = '100%';
                        el.style.boxSizing = 'border-box';
                    });
            }

            /* ── debounced body observer — catches accordion open + Gradio re-renders ──
               Observing document.body is intentionally broad: the debounce (50 ms) keeps
               it cheap; it fires applyAll at most ~20×/sec regardless of mutation rate.  */
            var _dTimer = null;
            var bodyObs = new MutationObserver(function() {
                if (_dTimer) clearTimeout(_dTimer);
                _dTimer = setTimeout(applyAll, 10);
            });
            bodyObs.observe(document.body, {childList: true, subtree: true, characterData: true});

            /* ── persistent poll: keeps re-styling for 30 s after load ──
               Handles accordions that open after the initial body-observer window.  */
            var _pollN = 0;
            var _pollId = setInterval(function() {
                applyAll();
                if (++_pollN >= 60) clearInterval(_pollId);
            }, 500);

            /* ── kick off ── */
            applyWidth();
            applyAll();
            setTimeout(applyAll, 300);
            setTimeout(applyAll, 800);
        }"""
        demo.load(fn=None, js=_JS_FIX_HEIGHT)

        demo.load(fn=None, js=_JS_AUTO_REFRESH)

        # Unlock Web Speech API for mobile (iOS Safari blocks async speechSynthesis
        # unless speak() is called at least once within a direct user gesture first)
        _JS_UNLOCK_TTS = """() => {
            function unlock() {
                var u = new SpeechSynthesisUtterance('');
                window.speechSynthesis.speak(u);
                window.speechSynthesis.cancel();
            }
            document.addEventListener('click',    unlock, {once: true});
            document.addEventListener('touchend', unlock, {once: true});
        }"""
        demo.load(fn=None, js=_JS_UNLOCK_TTS)

        # ── Blank state on every page load — user must be selected first ──
        def _on_page_load():
            """Runs on every page load: just pre-select Default User in the dropdown.
            The actual session restore is done by the chained .then(do_load_user) below."""
            return gr.update(choices=list_users(), value=DEFAULT_USER)

        (demo.load(fn=_on_page_load, outputs=[user_dd])
             .then(fn=do_load_user, inputs=[user_dd], outputs=_USER_LOAD_OUTPUTS))

        def _startup_prices(cur_sym_val, syms):
            """Fetch live prices for all tabs at startup; render hero + chart for active tab."""
            import time as _time
            import yfinance as yf
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout
            from agents.technical_agent import _session_info as _si_fn

            syms = [s for s in list(syms) if s]
            cur  = (cur_sym_val or "").strip().upper()
            if not syms:
                return [_hero_placeholder(), "", "", "", "", "", "*Run analysis to see AI report.*", ""]

            t       = _time.localtime()
            h12     = t.tm_hour % 12 or 12
            ampm    = "AM" if t.tm_hour < 12 else "PM"
            tz      = _time.strftime("%Z")
            now_str = f"{h12}:{t.tm_min:02d}:{t.tm_sec:02d} {ampm} {tz}"

            def _fetch_one(s):
                stock = yf.Ticker(s)
                ext_last = ext_time = pre_last = reg_last = post_last = ovn_last = None
                try:
                    df_1m = stock.history(period="1d", interval="1m", prepost=True)
                    if df_1m is not None and not df_1m.empty:
                        ext_last = float(df_1m["Close"].iloc[-1])
                        ts = df_1m.index[-1]
                        h, m = ts.hour, ts.minute
                        ap = "AM" if h < 12 else "PM"
                        ext_time = f"{h % 12 or 12}:{m:02d} {ap} ET"
                        for bar_ts in df_1m.index:
                            bh, bm = bar_ts.hour, bar_ts.minute
                            price  = float(df_1m.loc[bar_ts, "Close"])
                            if bh < 9 or (bh == 9 and bm < 30):
                                pre_last  = price
                            elif (bh == 9 and bm >= 30) or (10 <= bh < 16):
                                reg_last  = price
                            elif 16 <= bh < 20:
                                post_last = price
                            elif bh >= 20:
                                ovn_last  = price
                except Exception:
                    pass
                df_1d = None
                try:
                    df_1d = stock.history(period="2d", interval="1d", auto_adjust=True)
                    if df_1d is not None and not df_1d.empty:
                        df_1d.index = df_1d.index.tz_localize(None)
                    else:
                        df_1d = None
                except Exception:
                    pass
                info = {"_reg_last_price": reg_last}
                if pre_last  is not None: info["_pre_last_price"]  = pre_last
                if post_last is not None: info["_post_last_price"] = post_last
                if ovn_last  is not None: info["_ovn_last_price"]  = ovn_last
                if ext_last  is not None: info["_ext_last_price"]  = ext_last
                if ext_time  is not None: info["_ext_last_time"]   = ext_time
                si = _si_fn(info, df_1d)
                si["_refreshed_at"] = now_str
                return s, si

            # Fetch all symbols in parallel with a per-call timeout so a
            # hung yfinance request (HF rate-limit / cold-start) doesn't stall startup.
            _TIMEOUT = 20  # seconds per symbol
            with ThreadPoolExecutor(max_workers=len(syms)) as pool:
                futures = {pool.submit(_fetch_one, s): s for s in syms}
                for fut, s in futures.items():
                    try:
                        sym, si = fut.result(timeout=_TIMEOUT)
                        _analysis_cache.setdefault(sym, {})["session_info"] = si
                    except _FuturesTimeout:
                        logger.warning(f"_startup_prices [{s}]: timed out after {_TIMEOUT}s")
                    except Exception as e:
                        logger.error(f"_startup_prices [{s}]: {e}")

            if not cur or cur not in syms:
                cur = syms[0]
            data = _analysis_cache.get(cur, {})
            si   = data.get("session_info", {})
            owns = _owned_map.get(cur, False)
            hero = _hero_html(cur, "N/A", si.get("_reg_last_price") or 0, {}, owns, si) if si else _hero_placeholder(cur)
            return [
                hero,
                _tv_chart(cur, si), "", "", "", "",
                "*Run analysis to see AI report.*", "",
            ]

        # ── Watchlist ──────────────────────────────────────────────────────
        def do_wl_add(sym):
            sym = sym.strip().upper()
            if sym and sym not in _watchlist:
                _watchlist.append(sym)
                save_session([], _owned_map, _watchlist, snapshots=_analysis_cache, username=_current_user)
            return gr.update(choices=list(_watchlist), value=None), ""

        wl_add.click(fn=do_wl_add, inputs=[wl_in], outputs=[wl_radio, wl_in])
        wl_in.submit(fn=do_wl_add, inputs=[wl_in], outputs=[wl_radio, wl_in])

        def do_wl_delete(sym, syms):
            sym = (sym or "").strip().upper()
            syms = list(syms)
            if sym and sym in _watchlist:
                _watchlist.remove(sym)
                save_session(syms, _owned_map, _watchlist, snapshots=_analysis_cache, username=_current_user)
            return gr.update(choices=list(_watchlist), value=None)

        wl_del.click(fn=do_wl_delete, inputs=[wl_radio, syms_state], outputs=[wl_radio])

        def do_wl_select(sym, syms):
            """Add selected watchlist stock as a tab."""
            if not sym:
                return (list(syms), gr.update(), *_tab_updates(list(syms)), *_own_chk_updates(list(syms)))
            syms = list(syms)
            if sym in syms:
                return (syms,
                        f'<div style="color:#facc15;font-size:12px">{sym} already open.</div>',
                        *_tab_updates(syms), *_own_chk_updates(syms))
            if len(syms) >= MAX_SLOTS:
                return (syms,
                        f'<div style="color:#ef4444;font-size:12px">Max {MAX_SLOTS} tabs reached.</div>',
                        *_tab_updates(syms), *_own_chk_updates(syms))
            syms.append(sym)
            _owned_map.setdefault(sym, False)
            save_session(syms, _owned_map, _watchlist, snapshots=_analysis_cache, username=_current_user)
            return (syms,
                    f'<div style="color:#22c55e;font-size:12px"><b>{sym}</b> added.</div>',
                    *_tab_updates(syms), *_own_chk_updates(syms))

        (wl_radio.change(fn=do_wl_select, inputs=[wl_radio, syms_state],
                         outputs=[syms_state, status_msg] + tab_objs + own_chk_list)
                 .then(fn=_sync_tabs, inputs=[syms_state], outputs=tab_objs + own_chk_list)
                 .then(fn=lambda sym, syms: _startup_prices(sym, syms), inputs=[cur_sym, syms_state], outputs=PANEL))

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
                return "", False, gr.update(value="▶ READ", variant="secondary")
            ticker = (sym or "").strip().upper()
            summary = _analysis_cache.get(ticker, {}).get("llm_summary", "")
            text = f"{ticker} AI Analysis. {summary}" if summary else ""
            if not text:
                return "", False, gr.update(value="▶ READ", variant="secondary")
            return text, True, gr.update(value="⏹ STOP", variant="stop")

        (rd_rep.click(fn=toggle_rep_tts, inputs=[cur_sym, rep_reading],
                      outputs=[rep_tts_text, rep_reading, rd_rep])
               .then(fn=None, js=_JS_SPEAK_REP))

        # ── Chat ───────────────────────────────────────────────────────────
        async def do_chat(question, history, ticker, ct_mode=False):
            """Async generator — yields a thinking indicator first, then the real answer.
            This keeps the Gradio SSE stream alive and prevents BodyStreamBuffer abort."""
            q = (question or "").strip()
            if not q:
                yield history or [], ""
                return
            base_h = list(history or [])
            # Immediately push a thinking indicator so SSE stays alive
            yield base_h + [
                {"role": "user",      "content": q},
                {"role": "assistant", "content": "_Thinking…_"},
            ], ""
            # Route to Capitol Trades if mode is forced on OR keyword matches
            if ct_mode or _is_ct_question(q):
                answer  = await asyncio.to_thread(_ct_chat_api, q)
                # Strip any leading [Capitol Trades] / Capitol Trades: the LLM may echo
                answer = re.sub(
                    r'^\**\[?Capitol\s+Trades\]?\**[\s:,\-–—]*',
                    '', answer, flags=re.IGNORECASE,
                ).strip()
                labeled = f"**[Capitol Trades]:**\n{answer}"
                new_h   = base_h + [
                    {"role": "user",      "content": q},
                    {"role": "assistant", "content": labeled},
                ]
                _chat_history.setdefault("__ct__", [])
                _chat_history["__ct__"].append([q, labeled])
                if len(_chat_history["__ct__"]) > MAX_CHATBOT_MEMORY:
                    _chat_history["__ct__"] = _chat_history["__ct__"][-MAX_CHATBOT_MEMORY:]
                yield new_h, ""
                return
            # Per-stock routing
            if not ticker:
                yield history or [], ""
                return
            answer  = await asyncio.to_thread(_chat_api, ticker, q)
            # Strip leading "TICKER: " / "TICKER — " the LLM sometimes adds to avoid duplication
            answer = re.sub(
                rf'^\**\[?{re.escape(ticker.strip().upper())}\]?\**[\s:,\-–—]+',
                '', answer, flags=re.IGNORECASE,
            ).strip()
            labeled = f"**[{ticker.strip().upper()}]** {answer}"
            new_h   = base_h + [
                {"role": "user",      "content": q},
                {"role": "assistant", "content": labeled},
            ]
            _chat_history.setdefault(ticker, [])
            _chat_history[ticker].append([q, labeled])
            if len(_chat_history[ticker]) > MAX_CHATBOT_MEMORY:
                _chat_history[ticker] = _chat_history[ticker][-MAX_CHATBOT_MEMORY:]
            yield new_h, ""

        _JS_SCROLL_CHAT = """() => {
            function scrollChatToBottom() {
                // Locate chatbot by elem_id first, then fall back to class/role
                var host = document.getElementById('chatbot-box')
                         || document.querySelector('[id*="chatbot"]')
                         || document.querySelector('[class*="chatbot"]');
                if (!host) return;
                // Walk every descendant — scroll whichever element actually overflows
                var nodes = host.querySelectorAll('*');
                for (var i = 0; i < nodes.length; i++) {
                    var el = nodes[i];
                    if (el.scrollHeight > el.clientHeight + 4) {
                        el.scrollTop = el.scrollHeight;
                    }
                }
                // Also attempt on the host root itself
                host.scrollTop = host.scrollHeight;
            }
            // Fire at 150 ms (after Gradio DOM update) and again at 700 ms
            // (safety net for slow network / large responses)
            setTimeout(scrollChatToBottom, 150);
            setTimeout(scrollChatToBottom, 700);
        }"""

        (snd_btn.click(fn=do_chat, inputs=[chat_in, chatbot, cur_sym, ct_mode_state],
                       outputs=[chatbot, chat_in])
                .then(fn=None, js=_JS_SCROLL_CHAT))
        (chat_in.submit(fn=do_chat, inputs=[chat_in, chatbot, cur_sym, ct_mode_state],
                        outputs=[chatbot, chat_in])
                .then(fn=None, js=_JS_SCROLL_CHAT))

        # Stock sample questions — auto-submit on click
        for q, btn in zip(SAMPLE_QUESTIONS, qbtns):
            async def _qfn(history, ticker, ct_mode, question=q):
                async for state in do_chat(question, history, ticker, ct_mode):
                    yield state
            (btn.click(fn=_qfn, inputs=[chatbot, cur_sym, ct_mode_state], outputs=[chatbot, chat_in])
                .then(fn=None, js=_JS_SCROLL_CHAT))

        # Capitol Trades quick questions — auto-submit on click
        for q, btn in zip(CT_SAMPLE_QUESTIONS, ct_qbtns):
            async def _ct_qfn(history, ticker, ct_mode, question=q):
                async for state in do_chat(question, history, ticker, ct_mode):
                    yield state
            (btn.click(fn=_ct_qfn, inputs=[chatbot, cur_sym, ct_mode_state], outputs=[chatbot, chat_in])
                .then(fn=None, js=_JS_SCROLL_CHAT))

        # Mode toggle
        def toggle_ct_mode(current):
            new = not current
            if new:
                return new, gr.update(value="🏛 Politicians Trade Questions", variant="primary")
            return new, gr.update(value="📈 Analyzed Stock Questions", variant="secondary")

        mode_btn.click(fn=toggle_ct_mode, inputs=[ct_mode_state],
                       outputs=[ct_mode_state, mode_btn])

        # Chat READ (toggle)
        def toggle_chat_tts(history, is_reading):
            if is_reading:
                return "", False, gr.update(value="▶ READ", variant="secondary")
            text = ""
            for msg in reversed(history or []):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    text = _extract_msg_text(msg.get("content", ""))
                    break
                elif isinstance(msg, (list, tuple)) and len(msg) > 1:
                    text = _extract_msg_text(msg[1] or "")
                    break
            if not text:
                return "", False, gr.update(value="▶ READ", variant="secondary")
            return text, True, gr.update(value="⏹ STOP", variant="stop")

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
/* ── Global container & body ───────────────────────────────────────────────── */
body,
.gradio-container,
div.gradio-container {
    background: #0a0f1e !important;
    font-family: 'Segoe UI', system-ui, sans-serif !important;
    max-width: 1440px !important;
    margin: 0 auto !important;
    min-height: 100vh !important;
}

/* ── Prevent layout collapse before first analysis ─────────────────────────── */
#main_row,
div#main_row {
    min-height: 720px !important;
    align-items: stretch !important;
}
#main_col,
div#main_col {
    min-height: 680px !important;
}

/* ── Mode toggle button ─────────────────────────────────────────────────────── */
/* Stock Mode (default / secondary) — blue */
#mode-btn button,
div#mode-btn button {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6) !important;
    border: 2px solid #60a5fa !important;
    color: #fff !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    min-width: 150px !important;
    transition: background .2s, border-color .2s !important;
}
#mode-btn button:hover,
div#mode-btn button:hover {
    background: linear-gradient(135deg, #2563eb, #60a5fa) !important;
}
/* CT Mode ON — Gradio sets variant="primary", adds .primary or [data-variant~=primary] */
#mode-btn button.primary,
#mode-btn button[class*="primary"],
#mode-btn button[data-variant="primary"],
div#mode-btn button.primary,
div#mode-btn button[class*="primary"],
div#mode-btn button[data-variant="primary"] {
    background: linear-gradient(135deg, #c2410c, #ea580c) !important;
    border: 2px solid #fb923c !important;
    color: #fff !important;
}
#mode-btn button.primary:hover,
div#mode-btn button.primary:hover {
    background: linear-gradient(135deg, #9a3412, #f97316) !important;
}

/* ── READ / STOP buttons ────────────────────────────────────────────────────── */
/* Default (▶ READ) — blue */
#rd-rep-btn button,
div#rd-rep-btn button,
#rd-chat-btn button,
div#rd-chat-btn button {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6) !important;
    border: 1px solid #60a5fa !important;
    color: #fff !important;
    font-weight: 600 !important;
}
#rd-rep-btn button:hover,
div#rd-rep-btn button:hover,
#rd-chat-btn button:hover,
div#rd-chat-btn button:hover {
    background: linear-gradient(135deg, #2563eb, #60a5fa) !important;
}
/* STOP state — variant="stop" adds .stop class — orange */
#rd-rep-btn button.stop,
#rd-rep-btn button[class*="stop"],
div#rd-rep-btn button.stop,
#rd-chat-btn button.stop,
#rd-chat-btn button[class*="stop"],
div#rd-chat-btn button.stop {
    background: linear-gradient(135deg, #c2410c, #ea580c) !important;
    border: 1px solid #fb923c !important;
    color: #fff !important;
}

/* ── Hide Gradio footer ─────────────────────────────────────────────────────── */
footer,
.footer,
div.footer,
gradio-app > footer,
.built-with {
    display: none !important;
    visibility: hidden !important;
}

/* ── Header (app-header elem_id wrapper) ────────────────────────────────────── */
#app-header,
div#app-header {
    background: linear-gradient(135deg, #0d1b2a, #1a2744) !important;
    border-bottom: 2px solid #1e40af !important;
    padding: 0 !important;
    margin-bottom: 8px !important;
}
#app-header h1,
div#app-header h1,
#app-header p,
div#app-header p {
    color: #ffffff !important;
}

/* ── Toolbar row (elem_id="toolbar") ───────────────────────────────────────── */
#toolbar,
div#toolbar {
    background: #ffffff !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    padding: 6px 10px !important;
    margin-bottom: 6px !important;
}

/* ── Tab styling ────────────────────────────────────────────────────────────── */
button[role="tab"],
.tab-nav button {
    background: #1e293b !important;
    color: #ffffff !important;
    border: 1px solid #334155 !important;
    font-weight: 600 !important;
    border-radius: 6px 6px 0 0 !important;
}
button[role="tab"][aria-selected="true"],
.tab-nav button.selected,
button[role="tab"].selected {
    background: #1e40af !important;
    color: #ffffff !important;
    border-color: #3b82f6 !important;
    border-bottom: 3px solid #60a5fa !important;
}
button[role="tab"]:hover:not([aria-selected="true"]) {
    background: #334155 !important;
    color: #ffffff !important;
}

/* ── Inputs ─────────────────────────────────────────────────────────────────── */
textarea,
input[type=text] {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #f1f5f9 !important;
    font-family: monospace !important;
}

/* ── Progress bars ──────────────────────────────────────────────────────────── */
.progress-bar-wrap {
    background: #1e293b !important;
    border-radius: 8px !important;
}
.progress-bar {
    background: linear-gradient(90deg, #3b82f6, #60a5fa) !important;
    border-radius: 8px !important;
}

/* ── Hidden utility textboxes ───────────────────────────────────────────────── */
#chat_copy_buf,
div#chat_copy_buf,
#rep_tts_buf,
div#rep_tts_buf,
#chat_tts_buf,
div#chat_tts_buf,
#ar_secs_input,
div#ar_secs_input {
    display: none !important;
}

/* ── Auto-refresh dropdown ──────────────────────────────────────────────────── */
#ar_dd,
div#ar_dd {
    font-size: 11px !important;
    min-width: 100px !important;
}
#ar_dd label,
div#ar_dd label {
    font-size: 10px !important;
    margin-bottom: 1px !important;
}
#ar_dd .wrap-inner,
#ar_dd .wrap,
div#ar_dd .wrap-inner,
div#ar_dd .wrap {
    padding: 2px 6px !important;
    min-height: unset !important;
}
#ar_dd input,
#ar_dd select,
div#ar_dd input,
div#ar_dd select {
    font-size: 11px !important;
    padding: 1px 4px !important;
}

/* ── Add Symbol button (blue gradient) ─────────────────────────────────────── */
#add-btn button,
div#add-btn button {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
}
#add-btn button:hover,
div#add-btn button:hover {
    background: linear-gradient(135deg, #2563eb, #60a5fa) !important;
}

/* ── Analyze Stock / Refresh Selected button (sky-blue gradient) ────────────── */
#refresh-sel-btn button,
div#refresh-sel-btn button {
    background: linear-gradient(135deg, #0369a1, #0ea5e9) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
}
#refresh-sel-btn button:hover,
div#refresh-sel-btn button:hover {
    background: linear-gradient(135deg, #0284c7, #38bdf8) !important;
}

/* ── Analyze All / Refresh All button (green gradient) ─────────────────────── */
#refresh-all-btn button,
div#refresh-all-btn button {
    background: linear-gradient(135deg, #065f46, #10b981) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
}
#refresh-all-btn button:hover,
div#refresh-all-btn button:hover {
    background: linear-gradient(135deg, #047857, #34d399) !important;
}

/* ── Save Dashboard button (purple gradient) ────────────────────────────────── */
#save-btn button,
div#save-btn button {
    background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
}
#save-btn button:hover,
div#save-btn button:hover {
    background: linear-gradient(135deg, #6d28d9, #c084fc) !important;
}

/* ── Copy Chat button (teal) ────────────────────────────────────────────────── */
#cpy-btn button,
div#cpy-btn button {
    background: linear-gradient(135deg, #0e7490, #06b6d4) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 600 !important;
}
#cpy-btn button:hover,
div#cpy-btn button:hover {
    background: linear-gradient(135deg, #0891b2, #22d3ee) !important;
}

/* ── Clear Chat button (red) ────────────────────────────────────────────────── */
#clr-btn button,
div#clr-btn button {
    background: linear-gradient(135deg, #991b1b, #ef4444) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 600 !important;
}
#clr-btn button:hover,
div#clr-btn button:hover {
    background: linear-gradient(135deg, #b91c1c, #f87171) !important;
}

/* ── Watchlist radio ────────────────────────────────────────────────────────── */
#watchlist_radio .wrap,
div#watchlist_radio .wrap {
    gap: 3px !important;
    flex-direction: column !important;
}
#watchlist_radio label,
div#watchlist_radio label {
    font-size: 10px !important;
    padding: 3px 8px !important;
    border-radius: 12px !important;
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #60a5fa !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    cursor: pointer !important;
    transition: all .15s !important;
}
#watchlist_radio label:hover,
div#watchlist_radio label:hover {
    background: #1e3a5f !important;
    border-color: #3b82f6 !important;
    color: #93c5fd !important;
}
#watchlist_radio input[type=radio]:checked ~ span,
#watchlist_radio label:has(input:checked),
div#watchlist_radio input[type=radio]:checked ~ span,
div#watchlist_radio label:has(input:checked) {
    background: linear-gradient(135deg, #0369a1, #0ea5e9) !important;
    border-color: #38bdf8 !important;
    color: #fff !important;
}
"""
THEME = gr.themes.Soft(primary_hue="blue", secondary_hue="slate", neutral_hue="slate")


if __name__ == "__main__":
    from utils.config import FRONTEND_PORT
    build_app().launch(
        server_name="0.0.0.0",
        server_port=FRONTEND_PORT,
        share=IS_HF_SPACE,
        css=CSS,
        theme=THEME,
        allowed_paths=[tempfile.gettempdir()],
    )
