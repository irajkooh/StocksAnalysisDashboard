"""
frontend.py — Gradio Frontend (port 7860)
Compatibility: Gradio 4.x / 5.x / 6.x

Toolbar (top):
  Row 1: [Symbol input] [Add to List] [Save & Reload] [Delete Stock]
  Row 2: [Analyze Stock] [Analyze All] [Refresh Stock] [Refresh All]
         [Auto-Refresh dd] [Workflow ON/OFF]

- Tabs built at startup from session.json
- Add stages symbol; Save & Reload persists + reloads
- Delete Stock removes the CURRENTLY SELECTED tab's symbol from session + reloads
- Analyze/Refresh Stock targets CURRENTLY SELECTED tab (tracked via gr.State)
- No Analyze/Refresh buttons inside tabs (only I OWN checkbox + TTS + Chatbot)
"""

import logging
import requests as req_lib

import gradio as gr

from config import (
    BACKEND_URL, IS_HF_SPACE, LLM_PROVIDER, OLLAMA_MODEL, GROQ_MODEL,
    DEFAULT_WATCHLIST, REFRESH_OPTIONS, MAX_CHATBOT_MEMORY,
)
from utils.device import get_device_label
from utils.session_manager import load_session, save_session
from utils.tts_engine import text_to_speech_file

logger = logging.getLogger(__name__)

# ─── Module-level state ───────────────────────────────────────────────────────
_active_symbols: list = []
_pending_symbols: list = []
_owned_map:      dict = {}
_analysis_cache: dict = {}
_chat_history:   dict = {}
_chatbot_ctx:    dict = {}
_watchlist:      list = []
_tab_components: dict = {}   # ticker -> list of 7 output components

SAMPLE_QUESTIONS = [
    "Good entry point?",
    "Key support levels?",
    "What does RSI say?",
    "Over or undervalued?",
    "What is the risk?",
    "Explain MACD signal",
]

_JS_RELOAD = "(x) => { setTimeout(() => window.location.reload(), 1200); return x; }"

# ─── Backend helpers ──────────────────────────────────────────────────────────

def _api(path, method="GET", payload=None, timeout=180):
    try:
        url = f"{BACKEND_URL}{path}"
        r = req_lib.get(url, timeout=timeout) if method == "GET" \
            else req_lib.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"API {method} {path}: {e}")
        return {}

def _analyze_ticker(ticker, owns):
    return _api("/analyze", "POST", {"ticker": ticker, "owns_stock": owns})

def _chat_api(ticker, question):
    resp = _api("/chat", "POST", {
        "ticker": ticker, "question": question,
        "chatbot_ctx": _chatbot_ctx.get(ticker, ""),
        "history":     _chat_history.get(ticker, []),
    })
    return resp.get("response", "Unable to respond right now.")

def _get_workflow():
    return _api("/workflow").get("mermaid", "")

# ─── HTML builders ────────────────────────────────────────────────────────────

def _status_bar_html():
    env   = "HuggingFace Spaces" if IS_HF_SPACE else "Local"
    dev   = get_device_label()
    model = OLLAMA_MODEL if LLM_PROVIDER == "ollama" else GROQ_MODEL
    llm   = f"{'Ollama' if LLM_PROVIDER == 'ollama' else 'Groq'} / {model}"
    return (
        '<div style="background:linear-gradient(135deg,#0d1b2a,#1a2744);'
        'border-bottom:2px solid #1e40af;padding:10px 20px;'
        'display:flex;align-items:center;gap:20px;font-family:monospace;flex-wrap:wrap">'
        '<b style="color:#38bdf8;font-size:17px;letter-spacing:2px">&#128202; STOCKS ANALYSIS DASHBOARD</b>'
        f'<span style="color:#60a5fa;font-size:12px">| {env}</span>'
        f'<span style="color:#60a5fa;font-size:12px">| {dev}</span>'
        f'<span style="color:#60a5fa;font-size:12px">| {llm}</span>'
        '</div>'
    )

def _card(label, value, color="#38bdf8", sub=None):
    s = f'<div style="color:#64748b;font-size:10px;margin-top:2px">{sub}</div>' if sub else ""
    return (
        '<div style="background:#1e293b;border:1px solid #334155;border-radius:8px;'
        f'padding:10px 14px;text-align:center;min-width:90px;display:inline-block;margin:2px">'
        f'<div style="color:#64748b;font-size:10px;text-transform:uppercase;'
        f'letter-spacing:1px;margin-bottom:4px">{label}</div>'
        f'<div style="color:{color};font-size:15px;font-weight:700">{value}</div>{s}</div>'
    )

def _hero_html(ticker, action, price, dcf, owns):
    ac   = "#22c55e" if "BUY" in str(action) else "#ef4444" if "SELL" in str(action) else "#facc15"
    iv   = dcf.get("intrinsic_value")
    prem = dcf.get("premium_discount")
    iv_s = f"${iv:.2f}" if iv is not None else "N/A"
    pr_s = f"{prem:+.1f}%" if prem is not None else "N/A"
    pc   = "#22c55e" if (prem or 0) < 0 else "#ef4444"
    badge = ('<b style="background:#7c3aed;color:#fff;padding:2px 8px;border-radius:12px;'
             'font-size:11px;margin-left:10px">I OWN THIS</b>') if owns else ""
    return (
        '<div style="background:linear-gradient(135deg,#0f172a,#1e293b);'
        'border:2px solid #1e40af;border-radius:12px;padding:20px 28px;margin-bottom:12px;'
        'display:flex;align-items:center;gap:24px;flex-wrap:wrap">'
        f'<div><div style="color:#94a3b8;font-size:11px;font-family:monospace;'
        f'text-transform:uppercase;letter-spacing:2px">{ticker}</div>'
        f'<div style="color:#f8fafc;font-size:32px;font-weight:800;margin:4px 0">${price:.2f}{badge}</div>'
        f'<div style="color:{ac};font-size:22px;font-weight:700">{action}</div></div>'
        f'<div style="display:flex;gap:8px;flex-wrap:wrap">'
        f'{_card("Intrinsic Value", iv_s, "#a78bfa")}'
        f'{_card("Premium/Discount", pr_s, pc, "vs DCF")}'
        f'</div></div>'
    )

def _signals_html(decision, indicators, risk):
    reasons = decision.get("reasons", [])
    conf    = decision.get("confidence", 0)
    pp      = decision.get("probability_profit", 50)
    pl      = decision.get("probability_loss", 50)
    rsi     = indicators.get("rsi", 50)
    rc      = "#22c55e" if rsi < 35 else "#ef4444" if rsi > 65 else "#facc15"
    def dot_color(r):
        t = r.lower()
        if any(w in t for w in ["bull","buy","above","under","oversold"]): return "#22c55e"
        if any(w in t for w in ["bear","sell","below","over","risk"]):     return "#ef4444"
        return "#facc15"
    rows = "".join(
        f'<div style="color:#cbd5e1;font-size:12px;padding:3px 0;border-bottom:1px solid #1e293b">'
        f'<span style="color:{dot_color(r)}">&#9679;</span> {r}</div>'
        for r in reasons
    )
    return (
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px">'
        '<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:14px">'
        '<div style="color:#38bdf8;font-weight:700;font-size:13px;margin-bottom:8px">KEY SIGNALS</div>'
        + rows + '</div>'
        '<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:14px">'
        '<div style="color:#38bdf8;font-weight:700;font-size:13px;margin-bottom:8px">INDICATORS</div>'
        '<div style="display:flex;flex-wrap:wrap;gap:4px">'
        + _card("RSI",     f"{rsi:.1f}",                              rc,        indicators.get("rsi_state",""))
        + _card("ATR",     f'${indicators.get("atr",0):.2f}',         "#fb923c")
        + _card("Stoch K", f'{indicators.get("stoch_k",50):.1f}',     "#a78bfa")
        + _card("Stoch D", f'{indicators.get("stoch_d",50):.1f}',     "#818cf8")
        + _card("Ann.Vol", f'{risk.get("annual_volatility",0):.1f}%', "#f43f5e")
        + _card("Sharpe",  f'{risk.get("sharpe_ratio",0):.2f}',       "#22c55e")
        + _card("Max DD",  f'{risk.get("max_drawdown",0):.1f}%',      "#ef4444")
        + _card("Risk",    f'{risk.get("risk_score",5)}/10',          "#fb923c",  risk.get("risk_label",""))
        + '</div></div></div>'
        '<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;'
        'padding:14px;margin-bottom:12px">'
        '<div style="color:#38bdf8;font-weight:700;font-size:13px;margin-bottom:10px">TRADE PROBABILITY</div>'
        '<div style="display:flex;gap:16px;align-items:center;flex-wrap:wrap">'
        '<div style="flex:1;min-width:140px">'
        '<div style="color:#94a3b8;font-size:11px;margin-bottom:4px">Profit Probability</div>'
        '<div style="background:#0f172a;border-radius:6px;height:20px;overflow:hidden">'
        f'<div style="background:#22c55e;width:{pp}%;height:100%;display:flex;align-items:center;'
        f'justify-content:center;color:#fff;font-size:11px;font-weight:700">{pp}%</div></div></div>'
        '<div style="flex:1;min-width:140px">'
        '<div style="color:#94a3b8;font-size:11px;margin-bottom:4px">Loss Probability</div>'
        '<div style="background:#0f172a;border-radius:6px;height:20px;overflow:hidden">'
        f'<div style="background:#ef4444;width:{pl}%;height:100%;display:flex;align-items:center;'
        f'justify-content:center;color:#fff;font-size:11px;font-weight:700">{pl}%</div></div></div>'
        f'<div style="color:#60a5fa;font-size:13px;font-weight:700">Confidence: {conf}%</div>'
        '</div></div>'
    )

def _levels_html(supports, resistances, pivots, fibonacci):
    def fmt(lst, color):
        if not lst: return "<span style='color:#475569'>N/A</span>"
        return "  |  ".join(f'<span style="color:{color};font-weight:700">${v:.2f}</span>' for v in lst)
    fib_rows = "".join(
        f'<div style="display:flex;justify-content:space-between;padding:2px 0;border-bottom:1px solid #1e293b">'
        f'<span style="color:#94a3b8;font-size:11px">{k}</span>'
        f'<span style="color:#a78bfa;font-size:11px;font-weight:600">${v:.2f}</span></div>'
        for k, v in list(fibonacci.items())[:7]
    )
    piv_rows = "".join(
        f'<div style="display:flex;justify-content:space-between;padding:2px 0;border-bottom:1px solid #1e293b">'
        f'<span style="color:#94a3b8;font-size:11px">{k}</span>'
        f'<span style="color:#{"22c55e" if k.startswith("R") else "ef4444" if k.startswith("S") else "facc15"};'
        f'font-size:11px;font-weight:700">${v:.2f}</span></div>'
        for k, v in pivots.items()
    )
    return (
        '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:12px">'
        '<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:14px">'
        '<div style="color:#22c55e;font-weight:700;font-size:13px;margin-bottom:8px">SUPPORT ZONES</div>'
        f'<div style="margin-bottom:10px">{fmt(supports,"#22c55e")}</div>'
        '<div style="color:#ef4444;font-weight:700;font-size:13px;margin-bottom:8px">RESISTANCE ZONES</div>'
        f'<div>{fmt(resistances,"#ef4444")}</div></div>'
        '<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:14px">'
        '<div style="color:#a78bfa;font-weight:700;font-size:13px;margin-bottom:8px">FIBONACCI LEVELS</div>'
        + fib_rows + '</div>'
        '<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:14px">'
        '<div style="color:#facc15;font-weight:700;font-size:13px;margin-bottom:8px">PIVOT POINTS</div>'
        + piv_rows + '</div></div>'
    )

def _fundamentals_html(fund, dcf):
    def row(label, val, color="#f1f5f9"):
        v = str(val) if val is not None else "N/A"
        return (
            f'<div style="display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #1e293b">'
            f'<span style="color:#94a3b8;font-size:12px">{label}</span>'
            f'<span style="color:{color};font-size:12px;font-weight:600">{v}</span></div>'
        )
    dc  = dcf.get("label","N/A")
    dcc = "#22c55e" if "Under" in dc else "#ef4444" if "Over" in dc else "#facc15"
    dy  = fund.get("dividend_yield")
    roe = fund.get("return_on_equity")
    return (
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px">'
        '<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:14px">'
        '<div style="color:#38bdf8;font-weight:700;font-size:13px;margin-bottom:8px">FUNDAMENTALS</div>'
        + row("Sector",    fund.get("sector","N/A"))
        + row("P/E",       fund.get("pe_ratio"))
        + row("Fwd P/E",   fund.get("forward_pe"))
        + row("PEG",       fund.get("peg_ratio"))
        + row("P/B",       fund.get("price_to_book"))
        + row("Beta",      fund.get("beta"))
        + row("Div Yield", f"{dy*100:.2f}%" if dy else "N/A")
        + '</div>'
        '<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:14px">'
        '<div style="color:#a78bfa;font-weight:700;font-size:13px;margin-bottom:8px">DCF VALUATION</div>'
        + row("Intrinsic Value",  f'${dcf.get("intrinsic_value","N/A")}', "#a78bfa")
        + row("Current Price",    f'${dcf.get("current_price","N/A")}')
        + row("Premium/Disc.",    f'{dcf.get("premium_discount","N/A")}%', dcc)
        + row("Margin of Safety", f'{dcf.get("margin_of_safety","N/A")}%', "#22c55e")
        + row("Growth Rate",      f'{dcf.get("growth_rate_used","N/A")}%')
        + row("Discount Rate",    f'{dcf.get("discount_rate","N/A")}%')
        + row("Valuation",        dc, dcc)
        + row("ROE",              f"{roe*100:.1f}%" if roe else "N/A")
        + '</div></div>'
    )

def _sentiment_html(sentiment):
    agg = sentiment.get("aggregate_score", 0)
    lbl = sentiment.get("aggregate_label", "N/A")
    tw  = sentiment.get("twitter", {})
    rd  = sentiment.get("reddit",  {})
    nw  = sentiment.get("news",    {})
    sc  = sentiment.get("sec",     {})
    ac  = "#22c55e" if agg > 0.1 else "#ef4444" if agg < -0.1 else "#facc15"
    def bar(name, score, count, color):
        pct = int((float(score) + 1) / 2 * 100)
        return (
            f'<div style="margin-bottom:8px">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:2px">'
            f'<span style="color:#94a3b8;font-size:11px">{name}</span>'
            f'<span style="color:{color};font-size:11px">{score:.2f} ({count} items)</span></div>'
            f'<div style="background:#0f172a;border-radius:4px;height:8px">'
            f'<div style="background:{color};width:{pct}%;height:100%;border-radius:4px"></div>'
            '</div></div>'
        )
    return (
        '<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;'
        'padding:14px;margin-bottom:12px">'
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">'
        '<b style="color:#38bdf8;font-size:13px">SENTIMENT ANALYSIS</b>'
        f'<b style="color:{ac};font-size:16px">{lbl} ({agg:+.2f})</b></div>'
        + bar("X.com / Twitter", tw.get("score",0), tw.get("count",0), "#1d9bf0")
        + bar("Reddit",          rd.get("score",0), rd.get("count",0), "#ff4500")
        + bar("News Headlines",  nw.get("score",0), nw.get("count",0), "#38bdf8")
        + bar("SEC Filings",     sc.get("score",0), sc.get("count",0), "#a78bfa")
        + '</div>'
    )

def _workflow_html_static(mermaid_src=""):
    nodes = [
        ("#0f766e", "Data Agent",      "yfinance OHLCV + Fundamentals"),
        ("#7c3aed", "Technical Agent", "RSI · MACD · SMA · BB · Fibonacci"),
        ("#b45309", "Sentiment Agent", "News · Reddit · X.com · SEC"),
        ("#0369a1", "Valuation Agent", "DCF Intrinsic Value"),
        ("#b91c1c", "Risk Agent",      "Volatility · Drawdown · ATR"),
        ("#065f46", "Decision Agent",  "Buy / Hold / Sell + LLM Narrative"),
    ]
    arrow = '<div style="text-align:center;color:#38bdf8;font-size:22px;margin:2px 0">&#8595;</div>'
    boxes = arrow.join(
        f'<div style="background:{c};border-radius:10px;padding:12px 20px;text-align:center;'
        f'margin:0 auto;max-width:380px">'
        f'<div style="color:#fff;font-weight:700;font-size:14px">{t}</div>'
        f'<div style="color:rgba(255,255,255,0.75);font-size:11px;margin-top:3px">{s}</div></div>'
        for c, t, s in nodes
    )
    raw = (
        '<details style="margin-top:16px"><summary style="color:#475569;font-size:11px;'
        'cursor:pointer;font-family:monospace">Show raw Mermaid source</summary>'
        f'<pre style="color:#64748b;font-size:11px;margin-top:8px;background:#0a0f1e;'
        f'padding:12px;border-radius:6px;overflow-x:auto;white-space:pre-wrap">{mermaid_src}</pre>'
        '</details>'
    ) if mermaid_src else ""
    return (
        '<div style="background:#0f172a;border:2px solid #1e40af;border-radius:12px;'
        'padding:24px;margin-bottom:14px">'
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">'
        '<b style="color:#38bdf8;font-size:15px;font-family:monospace">LangGraph Multi-Agent Workflow</b>'
        '<span style="color:#475569;font-size:11px">Sequential pipeline</span></div>'
        '<div style="display:flex;flex-direction:column">'
        '<div style="background:#1e293b;border-radius:8px;padding:8px 20px;text-align:center;'
        'margin:0 auto;max-width:380px;color:#1e40af;font-weight:700;font-size:13px">START</div>'
        f'{arrow}{boxes}{arrow}'
        '<div style="background:#1e293b;border-radius:8px;padding:8px 20px;text-align:center;'
        'margin:0 auto;max-width:380px;color:#1e40af;font-weight:700;font-size:13px">END</div>'
        f'</div>{raw}</div>'
    )

def _watchlist_html(wl):
    items = "".join(
        f'<div style="background:#1e293b;border:1px solid #334155;border-radius:6px;'
        f'padding:6px 10px;color:#38bdf8;font-size:13px;font-weight:600;font-family:monospace">{s}</div>'
        for s in wl
    )
    return (f'<div style="display:flex;flex-direction:column;gap:5px">{items}</div>'
            if items else '<span style="color:#475569;font-size:11px">Empty</span>')

# ─── Analysis runner ──────────────────────────────────────────────────────────

def _run_and_pack(ticker):
    """Run analysis for ticker; return 7-tuple for tab outputs."""
    owns = _owned_map.get(ticker, False)
    data = _analyze_ticker(ticker, owns)
    if not data or (data.get("errors") and not data.get("indicators")):
        err = "; ".join(data.get("errors", ["Unknown error"]))
        return (
            f'<div style="color:#ef4444;padding:16px"><b>Error:</b> {err}</div>',
            None, "", "", "", "", f"**Analysis failed:** {err}"
        )
    ind  = data.get("indicators", {})
    dcf  = data.get("dcf", {})
    dec  = data.get("decision", {})
    fund = data.get("fundamentals", {})
    sent = data.get("sentiment", {})
    risk = data.get("risk", {})
    _chatbot_ctx[ticker]    = data.get("llm_chatbot_ctx", "")
    _analysis_cache[ticker] = data
    import plotly.io as pio
    fig = pio.from_json(data["chart_json"]) if data.get("chart_json") else None
    return (
        _hero_html(ticker, dec.get("action","N/A"), ind.get("price",0), dcf, owns),
        fig,
        _signals_html(dec, ind, risk),
        _levels_html(data.get("supports",[]), data.get("resistances",[]),
                     data.get("pivots",{}), data.get("fibonacci",{})),
        _fundamentals_html(fund, dcf),
        _sentiment_html(sent),
        data.get("llm_summary",""),
    )

# ─── Per-stock tab ────────────────────────────────────────────────────────────

def _build_stock_tab(ticker: str, selected_sym_state: gr.State):
    """
    Build tab contents. No analyze/refresh buttons inside.
    Registers a select handler so clicking this tab updates selected_sym_state.
    Returns list of 7 output components.
    """
    with gr.Column():
        owns_chk = gr.Checkbox(label=f"I OWN {ticker}",
                               value=_owned_map.get(ticker, False))

        hero_html    = gr.HTML(
            f'<div style="color:#64748b;padding:16px">'
            f'<b>{ticker}</b> loaded. Use toolbar buttons to analyze.</div>'
        )
        chart_plot   = gr.Plot(show_label=False)
        signals_html = gr.HTML()
        levels_html  = gr.HTML()
        fund_html    = gr.HTML()
        sent_html    = gr.HTML()

        with gr.Accordion("AI Analysis Report", open=True):
            summary_md = gr.Markdown("*Use Analyze Stock or Analyze All from the toolbar.*")
            read_btn   = gr.Button("READ (TTS)", size="sm", variant="secondary")
            audio_out  = gr.Audio(visible=False, autoplay=True)

        with gr.Accordion("Ask About This Stock", open=False):
            chatbot    = gr.Chatbot(height=260, show_label=False)
            with gr.Row():
                chat_input = gr.Textbox(placeholder="Ask a question...",
                                        show_label=False, scale=4)
                voice_btn  = gr.Button("MIC", size="sm", scale=1)
                send_btn   = gr.Button("Send", variant="primary", size="sm", scale=1)
            with gr.Row():
                for q in SAMPLE_QUESTIONS:
                    qb = gr.Button(q, size="sm", min_width=0)
                    qb.click(fn=lambda v=q: v, outputs=chat_input)
            clear_btn = gr.Button("Clear Chat", size="sm", variant="stop")
            audio_in  = gr.Audio(sources=["microphone"], type="filepath",
                                 label="Voice Input", visible=False)

    outs = [hero_html, chart_plot, signals_html, levels_html, fund_html, sent_html, summary_md]
    _tab_components[ticker] = outs

    # Update owns map when checkbox changes (no re-analyze)
    owns_chk.change(fn=lambda v: _owned_map.update({ticker: v}), inputs=[owns_chk])

    # TTS
    def make_tts(text):
        if not text or text.startswith("*") or text.startswith("Use"):
            return gr.update(visible=False)
        path = text_to_speech_file(text)
        return gr.update(visible=True, value=path) if path else gr.update(visible=False)
    read_btn.click(fn=make_tts, inputs=[summary_md], outputs=[audio_out])

    # Chat
    def respond(question, history):
        if not question.strip():
            return history or [], ""
        answer  = _chat_api(ticker, question)
        history = list(history or []) + [[question, answer]]
        _chat_history.setdefault(ticker, [])
        _chat_history[ticker].append([question, answer])
        if len(_chat_history[ticker]) > MAX_CHATBOT_MEMORY:
            _chat_history[ticker] = _chat_history[ticker][-MAX_CHATBOT_MEMORY:]
        return history, ""
    send_btn.click(fn=respond, inputs=[chat_input, chatbot], outputs=[chatbot, chat_input])
    chat_input.submit(fn=respond, inputs=[chat_input, chatbot], outputs=[chatbot, chat_input])
    clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, chat_input])
    voice_btn.click(fn=lambda: gr.update(visible=True), outputs=[audio_in])

    def transcribe(path):
        if not path: return ""
        try:
            import whisper
            return whisper.load_model("tiny").transcribe(path).get("text","")
        except Exception:
            return "Voice requires: pip install openai-whisper"
    audio_in.change(fn=transcribe, inputs=[audio_in], outputs=[chat_input])

    return outs


# ─── Main app ─────────────────────────────────────────────────────────────────

def build_app():
    session = load_session()
    _active_symbols.clear()
    _active_symbols.extend(session.get("symbols", []))
    _owned_map.update(session.get("owned", {}))
    _watchlist.clear()
    _watchlist.extend(session.get("watchlist", list(DEFAULT_WATCHLIST)))
    saved_ref = session.get("refresh_interval", "Off")

    # (sym, [7 output components]) pairs — built during tab construction
    all_tab_outs: list = []

    demo = gr.Blocks(title="Stocks Analysis Dashboard")

    with demo:
        gr.HTML(_status_bar_html())

        # ── Row 1: Symbol management ─────────────────────────────────────
        with gr.Row():
            sym_input = gr.Textbox(placeholder="Enter symbol e.g. AAPL",
                                   show_label=False, scale=3, max_lines=1)
            add_btn   = gr.Button("Add to List",    variant="secondary", scale=1)
            save_btn  = gr.Button("Save & Reload",  variant="primary",   scale=1)
            del_btn   = gr.Button("Delete Stock",   variant="stop",      scale=1)

        # ── Row 2: Analysis actions ──────────────────────────────────────
        with gr.Row():
            analyze_stock_btn = gr.Button("Analyze Stock",  variant="primary",   scale=1)
            analyze_all_btn   = gr.Button("Analyze All",    variant="primary",   scale=1)
            refresh_stock_btn = gr.Button("Refresh Stock",  variant="secondary", scale=1)
            refresh_all_btn   = gr.Button("Refresh All",    variant="secondary", scale=1)
            refresh_dd        = gr.Dropdown(choices=list(REFRESH_OPTIONS.keys()),
                                            value=saved_ref, label="Auto-Refresh", scale=1)
            workflow_btn      = gr.Button("Workflow ON/OFF", variant="secondary", scale=1)

        status_msg  = gr.HTML("")
        pending_msg = gr.HTML("")

        # ── Workflow panel ───────────────────────────────────────────────
        wf_panel  = gr.HTML(value="", visible=False)
        wf_state  = gr.State(False)
        wf_cached = gr.State("")

        # Tracks the currently-selected tab symbol
        # Defaults to first symbol (the tab visible on load)
        default_selected = _active_symbols[0] if _active_symbols else ""
        selected_sym = gr.State(value=default_selected)

        # ── Layout: watchlist + tabs ─────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1, min_width=130):
                gr.HTML('<b style="color:#38bdf8;font-size:12px;'
                        'text-transform:uppercase;letter-spacing:1px">Watchlist</b>')
                wl_display = gr.HTML(_watchlist_html(_watchlist))
                wl_input   = gr.Textbox(placeholder="Add to watchlist...",
                                        show_label=False, max_lines=1)
                wl_add_btn = gr.Button("+ Add", size="sm")

            with gr.Column(scale=6):
                if _active_symbols:
                    with gr.Tabs() as tabs_container:
                        for sym in _active_symbols:
                            with gr.Tab(label=sym) as tab_obj:
                                tab_outs = _build_stock_tab(sym, selected_sym)
                                all_tab_outs.append((sym, tab_outs))
                                # When this tab is selected, update selected_sym state
                                tab_obj.select(
                                    fn=lambda s=sym: s,
                                    outputs=[selected_sym],
                                )
                else:
                    gr.HTML(
                        '<div style="text-align:center;padding:60px;color:#475569">'
                        '<div style="font-size:48px;margin-bottom:16px">&#128202;</div>'
                        '<b style="font-size:18px;color:#94a3b8">Welcome to Stocks Analysis Dashboard</b>'
                        '<br><br><span style="font-size:13px">'
                        'Enter a symbol, click <b>Add to List</b>, then <b>Save &amp; Reload</b>.'
                        '</span></div>'
                    )

        # ── Watchlist ────────────────────────────────────────────────────
        def add_to_watchlist(sym):
            sym = sym.strip().upper()
            if sym and sym not in _watchlist:
                _watchlist.append(sym)
            return _watchlist_html(_watchlist), ""
        wl_add_btn.click(fn=add_to_watchlist, inputs=[wl_input],
                         outputs=[wl_display, wl_input])

        # ── Add to pending list ──────────────────────────────────────────
        def add_symbol(sym):
            sym = sym.strip().upper()
            if not sym:
                return "", '<div style="color:#ef4444;font-size:12px">Enter a symbol first.</div>'
            if sym in _active_symbols:
                return "", f'<div style="color:#facc15;font-size:12px">{sym} is already saved.</div>'
            if sym in _pending_symbols:
                return "", f'<div style="color:#facc15;font-size:12px">{sym} already pending.</div>'
            _pending_symbols.append(sym)
            pending_str = ", ".join(_pending_symbols)
            return "", (
                f'<div style="color:#38bdf8;font-size:12px;padding:4px 8px;'
                f'background:#1e293b;border-radius:6px;border:1px solid #334155">'
                f'Pending: <b>{pending_str}</b> — click <b>Save &amp; Reload</b> to open tabs.</div>'
            )
        add_btn.click(fn=add_symbol, inputs=[sym_input], outputs=[sym_input, pending_msg])
        sym_input.submit(fn=add_symbol, inputs=[sym_input], outputs=[sym_input, pending_msg])

        # ── Save & Reload ────────────────────────────────────────────────
        def do_save(rv):
            for sym in _pending_symbols:
                if sym not in _active_symbols:
                    _active_symbols.append(sym)
                    _owned_map.setdefault(sym, False)
            _pending_symbols.clear()
            ok = save_session(_active_symbols, _owned_map, _watchlist, rv)
            return (f'<div style="color:#22c55e;font-size:12px">Saved — reloading...</div>'
                    if ok else '<div style="color:#ef4444;font-size:12px">Save failed.</div>')
        save_btn.click(fn=do_save, inputs=[refresh_dd], outputs=[status_msg], js=_JS_RELOAD)

        # ── Delete Stock (selected tab) ──────────────────────────────────
        def delete_stock(sym, rv):
            sym = sym.strip().upper() if sym else ""
            if not sym or sym not in _active_symbols:
                # Fallback: nothing selected yet — remove last
                if not _active_symbols:
                    return '<div style="color:#ef4444;font-size:12px">No symbols to delete.</div>'
                sym = _active_symbols[-1]
            _active_symbols.remove(sym)
            _owned_map.pop(sym, None)
            save_session(_active_symbols, _owned_map, _watchlist, rv)
            return f'<div style="color:#facc15;font-size:12px"><b>{sym}</b> deleted — reloading...</div>'
        del_btn.click(fn=delete_stock, inputs=[selected_sym, refresh_dd],
                      outputs=[status_msg], js=_JS_RELOAD)

        # ── Collect all output components (flat) for Analyze All / Refresh All ──
        all_flat_outs = []
        for _, tab_outs in all_tab_outs:
            all_flat_outs.extend(tab_outs)   # 7 per tab

        # ── Analyze Stock (selected tab only) ────────────────────────────
        def analyze_stock(sym):
            sym = (sym or "").strip().upper()
            if not sym or sym not in _tab_components:
                return '<div style="color:#ef4444;font-size:12px">No tab selected.</div>'
            _run_and_pack(sym)   # fills cache
            # We can't target a specific tab's outputs from a top-level button without
            # returning all flat outs. Instead return all flat outs, only updating target.
            results = []
            for s, _ in all_tab_outs:
                if s == sym:
                    results.extend(_run_and_pack(s))
                else:
                    results.extend([gr.update()] * 7)
            return results

        # ── Refresh Stock (same as Analyze Stock — clears cache first) ───
        def refresh_stock(sym):
            sym = (sym or "").strip().upper()
            if sym in _analysis_cache:
                del _analysis_cache[sym]
            return analyze_stock(sym)

        # ── Analyze All ──────────────────────────────────────────────────
        def analyze_all():
            results = []
            for sym, _ in all_tab_outs:
                results.extend(_run_and_pack(sym))
            return results if results else [gr.update()] * max(len(all_flat_outs), 1)

        # ── Refresh All ──────────────────────────────────────────────────
        def refresh_all():
            for sym, _ in all_tab_outs:
                _analysis_cache.pop(sym, None)
            return analyze_all()

        if all_flat_outs:
            analyze_stock_btn.click(fn=analyze_stock, inputs=[selected_sym], outputs=all_flat_outs)
            refresh_stock_btn.click(fn=refresh_stock, inputs=[selected_sym], outputs=all_flat_outs)
            analyze_all_btn.click(fn=analyze_all,   inputs=[],              outputs=all_flat_outs)
            refresh_all_btn.click(fn=refresh_all,   inputs=[],              outputs=all_flat_outs)
        else:
            no_tabs_msg = lambda: '<div style="color:#facc15;font-size:12px">Add and save symbols first.</div>'
            analyze_stock_btn.click(fn=no_tabs_msg, outputs=[status_msg])
            analyze_all_btn.click(fn=no_tabs_msg,   outputs=[status_msg])
            refresh_stock_btn.click(fn=no_tabs_msg, outputs=[status_msg])
            refresh_all_btn.click(fn=no_tabs_msg,   outputs=[status_msg])

        # ── Workflow toggle ──────────────────────────────────────────────
        def toggle_workflow(visible, cached):
            new_vis = not visible
            if not new_vis:
                return gr.update(visible=False, value=""), False, cached
            src = cached if cached else _get_workflow()
            return gr.update(visible=True, value=_workflow_html_static(src)), True, src
        workflow_btn.click(fn=toggle_workflow, inputs=[wf_state, wf_cached],
                           outputs=[wf_panel, wf_state, wf_cached])

    return demo


# ─── CSS / Theme ─────────────────────────────────────────────────────────────

CSS = """
body, .gradio-container {
    background: #0a0f1e !important;
    font-family: 'Segoe UI', system-ui, sans-serif !important;
}
.tab-nav button {
    background: #1e293b !important; color: #94a3b8 !important;
    border: 1px solid #334155 !important;
    font-family: monospace !important; font-weight: 600 !important;
    border-radius: 6px 6px 0 0 !important;
}
.tab-nav button.selected {
    background: #1e40af !important; color: #fff !important;
    border-color: #3b82f6 !important;
}
textarea, input[type=text] {
    background: #1e293b !important; border: 1px solid #334155 !important;
    color: #f1f5f9 !important; font-family: monospace !important;
}
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
    )
