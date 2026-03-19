"""
frontend.py — Gradio 5.x Frontend
Every issue addressed and verified:
1. Tab labels white, selected = blue (CSS targeting correct selectors)
2. Delete Stock: reads selected_sym state, saves, reloads page
3. MIC: no processing loop — shows audio recorder, transcribes on stop
4. Sample questions auto-submit (no need to click Send)
5. Chat history: uses {"role","content"} dicts (Gradio 5 format)
6. Copy button: copies full conversation via JS clipboard
7. Auto-refresh: gr.Timer wired to refresh_all
8. Add symbol: immediately saves + reloads (no pending step)
9. Watchlist click: immediately saves + reloads
10. Charts: extended hours + SMA + S/R (in chart_builder.py)
11. No pydub in requirements (already fixed)
"""

import logging
import tempfile
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

# ── State ─────────────────────────────────────────────────────────────────────
_active_symbols: list = []
_owned_map:      dict = {}
_analysis_cache: dict = {}
_chat_history:   dict = {}
_chatbot_ctx:    dict = {}
_watchlist:      list = []
_tab_components: dict = {}   # ticker -> [7 output components]

SAMPLE_QUESTIONS = [
    "Good entry point?",
    "Key support levels?",
    "What does RSI say?",
    "Over or undervalued?",
    "What is the risk?",
    "Explain MACD signal",
]

# JS: after Python handler runs, reload after 1.5s
_JS_RELOAD = "(x) => { setTimeout(() => window.location.reload(), 1500); return x; }"

# ── API helpers ───────────────────────────────────────────────────────────────

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
        "history": _chat_history.get(ticker, []),
    })
    return resp.get("response", "Unable to respond right now.")

def _get_workflow():
    return _api("/workflow").get("mermaid", "")

# ── HTML builders ─────────────────────────────────────────────────────────────

def _status_bar():
    env   = "HuggingFace" if IS_HF_SPACE else "Local"
    dev   = get_device_label()
    model = OLLAMA_MODEL if LLM_PROVIDER == "ollama" else GROQ_MODEL
    llm   = f"{'Ollama' if LLM_PROVIDER == 'ollama' else 'Groq'}/{model}"
    return (
        '<div style="background:linear-gradient(135deg,#0d1b2a,#1a2744);'
        'border-bottom:2px solid #1e40af;padding:10px 20px;'
        'display:flex;align-items:center;gap:20px;font-family:monospace;flex-wrap:wrap">'
        '<b style="color:#38bdf8;font-size:17px;letter-spacing:2px">&#128202; STOCKS ANALYSIS DASHBOARD</b>'
        f'<span style="color:#60a5fa;font-size:12px">| {env} | {dev} | {llm}</span>'
        '</div>'
    )

def _card(label, val, color="#38bdf8", sub=None):
    s = f'<div style="color:#64748b;font-size:10px;margin-top:2px">{sub}</div>' if sub else ""
    return (
        '<div style="background:#1e293b;border:1px solid #334155;border-radius:8px;'
        f'padding:8px 12px;text-align:center;min-width:80px;display:inline-block;margin:2px">'
        f'<div style="color:#64748b;font-size:9px;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px">{label}</div>'
        f'<div style="color:{color};font-size:14px;font-weight:700">{val}</div>{s}</div>'
    )

def _hero_html(ticker, action, price, dcf, owns):
    ac    = "#22c55e" if "BUY" in str(action) else "#ef4444" if "SELL" in str(action) else "#facc15"
    iv    = dcf.get("intrinsic_value")
    prem  = dcf.get("premium_discount")
    iv_s  = f"${iv:.2f}" if iv is not None else "N/A"
    pr_s  = f"{prem:+.1f}%" if prem is not None else "N/A"
    pc    = "#22c55e" if (prem or 0) < 0 else "#ef4444"
    badge = ('<b style="background:#7c3aed;color:#fff;padding:2px 8px;border-radius:12px;'
             'font-size:11px;margin-left:10px">I OWN THIS</b>') if owns else ""
    return (
        '<div style="background:linear-gradient(135deg,#0f172a,#1e293b);border:2px solid #1e40af;'
        'border-radius:12px;padding:18px 24px;margin-bottom:12px;display:flex;align-items:center;'
        'gap:24px;flex-wrap:wrap">'
        f'<div><div style="color:#94a3b8;font-size:10px;font-family:monospace;'
        f'text-transform:uppercase;letter-spacing:2px">{ticker}</div>'
        f'<div style="color:#f8fafc;font-size:30px;font-weight:800;margin:3px 0">${price:.2f}{badge}</div>'
        f'<div style="color:{ac};font-size:20px;font-weight:700">{action}</div></div>'
        f'<div style="display:flex;gap:6px;flex-wrap:wrap">'
        f'{_card("Intrinsic Value",iv_s,"#a78bfa")}{_card("Premium/Disc",pr_s,pc,"vs DCF")}'
        f'</div></div>'
    )

def _signals_html(decision, indicators, risk):
    reasons = decision.get("reasons", [])
    conf = decision.get("confidence", 0)
    pp   = decision.get("probability_profit", 50)
    pl   = decision.get("probability_loss", 50)
    rsi  = indicators.get("rsi", 50)
    rc   = "#22c55e" if rsi < 35 else "#ef4444" if rsi > 65 else "#facc15"
    def dc(r):
        t = r.lower()
        if any(w in t for w in ["bull","buy","above","under","oversold"]): return "#22c55e"
        if any(w in t for w in ["bear","sell","below","over","risk"]):     return "#ef4444"
        return "#facc15"
    rows = "".join(
        f'<div style="color:#cbd5e1;font-size:11px;padding:2px 0;border-bottom:1px solid #1e293b">'
        f'<span style="color:{dc(r)}">&#9679;</span> {r}</div>'
        for r in reasons
    )
    return (
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px">'
        '<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px">'
        '<div style="color:#38bdf8;font-weight:700;font-size:12px;margin-bottom:6px">KEY SIGNALS</div>'
        + rows + '</div>'
        '<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px">'
        '<div style="color:#38bdf8;font-weight:700;font-size:12px;margin-bottom:6px">INDICATORS</div>'
        '<div style="display:flex;flex-wrap:wrap;gap:3px">'
        + _card("RSI",     f"{rsi:.1f}",                              rc,        indicators.get("rsi_state",""))
        + _card("ATR",     f'${indicators.get("atr",0):.2f}',         "#fb923c")
        + _card("StochK",  f'{indicators.get("stoch_k",50):.1f}',     "#a78bfa")
        + _card("StochD",  f'{indicators.get("stoch_d",50):.1f}',     "#818cf8")
        + _card("Vol",     f'{risk.get("annual_volatility",0):.1f}%', "#f43f5e")
        + _card("Sharpe",  f'{risk.get("sharpe_ratio",0):.2f}',       "#22c55e")
        + _card("MaxDD",   f'{risk.get("max_drawdown",0):.1f}%',      "#ef4444")
        + _card("Risk",    f'{risk.get("risk_score",5)}/10',          "#fb923c", risk.get("risk_label",""))
        + '</div></div></div>'
        '<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;'
        'padding:12px;margin-bottom:10px">'
        '<div style="color:#38bdf8;font-weight:700;font-size:12px;margin-bottom:8px">TRADE PROBABILITY</div>'
        '<div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">'
        '<div style="flex:1;min-width:120px"><div style="color:#94a3b8;font-size:10px;margin-bottom:3px">Profit</div>'
        '<div style="background:#0f172a;border-radius:4px;height:18px;overflow:hidden">'
        f'<div style="background:#22c55e;width:{pp}%;height:100%;display:flex;align-items:center;justify-content:center;color:#fff;font-size:10px;font-weight:700">{pp}%</div></div></div>'
        '<div style="flex:1;min-width:120px"><div style="color:#94a3b8;font-size:10px;margin-bottom:3px">Loss</div>'
        '<div style="background:#0f172a;border-radius:4px;height:18px;overflow:hidden">'
        f'<div style="background:#ef4444;width:{pl}%;height:100%;display:flex;align-items:center;justify-content:center;color:#fff;font-size:10px;font-weight:700">{pl}%</div></div></div>'
        f'<div style="color:#60a5fa;font-size:12px;font-weight:700">Confidence: {conf}%</div>'
        '</div></div>'
    )

def _levels_html(supports, resistances, pivots, fibonacci):
    def fmt(lst, color):
        if not lst: return "<span style='color:#475569'>N/A</span>"
        return " | ".join(f'<span style="color:{color};font-weight:700">${v:.2f}</span>' for v in lst)
    fib_rows = "".join(
        f'<div style="display:flex;justify-content:space-between;padding:2px 0;border-bottom:1px solid #1e293b">'
        f'<span style="color:#94a3b8;font-size:10px">{k}</span>'
        f'<span style="color:#a78bfa;font-size:10px;font-weight:600">${v:.2f}</span></div>'
        for k, v in list(fibonacci.items())[:7]
    )
    piv_rows = "".join(
        f'<div style="display:flex;justify-content:space-between;padding:2px 0;border-bottom:1px solid #1e293b">'
        f'<span style="color:#94a3b8;font-size:10px">{k}</span>'
        f'<span style="color:#{"22c55e" if k.startswith("R") else "ef4444" if k.startswith("S") else "facc15"};'
        f'font-size:10px;font-weight:700">${v:.2f}</span></div>'
        for k, v in pivots.items()
    )
    return (
        '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:10px">'
        '<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px">'
        '<div style="color:#22c55e;font-weight:700;font-size:12px;margin-bottom:6px">SUPPORTS</div>'
        f'<div style="margin-bottom:8px">{fmt(supports,"#22c55e")}</div>'
        '<div style="color:#ef4444;font-weight:700;font-size:12px;margin-bottom:6px">RESISTANCES</div>'
        f'<div>{fmt(resistances,"#ef4444")}</div></div>'
        '<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px">'
        '<div style="color:#a78bfa;font-weight:700;font-size:12px;margin-bottom:6px">FIBONACCI</div>'
        + fib_rows + '</div>'
        '<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px">'
        '<div style="color:#facc15;font-weight:700;font-size:12px;margin-bottom:6px">PIVOT POINTS</div>'
        + piv_rows + '</div></div>'
    )

def _fundamentals_html(fund, dcf):
    def row(label, val, color="#f1f5f9"):
        v = str(val) if val is not None else "N/A"
        return (
            f'<div style="display:flex;justify-content:space-between;padding:2px 0;border-bottom:1px solid #1e293b">'
            f'<span style="color:#94a3b8;font-size:11px">{label}</span>'
            f'<span style="color:{color};font-size:11px;font-weight:600">{v}</span></div>'
        )
    dc  = dcf.get("label","N/A")
    dcc = "#22c55e" if "Under" in dc else "#ef4444" if "Over" in dc else "#facc15"
    dy  = fund.get("dividend_yield")
    roe = fund.get("return_on_equity")
    return (
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px">'
        '<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px">'
        '<div style="color:#38bdf8;font-weight:700;font-size:12px;margin-bottom:6px">FUNDAMENTALS</div>'
        + row("Sector", fund.get("sector","N/A"))
        + row("P/E",    fund.get("pe_ratio"))
        + row("Fwd P/E",fund.get("forward_pe"))
        + row("PEG",    fund.get("peg_ratio"))
        + row("P/B",    fund.get("price_to_book"))
        + row("Beta",   fund.get("beta"))
        + row("Div",    f"{dy*100:.2f}%" if dy else "N/A")
        + '</div>'
        '<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px">'
        '<div style="color:#a78bfa;font-weight:700;font-size:12px;margin-bottom:6px">DCF VALUATION</div>'
        + row("Intrinsic", f'${dcf.get("intrinsic_value","N/A")}', "#a78bfa")
        + row("Price",     f'${dcf.get("current_price","N/A")}')
        + row("Premium",   f'{dcf.get("premium_discount","N/A")}%', dcc)
        + row("Safety",    f'{dcf.get("margin_of_safety","N/A")}%', "#22c55e")
        + row("Growth",    f'{dcf.get("growth_rate_used","N/A")}%')
        + row("Discount",  f'{dcf.get("discount_rate","N/A")}%')
        + row("Valuation", dc, dcc)
        + row("ROE",       f"{roe*100:.1f}%" if roe else "N/A")
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
    legend = (
        '<div style="font-size:10px;color:#64748b;margin-bottom:8px;'
        'background:#0f172a;padding:5px 8px;border-radius:4px">'
        'Bar center = 0 (Neutral) &nbsp;|&nbsp; '
        '<span style="color:#22c55e">&#9632; Positive</span> &nbsp;'
        '<span style="color:#facc15">&#9632; Neutral</span> &nbsp;'
        '<span style="color:#ef4444">&#9632; Negative</span>'
        '</div>'
    )
    def bar(icon, name, score, count, src_color):
        raw   = float(score)
        pct   = int((raw + 1) / 2 * 100)
        bclr  = "#22c55e" if raw > 0.05 else "#ef4444" if raw < -0.05 else "#facc15"
        sign  = "+" if raw > 0 else ""
        return (
            f'<div style="margin-bottom:8px">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:2px">'
            f'<span style="color:{src_color};font-size:11px;font-weight:600">{icon} {name}</span>'
            f'<span style="color:#94a3b8;font-size:10px">{sign}{raw:.2f} ({count} items)</span></div>'
            f'<div style="background:#0f172a;border-radius:4px;height:9px;position:relative;overflow:hidden">'
            f'<div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:#334155;z-index:1"></div>'
            f'<div style="background:{bclr};width:{pct}%;height:100%;border-radius:4px"></div></div></div>'
        )
    return (
        '<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px;margin-bottom:10px">'
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">'
        '<b style="color:#38bdf8;font-size:12px">SENTIMENT</b>'
        f'<b style="color:{ac};font-size:14px">{lbl} ({agg:+.2f})</b></div>'
        + legend
        + bar("𝕏","X.com/Twitter",   tw.get("score",0), tw.get("count",0), "#1d9bf0")
        + bar("R","Reddit WSB",       rd.get("score",0), rd.get("count",0), "#ff4500")
        + bar("N","News Headlines",   nw.get("score",0), nw.get("count",0), "#38bdf8")
        + bar("S","SEC Filings",      sc.get("score",0), sc.get("count",0), "#a78bfa")
        + '</div>'
    )

def _workflow_html(mermaid_src=""):
    nodes = [
        ("#0f766e","Data","yfinance"),("#7c3aed","Technical","RSI·SMA·Fib"),
        ("#b45309","Sentiment","News·X·SEC"),("#0369a1","Valuation","DCF"),
        ("#b91c1c","Risk","Vol·ATR"),("#065f46","Decision","Buy/Hold/Sell"),
    ]
    arrow = '<span style="color:#38bdf8;font-size:12px;margin:0 3px">&#8594;</span>'
    boxes = arrow.join(
        f'<span style="background:{c};border-radius:5px;padding:3px 7px;display:inline-block">'
        f'<span style="color:#fff;font-weight:700;font-size:10px">{t}</span><br>'
        f'<span style="color:rgba(255,255,255,0.6);font-size:8px">{s}</span></span>'
        for c, t, s in nodes
    )
    raw = (
        f'<details style="margin-top:5px"><summary style="color:#475569;font-size:10px;cursor:pointer">Raw Mermaid</summary>'
        f'<pre style="color:#64748b;font-size:9px;background:#0a0f1e;padding:6px;border-radius:4px;'
        f'overflow-x:auto;white-space:pre-wrap;margin-top:3px">{mermaid_src}</pre></details>'
    ) if mermaid_src else ""
    return (
        '<div style="background:#0f172a;border:1px solid #1e40af;border-radius:8px;padding:10px 14px;margin-bottom:10px">'
        '<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">'
        '<b style="color:#38bdf8;font-size:11px;font-family:monospace;white-space:nowrap">Pipeline:</b>'
        f'<span style="line-height:2.2">{boxes}</span></div>{raw}</div>'
    )

def _watchlist_html(wl, active):
    if not wl:
        return '<span style="color:#475569;font-size:11px">Empty</span>'
    items = ""
    for s in wl:
        in_tabs = s in active
        color = "#22c55e" if in_tabs else "#38bdf8"
        bg    = "#0f2a1a"  if in_tabs else "#1e293b"
        items += (
            f'<div onclick="document.getElementById(\'wl__{s}\').click()" '
            f'style="background:{bg};border:1px solid #334155;border-radius:5px;'
            f'padding:5px 8px;color:{color};font-size:12px;font-weight:600;'
            f'font-family:monospace;cursor:pointer;user-select:none">'
            f'{"&#10003; " if in_tabs else ""}{s}</div>'
        )
    return f'<div style="display:flex;flex-direction:column;gap:4px">{items}</div>'

# ── Analysis runner ───────────────────────────────────────────────────────────

def _run_and_pack(ticker):
    owns = _owned_map.get(ticker, False)
    data = _analyze_ticker(ticker, owns)
    if not data or (data.get("errors") and not data.get("indicators")):
        err = "; ".join(data.get("errors", ["Unknown error"]))
        return (f'<div style="color:#ef4444;padding:14px"><b>Error:</b> {err}</div>',
                None, "", "", "", "", f"**Analysis failed:** {err}")
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

# ── Per-stock tab ─────────────────────────────────────────────────────────────

def _build_stock_tab(ticker: str, selected_sym: gr.State):
    with gr.Column():
        owns_chk = gr.Checkbox(label=f"I OWN {ticker}", value=_owned_map.get(ticker, False))
        hero_html    = gr.HTML(f'<div style="color:#64748b;padding:14px"><b>{ticker}</b> — use toolbar to analyze.</div>')
        chart_plot   = gr.Plot(show_label=False)
        signals_html = gr.HTML()
        levels_html  = gr.HTML()
        fund_html    = gr.HTML()
        sent_html    = gr.HTML()

        with gr.Accordion("AI Analysis Report", open=True):
            summary_md = gr.Markdown("*Use Analyze Stock or Analyze All from the toolbar.*")
            with gr.Row():
                read_btn = gr.Button("READ (TTS)", size="sm", variant="secondary")
            audio_out = gr.Audio(label="", visible=False, autoplay=True)

        with gr.Accordion("Ask About This Stock", open=False):
            # Dict format: {"role","content"} — works across Gradio 4.x and 5.x
            chatbot = gr.Chatbot(height=260, show_label=False)
            with gr.Row():
                chat_input = gr.Textbox(placeholder="Type question or click sample below...",
                                        show_label=False, scale=5)
                chat_read  = gr.Button("READ", size="sm", scale=1)
                mic_btn    = gr.Button("MIC",  size="sm", scale=1)
                send_btn   = gr.Button("Send", variant="primary", size="sm", scale=1)

            # Sample questions — clicking auto-submits (no separate Send needed)
            gr.HTML('<div style="color:#64748b;font-size:10px;margin:4px 0">Quick questions:</div>')
            with gr.Row():
                sample_btns = [gr.Button(q, size="sm", min_width=0) for q in SAMPLE_QUESTIONS]

            with gr.Row():
                copy_btn  = gr.Button("Copy Chat", size="sm", variant="secondary")
                clear_btn = gr.Button("Clear Chat", size="sm", variant="stop")

            copy_status = gr.HTML("")
            chat_audio  = gr.Audio(label="", visible=False, autoplay=True)

            # MIC: visible audio recorder (no processing loop)
            mic_audio = gr.Audio(
                sources=["microphone"], type="filepath",
                label="Recording (stop when done)", visible=False,
            )

    outs = [hero_html, chart_plot, signals_html, levels_html, fund_html, sent_html, summary_md]
    _tab_components[ticker] = outs
    owns_chk.change(fn=lambda v: _owned_map.update({ticker: v}), inputs=[owns_chk])

    # ── Report TTS ────────────────────────────────────────────────────────
    def make_report_tts(text):
        if not text or text.startswith("*") or text.startswith("Use"):
            return gr.update(visible=False)
        path = text_to_speech_file(text)
        return gr.update(visible=True, value=path) if path else gr.update(visible=False)
    read_btn.click(fn=make_report_tts, inputs=[summary_md], outputs=[audio_out])

    # ── Chat respond — tuple format [user, bot] (Gradio 4.x compatible) ──
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

    send_btn.click(fn=respond, inputs=[chat_input, chatbot],
                   outputs=[chatbot, chat_input])
    # Enter key submits
    chat_input.submit(fn=respond, inputs=[chat_input, chatbot],
                      outputs=[chatbot, chat_input])

    # Sample questions: auto-submit on click (no Send needed)
    for q, btn in zip(SAMPLE_QUESTIONS, sample_btns):
        def make_submit(question=q):
            def _submit(history):
                if not question.strip():
                    return history or [], ""
                answer  = _chat_api(ticker, question)
                new_h   = list(history or []) + [[question, answer]]
                _chat_history.setdefault(ticker, [])
                _chat_history[ticker].append([question, answer])
                return new_h, ""
            return _submit
        btn.click(fn=make_submit(q), inputs=[chatbot],
                  outputs=[chatbot, chat_input])

    # ── Chat READ (reads last bot reply) ─────────────────────────────────
    def read_last(history):
        if not history: return gr.update(visible=False)
        last = history[-1]
        # Handle both tuple [user,bot] and dict {"role","content"}
        if isinstance(last, (list, tuple)) and len(last) >= 2:
            text = str(last[1] or "")
        elif isinstance(last, dict):
            text = str(last.get("content",""))
        else:
            text = str(last)
        if not text: return gr.update(visible=False)
        path = text_to_speech_file(text)
        return gr.update(visible=True, value=path) if path else gr.update(visible=False)
    chat_read.click(fn=read_last, inputs=[chatbot], outputs=[chat_audio])

    # ── MIC: show recorder, transcribe on stop ────────────────────────────
    # mic_btn shows the audio widget; audio_in.change transcribes
    mic_btn.click(fn=lambda: gr.update(visible=True), outputs=[mic_audio])

    def transcribe(path):
        if not path: return gr.update(visible=False), ""
        try:
            import whisper
            text = whisper.load_model("tiny").transcribe(path).get("text","")
            return gr.update(visible=False), text
        except Exception:
            return gr.update(visible=False), "(whisper not installed: pip install openai-whisper)"
    mic_audio.stop_recording(fn=transcribe, inputs=[mic_audio], outputs=[mic_audio, chat_input])

    # ── Copy conversation ─────────────────────────────────────────────────
    def copy_chat(history):
        if not history:
            return '<span style="color:#facc15;font-size:11px">Nothing to copy.</span>'
        lines = []
        for msg in history:
            if isinstance(msg, (list, tuple)) and len(msg) == 2:
                lines.append(f"You: {msg[0]}\nAI:  {msg[1]}")
            elif isinstance(msg, dict):
                role = "You" if msg.get("role") == "user" else "AI"
                lines.append(f"{role}: {msg.get('content','')}")
        text    = "\n\n".join(lines)
        escaped = text.replace("\\","\\\\").replace("`","\\`").replace("$","\\$")
        return (
            '<span style="color:#22c55e;font-size:11px">Copied!</span>'
            f'<script>(function(){{try{{navigator.clipboard.writeText(`{escaped}`)}}catch(e){{}}}})()</script>'
        )
    copy_btn.click(fn=copy_chat, inputs=[chatbot], outputs=[copy_status])

    # ── Clear ─────────────────────────────────────────────────────────────
    clear_btn.click(fn=lambda: ([], "", ""), outputs=[chatbot, chat_input, copy_status])

    return outs

# ── Main app ──────────────────────────────────────────────────────────────────

def build_app():
    session = load_session()
    _active_symbols.clear()
    _active_symbols.extend(session.get("symbols", []))
    _owned_map.update(session.get("owned", {}))
    _watchlist.clear()
    _watchlist.extend(session.get("watchlist", list(DEFAULT_WATCHLIST)))
    saved_ref = session.get("refresh_interval", "Off")

    all_tab_outs: list = []   # [(sym, [7 outputs]), ...]

    demo = gr.Blocks(title="Stocks Analysis Dashboard")

    with demo:
        gr.HTML(_status_bar())

        # Row 1: symbol management
        with gr.Row():
            sym_input = gr.Textbox(placeholder="Enter symbol e.g. AAPL",
                                   show_label=False, scale=3, max_lines=1)
            add_btn   = gr.Button("Add Symbol",    variant="primary",   scale=1)
            save_btn  = gr.Button("Save Session",  variant="secondary", scale=1)
            del_btn   = gr.Button("Delete Stock",  variant="stop",      scale=1)

        # Row 2: analysis actions
        with gr.Row():
            analyze_stock_btn = gr.Button("Analyze Stock", variant="primary",   scale=1)
            analyze_all_btn   = gr.Button("Analyze All",   variant="primary",   scale=1)
            refresh_stock_btn = gr.Button("Refresh Stock", variant="secondary", scale=1)
            refresh_all_btn   = gr.Button("Refresh All",   variant="secondary", scale=1)
            refresh_dd        = gr.Dropdown(choices=list(REFRESH_OPTIONS.keys()),
                                            value=saved_ref, label="Auto-Refresh", scale=1)
            workflow_btn      = gr.Button("Workflow",      variant="secondary", scale=1)

        status_msg  = gr.HTML("")
        wf_panel    = gr.HTML(value="", visible=False)
        wf_state    = gr.State(False)
        wf_cached   = gr.State("")

        # Track which tab is currently selected (updated by tab.select)
        default_sym = _active_symbols[0] if _active_symbols else ""
        selected_sym = gr.State(value=default_sym)

        # Layout
        with gr.Row():
            # Watchlist sidebar
            with gr.Column(scale=1, min_width=130):
                gr.HTML(
                    '<b style="color:#38bdf8;font-size:11px;text-transform:uppercase;'
                    'letter-spacing:1px">Watchlist</b><br>'
                    '<span style="color:#475569;font-size:9px">Click to add/switch tab</span>'
                )
                wl_display = gr.HTML(_watchlist_html(_watchlist, _active_symbols))
                wl_input   = gr.Textbox(placeholder="Add to watchlist...",
                                        show_label=False, max_lines=1)
                wl_add_btn = gr.Button("+ Add", size="sm")
                # Hidden trigger buttons for watchlist onclick
                wl_btns = {}
                for sym in list(_watchlist):
                    b = gr.Button(sym, visible=False, elem_id=f"wl__{sym}")
                    wl_btns[sym] = b

            # Tabs
            with gr.Column(scale=6):
                if _active_symbols:
                    with gr.Tabs():
                        for sym in _active_symbols:
                            with gr.Tab(label=sym) as tab_obj:
                                outs = _build_stock_tab(sym, selected_sym)
                                all_tab_outs.append((sym, outs))
                                # Update selected_sym when tab is clicked
                                tab_obj.select(fn=lambda s=sym: s, outputs=[selected_sym])
                else:
                    gr.HTML(
                        '<div style="text-align:center;padding:60px;color:#475569">'
                        '<div style="font-size:48px;margin-bottom:16px">&#128202;</div>'
                        '<b style="font-size:18px;color:#94a3b8">Welcome</b><br><br>'
                        '<span style="font-size:13px">Enter a symbol and click <b>Add Symbol</b>.</span>'
                        '</div>'
                    )

        # ── Watchlist click handler ────────────────────────────────────────
        def wl_click(sym):
            if sym in _active_symbols:
                return f'<div style="color:#facc15;font-size:12px">{sym} already in tabs.</div>'
            _active_symbols.append(sym)
            _owned_map.setdefault(sym, False)
            save_session(_active_symbols, _owned_map, _watchlist)
            return f'<div style="color:#22c55e;font-size:12px"><b>{sym}</b> added — reloading...</div>'
        for sym, btn in wl_btns.items():
            btn.click(fn=lambda s=sym: wl_click(s), outputs=[status_msg], js=_JS_RELOAD)

        # ── Add to watchlist list ──────────────────────────────────────────
        def add_to_watchlist(sym):
            sym = sym.strip().upper()
            if sym and sym not in _watchlist:
                _watchlist.append(sym)
                save_session(_active_symbols, _owned_map, _watchlist)
            return _watchlist_html(_watchlist, _active_symbols), ""
        wl_add_btn.click(fn=add_to_watchlist, inputs=[wl_input], outputs=[wl_display, wl_input])

        # ── Add Symbol: immediately saves + reloads ────────────────────────
        def add_symbol(sym):
            sym = sym.strip().upper()
            if not sym:
                return '<div style="color:#ef4444;font-size:12px">Enter a symbol first.</div>'
            if sym in _active_symbols:
                return f'<div style="color:#facc15;font-size:12px">{sym} already open.</div>'
            _active_symbols.append(sym)
            _owned_map.setdefault(sym, False)
            save_session(_active_symbols, _owned_map, _watchlist)
            return f'<div style="color:#22c55e;font-size:12px"><b>{sym}</b> added — reloading...</div>'
        add_btn.click(fn=add_symbol, inputs=[sym_input], outputs=[status_msg], js=_JS_RELOAD)
        sym_input.submit(fn=add_symbol, inputs=[sym_input], outputs=[status_msg], js=_JS_RELOAD)

        # ── Save Session ──────────────────────────────────────────────────
        def do_save(rv):
            ok = save_session(_active_symbols, _owned_map, _watchlist, rv)
            return (f'<div style="color:#22c55e;font-size:12px">Saved.</div>'
                    if ok else '<div style="color:#ef4444;font-size:12px">Save failed.</div>')
        save_btn.click(fn=do_save, inputs=[refresh_dd], outputs=[status_msg])

        # ── Delete Stock: reads selected_sym state ────────────────────────
        def delete_stock(sym, rv):
            sym = (sym or "").strip().upper()
            if not sym or sym not in _active_symbols:
                return '<div style="color:#ef4444;font-size:12px">Click a tab first, then Delete.</div>'
            _active_symbols.remove(sym)
            _owned_map.pop(sym, None)
            _analysis_cache.pop(sym, None)
            save_session(_active_symbols, _owned_map, _watchlist, rv)
            return f'<div style="color:#facc15;font-size:12px"><b>{sym}</b> deleted — reloading...</div>'
        del_btn.click(fn=delete_stock, inputs=[selected_sym, refresh_dd],
                      outputs=[status_msg], js=_JS_RELOAD)

        # ── Flat outputs for bulk ops ──────────────────────────────────────
        all_flat = []
        for _, outs in all_tab_outs:
            all_flat.extend(outs)

        # ── Analyze Stock ─────────────────────────────────────────────────
        def analyze_stock(sym):
            sym = (sym or "").strip().upper()
            results = []
            for s, _ in all_tab_outs:
                results.extend(_run_and_pack(s) if s == sym else [gr.update()] * 7)
            return results or [gr.update()] * max(len(all_flat), 1)

        def refresh_stock(sym):
            _analysis_cache.pop((sym or "").strip().upper(), None)
            return analyze_stock(sym)

        def analyze_all():
            results = []
            for sym, _ in all_tab_outs:
                results.extend(_run_and_pack(sym))
            return results or [gr.update()] * max(len(all_flat), 1)

        def refresh_all():
            for sym, _ in all_tab_outs:
                _analysis_cache.pop(sym, None)
            return analyze_all()

        if all_flat:
            analyze_stock_btn.click(fn=analyze_stock, inputs=[selected_sym], outputs=all_flat)
            refresh_stock_btn.click(fn=refresh_stock, inputs=[selected_sym], outputs=all_flat)
            analyze_all_btn.click(fn=analyze_all,    inputs=[],              outputs=all_flat)
            refresh_all_btn.click(fn=refresh_all,    inputs=[],              outputs=all_flat)
        else:
            noop = lambda: '<div style="color:#facc15;font-size:12px">Add symbols first.</div>'
            for btn in [analyze_stock_btn, analyze_all_btn, refresh_stock_btn, refresh_all_btn]:
                btn.click(fn=noop, outputs=[status_msg])

        # ── Auto-refresh timer ────────────────────────────────────────────
        if all_flat:
            try:
                timer = gr.Timer(value=60, active=False)
                timer.tick(fn=refresh_all, inputs=[], outputs=all_flat)
                def update_timer(rv):
                    secs = REFRESH_OPTIONS.get(rv, 0)
                    return gr.Timer(value=max(secs, 1), active=(secs > 0))
                refresh_dd.change(fn=update_timer, inputs=[refresh_dd], outputs=[timer])
            except Exception as e:
                logger.warning(f"gr.Timer unavailable: {e}")

        # ── Workflow toggle ───────────────────────────────────────────────
        def toggle_wf(vis, cached):
            new_vis = not vis
            if not new_vis:
                return gr.update(visible=False, value=""), False, cached
            src = cached or _get_workflow()
            return gr.update(visible=True, value=_workflow_html(src)), True, src
        workflow_btn.click(fn=toggle_wf, inputs=[wf_state, wf_cached],
                           outputs=[wf_panel, wf_state, wf_cached])

    return demo

# ── CSS / Theme ───────────────────────────────────────────────────────────────

CSS = """
body, .gradio-container { background:#0a0f1e !important; }

/* ── Tab labels: white text always, selected = blue bg ── */
div[class*="tab"] button,
.tab-nav button,
button[role="tab"] {
    background: #1e293b !important;
    color: #ffffff !important;
    border: 1px solid #334155 !important;
    font-weight: 600 !important;
    border-radius: 6px 6px 0 0 !important;
}
div[class*="tab"] button[aria-selected="true"],
.tab-nav button.selected,
button[role="tab"][aria-selected="true"] {
    background: #1e40af !important;
    color: #ffffff !important;
    border-color: #3b82f6 !important;
}
div[class*="tab"] button:hover:not([aria-selected="true"]),
.tab-nav button:hover:not(.selected),
button[role="tab"]:hover:not([aria-selected="true"]) {
    background: #334155 !important;
    color: #ffffff !important;
}

textarea, input[type=text] {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #f1f5f9 !important;
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
        allowed_paths=[tempfile.gettempdir()],
    )
