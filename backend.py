"""
backend.py — FastAPI Backend  (port 8000)
Exposes REST endpoints consumed by the Gradio frontend.
"""

import logging
import re
import time
import requests as _req_lib
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.supervisor import run_analysis, get_mermaid_diagram
from utils.session_manager import load_session, save_session
from utils.device import get_device, get_device_label
from utils.config import (
    IS_HF_SPACE, LLM_PROVIDER,
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT,
    GROQ_API_KEY, GROQ_MODEL,
    HF_TOKEN, HF_MODEL, REFRESH_OPTIONS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="StocksAnalysisDashboard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Pydantic Models ──────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    ticker:     str
    owns_stock: bool = False
    period:     str  = "3mo"
    interval:   str  = "1d"

class ChatRequest(BaseModel):
    ticker:      str
    question:    str
    chatbot_ctx: str
    history:     list = []

class SaveRequest(BaseModel):
    symbols:          list
    owned:            dict
    watchlist:        list = []
    refresh_interval: str  = "Off"

class LLMChatRequest(BaseModel):
    system_prompt: str
    question:      str
    history:       list = []

class CapitolTradesChatRequest(BaseModel):
    question: str
    history:  list = []

# ─── Cache (simple in-memory, per ticker) ─────────────────────────────────────
_analysis_cache: dict = {}
_charts_cache:   dict = {}   # ticker → {tf: base64_png}
_price_cache:    dict = {}   # ticker → {_ts, data} — lightweight price-only
CACHE_TTL       = 300  # seconds  (full analysis)
PRICE_CACHE_TTL =  60  # seconds  (price-only endpoint)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "device":       get_device_label(),
        "llm_provider": LLM_PROVIDER,
        "llm_model":    OLLAMA_MODEL if LLM_PROVIDER == "ollama" else GROQ_MODEL,
        "environment":  "HuggingFace Spaces" if IS_HF_SPACE else "Local",
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    ticker = req.ticker.upper().strip()
    if not ticker:
        raise HTTPException(400, "Ticker is required")

    cache_key = f"{ticker}_{req.owns_stock}_{req.period}"
    cached    = _analysis_cache.get(cache_key)
    if cached and (time.time() - cached["_ts"]) < CACHE_TTL:
        logger.info(f"Cache hit for {ticker}")
        return cached["data"]

    logger.info(f"Running full analysis for {ticker}")
    state = run_analysis(
        ticker     = ticker,
        owns_stock = req.owns_stock,
        period     = req.period,
        interval   = req.interval,
    )

    # Cache charts separately (large base64 blobs — not sent in main response)
    charts = state.get("charts", {})
    if charts:
        _charts_cache[ticker] = charts

    # Serialize: DataFrames are not JSON-serializable, remove them
    result = _serialize_state(state)
    result.pop("charts", None)   # strip from main response

    _analysis_cache[cache_key] = {"_ts": time.time(), "data": result}
    return result


@app.get("/price/{ticker}")
def get_price(ticker: str):
    """Lightweight price refresh: returns session_info with the latest traded price.
    Uses history(period='1d', interval='1m', prepost=True) so it always reflects
    the most recent trade — after-hours close, overnight ECN, or pre-market.
    60-second cache; no full analysis re-run."""
    ticker = ticker.upper().strip()
    if not ticker:
        raise HTTPException(400, "Ticker required")
    cached = _price_cache.get(ticker)
    if cached and (time.time() - cached["_ts"]) < PRICE_CACHE_TTL:
        # logger.debug(f"Price cache hit for {ticker}")
        return cached["data"]
    try:
        import yfinance as yf
        from agents.technical_agent import _session_info
        # Create a fresh Ticker to bypass yfinance's internal per-instance cache
        stock = yf.Ticker(ticker)

        # 1-min bars with prepost=True — last Close is always the most recent trade
        ext_last = ext_time = pre_last = reg_last = post_last = ovn_last = None
        try:
            df_1m = stock.history(period="1d", interval="1m", prepost=True,
                                  raise_errors=False)
            if df_1m is not None and not df_1m.empty:
                ext_last = float(df_1m["Close"].iloc[-1])
                ts = df_1m.index[-1]
                try:
                    h, m = ts.hour, ts.minute
                    ampm = "AM" if h < 12 else "PM"
                    h12  = h % 12 or 12
                    ext_time = f"{h12}:{m:02d} {ampm} ET"
                    # Classify last bar into its trading session
                    if h < 9 or (h == 9 and m < 30):
                        pre_last  = ext_last  # Pre-market:  4:00–9:30 AM ET
                    elif (h == 9 and m >= 30) or (10 <= h < 16):
                        reg_last  = ext_last  # Regular:     9:30 AM–4:00 PM ET
                    elif 16 <= h < 20:
                        post_last = ext_last  # After-hours: 4:00–8:00 PM ET
                    elif h >= 20:
                        ovn_last  = ext_last  # Overnight:   8:00 PM+ ET
                except Exception:
                    pass
        except Exception:
            pass

        # Daily bars for regular close + prev close reference
        df_1d = None
        try:
            df_1d = stock.history(period="2d", interval="1d", auto_adjust=True)
            if df_1d is not None and not df_1d.empty:
                df_1d.index = df_1d.index.tz_localize(None)
            else:
                df_1d = None
        except Exception:
            pass

        # Minimal info dict — skip slow stock.info entirely
        info = {}
        if ext_last  is not None: info["_ext_last_price"]  = ext_last
        if ext_time  is not None: info["_ext_last_time"]   = ext_time
        if pre_last  is not None: info["_pre_last_price"]  = pre_last
        if reg_last  is not None: info["_reg_last_price"]  = reg_last
        if post_last is not None: info["_post_last_price"] = post_last
        if ovn_last  is not None: info["_ovn_last_price"]  = ovn_last

        si = _session_info(info, df_1d)
        _price_cache[ticker] = {"_ts": time.time(), "data": si}
        return si
    except Exception as e:
        logger.error(f"price endpoint [{ticker}]: {e}")
        raise HTTPException(500, str(e))


@app.post("/chat")
def chat(req: ChatRequest):
    """Per-stock chatbot using LLM with conversation history."""
    system = req.chatbot_ctx or (
        f"You are a professional stock analyst assistant for {req.ticker}. "
        "Answer questions concisely and accurately based on technical and fundamental analysis."
    )
    messages = []
    for pair in req.history[-10:]:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            messages.append({"role": "user",      "content": pair[0]})
            messages.append({"role": "assistant", "content": pair[1]})
    messages.append({"role": "user", "content": req.question})
    try:
        return {"response": _call_llm(system, messages, max_tokens=600)}
    except Exception as e:
        logger.error(f"Chat error [{LLM_PROVIDER}]: {type(e).__name__}: {e}")
    return {"response": "I'm unable to answer right now. Please check the LLM configuration."}


# ── Capitol Trades helpers ────────────────────────────────────────────────────

def _format_ct_trades(trades: list) -> str:
    lines = []
    for t in trades:
        lines.append(
            f"- {t['politician'][:45]} | {t['issuer'][:30]} "
            f"| {t['type'].upper()} | {t['size']} | {t['trade_date']}"
        )
    return "\n".join(lines) if lines else "No trades found."


def _format_ct_issuers(issuers: list) -> str:
    lines = []
    for i in issuers:
        lines.append(
            f"- {i['name_ticker'][:40]} | Volume: {i['total_volume']} "
            f"| Trades: {i['trade_count']} | Politicians: {i['politician_count']}"
        )
    return "\n".join(lines) if lines else "No issuers found."


def _format_ct_politicians(politicians: list) -> str:
    lines = []
    for p in politicians:
        lines.append(
            f"- {p['name_info'][:55]} "
            f"| Trades: {p['trade_count']} | Volume: {p['total_volume']}"
        )
    return "\n".join(lines) if lines else "No politicians found."


def _call_llm(system: str, messages: list, max_tokens: int = 600) -> str:
    """Call the configured LLM and return the response text."""
    if LLM_PROVIDER == "groq":
        from groq import Groq, RateLimitError
        try:
            resp = Groq(api_key=GROQ_API_KEY).chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "system", "content": system}] + messages,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except RateLimitError:
            logger.warning("Groq rate limit hit — falling back to HF Inference API")
            # fall through to HF block below
        except Exception:
            raise
    if LLM_PROVIDER == "ollama":
        prompt = f"SYSTEM: {system}\n" + "\n".join(
            ("USER" if m["role"] == "user" else "ASSISTANT") + ": " + m["content"]
            for m in messages
        )
        r = _req_lib.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=OLLAMA_TIMEOUT,
        )
        if r.status_code == 200:
            return r.json().get("response", "").strip()
    # HF Inference API — also used as Groq rate-limit fallback
    from huggingface_hub import InferenceClient
    resp = InferenceClient(model=HF_MODEL, token=HF_TOKEN or None).chat_completion(
        messages=[{"role": "system", "content": system}] + messages,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


@app.post("/chat/capitol_trades")
def chat_capitol_trades(req: CapitolTradesChatRequest):
    """Chat endpoint that fetches live Capitol Trades data then queries the LLM."""
    from utils.capitol_trades_scraper import (
        get_recent_trades, get_trades_by_ticker, get_top_issuers,
        get_politicians, get_trade_summary_stats, get_latest_insights,
    )

    q = req.question
    q_lower = q.lower()

    # Detect an explicit ticker symbol in the question (2-5 uppercase letters)
    tickers_in_q = re.findall(r'\b([A-Z]{2,5})\b', q)

    data_context = ""
    try:
        if tickers_in_q and any(
            w in q_lower for w in ["trade", "bought", "sold", "buy", "sell", "politician", "congress"]
        ):
            for t in tickers_in_q[:1]:
                result = get_trades_by_ticker(t, page_size=10)
                if "error" not in result:
                    data_context = (
                        f"Recent Capitol Trades data for {t}:\n"
                        + _format_ct_trades(result["trades"][:10])
                    )
                    break

        if not data_context and any(
            w in q_lower for w in ["top", "most traded", "popular", "active", "issuer", "stocks"]
        ):
            result = get_top_issuers(page_size=10)
            data_context = "Top issuers traded by politicians:\n" + _format_ct_issuers(result["issuers"])

        if not data_context and any(
            w in q_lower for w in ["politician", "congress", "senator", "representative", "lawmaker", "who"]
        ):
            result = get_politicians(page_size=20)
            data_context = "Currently tracked politicians:\n" + _format_ct_politicians(result["politicians"])

        if not data_context and any(
            w in q_lower for w in ["stat", "total", "how many", "summary", "overview", "volume"]
        ):
            s = get_trade_summary_stats()
            data_context = (
                f"Capitol Trades statistics:\n"
                f"Total trades: {s.get('total_trades')}\n"
                f"Total volume: ${s.get('total_volume')}\n"
                f"Politicians tracked: {s.get('total_politicians')}\n"
                f"Issuers: {s.get('total_issuers')}"
            )

        if not data_context and any(
            w in q_lower for w in ["news", "insight", "article", "report", "latest"]
        ):
            result = get_latest_insights()
            data_context = "Latest Capitol Trades insights:\n" + "\n".join(
                f"- {a['title']}" for a in result.get("articles", [])[:8]
            )

        if not data_context:
            result = get_recent_trades(page_size=10)
            data_context = "Recent politician trades:\n" + _format_ct_trades(result["trades"][:10])

    except Exception as e:
        logger.error(f"Capitol Trades fetch error: {e}")
        data_context = "Could not fetch data from Capitol Trades at this time."

    system = (
        "You are a financial intelligence assistant specializing in US politician stock "
        "trade disclosures reported under the STOCK Act via Capitol Trades "
        "(capitoltrades.com). Answer the user's question concisely and helpfully using "
        "the real-time trade data provided. When listing trades, highlight notable "
        "patterns (large sizes, repeated buys/sells, sector clusters)."
    )
    user_msg = f"{q}\n\nRelevant Capitol Trades data:\n{data_context}"

    messages = []
    for pair in req.history[-6:]:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            messages.append({"role": "user",      "content": pair[0]})
            messages.append({"role": "assistant", "content": pair[1]})
    messages.append({"role": "user", "content": user_msg})

    try:
        return {"response": _call_llm(system, messages, max_tokens=800)}
    except Exception as e:
        logger.error(f"Capitol Trades LLM error [{LLM_PROVIDER}]: {type(e).__name__}: {e}")
    return {"response": "Unable to answer Capitol Trades questions right now."}


@app.get("/session")
def get_session():
    return load_session()


@app.post("/session/save")
def save_session_endpoint(req: SaveRequest):
    ok, err = save_session(req.symbols, req.owned, req.watchlist, req.refresh_interval)
    return {"ok": ok, "error": err}


@app.get("/chart/{ticker}/{tf}")
def get_chart(ticker: str, tf: str):
    ticker = ticker.upper().strip()
    charts = _charts_cache.get(ticker, {})
    b64    = charts.get(tf)
    if not b64:
        raise HTTPException(404, f"No chart for {ticker}/{tf}")
    return {"b64": b64}


@app.get("/workflow")
def workflow():
    return {"mermaid": get_mermaid_diagram()}


@app.get("/refresh_options")
def refresh_options():
    return {"options": list(REFRESH_OPTIONS.keys())}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _serialize_state(state: dict) -> dict:
    """Remove non-serializable objects (DataFrames) from state."""
    import math

    skip_keys = {"ohlcv_df", "financials", "earnings_hist"}
    out = {}
    for k, v in state.items():
        if k in skip_keys:
            continue
        try:
            import json
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = str(v)

    # Sanitize NaN/Inf in nested dicts
    return _sanitize(out)


def _sanitize(obj):
    import math
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(i) for i in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


# ─── Run directly ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from utils.config import BACKEND_PORT
    uvicorn.run("backend:app", host="0.0.0.0", port=BACKEND_PORT, reload=False)
