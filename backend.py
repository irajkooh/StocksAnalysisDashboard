"""
backend.py — FastAPI Backend  (port 8000)
Exposes REST endpoints consumed by the Gradio frontend.
"""

import logging
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.supervisor import run_analysis, get_mermaid_diagram
from utils.session_manager import load_session, save_session
from utils.device import get_device, get_device_label
from config import IS_HF_SPACE, LLM_PROVIDER, OLLAMA_MODEL, GROQ_MODEL, REFRESH_OPTIONS

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

# ─── Cache (simple in-memory, per ticker) ─────────────────────────────────────
_analysis_cache: dict = {}
CACHE_TTL = 300  # seconds


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

    # Serialize: DataFrames are not JSON-serializable, remove them
    result = _serialize_state(state)

    _analysis_cache[cache_key] = {"_ts": time.time(), "data": result}
    return result


@app.post("/chat")
def chat(req: ChatRequest):
    """Per-stock chatbot using LLM with conversation history."""
    from config import LLM_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, GROQ_API_KEY, GROQ_MODEL

    system = req.chatbot_ctx or (
        f"You are a professional stock analyst assistant for {req.ticker}. "
        "Answer questions concisely and accurately based on technical and fundamental analysis."
    )

    # Build message history
    messages = []
    for pair in req.history[-10:]:  # last 10 turns
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            messages.append({"role": "user",      "content": pair[0]})
            messages.append({"role": "assistant", "content": pair[1]})

    messages.append({"role": "user", "content": req.question})

    try:
        if LLM_PROVIDER == "ollama":
            import requests as req_lib
            prompt_parts = [f"SYSTEM: {system}\n"]
            for m in messages:
                prefix = "USER" if m["role"] == "user" else "ASSISTANT"
                prompt_parts.append(f"{prefix}: {m['content']}")
            full_prompt = "\n".join(prompt_parts)

            r = req_lib.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": full_prompt, "stream": False},
                timeout=OLLAMA_TIMEOUT,
            )
            if r.status_code == 200:
                return {"response": r.json().get("response", "").strip()}

        elif LLM_PROVIDER == "groq":
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            resp   = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "system", "content": system}] + messages,
                max_tokens=600,
            )
            return {"response": resp.choices[0].message.content.strip()}

    except Exception as e:
        logger.error(f"Chat error: {e}")

    return {"response": "I'm unable to answer right now. Please check the LLM configuration."}


@app.get("/session")
def get_session():
    return load_session()


@app.post("/session/save")
def save_session_endpoint(req: SaveRequest):
    ok = save_session(req.symbols, req.owned, req.watchlist, req.refresh_interval)
    return {"ok": ok}


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
    from config import BACKEND_PORT
    uvicorn.run("backend:app", host="0.0.0.0", port=BACKEND_PORT, reload=False)
