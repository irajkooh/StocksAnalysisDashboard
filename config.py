"""
StocksAnalysisDashboard — Central Configuration
All tunable parameters and environment settings in one place.
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).parent
SESSION_FILE  = ROOT_DIR / "session.json"
STATIC_DIR    = ROOT_DIR / "static"

# ─── Deployment ───────────────────────────────────────────────────────────────
IS_HF_SPACE   = os.getenv("SPACE_ID") is not None
HF_TOKEN      = os.getenv("HF_TOKEN", "")
HF_USER       = "irajkoohi"
HF_SPACE_NAME = "StocksAnalysisDashboard"

# ─── Server Ports ─────────────────────────────────────────────────────────────
BACKEND_PORT  = int(os.getenv("BACKEND_PORT",  8000))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", 7860))
BACKEND_URL   = os.getenv("BACKEND_URL", f"http://127.0.0.1:{BACKEND_PORT}")

# ─── LLM Configuration ────────────────────────────────────────────────────────
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL       = os.getenv("OLLAMA_MODEL",    "llama3.2:3b")
OLLAMA_TIMEOUT     = 120          # seconds

GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL         = "llama3-70b-8192"

# Prefer Ollama locally; fall back to Groq on HF if key available
LLM_PROVIDER = "groq" if (IS_HF_SPACE and GROQ_API_KEY) else "ollama"

# ─── News / Sentiment API Keys ────────────────────────────────────────────────
NEWS_API_KEY       = os.getenv("NEWS_API_KEY", "")
REDDIT_CLIENT_ID   = os.getenv("REDDIT_CLIENT_ID",  "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT  = "StocksDashboard/1.0"
SEC_EDGAR_BASE_URL = "https://efts.sec.gov/LATEST/search-index?q={ticker}&dateRange=custom&startdt={start}&enddt={end}&forms=8-K,10-K,10-Q"

# ─── Data / Charting ──────────────────────────────────────────────────────────
DEFAULT_PERIOD      = "3mo"       # yfinance period for day-trading context
DEFAULT_INTERVAL    = "1d"        # candle interval
INTRADAY_INTERVAL   = "5m"        # used when period <= 5d
SMA_PERIODS         = [20, 50, 200]
RSI_PERIOD          = 14
MACD_FAST           = 12
MACD_SLOW           = 26
MACD_SIGNAL         = 9
STOCH_K             = 14
STOCH_D             = 3
ATR_PERIOD          = 14
BB_PERIOD           = 20
BB_STD              = 2
FIB_LOOKBACK_DAYS   = 60          # window for swing high/low detection

# ─── Intrinsic Value — DCF Parameters ─────────────────────────────────────────
DCF_DISCOUNT_RATE        = 0.10   # WACC approximation
DCF_TERMINAL_GROWTH_RATE = 0.03   # perpetuity growth
DCF_PROJECTION_YEARS     = 5
DCF_EPS_GROWTH_FALLBACK  = 0.08   # if analyst estimates unavailable

# ─── Trading Thresholds ───────────────────────────────────────────────────────
RSI_OVERSOLD     = 30
RSI_OVERBOUGHT   = 70
RSI_NEUTRAL_LOW  = 40
RSI_NEUTRAL_HIGH = 60

UNDERVALUED_THRESHOLD  = -0.15   # intrinsic discount ≥ 15% → undervalued
OVERVALUED_THRESHOLD   =  0.15   # intrinsic premium ≥ 15% → overvalued

# ─── Sentiment Weights ────────────────────────────────────────────────────────
SENTIMENT_WEIGHTS = {
    "twitter": 0.30,
    "reddit":  0.25,
    "news":    0.30,
    "sec":     0.15,
}

# ─── Auto-refresh Options (seconds) ───────────────────────────────────────────
REFRESH_OPTIONS = {
    "Off":    0,
    "1 min":  60,
    "5 min":  300,
    "15 min": 900,
    "30 min": 1800,
}

# ─── UI Defaults ──────────────────────────────────────────────────────────────
DEFAULT_WATCHLIST  = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META"]
CHART_THEME        = "plotly_dark"
CHART_HEIGHT_MAIN  = 420
CHART_HEIGHT_RSI   = 160
CHART_HEIGHT_MACD  = 160
MAX_CHATBOT_MEMORY = 20           # message pairs to keep per stock

# ─── TTS Settings ─────────────────────────────────────────────────────────────
TTS_LANG           = "en"
TTS_SLOW           = False
TTS_ACCENT         = "com"        # American English accent

# ─── Agent Timeouts ───────────────────────────────────────────────────────────
AGENT_TIMEOUT = {
    "data":      30,
    "technical": 10,
    "sentiment": 45,
    "valuation": 15,
    "risk":      10,
    "decision":  60,
}
