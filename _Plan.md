# 📊 StocksAnalysisDashboard — Project Plan

## Project Overview

A production-grade, AI-powered multi-tab stock analysis dashboard built for short-term day trading.
Uses a LangGraph multi-agent backend, Gradio frontend, FastAPI REST API, and Ollama/Groq LLM.
Deployable locally (MPS/CUDA) and on HuggingFace Spaces.

---

## Goals

- Analyze any stock symbol on demand with a single click
- Present actionable Buy / Hold / Sell signals with probability scores
- Provide intrinsic value, technical indicators, sentiment, and risk metrics in one view
- Support multiple symbols simultaneously via a tabbed dashboard
- Persist sessions across restarts
- Be deployable both locally and to HuggingFace Spaces with zero code changes

---

## User Requirements

| # | Requirement |
|---|---|
| 1 | Stock symbol input → full analysis (charts, indicators, sentiment, valuation) |
| 2 | Add Symbol button creates a new tab per symbol without losing existing tabs |
| 3 | Delete button removes the current tab and auto-saves session |
| 4 | Save button persists all open tabs to session.json |
| 5 | Session auto-restores all tabs on next app launch |
| 6 | Ollama as local LLM; Groq API as HuggingFace fallback |
| 7 | Device detection: MPS (Apple), CUDA (NVIDIA), CPU fallback |
| 8 | Status bar shows: environment, device, LLM — in blue at top of UI |
| 9 | Each tab shows suggestion (Buy/Hold/Sell) and last price in bold blue large font |
| 10 | "I Own This Stock" toggle per tab adjusts recommendation logic |
| 11 | Candlestick chart with Volume, SMA 20/50/200, RSI sub-chart, MACD sub-chart |
| 12 | Data from yfinance |
| 13 | Fibonacci retracement levels overlaid on chart |
| 14 | Sentiment from X.com/Twitter, NewsAPI, Reddit (WSB/investing), SEC filings |
| 15 | DCF (Discounted Cash Flow) intrinsic value calculation |
| 16 | Watchlist sidebar with quick-add |
| 17 | Auto-refresh interval toggle (Off / 1min / 5min / 15min / 30min) |
| 18 | LangGraph Supervisor + Specialized sub-agents architecture |
| 19 | LangGraph Mermaid workflow diagram toggle on top of UI |
| 20 | Chatbot widget per tab with conversation memory |
| 21 | Sample question buttons (same width) that auto-fill and submit |
| 22 | Voice toggle button for voice input to chatbot |
| 23 | READ button (TTS) for long text sections — American English accent |
| 24 | Backend: FastAPI on port 8000 |
| 25 | Frontend: Gradio on port 7860 |
| 26 | app.py launches both services |
| 27 | All tools in utils/ folder |
| 28 | All parameters in config.py |
| 29 | Instructions.md install guide |
| 30 | HowItWorks.md technical explanation |
| 31 | HuggingFace Space: irajkoohi/StocksAnalysisDashboard |
| 32 | Well-designed, colorful, user-friendly UI — no wasted space |

---

## Technical Selections (Finalized)

| Category | Choice | Reason |
|---|---|---|
| Additional Indicators | Fibonacci Retracement | Natural price magnet levels for day trading |
| Sentiment Sources | X.com + NewsAPI + Reddit + SEC | Broadest signal coverage |
| Intrinsic Value Method | DCF (Discounted Cash Flow) | Most rigorous; uses FCF and growth estimates |
| Extra UX Features | Watchlist sidebar + Auto-refresh | Quick access and live monitoring |
| Agent Architecture | Supervisor + Specialized Sub-agents | Clean separation of concerns; parallel fan-out |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  Gradio Frontend  :7860                       │
│                                                              │
│  ┌─────────────┐  ┌──────────────────────────────────────┐  │
│  │  Watchlist  │  │           Tab Container               │  │
│  │  Sidebar    │  │  ┌─────────┐ ┌─────────┐ ┌────────┐  │  │
│  │             │  │  │  AAPL   │ │  TSLA   │ │  + Add │  │  │
│  │ [AAPL]      │  │  │  tab    │ │  tab    │ │        │  │  │
│  │ [TSLA]      │  │  └─────────┘ └─────────┘ └────────┘  │  │
│  │ [NVDA]      │  │                                        │  │
│  │ ...         │  │  Per-tab: Hero · Chart · Signals       │  │
│  │ [+ Add]     │  │           Levels · Valuation · Sent.  │  │
│  └─────────────┘  │           LLM Summary · Chatbot        │  │
│                   └──────────────────────────────────────┘  │
└─────────────────────────────┬────────────────────────────────┘
                              │ HTTP REST
┌─────────────────────────────▼────────────────────────────────┐
│                  FastAPI Backend  :8000                       │
│                                                              │
│  POST /analyze      → run full agent pipeline                │
│  POST /chat         → LLM chatbot with history               │
│  GET  /session      → load saved session                     │
│  POST /session/save → persist session to JSON                │
│  GET  /workflow     → return Mermaid diagram string          │
│  GET  /health       → device, LLM, env info                  │
└─────────────────────────────┬────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────┐
│           LangGraph Multi-Agent Supervisor                    │
│                                                              │
│  START → [Data Agent]                                        │
│               ├──→ [Technical Agent]  ─┐                    │
│               ├──→ [Sentiment Agent]  ─┤→ [Risk Agent]      │
│               └──→ [Valuation Agent]  ─┘       │            │
│                                         [Decision Agent]     │
│                                                │             │
│                                           LLM Synthesis      │
│                                         (Ollama / Groq)      │
│                                              END             │
└──────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
StocksAnalysisDashboard/
│
├── app.py                      # Launcher — starts backend + frontend threads
├── backend.py                  # FastAPI REST API (port 8000)
├── frontend.py                 # Gradio UI (port 7860)
├── config.py                   # All settings, thresholds, API keys, ports
├── requirements.txt            # Python dependencies
├── README.md                   # HuggingFace Space metadata + overview
├── Instructions.md             # Install guide (local + HF deployment)
├── HowItWorks.md               # Full technical explanation with formulas
├── .env.example                # Template for environment variables
├── .gitignore
├── session.json                # Auto-created; persists tabs and state
│
├── agents/
│   ├── __init__.py
│   ├── state.py                # TypedDict shared state schema for LangGraph
│   ├── supervisor.py           # LangGraph StateGraph builder + Mermaid generator
│   ├── data_agent.py           # yfinance OHLCV + info + financials
│   ├── technical_agent.py      # All TA indicators + Plotly chart
│   ├── sentiment_agent.py      # Aggregate sentiment dispatcher
│   ├── valuation_agent.py      # DCF intrinsic value + fundamentals
│   ├── risk_agent.py           # Volatility, drawdown, Sharpe, risk score
│   └── decision_agent.py       # Rule-based scoring + LLM narrative
│
└── utils/
    ├── __init__.py
    ├── device.py               # MPS / CUDA / CPU detection
    ├── indicators.py           # SMA, EMA, RSI, MACD, BB, ATR, Stoch, Fib, S/R, Pivots
    ├── intrinsic_value.py      # DCF model + fundamental metrics extraction
    ├── chart_builder.py        # 3-panel Plotly dark chart (candle + RSI + MACD)
    ├── sentiment_scraper.py    # NewsAPI + Reddit + SEC EDGAR + X.com scrapers
    ├── session_manager.py      # JSON save / load for dashboard state
    └── tts_engine.py           # gTTS American accent → MP3 file for Gradio Audio
```

---

## LangGraph Agent Details

### Agent 1 — Data Agent
- **Library:** yfinance
- **Fetches:** OHLCV DataFrame, company .info dict, financials, earnings history
- **Auto-switches** to 5-minute candles when period ≤ 5 days (intraday)
- **Output keys:** `ohlcv_df`, `info`, `financials`, `earnings_hist`

### Agent 2 — Technical Agent
- **Runs on:** OHLCV DataFrame
- **Indicators:** SMA 20/50/200, RSI(14), MACD(12,26,9), Bollinger Bands(20,2σ), ATR(14), Stochastic(14,3), Fibonacci retracement, Support/Resistance zones, Pivot Points (classic)
- **Builds:** 3-panel Plotly chart (serialized to JSON for transport)
- **Output keys:** `indicators`, `supports`, `resistances`, `fibonacci`, `pivots`, `chart_json`

### Agent 3 — Sentiment Agent
- **Sources:**
  - X.com via Nitter public instances (cashtag search)
  - NewsAPI REST (7-day news window)
  - Reddit via PRAW or public JSON fallback (WSB, investing, stocks subreddits)
  - SEC EDGAR full-text search (8-K, 10-K, 10-Q, 90-day window)
- **Scoring:** Keyword NLP → [-1.0, +1.0] per source → weighted aggregate
- **Weights:** Twitter 30%, News 30%, Reddit 25%, SEC 15%
- **Output key:** `sentiment`

### Agent 4 — Valuation Agent
- **Method:** DCF (Discounted Cash Flow)
- **Inputs:** Free Cash Flow (or Net Income), shares outstanding, EPS growth estimate
- **Parameters:** Discount rate 10%, terminal growth 3%, 5-year projection
- **Also extracts:** P/E, Fwd P/E, PEG, P/B, P/S, EV/EBITDA, ROE, ROA, beta, margins, analyst targets
- **Output keys:** `dcf`, `fundamentals`

### Agent 5 — Risk Agent
- **Computes:** Daily/Annual volatility, Max drawdown, Sharpe ratio, ATR-based stop/target, risk/reward ratio, short interest
- **Risk Score:** Composite 1–10 from volatility + drawdown + beta + short interest
- **Output key:** `risk`

### Agent 6 — Decision Agent
- **Step 1:** Rule-based signal scoring (-10 to +10) across 13 weighted conditions
- **Step 2:** Maps score to Buy / Hold / Sell + profit/loss probabilities
- **Step 3:** Builds structured prompt → sends to Ollama or Groq LLM
- **LLM writes:** 3-paragraph analyst report (verdict, risk factors, trade strategy with price levels)
- **Output keys:** `decision`, `llm_summary`, `llm_chatbot_ctx`

---

## Decision Scoring Logic

| Signal | Score |
|---|---|
| RSI < 30 (oversold) | +2.0 |
| RSI > 70 (overbought) | -2.0 |
| MACD bullish crossover | +2.5 |
| MACD bearish crossover | -2.5 |
| Price above all SMAs (20 > 50 > 200) | +1.5 |
| Price below all SMAs | -1.5 |
| Price at Bollinger lower band | +1.0 |
| Price at Bollinger upper band | -1.0 |
| DCF undervalued ≥ 15% | +2.0 |
| DCF overvalued ≥ 15% | -1.5 |
| Aggregate sentiment > 0.15 | +1.0 |
| Aggregate sentiment < -0.15 | -1.0 |
| Risk score ≥ 8/10 | -1.0 |

**Thresholds:** ≥ 3.0 = Strong BUY · 1.0–2.9 = BUY · ±0.9 = HOLD · -1.0 to -2.9 = SELL · ≤ -3.0 = Strong SELL

---

## UI Design Specifications

### Color Palette (Dark Trading Terminal Theme)
| Element | Color |
|---|---|
| Background | #0a0f1e (deep navy) |
| Cards / Panels | #1e293b (slate dark) |
| Borders | #334155 |
| Primary accent | #38bdf8 (sky blue) |
| BUY / bullish | #22c55e (green) |
| SELL / bearish | #ef4444 (red) |
| HOLD / neutral | #facc15 (yellow) |
| SMA 20 | #38bdf8 |
| SMA 50 | #fb923c |
| SMA 200 | #a78bfa |
| Bollinger Bands | #facc15 |
| Fibonacci | slate shades |
| Text primary | #f1f5f9 |
| Text secondary | #94a3b8 |

### Fonts
- **Headers / Status bar:** IBM Plex Mono (monospace — trading terminal feel)
- **Body / UI:** Space Grotesk (clean, modern)

### Layout Per Tab
```
┌─────────────────────────────────────────────────────────┐
│  [I OWN STOCK toggle]  [Analyze button]  [Refresh btn]  │
├─────────────────────────────────────────────────────────┤
│  HERO: BIG TICKER · $PRICE · BUY/HOLD/SELL · Intrinsic │
├─────────────────────────────────────────────────────────┤
│                  3-Panel Plotly Chart                   │
│  (Candlestick+Vol+SMA+BB+Fib+S/R | RSI | MACD)         │
├────────────────────────┬────────────────────────────────┤
│  KEY SIGNALS list      │  INDICATOR GRID (8 metrics)    │
├────────────────────────┴────────────────────────────────┤
│        TRADE PROBABILITY bars (profit / loss %)         │
├───────────────┬────────────────┬────────────────────────┤
│  SUPPORT /    │  FIBONACCI     │  PIVOT POINTS          │
│  RESISTANCE   │  LEVELS        │  (P R1 R2 R3 S1 S2 S3) │
├───────────────┴────────────────┴────────────────────────┤
│  FUNDAMENTALS (left)     DCF VALUATION (right)          │
├─────────────────────────────────────────────────────────┤
│  SENTIMENT BARS (Twitter / Reddit / News / SEC)         │
├─────────────────────────────────────────────────────────┤
│  [Accordion] AI Analysis Report + READ button + Audio   │
├─────────────────────────────────────────────────────────┤
│  [Accordion] Chatbot + Sample Q buttons + Voice button  │
└─────────────────────────────────────────────────────────┘
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Device, LLM, environment info |
| POST | `/analyze` | Full multi-agent analysis for a ticker |
| POST | `/chat` | LLM chatbot response with history |
| GET | `/session` | Load saved session JSON |
| POST | `/session/save` | Persist symbols, owned state, watchlist |
| GET | `/workflow` | Return Mermaid diagram of agent graph |
| GET | `/refresh_options` | Available auto-refresh intervals |

---

## Configuration Parameters (config.py)

| Parameter | Default | Description |
|---|---|---|
| `BACKEND_PORT` | 8000 | FastAPI port |
| `FRONTEND_PORT` | 7860 | Gradio port |
| `OLLAMA_MODEL` | llama3.2:3b | Local LLM |
| `GROQ_MODEL` | llama3-70b-8192 | HF Spaces LLM |
| `DEFAULT_PERIOD` | 3mo | Chart lookback period |
| `DEFAULT_INTERVAL` | 1d | Candle interval |
| `SMA_PERIODS` | [20, 50, 200] | Moving average windows |
| `RSI_PERIOD` | 14 | RSI calculation window |
| `RSI_OVERSOLD` | 30 | Oversold threshold |
| `RSI_OVERBOUGHT` | 70 | Overbought threshold |
| `MACD_FAST/SLOW/SIGNAL` | 12/26/9 | MACD parameters |
| `BB_PERIOD` | 20 | Bollinger Band period |
| `BB_STD` | 2 | Bollinger Band standard deviations |
| `ATR_PERIOD` | 14 | ATR calculation window |
| `FIB_LOOKBACK_DAYS` | 60 | Swing high/low detection window |
| `DCF_DISCOUNT_RATE` | 0.10 | WACC proxy (10%) |
| `DCF_TERMINAL_GROWTH_RATE` | 0.03 | Perpetuity growth (3%) |
| `DCF_PROJECTION_YEARS` | 5 | DCF forecast horizon |
| `UNDERVALUED_THRESHOLD` | -0.15 | 15% discount = undervalued |
| `OVERVALUED_THRESHOLD` | 0.15 | 15% premium = overvalued |
| `CACHE_TTL` | 300 | Analysis cache (seconds) |
| `MAX_CHATBOT_MEMORY` | 20 | Chat history pairs per stock |

---

## Sentiment Source Details

| Source | API / Method | Rate Limit | Key Required |
|---|---|---|---|
| X.com / Twitter | Nitter public instance scraping | Low | No |
| NewsAPI | REST API (newsapi.org) | 100 req/day free | Yes (NEWS_API_KEY) |
| Reddit | PRAW library or public JSON | 60 req/min public | Optional (REDDIT_CLIENT_ID) |
| SEC EDGAR | efts.sec.gov full-text search | No limit | No |

---

## DCF Intrinsic Value Formula

```
Intrinsic Value per Share =
    Σ(year 1→5) [ FCF_per_share × (1 + growth)^year / (1 + WACC)^year ]
  + Terminal Value / (1 + WACC)^5
  + max(Net Cash per Share, 0)

where:
  FCF_per_share      = Free Cash Flow / Shares Outstanding
  growth             = analyst EPS growth estimate (capped -10% to +35%)
  WACC               = 10% (config: DCF_DISCOUNT_RATE)
  Terminal Value     = FCF_year5 × (1 + 3%) / (10% - 3%)
  Net Cash per Share = (Total Cash - Total Debt) / Shares Outstanding
```

**Valuation labels:**
- Current Price < Intrinsic × 0.85 → **Undervalued** (≥15% margin of safety)
- Current Price > Intrinsic × 1.15 → **Overvalued**
- Otherwise → **Fairly Valued**

---

## Deployment

### Local
```bash
pip install -r requirements.txt
ollama serve && ollama pull llama3.2:3b
python app.py
# → http://localhost:7860
```

### HuggingFace Spaces
- Space: `irajkoohi/StocksAnalysisDashboard`
- SDK: Gradio
- App file: `app.py`
- Required secrets: `GROQ_API_KEY` (LLM), optionally `NEWS_API_KEY`, `REDDIT_CLIENT_ID/SECRET`
- LLM auto-switches to Groq when `SPACE_ID` env var detected

---

## Dependencies

| Package | Purpose |
|---|---|
| fastapi + uvicorn | REST API backend |
| gradio ≥ 4.31 | Frontend UI |
| langgraph ≥ 0.1 | Multi-agent orchestration |
| yfinance ≥ 0.2.40 | Market data |
| plotly ≥ 5.22 | Interactive charts |
| pandas / numpy | Data processing |
| groq | HF Spaces LLM fallback |
| requests | HTTP calls |
| praw | Reddit API (optional) |
| newsapi-python | NewsAPI (optional) |
| gTTS | Text-to-speech (American) |
| python-dotenv | .env file loading |
| openai-whisper | Voice input (optional, heavy) |

---

## Known Limitations & Future Improvements

| Item | Notes |
|---|---|
| Tab hot-reload | Gradio requires page refresh when adding/deleting tabs — known Gradio limitation |
| X.com sentiment | Nitter instances can be unstable; upgrade path is Twitter API v2 |
| Whisper voice | Commented out by default due to size; enable in requirements.txt |
| Options data | Not included; future: add IV, put/call ratio from yfinance options chain |
| Backtesting | Future: add signal backtesting with historical win rate per indicator |
| Real-time streaming | Future: WebSocket price streaming instead of polling |
| Portfolio P&L | Future: track cost basis and unrealized gains per owned stock |
| Alert system | Future: email/SMS alerts when RSI or price crosses thresholds |
