# 🔬 HowItWorks.md — Technical Deep Dive

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Gradio Frontend (:7860)                   │
│  Multi-tab UI · Watchlist · Chatbot · TTS · Voice · Charts  │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP REST (port 8000)
┌────────────────────────────▼────────────────────────────────┐
│                   FastAPI Backend (:8000)                    │
│   /analyze · /chat · /session · /workflow · /health         │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│              LangGraph Multi-Agent Supervisor                │
│                                                              │
│  ┌──────────┐     ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │  Data    │────▶│Technical │  │Sentiment │  │Valuation │ │
│  │  Agent   │     │  Agent   │  │  Agent   │  │  Agent   │ │
│  └──────────┘     └────┬─────┘  └────┬─────┘  └────┬─────┘ │
│                        │             │              │        │
│                        └──────┬──────┘──────────────┘        │
│                               ▼                              │
│                        ┌──────────┐                          │
│                        │  Risk    │                          │
│                        │  Agent   │                          │
│                        └────┬─────┘                          │
│                             ▼                                │
│                        ┌──────────┐                          │
│                        │Decision  │ ◀── LLM (Ollama/Groq)   │
│                        │  Agent   │                          │
│                        └──────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Analysis Pipeline

### Step 1: Data Agent (`agents/data_agent.py`)

**Input:** ticker symbol, period, interval

**Process:**
- Calls `yfinance.Ticker(symbol).history()` to fetch OHLCV (Open, High, Low, Close, Volume) data
- Fetches `yfinance.Ticker.info` for company fundamentals (PE, market cap, EPS, FCF, etc.)
- Fetches `financials` and `earnings_history` DataFrames
- Normalizes timestamps (removes timezone)
- Auto-switches to 5-minute intervals for intraday analysis (period ≤ 5d)

**Output:** `ohlcv_df`, `info`, `financials`, `earnings_hist`

---

### Step 2: Technical Agent (`agents/technical_agent.py`)

**Input:** `ohlcv_df`

**Indicators computed (`utils/indicators.py`):**

#### Moving Averages
- **SMA 20**: Short-term trend (day-trading signal)
- **SMA 50**: Medium-term trend  
- **SMA 200**: Long-term trend (golden/death cross detection)

#### RSI (Relative Strength Index) — 14-period
- Formula: `RSI = 100 - (100 / (1 + RS))` where RS = avg_gain / avg_loss
- `< 30` → Oversold (potential bounce)
- `> 70` → Overbought (potential pullback)
- Uses Wilder's exponential smoothing

#### MACD (12, 26, 9)
- MACD Line = EMA(12) - EMA(26)
- Signal Line = EMA(9) of MACD
- Histogram = MACD - Signal
- Bullish cross: MACD crosses above Signal → buy signal
- Bearish cross: MACD crosses below Signal → sell signal

#### Bollinger Bands (20, 2σ)
- Upper = SMA(20) + 2 × std
- Lower = SMA(20) - 2 × std
- Price touching lower band = potential mean-reversion buy
- Price touching upper band = overextended

#### ATR (Average True Range) — 14-period
- Measures volatility: `TR = max(H-L, |H-PrevC|, |L-PrevC|)`
- Used to set dynamic stop-loss and take-profit levels
- Stop Loss = Price - 1.5 × ATR
- Take Profit = Price + 2.5 × ATR (2.5:1.5 = 1.67 risk/reward)

#### Stochastic Oscillator (14, 3)
- `%K = (Close - Lowest Low) / (Highest High - Lowest Low) × 100`
- `%D = SMA(3) of %K`

#### Fibonacci Retracement
- Identifies swing high and swing low over the last 60 candles
- Plots 6 key retracement levels: 23.6%, 38.2%, 50%, 61.8%, 78.6%
- These are natural price magnet levels where reversals commonly occur

#### Support / Resistance Detection
- Scans price history for local minima (support) and maxima (resistance)
- Uses a 5-candle window to identify pivot lows and highs
- Clusters nearby levels within 1.5% tolerance
- Returns top 5 support and 5 resistance zones

#### Pivot Points (Classic Floor Trader)
- `P = (H + L + C) / 3`
- `R1 = 2P - L`, `R2 = P + (H-L)`, `R3 = H + 2(P-L)`
- `S1 = 2P - H`, `S2 = P - (H-L)`, `S3 = L - 2(H-P)`

**Output:** `indicators` snapshot dict, `supports`, `resistances`, `fibonacci`, `pivots`, `chart_json`

---

### Step 3: Sentiment Agent (`agents/sentiment_agent.py`)

**Sources and weights:**

| Source | Weight | Method |
|---|---|---|
| X.com / Twitter | 30% | Nitter public scraping ($TICKER cashtag search) |
| News Headlines | 30% | NewsAPI REST API (7-day window) |
| Reddit | 25% | r/wallstreetbets, r/investing, r/stocks (PRAW or public JSON) |
| SEC EDGAR | 15% | EDGAR full-text search (8-K, 10-K, 10-Q filings, 90-day window) |

**Sentiment Scoring (`utils/sentiment_scraper.py`):**
- Keyword-based NLP: 30+ positive words (beat, surge, rally…) vs 30+ negative words (miss, fall, decline…)
- Score = (positive_count - negative_count) / total_signal_words
- Range: [-1.0 (very bearish) → +1.0 (very bullish)]
- Weighted aggregate across all sources

**Output:** `sentiment` dict with per-source scores, labels, and sample headlines

---

### Step 4: Valuation Agent (`agents/valuation_agent.py`)

**DCF (Discounted Cash Flow) Intrinsic Value (`utils/intrinsic_value.py`):**

**Formula:**
```
Intrinsic Value = Σ(t=1 to N) [FCF_per_share × (1+g)^t / (1+r)^t]
                + Terminal Value / (1+r)^N
                + Net Cash per Share

where:
  FCF = Free Cash Flow (or Net Income fallback)
  g   = EPS growth rate (analyst estimate or 8% fallback)
  r   = Discount rate / WACC (default 10%)
  N   = Projection years (default 5)
  Terminal Value = FCF_year5 × (1+g_terminal) / (r - g_terminal)
  g_terminal = 3% (long-run GDP growth)
```

**Valuation Label:**
- `Intrinsic < Current × 0.85` → **Undervalued** (≥15% margin of safety)
- `Intrinsic > Current × 1.15` → **Overvalued** (premium ≥15%)
- Otherwise → **Fairly Valued**

**Fundamental Metrics extracted:** P/E, Fwd P/E, PEG, P/B, P/S, EV/EBITDA, ROE, ROA, margins, beta, 52-week range, analyst targets

---

### Step 5: Risk Agent (`agents/risk_agent.py`)

**Metrics computed:**

| Metric | Formula |
|---|---|
| Daily Volatility | `std(daily_returns)` |
| Annual Volatility | `daily_vol × √252` |
| Max Drawdown | `min((price - rolling_max) / rolling_max)` |
| Sharpe Ratio | `(avg_return - 0.05/252) / daily_vol × √252` |
| Risk Score | Composite 1-10 from vol + drawdown + beta + short interest |
| ATR Stop Loss | `current_price - 1.5 × ATR` |
| ATR Take Profit | `current_price + 2.5 × ATR` |
| Risk/Reward | `(target - price) / (price - stop)` |

---

### Step 6: Decision Agent (`agents/decision_agent.py`)

**Rule-based composite scoring (range: -10 to +10):**

| Signal | Weight | Condition |
|---|---|---|
| RSI Oversold | +2.0 | RSI < 30 |
| RSI Overbought | -2.0 | RSI > 70 |
| MACD Bullish Cross | +2.5 | MACD crosses above Signal |
| MACD Bearish Cross | -2.5 | MACD crosses below Signal |
| Above all SMAs (20>50>200) | +1.5 | Strong uptrend |
| Below all SMAs | -1.5 | Strong downtrend |
| BB Lower Band touch | +1.0 | Mean reversion signal |
| BB Upper Band touch | -1.0 | Overextended |
| DCF Undervalued ≥15% | +2.0 | Margin of safety |
| DCF Overvalued ≥15% | -1.5 | Overpriced |
| Positive Sentiment | +1.0 | Aggregate score > 0.15 |
| Negative Sentiment | -1.0 | Aggregate score < -0.15 |
| High Risk Penalty | -1.0 | Risk score ≥ 8/10 |

**Decision mapping:**
- Score ≥ 3.0 → **BUY** (high confidence)
- Score 1.0–2.9 → **BUY** (moderate)
- Score -0.9 to +0.9 → **HOLD**
- Score -1.0 to -2.9 → **SELL** (moderate)
- Score ≤ -3.0 → **SELL** (high confidence)

**Probability calculation:**
```
Profit Probability = min(75 + score × 2, 88)%  for strong BUY
Loss Probability   = 100 - Profit Probability
```

**LLM Synthesis:**
- Decision agent builds a structured prompt with all signals, indicators, valuation, and risk data
- Sends to Ollama (local) or Groq (HF Spaces)
- LLM writes a 3-paragraph professional analysis: verdict, risks, and specific trade strategy with price levels

---

## Session Persistence

- `session.json` stores: active symbols, ownership states, watchlist, refresh interval
- Created on first "Add Symbol" click
- Auto-updated on Delete and Save
- Loaded at startup to restore all tabs

---

## LLM Configuration

| Environment | LLM | Notes |
|---|---|---|
| Local | Ollama (`llama3.2:3b`) | Full offline, MPS/CUDA accelerated |
| HuggingFace Spaces | Groq (`llama3-70b-8192`) | Requires `GROQ_API_KEY` secret |

Switch models in `config.py` → `OLLAMA_MODEL` or `GROQ_MODEL`.

---

## Chart Architecture

Three-panel Plotly chart (dark theme):
1. **Main panel (55%)**: Candlesticks + Volume bars + SMA 20/50/200 + Bollinger Bands + Fibonacci levels + Support/Resistance zones
2. **RSI panel (22%)**: RSI line with colored overbought/oversold zones
3. **MACD panel (23%)**: MACD line + Signal line + Histogram bars (green/red)

All panels share the X-axis (time) and use unified hover for easy cross-panel reading.
