---
title: StocksAnalysisDashboard
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.25.0"
app_file: app.py
pinned: true
license: mit
---

# 📊 StocksAnalysisDashboard

Multi-tab AI-powered stock analysis dashboard with LangGraph multi-agent system.

## Features
- Multi-tab dashboard (one tab per stock symbol)
- LangGraph multi-agent: Data → Technical → Sentiment → Valuation → Risk → Decision
- Candlestick charts with SMA 20/50/200, Bollinger Bands, Fibonacci retracements
- RSI, MACD, ATR, Stochastic indicators
- DCF intrinsic value calculation
- Multi-source sentiment: NewsAPI + Reddit + SEC EDGAR + X.com
- Buy / Hold / Sell recommendation with probability scores
- Per-tab AI chatbot with memory
- TTS read button + voice input
- Session persistence (save/load JSON)
- Watchlist sidebar + auto-refresh

## LLM Setup
| Environment | Model | Notes |
|---|---|---|
| HF Spaces (primary) | Groq `llama-3.1-8b-instant` | Free: 500K tokens/day — requires `GROQ_API_KEY` secret |
| HF Spaces (fallback) | HF `meta-llama/Llama-3.2-3B-Instruct` | Free with `HF_TOKEN` — auto-used when Groq rate-limit hit |
| Local | Ollama `llama3.2:3b` | Runs fully offline |

## Environment Variables (set in HF Spaces Secrets)
| Variable | Description |
|---|---|
| `GROQ_API_KEY` | **Required** — Groq API key for Llama 3.1 8B (free at console.groq.com) |
| `HF_TOKEN` | **Required** — HuggingFace token for Llama 3.2 3B fallback |
| `NEWS_API_KEY` | NewsAPI key for headlines |
| `REDDIT_CLIENT_ID` | Reddit app client ID |
| `REDDIT_CLIENT_SECRET` | Reddit app client secret |
