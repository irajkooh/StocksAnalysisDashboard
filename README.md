---
title: StocksAnalysisDashboard
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.31.0"
app_file: app.py
pinned: false
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

## Environment Variables (set in HF Spaces Secrets)
| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key (LLM on HF Spaces) |
| `NEWS_API_KEY` | NewsAPI key for headlines |
| `REDDIT_CLIENT_ID` | Reddit app client ID |
| `REDDIT_CLIENT_SECRET` | Reddit app client secret |
| `HF_TOKEN` | HuggingFace token |
