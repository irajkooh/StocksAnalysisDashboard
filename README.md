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

## Environment Variables (set in HF Spaces Secrets)
| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key — primary LLM (`llama-3.1-8b-instant`, 500K tokens/day free) |
| `HF_TOKEN` | HuggingFace token — fallback LLM when Groq rate-limits (`Llama-3.2-3B-Instruct`) |
