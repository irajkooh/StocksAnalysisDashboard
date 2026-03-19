"""
agents/decision_agent.py — Decision Agent
Combines all signals into a final Buy/Hold/Sell recommendation
using a rule-based scoring model + LLM narrative synthesis.
"""

import json
import logging
from typing import Dict, Tuple
from agents.state import AnalysisState
from config import (
    LLM_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT,
    GROQ_API_KEY, GROQ_MODEL,
    RSI_OVERSOLD, RSI_OVERBOUGHT, UNDERVALUED_THRESHOLD, OVERVALUED_THRESHOLD
)

logger = logging.getLogger(__name__)


# ─── Rule-Based Scoring ───────────────────────────────────────────────────────

def _score_signals(state: AnalysisState) -> Tuple[float, list]:
    """
    Returns (composite_score [-10, +10], reasons[])
    Positive → bullish, Negative → bearish.
    """
    ind   = state.get("indicators", {})
    dcf   = state.get("dcf", {})
    sent  = state.get("sentiment", {})
    risk  = state.get("risk", {})
    fund  = state.get("fundamentals", {})
    owns  = state.get("owns_stock", False)

    score   = 0.0
    reasons = []

    # ── RSI ────────────────────────────────────────────────────────────────
    rsi = ind.get("rsi", 50)
    if rsi < RSI_OVERSOLD:
        score += 2.0
        reasons.append(f"RSI {rsi:.1f} — oversold (bullish reversal signal)")
    elif rsi > RSI_OVERBOUGHT:
        score -= 2.0
        reasons.append(f"RSI {rsi:.1f} — overbought (pullback risk)")
    else:
        reasons.append(f"RSI {rsi:.1f} — neutral territory")

    # ── MACD ───────────────────────────────────────────────────────────────
    cross = ind.get("macd_cross", "")
    macd  = ind.get("macd", 0)
    sig   = ind.get("macd_signal", 0)
    if cross == "Bullish Cross":
        score += 2.5
        reasons.append("MACD bullish crossover — momentum turning positive")
    elif cross == "Bearish Cross":
        score -= 2.5
        reasons.append("MACD bearish crossover — momentum turning negative")
    elif macd > sig:
        score += 0.5
        reasons.append("MACD above signal line — mild bullish momentum")
    else:
        score -= 0.5
        reasons.append("MACD below signal line — mild bearish momentum")

    # ── SMA Trend ──────────────────────────────────────────────────────────
    price  = ind.get("price", 0)
    sma20  = ind.get("sma_20", 0)
    sma50  = ind.get("sma_50", 0)
    sma200 = ind.get("sma_200", 0)
    if price > sma20 > sma50 > sma200:
        score += 1.5
        reasons.append("Price above all SMAs (20>50>200) — strong uptrend")
    elif price < sma20 < sma50 < sma200:
        score -= 1.5
        reasons.append("Price below all SMAs — strong downtrend")
    elif price > sma50:
        score += 0.5
        reasons.append("Price above SMA-50 — medium-term uptrend")

    # ── Bollinger Bands ────────────────────────────────────────────────────
    bb_low  = ind.get("bb_lower", 0)
    bb_high = ind.get("bb_upper", 0)
    if price <= bb_low:
        score += 1.0
        reasons.append("Price at/below Bollinger lower band — mean-reversion buy signal")
    elif price >= bb_high:
        score -= 1.0
        reasons.append("Price at/above Bollinger upper band — overextended")

    # ── Intrinsic Value / DCF ──────────────────────────────────────────────
    prem = dcf.get("premium_discount")
    if prem is not None:
        if prem <= UNDERVALUED_THRESHOLD * 100:
            score += 2.0
            reasons.append(f"DCF shows {abs(prem):.1f}% undervalued vs intrinsic value")
        elif prem >= OVERVALUED_THRESHOLD * 100:
            score -= 1.5
            reasons.append(f"DCF shows {prem:.1f}% overvalued vs intrinsic value")

    # ── Sentiment ──────────────────────────────────────────────────────────
    agg_sent = sent.get("aggregate_score", 0)
    if agg_sent > 0.15:
        score += 1.0
        reasons.append(f"Aggregate sentiment positive ({sent.get('aggregate_label','')})")
    elif agg_sent < -0.15:
        score -= 1.0
        reasons.append(f"Aggregate sentiment negative ({sent.get('aggregate_label','')})")

    # ── Risk Penalty ───────────────────────────────────────────────────────
    rsk = risk.get("risk_score", 5)
    if rsk >= 8:
        score -= 1.0
        reasons.append(f"High risk score ({rsk}/10) — increased caution warranted")

    # ── Ownership modifier ─────────────────────────────────────────────────
    if owns:
        reasons.append("⚠️ You own this stock — hold/sell logic applied with position context")

    return score, reasons


def _score_to_decision(score: float, owns: bool) -> Dict:
    if score >= 3.0:
        action, prob_profit, prob_loss = "BUY 🟢",  min(75 + score * 2, 88), max(12, 25 - score * 2)
    elif score >= 1.0:
        action, prob_profit, prob_loss = "BUY 🟢",  min(60 + score * 3, 75), max(25, 40 - score * 3)
    elif score <= -3.0:
        action, prob_profit, prob_loss = "SELL 🔴", max(12, 25 + score * 2), min(88, 75 - score * 2)
    elif score <= -1.0:
        action, prob_profit, prob_loss = "SELL 🔴", max(25, 40 + score * 3), min(75, 60 - score * 3)
    else:
        action, prob_profit, prob_loss = "HOLD 🟡", 50, 50

    if owns and action == "SELL 🔴":
        action = "SELL / REDUCE 🔴"
    if owns and action == "HOLD 🟡":
        action = "HOLD (You Own) 🟡"

    confidence = min(95, int(abs(score) / 10 * 100 + 50))

    return {
        "action":          action,
        "score":           round(score, 2),
        "confidence":      confidence,
        "probability_profit": round(prob_profit, 1),
        "probability_loss":   round(prob_loss, 1),
    }


# ─── LLM Narrative ───────────────────────────────────────────────────────────

def _build_llm_prompt(state: AnalysisState, decision: Dict, reasons: list) -> str:
    ticker = state.get("ticker", "")
    price  = state.get("indicators", {}).get("price", 0)
    dcf    = state.get("dcf", {})
    fund   = state.get("fundamentals", {})
    risk   = state.get("risk", {})
    sent   = state.get("sentiment", {})
    owns   = state.get("owns_stock", False)

    return f"""You are a professional equity analyst writing a concise day-trading analysis report.

STOCK: {ticker} | LAST PRICE: ${price:.2f}
OWNS STOCK: {"Yes" % owns if owns else "No"}

DECISION: {decision['action']}
COMPOSITE SCORE: {decision['score']}/10
CONFIDENCE: {decision['confidence']}%
PROFIT PROBABILITY: {decision['probability_profit']}% | LOSS PROBABILITY: {decision['probability_loss']}%

KEY SIGNALS:
{chr(10).join(f"• {r}" for r in reasons)}

TECHNICAL SNAPSHOT:
- RSI: {state.get('indicators', {}).get('rsi', 'N/A'):.1f}  ({state.get('indicators', {}).get('rsi_state', '')})
- MACD Cross: {state.get('indicators', {}).get('macd_cross', 'N/A')}
- ATR: ${state.get('indicators', {}).get('atr', 0):.2f}
- Stochastic K/D: {state.get('indicators', {}).get('stoch_k', 0):.1f} / {state.get('indicators', {}).get('stoch_d', 0):.1f}

VALUATION:
- DCF Intrinsic Value: ${dcf.get('intrinsic_value', 'N/A')}
- Premium/Discount: {dcf.get('premium_discount', 'N/A')}%
- PE Ratio: {fund.get('pe_ratio', 'N/A')}  |  Forward PE: {fund.get('forward_pe', 'N/A')}
- Beta: {fund.get('beta', 'N/A')}

RISK:
- Annual Volatility: {risk.get('annual_volatility', 'N/A')}%
- Max Drawdown: {risk.get('max_drawdown', 'N/A')}%
- Sharpe Ratio: {risk.get('sharpe_ratio', 'N/A')}
- Risk Score: {risk.get('risk_score', 'N/A')}/10

SENTIMENT: {sent.get('aggregate_label', 'N/A')} (score: {sent.get('aggregate_score', 0):.2f})

Write a 3-paragraph professional trading analysis:
1. MARKET POSITION & VERDICT — explain the {decision['action']} recommendation with key reasons
2. RISK FACTORS — what could go wrong, stop-loss considerations
3. TRADING STRATEGY — specific entry/exit levels based on support/resistance, timeframe

Be direct, data-driven, and concise. No preamble. Use $ amounts for price levels."""


def _call_llm(prompt: str) -> str:
    try:
        if LLM_PROVIDER == "ollama":
            import requests
            r = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                timeout=OLLAMA_TIMEOUT,
            )
            if r.status_code == 200:
                return r.json().get("response", "").strip()
        elif LLM_PROVIDER == "groq":
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            resp   = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
            )
            return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
    return "LLM analysis unavailable. Please check your Ollama/Groq configuration."


# ─── Agent Entry Point ────────────────────────────────────────────────────────

def decision_agent(state: AnalysisState) -> AnalysisState:
    """
    Produce final trade recommendation.
    Populates: decision, llm_summary, llm_chatbot_ctx
    """
    ticker = state.get("ticker", "")
    errors = list(state.get("errors", []))
    owns   = state.get("owns_stock", False)

    try:
        score, reasons = _score_signals(state)
        decision       = _score_to_decision(score, owns)
        decision["reasons"] = reasons

        # Build LLM narrative
        prompt      = _build_llm_prompt(state, decision, reasons)
        llm_summary = _call_llm(prompt)

        # Chatbot context (used as system message)
        llm_ctx = (
            f"You are an AI assistant analyzing {ticker} stock. "
            f"Current recommendation: {decision['action']}. "
            f"Last price: ${state.get('indicators', {}).get('price', 0):.2f}. "
            f"DCF intrinsic value: ${state.get('dcf', {}).get('intrinsic_value', 'N/A')}. "
            f"Key signals: {'; '.join(reasons[:5])}. "
            f"Answer questions about this stock in a professional, data-driven manner."
        )

        return {
            **state,
            "decision":        decision,
            "llm_summary":     llm_summary,
            "llm_chatbot_ctx": llm_ctx,
            "errors":          errors,
        }

    except Exception as e:
        errors.append(f"DecisionAgent error: {str(e)}")
        logger.error(f"DecisionAgent [{ticker}]: {e}")
        return {**state, "decision": {}, "llm_summary": "", "errors": errors}
