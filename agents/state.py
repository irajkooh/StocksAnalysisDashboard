"""
agents/state.py — Shared LangGraph State
Typed dict passed through every agent node in the analysis graph.
"""

from typing import Any, Dict, List, Optional, TypedDict


class AnalysisState(TypedDict, total=False):
    # ── Inputs ────────────────────────────────────────────────────────────────
    ticker:          str
    owns_stock:      bool
    period:          str
    interval:        str

    # ── Raw Data (from DataAgent) ──────────────────────────────────────────────
    ohlcv_df:        Any    # pd.DataFrame
    info:            Dict   # yfinance .info dict
    financials:      Any    # pd.DataFrame
    earnings_hist:   Any    # pd.DataFrame

    # ── Technical Analysis (from TechnicalAgent) ──────────────────────────────
    indicators:      Dict
    supports:        List[float]
    resistances:     List[float]
    fibonacci:       Dict[str, float]
    pivots:          Dict[str, float]

    # ── Sentiment (from SentimentAgent) ───────────────────────────────────────
    sentiment:       Dict

    # ── Valuation (from ValuationAgent) ───────────────────────────────────────
    dcf:             Dict
    fundamentals:    Dict

    # ── Risk (from RiskAgent) ─────────────────────────────────────────────────
    risk:            Dict

    # ── Decision (from DecisionAgent) ─────────────────────────────────────────
    decision:        Dict    # {action, confidence, reasons, probability_profit, probability_loss}
    llm_summary:     str
    llm_chatbot_ctx: str

    # ── Chart data ────────────────────────────────────────────────────────────
    chart_json:      str    # plotly fig.to_json()

    # ── Errors / Metadata ─────────────────────────────────────────────────────
    errors:          List[str]
    duration_ms:     float
