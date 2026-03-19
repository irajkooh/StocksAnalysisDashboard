"""
agents/technical_agent.py — Technical Analysis Agent
Computes all indicators, support/resistance, Fibonacci, and pivots.
"""

import logging
from agents.state import AnalysisState
from utils.indicators import (
    get_indicator_snapshot,
    compute_support_resistance,
    compute_fibonacci,
    compute_pivots,
)
from utils.chart_builder import build_stock_chart

logger = logging.getLogger(__name__)


def technical_agent(state: AnalysisState) -> AnalysisState:
    """
    Run all technical indicators on ohlcv_df.
    Populates: indicators, supports, resistances, fibonacci, pivots, chart_json
    """
    df     = state.get("ohlcv_df")
    ticker = state.get("ticker", "")
    errors = list(state.get("errors", []))

    if df is None or df.empty:
        errors.append("TechnicalAgent: no OHLCV data available")
        return {**state, "errors": errors}

    try:
        snapshot = get_indicator_snapshot(df)

        supports     = snapshot.pop("supports", [])
        resistances  = snapshot.pop("resistances", [])
        fibonacci    = snapshot.pop("fibonacci", {})
        pivots       = snapshot.pop("pivots", {})

        # Build the chart
        fig  = build_stock_chart(df, ticker, fibonacci, supports, resistances)
        chart_json = fig.to_json()

        return {
            **state,
            "indicators":  snapshot,
            "supports":    supports,
            "resistances": resistances,
            "fibonacci":   fibonacci,
            "pivots":      pivots,
            "chart_json":  chart_json,
            "errors":      errors,
        }

    except Exception as e:
        errors.append(f"TechnicalAgent error: {str(e)}")
        logger.error(f"TechnicalAgent [{ticker}]: {e}")
        return {**state, "errors": errors}
