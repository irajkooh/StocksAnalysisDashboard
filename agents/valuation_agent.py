"""
agents/valuation_agent.py — Valuation Agent
Runs DCF intrinsic value calculation and extracts fundamental metrics.
"""

import logging
from agents.state import AnalysisState
from utils.intrinsic_value import dcf_intrinsic_value, get_fundamental_metrics

logger = logging.getLogger(__name__)


def valuation_agent(state: AnalysisState) -> AnalysisState:
    """
    Compute DCF intrinsic value and extract key fundamental ratios.
    Populates: dcf, fundamentals
    """
    info   = state.get("info", {})
    ticker = state.get("ticker", "")
    errors = list(state.get("errors", []))

    try:
        dcf_result   = dcf_intrinsic_value(info)
        fundamentals = get_fundamental_metrics(info)
        return {**state, "dcf": dcf_result, "fundamentals": fundamentals, "errors": errors}
    except Exception as e:
        errors.append(f"ValuationAgent error: {str(e)}")
        logger.error(f"ValuationAgent [{ticker}]: {e}")
        return {**state, "dcf": {}, "fundamentals": {}, "errors": errors}
