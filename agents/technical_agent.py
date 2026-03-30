"""agents/technical_agent.py — Technical Analysis Agent"""
import logging
import pandas as pd
import yfinance as yf
from agents.state import AnalysisState
from utils.indicators import get_indicator_snapshot

logger = logging.getLogger(__name__)


def _session_info(info: dict, df=None) -> dict:
    """Compute pre-market / regular / after-hours price, change, pct."""
    # Regular: always compute from OHLCV df so it's never missing
    reg_price = reg_change = reg_pct = None
    if df is not None and len(df) >= 2:
        last  = float(df.iloc[-1]["Close"])
        prev  = float(df.iloc[-2]["Close"])
        reg_price  = last
        reg_change = last - prev
        reg_pct    = (reg_change / prev * 100) if prev else 0

    # yfinance info may override with live price
    if info.get("regularMarketPrice"):
        reg_price  = info["regularMarketPrice"]
        reg_change = info.get("regularMarketChange",        reg_change)
        reg_pct    = info.get("regularMarketChangePercent", reg_pct)

    def _g(key):
        v = info.get(key)
        return float(v) if v is not None else None

    return {
        "regular_price":  reg_price,
        "regular_change": reg_change,
        "regular_pct":    reg_pct,
        "pre_price":      _g("preMarketPrice"),
        "pre_change":     _g("preMarketChange"),
        "pre_pct":        _g("preMarketChangePercent"),
        "post_price":     _g("postMarketPrice"),
        "post_change":    _g("postMarketChange"),
        "post_pct":       _g("postMarketChangePercent"),
    }


def technical_agent(state: AnalysisState) -> AnalysisState:
    df     = state.get("ohlcv_df")
    ticker = state.get("ticker", "")
    errors = list(state.get("errors", []))

    if df is None or df.empty:
        errors.append("TechnicalAgent: no OHLCV data available")
        return {**state, "errors": errors}

    try:
        snapshot    = get_indicator_snapshot(df)
        supports    = snapshot.pop("supports", [])
        resistances = snapshot.pop("resistances", [])
        fibonacci   = snapshot.pop("fibonacci", {})
        pivots      = snapshot.pop("pivots", {})

        # Session price info from yfinance info dict
        info         = state.get("info", {}) or {}
        session_info = _session_info(info, df)

        return {
            **state,
            "indicators":  snapshot,
            "supports":    supports,
            "resistances": resistances,
            "fibonacci":   fibonacci,
            "pivots":      pivots,
            "session_info": session_info,
            "chart_json":  "",
            "errors":      errors,
        }
    except Exception as e:
        errors.append(f"TechnicalAgent error: {str(e)}")
        logger.error(f"TechnicalAgent [{ticker}]: {e}")
        return {**state, "errors": errors}
