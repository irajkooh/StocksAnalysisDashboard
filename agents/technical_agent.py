"""agents/technical_agent.py — Technical Analysis Agent"""
import logging
import pandas as pd
import yfinance as yf
from agents.state import AnalysisState
from utils.indicators import get_indicator_snapshot
from utils.chart_builder import build_plotly_chart

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

    pre_p  = _g("preMarketPrice")
    post_p = _g("postMarketPrice")

    # prev_close: best reference for computing extended-hours changes
    prev_close = _g("regularMarketPreviousClose") or (
        float(df.iloc[-2]["Close"]) if df is not None and len(df) >= 2 else None
    )

    # _ext_last_price: injected from history(period="1d", interval="1m", prepost=True).
    # yfinance postMarketPrice becomes None after 8 PM ET; this covers the gap.
    ext_last = _g("_ext_last_price")
    if post_p is None and pre_p is None and ext_last is not None:
        ref = prev_close or reg_price
        if ref and abs(ext_last - ref) > 0.001:
            post_p = ext_last

    # price_time: timestamp of the last traded bar (from 1-min prepost history)
    price_time = info.get("_ext_last_time")

    # Always compute changes dynamically so they work even when yfinance fields are absent
    def _chg(p, ref):
        if p is not None and ref and ref != 0:
            c = p - ref
            return c, c / ref * 100
        return None, None

    ref          = prev_close or reg_price
    pre_change,  pre_pct  = _chg(pre_p,  ref)
    post_change, post_pct = _chg(post_p, ref)

    # Overnight gap pill: most recent extended-hours price vs prev close
    ext_p = pre_p if pre_p is not None else post_p
    ovn_change, ovn_pct = _chg(ext_p, prev_close)
    if ext_p is None or ovn_change is None:
        ext_p = ovn_change = ovn_pct = None

    return {
        "regular_price":    reg_price,
        "regular_change":   reg_change,
        "regular_pct":      reg_pct,
        "pre_price":        pre_p,
        "pre_change":       pre_change,
        "pre_pct":          pre_pct,
        "post_price":       post_p,
        "post_change":      post_change,
        "post_pct":         post_pct,
        "overnight_price":  ext_p,
        "overnight_change": ovn_change,
        "overnight_pct":    ovn_pct,
        "prev_close":       prev_close,
        "price_time":       price_time,
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

        try:
            chart_html = build_plotly_chart(df, ticker)
        except Exception as ce:
            logger.warning(f"TechnicalAgent [{ticker}]: chart build failed: {ce}")
            chart_html = ""

        return {
            **state,
            "indicators":  snapshot,
            "supports":    supports,
            "resistances": resistances,
            "fibonacci":   fibonacci,
            "pivots":      pivots,
            "session_info": session_info,
            "chart_json":  chart_html,
            "errors":      errors,
        }
    except Exception as e:
        errors.append(f"TechnicalAgent error: {str(e)}")
        logger.error(f"TechnicalAgent [{ticker}]: {e}")
        return {**state, "errors": errors}
