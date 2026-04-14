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
    # Regular: always compute from OHLCV df so it's never missing.
    # prev_ohlcv = yesterday's close — kept separate so the _reg_last_price
    # override below always diffs against the correct prior-day close.
    reg_price = reg_change = reg_pct = None
    prev_ohlcv = None
    if df is not None and len(df) >= 2:
        last  = float(df.iloc[-1]["Close"])
        prev  = float(df.iloc[-2]["Close"])
        prev_ohlcv = prev
        reg_price  = last
        reg_change = last - prev
        reg_pct    = (reg_change / prev * 100) if prev else 0

    # yfinance info may override with live price
    if info.get("regularMarketPrice"):
        reg_price  = info["regularMarketPrice"]
        reg_change = info.get("regularMarketChange",        reg_change)
        reg_pct    = info.get("regularMarketChangePercent", reg_pct)
    elif info.get("_reg_last_price"):
        # Use prev_ohlcv (yesterday's OHLCV close) as reference, not reg_price,
        # which may already equal today's intraday bar — causing change = 0.
        ref = prev_ohlcv or reg_price
        reg_price  = info["_reg_last_price"]
        if ref and ref != 0:
            reg_change = reg_price - ref
            reg_pct    = reg_change / ref * 100

    def _g(key):
        v = info.get(key)
        return float(v) if v is not None else None

    # Prefer freshly-fetched 1-min bar prices (_pre/_post_last_price) over
    # yfinance stock.info fields, which can be stale (e.g. yesterday's pre-market close).
    pre_p  = _g("_pre_last_price")  or _g("preMarketPrice")
    post_p = _g("_post_last_price") or _g("postMarketPrice")
    ovn_p  = _g("_ovn_last_price")

    # prev_close: best reference for computing extended-hours changes
    prev_close = _g("regularMarketPreviousClose") or (
        float(df.iloc[-2]["Close"]) if df is not None and len(df) >= 2 else None
    )

    # _ext_last_price: injected from history(period="1d", interval="1m", prepost=True).
    # yfinance postMarketPrice becomes None after 8 PM ET; this covers the gap.
    # Only activate when NOT in regular session (i.e. _reg_last_price absent), otherwise
    # the regular-session live price would be mislabelled as "After-Hours".
    ext_last = _g("_ext_last_price")
    if post_p is None and pre_p is None and not info.get("_reg_last_price") and ext_last is not None:
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

    # Overnight pill: only show when actual overnight bars (8 PM ET+) exist
    ovn_change, ovn_pct = _chg(ovn_p, prev_close)
    if ovn_p is None or ovn_change is None:
        ovn_p = ovn_change = ovn_pct = None

    # Determine the currently active session (most recent one with data).
    # Priority: overnight > post > regular > pre — higher sessions overwrite earlier ones.
    if info.get("_ovn_last_price"):
        current_session = "overnight"
    elif info.get("_post_last_price"):
        current_session = "post"
    elif info.get("_reg_last_price"):
        current_session = "regular"
    elif info.get("_pre_last_price"):
        current_session = "pre"
    else:
        current_session = "regular"

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
        "overnight_price":  ovn_p,
        "overnight_change": ovn_change,
        "overnight_pct":    ovn_pct,
        "prev_close":       prev_close,
        "price_time":       price_time,
        "current_session":  current_session,
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
