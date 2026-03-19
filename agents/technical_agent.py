"""agents/technical_agent.py — Technical Analysis Agent + extended-hours data"""
import logging
import yfinance as yf
from agents.state import AnalysisState
from utils.indicators import get_indicator_snapshot
from utils.chart_builder import build_stock_chart

logger = logging.getLogger(__name__)

def technical_agent(state: AnalysisState) -> AnalysisState:
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

        # Try to fetch extended-hours data (pre/post market, 60-day window)
        df_ext = None
        try:
            stock  = yf.Ticker(ticker)
            df_ext = stock.history(period="5d", interval="1m", prepost=True)
            if df_ext is not None and not df_ext.empty:
                df_ext.index = df_ext.index.tz_localize(None)
                # Keep only pre/post market rows (outside 9:30–16:00 ET)
                import pandas as pd
                df_ext = df_ext[
                    (df_ext.index.time < __import__('datetime').time(9, 30)) |
                    (df_ext.index.time > __import__('datetime').time(16, 0))
                ]
        except Exception as e:
            logger.debug(f"Extended hours fetch skipped: {e}")
            df_ext = None

        fig = build_stock_chart(df, ticker, fibonacci, supports, resistances, df_ext)
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
