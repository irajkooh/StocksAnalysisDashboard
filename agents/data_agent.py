"""
agents/data_agent.py — Data Agent
Fetches OHLCV price data, company info, and financials via yfinance.
"""

import time
import logging
from agents.state import AnalysisState
from config import DEFAULT_PERIOD, DEFAULT_INTERVAL, INTRADAY_INTERVAL

logger = logging.getLogger(__name__)


def _fetch_with_retry(stock, period, interval, retries=3, delay=5):
    for attempt in range(retries):
        df = stock.history(period=period, interval=interval, auto_adjust=True)
        if not df.empty:
            return df
        if attempt < retries - 1:
            logger.warning(f"yfinance empty/rate-limited, retrying in {delay}s…")
            time.sleep(delay)
            delay *= 2
    return df


def data_agent(state: AnalysisState) -> AnalysisState:
    """
    Fetch raw market data for the given ticker.
    Populates: ohlcv_df, info, financials, earnings_hist
    """
    ticker  = state.get("ticker", "").upper().strip()
    period  = state.get("period",   DEFAULT_PERIOD)
    interval = state.get("interval", DEFAULT_INTERVAL)
    errors  = list(state.get("errors", []))

    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)

        # OHLCV — switch to intraday if short period requested
        if period in ("1d", "5d"):
            interval = INTRADAY_INTERVAL
        df = _fetch_with_retry(stock, period, interval)

        if df.empty:
            errors.append(f"No OHLCV data returned for {ticker}")
            return {**state, "errors": errors}

        df.index = df.index.tz_localize(None)
        df = df.dropna(subset=["Close"])

        # Company info
        try:
            info = stock.info or {}
        except Exception:
            info = {}

        # Financials
        try:
            financials = stock.financials
        except Exception:
            financials = None

        # Earnings history
        try:
            earnings_hist = stock.earnings_history
        except Exception:
            earnings_hist = None

        return {
            **state,
            "ohlcv_df":     df,
            "info":         info,
            "financials":   financials,
            "earnings_hist": earnings_hist,
            "errors":       errors,
        }

    except Exception as e:
        errors.append(f"DataAgent error: {str(e)}")
        logger.error(f"DataAgent [{ticker}]: {e}")
        return {**state, "errors": errors}
