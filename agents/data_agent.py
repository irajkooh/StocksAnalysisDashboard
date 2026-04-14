"""
agents/data_agent.py — Data Agent
Fetches OHLCV price data, company info, and financials via yfinance.
"""

import time
import logging
from agents.state import AnalysisState
from utils.config import DEFAULT_PERIOD, DEFAULT_INTERVAL, INTRADAY_INTERVAL

logger = logging.getLogger(__name__)


def _fetch_with_retry(stock, period, interval, retries=3, delay=5):
    last_exc = None
    for attempt in range(retries):
        try:
            df = stock.history(period=period, interval=interval, auto_adjust=True)
            if not df.empty:
                return df
        except Exception as e:
            last_exc = e
            logger.warning(f"yfinance error (attempt {attempt+1}/{retries}): {e}")
        if attempt < retries - 1:
            logger.warning(f"Retrying in {delay}s…")
            time.sleep(delay)
            delay *= 2
    if last_exc:
        raise last_exc
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

        # 1-min bars with prepost=True → last Close = most recent traded price at any hour.
        # postMarketPrice in stock.info becomes None after 8 PM ET; this covers that gap.
        try:
            df_ext = stock.history(period="1d", interval="1m", prepost=True)
            if df_ext is not None and not df_ext.empty:
                info["_ext_last_price"] = float(df_ext["Close"].iloc[-1])
                ts = df_ext.index[-1]
                try:
                    h, m = ts.hour, ts.minute
                    ampm = "AM" if h < 12 else "PM"
                    h12  = h % 12 or 12
                    info["_ext_last_time"] = f"{h12}:{m:02d} {ampm} ET"
                    # Classify last bar into its trading session so _session_info
                    # shows the correct live price pill without needing yfinance info fields.
                    p = info["_ext_last_price"]
                    if h < 9 or (h == 9 and m < 30):
                        info["_pre_last_price"]  = p  # Pre-market:  4:00–9:30 AM ET
                    elif (h == 9 and m >= 30) or (10 <= h < 16):
                        info["_reg_last_price"]  = p  # Regular:     9:30 AM–4:00 PM ET
                    elif 16 <= h < 20:
                        info["_post_last_price"] = p  # After-hours: 4:00–8:00 PM ET
                    elif h >= 20:
                        info["_ovn_last_price"]  = p  # Overnight:   8:00 PM+ ET
                except Exception:
                    pass
        except Exception:
            pass

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
