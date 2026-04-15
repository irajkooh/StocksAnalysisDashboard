"""
agents/data_agent.py — Data Agent
Fetches OHLCV price data, company info, and financials via yfinance.
"""

import time
import logging
from agents.state import AnalysisState
from utils.config import DEFAULT_PERIOD, DEFAULT_INTERVAL, INTRADAY_INTERVAL

logger = logging.getLogger(__name__)

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def _make_yf_session():
    """Return a requests.Session with a browser User-Agent.
    Prevents Yahoo Finance from blocking datacenter IPs (e.g. HF Spaces)."""
    try:
        import requests as _req
        s = _req.Session()
        s.headers["User-Agent"] = _BROWSER_UA
        return s
    except Exception:
        return None


def _fetch_info_with_retry(stock, retries=3, delay=3) -> dict:
    """Fetch stock.info with retries; returns {} on persistent failure."""
    for attempt in range(retries):
        try:
            info = stock.info
            if info:
                return info
        except Exception as e:
            logger.warning(f"stock.info attempt {attempt + 1}/{retries} failed: {e}")
        if attempt < retries - 1:
            time.sleep(delay)
            delay *= 2
    return {}


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
        session = _make_yf_session()
        stock = yf.Ticker(ticker, session=session) if session else yf.Ticker(ticker)

        # OHLCV — switch to intraday if short period requested
        if period in ("1d", "5d"):
            interval = INTRADAY_INTERVAL
        df = _fetch_with_retry(stock, period, interval)

        if df.empty:
            errors.append(f"No OHLCV data returned for {ticker}")
            return {**state, "errors": errors}

        df.index = df.index.tz_localize(None)
        df = df.dropna(subset=["Close"])

        # Company info — use retry helper to survive transient Yahoo Finance blocks
        info = _fetch_info_with_retry(stock)
        if not info:
            logger.warning(f"stock.info returned empty for {ticker} after retries")

        # 1-min bars with prepost=True — walk ALL bars so every session's
        # last price is captured independently (pre, regular, post, overnight).
        # Only looking at the last bar would miss e.g. pre-market during regular hours.
        try:
            df_ext = stock.history(period="1d", interval="1m", prepost=True)
            if df_ext is not None and not df_ext.empty:
                pre_last = reg_last = post_last = ovn_last = None
                for bar_ts in df_ext.index:
                    bh, bm = bar_ts.hour, bar_ts.minute
                    p = float(df_ext.loc[bar_ts, "Close"])
                    if bh < 9 or (bh == 9 and bm < 30):
                        pre_last  = p   # Pre-market:  4:00–9:30 AM ET
                    elif (bh == 9 and bm >= 30) or (10 <= bh < 16):
                        reg_last  = p   # Regular:     9:30 AM–4:00 PM ET
                    elif 16 <= bh < 20:
                        post_last = p   # After-hours: 4:00–8:00 PM ET
                    elif bh >= 20:
                        ovn_last  = p   # Overnight:   8:00 PM+ ET
                ts = df_ext.index[-1]
                h, m = ts.hour, ts.minute
                info["_ext_last_price"] = float(df_ext["Close"].iloc[-1])
                info["_ext_last_time"]  = f"{h % 12 or 12}:{m:02d} {'AM' if h < 12 else 'PM'} ET"
                # Always set reg so _session_info knows we have live data
                info["_reg_last_price"] = reg_last
                if pre_last  is not None: info["_pre_last_price"]  = pre_last
                if post_last is not None: info["_post_last_price"] = post_last
                if ovn_last  is not None: info["_ovn_last_price"]  = ovn_last
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
