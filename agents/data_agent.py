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
    """Return a requests.Session with browser UA + pre-primed Yahoo Finance cookies.
    Visiting yahoo.com first sets the consent cookies needed for the crumb endpoint."""
    try:
        import requests as _req
        s = _req.Session()
        s.headers.update({
            "User-Agent": _BROWSER_UA,
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.5",
        })
        # Prime Yahoo consent cookies so the crumb fetch works from datacenter IPs
        try:
            s.get("https://finance.yahoo.com/", timeout=8)
        except Exception as e:
            logger.debug(f"Yahoo cookie prime failed (non-fatal): {e}")
        return s
    except Exception:
        return None


def _fetch_info_direct(ticker: str, session) -> dict:
    """Directly call Yahoo Finance quoteSummary with a manually fetched crumb.
    Bypasses yfinance's internal CrumbManager, which can fail on datacenter IPs
    (e.g. HF Spaces) even when stock.history() works fine."""
    try:
        crumb_r = session.get(
            "https://query2.finance.yahoo.com/v1/test/getcrumb", timeout=10
        )
        logger.info(f"[{ticker}] crumb HTTP {crumb_r.status_code}")
        if crumb_r.status_code != 200 or not crumb_r.text.strip():
            return {}
        crumb = crumb_r.text.strip()

        modules = "summaryDetail,defaultKeyStatistics,assetProfile,price,financialData"
        url = (
            f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
            f"?modules={modules}&crumb={crumb}&formatted=false"
        )
        r = session.get(url, timeout=15)
        logger.info(f"[{ticker}] quoteSummary HTTP {r.status_code}")
        if r.status_code != 200:
            return {}

        result_list = r.json().get("quoteSummary", {}).get("result", [])
        if not result_list:
            return {}

        info: dict = {}
        for module_dict in result_list[0].values():
            if not isinstance(module_dict, dict):
                continue
            for k, v in module_dict.items():
                if isinstance(v, dict):
                    info[k] = v.get("raw", v.get("fmt"))
                else:
                    info[k] = v

        logger.info(
            f"[{ticker}] direct quoteSummary: {len(info)} fields, "
            f"sector={info.get('sector')}, PE={info.get('trailingPE')}"
        )
        return info
    except Exception as e:
        logger.warning(f"[{ticker}] direct quoteSummary fallback failed: {e}")
        return {}


def _fetch_info_with_retry(stock, ticker: str, session, retries: int = 2, delay: int = 3) -> dict:
    """Fetch stock.info; falls back to direct quoteSummary call if yfinance returns
    empty or incomplete data (missing fundamental fields like sector/trailingPE)."""
    info: dict = {}
    for attempt in range(retries):
        try:
            candidate = stock.info or {}
            if candidate.get("sector") or candidate.get("trailingPE"):
                logger.info(f"[{ticker}] stock.info OK on attempt {attempt + 1}")
                return candidate
            if candidate:
                logger.warning(
                    f"[{ticker}] stock.info partial on attempt {attempt + 1} "
                    f"({len(candidate)} keys, no sector/PE)"
                )
                info = candidate  # keep partial — might be merged below
        except Exception as e:
            logger.warning(f"[{ticker}] stock.info attempt {attempt + 1}/{retries}: {e}")
        if attempt < retries - 1:
            time.sleep(delay)

    # Fallback: bypass yfinance crumb manager, call API directly
    if session:
        logger.info(f"[{ticker}] falling back to direct quoteSummary fetch")
        direct = _fetch_info_direct(ticker, session)
        if direct:
            merged = {**info, **direct}   # direct wins for fundamental fields
            return merged

    if not info:
        logger.warning(f"[{ticker}] stock.info returned empty after {retries} attempts + direct fallback")
    return info


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

        # Company info — retry via yfinance, fall back to direct API call if needed
        info = _fetch_info_with_retry(stock, ticker, session)

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
