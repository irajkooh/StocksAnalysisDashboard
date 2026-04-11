"""
utils/indicators.py — Technical Analysis Calculations
Computes SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic,
Fibonacci retracements, support/resistance, and pivot points.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from utils.config import (
    SMA_PERIODS, RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD, ATR_PERIOD, FIB_LOOKBACK_DAYS,
    RSI_OVERSOLD, RSI_OVERBOUGHT
)


# ─── Moving Averages ──────────────────────────────────────────────────────────

def compute_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=1).mean()


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_all_smas(df: pd.DataFrame) -> pd.DataFrame:
    for p in SMA_PERIODS:
        df[f"SMA_{p}"] = compute_sma(df["Close"], p)
    return df


# ─── RSI ──────────────────────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


# ─── MACD ─────────────────────────────────────────────────────────────────────

def compute_macd(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast   = compute_ema(series, MACD_FAST)
    ema_slow   = compute_ema(series, MACD_SLOW)
    macd_line  = ema_fast - ema_slow
    signal     = compute_ema(macd_line, MACD_SIGNAL)
    histogram  = macd_line - signal
    return macd_line, signal, histogram


# ─── Bollinger Bands ──────────────────────────────────────────────────────────

def compute_bollinger_bands(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid  = compute_sma(series, BB_PERIOD)
    std  = series.rolling(window=BB_PERIOD, min_periods=1).std()
    upper = mid + BB_STD * std
    lower = mid - BB_STD * std
    return upper, mid, lower


# ─── ATR ──────────────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


# ─── Stochastic Oscillator ────────────────────────────────────────────────────

def compute_stochastic(df: pd.DataFrame, k: int = 14, d: int = 3) -> Tuple[pd.Series, pd.Series]:
    low_min  = df["Low"].rolling(k).min()
    high_max = df["High"].rolling(k).max()
    stoch_k  = 100 * (df["Close"] - low_min) / (high_max - low_min + 1e-9)
    stoch_d  = stoch_k.rolling(d).mean()
    return stoch_k.fillna(50), stoch_d.fillna(50)


# ─── Fibonacci Retracement ────────────────────────────────────────────────────

def compute_fibonacci(df: pd.DataFrame, lookback: int = FIB_LOOKBACK_DAYS) -> Dict[str, float]:
    """
    Identify swing high and swing low over lookback period,
    then calculate Fibonacci retracement levels.
    """
    recent = df.tail(lookback)
    swing_high = recent["High"].max()
    swing_low  = recent["Low"].min()
    diff       = swing_high - swing_low

    levels = {
        "0.0%  (Low)":  swing_low,
        "23.6%":        swing_low + 0.236 * diff,
        "38.2%":        swing_low + 0.382 * diff,
        "50.0%":        swing_low + 0.500 * diff,
        "61.8%":        swing_low + 0.618 * diff,
        "78.6%":        swing_low + 0.786 * diff,
        "100% (High)":  swing_high,
    }
    return levels


# ─── Support / Resistance ─────────────────────────────────────────────────────

def compute_support_resistance(df: pd.DataFrame, n_levels: int = 5) -> Tuple[List[float], List[float]]:
    """
    Detect support and resistance zones using local minima/maxima
    and volume-weighted price clustering.
    """
    closes = df["Close"].values
    highs  = df["High"].values
    lows   = df["Low"].values
    volumes= df["Volume"].values if "Volume" in df.columns else np.ones(len(closes))

    # Local extrema detection with window=5
    window = 5
    resistance_raw, support_raw = [], []
    for i in range(window, len(closes) - window):
        if highs[i] == max(highs[i-window:i+window+1]):
            resistance_raw.append(highs[i])
        if lows[i] == min(lows[i-window:i+window+1]):
            support_raw.append(lows[i])

    def cluster_levels(raw: List[float], n: int) -> List[float]:
        if not raw:
            return []
        arr = np.array(sorted(raw))
        tolerance = arr.mean() * 0.015  # 1.5% clustering tolerance
        clusters = []
        current = [arr[0]]
        for v in arr[1:]:
            if v - current[-1] <= tolerance:
                current.append(v)
            else:
                clusters.append(np.mean(current))
                current = [v]
        clusters.append(np.mean(current))
        return sorted(clusters)[-n:]

    supports    = cluster_levels(support_raw, n_levels)
    resistances = cluster_levels(resistance_raw, n_levels)
    return supports, resistances


# ─── Pivot Points ─────────────────────────────────────────────────────────────

def compute_pivots(df: pd.DataFrame) -> Dict[str, float]:
    """Classic floor trader pivot points from most recent completed candle."""
    last = df.iloc[-2] if len(df) >= 2 else df.iloc[-1]
    H, L, C = last["High"], last["Low"], last["Close"]
    P  = (H + L + C) / 3
    R1 = 2 * P - L
    S1 = 2 * P - H
    R2 = P + (H - L)
    S2 = P - (H - L)
    R3 = H + 2 * (P - L)
    S3 = L - 2 * (H - P)
    return {"P": P, "R1": R1, "R2": R2, "R3": R3,
            "S1": S1, "S2": S2, "S3": S3}


# ─── Full Indicator Bundle ────────────────────────────────────────────────────

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Run all indicators and attach as columns to df."""
    df = df.copy()
    df = compute_all_smas(df)

    df["RSI"]          = compute_rsi(df["Close"])
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = compute_macd(df["Close"])
    df["BB_Upper"], df["BB_Mid"], df["BB_Lower"]   = compute_bollinger_bands(df["Close"])
    df["ATR"]          = compute_atr(df)
    df["Stoch_K"], df["Stoch_D"] = compute_stochastic(df)

    return df


# ─── Summary Snapshot ─────────────────────────────────────────────────────────

def get_indicator_snapshot(df: pd.DataFrame) -> Dict:
    """Return last-row indicator values as a clean dict."""
    df = compute_all_indicators(df)
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    rsi_val = float(last["RSI"])
    rsi_state = (
        "Oversold"   if rsi_val < RSI_OVERSOLD  else
        "Overbought" if rsi_val > RSI_OVERBOUGHT else
        "Neutral"
    )

    macd_cross = "Bullish Cross" if (last["MACD"] > last["MACD_Signal"] and
                                      prev["MACD"] <= prev["MACD_Signal"]) else \
                 "Bearish Cross" if (last["MACD"] < last["MACD_Signal"] and
                                      prev["MACD"] >= prev["MACD_Signal"]) else "No Cross"

    supports, resistances = compute_support_resistance(df)
    fibs    = compute_fibonacci(df)
    pivots  = compute_pivots(df)

    return {
        "price":       float(last["Close"]),
        "open":        float(last["Open"]),
        "high":        float(last["High"]),
        "low":         float(last["Low"]),
        "volume":      float(last.get("Volume", 0)),
        "rsi":         rsi_val,
        "rsi_state":   rsi_state,
        "macd":        float(last["MACD"]),
        "macd_signal": float(last["MACD_Signal"]),
        "macd_hist":   float(last["MACD_Hist"]),
        "macd_cross":  macd_cross,
        "sma_20":      float(last.get("SMA_20", 0)),
        "sma_50":      float(last.get("SMA_50", 0)),
        "sma_200":     float(last.get("SMA_200", 0)),
        "bb_upper":    float(last["BB_Upper"]),
        "bb_lower":    float(last["BB_Lower"]),
        "atr":         float(last["ATR"]),
        "stoch_k":     float(last["Stoch_K"]),
        "stoch_d":     float(last["Stoch_D"]),
        "supports":    supports,
        "resistances": resistances,
        "fibonacci":   fibs,
        "pivots":      pivots,
    }
