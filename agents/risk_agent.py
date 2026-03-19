"""
agents/risk_agent.py — Risk Assessment Agent
Computes volatility, ATR-based risk/reward, drawdown, and short squeeze metrics.
"""

import logging
import numpy as np
from agents.state import AnalysisState

logger = logging.getLogger(__name__)


def risk_agent(state: AnalysisState) -> AnalysisState:
    """
    Assess trading risk metrics from OHLCV + indicators.
    Populates: risk
    """
    df          = state.get("ohlcv_df")
    indicators  = state.get("indicators", {})
    fundamentals= state.get("fundamentals", {})
    ticker      = state.get("ticker", "")
    errors      = list(state.get("errors", []))

    if df is None or df.empty:
        return {**state, "risk": {}, "errors": errors}

    try:
        closes  = df["Close"]
        returns = closes.pct_change().dropna()

        price   = indicators.get("price", float(closes.iloc[-1]))
        atr     = indicators.get("atr", 0)
        beta    = fundamentals.get("beta") or 1.0

        # Volatility
        daily_vol  = float(returns.std())
        annual_vol = daily_vol * (252 ** 0.5)

        # Max drawdown (rolling window)
        roll_max  = closes.cummax()
        drawdowns = (closes - roll_max) / roll_max
        max_drawdown = float(drawdowns.min())

        # ATR-based stop loss and take profit suggestions
        atr_stop    = price - 1.5 * atr if atr else None
        atr_target  = price + 2.5 * atr if atr else None
        risk_reward = (atr_target - price) / (price - atr_stop) if atr_stop and atr_stop < price else None

        # Sharpe approximation (risk-free 5%)
        excess_ret = returns.mean() - 0.05 / 252
        sharpe     = (excess_ret / daily_vol * (252 ** 0.5)) if daily_vol else 0

        # Short interest as risk factor
        short_pct  = fundamentals.get("shares_short_pct") or 0

        # Overall risk score 1-10
        risk_score = _compute_risk_score(annual_vol, max_drawdown, beta, short_pct)

        risk_label = (
            "Low Risk 🟢"    if risk_score <= 3 else
            "Moderate Risk 🟡" if risk_score <= 6 else
            "High Risk 🔴"
        )

        return {
            **state,
            "risk": {
                "daily_volatility":  round(daily_vol * 100, 2),
                "annual_volatility": round(annual_vol * 100, 2),
                "max_drawdown":      round(max_drawdown * 100, 2),
                "sharpe_ratio":      round(sharpe, 2),
                "beta":              round(beta, 2),
                "atr":               round(atr, 2),
                "stop_loss_atr":     round(atr_stop, 2) if atr_stop else None,
                "take_profit_atr":   round(atr_target, 2) if atr_target else None,
                "risk_reward_ratio": round(risk_reward, 2) if risk_reward else None,
                "short_interest_pct":round(short_pct * 100, 1) if short_pct else 0,
                "risk_score":        risk_score,
                "risk_label":        risk_label,
            },
            "errors": errors,
        }

    except Exception as e:
        errors.append(f"RiskAgent error: {str(e)}")
        logger.error(f"RiskAgent [{ticker}]: {e}")
        return {**state, "risk": {}, "errors": errors}


def _compute_risk_score(annual_vol: float, max_dd: float, beta: float, short_pct: float) -> int:
    score = 0
    score += min(3, int(annual_vol / 0.20))       # 0-3: 20% vol = 1 pt
    score += min(3, int(abs(max_dd) / 0.15))      # 0-3: 15% dd = 1 pt
    score += min(2, int(beta / 1.0))              # 0-2: beta > 1 = risky
    score += min(2, int(short_pct / 0.05))        # 0-2: 5% short = 1 pt
    return max(1, min(10, score + 1))
