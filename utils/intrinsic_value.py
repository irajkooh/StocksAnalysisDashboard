"""
utils/intrinsic_value.py — Intrinsic Value via DCF
Discounted Cash Flow model using yfinance fundamental data.
"""

import math
from typing import Dict, Optional
from utils.config import (
    DCF_DISCOUNT_RATE, DCF_TERMINAL_GROWTH_RATE,
    DCF_PROJECTION_YEARS, DCF_EPS_GROWTH_FALLBACK,
    UNDERVALUED_THRESHOLD, OVERVALUED_THRESHOLD
)


def dcf_intrinsic_value(info: Dict) -> Dict:
    """
    Compute intrinsic value via a simplified DCF using:
      - Free Cash Flow (or Net Income as proxy)
      - Analyst EPS growth estimate (or fallback)
      - Shares outstanding
      - Discount rate (WACC proxy from config)
      - Terminal growth rate

    Returns a dict with intrinsic_value, current_price, premium_discount, label.
    """
    try:
        shares  = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding") or 1
        price   = info.get("currentPrice") or info.get("regularMarketPrice") or 0

        # Free cash flow per share preferred; fall back to earnings
        fcf     = info.get("freeCashflow")
        net_inc = info.get("netIncomeToCommon")
        cash_base = fcf if fcf and fcf > 0 else net_inc

        if not cash_base or cash_base <= 0 or shares <= 0 or price <= 0:
            return _empty_dcf(price)

        cash_per_share = cash_base / shares

        # Growth rate: analyst forward EPS growth or fallback
        growth = (
            info.get("earningsGrowth") or
            info.get("revenueGrowth") or
            DCF_EPS_GROWTH_FALLBACK
        )
        growth = max(min(float(growth), 0.35), -0.10)  # clamp -10% to +35%

        r = DCF_DISCOUNT_RATE
        g = DCF_TERMINAL_GROWTH_RATE
        n = DCF_PROJECTION_YEARS

        # Project FCF/share and discount
        pv_sum = 0.0
        cf = cash_per_share
        for yr in range(1, n + 1):
            cf *= (1 + growth)
            pv_sum += cf / ((1 + r) ** yr)

        # Terminal value (Gordon Growth)
        terminal_cf = cf * (1 + g)
        terminal_pv = (terminal_cf / (r - g)) / ((1 + r) ** n)
        intrinsic    = pv_sum + terminal_pv

        # Add net cash per share (balance sheet buffer)
        net_cash = (info.get("totalCash", 0) or 0) - (info.get("totalDebt", 0) or 0)
        net_cash_per_share = net_cash / shares if shares else 0
        intrinsic += max(net_cash_per_share, 0)

        premium = (price - intrinsic) / intrinsic if intrinsic else 0

        if   premium <= UNDERVALUED_THRESHOLD:
            label = "Undervalued 🟢"
        elif premium >= OVERVALUED_THRESHOLD:
            label = "Overvalued 🔴"
        else:
            label = "Fairly Valued 🟡"

        # Margin of safety
        margin_of_safety = max(0, (intrinsic - price) / intrinsic) if intrinsic > 0 else 0

        return {
            "intrinsic_value":   round(intrinsic, 2),
            "current_price":     round(price, 2),
            "premium_discount":  round(premium * 100, 1),   # % positive = overvalued
            "label":             label,
            "margin_of_safety":  round(margin_of_safety * 100, 1),
            "cash_per_share":    round(cash_per_share, 2),
            "growth_rate_used":  round(growth * 100, 1),
            "discount_rate":     round(r * 100, 1),
            "projection_years":  n,
            "method":            "DCF (Free Cash Flow)",
            "error":             None,
        }

    except Exception as e:
        return _empty_dcf(0, str(e))


def _empty_dcf(price: float, error: str = "Insufficient data") -> Dict:
    return {
        "intrinsic_value":   None,
        "current_price":     round(price, 2),
        "premium_discount":  None,
        "label":             "N/A",
        "margin_of_safety":  None,
        "cash_per_share":    None,
        "growth_rate_used":  None,
        "discount_rate":     round(DCF_DISCOUNT_RATE * 100, 1),
        "projection_years":  DCF_PROJECTION_YEARS,
        "method":            "DCF (Free Cash Flow)",
        "error":             error,
    }


def get_fundamental_metrics(info: Dict) -> Dict:
    """Extract and normalize key fundamental metrics from yfinance info dict."""
    def safe(key, default=None):
        v = info.get(key)
        return round(float(v), 2) if v is not None else default

    return {
        "pe_ratio":          safe("trailingPE"),
        "forward_pe":        safe("forwardPE"),
        "peg_ratio":         safe("pegRatio"),
        "price_to_book":     safe("priceToBook"),
        "price_to_sales":    safe("priceToSalesTrailing12Months"),
        "ev_to_ebitda":      safe("enterpriseToEbitda"),
        "ev_to_revenue":     safe("enterpriseToRevenue"),
        "debt_to_equity":    safe("debtToEquity"),
        "return_on_equity":  safe("returnOnEquity"),
        "return_on_assets":  safe("returnOnAssets"),
        "profit_margin":     safe("profitMargins"),
        "operating_margin":  safe("operatingMargins"),
        "revenue_growth":    safe("revenueGrowth"),
        "earnings_growth":   safe("earningsGrowth"),
        "beta":              safe("beta"),
        "52w_high":          safe("fiftyTwoWeekHigh"),
        "52w_low":           safe("fiftyTwoWeekLow"),
        "market_cap":        info.get("marketCap"),
        "sector":            info.get("sector", "N/A"),
        "industry":          info.get("industry", "N/A"),
        "dividend_yield":    safe("dividendYield"),
        "shares_short_pct":  safe("shortPercentOfFloat"),
        "analyst_target":    safe("targetMeanPrice"),
        "analyst_rating":    info.get("recommendationKey", "N/A"),
    }
