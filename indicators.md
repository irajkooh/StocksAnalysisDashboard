# Indicator Reference

## RSI — Relative Strength Index

Measures **momentum** — whether a stock is being bought or sold too aggressively relative to recent history.

**Formula:**
```
RS  = Avg Gain / Avg Loss  (over 14 periods)
RSI = 100 - (100 / (1 + RS))
```

Result is always **0–100**.

**Dashboard thresholds:**
- **<35** → green (Oversold) — price has fallen too fast, potential bounce
- **35–65** → yellow (Neutral) — normal range
- **>65** → red (Overbought) — price has risen too fast, potential pullback

*(Uses 35/65 instead of the traditional 30/70 — slightly more sensitive.)*

RSI doesn't tell you *when* a reversal happens, just that conditions are stretched. A stock can stay overbought for weeks in a strong trend. Most useful as a confirmation signal alongside other indicators, not a standalone buy/sell trigger.

---

## Stochastic %K

Measures **where the current closing price sits within the recent high-low range**, as a percentage (0–100).

**Formula:**
```
%K = (Close - Lowest Low) / (Highest High - Lowest Low) × 100
```
over a 14-period lookback window.

**Dashboard thresholds:**
- **<20** → green (Oversold) — price near the bottom of its recent range, potential reversal up
- **20–80** → yellow (Neutral)
- **>80** → red (Overbought) — price near the top, potential reversal down

A reading of 36 means the close is 36% of the way up from the recent low to the recent high — leaning toward the lower end, but not yet oversold.

**vs RSI:** Both are 0–100 momentum oscillators. RSI compares gains vs losses over time; Stochastic compares close to the high-low range. Divergence between them can signal a stronger trade setup.

---

## ATR — Average True Range

Measures **how much a stock moves on average per bar**, in dollar terms.

**Formula:**
```
True Range = max(
    High - Low,
    |High - Previous Close|,
    |Low  - Previous Close|
)
ATR = 14-period rolling average of True Range
```

Uses the previous close to capture gaps (e.g. stock closes at $100, gaps up to open at $110).

**No direction** — it's pure volatility, not bullish/bearish. High ATR → wide swings, wider stop-losses needed. Low ATR → tight range, calmer price action.

**Dashboard thresholds** (ATR as % of price):
- **<2%** → green (Low)
- **2–5%** → yellow (Moderate)
- **>5%** → red (High)

So $0.61 on a $30 stock (~2%) is Moderate, while $0.61 on a $200 stock (0.3%) is Low.

---

## Annualized Volatility

Measures **how much a stock's daily returns fluctuate**, scaled to a yearly figure.

**Formula:**
```
Daily Returns = ln(Close / Previous Close)
Std Dev       = standard deviation of daily returns (over ~252 days)
Ann. Vol      = Std Dev × √252
```

`√252` converts daily volatility to annual (252 = trading days per year).

Ann. Vol = 30% means the stock's price could swing ±30% over a year (within 1 standard deviation). No direction — same as ATR, it's pure magnitude of movement.

**Dashboard thresholds:**
- **<20%** → green (Low) — stable, blue-chip territory
- **20–40%** → yellow (Moderate) — typical for growth stocks
- **>40%** → red (High) — volatile, speculative or crypto

**vs ATR:** ATR is in dollars per bar — useful for setting stop-losses. Ann. Vol is in percent annualized — useful for comparing risk across different stocks regardless of price.

---

## Sharpe Ratio

Measures **return per unit of risk** — how much reward you get for the volatility you're taking on.

**Formula:**
```
Sharpe = (Portfolio Return - Risk-Free Rate) / Std Dev of Returns
```

Simplified in the dashboard to:
```
Sharpe = Mean Daily Return / Std Dev of Daily Returns × √252
```

**Dashboard thresholds:**
- **>1** → green (Good) — solid risk-adjusted return
- **0–1** → yellow (Fair) — positive but not great
- **<0** → red (Poor) — losing money on a risk-adjusted basis

**Real-world benchmarks:** S&P 500 long-run Sharpe ≈ 0.5–0.6. Sharpe >2 is exceptional. A Sharpe of -4.97 means the stock is generating large losses relative to its volatility.

Two stocks with the same return can have very different Sharpes. If stock A returns 20% with low volatility and stock B returns 20% with wild swings, stock A has a better Sharpe — same reward for less risk.

---

## Risk Score

A **composite score from 1–10** combining multiple risk factors into a single number.

**Components:**

| Component | Points | Logic |
|---|---|---|
| Annualized Volatility | 0–3 | every 20% vol = +1 pt |
| Max Drawdown | 0–3 | every 15% drawdown = +1 pt |
| Beta | 0–2 | beta >1 = +1 pt, >2 = +2 pt |
| Short Interest % | 0–2 | every 5% short interest = +1 pt |

**Dashboard thresholds:**
- **1–3** → green (Low Risk)
- **4–6** → yellow (Moderate Risk)
- **7–10** → red (High Risk)

**What each component means:**
- **Volatility** — how wildly price swings day-to-day
- **Max Drawdown** — worst peak-to-trough loss in history
- **Beta** — how much the stock moves relative to the market (beta=2 → moves 2× the S&P)
- **Short Interest** — % of float sold short; high short interest signals that large traders expect it to fall
