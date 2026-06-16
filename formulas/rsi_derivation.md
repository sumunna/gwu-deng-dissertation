# RSI Derivation — Full Step-by-Step with Interpretations

**Dissertation:** Multimodal Stock Price Prediction Leveraging Insider Transactions and Market Sentiment  
**Author:** Stella Umunna — George Washington University SEAS, DEng in AI, 2026  
**Reference:** Chapter 3, Section 3.5.2

---

## Overview

The Relative Strength Index (RSI) is a bounded momentum oscillator that measures the speed and magnitude of recent price changes to identify overbought and oversold conditions. RSI ranges from 0 to 100. In this study, `rsi_14_rank` — the cross-sectional percentile rank of the 14-day RSI within the daily ticker universe — was the single most important feature in SHAP analysis for the S&P 500 universe.

---

## Step 1 — Compute Daily Price Changes

$$\Delta P_t = P_t - P_{t-1}$$

Separate gains and losses:

$$G_t = \max(\Delta P_t,\ 0) \qquad \text{(gain if price rose, else 0)}$$

$$L_t = \max(-\Delta P_t,\ 0) \qquad \text{(loss if price fell, else 0)}$$

**Example:**

| Day | Close | $\Delta P$ | $G_t$ | $L_t$ |
|-----|-------|------------|-------|-------|
| 1   | 44.34 | —          | —     | —     |
| 2   | 44.09 | -0.25      | 0.00  | 0.25  |
| 3   | 44.15 | +0.06      | 0.06  | 0.00  |
| 4   | 43.61 | -0.54      | 0.00  | 0.54  |
| 5   | 44.33 | +0.72      | 0.72  | 0.00  |
| 6   | 44.83 | +0.50      | 0.50  | 0.00  |
| 7   | 45.10 | +0.27      | 0.27  | 0.00  |
| 8   | 45.15 | +0.05      | 0.05  | 0.00  |
| 9   | 43.61 | -1.54      | 0.00  | 1.54  |
| 10  | 44.33 | +0.72      | 0.72  | 0.00  |
| 11  | 44.83 | +0.50      | 0.50  | 0.00  |
| 12  | 45.10 | +0.27      | 0.27  | 0.00  |
| 13  | 45.15 | +0.05      | 0.05  | 0.00  |
| 14  | 46.92 | +1.77      | 1.77  | 0.00  |
| 15  | 46.75 | -0.17      | 0.00  | 0.17  |

---

## Step 2 — Compute Initial Average Gain and Average Loss

For the first RSI value (day 15 in a 14-period RSI), use the simple average of the first 14 gains and losses:

$$\overline{G}_{14} = \frac{1}{14} \sum_{i=2}^{15} G_i$$

$$\overline{L}_{14} = \frac{1}{14} \sum_{i=2}^{15} L_i$$

Using the example above (days 2–15):  
$\overline{G}_{14} = (0 + 0.06 + 0 + 0.72 + 0.50 + 0.27 + 0.05 + 0 + 0.72 + 0.50 + 0.27 + 0.05 + 1.77 + 0) / 14 = 0.354$  
$\overline{L}_{14} = (0.25 + 0 + 0.54 + 0 + 0 + 0 + 0 + 1.54 + 0 + 0 + 0 + 0 + 0 + 0.17) / 14 = 0.179$

---

## Step 3 — Compute Subsequent Values Using Wilder's Smoothing

For all periods after the initial calculation, use Wilder's smoothing (equivalent to an EMA with $\alpha = 1/n$):

$$\overline{G}_t = \frac{(n-1) \cdot \overline{G}_{t-1} + G_t}{n}$$

$$\overline{L}_t = \frac{(n-1) \cdot \overline{L}_{t-1} + L_t}{n}$$

**Where** $n = 14$ (standard lookback period).

This smoothing gives more weight to recent gains/losses while retaining historical context, preventing the RSI from being overly reactive to single-day price spikes.

---

## Step 4 — Compute Relative Strength (RS)

$$RS_t = \frac{\overline{G}_t}{\overline{L}_t}$$

Using the example:  
$RS = 0.354 / 0.179 = 1.977$

---

## Step 5 — Compute RSI

$$RSI_t = 100 - \frac{100}{1 + RS_t}$$

Using the example:  
$RSI = 100 - (100 / (1 + 1.977)) = 100 - 33.59 = 66.41$

**Boundary behavior:**
- When $\overline{L}_t \to 0$ (all gains, no losses): $RS \to \infty$, $RSI \to 100$
- When $\overline{G}_t \to 0$ (all losses, no gains): $RS \to 0$, $RSI \to 0$

---

## Interpretation

| RSI Range | Market Condition | Trading Implication |
|-----------|-----------------|---------------------|
| RSI > 70  | Overbought | Potential downward correction; insider selling here is a strong bearish signal |
| RSI 50–70 | Bullish momentum | Price trending upward; insider buying reinforces uptrend |
| RSI = 50  | Neutral | No directional momentum bias |
| RSI 30–50 | Bearish momentum | Price trending downward |
| RSI < 30  | Oversold | Potential upward reversal; insider buying here is a strong contrarian signal |

---

## Cross-Sectional Rank Transformation

In this study, the raw RSI is transformed to `rsi_14_rank` — the percentile rank of each stock's RSI within the full 602-ticker universe on each day:

$$rsi\_14\_rank_{t,k} = \frac{\text{rank of } RSI_{t,k} \text{ among all tickers on day } t}{N_t}$$

**Where** $N_t$ = number of tickers with valid RSI values on day $t$.

This transformation:
- Removes absolute level bias (an RSI of 65 means different things in different sector environments)
- Produces a uniform [0, 1] distribution each day, improving model calibration
- Allows direct comparison of momentum signal strength across all 602 tickers simultaneously

`rsi_14_rank` ranked as the **#1 most important feature** in SHAP analysis for the S&P 500 universe (SHAP attribution: 8.3% of total model output variance).
