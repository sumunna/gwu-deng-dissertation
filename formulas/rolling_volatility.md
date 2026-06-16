# Rolling Volatility — Step-by-Step Derivation

**Dissertation:** Multimodal Stock Price Prediction Leveraging Insider Transactions and Market Sentiment  
**Author:** Stella Umunna — George Washington University SEAS, DEng in AI, 2026  
**Reference:** Chapter 3, Section 3.5.2

---

## Overview

Rolling volatility measures how much a stock's price has been fluctuating over a recent window of trading days. It is computed as the standard deviation of daily log returns over a fixed lookback period, recalculated each day as the window moves forward. This gives the model a continuously updated estimate of market uncertainty, which is used to calibrate the weight assigned to directional signals from insider transactions and sentiment.

---

## Step 1 — Compute Log Returns

For each trading day $t$, compute the log return from the prior closing price:

$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

**Where:**
- $P_t$ = closing price on day $t$
- $P_{t-1}$ = closing price on day $t-1$
- $r_t$ = log return on day $t$

**Why log returns?**  
Log returns are preferred over simple percentage returns because they are:
- Symmetric (a 10% gain followed by a 10% loss returns exactly to the origin)
- Approximately normally distributed for short time horizons
- Additive across time periods, simplifying multi-period calculations
- More stable across stocks with very different price levels

**Example:**

| Day | Close Price | Log Return |
|-----|-------------|------------|
| 1   | 100.00      | —          |
| 2   | 102.50      | ln(102.50/100.00) = 0.02469 |
| 3   | 101.00      | ln(101.00/102.50) = -0.01478 |
| 4   | 103.75      | ln(103.75/101.00) = 0.02693 |
| 5   | 102.00      | ln(102.00/103.75) = -0.01703 |

---

## Step 2 — Compute the Mean Log Return Over the Window

$$\bar{r}_t = \frac{1}{n} \sum_{i=t-n+1}^{t} r_i$$

**Where:**
- $n$ = window size (21 days in this study, approximating one trading month)
- $\bar{r}_t$ = mean log return over the window ending at day $t$

---

## Step 3 — Compute Rolling Volatility (Standard Deviation)

$$\sigma_t = \sqrt{\frac{1}{n-1} \sum_{i=t-n+1}^{t} (r_i - \bar{r}_t)^2}$$

**Where:**
- $\sigma_t$ = rolling volatility (standard deviation) at time $t$
- $n - 1$ = Bessel's correction for sample standard deviation (unbiased estimator)
- $r_i$ = individual log return within the window
- $\bar{r}_t$ = mean log return over the window

**Note on Bessel's correction:** The denominator uses $n-1$ rather than $n$ because the mean $\bar{r}_t$ is estimated from the same sample, consuming one degree of freedom. This produces an unbiased estimate of the population standard deviation.

---

## Step 4 — Annualise (Optional)

To express volatility as an annualised figure comparable to industry convention:

$$\sigma_{ann,t} = \sigma_t \times \sqrt{T}$$

**Where:**
- $T$ = number of trading days in a year (252 for US equities)

**Example:** A daily rolling volatility of 0.015 (1.5%) annualises to $0.015 \times \sqrt{252} \approx 0.238$ (23.8%).

---

## Worked Example (5-day window for illustration)

| Day | Close | Log Return $r_t$ | $(r_t - \bar{r})^2$ |
|-----|-------|-----------------|---------------------|
| 1   | 100.00 | —              | —                   |
| 2   | 102.50 | 0.02469        | —                   |
| 3   | 101.00 | -0.01478       | —                   |
| 4   | 103.75 | 0.02693        | —                   |
| 5   | 102.00 | -0.01703       | —                   |
| 6   | 104.50 | 0.02410        | (window: days 2–6)  |

For days 2–6:  
$\bar{r} = (0.02469 - 0.01478 + 0.02693 - 0.01703 + 0.02410) / 5 = 0.00878$

$$\sigma = \sqrt{\frac{(0.02469-0.00878)^2 + (-0.01478-0.00878)^2 + (0.02693-0.00878)^2 + (-0.01703-0.00878)^2 + (0.02410-0.00878)^2}{4}}$$

$$\sigma \approx 0.0197 \quad (1.97\% \text{ daily volatility})$$

---

## Interpretation in the Model

| Volatility Level | Regime Implication | Effect on Signals |
|-----------------|-------------------|-------------------|
| Low ($\sigma < 0.01$) | Calm, low-VIX market | Insider and sentiment signals carry full weight |
| Moderate ($0.01 \leq \sigma < 0.02$) | Normal market conditions | Standard signal weighting |
| High ($\sigma \geq 0.02$) | Elevated uncertainty | Regime gate compresses directional signal confidence |
| Extreme ($\sigma \geq 0.03$) | Crisis conditions (e.g., COVID-19 crash) | Training sample downweighted (0.4× BEAR regime weight) |

**Window used in this study:** 21-day rolling window (primary), 5-day rolling window (short-term regime signal input)
