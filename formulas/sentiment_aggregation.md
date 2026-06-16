# Sentiment Aggregation — Formulas and Methodology

**Dissertation:** Multimodal Stock Price Prediction Leveraging Insider Transactions and Market Sentiment  
**Author:** Stella Umunna — George Washington University SEAS, DEng in AI, 2026  
**Reference:** Chapter 3, Sections 3.2.3 and 3.5.3

---

## Overview

Sentiment features are derived from 62,994 financial news headlines and
lead paragraphs processed by FinBERT across January 2018 to May 2026.
Each article is scored at the individual level, then aggregated to the
ticker-day level to produce three daily features consumed by the
prediction model.

---

## Step 1 — FinBERT Article-Level Scoring

Each article is passed through FinBERT (Huang, Wang, & Yang, 2023), which
produces a three-class probability distribution:

$$[P(\text{positive}),\ P(\text{negative}),\ P(\text{neutral})]$$

The signed sentiment score for article $i$ is:

$$s_i = P(\text{positive})_i - P(\text{negative})_i$$

**Range:** $-1.0$ (maximum negative) to $+1.0$ (maximum positive)  
**Near zero:** neutral or uncertain tone

**Example:**

| Article | P(pos) | P(neg) | P(neu) | Score $s_i$ |
|---------|--------|--------|--------|-------------|
| "Earnings beat estimates" | 0.82 | 0.05 | 0.13 | +0.77 |
| "Revenue misses forecast" | 0.04 | 0.89 | 0.07 | -0.85 |
| "Company files quarterly report" | 0.12 | 0.11 | 0.77 | +0.01 |

---

## Step 2 — Timestamp Normalisation

Publication timestamps are normalised to trading-day boundaries:

- Articles published **after 16:00 ET** → assigned to **next trading day**
- Articles published **before 09:30 ET** → assigned to **next trading day**  
- Articles published **09:30–16:00 ET** → assigned to **current trading day**

This ensures the model only consumes sentiment that was publicly
available before the trading session it is predicting.

---

## Step 3 — Ticker Attribution

For articles mentioning a single ticker, the score $s_i$ is attributed
entirely to that ticker.

For articles mentioning multiple tickers, the score is attributed
proportionally by mention frequency:

$$s_{i,k} = s_i \times \frac{m_{i,k}}{\sum_{j} m_{i,j}}$$

**Where:**
- $s_{i,k}$ = sentiment score attributed to ticker $k$ from article $i$
- $m_{i,k}$ = number of mentions of ticker $k$ in article $i$
- $\sum_j m_{i,j}$ = total ticker mentions in article $i$

This prevents artificial inflation of sentiment when one ticker
dominates the article's discussion.

---

## Step 4 — Daily Aggregation Features

### Feature 1: `sentiment_mean_7` (Primary Sentiment Feature)

Exponentially weighted mean of signed sentiment scores over the
prior 7 trading days:

$$sentiment\_mean\_7_{t,k} = \frac{\sum_{d=0}^{6} \lambda^d \cdot \bar{s}_{t-d,k}}{\sum_{d=0}^{6} \lambda^d}$$

**Where:**
- $\bar{s}_{t-d,k}$ = mean signed sentiment score for ticker $k$ on day $t-d$
- $\lambda$ = decay factor (0.85 in this study; more recent days weighted higher)
- Days with no articles: $\bar{s}_{t-d,k} = 0$ (structural zero, not missing)

**SHAP ranking:** #2 most important feature for S&P 500; #1 for small-cap universe

---

### Feature 2: `sentiment_mean_7_rank` (Cross-Sectional Rank)

The percentile rank of `sentiment_mean_7` within the full ticker
universe on each day:

$$sentiment\_mean\_7\_rank_{t,k} = \frac{\text{rank of } sentiment\_mean\_7_{t,k} \text{ among all tickers on day } t}{N_t}$$

**Where** $N_t$ = number of tickers with valid sentiment values on day $t$.

This removes absolute level bias: a sentiment score of +0.3 means
something different in a market-wide negative environment versus a
positive one. The rank transformation produces a uniform [0, 1]
distribution each day.

---

### Feature 3: `news_count_30`

Count of articles mentioning ticker $k$ in the prior 30 trading days:

$$news\_count\_30_{t,k} = \sum_{d=1}^{30} \mathbb{1}[\text{article mentioning } k \text{ exists on day } t-d]$$

**Where** $\mathbb{1}[\cdot]$ = indicator function (1 if at least one article, 0 otherwise).

**Interpretation:** A proxy for information flow intensity. Sudden
increases in `news_count_30` frequently coincide with material
corporate events (earnings announcements, M&A activity, regulatory
actions) and carry independent predictive signal.

**SHAP ranking:** Feature rank #7 for S&P 500 universe

---

## Step 5 — Structural Zeros vs. Missing Values

On trading days where ticker $k$ receives no news coverage:

$$sentiment\_mean\_7_{t,k} = 0, \quad news\_count\_30_{t,k} = 0$$

These are recorded as **structural zeros**, not as missing values
(`NaN`). The distinction is methodologically important:

- **Structural zero:** silence is a genuine market signal — the absence
  of news coverage on a given day is informative about the information
  environment
- **Missing value:** would imply a data collection failure and would
  trigger imputation, distorting the true coverage distribution

Treating zero-coverage days as structural zeros preserves the
authentic information environment and prevents imputation procedures
from inflating apparent news coverage uniformity.

---

## Sentiment Feature Summary

| Feature | Formula Type | Window | SHAP Rank (S&P 500) | SHAP Rank (Small Cap) |
|---------|-------------|--------|---------------------|----------------------|
| `sentiment_mean_7` | EW mean of FinBERT scores | 7-day | #2 | #1 |
| `sentiment_mean_7_rank` | Cross-sectional percentile | Daily universe | #4 | #3 |
| `news_count_30` | Article count | 30-day | #7 | #6 |
