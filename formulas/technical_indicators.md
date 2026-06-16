# Technical Indicators — Full Derivations

**Dissertation:** Multimodal Stock Price Prediction Leveraging Insider Transactions and Market Sentiment  
**Author:** Stella Umunna — George Washington University SEAS, DEng in AI, 2026  
**Reference:** Chapter 3, Section 3.5.2

---

## 1. Simple Moving Average (SMA)

The Simple Moving Average computes the arithmetic mean of closing prices over a fixed lookback window of *n* days.

**Formula:**

$$SMA_t = \frac{1}{n} \sum_{i=t-n+1}^{t} P_i$$

**Where:**
- $P_i$ = closing price at time $i$
- $n$ = number of periods (e.g., 20 days, 50 days)
- $SMA_t$ = simple moving average at time $t$

**Interpretation:**  
The SMA smooths short-term price fluctuations to reveal the underlying trend direction. A price above its SMA signals bullish momentum; below signals bearish pressure. Crossovers between short-term (e.g., 20-day) and long-term (e.g., 50-day) SMAs are used to detect trend reversals.

**Windows used in this study:** 5-day, 20-day, 50-day, 200-day

---

## 2. Exponential Moving Average (EMA)

The Exponential Moving Average assigns exponentially decreasing weights to older prices, making it more responsive to recent price changes than the SMA.

**Formula:**

$$EMA_t = \alpha P_t + (1 - \alpha) EMA_{t-1}$$

**Where the smoothing factor $\alpha$ is:**

$$\alpha = \frac{2}{n + 1}$$

**Where:**
- $\alpha$ = smoothing factor (between 0 and 1)
- $P_t$ = closing price at time $t$
- $EMA_{t-1}$ = EMA value from the prior period
- $n$ = number of periods

**Initialization:** The first EMA value is set equal to the SMA over the first $n$ periods.

**Interpretation:**  
The EMA reacts faster to recent price changes than the SMA because it gives proportionally more weight to new data. This makes it preferable for detecting short-term momentum shifts. Used in MACD construction and trend strength assessment.

**Windows used in this study:** 12-day, 26-day (for MACD), 20-day, 50-day

---

## 3. Weighted Moving Average (WMA)

The Weighted Moving Average assigns linearly increasing weights to more recent prices, giving the most recent observation the highest weight.

**Formula:**

$$WMA_t = \frac{\sum_{i=1}^{n} w_i P_{t-n+i}}{\sum_{i=1}^{n} w_i}$$

**Where:**
- $w_i = i$ (weights increase linearly: 1, 2, 3, ..., n)
- $P_{t-n+i}$ = closing price at position $i$ within the window
- $n$ = number of periods

**Example (5-day WMA):**

| Day | Price | Weight | Weighted Price |
|-----|-------|--------|----------------|
| t-4 | 100   | 1      | 100            |
| t-3 | 102   | 2      | 204            |
| t-2 | 101   | 3      | 303            |
| t-1 | 104   | 4      | 416            |
| t   | 106   | 5      | 530            |
| **Sum** | | **15** | **1553** |

$$WMA = 1553 / 15 = 103.53$$

**Interpretation:**  
The WMA responds more quickly to recent price changes than the SMA but less smoothly than the EMA. Used as a complementary trend indicator alongside SMA and EMA crossovers.

---

## 4. Moving Average Convergence Divergence (MACD)

MACD measures the relationship between two EMAs to identify momentum and trend direction changes.

**Formula:**

$$MACD_t = EMA_{12}(P_t) - EMA_{26}(P_t)$$

**Signal Line:**

$$Signal_t = EMA_9(MACD_t)$$

**MACD Histogram:**

$$Histogram_t = MACD_t - Signal_t$$

**Interpretation:**
- MACD > 0: short-term momentum stronger than long-term (bullish)
- MACD < 0: short-term momentum weaker than long-term (bearish)
- MACD crossing above Signal Line: buy signal
- MACD crossing below Signal Line: sell signal
- Histogram bars growing: momentum accelerating; shrinking: momentum decelerating

---

## 5. Average True Range (ATR)

ATR measures market volatility by decomposing the full range of price movement over a period, accounting for overnight gaps.

**Step 1 — True Range (TR):**

$$TR_t = \max(H_t - L_t,\ |H_t - C_{t-1}|,\ |L_t - C_{t-1}|)$$

**Where:**
- $H_t$ = high price at time $t$
- $L_t$ = low price at time $t$
- $C_{t-1}$ = closing price at time $t-1$

The three components capture:
1. Current high-to-low range
2. Gap up from prior close to current high
3. Gap down from prior close to current low

**Step 2 — ATR (Wilder's smoothing, 14-day default):**

$$ATR_t = \frac{(n-1) \cdot ATR_{t-1} + TR_t}{n}$$

**Initialization:** The first ATR is the simple average of the first $n$ True Range values.

**Interpretation:**
- High ATR: increased volatility; insider signals may be amplified or obscured by market noise
- Low ATR: calm market conditions; insider signals carry stronger relative weight
- ATR does not indicate direction — only magnitude of price movement

**Window used in this study:** 14-day

---

## 6. Trend Strength

A normalized measure of how far the current price deviates from its moving average, used to identify whether insider signals occur within established trends or during consolidation.

**Formula:**

$$TrendStrength_t = \frac{P_t - MA_n}{MA_n}$$

**Where:**
- $P_t$ = current closing price
- $MA_n$ = moving average over $n$ periods

**Interpretation:**
- Positive values: price above moving average (uptrend)
- Negative values: price below moving average (downtrend)
- Values near zero: consolidation or trend transition

---

## 7. Relative Volume

Compares current trading volume to historical average volume to detect unusual institutional activity.

**Formula:**

$$RelativeVolume_t = \frac{V_t}{\bar{V}_n}$$

**Where:**
- $V_t$ = trading volume at time $t$
- $\bar{V}_n$ = average trading volume over the prior $n$ periods

**Interpretation:**
- RelativeVolume > 1.5: above-average activity; may indicate institutional accumulation or distribution
- RelativeVolume < 0.5: thin trading; signals less reliable
- Used alongside insider buy signals to distinguish high-conviction from low-liquidity trades

**Window used in this study:** 20-day average
