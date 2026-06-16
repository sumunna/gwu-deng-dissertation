# Insider Features — Full Definitions and Derivations

**Dissertation:** Multimodal Stock Price Prediction Leveraging Insider Transactions and Market Sentiment  
**Author:** Stella Umunna — George Washington University SEAS, DEng in AI, 2026  
**Reference:** Chapter 3, Sections 3.2.1 and 3.5.1

---

## Overview

Seven insider features are derived from SEC Form 4 filings across
602 tickers from January 2018 to May 2026. All features are lagged
to the SEC public disclosure date (not the trade execution date) to
prevent look-ahead bias. The mandatory two-business-day filing window
introduced by the Sarbanes-Oxley Act (2002) means that all insider
variables are anchored at least two days after the transaction
occurred, ensuring the model only consumes information that was
publicly available at prediction time.

---

## Data Source and Filtering

**Source:** SEC EDGAR Form 4 electronic filings  
**Universe:** All officer and director transactions for 602 S&P 500 + S&P 600 tickers  
**Period:** January 2018 to May 2026

**Inclusion criteria:**
- Transaction type: open-market purchases and sales only
- Filer role: officers (CEO, CFO, COO, President, VP) and directors
- Security type: common stock and preferred stock (combined)

**Exclusion criteria:**
- Option exercises and conversions
- Gift transfers and inheritances
- Administrative filings with no economic content
- Transactions with missing price or share count data

---

## Feature 1: `buy_count_lag2`

**Definition:** Aggregate count of open-market insider purchases for
ticker $k$, lagged two trading days to align with the SEC disclosure
date.

**Formula:**

$$buy\_count\_lag2_{t,k} = \sum_{j} \mathbb{1}[\text{purchase by insider } j \text{ in ticker } k \text{ disclosed on day } t-2]$$

**Rationale:** The two-day lag reflects the mandatory SOX filing
window. Clustered buying — multiple insiders purchasing within a
short period — is a stronger signal than isolated transactions
(Cohen, Malloy, & Pomorski, 2012). The count (rather than dollar
value) captures clustering behavior independently of transaction size.

---

## Feature 2: `sell_count_lag2`

**Definition:** Aggregate count of open-market insider sales for
ticker $k$, lagged two trading days.

**Formula:**

$$sell\_count\_lag2_{t,k} = \sum_{j} \mathbb{1}[\text{sale by insider } j \text{ in ticker } k \text{ disclosed on day } t-2]$$

**Rationale:** Insider sales carry weaker and more asymmetric
informational content than purchases (Jeng, Metrick, & Zeckhauser,
2003). Sales are frequently driven by personal liquidity needs,
diversification, or pre-planned 10b5-1 trading plans rather than
negative private information. The sell count is retained as a
feature but downweighted by the model relative to buy signals.

---

## Feature 3: `ceo_buy_flag_lag2`

**Definition:** Binary indicator equal to 1 if the CEO executed an
open-market purchase disclosed on day $t-2$, else 0.

**Formula:**

$$ceo\_buy\_flag\_lag2_{t,k} = \mathbb{1}[\text{CEO of ticker } k \text{ filed a purchase on day } t-2]$$

**Rationale:** CEO transactions are treated as a separate feature
because they carry disproportionately strong informational content.
CEOs hold the most comprehensive view of the firm's strategic
direction and internal performance metrics. A CEO purchase signals
conviction that the stock is undervalued relative to information
only they possess (Kaplan, Klebanov, & Sorensen, 2012). The binary
flag allows the model to learn a distinct coefficient for CEO
activity independent of the aggregate buy count.

---

## Feature 4: `net_transaction_value_lag2`

**Definition:** Net dollar value of insider transactions (purchases
minus sales) for ticker $k$, lagged two trading days.

**Formula:**

$$net\_txn\_value_{t,k} = \left(\sum_j shares\_bought_{j,t-2} \times price_{j,t-2}\right) - \left(\sum_j shares\_sold_{j,t-2} \times price_{j,t-2}\right)$$

**Where:**
- $shares\_bought_{j,t-2}$ = shares purchased by insider $j$ disclosed on day $t-2$
- $price_{j,t-2}$ = transaction price reported in the Form 4 filing

**Rationale:** The net transaction value captures the economic
magnitude of insider commitment, not just the directional count.
A CEO purchasing $5M of stock carries a stronger signal than a
director purchasing $50,000.

---

## Feature 5: `insider_strength`

**Definition:** Exponentially weighted decay sum of insider purchase
activity over a rolling window, giving higher weight to more recent
transactions.

**Formula:**

$$insider\_strength_{t,k} = \sum_{i=0}^{K} x_{t-i,k} \cdot e^{-\lambda i}$$

**Where:**
- $x_{t-i,k}$ = insider buy activity (buy count or net value) for ticker $k$ on day $t-i$
- $e^{-\lambda i}$ = exponential decay weight for lag $i$
- $\lambda$ = decay rate (0.1 in this study; half-life approximately 7 days)
- $K$ = maximum lookback window (30 trading days)

**Decay weights by lag:**

| Lag (days) | Weight $e^{-0.1i}$ |
|------------|-------------------|
| 0 (today)  | 1.000             |
| 1          | 0.905             |
| 3          | 0.741             |
| 7          | 0.497             |
| 14         | 0.247             |
| 21         | 0.123             |
| 30         | 0.050             |

**Rationale:** A single insider purchase is an event; a sequence of
purchases by multiple insiders over several weeks represents
accumulation. The exponential decay preserves the cumulative signal
from recent clusters while allowing older activity to fade, preventing
stale signals from dominating the feature.

---

## Feature 6: `relative_size`

**Definition:** Each insider transaction normalized by the stock's
average daily trading volume, measuring the economic significance of
the trade relative to normal market activity.

**Formula:**

$$relative\_size_{t,k} = \frac{\text{Insider Transaction Shares}_{t,k}}{\overline{ADV}_{t,k}}$$

**Where:**
- $\text{Insider Transaction Shares}_{t,k}$ = number of shares in the disclosed transaction
- $\overline{ADV}_{t,k}$ = average daily trading volume for ticker $k$ over the prior 20 trading days

**Rationale:** A purchase of 10,000 shares in a stock with average
daily volume of 50,000 (relative size = 0.20) is far more significant
than the same purchase in a stock with average daily volume of
5,000,000 (relative size = 0.002). Normalizing by ADV allows the
model to compare insider conviction across stocks of very different
sizes and liquidity levels.

---

## Feature 7: `insider_sell_silence_rank`

**Definition:** Cross-sectional percentile rank of the number of
trading days since the last insider sale, within the daily ticker
universe. High values indicate insiders have not sold recently —
a contrarian bullish signal.

**Formula:**

$$silence\_days_{t,k} = t - \max\{d \leq t : sell\_count_{d,k} > 0\}$$

$$insider\_sell\_silence\_rank_{t,k} = \frac{\text{rank of } silence\_days_{t,k} \text{ among all tickers on day } t}{N_t}$$

**Rationale:** The absence of insider selling is itself informative.
When insiders have not sold despite a rising stock price, it suggests
they do not believe the stock is overvalued relative to their private
information. This "sell silence" feature was identified in SHAP
analysis as a meaningful contrarian signal, particularly when combined
with the `ix_silence_gc` regime-aware interaction term (sell silence
in low-VIX environments).

---

## Lag Structure Summary

| Feature | Lag | Reason |
|---------|-----|--------|
| `buy_count_lag2` | 2 trading days | SOX mandatory disclosure window |
| `sell_count_lag2` | 2 trading days | SOX mandatory disclosure window |
| `ceo_buy_flag_lag2` | 2 trading days | SOX mandatory disclosure window |
| `net_transaction_value_lag2` | 2 trading days | SOX mandatory disclosure window |
| `insider_strength` | 2–30 trading days | EW decay over disclosure-date history |
| `relative_size` | 2 trading days | Normalized at disclosure date |
| `insider_sell_silence_rank` | 1 trading day | Prior-day silence count; ranked daily |

All lags are applied before any feature enters the training matrix.
No feature uses same-day or future transaction data at any point
in the pipeline.
