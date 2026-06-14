#!/usr/bin/env python3
# ============================================================
# features_insider.py — Standalone Windows script
#
# Reads  : features_technical.csv   (from Google Drive)
#          insider_raw_parsed.csv   (from Google Drive)
# Writes : features_insider.csv     (to Google Drive)
#
# Features computed per (ticker, date):
#
#  30-day rolling window
#   insider_buy_count_30       number of buy transactions
#   insider_sell_count_30      number of sell transactions
#   insider_net_count_30       buys - sells
#   insider_buy_value_30       total $ value of buys
#   insider_sell_value_30      total $ value of sells
#   insider_net_value_30       buy_value - sell_value
#   insider_buy_intensity      buy_value / volume_sma_20
#   insider_sell_intensity     sell_value / volume_sma_20
#   insider_sentiment_score    (buy_val - sell_val) / (buy_val + sell_val)
#   insider_cluster_buy        1 if >= 3 distinct insiders bought
#   insider_cluster_sell       1 if >= 3 distinct insiders sold
#   insider_any_recent         1 if any filing in last 14 days
#   ceo_buy_30                 CEO bought in last 30 days
#   ceo_sell_30                CEO sold in last 30 days
#   cfo_buy_30                 CFO bought in last 30 days
#   plan_sell_30               sell under 10b5-1 plan in last 30 days
#
#  Holding signals (silence = confidence)
#   insider_holding            1 if NO sells in last 90 days
#   insider_strong_hold        1 if no sells AND >= 1 buy in 90 days
#   insider_sell_silence       days since last insider sell (capped 180)
#
#  90-day rolling window
#   insider_buy_count_90
#   insider_sell_count_90
#   insider_net_count_90
#   insider_net_value_90
#
# Usage:
#   python features_insider.py
# ============================================================

import os
import sys
import logging
import numpy  as np
import pandas as pd
from datetime import datetime

# ============================================================
# CONFIG — adjust paths if needed
# ============================================================
GDRIVE      = r"G:\My Drive"
DATA_DIR    = os.path.join(GDRIVE, "AI_PROJECT", "Data")
TECH_CSV    = os.path.join(DATA_DIR, "features_technical.csv")
RAW_CSV     = os.path.join(DATA_DIR, "insider_raw_parsed.csv")
INSIDER_CSV = os.path.join(DATA_DIR, "features_insider.csv")
LOG_FILE    = os.path.join(DATA_DIR, "features_insider.log")
# ============================================================


# ── Logging ───────────────────────────────────────────────────
os.makedirs(DATA_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)

# ── Zero-fill dict for tickers with no insider data ───────────
ZERO_INSIDER = {
    "insider_buy_count_30":    0,
    "insider_sell_count_30":   0,
    "insider_net_count_30":    0,
    "insider_buy_value_30":    0.0,
    "insider_sell_value_30":   0.0,
    "insider_net_value_30":    0.0,
    "insider_buy_intensity":   0.0,
    "insider_sell_intensity":  0.0,
    "insider_sentiment_score": 0.0,
    "insider_cluster_buy":     0,
    "insider_cluster_sell":    0,
    "insider_any_recent":      0,
    "ceo_buy_30":              0,
    "ceo_sell_30":             0,
    "cfo_buy_30":              0,
    "plan_sell_30":            0,
    "insider_buy_count_90":    0,
    "insider_sell_count_90":   0,
    "insider_net_count_90":    0,
    "insider_net_value_90":    0.0,
    "insider_holding":         0,
    "insider_strong_hold":     0,
    "insider_sell_silence":    180,
}


# ============================================================
# Feature computation — vectorised with searchsorted
# ============================================================

def compute_insider_features(
    price_dates: pd.DatetimeIndex,
    ticker:      str,
    txns:        pd.DataFrame,
    vol_sma:     pd.Series,
) -> pd.DataFrame:
    """
    Compute all 23 insider features for one ticker across all dates.
    Uses numpy searchsorted for ~50x speedup vs row-by-row filtering.
    """
    tf = txns[txns["ticker"] == ticker].copy()
    tf["date"] = pd.to_datetime(tf["transaction_date"])
    tf = tf.sort_values("date").reset_index(drop=True)

    if len(tf) == 0:
        return pd.DataFrame({
            "date": price_dates, "ticker": ticker, **ZERO_INSIDER
        })

    eps        = 1e-8
    dates_arr  = tf["date"].values
    types_arr  = tf["transaction_type"].values
    vals_arr   = tf["value"].fillna(0).values.astype(float)

    names_arr  = (tf["insider_name"].values
                  if "insider_name"    in tf.columns else None)
    is_ceo_arr = (tf["is_ceo"].values
                  if "is_ceo"          in tf.columns else None)
    is_cfo_arr = (tf["is_cfo"].values
                  if "is_cfo"          in tf.columns else None)
    plan_arr   = (tf["is_10b5_1_plan"].values
                  if "is_10b5_1_plan"  in tf.columns else None)

    sell_mask  = types_arr == "sell"
    sell_dates = dates_arr[sell_mask]
    has_sell   = len(sell_dates) > 0

    rows = []
    for dt in price_dates:
        dt_np = np.datetime64(dt)
        vp    = vol_sma.get(dt, np.nan)
        row   = {"date": dt, "ticker": ticker}

        i_dt = int(np.searchsorted(dates_arr, dt_np, side="left"))

        # ── 14-day recent filing flag ─────────────────────────
        t14      = np.datetime64(dt - pd.Timedelta(days=14))
        i14      = int(np.searchsorted(dates_arr, t14, side="left"))
        any_recent = int(i14 < i_dt)

        # ── 30-day window ─────────────────────────────────────
        t30  = np.datetime64(dt - pd.Timedelta(days=30))
        i30  = int(np.searchsorted(dates_arr, t30, side="left"))
        sl30 = slice(i30, i_dt)
        tt30 = types_arr[sl30]
        tv30 = vals_arr[sl30]
        bm30 = tt30 == "buy"
        sm30 = tt30 == "sell"
        b_cnt = int(bm30.sum())
        s_cnt = int(sm30.sum())
        b_val = float(tv30[bm30].sum())
        s_val = float(tv30[sm30].sum())

        cluster_buy  = int(names_arr is not None and b_cnt > 0 and
                           len(set(names_arr[sl30][bm30])) >= 3)
        cluster_sell = int(names_arr is not None and s_cnt > 0 and
                           len(set(names_arr[sl30][sm30])) >= 3)
        ceo_buy      = int(is_ceo_arr is not None and b_cnt > 0 and
                           bool((is_ceo_arr[sl30][bm30] == 1).any()))
        ceo_sell     = int(is_ceo_arr is not None and s_cnt > 0 and
                           bool((is_ceo_arr[sl30][sm30] == 1).any()))
        cfo_buy      = int(is_cfo_arr is not None and b_cnt > 0 and
                           bool((is_cfo_arr[sl30][bm30] == 1).any()))
        plan_sell    = int(plan_arr is not None and s_cnt > 0 and
                           bool((plan_arr[sl30][sm30] == 1).any()))

        # ── 90-day window ─────────────────────────────────────
        t90  = np.datetime64(dt - pd.Timedelta(days=90))
        i90  = int(np.searchsorted(dates_arr, t90, side="left"))
        sl90 = slice(i90, i_dt)
        tt90 = types_arr[sl90]
        tv90 = vals_arr[sl90]
        bm90 = tt90 == "buy"
        sm90 = tt90 == "sell"
        b_cnt90 = int(bm90.sum())
        s_cnt90 = int(sm90.sum())
        b_val90 = float(tv90[bm90].sum())
        s_val90 = float(tv90[sm90].sum())

        # ── Holding signals ───────────────────────────────────
        holding = int(s_cnt90 == 0)
        strong  = int(s_cnt90 == 0 and b_cnt90 > 0)

        if has_sell:
            past_sells = sell_dates[sell_dates < dt_np]
            silence = (
                min(int((dt_np - past_sells[-1]) /
                        np.timedelta64(1, "D")), 180)
                if len(past_sells) else 180
            )
        else:
            silence = 180

        row.update({
            "insider_buy_count_30":    b_cnt,
            "insider_sell_count_30":   s_cnt,
            "insider_net_count_30":    b_cnt - s_cnt,
            "insider_buy_value_30":    b_val,
            "insider_sell_value_30":   s_val,
            "insider_net_value_30":    b_val - s_val,
            "insider_buy_intensity":   (b_val / (vp + eps)
                                        if not np.isnan(vp) else 0.0),
            "insider_sell_intensity":  (s_val / (vp + eps)
                                        if not np.isnan(vp) else 0.0),
            "insider_sentiment_score": (b_val - s_val) /
                                        (b_val + s_val + eps),
            "insider_cluster_buy":     cluster_buy,
            "insider_cluster_sell":    cluster_sell,
            "insider_any_recent":      any_recent,
            "ceo_buy_30":              ceo_buy,
            "ceo_sell_30":             ceo_sell,
            "cfo_buy_30":              cfo_buy,
            "plan_sell_30":            plan_sell,
            "insider_holding":         holding,
            "insider_strong_hold":     strong,
            "insider_sell_silence":    silence,
            "insider_buy_count_90":    b_cnt90,
            "insider_sell_count_90":   s_cnt90,
            "insider_net_count_90":    b_cnt90 - s_cnt90,
            "insider_net_value_90":    b_val90 - s_val90,
        })
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():
    start = datetime.now()
    log.info("=" * 60)
    log.info("  Insider Feature Engineering — Standalone")
    log.info("=" * 60)

    # ── Validate paths ────────────────────────────────────────
    for path, label in [(TECH_CSV, "features_technical.csv"),
                        (RAW_CSV,  "insider_raw_parsed.csv")]:
        if not os.path.exists(path):
            log.error(f"File not found: {path}")
            log.error(f"Check DATA_DIR in CONFIG: {DATA_DIR}")
            sys.exit(1)

    # ── Load technical features ───────────────────────────────
    log.info("Loading technical features...")
    tech    = pd.read_csv(TECH_CSV, parse_dates=["date"])
    tickers = sorted(tech["ticker"].unique().tolist())
    log.info(f"{len(tickers)} tickers | {len(tech):,} rows | "
             f"latest={tech['date'].max().date()}")

    # ── Load insider transactions ─────────────────────────────
    log.info("\nLoading insider transaction cache...")
    raw = pd.read_csv(RAW_CSV, parse_dates=["transaction_date"])
    raw = raw[raw["transaction_type"].isin(["buy", "sell"])].copy()
    raw["transaction_date"] = pd.to_datetime(
        raw["transaction_date"], errors="coerce")
    raw = raw.dropna(subset=["transaction_date"])

    for col in ["is_ceo", "is_cfo", "is_director", "is_10b5_1_plan"]:
        if col not in raw.columns:
            raw[col] = 0

    tickers_with_data = set(raw["ticker"].unique())
    missing_pct       = (1 - len(tickers_with_data) / len(tickers)) * 100

    log.info(f"{len(raw):,} transactions | "
             f"{raw['ticker'].nunique()} tickers covered")
    log.info(f"Buys  : {(raw['transaction_type']=='buy').sum():,}")
    log.info(f"Sells : {(raw['transaction_type']=='sell').sum():,}")
    log.info(f"Date range: {raw['transaction_date'].min().date()} "
             f"→ {raw['transaction_date'].max().date()}")
    log.info(f"Tickers with data   : {len(tickers_with_data)} "
             f"({100-missing_pct:.0f}%)")
    log.info(f"Tickers zero-filled : "
             f"{len(set(tickers) - tickers_with_data)} "
             f"({missing_pct:.0f}%)")

    # ── Pre-group tech by ticker ──────────────────────────────
    log.info("\nPre-grouping technical data by ticker...")
    tech_grouped = {
        t: grp.sort_values("date").reset_index(drop=True)
        for t, grp in tech.groupby("ticker")
    }

    # ── Compute insider features per ticker ───────────────────
    log.info("Computing rolling insider features...")
    ticker_frames = []
    total         = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        if i % 100 == 0 or i == total:
            elapsed = (datetime.now() - start).seconds
            log.info(f"  {i:>4}/{total} | elapsed={elapsed}s")

        t_grp = tech_grouped.get(ticker)
        if t_grp is None:
            continue

        price_dates = pd.DatetimeIndex(t_grp["date"].values)
        vol_sma     = t_grp.set_index("date")["volume_sma_20"]

        if ticker in tickers_with_data:
            feat_df = compute_insider_features(
                price_dates, ticker, raw, vol_sma)
        else:
            feat_df = pd.DataFrame({
                "date":   price_dates,
                "ticker": ticker,
                **ZERO_INSIDER,
            })

        ticker_frames.append(feat_df)

    # ── Merge and save ────────────────────────────────────────
    log.info("\nMerging onto technical feature matrix...")
    insider_feats         = pd.concat(ticker_frames, ignore_index=True)
    insider_feats["date"] = pd.to_datetime(insider_feats["date"])
    insider_cols          = [c for c in insider_feats.columns
                             if c not in ("ticker", "date")]

    tech["date"] = pd.to_datetime(tech["date"])
    merged       = tech.merge(
        insider_feats, on=["ticker", "date"], how="left")
    merged[insider_cols] = merged[insider_cols].fillna(0)
    merged.to_csv(INSIDER_CSV, index=False)

    # ── Summary ───────────────────────────────────────────────
    sp500_f   = merged[merged["universe"] == "sp500"]
    sc_f      = merged[merged["universe"] == "small_cap"]
    active    = (merged["insider_any_recent"]    > 0).mean() * 100
    buys_pct  = (merged["insider_buy_count_30"]  > 0).mean() * 100
    sells_pct = (merged["insider_sell_count_30"] > 0).mean() * 100
    ceo_pct   = (merged["ceo_buy_30"]           > 0).mean() * 100
    hold_pct  = (merged["insider_holding"]       > 0).mean() * 100

    elapsed_total = (datetime.now() - start).seconds
    log.info(f"\n[DONE] {INSIDER_CSV}")
    log.info(f"  Total rows              : {len(merged):,}")
    log.info(f"  Insider feature columns : {len(insider_cols)}")
    log.info(f"  Tickers with data       : {len(tickers_with_data)}")
    log.info(f"  S&P 500 rows            : {len(sp500_f):,}")
    log.info(f"  Small cap rows          : {len(sc_f):,}")
    log.info(f"  Latest date             : {merged['date'].max().date()}")
    log.info(f"  Total runtime           : {elapsed_total}s")
    log.info(f"\n  Signal sparsity (expect 5-15%):")
    log.info(f"    Any recent filing     : {active:.1f}%")
    log.info(f"    30d buy signal        : {buys_pct:.1f}%")
    log.info(f"    30d sell signal       : {sells_pct:.1f}%")
    log.info(f"    CEO buy signal        : {ceo_pct:.1f}%")
    log.info(f"    Insider holding       : {hold_pct:.1f}%")
    log.info(f"\nLog saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()