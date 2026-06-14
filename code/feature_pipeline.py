# =============================================================================
#  feature_pipeline.py
#  Feature Engineering + Fusion Pipeline
#  Multimodal Stock Advisory System | Stella Umunna, DEng AI, GWU
# =============================================================================
#
#  Replaces (Colab notebook cells):
#      feature_engineering_v3.py
#      fusion_v3.py
#
#  Run from terminal / command prompt:
#      python feature_pipeline.py
#
#  Optional — run one step only:
#      python feature_pipeline.py features   (build features_technical.csv only)
#      python feature_pipeline.py fusion     (merge + rank + interact only)
#
# =============================================================================
#  WEEKLY MAINTENANCE CHECKLIST
# =============================================================================
#  1. Update EARNINGS_CALENDAR below with next week's reporters (~2 min)
#  2. python feature_pipeline.py
#  3. python regime_gated_model.py
# =============================================================================

import os
import sys
import logging
import time
from datetime import datetime

import numpy  as np
import pandas as pd
from scipy import stats

# =============================================================================
#  PATH CONFIGURATION
#  Windows — Google Drive desktop app
#  Check which letter your Drive mounts to (G:, H:, etc.)
# =============================================================================

GDRIVE   = r"G:\My Drive"
DATA_DIR = os.path.join(GDRIVE, "AI_PROJECT", "Data")

# ── Input files ───────────────────────────────────────────────────────────────
PRICES_CSV  = os.path.join(DATA_DIR, "prices_cache.csv")
INSIDER_CSV = os.path.join(DATA_DIR, "features_insider.csv")
SENT_CSV    = os.path.join(DATA_DIR, "features_sentiment.csv")
MACRO_CSV   = os.path.join(DATA_DIR, "macro_features.csv")       # optional
GICS_CSV    = os.path.join(DATA_DIR, "ticker_sector.csv")        # optional

# ── Output files ──────────────────────────────────────────────────────────────
TECH_CSV  = os.path.join(DATA_DIR, "features_technical.csv")
SP500_CSV = os.path.join(DATA_DIR, "features_final_sp500_regime.csv")
SC_CSV    = os.path.join(DATA_DIR, "features_final_smallcap_regime.csv")

# ── Log file (appends each run) ───────────────────────────────────────────────
LOG_FILE  = os.path.join(DATA_DIR, "feature_pipeline.log")

EARNINGS_CALENDAR = {
    # ── Current cycle (update each week) ─────────────────────────────────────
    "CVNA":  "2026-05-07",
    "CRL":   "2026-05-07",
    "IFF":   "2026-05-06",
    "AMTM":  "2026-05-12",
    "CZR":   "2026-05-07",
    "ACLS":  "2026-05-07",
    # ── Template: add next week's reporters here ──────────────────────────────
    # "TICKER": "2026-MM-DD",
}

# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSE CONFIG  (unchanged from Colab version)
# ─────────────────────────────────────────────────────────────────────────────
UNIVERSE_CONFIG = {
    "sp500": {
        "sma_windows":   [5, 10, 20, 50],
        "ema_windows":   [12, 26],
        "rsi_period":    14,
        "macd_fast":     12,
        "macd_slow":     26,
        "macd_signal":   9,
        "atr_period":    14,
        "bb_period":     20,
        "bb_std":        2.0,
        "vol_window":    20,
        "momentum_days": [5, 10],
        "roc_window":    20,
        "volume_window": 20,
    },
    "small_cap": {
        "sma_windows":   [5, 10, 20, 50],
        "ema_windows":   [12, 26],
        "rsi_period":    14,
        "macd_fast":     12,
        "macd_slow":     26,
        "macd_signal":   9,
        "atr_period":    10,
        "bb_period":     20,
        "bb_std":        2.5,
        "vol_window":    20,
        "momentum_days": [5, 10],
        "roc_window":    20,
        "volume_window": 20,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# FUSION CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BASE_COLS = [
    "ticker", "date", "asof_date", "open", "high", "low",
    "close", "adj_close", "volume", "universe",
    "target_return", "target_direction",
    "target_5d_return", "target_5d_direction",
    "is_prediction_row",
]

RANK_COLS = [
    "rsi_14", "momentum_5", "momentum_10", "volume_ratio",
    "bb_pct", "macd_hist", "hist_vol_20",
    "sentiment_mean_7", "insider_sentiment_score",
    "insider_sell_silence", "pv_corr_10",
    "parkinson_vol", "vol_spike_ratio", "atr_pct",   # new v3
]

PASSTHROUGH_COLS = [
    "hv5", "hv60", "parkinson_vol", "vol_spike_ratio",
    "atr_pct", "hv20_rank_own",
    "breadth_1d", "breadth_5d",
    "market_ret_1d", "market_ret_5d",
    "sector_ret_5d", "sector_dispersion",
    "near_earnings",
]

SECTOR_RANK_COLS = [
    "rsi_14", "momentum_5", "momentum_10", "hist_vol_20",
    "volume_ratio", "bb_pct", "macd_hist", "sentiment_mean_7",
    "parkinson_vol", "vol_spike_ratio",
]

PT_DEFAULTS = {
    "breadth_5d":        0.50,
    "breadth_1d":        0.50,
    "market_ret_5d":     0.00,
    "market_ret_1d":     0.00,
    "sector_ret_5d":     0.00,
    "sector_dispersion": 0.02,
    "hv20_rank_own":     0.50,
    "near_earnings":     0,
    "vol_spike_ratio":   1.00,
    "parkinson_vol":     0.25,
    "atr_pct":           0.02,
    "hv5":               0.00,
    "hv60":              0.00,
}

# ─────────────────────────────────────────────────────────────────────────────
# GICS SECTOR MAP
# ─────────────────────────────────────────────────────────────────────────────
SECTOR_MAP = {
    "AAPL":"Technology","MSFT":"Technology","NVDA":"Technology",
    "AVGO":"Technology","ORCL":"Technology","CRM":"Technology",
    "AMD":"Technology","INTC":"Technology","QCOM":"Technology",
    "TXN":"Technology","NOW":"Technology","ADBE":"Technology",
    "AMAT":"Technology","KLAC":"Technology","LRCX":"Technology",
    "MU":"Technology","SNPS":"Technology","CDNS":"Technology",
    "AKAM":"Technology","DDOG":"Technology","DELL":"Technology",
    "HPQ":"Technology","HPE":"Technology","NTAP":"Technology",
    "JNPR":"Technology","CSCO":"Technology","ECG":"Technology",
    "AMTM":"Technology","ACLS":"Technology",
    "JPM":"Financials","BAC":"Financials","WFC":"Financials",
    "GS":"Financials","MS":"Financials","BLK":"Financials",
    "AXP":"Financials","COF":"Financials","USB":"Financials",
    "PNC":"Financials","TFC":"Financials","SCHW":"Financials",
    "CME":"Financials","ICE":"Financials","CBOE":"Financials",
    "SPGI":"Financials","MCO":"Financials","V":"Financials",
    "MA":"Financials","PYPL":"Financials","COIN":"Financials",
    "MARA":"Financials","BRO":"Financials","AUB":"Financials",
    "AX":"Financials","ABCB":"Financials","LNC":"Financials",
    "UNH":"Healthcare","JNJ":"Healthcare","ABT":"Healthcare",
    "TMO":"Healthcare","DHR":"Healthcare","BMY":"Healthcare",
    "AMGN":"Healthcare","GILD":"Healthcare","VRTX":"Healthcare",
    "BSX":"Healthcare","EW":"Healthcare","MDT":"Healthcare",
    "SYK":"Healthcare","ZBH":"Healthcare","TFX":"Healthcare",
    "CORT":"Healthcare","ALKS":"Healthcare","BTSG":"Healthcare",
    "ACT":"Healthcare","KRYS":"Healthcare","CNC":"Healthcare",
    "CRL":"Healthcare","REZI":"Healthcare",
    "AMZN":"ConsDisc","TSLA":"ConsDisc","HD":"ConsDisc",
    "MCD":"ConsDisc","NKE":"ConsDisc","LOW":"ConsDisc",
    "BKNG":"ConsDisc","CMG":"ConsDisc","TGT":"ConsDisc",
    "ROST":"ConsDisc","ORLY":"ConsDisc","AZO":"ConsDisc",
    "CVNA":"ConsDisc","EAT":"ConsDisc","LKQ":"ConsDisc",
    "BOOT":"ConsDisc","CZR":"ConsDisc","ACA":"ConsDisc",
    "PG":"ConsStaples","KO":"ConsStaples","PEP":"ConsStaples",
    "PM":"ConsStaples","MO":"ConsStaples","WMT":"ConsStaples",
    "COST":"ConsStaples","CL":"ConsStaples","KMB":"ConsStaples",
    "LW":"ConsStaples","DLTR":"ConsStaples",
    "XOM":"Energy","CVX":"Energy","COP":"Energy",
    "EOG":"Energy","SLB":"Energy","PSX":"Energy",
    "VLO":"Energy","MPC":"Energy","HES":"Energy",
    "CRC":"Energy","LBRT":"Energy",
    "GE":"Industrials","HON":"Industrials","CAT":"Industrials",
    "UPS":"Industrials","DE":"Industrials","RTX":"Industrials",
    "LMT":"Industrials","NOC":"Industrials","GD":"Industrials",
    "BLDR":"Industrials","CARR":"Industrials","PRIM":"Industrials",
    "FSS":"Industrials","AGX":"Industrials","CWST":"Industrials",
    "ADT":"Industrials",
    "LIN":"Materials","APD":"Materials","SHW":"Materials",
    "NEM":"Materials","FCX":"Materials","NUE":"Materials",
    "IFF":"Materials","CE":"Materials","CENX":"Materials",
    "EMN":"Materials",
    "NEE":"Utilities","DUK":"Utilities","SO":"Utilities",
    "D":"Utilities","AEP":"Utilities","PECO":"Utilities",
    "PLD":"RealEstate","AMT":"RealEstate","EQIX":"RealEstate",
    "CCI":"RealEstate","WELL":"RealEstate","RHP":"RealEstate",
    "GOOGL":"CommServices","META":"CommServices","NFLX":"CommServices",
    "DIS":"CommServices","CMCSA":"CommServices","T":"CommServices",
    "VZ":"CommServices","TMUS":"CommServices",
}


# =============================================================================
# LOGGING
# =============================================================================
# =============================================================================
# PART 1 — TECHNICAL FEATURE ENGINEERING
# =============================================================================

# ── Indicator helpers ─────────────────────────────────────────────────────────

def compute_sma(close, window):
    return close.rolling(window, min_periods=window).mean()

def compute_ema(close, span):
    return close.ewm(span=span, adjust=False).mean()

def compute_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    ag    = gain.ewm(alpha=1/period, adjust=False).mean()
    al    = loss.ewm(alpha=1/period, adjust=False).mean()
    rs    = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_macd(close, fast, slow, signal):
    ema_f = compute_ema(close, fast)
    ema_s = compute_ema(close, slow)
    line  = ema_f - ema_s
    sig   = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig

def compute_atr(high, low, close, period):
    prev = close.shift(1)
    tr   = pd.concat([high - low,
                      (high - prev).abs(),
                      (low  - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def compute_bollinger(close, period, n_std):
    mid   = close.rolling(period, min_periods=period).mean()
    std   = close.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    width = (upper - lower) / mid.replace(0, np.nan)
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return upper, lower, width, pct_b

def compute_obv(close, volume):
    return (np.sign(close.diff()).fillna(0) * volume).cumsum()

def compute_vwap_proxy(close, volume, window):
    return ((close * volume).rolling(window).sum()
            / volume.rolling(window).sum())

def compute_hist_vol(close, window):
    return (np.log(close / close.shift(1))
            .rolling(window).std(ddof=1) * np.sqrt(252))

def compute_parkinson_vol(high, low, window=20):
    """High-low range IV proxy (Parkinson 1980)."""
    log_hl = np.log(high / low.replace(0, np.nan))
    return ((log_hl ** 2)
            .rolling(window, min_periods=window // 2)
            .mean()
            .pipe(lambda x: np.sqrt(x / (4 * np.log(2))) * np.sqrt(252)))

def compute_hv20_own_rank(hv20_series, window=252):
    """Percentile rank of HV20 vs its own trailing history."""
    def _pct_rank(x):
        if len(x) < 5 or np.isnan(x.iloc[-1]):
            return np.nan
        return stats.percentileofscore(
            x.dropna().values, x.iloc[-1], kind="rank") / 100.0
    return hv20_series.rolling(window, min_periods=63).apply(
        _pct_rank, raw=False)

def compute_atr_pct(atr, close):
    return atr / close.replace(0, np.nan)

def compute_vol_spike_ratio(close, short=5, long=60):
    hv_s = compute_hist_vol(close, short)
    hv_l = compute_hist_vol(close, long)
    return hv_s / hv_l.replace(0, np.nan)


# ── Market-wide features (computed once, merged onto all tickers) ─────────────

def compute_market_breadth(prices: pd.DataFrame) -> pd.DataFrame:
    sp = (prices[prices["universe"] == "sp500"]
          .sort_values(["ticker", "date"])
          [["ticker", "date", "adj_close"]].copy())
    sp["ret_1d"] = sp.groupby("ticker")["adj_close"].pct_change(1)
    sp["ret_5d"] = sp.groupby("ticker")["adj_close"].pct_change(5)
    return (sp.groupby("date")
            .agg(
                breadth_1d    = ("ret_1d", lambda x: (x > 0).mean()),
                breadth_5d    = ("ret_5d", lambda x: (x > 0).mean()),
                market_ret_1d = ("ret_1d", "mean"),
                market_ret_5d = ("ret_5d", "mean"),
            )
            .reset_index())


def compute_sector_features(prices: pd.DataFrame,
                             sector_map: dict) -> pd.DataFrame:
    sp = (prices[prices["universe"] == "sp500"]
          .sort_values(["ticker", "date"])
          [["ticker", "date", "adj_close"]].copy())
    sp["sector"] = sp["ticker"].map(sector_map)
    sp["ret_5d"] = sp.groupby("ticker")["adj_close"].pct_change(5)

    sector_ret = (sp.groupby(["date", "sector"])["ret_5d"]
                  .mean().reset_index()
                  .rename(columns={"ret_5d": "sector_ret_5d"}))

    dispersion = (sector_ret.groupby("date")["sector_ret_5d"]
                  .std().reset_index()
                  .rename(columns={"sector_ret_5d": "sector_dispersion"}))

    ticker_sector = sp[["ticker", "date", "sector"]].drop_duplicates()
    merged = (ticker_sector
              .merge(sector_ret, on=["date", "sector"], how="left")
              .merge(dispersion, on="date", how="left"))
    return merged[["ticker", "date", "sector_ret_5d", "sector_dispersion"]]


def compute_earnings_flag(dates: pd.Series,
                           ticker: str,
                           earnings_calendar: dict,
                           window_days: int = 5) -> pd.Series:
    earn_str = earnings_calendar.get(ticker)
    if earn_str is None:
        return pd.Series(0, index=dates.index, dtype="int8")
    earn_dt = pd.Timestamp(earn_str)
    return ((dates - earn_dt).abs()
            .dt.days.le(window_days)
            .astype("int8"))


# ── Per-ticker feature builder ────────────────────────────────────────────────

def build_ticker_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    close  = df["adj_close"]
    high   = df["high"]
    low    = df["low"]
    open_  = df["open"]
    volume = df["volume"]

    for w in cfg["sma_windows"]:
        df[f"sma_{w}"] = compute_sma(close, w)
    for span in cfg["ema_windows"]:
        df[f"ema_{span}"] = compute_ema(close, span)

    sma20 = df.get("sma_20", compute_sma(close, 20))
    sma50 = df.get("sma_50", compute_sma(close, 50))
    df["price_vs_sma20"] = close / sma20.replace(0, np.nan)
    df["price_vs_sma50"] = close / sma50.replace(0, np.nan)
    df["golden_cross"]   = (sma20 > sma50).astype(int)

    df["rsi_14"] = compute_rsi(close, cfg["rsi_period"])
    macd_line, signal_line, macd_hist = compute_macd(
        close, cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"])
    df["macd"]        = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"]   = macd_hist

    for d in cfg["momentum_days"]:
        df[f"momentum_{d}"] = close.pct_change(d)
    df["roc_20"] = close.pct_change(cfg["roc_window"])

    atr = compute_atr(high, low, close, cfg["atr_period"])
    df["atr_14"] = atr
    bb_upper, bb_lower, bb_width, bb_pct = compute_bollinger(
        close, cfg["bb_period"], cfg["bb_std"])
    df["bb_upper"]    = bb_upper
    df["bb_lower"]    = bb_lower
    df["bb_width"]    = bb_width
    df["bb_pct"]      = bb_pct
    df["hist_vol_20"] = compute_hist_vol(close, cfg["vol_window"])
    df["vix_proxy"]   = df["hist_vol_20"]

    # New v3 vol features
    df["hv5"]             = compute_hist_vol(close, 5)
    df["hv60"]            = compute_hist_vol(close, 60)
    df["vol_spike_ratio"] = compute_vol_spike_ratio(close)
    df["parkinson_vol"]   = compute_parkinson_vol(high, low,
                                                    window=cfg["vol_window"])
    df["atr_pct"]         = compute_atr_pct(atr, close)
    df["hv20_rank_own"]   = np.nan   # filled in post-loop

    df["volume_sma_20"] = volume.rolling(cfg["volume_window"]).mean()
    df["volume_ratio"]  = volume / df["volume_sma_20"].replace(0, np.nan)
    df["obv"]           = compute_obv(close, volume)
    df["vwap_20"]       = compute_vwap_proxy(close, volume,
                                              cfg["volume_window"])
    vol_sma5            = volume.rolling(5).mean()
    vol_sma10           = volume.rolling(10).mean()
    df["volume_trend"]  = (vol_sma5 - vol_sma10) / vol_sma10.replace(0, np.nan)
    df["force_index"]   = close.diff(1) * volume
    log_ret             = np.log(close / close.shift(1))
    df["pv_corr_10"]    = log_ret.rolling(10).corr(volume)

    df["daily_return"]  = close.pct_change()
    df["hl_spread"]     = (high - low) / close.replace(0, np.nan)
    df["gap"]           = (open_ - close.shift(1)) / close.shift(1).replace(0, np.nan)

    df["return_lag1"]   = df["daily_return"].shift(1)
    df["return_lag2"]   = df["daily_return"].shift(2)
    df["return_lag5"]   = df["daily_return"].shift(5)
    df["vol_lag5"]      = df["hist_vol_20"].shift(5)
    df["rsi_lag5"]      = df["rsi_14"].shift(5)

    macd_above          = (macd_line > signal_line).astype(int)
    macd_above_prev     = macd_above.shift(1)
    df["macd_cross"]    = np.where(
        (macd_above == 1) & (macd_above_prev == 0),  1,
        np.where(
        (macd_above == 0) & (macd_above_prev == 1), -1, 0))

    bb_width_ma         = bb_width.rolling(20).mean()
    df["bb_squeeze"]    = (bb_width < bb_width_ma).astype(int)

    ret_20d             = close.pct_change(20)
    df["regime"]        = np.where(ret_20d >  0.02,  1,
                          np.where(ret_20d < -0.02, -1, 0))

    # Targets — 5-day primary, 1-day retained for compatibility
    df["target_5d_return"]    = close.pct_change(5).shift(-5)
    df["target_5d_direction"] = np.sign(df["target_5d_return"]).astype("Int8")
    df["target_return"]       = df["daily_return"].shift(-1)
    df["target_direction"]    = np.sign(df["target_return"]).astype("Int8")
    df["is_prediction_row"]   = df["target_5d_return"].isna().astype(int)

    return df


def run_feature_engineering(log, dry_run=False):
    """
    Step 1: build features_technical.csv from prices_cache.csv.
    Returns path to output file.
    """
    log.info("=" * 60)
    log.info("STEP 1 — TECHNICAL FEATURE ENGINEERING")
    log.info("=" * 60)
    log.info(f"Input  : {PRICES_CSV}")
    log.info(f"Output : {TECH_CSV}")

    if not os.path.exists(PRICES_CSV):
        log.error(f"prices_cache.csv not found: {PRICES_CSV}")
        sys.exit(1)

    # ── Load ──────────────────────────────────────────────────────────────────
    log.info("Loading prices...")
    prices = pd.read_csv(PRICES_CSV, parse_dates=["date"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    log.info(f"  {len(prices):,} rows | {prices['ticker'].nunique()} tickers "
             f"| latest={prices['date'].max().date()}")

    if dry_run:
        log.info("[DRY RUN] Skipping computation — would write "
                 f"{output_path}")
        return TECH_CSV

    # ── Market-wide features ──────────────────────────────────────────────────
    log.info("Computing market breadth features...")
    breadth_df = compute_market_breadth(prices)

    log.info("Computing sector rotation features...")
    sector_df  = compute_sector_features(prices, SECTOR_MAP)

    # ── Per-ticker features ───────────────────────────────────────────────────
    log.info("Building per-ticker features...")
    all_frames = []
    tickers    = prices["ticker"].unique()
    total      = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        if i % 100 == 0 or i == total:
            log.info(f"  Processing {i}/{total} tickers...")
        df_t     = prices[prices["ticker"] == ticker].copy().reset_index(drop=True)
        universe = df_t["universe"].iloc[0]
        cfg      = UNIVERSE_CONFIG.get(universe, UNIVERSE_CONFIG["sp500"])
        try:
            df_t = build_ticker_features(df_t, cfg)
            all_frames.append(df_t)
        except Exception as e:
            log.warning(f"  {ticker}: {e}")

    features = pd.concat(all_frames, ignore_index=True)

    # ── HV20 own-history rank (expensive rolling apply — post-loop) ───────────
    log.info("Computing HV20 own-history rank (rolling percentile)...")
    hv20_rank_list = []
    for ticker, grp in features.groupby("ticker"):
        rank = compute_hv20_own_rank(grp["hist_vol_20"])
        hv20_rank_list.append(rank)
    features["hv20_rank_own"] = pd.concat(hv20_rank_list)

    # ── Merge market-wide features ────────────────────────────────────────────
    log.info("Merging breadth and sector features...")
    features = features.merge(breadth_df, on="date", how="left")
    features = features.merge(sector_df,  on=["ticker", "date"], how="left")

    for col in ["sector_ret_5d", "sector_dispersion"]:
        features[col] = features[col].fillna(
            features.groupby("date")[col].transform("mean"))

    # ── Earnings proximity flag ───────────────────────────────────────────────
    log.info("Computing earnings proximity flags...")
    earn_flags = []
    for ticker, grp in features.groupby("ticker"):
        flag = compute_earnings_flag(grp["date"], ticker, EARNINGS_CALENDAR)
        earn_flags.append(flag)
    features["near_earnings"] = (pd.concat(earn_flags)
                                   .fillna(0).astype("int8"))

    n_earn = features["near_earnings"].sum()
    n_earn_tickers = features[features["near_earnings"] == 1]["ticker"].nunique()
    log.info(f"  Earnings flags: {n_earn:,} rows | {n_earn_tickers} tickers")

    # ── Finalise ──────────────────────────────────────────────────────────────
    base_and_target = BASE_COLS
    feature_cols    = [c for c in features.columns if c not in base_and_target]
    features[feature_cols] = features[feature_cols].astype(float).round(6)
    features = features.sort_values(
        ["universe", "ticker", "date"]).reset_index(drop=True)

    features.to_csv(TECH_CSV, index=False)

    pred_rows = features[features["is_prediction_row"] == 1]
    log.info(f"")
    log.info(f"[SAVED] {TECH_CSV}")
    log.info(f"  Total rows       : {len(features):,}")
    log.info(f"  Feature columns  : {len(feature_cols)}")
    log.info(f"  Prediction rows  : {len(pred_rows):,}")
    log.info(f"  Prediction base  : {pred_rows['date'].max().date()}")

    return TECH_CSV


# =============================================================================
# PART 2 — FEATURE FUSION
# =============================================================================

def safe_col(df: pd.DataFrame, col: str, default=0.0) -> pd.Series:
    """Return column if present, else Series of default values."""
    return (df[col] if col in df.columns
            else pd.Series(default, index=df.index))


def run_fusion(log):
    """
    Step 2: merge technical + insider + sentiment + macro, add ranks
    and interaction features, save final feature matrices.
    """


    log.info("")
    log.info("=" * 60)
    log.info("STEP 2 — FEATURE FUSION")
    log.info("=" * 60)
    log.info(f"Technical : {TECH_CSV}")
    log.info(f"Insider   : {INSIDER_CSV}")
    log.info(f"Sentiment : {SENT_CSV}")
    log.info(f"SP500 out : {SP500_CSV}")
    log.info(f"SC out    : {SC_CSV}")

    for path in [TECH_CSV, INSIDER_CSV, SENT_CSV]:
        if not os.path.exists(path):
            log.error(f"Required input not found: {path}")
            sys.exit(1)

    # ── Load ──────────────────────────────────────────────────────────────────
    log.info("Loading feature files...")
    tech    = pd.read_csv(TECH_CSV,    parse_dates=["date"])
    insider = pd.read_csv(INSIDER_CSV, parse_dates=["date"])
    sent    = pd.read_csv(SENT_CSV,    parse_dates=["date"])

    log.info(f"  Technical : {len(tech):,} rows × {tech.shape[1]} cols")
    log.info(f"  Insider   : {len(insider):,} rows × {insider.shape[1]} cols")
    log.info(f"  Sentiment : {len(sent):,} rows × {sent.shape[1]} cols")

    # Check v3 columns arrived
    v3_missing = [c for c in PASSTHROUGH_COLS if c not in tech.columns]
    if v3_missing:
        log.warning(f"v3 passthrough columns missing (run feature_pipeline "
                    f"--step features first): {v3_missing}")

    # Identify per-source feature columns
    tech_feat_cols    = [c for c in tech.columns if c not in BASE_COLS]
    insider_feat_cols = [c for c in insider.columns
                         if c not in BASE_COLS + tech_feat_cols]
    sent_feat_cols    = [c for c in sent.columns
                         if c not in BASE_COLS + tech_feat_cols
                                               + insider_feat_cols]

    log.info(f"  Technical  : {len(tech_feat_cols)} feature cols")
    log.info(f"  Insider    : {len(insider_feat_cols)} feature cols")
    log.info(f"  Sentiment  : {len(sent_feat_cols)} feature cols")

    # ── Filter thin tickers ───────────────────────────────────────────────────
    log.info("Filtering tickers with sufficient history (≥50 rows)...")
    valid = tech.groupby("ticker").size()
    valid = valid[valid >= 50].index
    before = tech["ticker"].nunique()
    tech    = tech[tech["ticker"].isin(valid)].copy()
    insider = insider[insider["ticker"].isin(valid)].copy()
    sent    = sent[sent["ticker"].isin(valid)].copy()
    log.info(f"  Dropped {before - tech['ticker'].nunique()} thin tickers")

    # ── Merge ─────────────────────────────────────────────────────────────────
    log.info("Merging all feature sources...")
    merged = tech.copy()
    merged = merged.merge(
        insider[["ticker", "date"] + insider_feat_cols],
        on=["ticker", "date"], how="left")
    merged = merged.merge(
        sent[["ticker", "date"] + sent_feat_cols],
        on=["ticker", "date"], how="left")

    all_feat_cols = tech_feat_cols + insider_feat_cols + sent_feat_cols

    non_pt = [c for c in all_feat_cols if c not in PASSTHROUGH_COLS]
    merged[non_pt] = merged[non_pt].fillna(0)
    for col, default in PT_DEFAULTS.items():
        if col in merged.columns:
            merged[col] = merged[col].fillna(default)

    log.info(f"  Merged: {len(merged):,} rows | "
             f"latest={merged['date'].max().date()}")

    # ── near_earnings (from technical file) ───────────────────────────────────
    if "near_earnings" not in merged.columns:
        log.warning("near_earnings not in technical file — set to 0")
        merged["near_earnings"] = 0
    else:
        pct = merged["near_earnings"].mean() * 100
        log.info(f"  near_earnings: {pct:.1f}% rows flagged")
    merged["near_earnings"] = merged["near_earnings"].fillna(0).astype(int)

    # ── Cross-sectional ranks ─────────────────────────────────────────────────
    log.info("Computing cross-sectional rank features...")
    rank_added = []
    for col in RANK_COLS:
        if col not in merged.columns:
            log.info(f"  [SKIP] {col} not in merged")
            continue
        rank_col = f"{col}_rank"
        merged[rank_col] = (
            merged.groupby(["date", "universe"])[col]
            .rank(pct=True, na_option="keep"))
        rank_added.append(rank_col)
    log.info(f"  Added {len(rank_added)} rank features")

    # ── 5-day target ──────────────────────────────────────────────────────────
    log.info("Computing 5-day return targets...")
    if "target_5d_return" not in merged.columns:
        merged["target_5d_return"] = (
            merged.groupby("ticker")["adj_close"]
            .transform(lambda x: x.shift(-5) / x - 1)
            .clip(-0.20, 0.20))
    else:
        merged["target_5d_return"] = merged["target_5d_return"].clip(-0.20, 0.20)

    merged["_mkt"] = np.nan
    for univ in ["sp500", "small_cap"]:
        mask      = merged["universe"] == univ
        valid_m   = merged[mask & merged["target_5d_return"].notna()]
        daily_mkt = valid_m.groupby("date")["target_5d_return"].mean()
        idx       = merged.index[mask]
        merged.loc[idx, "_mkt"] = merged.loc[idx, "date"].map(daily_mkt).values

    merged["target_excess_return"] = (
        merged["target_5d_return"] - merged["_mkt"]).clip(-0.15, 0.15)
    merged["target_excess_direction"] = (
        merged["target_excess_return"] > 0).astype("Int8")
    merged = merged.drop(columns=["_mkt"])

    merged["is_prediction_row"] = merged["target_5d_return"].isna().astype(int)
    merged["target_excess_return"]    = merged["target_excess_return"].fillna(0)
    merged["target_excess_direction"] = (merged["target_excess_direction"]
                                          .fillna(0).astype("Int8"))

    pred_rows  = merged[merged["is_prediction_row"] == 1]
    train_rows = merged[merged["is_prediction_row"] == 0]
    log.info(f"  Training rows   : {len(train_rows):,}")
    log.info(f"  Prediction rows : {len(pred_rows):,}")

    # ── Macro features ────────────────────────────────────────────────────────
    macro_cols = []
    if os.path.exists(MACRO_CSV):
        log.info("Merging macro features...")
        macro      = pd.read_csv(MACRO_CSV, parse_dates=["date"])
        macro_cols = [c for c in macro.columns if c != "date"]
        merged     = merged.merge(macro, on="date", how="left")
        merged[macro_cols] = merged[macro_cols].ffill().fillna(0)
        log.info(f"  Added {len(macro_cols)} macro features")
        all_feat_cols += macro_cols
    else:
        log.warning("macro_features.csv not found — skipping macro features")

    # ── Sector-relative ranks ─────────────────────────────────────────────────
    sector_rank_added = []
    if os.path.exists(GICS_CSV):
        log.info("Computing sector-relative rank features (from GICS file)...")
        gics   = pd.read_csv(GICS_CSV)[["ticker", "sector"]]
        merged = merged.merge(gics, on="ticker", how="left")
        merged["sector"] = merged["sector"].fillna("Unknown")

        for col in SECTOR_RANK_COLS:
            if col not in merged.columns:
                continue
            rank_col = f"{col}_sector_rank"
            merged[rank_col] = (
                merged.groupby(["date", "sector"])[col]
                .rank(pct=True, na_option="keep"))
            sector_rank_added.append(rank_col)

        log.info(f"  Added {len(sector_rank_added)} sector-relative ranks")
        all_feat_cols += sector_rank_added
    else:
        log.info("Computing sector-relative rank features (built-in SECTOR_MAP)...")
        # Apply built-in map
        merged["sector"] = merged["ticker"].map(SECTOR_MAP).fillna("Unknown")
        for col in SECTOR_RANK_COLS:
            if col not in merged.columns:
                continue
            rank_col = f"{col}_sector_rank"
            merged[rank_col] = (
                merged.groupby(["date", "sector"])[col]
                .rank(pct=True, na_option="keep"))
            sector_rank_added.append(rank_col)
        log.info(f"  Added {len(sector_rank_added)} sector-relative ranks "
                 "(from built-in SECTOR_MAP)")
        all_feat_cols += sector_rank_added

    # ── Interaction features (9 total) ────────────────────────────────────────
    log.info("Computing interaction features (7 existing + 2 new v3)...")

    merged["ix_rsi_vix"] = (
        safe_col(merged, "rsi_14_rank") *
        (1 - safe_col(merged, "vix_proxy")
              .rank(pct=True).reindex(merged.index).fillna(0.5)))
    merged["ix_sent_riskOn"] = (
        safe_col(merged, "sentiment_mean_7_rank") *
        safe_col(merged, "risk_on_score", 0.5))
    merged["ix_silence_gc"] = (
        safe_col(merged, "insider_sell_silence_rank") *
        safe_col(merged, "golden_cross", 0))
    merged["ix_mom_breadth"] = (
        safe_col(merged, "momentum_5_rank") *
        safe_col(merged, "spy_20d_return", 0.0))
    merged["ix_vol_spyMom"] = (
        safe_col(merged, "hist_vol_20_rank") *
        safe_col(merged, "spy_20d_return", 0.0))
    merged["ix_mom_yc"] = (
        safe_col(merged, "momentum_5_rank") *
        safe_col(merged, "yield_curve", 0.5))
    merged["ix_rsi_regime"] = (
        safe_col(merged, "rsi_14_rank") *
        safe_col(merged, "regime_score", 0.5))

    # New v3
    merged["ix_volspike_breadth"] = (
        safe_col(merged, "vol_spike_ratio_rank") *
        (1 - safe_col(merged, "breadth_5d", 0.5)))
    merged["ix_breadth_mom"] = (
        safe_col(merged, "momentum_5_rank") *
        safe_col(merged, "breadth_5d", 0.5))

    ix_cols = [
        "ix_rsi_vix", "ix_sent_riskOn", "ix_silence_gc",
        "ix_mom_breadth", "ix_vol_spyMom", "ix_mom_yc", "ix_rsi_regime",
        "ix_volspike_breadth", "ix_breadth_mom",
    ]
    all_feat_cols += rank_added + ix_cols

    # ── Sanity check ──────────────────────────────────────────────────────────
    log.info("Sanity check...")
    feat_check = [c for c in all_feat_cols if c in merged.columns]
    nan_check  = merged[feat_check].isna().sum()
    nan_check  = nan_check[nan_check > 0]
    if len(nan_check):
        log.warning(f"NaNs in {len(nan_check)} cols — filling 0")
        merged[feat_check] = merged[feat_check].fillna(0)
    else:
        log.info("  ✓ No NaNs in feature columns")

    # ── Save ──────────────────────────────────────────────────────────────────
    merged = merged.sort_values(
        ["universe", "ticker", "date"]).reset_index(drop=True)
    df_sp500 = merged[merged["universe"] == "sp500"].copy()
    df_sc    = merged[merged["universe"] == "small_cap"].copy()

    df_sp500.to_csv(SP500_CSV, index=False)
    df_sc.to_csv(SC_CSV,       index=False)

    all_feat_dedup = list(dict.fromkeys(all_feat_cols))
    n_total = (len(tech_feat_cols) + len(insider_feat_cols)
               + len(sent_feat_cols) + len(macro_cols)
               + len(sector_rank_added) + len(rank_added) + len(ix_cols))

    log.info("")
    log.info(f"[SAVED] {SP500_CSV}")
    log.info(f"        {len(df_sp500):,} rows | "
             f"{df_sp500['ticker'].nunique()} tickers | "
             f"latest={df_sp500['date'].max().date()}")
    log.info(f"[SAVED] {SC_CSV}")
    log.info(f"        {len(df_sc):,} rows | "
             f"{df_sc['ticker'].nunique()} tickers | "
             f"latest={df_sc['date'].max().date()}")
    log.info("")
    log.info("FEATURE MATRIX SUMMARY")
    log.info(f"  Technical base        : {len(tech_feat_cols)}")
    log.info(f"  Insider               : {len(insider_feat_cols)}")
    log.info(f"  Sentiment             : {len(sent_feat_cols)}")
    log.info(f"  Macro                 : {len(macro_cols)}")
    log.info(f"  Sector-relative ranks : {len(sector_rank_added)}")
    log.info(f"  Cross-sect ranks      : {len(rank_added)}")
    log.info(f"  Interaction features  : {len(ix_cols)}")
    log.info(f"  Total (approx)        : {n_total}")
    log.info(f"  Prediction base       : {merged['date'].max().date()}")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================


# =============================================================================
#  LOGGING  (appends to LOG_FILE each run)
# =============================================================================

def setup_logging():
    fmt      = "%(asctime)s [%(levelname)s] %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    return logging.getLogger("feature_pipeline")


# =============================================================================
#  MAIN
# =============================================================================

if __name__ == "__main__":
    # Optional: pass "features" or "fusion" as first argument to run one step.
    # Default (no argument): runs both steps in sequence.
    step = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    if step not in ("all", "features", "fusion"):
        print("Usage: python feature_pipeline.py [all | features | fusion]")
        sys.exit(1)

    os.makedirs(DATA_DIR, exist_ok=True)
    log = setup_logging()

    t0 = time.time()
    log.info("=" * 60)
    log.info("  FEATURE PIPELINE  —  Multimodal Stock Advisory System")
    log.info("  Stella Umunna | DEng AI | GWU")
    log.info(f"  Run     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Data    : {DATA_DIR}")
    log.info(f"  Step    : {step}")
    log.info("=" * 60)

    if step in ("all", "features"):
        run_feature_engineering(log)

    if step in ("all", "fusion"):
        run_fusion(log)

    elapsed = time.time() - t0
    log.info("")
    log.info(f"Pipeline complete in {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    log.info("")
    log.info("NEXT STEP:  python regime_gated_model.py")