# ============================================================
# build_macro_features.py
#
# Standalone script — no Google Colab required.
# Run from any machine with yfinance, pandas, numpy installed:
#
#   pip install yfinance pandas numpy
#   python build_macro_features.py
#
# CONFIG (edit the CONFIG block below):
#   DATA_DIR  — folder where macro_features.csv will be saved
#               and where the four feature files live
#   START     — earliest date to pull from Yahoo Finance
#   END       — set to "today" or a fixed date string "YYYY-MM-DD"
#   MERGE_FILES — set to False to skip merging into feature CSVs
#
# Features built (~35 total):
#   VIX level + regime (calm/normal/elevated/fear)
#   VIX term structure (VIX9D vs VIX)
#   10Y / 3M / 30Y treasury yields + changes
#   Yield curve slope + inversion flag
#   Dollar index level + trend
#   SPY / QQQ momentum
#   Credit spread proxy (HYG vs LQD)
#   Gold signal (GLD 5d / 20d returns)
#   TLT bond signal
#   Market breadth proxy (% days SPY up over 20d)
#   Composite risk-on / risk-off score + regime flag
# ============================================================

import os
import sys
import warnings
from datetime import date

import numpy  as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG — edit these paths before running
# ============================================================
DATA_DIR    = "/content/drive/MyDrive/AI_PROJECT/Data"   # <-- change me
START       = "2017-01-01"
END         = "today"          # or e.g. "2026-05-25"
MERGE_FILES = True             # merge macro cols into feature CSVs?

SP500_REG = os.path.join(DATA_DIR, "features_final_sp500_regime.csv")
SC_REG    = os.path.join(DATA_DIR, "features_final_smallcap_regime.csv")
SP500_STD = os.path.join(DATA_DIR, "features_final_sp500.csv")
SC_STD    = os.path.join(DATA_DIR, "features_final_smallcap.csv")
MACRO_CSV = os.path.join(DATA_DIR, "macro_features.csv")

# ============================================================
# Step 1: Download macro data from Yahoo Finance
# ============================================================
end_str = str(date.today()) if END == "today" else END
print(f"[INFO] Downloading macro data  {START} → {end_str}")

TICKERS = {
    "^VIX":     "vix",        # CBOE VIX
    "^VIX9D":   "vix9d",      # 9-day VIX
    "^TNX":     "yield_10y",  # 10-year treasury yield
    "^IRX":     "yield_3m",   # 3-month treasury yield
    "^TYX":     "yield_30y",  # 30-year treasury yield
    "DX-Y.NYB": "dxy",        # US dollar index
    "SPY":      "spy",        # S&P 500 ETF
    "QQQ":      "qqq",        # Nasdaq ETF
    "HYG":      "hyg",        # High yield bonds
    "LQD":      "lqd",        # Investment grade bonds
    "GLD":      "gld",        # Gold
    "TLT":      "tlt",        # 20+ year treasury ETF
    "IWM":      "iwm",        # Russell 2000 small cap
}

raw = {}
for ticker, name in TICKERS.items():
    try:
        df_t = yf.download(ticker, start=START, end=end_str,
                           progress=False, auto_adjust=True)
        if len(df_t) > 0:
            raw[name] = df_t["Close"].squeeze()
            print(f"  ✓ {ticker:<12} ({name:<12}) "
                  f"{len(df_t)} rows  latest={df_t.index[-1].date()}")
        else:
            print(f"  ✗ {ticker:<12} — no data returned")
    except Exception as e:
        print(f"  ✗ {ticker:<12} — error: {e}")

if not raw:
    sys.exit("[ERROR] No data downloaded. Check your internet connection.")

macro_raw = pd.DataFrame(raw)
macro_raw.index.name = "date"
macro_raw = macro_raw.sort_index()
print(f"\n[INFO] Raw macro: {len(macro_raw)} dates × {len(macro_raw.columns)} series")

# ============================================================
# Step 2: Engineer macro features
# ============================================================
print("\n[INFO] Engineering macro features...")
macro = pd.DataFrame(index=macro_raw.index)

# ── VIX ──────────────────────────────────────────────────────
if "vix" in macro_raw.columns:
    macro["vix_level"]       = macro_raw["vix"]
    macro["vix_5d_change"]   = macro_raw["vix"].pct_change(5)
    macro["vix_20d_change"]  = macro_raw["vix"].pct_change(20)
    macro["vix_ma20"]        = macro_raw["vix"].rolling(20).mean()
    macro["vix_above_ma20"]  = (macro_raw["vix"] > macro["vix_ma20"]).astype(int)
    macro["vix_regime"]      = np.select(          # 0=calm 1=normal 2=elevated 3=fear
        [macro_raw["vix"] < 15, macro_raw["vix"] < 25, macro_raw["vix"] < 35],
        [0, 1, 2], default=3)
    print("  VIX features: 6 ✓")

# ── VIX term structure ────────────────────────────────────────
if "vix9d" in macro_raw.columns and "vix" in macro_raw.columns:
    macro["vix_term_structure"] = macro_raw["vix"] - macro_raw["vix9d"]
    macro["vix_backwardation"]  = (macro["vix_term_structure"] < 0).astype(int)
    print("  VIX term structure: 2 ✓")

# ── Treasury yields ───────────────────────────────────────────
if "yield_10y" in macro_raw.columns:
    macro["yield_10y"]         = macro_raw["yield_10y"]
    macro["yield_10y_5d_chg"]  = macro_raw["yield_10y"].diff(5)
    macro["yield_10y_20d_chg"] = macro_raw["yield_10y"].diff(20)
    print("  10Y yield: 3 ✓")

if "yield_3m" in macro_raw.columns:
    macro["yield_3m"] = macro_raw["yield_3m"]
    print("  3M yield: 1 ✓")

if "yield_30y" in macro_raw.columns:
    macro["yield_30y"] = macro_raw["yield_30y"]
    print("  30Y yield: 1 ✓")

# ── Yield curve ───────────────────────────────────────────────
if "yield_10y" in macro_raw.columns and "yield_3m" in macro_raw.columns:
    macro["yield_curve_slope"]    = macro_raw["yield_10y"] - macro_raw["yield_3m"]
    macro["yield_curve_inverted"] = (macro["yield_curve_slope"] < 0).astype(int)
    print("  Yield curve: 2 ✓")

# ── Dollar index ──────────────────────────────────────────────
if "dxy" in macro_raw.columns:
    macro["dxy_level"]      = macro_raw["dxy"]
    macro["dxy_5d_ret"]     = macro_raw["dxy"].pct_change(5)
    macro["dxy_20d_ret"]    = macro_raw["dxy"].pct_change(20)
    macro["dxy_ma20"]       = macro_raw["dxy"].rolling(20).mean()
    macro["dxy_above_ma20"] = (macro_raw["dxy"] > macro["dxy_ma20"]).astype(int)
    print("  DXY: 5 ✓")

# ── SPY momentum ─────────────────────────────────────────────
if "spy" in macro_raw.columns:
    macro["spy_5d_ret"]     = macro_raw["spy"].pct_change(5)
    macro["spy_20d_ret"]    = macro_raw["spy"].pct_change(20)
    macro["spy_ma50"]       = macro_raw["spy"].rolling(50).mean()
    macro["spy_above_ma50"] = (macro_raw["spy"] > macro["spy_ma50"]).astype(int)
    print("  SPY momentum: 4 ✓")

# ── QQQ momentum ─────────────────────────────────────────────
if "qqq" in macro_raw.columns:
    macro["qqq_5d_ret"]  = macro_raw["qqq"].pct_change(5)
    macro["qqq_20d_ret"] = macro_raw["qqq"].pct_change(20)
    print("  QQQ momentum: 2 ✓")

# ── Credit spread proxy (HYG vs LQD) ─────────────────────────
if "hyg" in macro_raw.columns and "lqd" in macro_raw.columns:
    macro["credit_spread_proxy"]  = macro_raw["lqd"] / macro_raw["hyg"]
    macro["credit_spread_5d_chg"] = macro["credit_spread_proxy"].pct_change(5)
    macro["credit_risk_off"]      = (macro["credit_spread_5d_chg"] > 0.01).astype(int)
    print("  Credit spread: 3 ✓")

# ── Gold ──────────────────────────────────────────────────────
if "gld" in macro_raw.columns:
    macro["gld_5d_ret"]  = macro_raw["gld"].pct_change(5)
    macro["gld_20d_ret"] = macro_raw["gld"].pct_change(20)
    macro["gld_risk_off"] = (macro["gld_5d_ret"] > 0.02).astype(int)
    print("  Gold: 3 ✓")

# ── TLT (long bonds) ─────────────────────────────────────────
if "tlt" in macro_raw.columns:
    macro["tlt_5d_ret"]  = macro_raw["tlt"].pct_change(5)
    macro["tlt_20d_ret"] = macro_raw["tlt"].pct_change(20)
    macro["bond_demand"] = (macro["tlt_5d_ret"] > 0).astype(int)
    print("  TLT bonds: 3 ✓")

# ── Market breadth proxy ──────────────────────────────────────
if "spy" in macro_raw.columns:
    spy_up = (macro_raw["spy"].pct_change() > 0).rolling(20).mean()
    macro["market_breadth_20d"] = spy_up
    macro["breadth_expanding"]  = (spy_up > 0.6).astype(int)
    print("  Market breadth: 2 ✓")

# ── Composite risk-on / risk-off score ────────────────────────
risk_off_signals = [
    macro[col] for col in
    ["vix_above_ma20", "yield_curve_inverted",
     "credit_risk_off", "bond_demand", "gld_risk_off"]
    if col in macro.columns
]
if risk_off_signals:
    macro["risk_off_score"] = sum(risk_off_signals) / len(risk_off_signals)
    macro["risk_on_score"]  = 1 - macro["risk_off_score"]
    macro["risk_regime"]    = (macro["risk_on_score"] > 0.6).astype(int)
    print("  Composite risk score: 3 ✓")

# ── Finalise ──────────────────────────────────────────────────
macro = macro.ffill().bfill().reset_index()
macro["date"] = pd.to_datetime(macro["date"])
if macro["date"].dt.tz is not None:
    macro["date"] = macro["date"].dt.tz_localize(None)

n_features = len(macro.columns) - 1   # exclude date
print(f"\n[INFO] Total macro features : {n_features}")
print(f"[INFO] Date range           : "
      f"{macro['date'].min().date()} → {macro['date'].max().date()}")
print(f"[INFO] Rows                 : {len(macro):,}")

os.makedirs(DATA_DIR, exist_ok=True)
macro.to_csv(MACRO_CSV, index=False)
print(f"[SAVED] {MACRO_CSV}")

# ============================================================
# Step 3 (optional): Merge into feature files
# ============================================================
if not MERGE_FILES:
    print("\n[INFO] MERGE_FILES=False — skipping merge step.")
    print("[DONE]")
    sys.exit(0)

MACRO_COLS = [c for c in macro.columns if c != "date"]
print(f"\n[INFO] Merging {len(MACRO_COLS)} macro features into feature files...")

for fname, label in [
    (SP500_REG, "S&P 500 regime"),
    (SC_REG,    "Small cap regime"),
    (SP500_STD, "S&P 500 standard"),
    (SC_STD,    "Small cap standard"),
]:
    if not os.path.exists(fname):
        print(f"  [SKIP] {label} — file not found: {fname}")
        continue

    print(f"\n  Loading {label}...")
    df = pd.read_csv(fname, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    n_before = len(df)

    # Drop any stale macro columns before re-merging
    stale = [c for c in MACRO_COLS if c in df.columns]
    if stale:
        df = df.drop(columns=stale)
        print(f"    Dropped {len(stale)} existing macro cols")

    df = df.merge(macro[["date"] + MACRO_COLS], on="date", how="left")

    # Forward-fill within each ticker (handles missing macro dates)
    df = df.sort_values(["ticker", "date"])
    df[MACRO_COLS] = (
        df.groupby("ticker")[MACRO_COLS]
          .transform(lambda x: x.ffill().bfill())
    )
    df[MACRO_COLS] = df[MACRO_COLS].fillna(0)

    assert len(df) == n_before, \
        f"Row count changed {n_before} → {len(df)} in {label}"

    df.to_csv(fname, index=False)
    print(f"    Saved {len(df):,} rows → {fname}")

print("\n[DONE] macro_features.csv built and merged into all feature files.")