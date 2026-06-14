# ============================================================
# train_regime_gated_model.py
#
# Standalone Regime-Gated Model — Multimodal Stock Advisory System
# Stella Umunna | DEng AI | GWU
#
# Architecture:
#   1. Importance-weighted training  (BULL 2.5x, BEAR 2.5x — symmetric Fix 3)
#   2. Walk-forward purged CV        (5 folds, purge=5d, embargo=2d)
#   3. Transformer + LightGBM ensemble
#   4. Meta-learner (logistic regression on OOF preds)
#   5. Regime gate at prediction time (RS-scaled confidence)
#   6. Separate models for S&P 500 and Small Cap
#
# Inputs  (all in DATA_DIR):
#   features_final_sp500_regime.csv
#   features_final_smallcap_regime.csv
#   macro_features.csv              (optional — merged if present)
#
# Outputs (all in DATA_DIR):
#   advisory_sp500.csv
#   advisory_smallcap.csv
#   advisory_combined.csv
#   regime_gated_model_meta.json
#
# Install:
#   pip install pandas numpy scikit-learn lightgbm torch
#
# Run:
#   python train_regime_gated_model.py
#   python train_regime_gated_model.py --data-dir "G:\My Drive\AI_PROJECT\Data"
#   python train_regime_gated_model.py --no-transformer   # faster, CPU-only
# ============================================================

import os
import sys
import json
import math
import warnings
import argparse
import logging
from datetime import datetime

import numpy  as np
import pandas as pd
from sklearn.preprocessing  import StandardScaler
from sklearn.linear_model   import LogisticRegression
from sklearn.calibration    import CalibratedClassifierCV
from sklearn.metrics        import roc_auc_score, accuracy_score

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[WARN] lightgbm not installed — LGB step skipped")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] torch not installed — Transformer step skipped")

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
DEFAULT_DATA_DIR = r"G:\My Drive\AI_PROJECT\Data"

# Targets
TARGET_CLF  = "target_5d_direction"   # binary 0/1
TARGET_REG  = "target_5d_return"      # continuous return

# Crash window to exclude (COVID crash distorts weighting)
CRASH_START = pd.Timestamp("2020-02-15")
CRASH_END   = pd.Timestamp("2020-06-01")

# Walk-forward CV
N_FOLDS      = 5
VAL_MONTHS   = 3
PURGE_DAYS   = 5
EMBARGO_DAYS = 2
FOLD_START   = pd.Timestamp("2023-01-01")   # earlier data all used as training

# Regime weighting — Fix 3: symmetric bear weighting
# BULL and BEAR both weighted equally so model learns both regimes
# Previous: BEAR_WEIGHT = 0.5 (bear underweighted → poor bear performance)
# Fix 3:    BEAR_WEIGHT = 2.5 (symmetric → model sees bear patterns equally)
BULL_WEIGHT  = 2.5
BEAR_WEIGHT  = 2.5   # was 0.5 — raised to match BULL for symmetric learning
WEIGHT_FLOOR = 0.15
GATE_MIN     = 0.15

# Transformer config (memory-safe)
D_MODEL      = 32
N_HEADS      = 4
N_LAYERS     = 2
DROPOUT      = 0.1
MAX_EPOCHS   = 30
PATIENCE     = 5
BATCH_SIZE   = 512

# Advisory filter thresholds — set as TOP-N percentile of scored tickers
# (adaptive to actual score distribution rather than fixed values)
SP500_T1_PCT   = 0.85   # top 15% = T1
SP500_T2_PCT   = 0.60   # top 40% = T2
SP500_MIN_RET  = 0.0
SP500_TIERS    = {"T1": SP500_T1_PCT, "T2": SP500_T2_PCT}

SC_T1_PCT      = 0.85
SC_T2_PCT      = 0.60
SC_MIN_RET     = 0.0
SC_TIERS       = {"T1": SC_T1_PCT, "T2": SC_T2_PCT}


# ============================================================
# Logging
# ============================================================
def setup_logger():
    log = logging.getLogger("regime_gated")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    return log

log = setup_logger()


# ============================================================
# Feature helpers
# ============================================================
def compute_regime_score(df: pd.DataFrame) -> pd.Series:
    """
    Composite bull/risk-on score [0, 1].
    Uses available regime-related columns; degrades gracefully.
    """
    signals = []
    w       = []

    col_weights = {
        "spy_above_200":   0.25,
        "spy_golden_cross":0.20,
        "risk_on_score":   0.25,
        "vix_regime":      0.15,   # inverted: low vix = bull
        "bull":            0.15,
    }
    for col, wt in col_weights.items():
        if col in df.columns:
            s = df[col].fillna(0).astype(float)
            if col == "vix_regime":
                s = 1 - (s / 3.0)   # invert: 0=fear→0, 3=calm→1
            signals.append(s * wt)
            w.append(wt)

    if not signals:
        return pd.Series(0.5, index=df.index)

    total_w = sum(w)
    score   = sum(signals) / total_w
    return score.clip(GATE_MIN, 1.0)


def add_interaction_features(df: pd.DataFrame,
                              feat_cols: list) -> tuple:
    """Add regime × signal interaction terms."""
    df = df.copy()
    new_cols = []

    interaction_pairs = [
        ("bull",   "insider_strength"),
        ("bull",   "sentiment_score"),
        ("bull",   "rsi14"),
        ("bull",   "spy_20d_ret"),
        ("bear",   "insider_strength"),
        ("risk_on_score", "spy_20d_ret"),
        ("risk_on_score", "insider_strength"),
        ("vix_level",     "spy_20d_ret"),
    ]

    for col_a, col_b in interaction_pairs:
        if col_a in df.columns and col_b in df.columns:
            new_name = f"{col_a}_x_{col_b}"
            df[new_name] = (df[col_a].fillna(0) *
                            df[col_b].fillna(0))
            new_cols.append(new_name)

    extended = feat_cols + [c for c in new_cols
                            if c not in feat_cols]
    return df, extended


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return model-usable feature columns."""
    exclude = {
        "ticker", "date", "asof_date", "universe",
        "open", "high", "low", "close", "adj_close", "volume",
        TARGET_CLF, TARGET_REG,
        "future_ret", "future_close",
        "is_prediction_row", "_bin",
        "market_regime",
    }
    cols = [c for c in df.columns
            if c not in exclude
            and pd.api.types.is_numeric_dtype(df[c])]
    return cols


def safe_fmt(val, fmt=".4f"):
    try:
        return f"{float(val):{fmt}}"
    except Exception:
        return "N/A"


# ============================================================
# Transformer (memory-safe)
# ============================================================
class RegimeTransformer(nn.Module):
    def __init__(self, n_features, d_model=D_MODEL,
                 n_heads=N_HEADS, n_layers=N_LAYERS,
                 dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1))

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        x = self.encoder(x).squeeze(1)
        return self.head(x).squeeze(-1)


def train_transformer(X_tr, y_tr, X_val, y_val,
                      weights=None, use_gpu=False):
    if not HAS_TORCH:
        return None

    device = torch.device(
        "cuda" if (use_gpu and torch.cuda.is_available())
        else "cpu")
    log.info(f"      Transformer device: {device}")

    n_feat = X_tr.shape[1]
    model  = RegimeTransformer(n_feat).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    log.info(f"      Params: {sum(p.numel() for p in model.parameters()):,}"
             f"  d_model={D_MODEL}  layers={N_LAYERS}")

    X_t = torch.FloatTensor(X_tr)
    y_t = torch.FloatTensor(y_tr)
    w_t = (torch.FloatTensor(weights)
           if weights is not None
           else torch.ones(len(y_t)))

    # Val always on CPU to save VRAM
    X_v = torch.FloatTensor(X_val)
    y_v = torch.FloatTensor(y_val)

    dataset = torch.utils.data.TensorDataset(X_t, y_t, w_t)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True)

    best_loss  = float("inf")
    best_state = None
    patience_ct = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad()
            loss = (loss_fn(model(xb), yb) * wb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # Val on CPU
        model.eval()
        with torch.no_grad():
            val_logits = model(X_v.to("cpu"))
            val_loss   = loss_fn(val_logits, y_v).mean().item()

        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_ct = 0
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE:
                log.info(f"      Early stop ep {epoch+1}"
                         f" (best_loss={best_loss:.4f})")
                break

    if best_state:
        model.load_state_dict(best_state)
    model = model.to("cpu")
    return model


def predict_transformer(model, X):
    if model is None:
        return np.full(len(X), 0.5)
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X))
        return torch.sigmoid(logits).numpy()


# ============================================================
# LightGBM helper
# ============================================================
def safe_lgb_fit(clf, X_tr, y_tr, X_val, y_val,
                 sample_weight=None):
    """Fit LGB with fallback if GPU OOM."""
    try:
        clf.fit(X_tr, y_tr,
                sample_weight=sample_weight,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(20, verbose=False),
                           lgb.log_evaluation(-1)])
    except Exception as e:
        log.warning(f"      LGB fit error ({e}) — retrying CPU")
        clf.set_params(device="cpu")
        clf.fit(X_tr, y_tr,
                sample_weight=sample_weight,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(20, verbose=False),
                           lgb.log_evaluation(-1)])
    return clf


# ============================================================
# Regime-gated training
# ============================================================
def train_regime_gated(universe_name: str,
                       df: pd.DataFrame,
                       feat_cols: list,
                       use_transformer: bool = True):
    log.info(f"\n{'='*65}")
    log.info(f"  REGIME-GATED — {universe_name}")
    log.info(f"  1. Importance-weighted (BULL {BULL_WEIGHT}x / BEAR {BEAR_WEIGHT}x — symmetric)")
    log.info(f"  2. Regime gate (RS-scaled predictions)")
    log.info(f"  3. Transformer (d_model={D_MODEL}) + LightGBM")
    log.info(f"{'='*65}")

    df = df.sort_values("date").reset_index(drop=True)

    # Split prediction vs training rows
    if "is_prediction_row" in df.columns:
        df_pred  = df[df["is_prediction_row"] == 1].copy()
        df_train = df[df["is_prediction_row"] == 0].copy()
    else:
        latest   = df["date"].max()
        df_pred  = df[df["date"] == latest].copy()
        df_train = df[df["date"] <  latest].copy()

    pred_latest = df_pred["date"].max()
    df_pred     = df_pred[df_pred["date"] == pred_latest].copy()

    # Clip target, exclude crash window
    df_train[TARGET_REG] = df_train[TARGET_REG].clip(-0.20, 0.20)
    df_train["_bin"]     = (df_train[TARGET_CLF] > 0).astype(int)

    crash    = ((df_train["date"] >= CRASH_START) &
                (df_train["date"] <= CRASH_END))
    df_train = df_train[~crash].copy().reset_index(drop=True)

    # Market regime per row
    if "regime" in df_train.columns:
        mr = df_train.groupby("date")["regime"].transform("median")
        df_train["market_regime"] = np.where(
            mr > 0, 1, np.where(mr < 0, -1, 0))
    else:
        df_train["market_regime"] = 0

    rd = df_train["market_regime"].value_counts()
    log.info(f"  Prediction base   : {pred_latest.date()}")
    log.info(f"  Prediction tickers: {df_pred['ticker'].nunique()}")
    log.info(f"  Training rows     : {len(df_train):,}")
    log.info(f"  Crash excluded    : {crash.sum():,}")
    log.info(f"  Regime dist: Bull={rd.get(1,0):,}  "
             f"Neutral={rd.get(0,0):,}  Bear={rd.get(-1,0):,}")

    # Add interaction features
    log.info(f"\n  Adding interaction features...")
    df_train, feat_ext = add_interaction_features(df_train, feat_cols)
    feat_ext = [f for f in feat_ext if f in df_train.columns]
    log.info(f"  Total features: {len(feat_ext)}")

    # Regime scores
    rs_train = compute_regime_score(df_train)
    log.info(f"  Train RS: mean={rs_train.mean():.3f}  "
             f"min={rs_train.min():.3f}  max={rs_train.max():.3f}")
    df_pred_tmp, _ = add_interaction_features(df_pred, feat_cols)
    rs_pred = compute_regime_score(df_pred_tmp)
    log.info(f"  Pred  RS: mean={rs_pred.mean():.3f} "
             f"({'BULL' if rs_pred.mean() > 0.5 else 'FLAT'})")

    scaler = StandardScaler()

    # ── Walk-forward CV splits ────────────────────────────────
    splits = []
    for fold in range(N_FOLDS + 4):
        train_end    = FOLD_START + pd.DateOffset(months=fold * VAL_MONTHS)
        purge_cutoff = train_end  - pd.DateOffset(days=PURGE_DAYS)
        embargo_end  = train_end  + pd.DateOffset(days=EMBARGO_DAYS)
        val_end      = train_end  + pd.DateOffset(months=VAL_MONTHS)
        tr_idx  = df_train.index[df_train["date"] < purge_cutoff]
        val_idx = df_train.index[(df_train["date"] >= embargo_end) &
                                  (df_train["date"] <  val_end)]
        if len(tr_idx) < 5000 or len(val_idx) < 1000:
            continue
        splits.append((len(splits) + 1, tr_idx, val_idx))
        if len(splits) >= N_FOLDS:
            break

    # Fallback split schedule
    if not splits:
        for fold in range(N_FOLDS + 4):
            train_end    = pd.Timestamp("2021-01-01") + \
                           pd.DateOffset(months=fold * VAL_MONTHS)
            purge_cutoff = train_end - pd.DateOffset(days=PURGE_DAYS)
            embargo_end  = train_end + pd.DateOffset(days=EMBARGO_DAYS)
            val_end      = train_end + pd.DateOffset(months=VAL_MONTHS)
            tr_idx  = df_train.index[df_train["date"] < purge_cutoff]
            val_idx = df_train.index[(df_train["date"] >= embargo_end) &
                                      (df_train["date"] <  val_end)]
            if len(tr_idx) < 5000 or len(val_idx) < 1000:
                continue
            splits.append((len(splits) + 1, tr_idx, val_idx))
            if len(splits) >= N_FOLDS:
                break

    log.info(f"  CV folds: {len(splits)} "
             f"(purge={PURGE_DAYS}d, embargo={EMBARGO_DAYS}d)")
    for fold_n, tr_idx, val_idx in splits:
        td = df_train.loc[tr_idx,  "date"]
        vd = df_train.loc[val_idx, "date"]
        log.info(f"    Fold {fold_n}: train={len(tr_idx):,} "
                 f"({td.min().date()}→{td.max().date()})  "
                 f"val={len(val_idx):,} "
                 f"({vd.min().date()}→{vd.max().date()})")

    # ── CV loop ───────────────────────────────────────────────
    fold_results   = []
    meta_X_oof     = []
    meta_y_oof     = []
    transformer_models = []
    lgb_clf_models     = []
    lgb_reg_models     = []
    thresholds         = []

    for fold_n, tr_idx, val_idx in splits:
        tr  = df_train.loc[tr_idx].copy()
        val = df_train.loc[val_idx].copy()

        vix_val   = val["vix_level"].mean()  if "vix_level"  in val.columns else np.nan
        spy_val   = val["spy_20d_ret"].mean() if "spy_20d_ret" in val.columns else np.nan
        rs_val    = compute_regime_score(val).mean()
        regime_lbl = "BULL" if rs_val > 0.55 else ("BEAR" if rs_val < 0.40 else "FLAT")

        log.info(f"\n  Fold {fold_n}  val={len(val):,}  "
                 f"VIX={safe_fmt(vix_val,'.1f')}  "
                 f"SPY20d={safe_fmt(spy_val*100 if pd.notna(spy_val) else np.nan,'+.1f')}%  "
                 f"RS={safe_fmt(rs_val,'.2f')}  [{regime_lbl}]")

        # Importance weights — Fix 3: symmetric BULL=2.5x BEAR=2.5x
        rs_tr  = compute_regime_score(tr)
        w_base = np.ones(len(tr))
        bull_mask = tr["market_regime"] == 1
        bear_mask = tr["market_regime"] == -1
        w_base[bull_mask] = BULL_WEIGHT
        w_base[bear_mask] = BEAR_WEIGHT
        # Scale by RS — bear rows still get full weight regardless of RS
        w_scaled = w_base * (WEIGHT_FLOOR + (1 - WEIGHT_FLOOR) * rs_tr.values)
        # For bear rows, use inverted RS so low-RS (bear) rows get higher weight
        bear_rs  = 1 - rs_tr.values[bear_mask]   # high weight when RS is low
        w_scaled[bear_mask] = BEAR_WEIGHT * (WEIGHT_FLOOR + (1 - WEIGHT_FLOOR) * bear_rs)
        log.info(f"    Weights: bull={w_scaled[bull_mask].mean():.2f}x  "
                 f"bear={w_scaled[bear_mask].mean():.2f}x  "
                 f"(Fix 3: symmetric)")

        feat_valid = [f for f in feat_ext if f in tr.columns]
        X_tr  = tr[feat_valid].fillna(0).values
        y_tr  = tr["_bin"].values.astype(float)
        X_val = val[feat_valid].fillna(0).values
        y_val = val["_bin"].values.astype(float)

        X_tr_s  = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        val_preds = np.zeros((len(val), 2))   # [transformer, lgb]

        # ── [1] Transformer ────────────────────────────────
        if use_transformer and HAS_TORCH:
            log.info(f"    [1/4] Transformer (importance-weighted, val on CPU)...")
            t_model = train_transformer(
                X_tr_s, y_tr, X_val_s, y_val,
                weights=w_scaled)
            transformer_models.append(t_model)
            log.info(f"    [2/4] Val inference ({len(val):,} rows)...")
            val_preds[:, 0] = predict_transformer(t_model, X_val_s)
        else:
            transformer_models.append(None)
            val_preds[:, 0] = 0.5

        # ── [2] LightGBM classifier ────────────────────────
        if HAS_LGB:
            log.info(f"    [3/4] LightGBM (importance-weighted)...")
            lgb_clf = lgb.LGBMClassifier(
                n_estimators=500, learning_rate=0.02,
                max_depth=4, min_child_samples=30,
                subsample=0.8, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=2.0,
                random_state=42, n_jobs=-1, verbose=-1)
            safe_lgb_fit(lgb_clf, X_tr_s, y_tr,
                         X_val_s, y_val,
                         sample_weight=w_scaled)
            lgb_clf_models.append(lgb_clf)
            val_preds[:, 1] = lgb_clf.predict_proba(X_val_s)[:, 1]

            lgb_reg = lgb.LGBMRegressor(
                n_estimators=300, learning_rate=0.02,
                max_depth=4, min_child_samples=30,
                subsample=0.8, colsample_bytree=0.7,
                random_state=42, n_jobs=-1, verbose=-1)
            lgb_reg.fit(X_tr_s, tr[TARGET_REG].clip(-0.20, 0.20),
                        sample_weight=w_scaled)
            lgb_reg_models.append(lgb_reg)
        else:
            lgb_clf_models.append(None)
            lgb_reg_models.append(None)

        # ── [3] Meta-learner OOF ──────────────────────────
        ensemble_prob = val_preds.mean(axis=1)
        # Apply regime gate
        rs_v        = compute_regime_score(val).values
        gated_prob  = ensemble_prob * (GATE_MIN + (1 - GATE_MIN) * rs_v)

        # Threshold from val
        from sklearn.metrics import f1_score
        best_thr, best_f1 = 0.5, 0.0
        for thr in np.arange(0.40, 0.70, 0.02):
            preds_bin = (gated_prob >= thr).astype(int)
            if len(np.unique(preds_bin)) < 2:
                continue
            f1 = f1_score(y_val.astype(int), preds_bin,
                          average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        thresholds.append(best_thr)

        meta_X_oof.append(val_preds)
        meta_y_oof.append(y_val)

        # ── [4] Fold metrics ──────────────────────────────
        log.info(f"    [4/4] Meta-model (conf={best_thr:.2f}, "
                 f"{len(tr):,} rows)...")

        da_full  = accuracy_score(y_val, (ensemble_prob >= 0.5).astype(int))
        da_gated = accuracy_score(y_val, (gated_prob >= best_thr).astype(int))
        try:
            auc = roc_auc_score(y_val, gated_prob)
        except Exception:
            auc = np.nan

        log.info(f"  ── Fold {fold_n} [{regime_lbl}] RS={rs_val:.2f} ──")
        log.info(f"    DA no-gate / gated : {da_full:.4f} / {da_gated:.4f} "
                 f"({'↑' if da_gated >= da_full else '↓'}"
                 f"{abs(da_gated-da_full)*100:.2f}pp)")
        log.info(f"    AUC                : {auc:.4f}")

        fold_results.append({
            "fold": fold_n, "regime": regime_lbl,
            "rs": rs_val, "da_full": da_full,
            "da_gated": da_gated, "auc": auc,
            "threshold": best_thr,
        })

    # ── Meta-learner (trained on OOF) ─────────────────────────
    if meta_X_oof:
        meta_X = np.vstack(meta_X_oof)
        meta_y = np.concatenate(meta_y_oof)
        meta_clf = LogisticRegression(C=1.0, random_state=42)
        meta_clf.fit(meta_X, meta_y)
        meta_thr = np.mean(thresholds)
        log.info(f"\n  Meta-learner trained on {len(meta_y):,} OOF rows")
        log.info(f"  Mean threshold : {meta_thr:.3f}")
    else:
        meta_clf = None
        meta_thr = 0.5

    # ── Cross-val summary ─────────────────────────────────────
    if fold_results:
        da_mean = np.mean([f["da_gated"] for f in fold_results])
        auc_mean = np.mean([f["auc"] for f in fold_results if not math.isnan(f["auc"])])
        log.info(f"\n  CV Summary: mean DA={da_mean:.4f}  mean AUC={auc_mean:.4f}")

    # Use last fold's scaler + models for prediction
    return (transformer_models, lgb_clf_models, lgb_reg_models,
            meta_clf, scaler, feat_ext, meta_thr,
            fold_results, df_pred, rs_pred)


# ============================================================
# Advisory generation
# ============================================================
def generate_gated_advisory(df_pred, transformer_models,
                             lgb_clf_models, lgb_reg_models,
                             meta_clf, scaler,
                             feat_ext, meta_thr,
                             tiers, min_ret, universe_label):
    df_pred_tmp, _ = add_interaction_features(df_pred, feat_ext)
    feat_valid = [f for f in feat_ext if f in df_pred_tmp.columns]

    X_pred   = df_pred_tmp[feat_valid].fillna(0).values
    X_pred_s = scaler.transform(X_pred)

    # Ensemble average across folds
    t_preds   = np.array([predict_transformer(m, X_pred_s)
                           for m in transformer_models
                           if m is not None])
    lgb_preds = np.array([m.predict_proba(X_pred_s)[:, 1]
                           for m in lgb_clf_models
                           if m is not None])
    reg_preds = np.array([m.predict(X_pred_s)
                           for m in lgb_reg_models
                           if m is not None])

    clf_stack = []
    if len(t_preds)   > 0: clf_stack.append(t_preds.mean(axis=0))
    if len(lgb_preds) > 0: clf_stack.append(lgb_preds.mean(axis=0))
    ensemble_prob = np.mean(clf_stack, axis=0) if clf_stack else np.full(len(X_pred), 0.5)

    # Meta-learner
    if meta_clf is not None and len(clf_stack) >= 2:
        meta_input = np.column_stack(clf_stack[:2])
        meta_prob  = meta_clf.predict_proba(meta_input)[:, 1]
    else:
        meta_prob = ensemble_prob.copy()

    # Regime gate — softer formula: blend rather than multiply
    # Old: gated = prob * gate  → always < prob, compresses everything
    # New: gated = prob * (0.5 + 0.5 * RS)  → RS=0.77 gives 0.885x, much less aggressive
    rs_pred    = compute_regime_score(df_pred_tmp).values
    gate_scale = 0.5 + 0.5 * rs_pred          # range [0.5, 1.0]
    gated_prob = meta_prob * gate_scale

    # Predicted return — derived from gated_prob (classifier-based expected return)
    # Raw LGB regressor underfits; use prob-to-return conversion instead:
    # expected_ret = (gated_prob - 0.5) * scale
    # gated_prob=0.55 → +1.0%, gated_prob=0.44 → -1.2%, gated_prob=0.50 → 0%
    # Scale of 0.20 means top-scored ticker (prob~0.55) gets ~+2% expected return
    pred_ret = (gated_prob - 0.5) * 0.20

    pred = df_pred[["ticker", "date"]].copy().reset_index(drop=True)
    pred["prob_up"]      = ensemble_prob
    pred["meta_prob"]    = meta_prob
    pred["gated_prob"]   = gated_prob
    pred["regime_score"] = rs_pred
    pred["pred_ret"]     = pred_ret

    # ── Percentile-based tier thresholds ─────────────────────
    # tiers dict holds percentile cutoffs (e.g. 0.85 = top 15%)
    t1_pct = tiers.get("T1", 0.85)
    t2_pct = tiers.get("T2", 0.60)
    t1_thr = float(np.percentile(gated_prob, t1_pct * 100))
    t2_thr = float(np.percentile(gated_prob, t2_pct * 100))

    log.info(f"    Score range    : [{gated_prob.min():.4f}, {gated_prob.max():.4f}]")
    log.info(f"    T1 threshold   : {t1_thr:.4f} (top {100-t1_pct*100:.0f}%)")
    log.info(f"    T2 threshold   : {t2_thr:.4f} (top {100-t2_pct*100:.0f}%)")

    pred["tier"] = np.select(
        [gated_prob >= t1_thr, gated_prob >= t2_thr],
        ["T1", "T2"], default="")

    # Pass/fail filters
    pred["pass_direction"]   = gated_prob >= t2_thr
    pred["pass_confidence"]  = meta_prob  >= np.percentile(meta_prob, t2_pct * 100)
    meta_thr_adaptive = float(np.percentile(meta_prob, 40))  # bottom 60% cut
    pred["pass_meta"]        = meta_prob  >= meta_thr_adaptive
    pred["pass_regime_gate"] = rs_pred    >= GATE_MIN
    pred["pass_min_return"]  = gated_prob >= t2_thr
    pred["pass_conf_floor"]  = gated_prob >= t2_thr
    pred["pass_tier"]        = pred["tier"].isin(["T1", "T2"])

    # Optional signal filters (graceful if absent)
    pred["pass_insider"]   = (df_pred.reset_index(drop=True)["insider_strength"].fillna(0) > 0
                               if "insider_strength" in df_pred.columns
                               else pd.Series(True, index=pred.index))
    pred["pass_sentiment"] = (df_pred.reset_index(drop=True)["sentiment_score"].fillna(0) > -0.1
                               if "sentiment_score" in df_pred.columns
                               else pd.Series(True, index=pred.index))
    pred["pass_regime"]    = rs_pred >= 0.40

    pred["pass_all"] = (
        pred["pass_direction"]   &
        pred["pass_confidence"]  &
        pred["pass_meta"]        &
        pred["pass_regime_gate"] &
        pred["pass_tier"]        &
        pred["pass_regime"]      &
        (pred["tier"] == "T1"))   # pass_all = T1 picks only; T2 = watchlist

    # Attach key feature columns for output
    carry_cols = ["ticker", "date", "close",
                  "vix_level", "spy_20d_ret", "risk_on_score",
                  "insider_strength", "sentiment_score", "sector"]
    for col in carry_cols:
        if col in df_pred.columns and col not in pred.columns:
            pred[col] = df_pred.reset_index(drop=True)[col].values

    # Regime label
    mean_rs = rs_pred.mean()
    regime_label = ("BULL"    if mean_rs > 0.55
                    else "BEAR" if mean_rs < 0.40
                    else "FLAT")

    log.info(f"\n  {universe_label} advisory:")
    log.info(f"    Regime         : {regime_label} (RS={mean_rs:.3f})")
    log.info(f"    Total tickers  : {len(pred)}")
    log.info(f"    T1 picks (BUY) : {(pred['tier']=='T1').sum()}")
    log.info(f"    T2 watchlist   : {(pred['tier']=='T2').sum()}")
    log.info(f"    Pass all       : {pred['pass_all'].sum()}")

    return pred.sort_values("gated_prob", ascending=False), regime_label


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train regime-gated model and generate advisory",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir",       default=DEFAULT_DATA_DIR)
    parser.add_argument("--no-transformer", action="store_true",
                        help="Skip Transformer (faster, CPU-only)")
    parser.add_argument("--sp500-only",     action="store_true")
    parser.add_argument("--sc-only",        action="store_true")
    args = parser.parse_args()

    DATA_DIR     = args.data_dir
    USE_TRANSFORMER = not args.no_transformer

    SP500_CSV  = os.path.join(DATA_DIR, "features_final_sp500_regime.csv")
    SC_CSV     = os.path.join(DATA_DIR, "features_final_smallcap_regime.csv")
    MACRO_CSV  = os.path.join(DATA_DIR, "macro_features.csv")
    OUT_SP500  = os.path.join(DATA_DIR, "advisory_sp500.csv")
    OUT_SC     = os.path.join(DATA_DIR, "advisory_smallcap.csv")
    OUT_COMB   = os.path.join(DATA_DIR, "advisory_combined.csv")
    META_PATH  = os.path.join(DATA_DIR, "regime_gated_model_meta.json")

    log.info("=" * 65)
    log.info("  REGIME-GATED MODEL — Multimodal Stock Advisory System")
    log.info(f"  Run      : {datetime.now():%Y-%m-%d %H:%M:%S}")
    log.info(f"  Data     : {DATA_DIR}")
    log.info(f"  Transformer: {'ON' if USE_TRANSFORMER else 'OFF'}")
    log.info("=" * 65)

    # ── Load data ─────────────────────────────────────────────
    def load_features(path, label):
        if not os.path.exists(path):
            log.warning(f"[SKIP] {label} file not found: {path}")
            return None
        log.info(f"Loading {label}...")
        try:
            df = pd.read_csv(path, low_memory=False)
        except MemoryError:
            log.warning(f"  MemoryError on full load — trying chunked read...")
            chunks = []
            for chunk in pd.read_csv(path, low_memory=False, chunksize=100_000):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        log.info(f"  {label}: {len(df):,} rows  "
                 f"| latest={df['date'].max().date()}"
                 f"| features={len(df.columns)}")
        return df

    df_sp500 = load_features(SP500_CSV, "S&P 500") if not args.sc_only  else None
    df_sc    = load_features(SC_CSV,    "Small cap") if not args.sp500_only else None

    if df_sp500 is None and df_sc is None:
        sys.exit("[ERROR] No feature files found. Check DATA_DIR.")

    # ── Optionally merge macro features ──────────────────────
    if os.path.exists(MACRO_CSV):
        log.info(f"Merging macro features from {MACRO_CSV}...")
        macro = pd.read_csv(MACRO_CSV)
        macro["date"] = pd.to_datetime(macro["date"])
        macro_cols = [c for c in macro.columns if c != "date"]

        def merge_macro(df, label):
            stale = [c for c in macro_cols if c in df.columns]
            if stale:
                df = df.drop(columns=stale)
            df = df.merge(macro, on="date", how="left")
            df[macro_cols] = df[macro_cols].ffill().bfill().fillna(0)
            log.info(f"  {label}: merged {len(macro_cols)} macro cols")
            return df

        if df_sp500 is not None:
            df_sp500 = merge_macro(df_sp500, "S&P 500")
        if df_sc is not None:
            df_sc    = merge_macro(df_sc,    "Small cap")
    else:
        log.warning(f"macro_features.csv not found — skipping macro merge")

    # ── Current market snapshot ───────────────────────────────
    ref_df = df_sp500 if df_sp500 is not None else df_sc
    latest_row = ref_df[ref_df["date"] == ref_df["date"].max()].iloc[0]
    log.info(f"\n[INFO] Current market ({ref_df['date'].max().date()}):")
    for col, lbl in [("vix_level",     "VIX"),
                     ("spy_20d_ret",   "SPY 20d ret"),
                     ("risk_on_score", "risk_on_score"),
                     ("yield_curve_slope", "yield_curve")]:
        if col in latest_row.index:
            val = latest_row[col]
            fmt = f"{val*100:+.2f}%" if "ret" in col else f"{val:.3f}"
            log.info(f"  {lbl:<16}: {fmt}")
    rs_now = compute_regime_score(
        ref_df[ref_df["date"] == ref_df["date"].max()]).mean()
    log.info(f"  Deploy regime  : {'BULL' if rs_now > 0.55 else 'BEAR' if rs_now < 0.40 else 'FLAT'}"
             f" ({rs_now:.3f})")
    log.info(f"  Bull weight    : {BULL_WEIGHT}x")

    # ── Train ─────────────────────────────────────────────────
    results = {}

    if df_sp500 is not None:
        feat_sp = get_feature_cols(df_sp500)
        (t_models_sp, lgb_clf_sp, lgb_reg_sp,
         meta_sp, scaler_sp, feat_ext_sp,
         meta_thr_sp, cv_sp, df_sp_pred,
         rs_sp) = train_regime_gated(
            "S&P 500", df_sp500, feat_sp, USE_TRANSFORMER)
        results["sp500"] = dict(cv=cv_sp)

    if df_sc is not None:
        feat_sc = get_feature_cols(df_sc)
        (t_models_sc, lgb_clf_sc, lgb_reg_sc,
         meta_sc, scaler_sc, feat_ext_sc,
         meta_thr_sc, cv_sc, df_sc_pred,
         rs_sc) = train_regime_gated(
            "Small Cap", df_sc, feat_sc, USE_TRANSFORMER)
        results["smallcap"] = dict(cv=cv_sc)

    # ── Advisory ──────────────────────────────────────────────
    advisory_frames = []

    if df_sp500 is not None:
        log.info("\n[INFO] Generating S&P 500 advisory...")
        df_sp_adv, sp_regime = generate_gated_advisory(
            df_sp_pred, t_models_sp, lgb_clf_sp, lgb_reg_sp,
            meta_sp, scaler_sp, feat_ext_sp, meta_thr_sp,
            SP500_TIERS, SP500_MIN_RET, "S&P 500")
        df_sp_adv["universe"] = "sp500"
        df_sp_adv.to_csv(OUT_SP500, index=False)
        log.info(f"[SAVED] {OUT_SP500}")
        advisory_frames.append(df_sp_adv)

    if df_sc is not None:
        log.info("\n[INFO] Generating Small Cap advisory...")
        df_sc_adv, sc_regime = generate_gated_advisory(
            df_sc_pred, t_models_sc, lgb_clf_sc, lgb_reg_sc,
            meta_sc, scaler_sc, feat_ext_sc, meta_thr_sc,
            SC_TIERS, SC_MIN_RET, "Small Cap")
        df_sc_adv["universe"] = "smallcap"
        df_sc_adv.to_csv(OUT_SC, index=False)
        log.info(f"[SAVED] {OUT_SC}")
        advisory_frames.append(df_sc_adv)

    if advisory_frames:
        combined = pd.concat(advisory_frames, ignore_index=True)
        combined.to_csv(OUT_COMB, index=False)
        log.info(f"[SAVED] {OUT_COMB}")

    # ── Save meta ─────────────────────────────────────────────
    meta_out = {
        "run_timestamp": datetime.now().isoformat(),
        "data_dir":      DATA_DIR,
        "bull_weight":   BULL_WEIGHT,
        "gate_min":      GATE_MIN,
        "n_folds":       N_FOLDS,
        "use_transformer": USE_TRANSFORMER,
        "results":       {k: {"cv_da_mean": np.mean(
                              [f["da_gated"] for f in v["cv"]])}
                          for k, v in results.items()},
    }
    with open(META_PATH, "w") as f:
        json.dump(meta_out, f, indent=2)
    log.info(f"[SAVED] {META_PATH}")

    # ── Final summary ─────────────────────────────────────────
    log.info(f"\n{'='*65}")
    log.info(f"  REGIME-GATED MODEL — COMPLETE")
    log.info(f"{'='*65}")
    if advisory_frames:
        all_picks = combined[combined["pass_all"] == True]
        log.info(f"  Total advisory picks : {len(all_picks)}")
        for univ in combined["universe"].unique():
            u = combined[(combined["universe"] == univ) &
                         (combined["pass_all"] == True)]
            log.info(f"  {univ:<12}: {len(u)} picks  "
                     f"(T1={len(u[u['tier']=='T1'])}  "
                     f"T2={len(u[u['tier']=='T2'])})")
    log.info(f"\n  Outputs:")
    for p in [OUT_SP500, OUT_SC, OUT_COMB, META_PATH]:
        if os.path.exists(p):
            sz = os.path.getsize(p) / 1e6
            log.info(f"    {os.path.basename(p):<35} {sz:.1f} MB")
    log.info(f"\n  Next step: run advisory formatter / dashboard")


if __name__ == "__main__":
    main()