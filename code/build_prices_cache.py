# ============================================================
# build_prices_cache.py
#
# Standalone script — no Google Colab required.
# Downloads OHLCV price data from Yahoo Finance → prices_cache.csv
# DUAL UNIVERSE: S&P 500 (large cap) + S&P 600 top-100 (small cap)
# Tickers fetched LIVE from Wikipedia — always current.
#
# INCREMENTAL MODE: only fetches dates not already in the file.
# Small cap tickers with < MIN_SC_DAYS rows are force-re-downloaded
# from FALLBACK_START to ensure full history.
#
# Schema: ticker, date, asof_date, open, high, low,
#         close, adj_close, volume, universe
#
# Install:
#   pip install yfinance pandas requests
#
# Run:
#   python build_prices_cache.py
#   python build_prices_cache.py --data-dir "G:\My Drive\AI_PROJECT\Data"
#   python build_prices_cache.py --data-dir /path/to/data --start 2022-01-01
# ============================================================

import os
import io
import sys
import time
import argparse
import requests
import pandas as pd
import yfinance as yf

# ============================================================
# CONFIG — defaults (all overridable via CLI args)
# ============================================================
DEFAULT_DATA_DIR    = r"G:\My Drive\AI_PROJECT\Data"
DEFAULT_START       = "2023-01-01"
BATCH_SIZE          = 20
SMALL_CAP_TOP_N     = 100
MIN_SC_DAYS         = 200   # tickers below this row count get a full re-download

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


# ============================================================
# Helpers
# ============================================================
def wiki_tables(url: str) -> list:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text))


def fetch_sp500_tickers() -> list:
    print("[INFO] Fetching S&P 500 tickers from Wikipedia...")
    tables  = wiki_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    tickers = sorted(set(tickers))
    print(f"[INFO] S&P 500: {len(tickers)} tickers loaded.")
    return tickers


def fetch_sp600_tickers(top_n: int = 100) -> list:
    print("[INFO] Fetching S&P 600 SmallCap tickers from Wikipedia...")
    tables = wiki_tables("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies")
    df     = tables[0]

    ticker_col = next(
        (c for c in df.columns if any(k in str(c).lower() for k in ("ticker", "symbol"))),
        None,
    )
    if ticker_col is None:
        raise ValueError(f"Cannot find ticker column. Columns: {df.columns.tolist()}")

    raw = df[ticker_col].str.replace(".", "-", regex=False).dropna().tolist()
    raw = sorted(set(raw))
    print(f"[INFO] S&P 600 raw list: {len(raw)} tickers loaded.")

    print(f"[INFO] Fetching shares outstanding to rank top {top_n}...")
    market_caps: dict = {}
    batches = [raw[i:i + 25] for i in range(0, len(raw), 25)]

    for i, batch in enumerate(batches):
        print(f"  MC batch {i+1}/{len(batches)}...")
        for t in batch:
            try:
                info   = yf.Ticker(t).info
                shares = info.get("sharesOutstanding")
                price  = info.get("previousClose") or info.get("regularMarketPrice")
                if shares and price:
                    market_caps[t] = float(shares) * float(price)
            except Exception:
                pass
        time.sleep(1.0)

    covered = len(market_caps)
    print(f"[INFO] Market cap data retrieved for {covered}/{len(raw)} tickers.")

    ranked   = sorted(market_caps, key=market_caps.get, reverse=True)
    missing  = [t for t in raw if t not in market_caps]
    selected = (ranked + missing)[:top_n]

    if covered >= top_n:
        print(f"[INFO] Selected top {top_n} small caps by market cap.")
    else:
        print(f"[WARN] Only {covered} ranked — filled rest alphabetically.")

    return selected


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to standard lowercase names, handling yfinance version differences."""
    col_map = {}
    for col in df.columns:
        cl = str(col).lower().strip()
        if cl in ("date", "datetime", "price", "index"):
            col_map[col] = "date"
        elif cl == "open":
            col_map[col] = "open"
        elif cl == "high":
            col_map[col] = "high"
        elif cl == "low":
            col_map[col] = "low"
        elif cl == "close":
            col_map[col] = "close"
        elif cl in ("adj close", "adj_close", "adjclose"):
            col_map[col] = "adj_close"
        elif cl == "volume":
            col_map[col] = "volume"
    return df.rename(columns=col_map)


def download_batch(tickers: list, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if raw.empty:
        return pd.DataFrame()

    # Flatten MultiIndex columns if present (multi-ticker download)
    if isinstance(raw.columns, pd.MultiIndex):
        frames = []
        for ticker in tickers:
            try:
                df = raw.xs(ticker, axis=1, level=1).copy()
            except KeyError:
                print(f"  [WARN] {ticker}: no data returned")
                continue
            df = normalize_columns(df.reset_index())
            if "date" not in df.columns:
                print(f"  [WARN] {ticker}: could not find date column, skipping")
                continue
            df.insert(0, "ticker", ticker)
            frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        ticker = tickers[0]
        df = normalize_columns(raw.reset_index())
        if "date" not in df.columns:
            print(f"  [WARN] {ticker}: could not find date column, skipping")
            return pd.DataFrame()
        df.insert(0, "ticker", ticker)
        return df


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Build / update prices_cache.csv from Yahoo Finance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_prices_cache.py
  python build_prices_cache.py --data-dir "G:\\My Drive\\AI_PROJECT\\Data"
  python build_prices_cache.py --data-dir /data --start 2022-01-01
  python build_prices_cache.py --small-cap-n 50
        """)
    parser.add_argument("--data-dir",     default=DEFAULT_DATA_DIR,
                        help=f"Data directory (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--start",        default=DEFAULT_START,
                        help=f"Fallback start date for full downloads (default: {DEFAULT_START})")
    parser.add_argument("--small-cap-n",  type=int, default=SMALL_CAP_TOP_N,
                        help=f"Number of S&P 600 small caps to include (default: {SMALL_CAP_TOP_N})")
    parser.add_argument("--min-sc-days",  type=int, default=MIN_SC_DAYS,
                        help=f"Min rows before forcing re-download (default: {MIN_SC_DAYS})")
    parser.add_argument("--batch-size",   type=int, default=BATCH_SIZE,
                        help=f"Tickers per yfinance batch (default: {BATCH_SIZE})")
    args = parser.parse_args()

    DATA_DIR     = args.data_dir
    FALLBACK_START = args.start
    END_DATE     = pd.Timestamp.today().strftime("%Y-%m-%d")
    PRICES_CSV   = os.path.join(DATA_DIR, "prices_cache.csv")

    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("  BUILD PRICES CACHE — Multimodal Stock Advisory System")
    print(f"  Data dir  : {DATA_DIR}")
    print(f"  End date  : {END_DATE}")
    print("=" * 60)

    # ── Step 1: Fetch live ticker lists ──────────────────────
    SP500_TICKERS     = fetch_sp500_tickers()
    SMALL_CAP_TICKERS = fetch_sp600_tickers(top_n=args.small_cap_n)

    UNIVERSE_MAP: dict = {}
    for t in SP500_TICKERS:
        UNIVERSE_MAP[t] = "sp500"
    for t in SMALL_CAP_TICKERS:
        if t not in UNIVERSE_MAP:
            UNIVERSE_MAP[t] = "small_cap"

    ALL_TICKERS = list(UNIVERSE_MAP.keys())
    n_sp500 = sum(v == "sp500"     for v in UNIVERSE_MAP.values())
    n_sc    = sum(v == "small_cap" for v in UNIVERSE_MAP.values())
    print(f"\n[INFO] Total tickers : {len(ALL_TICKERS)}  |  "
          f"S&P 500: {n_sp500}  |  Small cap: {n_sc}")

    # ── Step 2: Load existing file ────────────────────────────
    if os.path.exists(PRICES_CSV):
        existing         = pd.read_csv(PRICES_CSV, dtype=str)
        existing["date"] = existing["date"].astype(str)

        if "universe" not in existing.columns:
            existing["universe"] = existing["ticker"].map(UNIVERSE_MAP).fillna("unknown")
        else:
            mask = existing["universe"].isna() | (existing["universe"] == "")
            existing.loc[mask, "universe"] = (
                existing.loc[mask, "ticker"].map(UNIVERSE_MAP))

        sc_counts    = (existing[existing["universe"] == "small_cap"]
                        .groupby("ticker")["date"].count())
        thin_sc      = set(sc_counts[sc_counts < args.min_sc_days].index)
        existing_tickers = set(existing["ticker"].unique())
        new_sc       = set(SMALL_CAP_TICKERS) - existing_tickers
        force_tickers = thin_sc | new_sc

        if force_tickers:
            print(f"\n[INFO] {len(force_tickers)} small cap tickers need full re-download "
                  f"({len(thin_sc)} thin, {len(new_sc)} new).")
            existing      = existing[~existing["ticker"].isin(force_tickers)].copy()
            existing_keys = set(zip(existing["ticker"], existing["date"]))
        else:
            existing_keys = set(zip(existing["ticker"], existing["date"]))

        last_date  = existing["date"].max() if len(existing) > 0 else FALLBACK_START
        START_DATE = last_date
        print(f"[INFO] Existing file : {len(existing):,} rows, last date: {last_date}")
        print(f"[INFO] Incremental fetch from {START_DATE} → {END_DATE}")

    else:
        existing      = pd.DataFrame()
        existing_keys = set()
        force_tickers = set(SMALL_CAP_TICKERS)
        START_DATE    = FALLBACK_START
        print(f"\n[INFO] No existing file — full download from {START_DATE} → {END_DATE}")

    # ── Step 3: Build download schedule ──────────────────────
    schedule = []
    for t in ALL_TICKERS:
        if t in force_tickers:
            schedule.append((t, FALLBACK_START, END_DATE))
        else:
            schedule.append((t, START_DATE, END_DATE))

    schedule_df  = pd.DataFrame(schedule, columns=["ticker", "start", "end"])
    start_groups = schedule_df.groupby("start")["ticker"].apply(list).to_dict()

    # ── Step 4: Download ──────────────────────────────────────
    all_frames = []

    for start_date, ticker_group in start_groups.items():
        label   = "FULL HISTORY" if start_date == FALLBACK_START else "incremental"
        batches = [ticker_group[i:i + args.batch_size]
                   for i in range(0, len(ticker_group), args.batch_size)]
        print(f"\n[INFO] {label} download ({start_date} → {END_DATE}) — "
              f"{len(ticker_group)} tickers, {len(batches)} batches")

        for i, batch in enumerate(batches):
            print(f"  Batch {i+1}/{len(batches)}: "
                  f"{batch[:4]}{'...' if len(batch) > 4 else ''}")
            try:
                df = download_batch(batch, start_date, END_DATE)
                if not df.empty:
                    all_frames.append(df)
            except Exception as e:
                print(f"  [ERROR] batch {i+1} failed: {e}")
            time.sleep(1.5)

    # ── Step 5: Clean, tag, deduplicate, save ────────────────
    if not all_frames:
        print("\n[INFO] No new data downloaded — file is already up to date.")
        sys.exit(0)

    new_data              = pd.concat(all_frames, ignore_index=True)
    new_data["date"]      = pd.to_datetime(new_data["date"]).dt.date.astype(str)
    new_data["asof_date"] = new_data["date"]
    new_data              = new_data.dropna(subset=["close"])

    for col in ["open", "high", "low", "close", "adj_close"]:
        new_data[col] = pd.to_numeric(new_data[col], errors="coerce").round(4)
    new_data["volume"] = pd.to_numeric(new_data["volume"], errors="coerce").round(0)

    new_data["universe"] = new_data["ticker"].map(UNIVERSE_MAP).fillna("unknown")
    new_data = new_data[[
        "ticker", "date", "asof_date",
        "open", "high", "low", "close", "adj_close", "volume", "universe",
    ]]

    before   = len(new_data)
    new_data = new_data[
        ~new_data.apply(lambda r: (r["ticker"], r["date"]) in existing_keys, axis=1)
    ]
    print(f"\n[INFO] Skipped   {before - len(new_data):,} duplicate rows.")
    print(f"[INFO] Appending {len(new_data):,} new rows.")

    combined = (
        pd.concat([existing, new_data], ignore_index=True)
          .sort_values(["universe", "ticker", "date"])
          .reset_index(drop=True)
    )
    combined.to_csv(PRICES_CSV, index=False)

    sp500_c = combined[combined["universe"] == "sp500"]
    sc_c    = combined[combined["universe"] == "small_cap"]
    sc_rpd  = len(sc_c) / max(sc_c["ticker"].nunique(), 1)

    print(f"\n[DONE] {PRICES_CSV}")
    print(f"       Total rows            : {len(combined):,}")
    print(f"       Date range            : "
          f"{combined['date'].min()}  →  {combined['date'].max()}")
    print(f"       S&P 500 tickers       : {sp500_c['ticker'].nunique()}")
    print(f"       Small cap tickers     : {sc_c['ticker'].nunique()}")
    print(f"       Small cap rows        : {len(sc_c):,}")
    print(f"       Avg rows/ticker (SC)  : {sc_rpd:.0f} trading days")

    if sc_rpd < args.min_sc_days:
        print(f"\n[WARN] Small cap still thin (~{sc_rpd:.0f} rows/ticker). "
              f"Check tickers for delistings or data gaps.")


if __name__ == "__main__":
    main()