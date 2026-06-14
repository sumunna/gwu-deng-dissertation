#!/usr/bin/env python3
# ============================================================
# fetch_form4_sec.py  —  Standalone SEC Form 4 fetcher
#
# Runs locally on your machine, reads/writes directly to your
# mounted Google Drive folder.
#
# Usage:
#   pip install requests pandas
#   python fetch_form4_sec.py
#
# Set your Google Drive path in the CONFIG section below.
# ============================================================

import os
import re
import glob
import time
import logging
import requests
import pandas as pd
from xml.etree import ElementTree as ET
from datetime  import timedelta


# ============================================================
# CONFIG — set your Google Drive path here
# ============================================================

# Windows (Google Drive desktop app — check which letter your Drive mounts to)
GDRIVE = r"G:\My Drive"

DATA_DIR    = os.path.join(GDRIVE, "AI_PROJECT", "Data")
TECH_CSV    = os.path.join(DATA_DIR, "features_technical.csv")
RAW_CSV     = os.path.join(DATA_DIR, "insider_raw_parsed.csv")
SEC_XML_DIR = os.path.join(DATA_DIR, "SEC", "form4_xml")
LOG_FILE    = os.path.join(DATA_DIR, "fetch_form4.log")

HEADERS    = {'User-Agent': 'ResearchProject Stella.Umunna@gwu.edu'}
SEC_WWW    = "https://www.sec.gov"
SEC_DATA   = "https://data.sec.gov"
RATE_SLEEP = 0.12    # stay under SEC's 10 req/sec limit
SAVE_EVERY = 25      # save cache every N tickers
# ============================================================


def setup_logging():
    os.makedirs(DATA_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),                              # terminal
            logging.FileHandler(LOG_FILE, encoding="utf-8"),     # Drive log file
        ]
    )

log = logging.getLogger(__name__)


# ============================================================
# SEC helpers
# ============================================================

def sec_get(url: str, retries: int = 3, timeout: int = 15):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 503):
                wait = 2 ** attempt * 5
                log.warning(f"Rate limit {r.status_code} — waiting {wait}s...")
                time.sleep(wait)
                continue
            return r
        except requests.RequestException as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                log.warning(f"Request failed: {url} — {e}")
    return None


def get_xml_url(cik_int: int, acc_clean: str) -> str | None:
    """
    Fetch filing directory listing to find the real XML filename.
    SEC XMLs are named form4.xml — NOT {accession}.xml.
    """
    dir_r = sec_get(
        f"{SEC_WWW}/Archives/edgar/data/{cik_int}/{acc_clean}/",
        timeout=10
    )
    time.sleep(RATE_SLEEP)
    if dir_r is None or dir_r.status_code != 200:
        return None

    xml_files = re.findall(
        rf'href="(/Archives/edgar/data/{cik_int}/{acc_clean}/([^"]+\.xml))"',
        dir_r.text, re.IGNORECASE
    )
    if not xml_files:
        return None

    for path, fname in xml_files:
        if "form4" in fname.lower():
            return f"{SEC_WWW}{path}"
    return f"{SEC_WWW}{xml_files[0][0]}"


# ============================================================
# XML parsers
# ============================================================

def parse_form4_xml(path: str, ticker: str, accession: str) -> list[dict]:
    """Parse one Form 4 XML — P/S transactions with valid price only."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except Exception:
        return []

    xml_start = raw.find("<?xml")
    if xml_start == -1:
        xml_start = raw.find("<ownershipDocument")
    if xml_start == -1:
        return parse_regex(raw, ticker, accession)
    try:
        root = ET.fromstring(raw[xml_start:])
    except ET.ParseError:
        return parse_regex(raw, ticker, accession)

    period_n = root.find(".//periodOfReport")
    period   = (period_n.text or "").strip() if period_n is not None else ""
    name_n   = root.find(".//reportingOwnerId/rptOwnerName")
    title_n  = root.find(".//reportingOwnerRelationship/officerTitle")
    insider  = (name_n.text  or "").strip() if name_n  is not None else ""
    title    = (title_n.text or "").strip() if title_n is not None else ""
    tl       = title.lower()
    is_ceo   = int("chief executive" in tl or "ceo" in tl)
    is_cfo   = int("chief financial"  in tl or "cfo" in tl)
    dir_n    = root.find(".//reportingOwnerRelationship/isDirector")
    is_dir   = int((dir_n.text or "0").strip() == "1") if dir_n is not None else 0

    records = []
    for txn in root.findall(".//nonDerivativeTransaction"):
        code_n = txn.find(".//transactionCoding/transactionCode")
        if code_n is None or code_n.text not in ("P", "S"):
            continue
        date_n   = txn.find(".//transactionDate/value")
        shares_n = txn.find(".//transactionAmounts/transactionShares/value")
        price_n  = txn.find(".//transactionAmounts/transactionPricePerShare/value")
        plan_n   = txn.find(".//transactionCoding/Rule10b5One")

        txn_date = (date_n.text or period).strip() if date_n is not None else period
        try:
            shares = float(shares_n.text) if shares_n is not None else None
            price  = float(price_n.text)  if price_n  is not None else None
        except (ValueError, TypeError):
            shares, price = None, None

        if not price or price <= 0:
            continue

        records.append({
            "ticker":           ticker,
            "accession":        accession,
            "filed_date":       pd.NaT,
            "transaction_date": txn_date,
            "transaction_type": "buy" if code_n.text == "P" else "sell",
            "shares":           shares,
            "price_per_share":  price,
            "value":            shares * price if (shares and price) else None,
            "insider_name":     insider,
            "insider_title":    title,
            "is_ceo":           is_ceo,
            "is_cfo":           is_cfo,
            "is_director":      is_dir,
            "is_10b5_1_plan":   int(plan_n is not None and
                                    (plan_n.text or "0").strip() == "1"),
        })
    return records


def parse_regex(text: str, ticker: str, accession: str) -> list[dict]:
    """Regex fallback for malformed XML."""
    def find(tag):
        m = re.search(rf"<{tag}[^>]*>\s*<value>\s*([^<]+)\s*</value>", text)
        if not m:
            m = re.search(rf"<{tag}[^>]*>\s*([^<]+)\s*</{tag}>", text)
        return m.group(1).strip() if m else ""

    codes      = re.findall(r"<transactionCode>\s*([A-Z])\s*</transactionCode>", text)
    dates      = re.findall(r"<transactionDate>\s*<value>\s*(\d{4}-\d{2}-\d{2})", text)
    shares_all = re.findall(r"<transactionShares>\s*<value>\s*([\d.]+)", text)
    prices_all = re.findall(r"<transactionPricePerShare>\s*<value>\s*([\d.]+)", text)
    insider    = find("rptOwnerName")
    title      = find("officerTitle")
    tl         = title.lower()

    records = []
    for idx, code in enumerate(codes):
        if code not in ("P", "S"):
            continue
        try:
            shares = float(shares_all[idx]) if idx < len(shares_all) else None
            price  = float(prices_all[idx]) if idx < len(prices_all) else None
        except (ValueError, IndexError):
            shares, price = None, None
        if not price or price <= 0:
            continue
        records.append({
            "ticker":           ticker,
            "accession":        accession,
            "filed_date":       pd.NaT,
            "transaction_date": dates[idx] if idx < len(dates) else "",
            "transaction_type": "buy" if code == "P" else "sell",
            "shares":           shares,
            "price_per_share":  price,
            "value":            shares * price if (shares and price) else None,
            "insider_name":     insider,
            "insider_title":    title,
            "is_ceo":           int("chief executive" in tl or "ceo" in tl),
            "is_cfo":           int("chief financial"  in tl or "cfo" in tl),
            "is_director":      0,
            "is_10b5_1_plan":   0,
        })
    return records


# ============================================================
# Main
# ============================================================

def main():
    setup_logging()
    os.makedirs(SEC_XML_DIR, exist_ok=True)

    # ── Verify Drive is accessible ────────────────────────────
    if not os.path.exists(DATA_DIR):
        log.error(f"DATA_DIR not found: {DATA_DIR}")
        log.error("Check your GDRIVE path in the CONFIG section.")
        return

    log.info(f"Writing to: {DATA_DIR}")

    # ── Load ticker list + date window ────────────────────────
    log.info("Loading technical features for ticker list...")
    tech     = pd.read_csv(TECH_CSV, parse_dates=["date"])
    tickers  = sorted(tech["ticker"].unique().tolist())
    min_date = (tech["date"].min() - timedelta(days=90)).strftime("%Y-%m-%d")
    max_date = tech["date"].max().strftime("%Y-%m-%d")
    log.info(f"{len(tickers)} tickers | window: {min_date} → {max_date}")

    # ── Load existing cache ───────────────────────────────────
    if os.path.exists(RAW_CSV):
        cached       = pd.read_csv(RAW_CSV, parse_dates=["transaction_date"])
        done_tickers = set(cached["ticker"].unique())
        log.info(f"Cache: {len(cached):,} rows | {len(done_tickers)} tickers done.")
    else:
        cached       = pd.DataFrame()
        done_tickers = set()
        log.info("No cache found — run parse_form4_to_cache.py first.")

    # Tickers with XML folder on disk already count as done
    for t in tickers:
        folder = os.path.join(SEC_XML_DIR, t)
        if os.path.isdir(folder) and glob.glob(os.path.join(folder, "*.xml")):
            done_tickers.add(t)

    missing = sorted(set(tickers) - done_tickers)
    log.info(f"Tickers still to fetch: {len(missing)}")

    if not missing:
        log.info("All tickers already fetched — nothing to do.")
        return

    # ── Fetch CIK map ─────────────────────────────────────────
    log.info("Fetching CIK map from SEC...")
    cik_r = sec_get(f"{SEC_WWW}/files/company_tickers.json")
    if cik_r is None or cik_r.status_code != 200:
        log.error("Cannot reach SEC — check internet connection.")
        return

    cik_map = {
        e["ticker"].upper(): str(e["cik_str"]).zfill(10)
        for e in cik_r.json().values() if e.get("ticker")
    }
    log.info(f"CIK map: {len(cik_map):,} companies.")
    time.sleep(RATE_SLEEP)

    # ── Fetch loop ────────────────────────────────────────────
    new_rows = []
    t_start  = time.time()
    n        = len(missing)
    log.info(f"Starting fetch for {n} tickers | saving every {SAVE_EVERY}...\n")

    for i, ticker in enumerate(missing, 1):
        cik = cik_map.get(ticker.upper())
        if not cik:
            done_tickers.add(ticker)
            continue

        cik_int = int(cik)
        folder  = os.path.join(SEC_XML_DIR, ticker)
        os.makedirs(folder, exist_ok=True)

        sub_r = sec_get(f"{SEC_DATA}/submissions/CIK{cik}.json")
        time.sleep(RATE_SLEEP)
        if sub_r is None or sub_r.status_code != 200:
            continue

        recent  = sub_r.json().get("filings", {}).get("recent", {})
        forms   = recent.get("form",            [])
        dates   = recent.get("filingDate",      [])
        accnums = recent.get("accessionNumber", [])

        ticker_rows = []
        for j, form in enumerate(forms):
            if form not in ("4", "4/A"):
                continue
            if dates[j] < min_date or dates[j] > max_date:
                continue

            acc       = accnums[j]
            acc_clean = acc.replace("-", "")
            save_path = os.path.join(folder, f"{acc}.xml")

            # Already on disk — just parse
            if os.path.exists(save_path):
                ticker_rows.extend(parse_form4_xml(save_path, ticker, acc))
                continue

            # Find real XML URL then download
            xml_url = get_xml_url(cik_int, acc_clean)
            if not xml_url:
                continue

            xml_r = sec_get(xml_url, timeout=10)
            time.sleep(RATE_SLEEP)
            if xml_r is None or xml_r.status_code != 200:
                continue

            with open(save_path, "w", encoding="utf-8") as f:
                f.write(xml_r.text)

            ticker_rows.extend(parse_form4_xml(save_path, ticker, acc))

        new_rows.extend(ticker_rows)
        done_tickers.add(ticker)

        # Progress every 10 tickers
        if i % 10 == 0 or i == n:
            elapsed   = time.time() - t_start
            remaining = (n - i) * (elapsed / i) / 60
            log.info(
                f"{i:>4}/{n} | {ticker:<8} | "
                f"{len(new_rows):>6} new transactions | "
                f"~{remaining:.0f} min remaining"
            )

        # Save to Drive every SAVE_EVERY tickers
        if (i % SAVE_EVERY == 0 or i == n) and new_rows:
            new_df = pd.DataFrame(new_rows)
            new_df["transaction_date"] = pd.to_datetime(
                new_df["transaction_date"], errors="coerce"
            )
            cached = (
                pd.concat([cached, new_df], ignore_index=True)
                  .sort_values(["ticker", "transaction_date"])
                  .reset_index(drop=True)
            )
            cached.to_csv(RAW_CSV, index=False)
            new_rows = []
            log.info(f"  [SAVED → Drive] {len(cached):,} total rows.")

    # ── Final summary ─────────────────────────────────────────
    log.info("\nFetch complete.")
    log.info(f"  Cache rows      : {len(cached):,}")
    log.info(f"  Tickers covered : {cached['ticker'].nunique()}")
    log.info(f"  Buys            : {(cached['transaction_type']=='buy').sum():,}")
    log.info(f"  Sells           : {(cached['transaction_type']=='sell').sum():,}")
    if len(cached):
        log.info(
            f"  Date range      : {cached['transaction_date'].min().date()} "
            f"→ {cached['transaction_date'].max().date()}"
        )
    log.info(f"\nLog saved to: {LOG_FILE}")
    log.info("Run features_insider.py next to compute rolling features.")


if __name__ == "__main__":
    main()
