# ============================================================
# advisory_dashboard.py
#
# Standalone Advisory Formatter + HTML Dashboard
# Multimodal Stock Advisory System — Stella Umunna | DEng AI | GWU
#
# Reads advisory_combined.csv and produces:
#   1. advisory_report.txt   — plain-text ranked pick list
#   2. advisory_dashboard.html — interactive HTML dashboard
#
# Install: pip install pandas numpy jinja2
# Run:     python advisory_dashboard.py
#          python advisory_dashboard.py --data-dir "G:\My Drive\AI_PROJECT\Data"
# ============================================================

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = r"G:\My Drive\AI_PROJECT\Data"


# ============================================================
# Load + format
# ============================================================
def load_advisory(data_dir):
    path = os.path.join(data_dir, "advisory_combined.csv")
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Not found: {path}\nRun train_regime_gated_model.py first.")
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    return df


def fmt_pct(v, decimals=1):
    try:
        return f"{float(v)*100:+.{decimals}f}%"
    except Exception:
        return "N/A"


def fmt_prob(v, decimals=1):
    try:
        return f"{float(v)*100:.{decimals}f}%"
    except Exception:
        return "N/A"


def fmt_price(v):
    try:
        return f"${float(v):.2f}"
    except Exception:
        return "N/A"


# ============================================================
# Plain-text report
# ============================================================
def build_text_report(df, data_dir):
    t1 = df[df["pass_all"] == True].copy().sort_values("gated_prob", ascending=False)
    t2 = df[(df["tier"] == "T2") & (df["pass_all"] != True)].copy().sort_values("gated_prob", ascending=False)

    lines = []
    sep   = "=" * 72

    # Header
    lines += [
        sep,
        "  MULTIMODAL STOCK ADVISORY SYSTEM",
        "  Regime-Gated Model — Weekly Advisory Report",
        f"  Generated : {datetime.now():%Y-%m-%d %H:%M}",
        f"  As-of date: {df['date'].max().date()}",
        sep,
    ]

    # Market snapshot
    row = df.iloc[0]
    rs  = float(row.get("regime_score", 0))
    regime_lbl = "BULL 🟢" if rs > 0.55 else "BEAR 🔴" if rs < 0.40 else "FLAT 🟡"
    lines += [
        "",
        "  MARKET SNAPSHOT",
        "-" * 72,
        f"  Regime         : {regime_lbl} (RS={rs:.3f})",
        f"  VIX            : {row.get('vix_level', 'N/A')}",
        f"  SPY 20d return : {fmt_pct(row.get('spy_20d_ret', None))}",
        f"  Risk-on score  : {fmt_prob(row.get('risk_on_score', None))}",
        "",
    ]

    # T1 picks
    lines += [
        sep,
        f"  TIER 1 — BUY PICKS  ({len(t1)} stocks)",
        "  High-conviction. Top 15% of model score in BULL regime.",
        sep,
        f"  {'#':<4} {'TICKER':<8} {'UNIVERSE':<12} {'SECTOR':<22} {'PRICE':<10} {'SCORE':<8} {'CONF':<8} {'PRED 5d':<8}",
        "-" * 72,
    ]
    for i, (_, r) in enumerate(t1.iterrows(), 1):
        sector  = str(r.get("sector", ""))[:20] if pd.notna(r.get("sector")) else "—"
        univ    = "S&P 500" if str(r.get("universe","")) == "sp500" else "Small Cap"
        lines.append(
            f"  {i:<4} {r['ticker']:<8} {univ:<12} {sector:<22} "
            f"{fmt_price(r.get('close')):<10} "
            f"{fmt_prob(r.get('gated_prob')):<8} "
            f"{fmt_prob(r.get('meta_prob')):<8} "
            f"{fmt_pct(r.get('pred_ret')):<8}"
        )

    # T2 watchlist
    lines += [
        "",
        sep,
        f"  TIER 2 — WATCHLIST  ({len(t2)} stocks)",
        "  Elevated score. Monitor for entry. Top 40% of model score.",
        sep,
        f"  {'#':<4} {'TICKER':<8} {'UNIVERSE':<12} {'SECTOR':<22} {'PRICE':<10} {'SCORE':<8}",
        "-" * 72,
    ]
    for i, (_, r) in enumerate(t2.iterrows(), 1):
        sector = str(r.get("sector", ""))[:20] if pd.notna(r.get("sector")) else "—"
        univ   = "S&P 500" if str(r.get("universe","")) == "sp500" else "Small Cap"
        lines.append(
            f"  {i:<4} {r['ticker']:<8} {univ:<12} {sector:<22} "
            f"{fmt_price(r.get('close')):<10} "
            f"{fmt_prob(r.get('gated_prob')):<8}"
        )

    # Sector breakdown (T1)
    if "sector" in t1.columns and t1["sector"].notna().any():
        sec_counts = t1["sector"].value_counts()
        lines += [
            "",
            sep,
            "  SECTOR BREAKDOWN — T1 PICKS",
            "-" * 72,
        ]
        for sec, cnt in sec_counts.items():
            bar = "█" * cnt
            lines.append(f"  {str(sec):<30} {cnt:>3}  {bar}")

    lines += ["", sep, "  END OF REPORT", sep]
    out_path = os.path.join(data_dir, "advisory_report.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[SAVED] {out_path}")
    return "\n".join(lines)


# ============================================================
# HTML Dashboard
# ============================================================
def build_html_dashboard(df, data_dir):
    t1 = df[df["pass_all"] == True].copy().sort_values("gated_prob", ascending=False)
    t2 = df[(df["tier"] == "T2") & (df["pass_all"] != True)].copy().sort_values("gated_prob", ascending=False)

    row    = df.iloc[0]
    rs     = float(row.get("regime_score", 0.5))
    regime = "BULL" if rs > 0.55 else "BEAR" if rs < 0.40 else "FLAT"
    regime_color = {"BULL": "#00d4aa", "BEAR": "#ff4d6d", "FLAT": "#ffd166"}[regime]

    as_of  = df["date"].max().strftime("%B %d, %Y")
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    vix_val  = row.get("vix_level", "—")
    spy_val  = fmt_pct(row.get("spy_20d_ret"))
    risk_val = fmt_prob(row.get("risk_on_score"))

    # Sector breakdown for T1
    sec_data = ""
    if "sector" in t1.columns and t1["sector"].notna().any():
        sec_counts = t1["sector"].value_counts().head(10)
        max_cnt    = sec_counts.max()
        for sec, cnt in sec_counts.items():
            pct = cnt / max_cnt * 100
            sec_data += f"""
            <div class="sector-row">
              <span class="sector-name">{sec}</span>
              <div class="sector-bar-wrap">
                <div class="sector-bar" style="width:{pct:.0f}%"></div>
              </div>
              <span class="sector-cnt">{cnt}</span>
            </div>"""

    def make_rows(subset, show_tier=True):
        rows = ""
        for i, (_, r) in enumerate(subset.iterrows(), 1):
            sector  = str(r.get("sector", ""))[:28] if pd.notna(r.get("sector")) else "—"
            univ    = "S&P 500" if str(r.get("universe","")) == "sp500" else "Small Cap"
            score   = float(r.get("gated_prob", 0))
            conf    = float(r.get("meta_prob", 0))
            score_w = min(score * 200, 100)
            tier_badge = f'<span class="badge t1">T1</span>' if r.get("tier") == "T1" else f'<span class="badge t2">T2</span>'
            pred_ret_val = float(r.get('pred_ret', 0) or 0)
            pred_price   = float(r.get('close', 0) or 0) * (1 + pred_ret_val)
            pred_price_str = f"${pred_price:.2f}" if pred_price > 0 else "N/A"
            rows += f"""
            <tr>
              <td class="rank">{i}</td>
              <td class="ticker">{r['ticker']}</td>
              {"<td>" + tier_badge + "</td>" if show_tier else ""}
              <td>{univ}</td>
              <td class="sector">{sector}</td>
              <td>{fmt_price(r.get('close'))}</td>
              <td class="pred-ret {'pos' if pred_price_str != 'N/A' and pred_price > float(r.get('close',0) or 0) else 'neg'}">{pred_price_str}</td>
              <td>
                <div class="score-wrap">
                  <div class="score-bar" style="width:{score_w:.0f}%"></div>
                  <span>{fmt_prob(score)}</span>
                </div>
              </td>
              <td>{fmt_prob(conf)}</td>
              <td class="pred-ret {'pos' if float(r.get('pred_ret',0) or 0) >= 0 else 'neg'}">{fmt_pct(r.get('pred_ret'))}</td>
            </tr>"""
        return rows

    t1_rows = make_rows(t1, show_tier=False)
    t2_rows = make_rows(t2, show_tier=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Advisory Dashboard — {as_of}</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:       #0a0c10;
    --bg2:      #10141c;
    --bg3:      #161b26;
    --border:   #1e2535;
    --text:     #c8d0e0;
    --muted:    #4a5568;
    --accent:   {regime_color};
    --t1:       #00d4aa;
    --t2:       #ffd166;
    --bull:     #00d4aa;
    --bear:     #ff4d6d;
    --flat:     #ffd166;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    line-height: 1.6;
    min-height: 100vh;
  }}

  /* ── Header ── */
  .header {{
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
    padding: 28px 40px 24px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
  }}
  .header-left h1 {{
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #fff;
  }}
  .header-left p {{
    color: var(--muted);
    font-size: 11px;
    margin-top: 4px;
    letter-spacing: 0.5px;
  }}
  .header-right {{
    text-align: right;
    font-size: 11px;
    color: var(--muted);
  }}
  .header-right .as-of {{
    font-family: 'Syne', sans-serif;
    font-size: 14px;
    font-weight: 600;
    color: var(--accent);
  }}

  /* ── Layout ── */
  .main {{ padding: 32px 40px; max-width: 1400px; margin: 0 auto; }}

  /* ── Stats bar ── */
  .stats-bar {{
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 12px;
    margin-bottom: 32px;
  }}
  .stat-card {{
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 18px;
  }}
  .stat-card .label {{
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 6px;
  }}
  .stat-card .value {{
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #fff;
  }}
  .stat-card .value.accent {{ color: var(--accent); }}
  .stat-card .value.bull   {{ color: var(--bull); }}
  .stat-card .value.bear   {{ color: var(--bear); }}
  .stat-card .sub {{
    font-size: 10px;
    color: var(--muted);
    margin-top: 2px;
  }}

  /* ── Two-column layout ── */
  .two-col {{
    display: grid;
    grid-template-columns: 1fr 280px;
    gap: 20px;
    margin-bottom: 24px;
  }}

  /* ── Tables ── */
  .panel {{
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 24px;
  }}
  .panel-header {{
    padding: 16px 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .panel-header h2 {{
    font-family: 'Syne', sans-serif;
    font-size: 14px;
    font-weight: 700;
    color: #fff;
    letter-spacing: 0.3px;
  }}
  .panel-header .pill {{
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 11px;
    color: var(--muted);
  }}
  .panel-header .pill.t1 {{ border-color: var(--t1); color: var(--t1); }}
  .panel-header .pill.t2 {{ border-color: var(--t2); color: var(--t2); }}

  table {{
    width: 100%;
    border-collapse: collapse;
  }}
  thead th {{
    padding: 10px 14px;
    text-align: left;
    font-size: 10px;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    background: var(--bg3);
    font-weight: 400;
  }}
  tbody tr {{
    border-bottom: 1px solid var(--border);
    transition: background 0.15s;
  }}
  tbody tr:last-child {{ border-bottom: none; }}
  tbody tr:hover {{ background: var(--bg3); }}
  tbody td {{
    padding: 10px 14px;
    vertical-align: middle;
  }}
  .rank {{ color: var(--muted); width: 36px; }}
  .ticker {{
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 13px;
    color: #fff;
  }}
  .sector {{ color: var(--muted); font-size: 11px; }}
  .pred-ret {{ font-weight: 500; }}
  .pred-ret.pos {{ color: var(--bull); }}
  .pred-ret.neg {{ color: var(--bear); }}

  .badge {{
    display: inline-block;
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.5px;
  }}
  .badge.t1 {{ background: rgba(0,212,170,0.15); color: var(--t1); border: 1px solid var(--t1); }}
  .badge.t2 {{ background: rgba(255,209,102,0.15); color: var(--t2); border: 1px solid var(--t2); }}

  .score-wrap {{
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .score-bar {{
    height: 4px;
    border-radius: 2px;
    background: var(--accent);
    opacity: 0.7;
    min-width: 4px;
  }}

  /* ── Regime indicator ── */
  .regime-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--bg3);
    border: 1px solid var(--accent);
    border-radius: 6px;
    padding: 4px 12px;
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 1px;
  }}
  .regime-dot {{
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 2s infinite;
  }}
  @keyframes pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50%       {{ opacity: 0.5; transform: scale(0.85); }}
  }}

  /* ── Sector chart ── */
  .sector-row {{
    display: grid;
    grid-template-columns: 160px 1fr 32px;
    align-items: center;
    gap: 10px;
    padding: 8px 20px;
    border-bottom: 1px solid var(--border);
  }}
  .sector-row:last-child {{ border-bottom: none; }}
  .sector-name {{ font-size: 11px; color: var(--text); }}
  .sector-bar-wrap {{
    background: var(--bg3);
    border-radius: 2px;
    height: 6px;
    overflow: hidden;
  }}
  .sector-bar {{
    height: 100%;
    background: var(--accent);
    border-radius: 2px;
    opacity: 0.8;
  }}
  .sector-cnt {{ color: var(--muted); font-size: 11px; text-align: right; }}

  /* ── Footer ── */
  .footer {{
    border-top: 1px solid var(--border);
    padding: 20px 40px;
    color: var(--muted);
    font-size: 10px;
    letter-spacing: 0.5px;
    display: flex;
    justify-content: space-between;
  }}

  /* ── Tab system ── */
  .tabs {{
    display: flex;
    gap: 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0;
  }}
  .tab-btn {{
    padding: 12px 24px;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: var(--muted);
    background: none;
    border: none;
    border-bottom: 2px solid transparent;
    cursor: pointer;
    letter-spacing: 0.5px;
    transition: all 0.15s;
  }}
  .tab-btn.active {{
    color: var(--accent);
    border-bottom-color: var(--accent);
  }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}

  /* ── Search ── */
  .search-bar {{
    padding: 12px 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    gap: 10px;
    align-items: center;
  }}
  .search-bar input {{
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    padding: 6px 12px;
    width: 200px;
    outline: none;
  }}
  .search-bar input:focus {{ border-color: var(--accent); }}
  .search-bar label {{ color: var(--muted); font-size: 11px; }}
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <h1>📊 Multimodal Stock Advisory System</h1>
    <p>REGIME-GATED MODEL &nbsp;·&nbsp; TRANSFORMER + LIGHTGBM ENSEMBLE &nbsp;·&nbsp; STELLA UMUNNA | DENG AI | GWU</p>
  </div>
  <div class="header-right">
    <div class="as-of">As of {as_of}</div>
    <div>Generated {run_ts}</div>
  </div>
</div>

<div class="main">

  <!-- Stats bar -->
  <div class="stats-bar">
    <div class="stat-card">
      <div class="label">Market Regime</div>
      <div class="value accent">{regime}</div>
      <div class="sub">RS = {rs:.3f}</div>
    </div>
    <div class="stat-card">
      <div class="label">VIX Level</div>
      <div class="value">{vix_val:.1f}</div>
      <div class="sub">Volatility index</div>
    </div>
    <div class="stat-card">
      <div class="label">SPY 20d Return</div>
      <div class="value {'bull' if '+' in spy_val else 'bear'}">{spy_val}</div>
      <div class="sub">Market momentum</div>
    </div>
    <div class="stat-card">
      <div class="label">Risk-On Score</div>
      <div class="value">{risk_val}</div>
      <div class="sub">Composite signal</div>
    </div>
    <div class="stat-card">
      <div class="label">T1 Buy Picks</div>
      <div class="value accent">{len(t1)}</div>
      <div class="sub">High conviction</div>
    </div>
    <div class="stat-card">
      <div class="label">T2 Watchlist</div>
      <div class="value">{len(t2)}</div>
      <div class="sub">Monitor for entry</div>
    </div>
  </div>

  <!-- T1 Picks + Sector -->
  <div class="two-col">
    <div>
      <div class="panel">
        <div class="panel-header">
          <h2>TIER 1 — BUY PICKS</h2>
          <span class="pill t1">{len(t1)} stocks · top 15%</span>
          <span style="margin-left:auto">
            <span class="regime-badge">
              <span class="regime-dot"></span>
              {regime} REGIME
            </span>
          </span>
        </div>
        <div class="search-bar">
          <label>Filter:</label>
          <input type="text" id="t1-search" placeholder="ticker or sector..." oninput="filterTable('t1-body', this.value)">
        </div>
        <table>
          <thead>
            <tr>
              <th>#</th><th>Ticker</th><th>Universe</th><th>Sector</th>
              <th>Price</th><th>Pred Price</th><th>Score</th><th>Confidence</th><th>Pred 5d</th>
            </tr>
          </thead>
          <tbody id="t1-body">
            {t1_rows}
          </tbody>
        </table>
      </div>
    </div>

    <!-- Sector breakdown -->
    <div>
      <div class="panel">
        <div class="panel-header">
          <h2>SECTOR MIX</h2>
          <span class="pill t1">T1 picks</span>
        </div>
        {sec_data if sec_data else '<div style="padding:20px;color:var(--muted)">No sector data</div>'}
      </div>
    </div>
  </div>

  <!-- T2 Watchlist -->
  <div class="panel">
    <div class="panel-header">
      <h2>TIER 2 — WATCHLIST</h2>
      <span class="pill t2">{len(t2)} stocks · top 40%</span>
    </div>
    <div class="search-bar">
      <label>Filter:</label>
      <input type="text" id="t2-search" placeholder="ticker or sector..." oninput="filterTable('t2-body', this.value)">
    </div>
    <table>
      <thead>
        <tr>
          <th>#</th><th>Ticker</th><th>Universe</th><th>Sector</th>
          <th>Price</th><th>Pred Price</th><th>Score</th><th>Confidence</th><th>Pred 5d</th>
        </tr>
      </thead>
      <tbody id="t2-body">
        {t2_rows}
      </tbody>
    </table>
  </div>

</div>

<div class="footer">
  <span>MULTIMODAL STOCK ADVISORY SYSTEM &nbsp;·&nbsp; REGIME-GATED MODEL</span>
  <span>FOR RESEARCH PURPOSES ONLY — NOT FINANCIAL ADVICE</span>
</div>

<script>
function filterTable(bodyId, query) {{
  const q = query.toLowerCase();
  const rows = document.getElementById(bodyId).querySelectorAll('tr');
  rows.forEach(row => {{
    row.style.display = row.textContent.toLowerCase().includes(q) ? '' : 'none';
  }});
}}
</script>
</body>
</html>"""

    out_path = os.path.join(data_dir, "advisory_dashboard.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[SAVED] {out_path}")
    return out_path


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Generate advisory report and dashboard")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    print(f"[INFO] Loading advisory data from {DATA_DIR}...")
    df = load_advisory(DATA_DIR)

    t1 = df[df["pass_all"] == True]
    t2 = df[(df["tier"] == "T2") & (df["pass_all"] != True)]
    print(f"[INFO] T1 picks   : {len(t1)}")
    print(f"[INFO] T2 watchlist: {len(t2)}")

    print("\n[INFO] Building text report...")
    build_text_report(df, DATA_DIR)

    print("[INFO] Building HTML dashboard...")
    build_html_dashboard(df, DATA_DIR)

    print(f"\n[DONE] Open advisory_dashboard.html in your browser.")

if __name__ == "__main__":
    main()