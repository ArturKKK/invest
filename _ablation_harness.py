#!/usr/bin/env python3
"""
Ablation harness for 3 audit fixes (see commit b325afc).

Controls via env vars (default=ON for all):
  FIX1=0|1  — inf→nan cleanup in _ic_scanner.build_features_minimal
  FIX2=0|1  — 6 market-level features in MARKET_LEVEL_FEATURES
  FIX3=0|1  — cum_funding_24h Binance override (LIVE-only; no-op for backtest)

Strategy:
  .ablation_bak/ holds the "NO FIXES" baseline copy of each file.
  At runtime we restore baseline, then inject each requested fix on top.
  First bootstrap: if backups absent, strip fixes from current committed files.
"""
import os, shutil, pathlib, re, sys

ROOT = pathlib.Path(__file__).parent.resolve()
BAK = ROOT / ".ablation_bak"
BAK.mkdir(exist_ok=True)

FIX1_BLOCK = (
    "\n    # ── Clean inf from pct_change on zero denominators ─────────\n"
    "    for col in df.select_dtypes(include=[np.number]).columns:\n"
    "        df[col] = df[col].replace([np.inf, -np.inf], np.nan)\n"
)
FIX1_ANCHOR = 'df["premium_zscore"] = (df["premium_index"] - pi_mean) / pi_std\n'
FIX1_REGEX = re.compile(
    r"\n    # ── Clean inf from pct_change on zero denominators[^\n]*\n"
    r"    for col in df\.select_dtypes\(include=\[np\.number\]\)\.columns:\n"
    r"        df\[col\] = df\[col\]\.replace\(\[np\.inf, -np\.inf\], np\.nan\)\n"
)

FIX2_FEATS = [
    "pct_coins_up_12h", "pct_coins_up_1h",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
FIX2_BLOCK = (
    '    # ablation: fix#2 (6 market-level features, skip CS-rank)\n'
    + "".join(f'    "{f}",\n' for f in FIX2_FEATS)
)
FIX2_ANCHOR = '    "mkt_oi_extreme_pct",\n}'
FIX2_REGEX = re.compile(
    r'    # ablation: fix#2[^\n]*\n(?:    "(?:pct_coins_up_12h|pct_coins_up_1h|hour_sin|hour_cos|dow_sin|dow_cos)",\n)+'
)
FIX2_ORIG_REGEX = re.compile(
    r'    # ── Same value for all symbols at each timestamp ──\n'
    r'    # Without this[^\n]*\n'
    r'(?:    "(?:pct_coins_up_12h|pct_coins_up_1h|hour_sin|hour_cos|dow_sin|dow_cos)",\n)+'
)

def _strip_fix1(text: str) -> str:
    return FIX1_REGEX.sub("\n", text, count=1)

def _strip_fix2(text: str) -> str:
    text = FIX2_REGEX.sub("", text, count=1)
    text = FIX2_ORIG_REGEX.sub("", text, count=1)
    return text

def _bootstrap_backup():
    targets = {
        "_ic_scanner.py": _strip_fix1,
        "_research_r35_new_features.py": _strip_fix2,
    }
    for rel, stripper in targets.items():
        dst = BAK / rel
        if dst.exists():
            continue
        src = ROOT / rel
        txt = src.read_text()
        stripped = stripper(txt)
        dst.write_text(stripped)
        print(f"[bootstrap] saved NO-FIX baseline: {dst}")

def _restore_baseline(rel: str):
    shutil.copy2(BAK / rel, ROOT / rel)

def apply_fix1(enable: bool):
    rel = "_ic_scanner.py"
    _restore_baseline(rel)
    if enable:
        p = ROOT / rel
        src = p.read_text()
        if FIX1_ANCHOR not in src:
            print(f"ERR fix1 enable: anchor not found in {rel}")
            sys.exit(2)
        p.write_text(src.replace(FIX1_ANCHOR, FIX1_ANCHOR + FIX1_BLOCK, 1))

def apply_fix2(enable: bool):
    rel = "_research_r35_new_features.py"
    _restore_baseline(rel)
    if enable:
        p = ROOT / rel
        src = p.read_text()
        if FIX2_ANCHOR not in src:
            print(f"ERR fix2 enable: anchor not found in {rel}")
            sys.exit(2)
        injected = '    "mkt_oi_extreme_pct",\n' + FIX2_BLOCK + '}'
        p.write_text(src.replace(FIX2_ANCHOR, injected, 1))

def apply_fix3(enable: bool):
    pass  # LIVE-only; no-op for backtest.

def main():
    _bootstrap_backup()
    f1 = os.environ.get("FIX1", "1") == "1"
    f2 = os.environ.get("FIX2", "1") == "1"
    f3 = os.environ.get("FIX3", "1") == "1"
    apply_fix1(f1)
    apply_fix2(f2)
    apply_fix3(f3)
    ic = (ROOT / "_ic_scanner.py").read_text()
    r35 = (ROOT / "_research_r35_new_features.py").read_text()
    have_fix1 = "replace([np.inf, -np.inf], np.nan)" in ic
    have_fix2 = all(f'"{name}"' in r35 for name in FIX2_FEATS)
    print(f"Ablation applied: FIX1={int(f1)} FIX2={int(f2)} FIX3={int(f3)} "
          f"(verify fix1={int(have_fix1)} fix2={int(have_fix2)})")
    ok = (have_fix1 == f1) and (have_fix2 == f2)
    if not ok:
        print("ERR verification failed")
        sys.exit(3)

if __name__ == "__main__":
    main()
