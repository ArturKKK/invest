"""Run R68 continuous WF with replacement features instead of 6 dead slots.

Usage:
  EXP=A_seasonal_x_symbol python _run_replacement.py
  EXP=BASELINE python _run_replacement.py   # sanity check (no changes)

Env:
  EXP — experiment key from _replacement_features.EXPERIMENTS, or BASELINE.

Writes:
  /data/datasets/replacement_${EXP}.csv   (from results_r68_continuous_wf.csv)
  /data/datasets/replacement_${EXP}.log   (via tee from runner)
"""
from __future__ import annotations
import os
import shutil
from pathlib import Path

import _research_r68_continuous_wf as r68
import _replacement_features as rf

DEAD = rf.DEAD_FEATS
EXP = os.environ.get("EXP", "BASELINE").strip()

_orig_load = r68.load_data


def patched_load():
    df, regime_df = _orig_load()
    if EXP in ("BASELINE", "", "NONE"):
        print(f"[REPLACEMENT] EXP={EXP} — no changes")
        return df, regime_df
    if EXP not in rf.EXPERIMENTS:
        raise SystemExit(f"Unknown EXP={EXP}; valid: {list(rf.EXPERIMENTS)}")
    added = rf.EXPERIMENTS[EXP](df)
    # Mutate module-level CHAMPION_FEAT_31: drop DEAD, append added
    champ = [f for f in r68.CHAMPION_FEAT_31 if f not in DEAD]
    for a in added:
        if a in df.columns and a not in champ:
            champ.append(a)
    r68.CHAMPION_FEAT_31[:] = champ
    # Ensure new feats aren't accidentally market-level excluded
    present_market = [f for f in champ if f in r68.MARKET_LEVEL_FEATURES]
    print(f"[REPLACEMENT] EXP={EXP}")
    print(f"[REPLACEMENT] dropped DEAD: {[f for f in DEAD if f in r68.CHAMPION_FEAT_31]} (should be empty)")
    print(f"[REPLACEMENT] added: {added}")
    print(f"[REPLACEMENT] feats total: {len(champ)}  market-level: {present_market}")
    return df, regime_df


r68.load_data = patched_load


if __name__ == "__main__":
    r68.main()
    src = Path("/data/datasets/results_r68_continuous_wf.csv")
    if src.exists():
        dst = Path(f"/data/datasets/replacement_{EXP}.csv")
        shutil.copy2(src, dst)
        print(f"[REPLACEMENT] CSV -> {dst}")
