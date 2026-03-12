#!/usr/bin/env python3
"""
Parse walk-forward validation results.
Shows per-window performance + stability metrics across windows.
"""
import re, os, sys, glob
import numpy as np

log_dir = sys.argv[1] if len(sys.argv) > 1 else 'walkforward_results'

WINDOWS = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6']
WINDOW_LABELS = {
    'W1': '2024-09→12', 'W2': '2024-12→03',
    'W3': '2025-03→06', 'W4': '2025-06→09',
    'W5': '2025-09→12', 'W6': '2025-12→03',
}

# Auto-detect configs from log files, or fall back to known list
_default_configs = [
    'v7_solo', 'ens3', 'ens4',
    'ens4+deriv', 'ens4+meta_lgb', 'ens4+meta_ridge',
]

def detect_configs(log_dir):
    """Auto-detect config names from log filenames like '{config}_{W1..W6}.log'."""
    configs = set()
    for fname in os.listdir(log_dir):
        if not fname.endswith('.log'):
            continue
        for w in WINDOWS:
            suffix = f'_{w}.log'
            if fname.endswith(suffix):
                cfg = fname[:-len(suffix)]
                configs.add(cfg)
    if configs:
        return sorted(configs)
    return _default_configs

CONFIGS = detect_configs(log_dir)

def parse_log(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()
    def get(pattern, default=None):
        m = re.search(pattern, text)
        return m.group(1) if m else default
    ret = get(r'Return:\s+([+-]?[\d.]+)%')
    if ret is None:
        return None
    sharpe = get(r'Sharpe:\s+([+-]?[\d.]+)')
    dd = get(r'Max DD:\s+(-?[\d.]+)%')
    wr = get(r'Win Rate:\s+(\d+)%')
    pf = get(r'PF:\s+([\d.]+)')
    calmar = get(r'Calmar:\s+([+-]?[\d.]+)')
    return {
        'ret': float(ret) if ret else None,
        'sharpe': float(sharpe) if sharpe else None,
        'dd': float(dd) if dd else None,
        'wr': float(wr) if wr else None,
        'pf': float(pf) if pf else None,
        'calmar': float(calmar) if calmar else None,
    }

# ── Collect all results ──────────────────────────────────────
data = {}  # data[config][window] = parsed_result
for cfg in CONFIGS:
    data[cfg] = {}
    for w in WINDOWS:
        path = os.path.join(log_dir, f'{cfg}_{w}.log')
        data[cfg][w] = parse_log(path)

# ── 1. Per-window Sharpe table ───────────────────────────────
print()
print("=" * 100)
print("  WALK-FORWARD RESULTS — Sharpe per window (6 × 90d)")
print("=" * 100)

# Header
hdr = f"  {'Config':<20s}"
for w in WINDOWS:
    hdr += f" {WINDOW_LABELS[w]:>11s}"
hdr += f" {'Mean':>7s} {'StdDev':>7s} {'Min':>7s} {'Wins':>5s}"
print(hdr)
print("-" * len(hdr))

# Rows
config_stats = {}
for cfg in CONFIGS:
    row = f"  {cfg:<20s}"
    sharpes = []
    for w in WINDOWS:
        r = data[cfg][w]
        if r and r['sharpe'] is not None:
            sharpes.append(r['sharpe'])
            row += f" {r['sharpe']:>11.2f}"
        else:
            row += f" {'ERR':>11s}"
    
    if sharpes:
        mean_s = np.mean(sharpes)
        std_s = np.std(sharpes)
        min_s = np.min(sharpes)
        n_pos = sum(1 for s in sharpes if s > 0)
        row += f" {mean_s:>7.2f} {std_s:>7.2f} {min_s:>7.2f} {n_pos:>3d}/{len(sharpes)}"
        config_stats[cfg] = {
            'mean_sharpe': mean_s, 'std_sharpe': std_s,
            'min_sharpe': min_s, 'n_positive': n_pos,
            'n_windows': len(sharpes), 'sharpes': sharpes,
        }
    else:
        row += f" {'N/A':>7s} {'N/A':>7s} {'N/A':>7s} {'N/A':>5s}"
    print(row)

# ── 2. Per-window Return table ───────────────────────────────
print()
print("=" * 100)
print("  WALK-FORWARD RESULTS — Return% per window")
print("=" * 100)

hdr = f"  {'Config':<20s}"
for w in WINDOWS:
    hdr += f" {WINDOW_LABELS[w]:>11s}"
hdr += f" {'Total':>7s} {'Mean':>7s}"
print(hdr)
print("-" * len(hdr))

for cfg in CONFIGS:
    row = f"  {cfg:<20s}"
    rets = []
    for w in WINDOWS:
        r = data[cfg][w]
        if r and r['ret'] is not None:
            rets.append(r['ret'])
            row += f" {r['ret']:>+10.1f}%"
        else:
            row += f" {'ERR':>11s}"
    if rets:
        # Compound return
        total = 100 * (np.prod([1 + r/100 for r in rets]) - 1)
        row += f" {total:>+6.1f}% {np.mean(rets):>+6.1f}%"
    print(row)

# ── 3. Per-window MaxDD table ────────────────────────────────
print()
print("=" * 100)
print("  WALK-FORWARD RESULTS — MaxDD% per window")
print("=" * 100)

hdr = f"  {'Config':<20s}"
for w in WINDOWS:
    hdr += f" {WINDOW_LABELS[w]:>11s}"
hdr += f" {'WorstDD':>8s}"
print(hdr)
print("-" * len(hdr))

for cfg in CONFIGS:
    row = f"  {cfg:<20s}"
    dds = []
    for w in WINDOWS:
        r = data[cfg][w]
        if r and r['dd'] is not None:
            dds.append(r['dd'])
            row += f" {r['dd']:>10.1f}%"
        else:
            row += f" {'ERR':>11s}"
    if dds:
        row += f" {min(dds):>7.1f}%"
    print(row)

# ── 4. Stability ranking ─────────────────────────────────────
print()
print("=" * 100)
print("  STABILITY RANKING (sorted by Mean Sharpe, with consistency metrics)")
print("=" * 100)

ranked = sorted(config_stats.items(), key=lambda x: x[1]['mean_sharpe'], reverse=True)
print(f"  {'#':>2s}  {'Config':<20s} {'MeanSharpe':>11s} {'StdSharpe':>10s} {'MinSharpe':>10s} {'Sharpe/Std':>11s} {'Positive':>9s}")
print("-" * 85)
for i, (cfg, st) in enumerate(ranked):
    info_ratio = st['mean_sharpe'] / st['std_sharpe'] if st['std_sharpe'] > 0 else float('inf')
    mark = '>>' if i == 0 else '  '
    print(f"{mark}{i+1:2d}  {cfg:<20s} {st['mean_sharpe']:>+10.2f} {st['std_sharpe']:>10.2f} {st['min_sharpe']:>+10.2f} {info_ratio:>11.2f} {st['n_positive']:>4d}/{st['n_windows']}")

print()
best_cfg = ranked[0][0]
best_st = ranked[0][1]
print(f"  BEST: {best_cfg}")
print(f"    Mean Sharpe: {best_st['mean_sharpe']:+.2f} (std {best_st['std_sharpe']:.2f})")
print(f"    Worst window: {best_st['min_sharpe']:+.2f}")
print(f"    Positive: {best_st['n_positive']}/{best_st['n_windows']} windows")

# ── 5. Impact analysis across windows ────────────────────────
print()
print("=" * 100)
print("  IMPACT ANALYSIS (per-window Sharpe deltas vs baseline)")
print("=" * 100)

# Auto-build comparisons: find a baseline config and compare all others
# Priority: ens4 > baseline_10 > first config
_baseline_priority = ['ens4', 'baseline_10', 'baseline_7']
base_cfg = None
for bp in _baseline_priority:
    if bp in config_stats:
        base_cfg = bp
        break
if base_cfg is None and config_stats:
    base_cfg = list(config_stats.keys())[0]

if base_cfg:
    comparisons = [(f"{base_cfg} → {cfg}", base_cfg, cfg)
                    for cfg in CONFIGS if cfg != base_cfg and cfg in config_stats]
else:
    comparisons = []

for label, base, test in comparisons:
    print(f"\n  {label}")
    deltas = []
    for w in WINDOWS:
        r_base = data.get(base, {}).get(w)
        r_test = data.get(test, {}).get(w)
        if r_base and r_test and r_base['sharpe'] is not None and r_test['sharpe'] is not None:
            d = r_test['sharpe'] - r_base['sharpe']
            deltas.append(d)
            winner = "✓" if d > 0 else "✗"
            print(f"    {WINDOW_LABELS[w]}: {r_base['sharpe']:+.2f} → {r_test['sharpe']:+.2f}  ({d:+.2f}) {winner}")
    if deltas:
        mean_d = np.mean(deltas)
        n_pos = sum(1 for d in deltas if d > 0)
        verdict = "HELPS" if mean_d > 0.3 else ("HURTS" if mean_d < -0.3 else "NEUTRAL")
        print(f"    AVG: {mean_d:+.2f} | Wins: {n_pos}/{len(deltas)} | → {verdict}")

print()
