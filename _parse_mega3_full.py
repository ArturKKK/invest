#!/usr/bin/env python3
"""Parse mega_comparison3.log → clean CSV + markdown with proper leverage labels."""
import re
import csv

with open("mega_comparison3.log") as f:
    lines = f.readlines()

results = []
current_sim = None
sim_num = 0

for line in lines:
    m = re.search(r'SIM #(\d+): (.+)', line)
    if m:
        sim_num = int(m.group(1))
        current_sim = m.group(2).strip()
        continue

    m = re.search(
        r'Return=([^\s]+)\s+HAC=([^\s]+)\s+MaxDD=([^\s]+)\s+WR=([^\s]+)\s+PF=([^\s]+)',
        line,
    )
    if m and current_sim:
        parts = current_sim.split('__')
        window = parts[0] if parts else current_sim
        model = parts[1] if len(parts) > 1 else ''
        sim_cfg = parts[2] if len(parts) > 2 else ''

        # Extract leverage from sim_cfg: "3x_base" → 3x, "5x_base" → 5x, "base" → 1x
        lev_m = re.match(r'^(\d+)x', sim_cfg)
        lev = lev_m.group(1) + 'x' if lev_m else '1x'

        # Extract flags: strip leverage prefix → "3x_base" → "base", "1x_smooth" → "smooth"
        flags = re.sub(r'^\d+x_', '', sim_cfg) if lev_m else sim_cfg

        def safe_float(s):
            s = s.replace('%', '').replace('+', '')
            try:
                return float(s)
            except ValueError:
                return None

        results.append({
            'sim': sim_num,
            'window': window,
            'model': model,
            'lev': lev,
            'flags': flags,
            'sim_cfg': sim_cfg,
            'ret': safe_float(m.group(1)),
            'hac': safe_float(m.group(2)),
            'maxdd': safe_float(m.group(3)),
            'wr': safe_float(m.group(4)),
            'pf': safe_float(m.group(5)),
        })
        current_sim = None

# ── Write CSV ──
csv_path = "results/mega_comparison3_all_sims.csv"
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=[
        'sim', 'window', 'model', 'lev', 'flags', 'sim_cfg',
        'ret', 'hac', 'maxdd', 'wr', 'pf',
    ])
    w.writeheader()
    w.writerows(results)
print(f"CSV: {csv_path} ({len(results)} sims)")

# ── Helpers ──
win_labels = {
    'WinA_train2024H1': ('WinA', '2024-07 → 2024-12'),
    'WinB_train2024':   ('WinB', '2025-01 → 2025-06'),
    'WinC_train2025H1': ('WinC', '2025-07 → 2025-12'),
}
wins = list(win_labels.keys())

def fmt(val, dp=1, plus=False):
    if val is None:
        return 'N/A'
    return f"{val:+.{dp}f}" if plus else f"{val:.{dp}f}"

# ── Build Markdown ──
md = []
md.append("# Mega Comparison 3 — Full Results (fixed)")
md.append("")
md.append(f"Total sims: {len(results)}")
md.append("")

# ═══════════════════
# SECTION 1: FULL TABLE PER WINDOW (all sims, proper leverage)
# ═══════════════════
for win_full, (ws, period) in win_labels.items():
    win_r = [r for r in results if r['window'] == win_full]
    win_r.sort(key=lambda x: -(x['hac'] or -999))

    md.append(f"## {ws} — OOS: {period} ({len(win_r)} sims)")
    md.append("")
    md.append("| # | Model | Lev | Flags | Ret% | HAC | MaxDD% | WR% | PF |")
    md.append("|--:|-------|:---:|-------|-----:|----:|-------:|----:|---:|")

    for i, r in enumerate(win_r, 1):
        md.append(
            f"| {i} | {r['model']} | {r['lev']} | {r['flags']} | "
            f"{fmt(r['ret'], plus=True)} | {fmt(r['hac'], 2, True)} | "
            f"{fmt(r['maxdd'])} | {fmt(r['wr'], 0)} | {fmt(r['pf'], 2)} |"
        )
    md.append("")

# ═══════════════════
# SECTION 2: CLEAN CROSS-WINDOW (base config only, by leverage)
# ═══════════════════
for lev in ['1x', '3x']:
    md.append(f"## Cross-Window: {lev} leverage, base config only")
    md.append("")
    md.append("| Model | WinA HAC | WinA Ret | WinB HAC | WinB Ret | WinC HAC | WinC Ret | Avg HAC |")
    md.append("|-------|--------:|--------:|--------:|--------:|--------:|--------:|--------:|")

    lookup = {}
    all_models = set()
    for r in results:
        if r['lev'] == lev and r['flags'] == 'base' and r['hac'] is not None:
            key = (r['window'], r['model'])
            if key not in lookup or r['hac'] > lookup[key]['hac']:
                lookup[key] = r
            all_models.add(r['model'])

    model_rows = []
    for model in all_models:
        hacs, rets = [], []
        for w in wins:
            r = lookup.get((w, model))
            if r and r['hac'] is not None:
                hacs.append(r['hac'])
                rets.append(r['ret'])
            else:
                hacs.append(None)
                rets.append(None)
        valid_hacs = [h for h in hacs if h is not None]
        avg = sum(valid_hacs) / len(valid_hacs) if valid_hacs else -999
        model_rows.append((model, hacs, rets, avg))

    model_rows.sort(key=lambda x: x[3], reverse=True)
    for model, hacs, rets, avg in model_rows:
        cells = []
        for h, ret in zip(hacs, rets):
            cells.append(fmt(h, 2, True))
            cells.append(fmt(ret, 1, True))
        md.append(f"| {model} | {' | '.join(cells)} | {fmt(avg, 2, True)} |")
    md.append("")

# ═══════════════════
# SECTION 3: SOLO vs ENSEMBLE detail (1x base)
# ═══════════════════
md.append("## Solo vs Ensemble (1x base)")
md.append("")
md.append("| Category | Model | WinA | WinB | WinC | Avg |")
md.append("|----------|-------|-----:|-----:|-----:|----:|")

lookup_1x = {}
all_models_1x = set()
for r in results:
    if r['lev'] == '1x' and r['flags'] == 'base' and r['hac'] is not None:
        key = (r['window'], r['model'])
        if key not in lookup_1x or r['hac'] > lookup_1x[key]['hac']:
            lookup_1x[key] = r
        all_models_1x.add(r['model'])

categories = [
    ('CatBoost solo', lambda m: 'cb_solo' in m),
    ('LGB v6 solo', lambda m: 'v6_solo' in m),
    ('LGB v7 solo', lambda m: 'v7_solo' in m),
    ('XGB solo', lambda m: 'xgb_solo' in m),
    ('Ensemble 2', lambda m: 'ens2' in m),
    ('Ensemble 3', lambda m: 'ens3' in m),
    ('Ensemble 4', lambda m: 'ens4' in m),
]

for cat, match_fn in categories:
    models_in = sorted(m for m in all_models_1x if match_fn(m))
    for model in models_in:
        hacs = []
        row_parts = []
        for w in wins:
            r = lookup_1x.get((w, model))
            if r:
                row_parts.append(fmt(r['hac'], 2, True))
                hacs.append(r['hac'])
            else:
                row_parts.append('N/A')
        avg = sum(hacs) / len(hacs) if hacs else None
        md.append(f"| {cat} | {model} | {' | '.join(row_parts)} | {fmt(avg, 2, True)} |")

md.append("")

# ═══════════════════
# SECTION 4: LEVERAGE COMPARISON
# ═══════════════════
md.append("## Leverage Comparison (base config)")
md.append("")
md.append("| Window | Model | 1x HAC | 1x Ret | 1x DD | 3x HAC | 3x Ret | 3x DD | 5x HAC | 5x Ret | 5x DD |")
md.append("|--------|-------|------:|------:|-----:|------:|------:|-----:|------:|------:|-----:|")

lev_data = {}
for r in results:
    if r['flags'] == 'base' and r['hac'] is not None:
        key = (r['window'], r['model'])
        if key not in lev_data:
            lev_data[key] = {}
        existing = lev_data[key].get(r['lev'])
        if existing is None or r['hac'] > existing['hac']:
            lev_data[key][r['lev']] = r

for (w, model), levs in sorted(lev_data.items()):
    if len(levs) < 2:
        continue
    ws = w.split('_')[0]
    cells = [ws, model]
    for lv in ['1x', '3x', '5x']:
        r = levs.get(lv)
        if r:
            cells.extend([fmt(r['hac'], 2, True), fmt(r['ret'], 1, True), fmt(r['maxdd'])])
        else:
            cells.extend(['—', '—', '—'])
    md.append(f"| {' | '.join(cells)} |")

md.append("")

# ═══════════════════
# SECTION 5: EXECUTION FLAGS IMPACT (1x only)
# ═══════════════════
md.append("## Execution Flags Impact (1x)")
md.append("")
md.append("| Window | Model | Flag | HAC | vs Base Δ | Ret% | MaxDD% |")
md.append("|--------|-------|------|----:|----------:|-----:|-------:|")

for w in wins:
    ws = w.split('_')[0]
    for r in sorted(
        [r for r in results if r['window'] == w and r['lev'] == '1x' and r['flags'] != 'base' and r['hac'] is not None],
        key=lambda x: -(x['hac'] or -999),
    ):
        base = lookup_1x.get((w, r['model']))
        base_hac = base['hac'] if base else None
        delta = r['hac'] - base_hac if base_hac is not None else None
        md.append(
            f"| {ws} | {r['model']} | {r['flags']} | "
            f"{fmt(r['hac'], 2, True)} | {fmt(delta, 2, True)} | "
            f"{fmt(r['ret'], 1, True)} | {fmt(r['maxdd'])} |"
        )

md.append("")

md_path = "results/MEGA_COMPARISON3_RESULTS.md"
with open(md_path, 'w') as f:
    f.write('\n'.join(md))
print(f"MD:  {md_path}")
