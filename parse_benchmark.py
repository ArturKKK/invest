#!/usr/bin/env python3
"""Parse benchmark logs and display sorted results table."""
import re, os

log_dir = 'benchmark_results'
configs = [
    'v6_solo', 'v7_solo', 'v6_solo+deriv', 'v7_solo+deriv',
    'ensemble_no_deriv', 'ensemble+deriv',
    'ensemble+meta_lgb', 'ensemble+meta_ridge',
    'ensemble+deriv+meta_lgb', 'ensemble+deriv+meta_ridge'
]

def parse_log(path):
    with open(path) as f:
        text = f.read()
    def get(pattern, default='N/A'):
        m = re.search(pattern, text)
        return m.group(1) if m else default
    return {
        'ret': get(r'Return:\s+([+-]?[\d.]+)%'),
        'ann': get(r'ann\. ~([+-]?[\d.]+)%'),
        'dd': get(r'Max DD:\s+(-?[\d.]+)%'),
        'sharpe': get(r'Sharpe:\s+([+-]?[\d.]+)'),
        'sharpe_hac': get(r'Sharpe HAC:\s+([+-]?[\d.]+)'),
        'calmar': get(r'Calmar:\s+([+-]?[\d.]+)'),
        'wr': get(r'Win Rate:\s+(\d+)%'),
        'pf': get(r'PF:\s+([\d.]+)'),
        'trades': get(r'Trades:\s+(\d+)'),
    }

results = []
for cfg in configs:
    path = os.path.join(log_dir, f'{cfg}.log')
    if os.path.exists(path):
        results.append((cfg, parse_log(path)))
    else:
        results.append((cfg, None))

# Sort by Sharpe descending
results.sort(key=lambda x: float(x[1]['sharpe']) if x[1] and x[1]['sharpe'] != 'N/A' else -999, reverse=True)

# Print table
hdr = f"{'#':>2}  {'Config':<30s} {'Return':>7s} {'Sharpe':>7s} {'MaxDD':>6s} {'Calmar':>7s} {'WR%':>4s} {'PF':>5s} {'Trades':>6s}"
sep = '-' * len(hdr)
print()
print(sep)
print(hdr)
print(sep)
for i, (cfg, r) in enumerate(results):
    if r and r['sharpe'] != 'N/A':
        mark = ' *' if i == 0 else '  '
        print(f"{mark}{i+1:2d} {cfg:<30s} {r['ret']:>6s}% {r['sharpe']:>7s} {r['dd']:>6s}% {r['calmar']:>7s} {r['wr']:>4s}% {r['pf']:>5s} {r['trades']:>6s}")
    else:
        print(f"  {i+1:2d} {cfg:<30s}    ERR     ERR    ERR     ERR  ERR   ERR    ERR")
print(sep)

best_cfg, best_r = results[0]
print()
print(f"BEST by Sharpe: {best_cfg}")
print(f"  Sharpe {best_r['sharpe']} | Return {best_r['ret']}% | DD {best_r['dd']}% | Calmar {best_r['calmar']} | PF {best_r['pf']}")
print()

# Group analysis
print("=== ANALYSIS ===")
print()

# Deriv gate effect
for base, deriv in [('v6_solo', 'v6_solo+deriv'), ('v7_solo', 'v7_solo+deriv'),
                     ('ensemble_no_deriv', 'ensemble+deriv')]:
    r_base = dict(results)[base]
    r_deriv = dict(results)[deriv]
    if r_base and r_deriv:
        s_b = float(r_base['sharpe'])
        s_d = float(r_deriv['sharpe'])
        delta = s_d - s_b
        arrow = '+' if delta > 0 else ''
        print(f"Deriv gate: {base} -> {deriv}: Sharpe {s_b:+.2f} -> {s_d:+.2f} ({arrow}{delta:.2f})")

print()
# Meta-model effect
for base, meta in [('ensemble_no_deriv', 'ensemble+meta_lgb'),
                    ('ensemble_no_deriv', 'ensemble+meta_ridge'),
                    ('ensemble+deriv', 'ensemble+deriv+meta_lgb'),
                    ('ensemble+deriv', 'ensemble+deriv+meta_ridge')]:
    r_base = dict(results)[base]
    r_meta = dict(results)[meta]
    if r_base and r_meta:
        s_b = float(r_base['sharpe'])
        s_m = float(r_meta['sharpe'])
        delta = s_m - s_b
        arrow = '+' if delta > 0 else ''
        print(f"Meta: {base} -> {meta}: Sharpe {s_b:+.2f} -> {s_m:+.2f} ({arrow}{delta:.2f})")
