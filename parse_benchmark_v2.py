#!/usr/bin/env python3
"""Parse benchmark v2 logs (30d + 60d) and display sorted results."""
import re, os, sys, glob

log_dir = sys.argv[1] if len(sys.argv) > 1 else 'benchmark_results_v2'

configs = [
    'v6_solo', 'v7_solo', 'v6+deriv', 'v7+deriv',
    'ens3', 'ens3+deriv',
    'ens3+meta_lgb', 'ens3+deriv+meta_lgb',
    'ens4', 'ens4+deriv',
    'ens4+meta_lgb', 'ens4+deriv+meta_lgb',
]

def parse_log(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()
    def get(pattern, default='N/A'):
        m = re.search(pattern, text)
        return m.group(1) if m else default
    ret = get(r'Return:\s+([+-]?[\d.]+)%')
    if ret == 'N/A':
        return None
    return {
        'ret': ret,
        'ann': get(r'ann\. ~([+-]?[\d.]+)%'),
        'dd': get(r'Max DD:\s+(-?[\d.]+)%'),
        'sharpe': get(r'Sharpe:\s+([+-]?[\d.]+)'),
        'sharpe_hac': get(r'Sharpe HAC:\s+([+-]?[\d.]+)'),
        'calmar': get(r'Calmar:\s+([+-]?[\d.]+)'),
        'wr': get(r'Win Rate:\s+(\d+)%'),
        'pf': get(r'PF:\s+([\d.]+)'),
        'trades': get(r'Trades:\s+(\d+)'),
    }

for days in [30, 60]:
    print()
    print(f"{'='*80}")
    print(f"  {days}d RESULTS  (sorted by Sharpe)")
    print(f"{'='*80}")
    
    results = []
    for cfg in configs:
        path = os.path.join(log_dir, f'{cfg}_{days}d.log')
        r = parse_log(path)
        results.append((cfg, r))
    
    results.sort(key=lambda x: float(x[1]['sharpe']) if x[1] else -999, reverse=True)
    
    hdr = f" {'#':>2}  {'Config':<25s} {'Ret%':>6s} {'Sharpe':>7s} {'DD%':>6s} {'Calmar':>7s} {'WR%':>4s} {'PF':>5s}"
    print(hdr)
    print('-' * len(hdr))
    
    for i, (cfg, r) in enumerate(results):
        if r:
            mark = '>>' if i == 0 else '  '
            print(f"{mark}{i+1:2d}  {cfg:<25s} {r['ret']:>6s} {r['sharpe']:>7s} {r['dd']:>6s} {r['calmar']:>7s} {r['wr']:>4s} {r['pf']:>5s}")
        else:
            print(f"  {i+1:2d}  {cfg:<25s}   ERR     ERR    ERR     ERR  ERR   ERR")
    
    print()
    best_cfg, best_r = results[0]
    if best_r:
        print(f"  BEST: {best_cfg}  ->  Sharpe {best_r['sharpe']} | Ret {best_r['ret']}% | DD {best_r['dd']}% | PF {best_r['pf']}")

# Cross-analysis
print()
print(f"{'='*80}")
print(f"  IMPACT ANALYSIS")
print(f"{'='*80}")

for days in [30, 60]:
    print(f"\n--- {days}d ---")
    data = {}
    for cfg in configs:
        path = os.path.join(log_dir, f'{cfg}_{days}d.log')
        r = parse_log(path)
        if r:
            data[cfg] = float(r['sharpe'])
    
    # Deriv gate effect
    for base, deriv in [('v6_solo', 'v6+deriv'), ('v7_solo', 'v7+deriv'),
                         ('ens3', 'ens3+deriv'), ('ens4', 'ens4+deriv')]:
        if base in data and deriv in data:
            d = data[deriv] - data[base]
            print(f"  Deriv:  {base:25s} -> {deriv:25s}  {data[base]:+.2f} -> {data[deriv]:+.2f}  ({d:+.2f})")
    
    # XGBoost effect (ens3 vs ens4)
    for e3, e4 in [('ens3', 'ens4'), ('ens3+deriv', 'ens4+deriv'),
                    ('ens3+meta_lgb', 'ens4+meta_lgb'),
                    ('ens3+deriv+meta_lgb', 'ens4+deriv+meta_lgb')]:
        if e3 in data and e4 in data:
            d = data[e4] - data[e3]
            print(f"  +XGB:   {e3:25s} -> {e4:25s}  {data[e3]:+.2f} -> {data[e4]:+.2f}  ({d:+.2f})")
    
    # Meta-model effect
    for base, meta in [('ens3', 'ens3+meta_lgb'), ('ens3+deriv', 'ens3+deriv+meta_lgb'),
                        ('ens4', 'ens4+meta_lgb'), ('ens4+deriv', 'ens4+deriv+meta_lgb')]:
        if base in data and meta in data:
            d = data[meta] - data[base]
            print(f"  +Meta:  {base:25s} -> {meta:25s}  {data[base]:+.2f} -> {data[meta]:+.2f}  ({d:+.2f})")
