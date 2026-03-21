#!/usr/bin/env python3
"""Parse overnight_v11.log and print metrics table."""
import re

lines = open('overnight_v11.log').readlines()
exp = ''
results = []
w_metrics = {}
for l in lines:
    m = re.search(r'EXP #(\d+): (.+)', l)
    if m:
        exp = f'#{m.group(1):>2} {m.group(2).strip()}'
        w_metrics = {}
    for key, pat in [
        ('icir', r'Rank_ICIR\s+([\d.]+)'),
        ('sharpe', r'LS_Sharpe_net\s+([-\d.]+)'),
        ('dd', r'LS_MaxDD_net_%\s+([-\d.]+)'),
        ('ddstop', r'LS_DDStop_Sharpe\s+([-\d.]+)'),
        ('ret', r'LS_Total_net_%\s+([-\d.]+)'),
    ]:
        m2 = re.search(pat, l)
        if m2:
            w_metrics.setdefault(key, []).append(float(m2.group(1)))
    if 'Results saved to' in l and w_metrics:
        results.append((exp, w_metrics))
        w_metrics = {}

hdr = f"{'Experiment':<40} {'R1_Sh':>6} {'R2_Sh':>6} {'R1_DD%':>7} {'R2_DD%':>7} {'R1_ICIR':>7} {'R2_ICIR':>7} {'R1_Ret%':>8} {'R2_Ret%':>8}"
print(hdr)
print('-' * len(hdr))
for exp, m in results:
    sh = m.get('sharpe', [])
    dd = m.get('dd', [])
    ic = m.get('icir', [])
    rt = m.get('ret', [])
    vals = []
    for arr in [sh, sh, dd, dd, ic, ic, rt, rt]:
        pass
    r1s = f'{sh[0]:+.2f}' if len(sh) > 0 else '?'
    r2s = f'{sh[1]:+.2f}' if len(sh) > 1 else '?'
    r1d = f'{dd[0]:.1f}' if len(dd) > 0 else '?'
    r2d = f'{dd[1]:.1f}' if len(dd) > 1 else '?'
    r1i = f'{ic[0]:.3f}' if len(ic) > 0 else '?'
    r2i = f'{ic[1]:.3f}' if len(ic) > 1 else '?'
    r1r = f'{rt[0]:+.1f}' if len(rt) > 0 else '?'
    r2r = f'{rt[1]:+.1f}' if len(rt) > 1 else '?'
    print(f'{exp:<40} {r1s:>6} {r2s:>6} {r1d:>7} {r2d:>7} {r1i:>7} {r2i:>7} {r1r:>8} {r2r:>8}')
