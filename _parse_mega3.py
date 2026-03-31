#!/usr/bin/env python3
"""Parse mega_comparison3.log and produce analytics."""
import re
import sys

with open("mega_comparison3.log") as f:
    lines = f.readlines()

results = []
current_sim = None

for line in lines:
    m = re.search(r'SIM #(\d+): (.+)', line)
    if m:
        current_sim = m.group(2).strip()
        continue

    m = re.search(
        r'Return=([^\s]+)\s+HAC=([^\s]+)\s+MaxDD=([^\s]+)\s+WR=([^\s]+)\s+PF=([^\s]+)',
        line,
    )
    if m and current_sim:
        ret = m.group(1).replace('%', '')
        hac = m.group(2)
        maxdd = m.group(3).replace('%', '')
        wr = m.group(4).replace('%', '')
        pf = m.group(5)

        parts = current_sim.split('__')
        if len(parts) == 3:
            window, model, sim_cfg = parts
        else:
            window = parts[0] if parts else current_sim
            model = parts[1] if len(parts) > 1 else ''
            sim_cfg = '__'.join(parts[2:]) if len(parts) > 2 else ''

        try:
            results.append({
                'window': window,
                'model': model,
                'sim_cfg': sim_cfg,
                'ret': float(ret.replace('+', '')),
                'hac': float(hac.replace('+', '')),
                'maxdd': float(maxdd.replace('+', '')),
                'wr': float(wr.replace('+', '')),
                'pf': float(pf),
            })
        except ValueError:
            pass  # skip N/A results
        current_sim = None

print(f"Total sims parsed: {len(results)}")
print()

# ═══════════════════════════════════════════
# TABLE 1: All 1x results by window, sorted by HAC Sharpe
# ═══════════════════════════════════════════
test_periods = {
    'WinA': '2024-07 → 2024-12 (H2 2024, bull)',
    'WinB': '2025-01 → 2025-06 (H1 2025, mixed)',
    'WinC': '2025-07 → 2025-12 (H2 2025, low disp)',
}

for win_label in ['WinA_train2024H1', 'WinB_train2024', 'WinC_train2025H1']:
    win_results = [r for r in results if r['window'] == win_label and '1x' in r['sim_cfg']]
    win_results.sort(key=lambda x: x['hac'], reverse=True)

    win_short = win_label.split('_')[0]

    print(f"{'═' * 95}")
    print(f"  {win_short} — OOS: {test_periods.get(win_short, '?')}  ({len(win_results)} sims, 1x lev)")
    print(f"{'═' * 95}")
    print(f"  {'#':>3}  {'Model':<42} {'Ret%':>7} {'HAC':>6} {'MaxDD':>7} {'WR%':>5} {'PF':>5}")
    print(f"  {'─' * 3}  {'─' * 42} {'─' * 7} {'─' * 6} {'─' * 7} {'─' * 5} {'─' * 5}")

    for i, r in enumerate(win_results):
        marker = " ★" if i == 0 else ""
        print(
            f"  {i + 1:>3}  {r['model']:<42} "
            f"{r['ret']:>+7.1f} {r['hac']:>+6.2f} {r['maxdd']:>7.1f} "
            f"{r['wr']:>5.0f} {r['pf']:>5.2f}{marker}"
        )
    print()

# ═══════════════════════════════════════════
# TABLE 2: Cross-window comparison (1x only)
# ═══════════════════════════════════════════
all_models = set()
for r in results:
    if '1x' in r['sim_cfg']:
        all_models.add(r['model'])

lookup = {}
for r in results:
    if '1x' in r['sim_cfg']:
        lookup[(r['window'], r['model'])] = r

wins = ['WinA_train2024H1', 'WinB_train2024', 'WinC_train2025H1']

print(f"\n{'═' * 105}")
print(f"  CROSS-WINDOW COMPARISON (1x leverage, HAC Sharpe)")
print(f"{'═' * 105}")
print(
    f"  {'Model':<42} {'WinA':>8} {'WinB':>8} {'WinC':>8} "
    f"{'Avg':>7} {'Std':>6} {'Min':>7}"
)
print(f"  {'─' * 42} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 7} {'─' * 6} {'─' * 7}")

model_avgs = []
for model in sorted(all_models):
    hacs = []
    row = f"  {model:<42}"
    for w in wins:
        r = lookup.get((w, model))
        if r:
            row += f" {r['hac']:>+8.2f}"
            hacs.append(r['hac'])
        else:
            row += f" {'N/A':>8}"

    if len(hacs) >= 2:
        avg = sum(hacs) / len(hacs)
        std = (sum((h - avg) ** 2 for h in hacs) / len(hacs)) ** 0.5
        mn = min(hacs)
        row += f" {avg:>+7.2f} {std:>6.2f} {mn:>+7.2f}"
        model_avgs.append((model, avg, std, mn, hacs))
    print(row)

# ═══════════════════════════════════════════
# TABLE 3: Top 10 by average HAC
# ═══════════════════════════════════════════
print(f"\n{'═' * 105}")
print(f"  TOP 10 MODELS BY AVERAGE HAC SHARPE (1x, across all 3 windows)")
print(f"{'═' * 105}")
model_avgs.sort(key=lambda x: x[1], reverse=True)
for i, (model, avg, std, mn, hacs) in enumerate(model_avgs[:10]):
    hac_str = " | ".join(f"{h:+.2f}" for h in hacs)
    flag = "⚠️ unstable" if std > 1.5 else "✓ stable" if std < 1.0 else "~ ok"
    print(f"  {i + 1:>2}. {model:<42} avg={avg:>+.2f}  std={std:.2f}  min={mn:>+.2f}  [{hac_str}]  {flag}")

# ═══════════════════════════════════════════
# TABLE 4: Solo vs Ensemble (focus question)
# ═══════════════════════════════════════════
print(f"\n{'═' * 105}")
print(f"  SOLO vs ENSEMBLE (1x, by window)")
print(f"{'═' * 105}")
solo_models = [m for m in all_models if 'solo' in m]
ens_models = [m for m in all_models if 'ens' in m or 'v6v7' in m or 'trio' in m or 'quad' in m or 'full' in m]

for win_label in wins:
    win_short = win_label.split('_')[0]
    best_solo = max(
        [(m, lookup[(win_label, m)]['hac']) for m in solo_models if (win_label, m) in lookup],
        key=lambda x: x[1],
        default=None,
    )
    best_ens = max(
        [(m, lookup[(win_label, m)]['hac']) for m in ens_models if (win_label, m) in lookup],
        key=lambda x: x[1],
        default=None,
    )
    if best_solo and best_ens:
        delta = best_ens[1] - best_solo[1]
        winner = "ENS" if delta > 0 else "SOLO"
        print(
            f"  {win_short}: Best solo: {best_solo[0]:<35} HAC={best_solo[1]:+.2f}  |  "
            f"Best ens: {best_ens[0]:<35} HAC={best_ens[1]:+.2f}  → {winner} wins (Δ={delta:+.2f})"
        )

# ═══════════════════════════════════════════
# TABLE 5: CatBoost variants head-to-head
# ═══════════════════════════════════════════
print(f"\n{'═' * 105}")
print(f"  CATBOOST VARIANT COMPARISON (solo, 1x)")
print(f"{'═' * 105}")
cb_models = sorted([m for m in all_models if 'cb_solo' in m])
print(f"  {'Variant':<42} {'WinA':>8} {'WinB':>8} {'WinC':>8} {'Avg':>7}")
print(f"  {'─' * 42} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 7}")
for model in cb_models:
    hacs = []
    row = f"  {model:<42}"
    for w in wins:
        r = lookup.get((w, model))
        if r:
            row += f" {r['hac']:>+8.2f}"
            hacs.append(r['hac'])
        else:
            row += f" {'N/A':>8}"
    if hacs:
        row += f" {sum(hacs)/len(hacs):>+7.2f}"
    print(row)

# ═══════════════════════════════════════════
# TABLE 6: News impact
# ═══════════════════════════════════════════
print(f"\n{'═' * 105}")
print(f"  NEWS IMPACT: with_news vs no_news vs market_news (CatBoost solo, 1x)")
print(f"{'═' * 105}")
news_pairs = [
    ('cb_solo_news', 'cb_solo_no_news', 'cb_solo_market_news'),
    ('cb_solo_news_no_deriv', 'cb_solo_no_deriv', None),
]
for news, no_news, market in news_pairs:
    print(f"  Pair: {news} vs {no_news}" + (f" vs {market}" if market else ""))
    for w in wins:
        ws = w.split('_')[0]
        r_news = lookup.get((w, news))
        r_no = lookup.get((w, no_news))
        r_mkt = lookup.get((w, market)) if market else None
        parts = [f"  {ws}:"]
        if r_news:
            parts.append(f"news={r_news['hac']:+.2f}")
        if r_no:
            parts.append(f"no_news={r_no['hac']:+.2f}")
        if r_mkt:
            parts.append(f"market={r_mkt['hac']:+.2f}")
        if r_news and r_no:
            delta = r_news['hac'] - r_no['hac']
            parts.append(f"Δ={delta:+.2f} ({'news+' if delta > 0 else 'news-'})")
        print("  ".join(parts))
    print()

# ═══════════════════════════════════════════
# TABLE 7: Derivatives impact
# ═══════════════════════════════════════════
print(f"{'═' * 105}")
print(f"  DERIVATIVES IMPACT: with_deriv vs no_deriv (CatBoost solo, 1x)")
print(f"{'═' * 105}")
deriv_pairs = [
    ('cb_solo_no_news', 'cb_solo_no_deriv'),
    ('cb_solo_news', 'cb_solo_news_no_deriv'),
    ('cb_solo_huber', 'cb_solo_huber_no_deriv'),
]
for with_d, no_d in deriv_pairs:
    print(f"  Pair: {with_d} vs {no_d}")
    for w in wins:
        ws = w.split('_')[0]
        r_wd = lookup.get((w, with_d))
        r_nd = lookup.get((w, no_d))
        if r_wd and r_nd:
            delta = r_wd['hac'] - r_nd['hac']
            print(f"    {ws}: with={r_wd['hac']:+.2f}  no={r_nd['hac']:+.2f}  Δ={delta:+.2f} ({'deriv+' if delta > 0 else 'deriv-'})")
    print()

# ═══════════════════════════════════════════
# TABLE 8: 3x leverage analysis
# ═══════════════════════════════════════════
print(f"{'═' * 105}")
print(f"  3x LEVERAGE: Sharpe degradation")
print(f"{'═' * 105}")
print(f"  {'Window':<6} {'Model':<35} {'1x HAC':>8} {'3x HAC':>8} {'Ratio':>6} {'3x Ret%':>8} {'3x DD%':>7}")
print(f"  {'─' * 6} {'─' * 35} {'─' * 8} {'─' * 8} {'─' * 6} {'─' * 8} {'─' * 7}")
for r in results:
    if '3x' in r['sim_cfg']:
        r1x = lookup.get((r['window'], r['model']))
        hac_1x = r1x['hac'] if r1x else 0
        ratio = r['hac'] / hac_1x if hac_1x != 0 else 0
        ws = r['window'].split('_')[0]
        print(
            f"  {ws:<6} {r['model']:<35} {hac_1x:>+8.2f} {r['hac']:>+8.2f} "
            f"{ratio:>6.2f} {r['ret']:>+8.1f} {r['maxdd']:>7.1f}"
        )

# ═══════════════════════════════════════════
# FINAL VERDICT
# ═══════════════════════════════════════════
print(f"\n{'═' * 105}")
print(f"  KEY FINDINGS SUMMARY")
print(f"{'═' * 105}")

# Best model per window
for w in wins:
    ws = w.split('_')[0]
    best = max(
        [(m, lookup[(w, m)]) for m in all_models if (w, m) in lookup],
        key=lambda x: x[1]['hac'],
    )
    print(f"  {ws} best: {best[0]:<40} HAC={best[1]['hac']:+.2f}  Ret={best[1]['ret']:+.1f}%")

# Overall best
print()
best_overall = max(model_avgs, key=lambda x: x[1])
print(f"  Overall best (avg HAC): {best_overall[0]:<35} avg={best_overall[1]:+.2f}")
worst_winc = min(
    [(m, lookup[('WinC_train2025H1', m)]['hac']) for m in all_models if ('WinC_train2025H1', m) in lookup],
    key=lambda x: abs(x[1]),
)
best_winc = max(
    [(m, lookup[('WinC_train2025H1', m)]['hac']) for m in all_models if ('WinC_train2025H1', m) in lookup],
    key=lambda x: x[1],
)
print(f"  WinC best:  {best_winc[0]:<35} HAC={best_winc[1]:+.2f}")
print(f"  WinC worst: {worst_winc[0]:<35} HAC={worst_winc[1]:+.2f}")
