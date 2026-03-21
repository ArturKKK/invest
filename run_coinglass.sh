#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  CoinGlass Data Download — Step-by-step runner
# ─────────────────────────────────────────────────────────────
#
#  Скачивает историю деривативных данных с CoinGlass API.
#  План: 3 дня, по приоритету.
#
#  Перед запуском:
#    1. Купить подписку: https://www.coinglass.com/pricing (Hobbyist $29/mo)
#    2. Прописать ключ:
#       echo 'COINGLASS_API_KEY=your_key_here' >> .env
#
#  Использование:
#    ./run_coinglass.sh probe       # проверить ключ
#    ./run_coinglass.sh day1        # День 1: ликвидации (приоритет)
#    ./run_coinglass.sh day2        # День 2: OI + funding
#    ./run_coinglass.sh day3        # День 3: L/S + taker + netflow + premium
#    ./run_coinglass.sh all         # всё сразу (~4 часа)
# ─────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${SCRIPT_DIR}/venv/bin/python"

# Fallback to system python if venv not found
if [[ ! -f "$PYTHON" ]]; then
    PYTHON="python3"
fi

DOWNLOADER="src/data/download_coinglass.py"

# Start date: aligned with binance_futures_metrics (Dec 2021)
START="2021-12-01"

echo "═══════════════════════════════════════════════════"
echo "  CoinGlass Downloader"
echo "  Python: $PYTHON"
echo "  Start:  $START"
echo "═══════════════════════════════════════════════════"

case "${1:-help}" in
    probe)
        echo "🔍 Testing API key..."
        $PYTHON "$DOWNLOADER" --probe --start "$START"
        ;;

    day1)
        echo "📊 Day 1: Liquidations (main priority)"
        echo "   Estimated: ~50 symbols × ~500 req ≈ 5-6 min"
        $PYTHON "$DOWNLOADER" --only liquidations --start "$START"
        ;;

    day2)
        echo "📊 Day 2: OI History + Funding Rates"
        $PYTHON "$DOWNLOADER" --only oi,funding --start "$START"
        ;;

    day3)
        echo "📊 Day 3: Long/Short + Taker + Netflow + Premium"
        $PYTHON "$DOWNLOADER" --only longshort,taker,netflow,premium --start "$START"
        ;;

    all)
        echo "📊 Downloading ALL endpoints (~4 hours)"
        $PYTHON "$DOWNLOADER" --start "$START"
        ;;

    status)
        echo "📁 Downloaded files:"
        echo ""
        DATA_DIR="data/raw/coinglass"
        if [[ -d "$DATA_DIR" ]]; then
            ls -lh "$DATA_DIR"/*.parquet 2>/dev/null || echo "   No parquet files yet"
            echo ""
            $PYTHON -c "
import os, pandas as pd
d = '$DATA_DIR'
for f in sorted(os.listdir(d)):
    if f.endswith('.parquet'):
        df = pd.read_parquet(os.path.join(d, f))
        ts_min = pd.to_datetime(df['timestamp']).min()
        ts_max = pd.to_datetime(df['timestamp']).max()
        nsym = df['symbol'].nunique() if 'symbol' in df.columns else '-'
        print(f'  {f:40s} {len(df):>10,} rows  {nsym} syms  {ts_min:%Y-%m-%d} → {ts_max:%Y-%m-%d}')
" 2>/dev/null || true
        else
            echo "   Directory $DATA_DIR does not exist yet"
        fi
        ;;

    *)
        echo "Usage: $0 {probe|day1|day2|day3|all|status}"
        echo ""
        echo "  probe  — test API key"
        echo "  day1   — liquidations (priority #1)"
        echo "  day2   — OI history + funding rates"
        echo "  day3   — L/S ratio + taker volume + netflow + premium"
        echo "  all    — everything at once (~4 hours)"
        echo "  status — show downloaded data summary"
        echo ""
        echo "Before running, set your API key:"
        echo "  echo 'COINGLASS_API_KEY=your_key' >> .env"
        ;;
esac
