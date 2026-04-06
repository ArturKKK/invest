#!/bin/bash
# Ждем завершения Phase 1+2 (PID 87276), потом запускаем Phase 3+4

cd /Users/a.s.tabakov/Developer/invest

echo "=== WAITING FOR Phase 1+2 (PID 87276) to finish ==="
echo "Started waiting: $(date)"

# Ждем пока процесс 87276 жив
while kill -0 87276 2>/dev/null; do
    LINES=$(wc -l < results_r48_phase12.log 2>/dev/null || echo 0)
    echo "  [$(date '+%H:%M:%S')] Phase1+2 still running... log=${LINES} lines"
    sleep 60
done

echo ""
echo "=== Phase 1+2 DONE: $(date) ==="
echo ""

# Сохраним результат Phase 1+2 для истории
cp results_r48_phase12.log results_r48_phase12_FINAL.log 2>/dev/null

# Запускаем Phase 3+4
echo "=== STARTING Phase 3+4: $(date) ==="
./venv/bin/python3.10 _research_r48_phase34.py >> results_r48_phase34.log 2>&1
EXIT_CODE=$?

echo ""
echo "=== Phase 3+4 EXIT CODE: $EXIT_CODE at $(date) ==="
cat results_r48_phase34.log >> results_r48_master_full.log
echo "=== ALL DONE: $(date) ===" >> results_r48_master_full.log
