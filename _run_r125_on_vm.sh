#!/bin/bash
# R125: FinBERT News Sentiment — GPU VM Runner
# Run with: setsid bash _run_r125_on_vm.sh > results/r125_log.txt 2>&1 < /dev/null &
set -e
cd ~/Dev/invest
mkdir -p results

echo "=== R125 FinBERT News Sentiment ==="
echo "Started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# Step 1: Re-score all news with FinBERT
echo "=== Step 1/3: FinBERT Re-scoring ==="
echo "This will score ~954K news items on GPU..."
python3 fetch_crypto_news.py --skip-fetch --scorer finbert
echo "Step 1 done: $(date)"
echo ""

# Step 2: Backup old results, then run R123 experiments with FinBERT features
echo "=== Step 2/3: Running R125 Experiments ==="
cp results/r123_news_sentiment.json results/r123_news_sentiment_vader_backup.json 2>/dev/null || true
python3 _research_r123_news_sentiment.py
echo "Step 2 done: $(date)"
echo ""

# Step 3: Rename output to R125
echo "=== Step 3/3: Saving Results ==="
cp results/r123_news_sentiment.json results/r125_finbert.json
echo ""

echo "=== R125 COMPLETE ==="
echo "Finished: $(date)"
echo "Results: results/r125_finbert.json"
echo "Log: results/r125_log.txt"

# Signal completion
touch results/r125_done.flag
