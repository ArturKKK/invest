#!/usr/bin/env python3
"""Quick baseline verification: run ONLY EXP A (31 champion features) and print Sharpe."""
import sys, time, warnings
warnings.filterwarnings("ignore")

from _research_r123_news_sentiment import (
    load_data, CHAMPION_FEAT_31, MARKET_LEVEL_FEATURES,
    CONTINUOUS_WINDOWS, SEEDS, train_ensemble, run_experiment
)

def main():
    t0 = time.time()
    print("=" * 60)
    print("BASELINE VERIFICATION (EXP A only)")
    print("=" * 60)

    print("\n[1/2] Loading data...")
    df, regime_df = load_data()
    print(f"  df shape: {df.shape}")

    print("\n[2/2] Training baseline (31 champion features)...")
    base_feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    no_rank_base = [f for f in base_feats if f in MARKET_LEVEL_FEATURES]
    print(f"  Features: {len(base_feats)}")

    t1 = time.time()
    preds_a = train_ensemble(df, base_feats, CONTINUOUS_WINDOWS,
                             seeds=SEEDS, cs_rank_exclude=no_rank_base)
    print(f"  Trained in {time.time()-t1:.0f}s")

    result = run_experiment(preds_a, regime_df, "A_baseline_verify")

    print("\n" + "=" * 60)
    print("RESULT:")
    print(f"  Net Sharpe:  {result.get('net_sharpe', 'N/A')}")
    print(f"  Max DD:      {result.get('max_dd', 'N/A')}")
    print(f"  Calmar:      {result.get('calmar', 'N/A')}")
    print(f"  Total time:  {time.time()-t0:.0f}s")
    print("=" * 60)

    expected = 2.831
    actual = result.get('net_sharpe', 0)
    if abs(actual - expected) < 0.01:
        print(f"\n✅ MATCH: {actual:.3f} ≈ {expected:.3f} (S6 champion)")
    else:
        print(f"\n❌ MISMATCH: {actual:.3f} != {expected:.3f}")

if __name__ == "__main__":
    main()
