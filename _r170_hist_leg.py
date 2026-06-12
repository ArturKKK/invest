#!/usr/bin/env python3
"""R170 — HIST transformer companion leg, training half. GPU VM ONLY.

Adapts run_hist_v2.py (old best: Rank IC 0.0752, HIST+LGB was the top gross
combo) to the current honest protocol:
  - W2/W3 walk-forward windows (same splits as r68.CONTINUOUS_WINDOWS[1:])
  - 12h target (champion horizon), target = per-timestamp pct-rank of fwd ret
  - 3+3 torch seeds: std [0,7,13], alt [1,8,14]; preds seed-averaged per batch
  - known-good config kept: d_model 128, 2L/4H, ic_weight 0.5, lr 1e-4

Outputs (per batch): cache/r170_hist_{std,alt}_preds.parquet with
(timestamp, symbol, pred_hist = seed-mean raw score over ALL 50 frame symbols).
The blend half (_r170b, CPU VM) subsets to the champion universe, re-ranks
per timestamp, and runs the pre-registered gate vs the frozen stack.

Inputs: data/features/crypto_features_1h.parquet (+ optional data/sentiment/
{fear_greed,funding_rates,long_short_ratio}.parquet) — ship via hist_data.tar.gz.
Run:  python3 _r170_hist_leg.py --device cuda
Ship: cp cache/r170_hist_*_preds.parquet /data/datasets/
"""
import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_hist_v2 import (EXCLUDE_COLS, REGIME_COLS, CRYPTO_CONCEPTS, N_CONCEPTS,
                         add_cross_asset_features, add_sentiment_features,
                         cross_sectional_rank)

HORIZON = 12
SEED_BATCHES = {"std": [0, 7, 13], "alt": [1, 8, 14]}
WINDOWS = [
    {"name": "W2", "train_end": "2025-01-01", "val_start": "2025-01-01",
     "val_end": "2025-04-30", "test_start": "2025-05-15", "test_end": "2025-11-14"},
    {"name": "W3", "train_end": "2025-07-01", "val_start": "2025-07-01",
     "val_end": "2025-10-31", "test_start": "2025-11-15", "test_end": "2026-03-17"},
]


def build_frame(project_root):
    feat_path = os.path.join(project_root, "data", "features", "crypto_features_1h.parquet")
    df = pd.read_parquet(feat_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    print(f"frame: {df.shape}, syms={df['symbol'].nunique()}, "
          f"{df['timestamp'].min()} -> {df['timestamp'].max()}", flush=True)
    df[f"target_ret_{HORIZON}h"] = df.groupby("symbol")["close"].transform(
        lambda x: x.pct_change(HORIZON).shift(-HORIZON))
    df = add_cross_asset_features(df)
    df = add_sentiment_features(df, project_root)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[f"target_ret_{HORIZON}h"])
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS
                 and not c.startswith("target_") and c not in REGIME_COLS]
    feat_cols = [c for c in feat_cols if df[c].dtype in ["float64", "float32", "int64", "int32"]]
    df[feat_cols] = df[feat_cols].fillna(0)
    df = cross_sectional_rank(df, feat_cols)
    df["target_rank"] = df.groupby("timestamp")[f"target_ret_{HORIZON}h"].rank(pct=True)
    print(f"features: {len(feat_cols)}", flush=True)
    return df, feat_cols


def make_xs(df, feat_cols, symbols, sym2idx, lo, hi):
    """(T,N,F) cross-section tensors for timestamp range [lo, hi)."""
    s = df[(df["timestamp"] >= lo) & (df["timestamp"] < hi)].copy()
    timestamps = sorted(s["timestamp"].unique())
    ts2idx = {ts: i for i, ts in enumerate(timestamps)}
    T, N, F = len(timestamps), len(symbols), len(feat_cols)
    X = np.zeros((T, N, F), dtype=np.float32)
    y = np.zeros((T, N), dtype=np.float32)
    mask = np.zeros((T, N), dtype=np.float32)
    ti = s["timestamp"].map(ts2idx).values
    si = s["symbol"].map(sym2idx).values
    for i, c in enumerate(feat_cols):
        X[ti, si, i] = s[c].values.astype(np.float32)
    y[ti, si] = s["target_rank"].values.astype(np.float32)
    mask[ti, si] = 1.0
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)
    return X, y, mask, timestamps


def train_one(Xtr, ytr, mtr, Xva, yva, mva, Xte, mte, concept_ids,
              feat_cols, seed, device_str):
    import torch
    import torch.nn as nn
    import torch.nn.functional as TF
    from torch.utils.data import TensorDataset, DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")

    sent_keywords = ["fng", "funding", "reversal", "surge", "dispersion",
                     "long_short", "btc_beta", "market_avg_funding", "market_funding"]
    sent_idx = [i for i, c in enumerate(feat_cols) if any(k in c for k in sent_keywords)]
    tech_idx = [i for i in range(len(feat_cols)) if i not in sent_idx]

    # model is imported lazily from run_hist_v2's closure-defined class — rebuild
    # it here verbatim instead (the original defines it inside a function).
    d_model, n_heads, n_layers, dropout = 128, 4, 2, 0.1

    class HISTv2Model(nn.Module):
        def __init__(self, n_tech, n_sent):
            super().__init__()
            self.tech_embed = nn.Sequential(
                nn.Linear(n_tech, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
            sent_hidden = max(32, min(d_model // 2, n_sent * 4))
            self.sent_embed = nn.Sequential(
                nn.Linear(n_sent, sent_hidden), nn.LayerNorm(sent_hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(sent_hidden, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
            self.fusion_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
            self.concept_embed = nn.Embedding(N_CONCEPTS, d_model)
            self.concept_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
            enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                             dim_feedforward=d_model * 4, dropout=dropout,
                                             batch_first=True, activation="gelu")
            self.cross_attn = nn.TransformerEncoder(enc, num_layers=n_layers)
            self.hidden_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
            self.head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(d_model // 2, 1))

        def forward(self, x, concepts_t, tech_i, sent_i):
            h_tech = self.tech_embed(x[:, :, tech_i])
            h_sent = self.sent_embed(x[:, :, sent_i])
            gate = self.fusion_gate(torch.cat([h_tech, h_sent], dim=-1))
            h = h_tech + gate * h_sent
            B = x.shape[0]
            concepts = self.concept_embed(concepts_t).unsqueeze(0).expand(B, -1, -1)
            cgate = self.concept_gate(torch.cat([h, concepts], dim=-1))
            hc = h + cgate * concepts
            hidden = self.cross_attn(hc)
            hgate = self.hidden_gate(torch.cat([hc, hidden], dim=-1))
            hf = hc + hgate * (hidden - hc)
            return self.head(hf).squeeze(-1)

    def ic_loss(pred, target, mask):
        losses = []
        for i in range(pred.shape[0]):
            m = mask[i] > 0.5
            if m.sum() < 5: continue
            p = pred[i][m] - pred[i][m].mean()
            t = target[i][m] - target[i][m].mean()
            losses.append(-(p * t).sum() / (p.norm() * t.norm() + 1e-8))
        if not losses:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return torch.stack(losses).mean()

    model = HISTv2Model(len(tech_idx), len(sent_idx)).to(device)
    concepts_t = torch.LongTensor(concept_ids).to(device)
    tech_t = torch.LongTensor(tech_idx).to(device)
    sent_t = torch.LongTensor(sent_idx).to(device)

    EPOCHS, PATIENCE, LR, BS, IC_W = 80, 15, 1e-4, 64, 0.5
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    warm = 5
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (e + 1) / warm if e < warm
                                              else 0.5 * (1 + np.cos(np.pi * (e - warm) / max(1, EPOCHS - warm))))
    g = torch.Generator(); g.manual_seed(seed)
    tr_loader = DataLoader(TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(ytr),
                                         torch.FloatTensor(mtr)),
                           batch_size=BS, shuffle=True, generator=g)
    va_X = torch.FloatTensor(Xva)

    best_ic, best_state, best_ep, bad = -999, None, 0, 0
    for ep in range(EPOCHS):
        model.train()
        for xb, yb, mb in tr_loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            pred = model(xb, concepts_t, tech_t, sent_t)
            loss = 0.5 * TF.mse_loss(pred[mb > 0.5], yb[mb > 0.5]) + IC_W * ic_loss(pred, yb, mb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vp = np.concatenate([model(va_X[i:i + 256].to(device), concepts_t, tech_t, sent_t).cpu().numpy()
                                 for i in range(0, len(va_X), 256)], axis=0)
        ics = []
        for t in range(vp.shape[0]):
            m = mva[t] > 0.5
            if m.sum() < 10: continue
            c, _ = spearmanr(vp[t][m], yva[t][m])
            if not np.isnan(c): ics.append(c)
        vic = float(np.mean(ics)) if ics else 0.0
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"      ep{ep+1:3d} val_ic={vic:+.4f}", flush=True)
        if vic > best_ic:
            best_ic, best_ep, bad = vic, ep + 1, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()
    te_X = torch.FloatTensor(Xte)
    with torch.no_grad():
        tp = np.concatenate([model(te_X[i:i + 256].to(device), concepts_t, tech_t, sent_t).cpu().numpy()
                             for i in range(0, len(te_X), 256)], axis=0)
    print(f"    seed {seed}: best ep {best_ep} val_ic {best_ic:+.4f}", flush=True)
    return tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(root, "cache"), exist_ok=True)

    df, feat_cols = build_frame(root)
    symbols = sorted(df["symbol"].unique())
    sym2idx = {s: i for i, s in enumerate(symbols)}
    concept_ids = np.array([CRYPTO_CONCEPTS.get(s, 7) for s in symbols])
    tz = df["timestamp"].dt.tz
    t0 = df["timestamp"].min()

    meta = {}
    for tag, seeds in SEED_BATCHES.items():
        frames = []
        for w in WINDOWS:
            tr_end = pd.Timestamp(w["train_end"], tz=tz)
            va_s, va_e = pd.Timestamp(w["val_start"], tz=tz), pd.Timestamp(w["val_end"], tz=tz)
            te_s, te_e = pd.Timestamp(w["test_start"], tz=tz), pd.Timestamp(w["test_end"], tz=tz) + pd.Timedelta(hours=23)
            print(f"\n[{tag}/{w['name']}] building tensors...", flush=True)
            Xtr, ytr, mtr, _ = make_xs(df, feat_cols, symbols, sym2idx, t0, tr_end)
            Xva, yva, mva, _ = make_xs(df, feat_cols, symbols, sym2idx, va_s, va_e)
            Xte, _, mte, te_ts = make_xs(df, feat_cols, symbols, sym2idx, te_s, te_e)
            print(f"  train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}", flush=True)
            preds = []
            for seed in seeds:
                preds.append(train_one(Xtr, ytr, mtr, Xva, yva, mva, Xte, mte,
                                       concept_ids, feat_cols, seed, args.device))
            tp = np.mean(preds, axis=0)
            rows = []
            for t_idx, ts in enumerate(te_ts):
                m = mte[t_idx] > 0.5
                for s_idx in np.where(m)[0]:
                    rows.append((ts, symbols[s_idx], float(tp[t_idx, s_idx]), w["name"]))
            frames.append(pd.DataFrame(rows, columns=["timestamp", "symbol", "pred_hist", "window"]))
            del Xtr, ytr, mtr, Xva, yva, mva, Xte, mte
        out = pd.concat(frames, ignore_index=True)
        path = os.path.join(root, "cache", f"r170_hist_{tag}_preds.parquet")
        out.to_parquet(path, index=False)
        meta[tag] = {"rows": len(out), "path": path}
        print(f"\n[{tag}] SAVED {len(out):,} rows -> {path}", flush=True)

    with open(os.path.join(root, "results_r170_hist_train.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print("\nShip preds: cp cache/r170_hist_*_preds.parquet /data/datasets/")
    print("R170 done.")


if __name__ == "__main__":
    main()
