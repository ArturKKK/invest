#!/usr/bin/env python3
"""Download public stablecoin data from stablecoins.llama.fi."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import quote

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "sentiment"
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://stablecoins.llama.fi"

OUT_ASSETS = DATA_DIR / "llama_stablecoins_assets.parquet"
OUT_CHAINS = DATA_DIR / "llama_stablecoin_chains.parquet"
OUT_GLOBAL = DATA_DIR / "llama_stablecoin_chart_all.parquet"
OUT_CHAIN_CHARTS = DATA_DIR / "llama_stablecoin_chart_by_chain.parquet"


def get_json(path: str):
    response = requests.get(f"{BASE_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def flatten_mapping(prefix: str, mapping: Dict | None) -> Dict:
    out: Dict = {}
    if not isinstance(mapping, dict):
        return out
    for key, value in mapping.items():
        out[f"{prefix}_{key}"] = value
    return out


def normalize_assets(pegged_assets: Iterable[Dict]) -> pd.DataFrame:
    rows: List[Dict] = []
    for asset in pegged_assets:
        row = {
            "id": asset.get("id"),
            "name": asset.get("name"),
            "symbol": asset.get("symbol"),
            "gecko_id": asset.get("gecko_id"),
            "pegType": asset.get("pegType"),
            "pegMechanism": asset.get("pegMechanism"),
            "priceSource": asset.get("priceSource"),
            "circulating_peggedUSD": asset.get("circulating", {}).get("peggedUSD") if isinstance(asset.get("circulating"), dict) else None,
            "chains": ",".join(asset.get("chains", [])) if isinstance(asset.get("chains"), list) else None,
            "chain_count": len(asset.get("chains", [])) if isinstance(asset.get("chains"), list) else None,
        }
        row.update(flatten_mapping("circulating", asset.get("circulating")))
        row.update(flatten_mapping("circulatingPrevDay", asset.get("circulatingPrevDay")))
        row.update(flatten_mapping("circulatingPrevWeek", asset.get("circulatingPrevWeek")))
        rows.append(row)
    return pd.DataFrame(rows)


def normalize_chains(chains: Iterable[Dict]) -> pd.DataFrame:
    rows: List[Dict] = []
    for chain in chains:
        row = {
            "name": chain.get("name"),
            "tokenSymbol": chain.get("tokenSymbol"),
            "gecko_id": chain.get("gecko_id"),
        }
        row.update(flatten_mapping("totalCirculatingUSD", chain.get("totalCirculatingUSD")))
        row.update(flatten_mapping("totalCirculating", chain.get("totalCirculating")))
        rows.append(row)
    return pd.DataFrame(rows)


def normalize_chart(rows: Iterable[Dict], label: str, label_name: str) -> pd.DataFrame:
    out_rows: List[Dict] = []
    for row in rows:
        date_value = pd.to_datetime(int(row["date"]), unit="s", utc=True)
        flat = {label_name: label, "date": date_value}
        flat.update(flatten_mapping("totalCirculating", row.get("totalCirculating")))
        flat.update(flatten_mapping("totalCirculatingUSD", row.get("totalCirculatingUSD")))
        flat.update(flatten_mapping("totalMintedUSD", row.get("totalMintedUSD")))
        out_rows.append(flat)
    return pd.DataFrame(out_rows)


def fetch_chain_chart(chain_name: str) -> Optional[pd.DataFrame]:
    try:
        payload = get_json(f"/stablecoincharts/{quote(chain_name)}")
        if not isinstance(payload, list) or not payload:
            return None
        return normalize_chart(payload, chain_name, "chain")
    except Exception:
        return None


def main() -> None:
    print("=" * 70)
    print("  DEFI LLAMA STABLECOINS DOWNLOADER")
    print("=" * 70)

    print("\n[1] Fetching asset snapshot...")
    assets_payload = get_json("/stablecoins")
    pegged_assets = assets_payload.get("peggedAssets", [])
    assets_df = normalize_assets(pegged_assets)
    assets_df.to_parquet(OUT_ASSETS, index=False)
    print(f"  Assets: {len(assets_df):,} rows -> {OUT_ASSETS.name}")

    print("\n[2] Fetching chain snapshot...")
    chains_payload = get_json("/stablecoinchains")
    chains_df = normalize_chains(chains_payload)
    chains_df.to_parquet(OUT_CHAINS, index=False)
    print(f"  Chains: {len(chains_df):,} rows -> {OUT_CHAINS.name}")

    print("\n[3] Fetching global history...")
    global_payload = get_json("/stablecoincharts/all")
    global_df = normalize_chart(global_payload, "all", "scope")
    global_df.to_parquet(OUT_GLOBAL, index=False)
    print(f"  Global history: {len(global_df):,} rows -> {OUT_GLOBAL.name}")

    print("\n[4] Fetching per-chain history...")
    chain_names = sorted(chains_df["name"].dropna().astype(str).unique().tolist())
    chain_frames: List[pd.DataFrame] = []
    failures: List[str] = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(fetch_chain_chart, chain_name): chain_name for chain_name in chain_names}
        for index, future in enumerate(as_completed(futures), start=1):
            chain_name = futures[future]
            frame = future.result()
            if frame is None or frame.empty:
                failures.append(chain_name)
            else:
                chain_frames.append(frame)
            if index % 25 == 0 or index == len(chain_names):
                print(f"  Progress: {index}/{len(chain_names)} chains")
            time.sleep(0.02)

    if chain_frames:
        chain_history_df = pd.concat(chain_frames, ignore_index=True)
        chain_history_df.to_parquet(OUT_CHAIN_CHARTS, index=False)
        print(f"  Chain history: {len(chain_history_df):,} rows -> {OUT_CHAIN_CHARTS.name}")
    else:
        chain_history_df = pd.DataFrame()
        print("  Chain history: no rows saved")

    print("\n[5] Summary")
    print(f"  Asset snapshot rows: {len(assets_df):,}")
    print(f"  Chain snapshot rows: {len(chains_df):,}")
    print(f"  Global history rows: {len(global_df):,}")
    print(f"  Chain history rows: {len(chain_history_df):,}")
    print(f"  Chain fetch failures: {len(failures):,}")


if __name__ == "__main__":
    main()