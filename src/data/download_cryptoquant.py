#!/usr/bin/env python3
"""
Download on-chain data from CryptoQuant API (v1).

Free tier: 50 req/day, 7 days history, daily resolution.
Professional ($99/mo): 20 req/min, 1 year history.

Phase 1: Probe API to discover available endpoints per coin
Phase 2: Download all available metrics for our universe
Phase 3: Save as parquet for IC analysis

Usage:
  # Set API key:
  export CRYPTOQUANT_API_KEY=your_key_here

  # Run:
  python src/data/download_cryptoquant.py

  # Probe only (discover endpoints, no download):
  python src/data/download_cryptoquant.py --probe-only

  # Specific coin:
  python src/data/download_cryptoquant.py --coin btc
"""

import os
import sys
import json
import time
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Load .env if python-dotenv available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass

# ─── Config ──────────────────────────────────────────────────────────────────

BASE_URL = "https://api.cryptoquant.com/v1"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cryptoquant"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Rate limit: Free=50/day, Pro=20/min. We pace conservatively.
RATE_LIMIT_DELAY = 1.5  # seconds between requests (safe for all tiers)

# Our trading universe (lowercase for CQ API)
COINS = [
    "btc", "eth", "bnb", "sol", "xrp",
    "ada", "doge", "avax", "dot", "link",
    "matic", "uni", "atom", "ltc", "etc",
    "fil", "apt", "arb", "op", "near",
    "aave", "mkr", "grt", "inj", "ftm",
    "algo", "sand", "mana", "axs", "theta",
    "rune", "egld", "xtz", "flow", "chz",
]

# Map CQ coin → our symbol format
def to_our_symbol(cq_coin: str) -> str:
    return f"{cq_coin.upper()}/USDT"

# ─── Known endpoint structure ────────────────────────────────────────────────
# CryptoQuant API v1: /v1/{coin}/{category}/{metric}
# Some metrics are BTC/ETH-only (UTXO-based: SOPR, MVRV, etc.)
#
# Endpoint discovery: we probe each coin+category+metric combo.
# Working endpoints are cached in data/cryptoquant/_endpoints.json

# Categories and metrics to try (ordered by expected usefulness)
ENDPOINTS_TO_PROBE = {
    "exchange-flows": [
        "netflow",          # Net flow to/from exchanges (inflow - outflow)
        "inflow",           # Flow into exchanges (sell pressure)
        "outflow",          # Flow out of exchanges (accumulation)
        "reserve",          # Total balance on exchanges
        "netflow-total",    # Netflow total
        "inflow-total",     # Inflow total
        "outflow-total",    # Outflow total
        "transactions-count-inflow",
        "transactions-count-outflow",
    ],
    "network-data": [
        "active-addresses",
        "transactions-count",
        "addresses-count",
        "supply-on-exchanges",
        "tokens-transferred-total",
        "tokens-transferred-mean",
    ],
    "market-data": [
        "open-interest",
        "funding-rates",
        "liquidations",
        "taker-buy-sell-ratio",
        "taker-buy-sell-volume",
        "price-ohlcv",
    ],
    "mining-data": [         # BTC/ETH PoW only
        "hash-rate",
        "miner-reserve",
        "miner-outflow",
        "difficulty",
        "puell-multiple",
    ],
    "market-indicator": [    # Mostly BTC-specific
        "sopr",
        "mvrv",
        "nupl",
        "nvm",
        "realized-price",
        "sopr-adjusted",
    ],
    "fund-data": [
        "etf-netflow",
        "etf-volume",
        "grayscale-holdings",
    ],
    "flow-indicator": [
        "exchange-whale-ratio",
        "fund-flow-ratio",
        "stablecoin-supply-ratio",
    ],
}

# ─── API Client ──────────────────────────────────────────────────────────────

class CryptoQuantAPI:
    """Thin wrapper around CryptoQuant v1 REST API."""

    def __init__(self, api_key: str, verify_ssl: bool = True):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        })
        self.session.verify = verify_ssl
        self._last_request_time = 0
        self.total_requests = 0
        self.daily_budget = 50  # free tier; Pro = unlimited (20/min)

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def get(self, endpoint: str, params: dict = None, retries: int = 2) -> dict | None:
        url = f"{BASE_URL}{endpoint}"
        for attempt in range(retries):
            self._rate_limit()
            self.total_requests += 1
            try:
                resp = self.session.get(url, params=params, timeout=30)

                if resp.status_code == 429:
                    wait = min(60, 10 * (attempt + 1))
                    print(f"\n   ⚠️  Rate limited (429), waiting {wait}s...")
                    time.sleep(wait)
                    continue

                if resp.status_code in (401, 403):
                    print(f"\n   ❌ Auth error ({resp.status_code}): check API key")
                    return None

                if resp.status_code == 404:
                    return None  # endpoint doesn't exist

                if resp.status_code == 400:
                    return None  # bad request (wrong params)

                if resp.status_code != 200:
                    print(f"\n   ⚠️  HTTP {resp.status_code} for {endpoint}")
                    return None

                data = resp.json()
                return data

            except requests.exceptions.Timeout:
                print(f"\n   ⚠️  Timeout for {endpoint}, retry {attempt+1}")
                continue
            except Exception as e:
                print(f"\n   ⚠️  Error: {e}")
                return None
        return None

    def budget_remaining(self) -> int:
        return max(0, self.daily_budget - self.total_requests)


# ─── Phase 1: Endpoint Discovery ────────────────────────────────────────────

def probe_endpoints(api: CryptoQuantAPI, coins: list = None) -> dict:
    """
    Discover which endpoints work for which coins.
    Returns: {endpoint_path: [list_of_working_coins]}
    """
    cache_path = DATA_DIR / "_endpoints.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        print(f"   📋 Loaded cached endpoints ({len(cached)} working)")
        return cached

    if coins is None:
        # Probe with BTC first (most endpoints), then check a few altcoins
        coins = ["btc"]

    print(f"\n🔍 Phase 1: Probing CryptoQuant endpoints...")
    working = {}
    total_probes = 0

    for category, metrics in ENDPOINTS_TO_PROBE.items():
        for metric in metrics:
            for coin in coins:
                if api.budget_remaining() < 5:
                    print(f"\n   ⚠️  Budget nearly exhausted ({api.total_requests} requests used)")
                    break

                endpoint = f"/{coin}/{category}/{metric}"
                params = {"window": "day", "limit": 2}
                sys.stdout.write(f"\r   Probing {endpoint}...{' ' * 20}")

                resp = api.get(endpoint, params)
                total_probes += 1

                if resp is not None and "result" in resp:
                    data_points = len(resp.get("result", {}).get("data", []))
                    if data_points > 0:
                        if endpoint not in working:
                            working[endpoint] = {"coins": [], "sample_keys": []}
                        working[endpoint]["coins"].append(coin)
                        # Record the data keys for understanding the schema
                        sample = resp["result"]["data"][0]
                        working[endpoint]["sample_keys"] = list(sample.keys())
                        sys.stdout.write(f"\r   ✅ {endpoint} — {data_points} rows, keys={list(sample.keys())}\n")
                elif resp is not None and "status" in resp:
                    # Some APIs return data differently
                    working_key = endpoint
                    if resp.get("status", {}).get("code") == "SUCCESS":
                        if working_key not in working:
                            working[working_key] = {"coins": [], "sample_keys": []}
                        working[working_key]["coins"].append(coin)
                        sys.stdout.write(f"\r   ✅ {endpoint} — SUCCESS\n")

    sys.stdout.write(f"\r{' ' * 80}\r")
    print(f"\n   🔍 Probed {total_probes} endpoints, {len(working)} working")

    # Now probe altcoins for working endpoints
    if len(coins) == 1 and working:
        altcoins_to_test = ["eth", "sol", "link", "avax"]
        print(f"\n   Testing altcoin coverage for {len(working)} endpoints...")
        for endpoint, info in list(working.items()):
            for alt in altcoins_to_test:
                if api.budget_remaining() < 3:
                    break
                alt_endpoint = endpoint.replace("/btc/", f"/{alt}/")
                params = {"window": "day", "limit": 2}
                resp = api.get(alt_endpoint, params)
                total_probes += 1
                if resp is not None:
                    result = resp.get("result", {})
                    data = result.get("data", []) if isinstance(result, dict) else []
                    if data:
                        if alt_endpoint not in working:
                            working[alt_endpoint] = {"coins": [], "sample_keys": list(data[0].keys())}
                        working[alt_endpoint]["coins"].append(alt)
                        sys.stdout.write(f"\r   ✅ {alt_endpoint}\n")

    # Save cache
    cache_path.write_text(json.dumps(working, indent=2))
    print(f"\n   💾 Saved endpoint cache → {cache_path}")
    print(f"   📊 Total API requests used: {api.total_requests}/{api.daily_budget}")

    return working


# ─── Phase 2: Download Data ─────────────────────────────────────────────────

def download_metric(api: CryptoQuantAPI, coin: str, category: str, metric: str,
                    window: str = "day", limit: int = 10) -> pd.DataFrame | None:
    """Download a single metric for a single coin."""
    endpoint = f"/{coin}/{category}/{metric}"
    params = {"window": window, "limit": limit}

    resp = api.get(endpoint, params)
    if resp is None:
        return None

    # Parse response — CryptoQuant has varying response formats
    result = resp.get("result", resp.get("data", {}))
    if isinstance(result, dict):
        data = result.get("data", [])
    elif isinstance(result, list):
        data = result
    else:
        return None

    if not data:
        return None

    df = pd.DataFrame(data)
    df["symbol"] = to_our_symbol(coin)
    df["metric_name"] = f"{category}/{metric}"

    # Parse timestamps
    for ts_col in ["datetime", "date", "timestamp", "time", "t"]:
        if ts_col in df.columns:
            try:
                # Try unix timestamp (seconds or milliseconds)
                vals = pd.to_numeric(df[ts_col], errors="coerce")
                if vals.notna().any():
                    if vals.max() > 1e12:  # milliseconds
                        df["timestamp"] = pd.to_datetime(vals, unit="ms", utc=True)
                    else:
                        df["timestamp"] = pd.to_datetime(vals, unit="s", utc=True)
                else:
                    df["timestamp"] = pd.to_datetime(df[ts_col], utc=True)
            except Exception:
                df["timestamp"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
            break

    return df


def download_all(api: CryptoQuantAPI, working_endpoints: dict,
                 coins: list = None, limit: int = 10) -> pd.DataFrame:
    """Download all working metrics for all coins."""
    if coins is None:
        coins = COINS

    print(f"\n📥 Phase 2: Downloading data ({len(working_endpoints)} endpoints × coins)...")
    all_frames = []
    downloaded = 0

    # Group endpoints by category/metric (strip coin prefix)
    seen_cat_metrics = set()
    for ep in working_endpoints:
        parts = ep.strip("/").split("/")  # coin/category/metric
        if len(parts) >= 3:
            cat_metric = f"{parts[1]}/{parts[2]}"
            seen_cat_metrics.add(cat_metric)

    for cat_metric in sorted(seen_cat_metrics):
        category, metric = cat_metric.split("/", 1)

        for coin in coins:
            if api.budget_remaining() < 2:
                print(f"\n   ⚠️  Budget exhausted! ({api.total_requests} requests)")
                break

            sys.stdout.write(f"\r   [{downloaded+1}] {coin}/{category}/{metric}...{' '*20}")
            df = download_metric(api, coin, category, metric, limit=limit)

            if df is not None and len(df) > 0:
                all_frames.append(df)
                downloaded += 1
                sys.stdout.write(f"\r   ✅ {coin}/{category}/{metric} — {len(df)} rows\n")

        if api.budget_remaining() < 2:
            break

    sys.stdout.write(f"\r{' ' * 80}\r")

    if not all_frames:
        print("   ❌ No data downloaded!")
        return pd.DataFrame()

    print(f"\n   📊 Downloaded {downloaded} metric/coin combos")
    print(f"   📊 Total API requests: {api.total_requests}")

    # Combine all data
    combined = pd.concat(all_frames, ignore_index=True)
    return combined


# ─── Phase 3: Save & Summary ────────────────────────────────────────────────

def save_results(df: pd.DataFrame):
    """Save downloaded data as parquet."""
    if df.empty:
        print("   ⚠️  Nothing to save")
        return

    out_path = DATA_DIR / "cryptoquant_raw.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n   💾 Saved {len(df):,} rows → {out_path}")

    # Summary
    print(f"\n📋 Data Summary:")
    print(f"   Coins: {df['symbol'].nunique()}")
    print(f"   Metrics: {df['metric_name'].nunique()}")
    if "timestamp" in df.columns:
        print(f"   Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    print(f"\n   Metrics breakdown:")
    for metric, grp in df.groupby("metric_name"):
        n_coins = grp["symbol"].nunique()
        n_rows = len(grp)
        print(f"     {metric}: {n_coins} coins, {n_rows} rows")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download CryptoQuant on-chain data")
    parser.add_argument("--api-key", type=str, default=os.environ.get("CRYPTOQUANT_API_KEY", ""),
                        help="CryptoQuant API key (or set CRYPTOQUANT_API_KEY env)")
    parser.add_argument("--probe-only", action="store_true",
                        help="Only discover endpoints, don't download")
    parser.add_argument("--coin", type=str, default=None,
                        help="Download for a specific coin only (e.g., btc)")
    parser.add_argument("--limit", type=int, default=10,
                        help="Number of data points to fetch per request (default: 10, max for free: ~7)")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear endpoint discovery cache and re-probe")
    parser.add_argument("--budget", type=int, default=50,
                        help="Daily request budget (free=50, pro=unlimited)")
    parser.add_argument("--no-verify-ssl", action="store_true",
                        help="Disable SSL verification (for corporate proxies)")
    args = parser.parse_args()

    if not args.api_key:
        print("❌ No API key provided!")
        print("   Set CRYPTOQUANT_API_KEY env or pass --api-key")
        print("   Get free key at: https://cryptoquant.com/pricing")
        sys.exit(1)

    api = CryptoQuantAPI(args.api_key, verify_ssl=not args.no_verify_ssl)
    api.daily_budget = args.budget

    print("=" * 60)
    print(f"  CryptoQuant Data Download")
    print(f"  Budget: {api.daily_budget} req/day")
    print(f"  Limit:  {args.limit} data points")
    print("=" * 60)

    # Clear cache if requested
    if args.clear_cache:
        cache = DATA_DIR / "_endpoints.json"
        if cache.exists():
            cache.unlink()
            print("   🗑  Cleared endpoint cache")

    # Phase 1: Discover endpoints
    working = probe_endpoints(api)

    if args.probe_only:
        print(f"\n✅ Probe complete. {len(working)} working endpoints found.")
        print(f"   Requests used: {api.total_requests}/{api.daily_budget}")
        return

    if not working:
        print("\n❌ No working endpoints found! Check API key and tier.")
        sys.exit(1)

    # Phase 2: Download
    coins = [args.coin] if args.coin else COINS
    df = download_all(api, working, coins=coins, limit=args.limit)

    # Phase 3: Save
    save_results(df)

    print(f"\n✅ Done! Requests used: {api.total_requests}/{api.daily_budget}")
    print(f"\n📌 Next step: run IC analysis:")
    print(f"   python src/data/analyze_cryptoquant.py")


if __name__ == "__main__":
    main()
