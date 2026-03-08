#!/usr/bin/env python3
"""
Fetch historical crypto news and compute per-coin sentiment features.

Data sources:
  1. CryptoCompare News API (free, no key required for basic access)
     - Returns news with title, body, categories, source, published_at
     - 50 items per page, paginate with lTs parameter
  2. VADER sentiment analysis on news titles (lightweight, fast)

Output: data/sentiment/crypto_news.parquet
  Columns: timestamp (hourly), symbol, news_count_1h, news_count_24h,
           news_count_7d, sentiment_avg_24h, sentiment_avg_7d,
           sentiment_momentum, abnormal_news_zscore, market_news_count_24h,
           market_sentiment_avg_24h

Usage:
  python fetch_crypto_news.py                       # fetch all history
  python fetch_crypto_news.py --days 730            # last 2 years
  python fetch_crypto_news.py --api-key YOUR_KEY    # CryptoPanic (better data)
  python fetch_crypto_news.py --resume              # resume from last fetch
"""

import os
import sys
import json
import time
import argparse
import warnings
from datetime import datetime, timedelta, timezone
from collections import defaultdict

import pandas as pd
import numpy as np
import requests

warnings.filterwarnings("ignore")

# ─── Config ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "sentiment")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "crypto_news.parquet")
RAW_NEWS_PATH = os.path.join(OUTPUT_DIR, "raw_news.parquet")

# Map our trading symbols to CryptoCompare/CryptoPanic tickers
SYMBOLS = [
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT", "LINK",
    "MATIC", "UNI", "ATOM", "LTC", "FIL", "APT", "ARB", "OP", "NEAR", "AAVE",
    "INJ", "FTM", "ALGO", "SAND", "MANA", "AXS", "THETA", "RUNE", "EGLD", "XTZ",
    "FLOW", "CHZ", "CRV", "LDO", "SNX", "COMP", "YFI", "SUSHI", "ENJ", "BAT",
    "ZIL", "ONE", "IOTA", "ICX", "ENS", "IMX", "GALA", "MKR", "GRT", "ETC",
]

# Additional aliases for matching (some news sources use different names)
ALIASES = {
    "MATIC": ["MATIC", "POLYGON", "POL"],
    "FTM": ["FTM", "FANTOM"],
    "DOGE": ["DOGE", "DOGECOIN"],
    "SOL": ["SOL", "SOLANA"],
    "AVAX": ["AVAX", "AVALANCHE"],
    "DOT": ["DOT", "POLKADOT"],
    "LINK": ["LINK", "CHAINLINK"],
    "UNI": ["UNI", "UNISWAP"],
    "ATOM": ["ATOM", "COSMOS"],
    "NEAR": ["NEAR", "NEAR PROTOCOL"],
    "AAVE": ["AAVE"],
    "ARB": ["ARB", "ARBITRUM"],
    "OP": ["OP", "OPTIMISM"],
    "ADA": ["ADA", "CARDANO"],
    "BNB": ["BNB", "BINANCE"],
    "XRP": ["XRP", "RIPPLE"],
    "LTC": ["LTC", "LITECOIN"],
    "SAND": ["SAND", "SANDBOX"],
    "MANA": ["MANA", "DECENTRALAND"],
    "AXS": ["AXS", "AXIE"],
    "THETA": ["THETA"],
    "RUNE": ["RUNE", "THORCHAIN"],
    "ENS": ["ENS", "ETHEREUM NAME"],
    "IMX": ["IMX", "IMMUTABLE"],
    "GALA": ["GALA"],
    "MKR": ["MKR", "MAKER"],
    "GRT": ["GRT", "GRAPH"],
    "LDO": ["LDO", "LIDO"],
    "CRV": ["CRV", "CURVE"],
    "SNX": ["SNX", "SYNTHETIX"],
    "COMP": ["COMP", "COMPOUND"],
    "YFI": ["YFI", "YEARN"],
    "SUSHI": ["SUSHI", "SUSHISWAP"],
    "ETC": ["ETC", "ETHEREUM CLASSIC"],
    "ALGO": ["ALGO", "ALGORAND"],
    "INJ": ["INJ", "INJECTIVE"],
    "FIL": ["FIL", "FILECOIN"],
    "APT": ["APT", "APTOS"],
    "CHZ": ["CHZ", "CHILIZ"],
    "EGLD": ["EGLD", "MULTIVERSX", "ELROND"],
}

# Rate limit settings
CRYPTOCOMPARE_DELAY = 0.25  # seconds between requests (free tier: ~50/min)
CRYPTOPANIC_DELAY = 1.0     # CryptoPanic free tier is more restrictive


def _save_checkpoint(all_news):
    """Save intermediate results during long fetches."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    checkpoint_df = pd.DataFrame(all_news)
    checkpoint_path = RAW_NEWS_PATH + ".checkpoint"
    checkpoint_df.to_parquet(checkpoint_path, index=False)
    print(f"   💾 Checkpoint: {len(all_news):,} items → {checkpoint_path}")


# ─── VADER Sentiment ──────────────────────────────────────────────
def get_vader_analyzer():
    """Lazy-load VADER sentiment analyzer."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except ImportError:
        print("⚠️  vaderSentiment not installed. Installing...")
        os.system(f"{sys.executable} -m pip install vaderSentiment")
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()


def analyze_sentiment(text, analyzer):
    """Get VADER compound sentiment score [-1, +1]."""
    if not text or not isinstance(text, str):
        return 0.0
    scores = analyzer.polarity_scores(text)
    return scores["compound"]


# ─── CryptoCompare Fetcher ─────────────────────────────────────────
def fetch_cryptocompare_news(days=730, resume_from_ts=None, api_key=None):
    """
    Fetch historical crypto news from CryptoCompare.
    Free API, no key needed (but key increases limits).
    
    Rate limits (free, no key): 3000/hour, 7500/day
    Rate limits (free key): 100k/month
    
    Returns list of news dicts.
    """
    base_url = "https://min-api.cryptocompare.com/data/v2/news/"
    
    all_news = []
    cutoff_ts = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
    
    # Start from most recent or resume point
    lTs = resume_from_ts or int(datetime.now(timezone.utc).timestamp())
    
    page = 0
    oldest_seen = lTs
    consecutive_empty = 0
    
    print(f"📡 Fetching CryptoCompare news (target: {days} days back)...")
    print(f"   Cutoff: {datetime.fromtimestamp(cutoff_ts, tz=timezone.utc).strftime('%Y-%m-%d')}")
    
    while oldest_seen > cutoff_ts:
        try:
            params = {"lang": "EN", "lTs": lTs, "sortOrder": "latest"}
            if api_key:
                params["api_key"] = api_key
            
            resp = requests.get(base_url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            # Check rate limit
            rate_limit = data.get("RateLimit", {})
            calls_made = rate_limit.get("calls_made", {})
            max_calls = rate_limit.get("max_calls", {})
            hour_used = calls_made.get("hour", 0)
            hour_max = max_calls.get("hour", 3000)
            
            if hour_used >= hour_max - 10:
                wait_mins = 5
                print(f"\n   ⏳ Rate limit approaching ({hour_used}/{hour_max}). "
                      f"Waiting {wait_mins} min...")
                time.sleep(wait_mins * 60)
                continue
            
            # Type 99 = rate limited / error
            if data.get("Type") == 99 or not data.get("Data"):
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    print(f"\n   ⏳ Rate limited (Type={data.get('Type')}). Waiting 5 min...")
                    time.sleep(300)
                    consecutive_empty = 0
                    continue
                time.sleep(2)
                continue
            
            news_items = data["Data"]
            if not isinstance(news_items, list) or not news_items:
                consecutive_empty += 1
                if consecutive_empty >= 5:
                    print(f"   ⚠️  No more data available, stopping.")
                    break
                time.sleep(1)
                continue
            
            consecutive_empty = 0
            
            for item in news_items:
                all_news.append({
                    "id": item.get("id"),
                    "title": item.get("title", ""),
                    "body": item.get("body", "")[:500],  # truncate body
                    "categories": item.get("categories", ""),
                    "source": item.get("source_info", {}).get("name", ""),
                    "published_on": item.get("published_on", 0),
                    "url": item.get("url", ""),
                    "tags": item.get("tags", ""),
                })
            
            oldest_ts = min(item["published_on"] for item in news_items)
            newest_ts = max(item["published_on"] for item in news_items)
            lTs = oldest_ts  # paginate backward
            oldest_seen = oldest_ts
            page += 1
            
            if page % 50 == 0:
                oldest_date = datetime.fromtimestamp(oldest_ts, tz=timezone.utc)
                print(f"   Page {page}: {len(all_news):,} news, "
                      f"oldest: {oldest_date.strftime('%Y-%m-%d')}, "
                      f"rate: {hour_used}/{hour_max}/hr")
            
            # Checkpoint save every 500 pages
            if page % 500 == 0 and all_news:
                _save_checkpoint(all_news)
            
            time.sleep(CRYPTOCOMPARE_DELAY)
            
        except requests.exceptions.RequestException as e:
            print(f"   ⚠️  Request error: {e}. Retrying in 5s...")
            time.sleep(5)
            continue
        except Exception as e:
            print(f"   ❌ Error: {e}")
            break
    
    print(f"   ✅ Fetched {len(all_news):,} news items across {page} pages")
    return all_news


# ─── CryptoPanic Fetcher ──────────────────────────────────────────
def fetch_cryptopanic_news(api_key, days=730):
    """
    Fetch from CryptoPanic API (better sentiment data with votes).
    Requires free API key from https://cryptopanic.com/developers/api/
    
    Returns news with: title, currencies, votes (positive/negative/important), published_at
    """
    base_url = "https://cryptopanic.com/api/v1/posts/"
    
    all_news = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    
    print(f"📡 Fetching CryptoPanic news (target: {days} days back)...")
    print(f"   Cutoff: {cutoff.strftime('%Y-%m-%d')}")
    
    next_url = base_url
    page = 0
    
    while next_url:
        try:
            params = {"auth_token": api_key, "public": "true"}
            if "?" in next_url and "page" in next_url:
                # next_url already has params
                resp = requests.get(next_url, params={"auth_token": api_key}, timeout=30)
            else:
                resp = requests.get(next_url, params=params, timeout=30)
            
            resp.raise_for_status()
            data = resp.json()
            
            results = data.get("results", [])
            if not results:
                break
            
            for item in results:
                published = item.get("published_at", "")
                if published:
                    pub_dt = pd.to_datetime(published)
                    if pub_dt.tzinfo is None:
                        pub_dt = pub_dt.tz_localize("UTC")
                    if pub_dt < cutoff:
                        print(f"   Reached cutoff at page {page}")
                        return all_news
                
                currencies = []
                for c in item.get("currencies", []) or []:
                    currencies.append(c.get("code", ""))
                
                votes = item.get("votes", {})
                
                all_news.append({
                    "id": item.get("id"),
                    "title": item.get("title", ""),
                    "body": "",
                    "categories": ",".join(currencies),
                    "source": item.get("source", {}).get("title", ""),
                    "published_on": int(pub_dt.timestamp()) if published else 0,
                    "url": item.get("url", ""),
                    "tags": ",".join(currencies),
                    "votes_positive": votes.get("positive", 0),
                    "votes_negative": votes.get("negative", 0),
                    "votes_important": votes.get("important", 0),
                    "votes_lol": votes.get("lol", 0),
                    "votes_toxic": votes.get("toxic", 0),
                    "votes_liked": votes.get("liked", 0),
                    "votes_disliked": votes.get("disliked", 0),
                })
            
            next_url = data.get("next")
            page += 1
            
            if page % 50 == 0:
                print(f"   Page {page}: {len(all_news):,} news")
            
            time.sleep(CRYPTOPANIC_DELAY)
            
        except requests.exceptions.RequestException as e:
            print(f"   ⚠️  Request error: {e}. Retrying in 5s...")
            time.sleep(5)
            continue
        except Exception as e:
            print(f"   ❌ Error: {e}")
            break
    
    print(f"   ✅ Fetched {len(all_news):,} news items across {page} pages")
    return all_news


# ─── News → Per-Coin Mapping ──────────────────────────────────────
def map_news_to_coins(news_items, symbols=SYMBOLS):
    """
    Map each news item to relevant coins based on categories/tags/title.
    Returns list of (news_item, [symbols]) tuples.
    """
    # Build lookup: alias → symbol
    alias_to_sym = {}
    for sym in symbols:
        alias_to_sym[sym.upper()] = sym
        if sym in ALIASES:
            for alias in ALIASES[sym]:
                alias_to_sym[alias.upper()] = sym
        # Also add full trading pair name
        alias_to_sym[f"{sym}/USDT"] = sym
    
    mapped = []
    for item in news_items:
        coins = set()
        
        # 1. Check categories field (CryptoCompare uses "|" separator)
        cats = str(item.get("categories", "")).upper()
        for alias, sym in alias_to_sym.items():
            if alias in cats.split("|"):
                coins.add(sym)
        
        # 2. Check tags field
        tags = str(item.get("tags", "")).upper()
        for alias, sym in alias_to_sym.items():
            if alias in tags.split(","):
                coins.add(sym)
        
        # 3. Check title for coin mentions (word boundary matching)
        title_upper = str(item.get("title", "")).upper()
        for alias, sym in alias_to_sym.items():
            if len(alias) >= 3:  # skip very short aliases to avoid false positives
                # Check for word boundary-ish match
                for i in range(len(title_upper)):
                    idx = title_upper.find(alias, i)
                    if idx == -1:
                        break
                    # Check boundaries
                    before_ok = (idx == 0 or not title_upper[idx-1].isalnum())
                    after_ok = (idx + len(alias) >= len(title_upper) or 
                               not title_upper[idx + len(alias)].isalnum())
                    if before_ok and after_ok:
                        coins.add(sym)
                        break
        
        if coins:
            mapped.append((item, list(coins)))
    
    total_mappings = sum(len(c) for _, c in mapped)
    print(f"   📋 Mapped {len(mapped):,}/{len(news_items):,} news → "
          f"{total_mappings:,} coin-news pairs")
    return mapped


# ─── Build Sentiment Features ────────────────────────────────────
def build_news_features(mapped_news, symbols=SYMBOLS):
    """
    Build per-coin, per-hour sentiment features from mapped news.
    
    Features per coin per hour:
      - news_count_1h: raw count in this hour
      - news_count_24h: rolling 24h count
      - news_count_7d: rolling 7d count
      - news_sentiment_1h: VADER sentiment of news in this hour (mean)
      - news_sentiment_24h: rolling 24h mean sentiment
      - news_sentiment_7d: rolling 7d mean sentiment
      - news_sentiment_momentum: 24h sentiment - 7d sentiment
      - news_volume_zscore: z-score of 24h count vs 30d rolling mean/std
    
    Market-level features:
      - market_news_count_24h: total crypto news in 24h
      - market_news_sentiment_24h: market-wide sentiment
    """
    analyzer = get_vader_analyzer()
    
    print("🔍 Computing sentiment scores...")
    
    # Build per-coin hourly raw data
    # coin → list of (hour_ts, sentiment_score)
    coin_events = defaultdict(list)
    market_events = []
    
    for item, coins in mapped_news:
        ts = item.get("published_on", 0)
        if ts == 0:
            continue
        
        # Round to hour
        hour_ts = pd.Timestamp(ts, unit="s", tz="UTC").floor("h")
        
        # Compute sentiment
        title = item.get("title", "")
        sentiment = analyze_sentiment(title, analyzer)
        
        # If CryptoPanic votes available, boost with vote sentiment
        if item.get("votes_positive") or item.get("votes_negative"):
            pos = item.get("votes_positive", 0)
            neg = item.get("votes_negative", 0)
            total_votes = pos + neg
            if total_votes > 0:
                vote_sentiment = (pos - neg) / total_votes  # [-1, +1]
                # Blend: 60% VADER + 40% votes
                sentiment = 0.6 * sentiment + 0.4 * vote_sentiment
        
        for coin in coins:
            coin_events[coin].append((hour_ts, sentiment))
        market_events.append((hour_ts, sentiment))
    
    print(f"   Total events: {sum(len(v) for v in coin_events.values()):,} coin-events, "
          f"{len(market_events):,} market events")
    
    # Determine time range
    all_hours = set()
    for events in coin_events.values():
        for h, _ in events:
            all_hours.add(h)
    for h, _ in market_events:
        all_hours.add(h)
    
    if not all_hours:
        print("   ❌ No events to process!")
        return pd.DataFrame()
    
    min_hour = min(all_hours)
    max_hour = max(all_hours)
    hourly_range = pd.date_range(min_hour, max_hour, freq="h", tz="UTC")
    
    print(f"   Time range: {min_hour} → {max_hour} ({len(hourly_range):,} hours)")
    
    # ─── Per-coin features ────────────────────────────────────────
    print("📊 Building per-coin features...")
    
    all_rows = []
    
    for sym in symbols:
        events = coin_events.get(sym, [])
        
        # Build hourly aggregates
        hourly_count = defaultdict(int)
        hourly_sentiment_sum = defaultdict(float)
        
        for h, s in events:
            hourly_count[h] += 1
            hourly_sentiment_sum[h] += s
        
        # Create series
        counts = pd.Series(
            [hourly_count.get(h, 0) for h in hourly_range],
            index=hourly_range, dtype=float, name="count"
        )
        sentiments = pd.Series(
            [hourly_sentiment_sum.get(h, 0) / max(hourly_count.get(h, 0), 1) 
             for h in hourly_range],
            index=hourly_range, dtype=float, name="sentiment"
        )
        
        # Rolling features
        news_count_24h = counts.rolling(24, min_periods=1).sum()
        news_count_7d = counts.rolling(168, min_periods=1).sum()
        sentiment_24h = sentiments.rolling(24, min_periods=1).mean()
        sentiment_7d = sentiments.rolling(168, min_periods=1).mean()
        
        # Momentum: short-term vs long-term sentiment
        sentiment_momentum = sentiment_24h - sentiment_7d
        
        # Abnormal news volume: z-score of 24h count vs 30d rolling stats
        count_30d_mean = news_count_24h.rolling(720, min_periods=24).mean()
        count_30d_std = news_count_24h.rolling(720, min_periods=24).std().clip(lower=0.1)
        news_volume_zscore = (news_count_24h - count_30d_mean) / count_30d_std
        
        for i, h in enumerate(hourly_range):
            all_rows.append({
                "timestamp": h,
                "symbol": f"{sym}/USDT",
                "news_count_1h": counts.iloc[i],
                "news_count_24h": news_count_24h.iloc[i],
                "news_count_7d": news_count_7d.iloc[i],
                "news_sentiment_1h": sentiments.iloc[i],
                "news_sentiment_24h": sentiment_24h.iloc[i],
                "news_sentiment_7d": sentiment_7d.iloc[i],
                "news_sentiment_momentum": sentiment_momentum.iloc[i],
                "news_volume_zscore": news_volume_zscore.iloc[i],
            })
    
    df = pd.DataFrame(all_rows)
    
    # ─── Market-level features ────────────────────────────────────
    print("📊 Building market-level features...")
    
    market_count = defaultdict(int)
    market_sent_sum = defaultdict(float)
    
    for h, s in market_events:
        market_count[h] += 1
        market_sent_sum[h] += s
    
    m_counts = pd.Series(
        [market_count.get(h, 0) for h in hourly_range],
        index=hourly_range, dtype=float
    )
    m_sents = pd.Series(
        [market_sent_sum.get(h, 0) / max(market_count.get(h, 0), 1) for h in hourly_range],
        index=hourly_range, dtype=float
    )
    
    market_news_24h = m_counts.rolling(24, min_periods=1).sum()
    market_sentiment_24h = m_sents.rolling(24, min_periods=1).mean()
    
    market_df = pd.DataFrame({
        "timestamp": hourly_range,
        "market_news_count_24h": market_news_24h.values,
        "market_news_sentiment_24h": market_sentiment_24h.values,
    })
    
    df = df.merge(market_df, on="timestamp", how="left")
    
    # Fill NaN
    for col in df.columns:
        if col not in ("timestamp", "symbol"):
            df[col] = df[col].fillna(0)
    
    print(f"   ✅ Features shape: {df.shape}")
    print(f"   Symbols with news: {df[df['news_count_7d'] > 0]['symbol'].nunique()}")
    print(f"   Coverage: {(df['news_count_7d'] > 0).mean():.1%} rows have some news")
    
    return df


# ─── Summary Stats ────────────────────────────────────────────────
def print_news_summary(df):
    """Print summary statistics of news features."""
    print("\n" + "=" * 60)
    print("📊 NEWS SENTIMENT FEATURES SUMMARY")
    print("=" * 60)
    
    print(f"\nTime range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"Total rows: {len(df):,}")
    print(f"Symbols: {df['symbol'].nunique()}")
    
    # Top coins by news coverage
    coin_coverage = (
        df.groupby("symbol")["news_count_7d"]
        .apply(lambda x: (x > 0).mean())
        .sort_values(ascending=False)
    )
    print(f"\nTop 10 coins by news coverage:")
    for sym, cov in coin_coverage.head(10).items():
        total_news = df[df["symbol"] == sym]["news_count_1h"].sum()
        avg_sent = df[df["symbol"] == sym]["news_sentiment_24h"].mean()
        print(f"  {sym:15s} coverage={cov:.1%}  total_news={total_news:.0f}  "
              f"avg_sentiment={avg_sent:+.3f}")
    
    # Feature distributions
    feat_cols = [c for c in df.columns if c.startswith("news_") or c.startswith("market_")]
    print(f"\nFeature distributions:")
    for col in feat_cols:
        vals = df[col]
        print(f"  {col:30s}  mean={vals.mean():+8.4f}  std={vals.std():8.4f}  "
              f"min={vals.min():+8.4f}  max={vals.max():+8.4f}")


# ─── Main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fetch crypto news and build sentiment features")
    parser.add_argument("--days", type=int, default=730, help="Days of history to fetch (default: 730)")
    parser.add_argument("--api-key", type=str, default=None, help="CryptoPanic API key (optional, better data)")
    parser.add_argument("--cc-api-key", type=str, default=None, help="CryptoCompare API key (optional, more calls)")
    parser.add_argument("--resume", action="store_true", help="Resume from last fetch point")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip fetching, only rebuild features from raw data")
    parser.add_argument("--source", choices=["cryptocompare", "cryptopanic", "both"], 
                       default="cryptocompare", help="News source (default: cryptocompare)")
    args = parser.parse_args()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ─── Fetch raw news ──────────────────────────────────────────
    if not args.skip_fetch:
        all_news = []
        
        # CryptoCompare (always available, free)
        if args.source in ("cryptocompare", "both"):
            resume_ts = None
            if args.resume and os.path.exists(RAW_NEWS_PATH):
                existing = pd.read_parquet(RAW_NEWS_PATH)
                resume_ts = int(existing["published_on"].min())
                print(f"📂 Resuming from {datetime.fromtimestamp(resume_ts, tz=timezone.utc).strftime('%Y-%m-%d')}")
            
            cc_news = fetch_cryptocompare_news(
                days=args.days, resume_from_ts=resume_ts,
                api_key=args.cc_api_key
            )
            all_news.extend(cc_news)
        
        # CryptoPanic (better sentiment, needs API key)
        if args.source in ("cryptopanic", "both") and args.api_key:
            cp_news = fetch_cryptopanic_news(api_key=args.api_key, days=args.days)
            all_news.extend(cp_news)
        elif args.source == "cryptopanic" and not args.api_key:
            print("❌ CryptoPanic requires --api-key. Get free key at https://cryptopanic.com/developers/api/")
            sys.exit(1)
        
        # Deduplicate by title + published_on
        seen = set()
        unique_news = []
        for item in all_news:
            key = (item.get("title", ""), item.get("published_on", 0))
            if key not in seen:
                seen.add(key)
                unique_news.append(item)
        
        print(f"\n📦 Total unique news: {len(unique_news):,} (deduped from {len(all_news):,})")
        
        # Save raw news
        raw_df = pd.DataFrame(unique_news)
        
        # Merge with existing if resuming
        if args.resume and os.path.exists(RAW_NEWS_PATH):
            existing = pd.read_parquet(RAW_NEWS_PATH)
            raw_df = pd.concat([existing, raw_df]).drop_duplicates(
                subset=["title", "published_on"]).reset_index(drop=True)
            print(f"   Merged with existing: {len(raw_df):,} total")
        
        raw_df.to_parquet(RAW_NEWS_PATH, index=False)
        print(f"   💾 Saved raw news → {RAW_NEWS_PATH}")
        
        news_items = raw_df.to_dict("records")
    else:
        # Load from existing raw data
        if not os.path.exists(RAW_NEWS_PATH):
            print(f"❌ No raw news at {RAW_NEWS_PATH}. Run without --skip-fetch first.")
            sys.exit(1)
        raw_df = pd.read_parquet(RAW_NEWS_PATH)
        news_items = raw_df.to_dict("records")
        print(f"📂 Loaded {len(news_items):,} raw news items")
    
    # ─── Map to coins ────────────────────────────────────────────
    print("\n🔗 Mapping news to coins...")
    mapped = map_news_to_coins(news_items)
    
    # ─── Build features ──────────────────────────────────────────
    print("\n🛠️  Building features...")
    features_df = build_news_features(mapped)
    
    if features_df.empty:
        print("❌ No features generated!")
        sys.exit(1)
    
    # ─── Save ─────────────────────────────────────────────────────
    features_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\n💾 Saved features → {OUTPUT_PATH}")
    
    # ─── Summary ──────────────────────────────────────────────────
    print_news_summary(features_df)
    
    print(f"\n✅ Done! Next steps:")
    print(f"   1. Upload {OUTPUT_PATH} to cluster")
    print(f"   2. Retrain models with news features")
    print(f"   3. Compare with current champion (Sharpe 8.04)")


if __name__ == "__main__":
    main()
