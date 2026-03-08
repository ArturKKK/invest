#!/usr/bin/env python3
"""
Fetch historical crypto + political/macro news and compute sentiment features.

Data sources:
  1. CryptoCompare News API (free, no key required for basic access)
     - Crypto-specific news with title, body, categories, source
     - 50 items per page, paginate with lTs parameter
     - Rate limit: 3000/hour (free), 100k/month (free API key)
  2. GDELT Project (free, no key, unlimited)
     - Global news/events database, updated every 15 minutes
     - Political/macro events (Trump, regulators, sanctions, etc.)
     - Full-text search by keywords
  3. Sentiment scorers (configurable via --scorer):
     - vader: VADER rule-based (fast, CPU, ~60% accuracy on finance)
     - finbert: ProsusAI/finbert (GPU, ~87% accuracy on finance)
     - cryptobert: ElKulako/cryptobert (GPU, trained on 3.2M crypto tweets)

Output: data/sentiment/crypto_news.parquet
  Per-coin + market-level + political sentiment features (hourly)

Usage:
  python fetch_crypto_news.py                               # fetch all (crypto + political)
  python fetch_crypto_news.py --days 730                    # last 2 years
  python fetch_crypto_news.py --source crypto               # crypto only
  python fetch_crypto_news.py --source political            # political only  
  python fetch_crypto_news.py --source all                  # both (default)
  python fetch_crypto_news.py --resume                      # resume from last fetch
  python fetch_crypto_news.py --skip-fetch                  # rebuild features only
  python fetch_crypto_news.py --cc-api-key KEY              # CryptoCompare key (more calls)
  python fetch_crypto_news.py --scorer finbert              # use FinBERT (GPU)
  python fetch_crypto_news.py --scorer cryptobert           # use CryptoBERT (GPU, best for crypto)
  python fetch_crypto_news.py --scorer vader                # use VADER (CPU, default)
  python fetch_crypto_news.py --skip-fetch --scorer finbert # re-score existing data with FinBERT
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

# Political/macro keywords that move crypto markets
POLITICAL_KEYWORDS = [
    # US Politics + Crypto
    "Trump crypto", "Trump bitcoin", "Trump tariff", "Trump sanctions",
    "Trump executive order", "Biden crypto", "Biden regulation",
    "SEC crypto", "SEC bitcoin", "SEC ethereum", "Gary Gensler",
    "crypto regulation", "crypto ban", "crypto tax",
    "stablecoin regulation", "CBDC", "digital dollar",
    # Macro events
    "Federal Reserve rate", "Fed rate decision", "interest rate hike",
    "interest rate cut", "inflation data", "CPI report",
    "US debt ceiling", "government shutdown",
    "bank failure", "banking crisis", "Silicon Valley Bank",
    # Geopolitics affecting crypto
    "Russia sanctions", "China crypto ban", "China bitcoin",
    "EU crypto regulation", "MiCA regulation",
    "Tether USDT", "stablecoin depeg",
    # Major crypto events via politics
    "Bitcoin ETF", "Ethereum ETF", "spot ETF approval",
    "crypto strategic reserve", "Binance SEC", "Coinbase SEC",
    "crypto executive order", "digital asset framework",
]

# Simplified keyword groups for GDELT queries (max ~250 chars per query)
GDELT_QUERY_GROUPS = [
    '(Trump AND (crypto OR bitcoin OR tariff))',
    '(SEC AND (crypto OR bitcoin OR ethereum))',
    '("Federal Reserve" OR "interest rate" OR "CPI" OR inflation)',
    '(crypto AND (regulation OR ban OR tax OR ETF))',
    '((Russia OR China) AND (sanctions OR crypto OR bitcoin))',
    '("Bitcoin ETF" OR "Ethereum ETF" OR "stablecoin" OR CBDC)',
]

RAW_POLITICAL_PATH = os.path.join(OUTPUT_DIR, "raw_political_news.parquet")


def _save_checkpoint(all_news):
    """Save intermediate results during long fetches."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    checkpoint_df = pd.DataFrame(all_news)
    checkpoint_path = RAW_NEWS_PATH + ".checkpoint"
    checkpoint_df.to_parquet(checkpoint_path, index=False)
    print(f"   💾 Checkpoint: {len(all_news):,} items → {checkpoint_path}")


# ─── Sentiment Scorers ─────────────────────────────────────────────

# --- VADER (rule-based, fast, CPU) ---
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


# --- FinBERT / CryptoBERT (transformer, GPU-accelerated) ---
def _detect_device():
    """Detect best available device: cuda > mps > cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            dev = "cuda"
            name = torch.cuda.get_device_name(0)
            print(f"   🚀 Using CUDA: {name}")
            return dev
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("   🍎 Using MPS (Apple Silicon)")
            return "mps"
    except ImportError:
        pass
    print("   💻 Using CPU (slow for transformers!)")
    return "cpu"


def get_transformer_scorer(model_name="finbert"):
    """
    Load a transformer-based sentiment model.
    
    Supported models:
      - 'finbert':     ProsusAI/finbert  (financial news, ~87% accuracy)
      - 'cryptobert':  ElKulako/cryptobert (crypto tweets, 3.2M training samples)
    
    Returns (pipeline, label_map) tuple.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline as hf_pipeline
        import torch
    except ImportError:
        print("❌ transformers / torch not installed. Install with:")
        print(f"   {sys.executable} -m pip install transformers torch")
        sys.exit(1)
    
    MODEL_MAP = {
        "finbert": {
            "hf_name": "ProsusAI/finbert",
            # FinBERT labels: positive, negative, neutral → map to [-1, +1]
            "label_map": {"positive": 1.0, "negative": -1.0, "neutral": 0.0},
        },
        "cryptobert": {
            "hf_name": "ElKulako/cryptobert",
            # CryptoBERT labels: Bullish, Bearish, Neutral → map to [-1, +1]
            "label_map": {"Bullish": 1.0, "Bearish": -1.0, "Neutral": 0.0},
        },
    }
    
    if model_name not in MODEL_MAP:
        print(f"❌ Unknown model '{model_name}'. Choices: {list(MODEL_MAP.keys())}")
        sys.exit(1)
    
    cfg = MODEL_MAP[model_name]
    hf_name = cfg["hf_name"]
    label_map = cfg["label_map"]
    
    device = _detect_device()
    device_id = 0 if device == "cuda" else (-1 if device == "cpu" else device)
    
    print(f"🤖 Loading {model_name} ({hf_name})...")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForSequenceClassification.from_pretrained(hf_name)
    
    # Use float16 on GPU for speed
    import torch
    if device == "cuda":
        model = model.half()
    
    pipe = hf_pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device_id,
        truncation=True,
        max_length=512,
        batch_size=64,  # adjust for GPU memory
    )
    
    print(f"   ✅ Model loaded on {device}")
    return pipe, label_map


def score_batch_transformer(texts, pipe, label_map):
    """
    Score a batch of texts using a transformer pipeline.
    Returns list of float scores in [-1, +1].
    
    Uses weighted score: sum(label_value * probability) for each text,
    so a "60% positive, 40% neutral" headline gets +0.6, not +1.0.
    """
    if not texts:
        return []
    
    # Clean texts
    clean = []
    for t in texts:
        if not t or not isinstance(t, str) or len(t.strip()) == 0:
            clean.append("neutral")  # placeholder
        else:
            clean.append(t[:512])  # truncate to model max
    
    # Score in batches with return_all_scores to get probabilities
    try:
        results = pipe(clean, return_all_scores=True)
    except Exception as e:
        print(f"   ⚠️  Batch scoring error: {e}. Falling back to single-item scoring.")
        scores = []
        for t in clean:
            try:
                res = pipe(t, return_all_scores=True)
                weighted = sum(
                    label_map.get(r["label"], 0.0) * r["score"]
                    for r in res[0]
                )
                scores.append(np.clip(weighted, -1.0, 1.0))
            except Exception:
                scores.append(0.0)
        return scores
    
    # Compute weighted scores from all-class probabilities
    scores = []
    for res in results:
        weighted = sum(
            label_map.get(r["label"], 0.0) * r["score"]
            for r in res
        )
        scores.append(np.clip(weighted, -1.0, 1.0))
    
    return scores


def prescore_all_news(news_items, scorer="vader"):
    """
    Pre-compute sentiment scores for all news items.
    Adds 'sentiment_score' field to each item dict (in-place).
    
    For transformers: batch processing on GPU (fast).
    For VADER: sequential (still fast since rule-based).
    """
    n = len(news_items)
    print(f"\n🎯 Scoring {n:,} news items with '{scorer}'...")
    
    if scorer == "vader":
        analyzer = get_vader_analyzer()
        for i, item in enumerate(news_items):
            item["sentiment_score"] = analyze_sentiment(item.get("title", ""), analyzer)
            if (i + 1) % 50000 == 0:
                print(f"   Scored {i + 1:,}/{n:,}")
    else:
        # Transformer-based (finbert / cryptobert)
        pipe, label_map = get_transformer_scorer(scorer)
        
        titles = [item.get("title", "") for item in news_items]
        
        # Process in mega-batches to show progress
        MEGA_BATCH = 10000
        all_scores = []
        
        for start in range(0, n, MEGA_BATCH):
            end = min(start + MEGA_BATCH, n)
            batch_titles = titles[start:end]
            batch_scores = score_batch_transformer(batch_titles, pipe, label_map)
            all_scores.extend(batch_scores)
            print(f"   Scored {end:,}/{n:,}  "
                  f"(avg={np.mean(batch_scores):+.4f}, "
                  f"pos={sum(1 for s in batch_scores if s > 0.1)}, "
                  f"neg={sum(1 for s in batch_scores if s < -0.1)})")
        
        for i, score in enumerate(all_scores):
            news_items[i]["sentiment_score"] = score
        
        # Free GPU memory
        del pipe
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    
    # Stats
    scores = [item.get("sentiment_score", 0) for item in news_items]
    pos = sum(1 for s in scores if s > 0.1)
    neg = sum(1 for s in scores if s < -0.1)
    neu = n - pos - neg
    print(f"   ✅ Sentiment distribution: {pos:,} positive ({pos/n:.1%}), "
          f"{neg:,} negative ({neg/n:.1%}), {neu:,} neutral ({neu/n:.1%})")
    print(f"   Mean={np.mean(scores):+.4f}, Std={np.std(scores):.4f}")
    
    return news_items


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


# ─── GDELT Political/Macro News Fetcher ───────────────────────────
def fetch_gdelt_political_news(days=730):
    """
    Fetch political/macro news from GDELT DOC API (free, unlimited, no key).
    
    GDELT indexes global news in near-real-time.
    DOC API: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
    
    - Full-text search across global media
    - No rate limits (but be reasonable)
    - Returns: url, title, seendate, source, language, domain
    - Max 250 results per query, paginate by date ranges
    
    We fetch political/macro events that affect crypto markets.
    """
    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    
    all_news = []
    seen_urls = set()
    
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    end_date = datetime.now(timezone.utc)
    
    print(f"📡 Fetching GDELT political/macro news (target: {days} days back)...")
    print(f"   Cutoff: {cutoff.strftime('%Y-%m-%d')}")
    print(f"   Query groups: {len(GDELT_QUERY_GROUPS)}")
    
    for qi, query in enumerate(GDELT_QUERY_GROUPS):
        print(f"\n   [{qi+1}/{len(GDELT_QUERY_GROUPS)}] Query: {query[:60]}...")
        
        # GDELT DOC API supports date ranges, paginate in 30-day chunks
        chunk_end = end_date
        chunk_days = 30  # process in 30-day windows
        query_total = 0
        
        while chunk_end > cutoff:
            chunk_start = max(chunk_end - timedelta(days=chunk_days), cutoff)
            
            try:
                params = {
                    "query": query,
                    "mode": "artlist",
                    "maxrecords": "250",
                    "format": "json",
                    "startdatetime": chunk_start.strftime("%Y%m%d%H%M%S"),
                    "enddatetime": chunk_end.strftime("%Y%m%d%H%M%S"),
                    "sourcelang": "eng",  # English only
                }
                
                resp = requests.get(base_url, params=params, timeout=60)
                
                if resp.status_code == 429:
                    print(f"      ⏳ Rate limited, waiting 30s...")
                    time.sleep(30)
                    continue  # retry same chunk
                elif resp.status_code != 200:
                    print(f"      ⚠️  HTTP {resp.status_code} for {chunk_start.date()}→{chunk_end.date()}")
                    chunk_end = chunk_start
                    time.sleep(2)
                    continue
                
                try:
                    data = resp.json()
                except Exception:
                    chunk_end = chunk_start
                    time.sleep(1)
                    continue
                
                articles = data.get("articles", [])
                
                for art in articles:
                    url = art.get("url", "")
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)
                    
                    title = art.get("title", "")
                    seendate = art.get("seendate", "")
                    
                    # Parse GDELT date format: "20260305T143000Z"
                    try:
                        if seendate:
                            pub_ts = int(pd.to_datetime(seendate).timestamp())
                        else:
                            continue
                    except Exception:
                        continue
                    
                    all_news.append({
                        "id": f"gdelt_{hash(url) & 0xFFFFFFFF}",
                        "title": title,
                        "body": "",
                        "categories": "POLITICAL|MACRO",
                        "source": art.get("domain", ""),
                        "published_on": pub_ts,
                        "url": url,
                        "tags": "political,macro",
                        "news_type": "political",
                    })
                    query_total += 1
                
                chunk_end = chunk_start
                time.sleep(1)  # pause between chunks
                
            except requests.exceptions.RequestException as e:
                print(f"      ⚠️  Request error: {e}. Retrying in 10s...")
                time.sleep(10)
                continue
            except Exception as e:
                print(f"      ❌ Error: {e}")
                chunk_end = chunk_start
                continue
        
        print(f"      → {query_total:,} articles for this query")
        time.sleep(3)  # pause between query groups
    
    print(f"\n   ✅ Total GDELT political news: {len(all_news):,}")
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
    political_unmapped = []  # political news not matched to specific coins → market-level
    
    for item in news_items:
        coins = set()
        is_political = item.get("news_type") == "political" or "POLITICAL" in str(item.get("categories", "")).upper()
        
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
        elif is_political:
            # Political news not mapped to specific coin → affects entire market
            political_unmapped.append(item)
    
    total_mappings = sum(len(c) for _, c in mapped)
    print(f"   📋 Mapped {len(mapped):,}/{len(news_items):,} news → "
          f"{total_mappings:,} coin-news pairs")
    print(f"   🏛️  Political/macro (market-level): {len(political_unmapped):,} items")
    return mapped, political_unmapped


# ─── Build Sentiment Features ────────────────────────────────────
def build_news_features(mapped_news, political_unmapped=None, symbols=SYMBOLS):
    """
    Build per-coin, per-hour sentiment features from mapped news.
    
    Expects items to have pre-computed 'sentiment_score' field
    (from prescore_all_news). Falls back to VADER if missing.
    
    Features per coin per hour:
      - news_count_1h: raw count in this hour
      - news_count_24h: rolling 24h count
      - news_count_7d: rolling 7d count
      - news_sentiment_1h: sentiment of news in this hour (mean)
      - news_sentiment_24h: rolling 24h mean sentiment
      - news_sentiment_7d: rolling 7d mean sentiment
      - news_sentiment_momentum: 24h sentiment - 7d sentiment
      - news_volume_zscore: z-score of 24h count vs 30d rolling mean/std
    
    Market-level features:
      - market_news_count_24h: total crypto news in 24h
      - market_news_sentiment_24h: market-wide sentiment
    
    Political/macro features (same for all coins):
      - political_news_count_24h: political/macro news volume
      - political_sentiment_24h: political news sentiment
      - political_sentiment_7d: 7d rolling political sentiment
      - political_sentiment_shock: |24h - 7d| political sentiment (sudden shift)
      - political_news_volume_zscore: z-score of political news volume
    """
    # Fallback VADER analyzer only if some items lack pre-computed scores
    _vader = None
    def _get_score(item):
        nonlocal _vader
        if "sentiment_score" in item:
            return item["sentiment_score"]
        # Fallback to VADER
        if _vader is None:
            _vader = get_vader_analyzer()
        return analyze_sentiment(item.get("title", ""), _vader)
    
    print("🔍 Building sentiment features...")
    
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
        
        # Get pre-computed or fallback sentiment
        sentiment = _get_score(item)
        
        # If CryptoPanic votes available, blend with vote sentiment
        if item.get("votes_positive") or item.get("votes_negative"):
            pos = item.get("votes_positive", 0)
            neg = item.get("votes_negative", 0)
            total_votes = pos + neg
            if total_votes > 0:
                vote_sentiment = (pos - neg) / total_votes  # [-1, +1]
                # Blend: 60% model + 40% votes
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
    
    # ─── Political/macro features (market-level) ──────────────────
    if political_unmapped:
        print("🏛️  Building political/macro features...")
        
        pol_count = defaultdict(int)
        pol_sent_sum = defaultdict(float)
        
        for item in political_unmapped:
            ts = item.get("published_on", 0)
            if ts == 0:
                continue
            hour_ts = pd.Timestamp(ts, unit="s", tz="UTC").floor("h")
            if hour_ts in set(hourly_range):
                sentiment = _get_score(item)
                pol_count[hour_ts] += 1
                pol_sent_sum[hour_ts] += sentiment
        
        p_counts = pd.Series(
            [pol_count.get(h, 0) for h in hourly_range],
            index=hourly_range, dtype=float
        )
        p_sents = pd.Series(
            [pol_sent_sum.get(h, 0) / max(pol_count.get(h, 0), 1) for h in hourly_range],
            index=hourly_range, dtype=float
        )
        
        pol_news_24h = p_counts.rolling(24, min_periods=1).sum()
        pol_sentiment_24h = p_sents.rolling(24, min_periods=1).mean()
        pol_sentiment_7d = p_sents.rolling(168, min_periods=1).mean()
        pol_sentiment_shock = (pol_sentiment_24h - pol_sentiment_7d).abs()
        pol_30d_mean = pol_news_24h.rolling(720, min_periods=24).mean()
        pol_30d_std = pol_news_24h.rolling(720, min_periods=24).std().clip(lower=0.1)
        pol_volume_zscore = (pol_news_24h - pol_30d_mean) / pol_30d_std
        
        pol_df = pd.DataFrame({
            "timestamp": hourly_range,
            "political_news_count_24h": pol_news_24h.values,
            "political_sentiment_24h": pol_sentiment_24h.values,
            "political_sentiment_7d": pol_sentiment_7d.values,
            "political_sentiment_shock": pol_sentiment_shock.values,
            "political_news_volume_zscore": pol_volume_zscore.values,
        })
        
        df = df.merge(pol_df, on="timestamp", how="left")
        n_pol = (df.get("political_news_count_24h", pd.Series([0])) > 0).sum()
        print(f"      Political features: {n_pol:,} rows with political news")
    
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
    feat_cols = [c for c in df.columns 
                 if c.startswith("news_") or c.startswith("market_") or c.startswith("political_")]
    print(f"\nFeature distributions:")
    for col in feat_cols:
        vals = df[col]
        print(f"  {col:30s}  mean={vals.mean():+8.4f}  std={vals.std():8.4f}  "
              f"min={vals.min():+8.4f}  max={vals.max():+8.4f}")


# ─── Main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fetch crypto + political news and build sentiment features")
    parser.add_argument("--days", type=int, default=730, help="Days of history to fetch (default: 730)")
    parser.add_argument("--api-key", type=str, default=None, help="CryptoPanic API key (optional, better data)")
    parser.add_argument("--cc-api-key", type=str, default=None, help="CryptoCompare API key (optional, more calls)")
    parser.add_argument("--resume", action="store_true", help="Resume from last fetch point")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip fetching, only rebuild features from raw data")
    parser.add_argument("--source", choices=["crypto", "political", "all"], 
                       default="all", help="News source: crypto, political, or all (default)")
    parser.add_argument("--scorer", choices=["vader", "finbert", "cryptobert"],
                       default="vader",
                       help="Sentiment scorer: vader (fast/CPU), finbert (GPU/finance), "
                            "cryptobert (GPU/crypto, best) [default: vader]")
    args = parser.parse_args()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ─── Fetch raw news ──────────────────────────────────────────
    if not args.skip_fetch:
        all_news = []
        
        # CryptoCompare (crypto news)
        if args.source in ("crypto", "all"):
            resume_ts = None
            if args.resume and os.path.exists(RAW_NEWS_PATH):
                existing = pd.read_parquet(RAW_NEWS_PATH)
                if len(existing) > 0 and "published_on" in existing.columns:
                    resume_ts = int(existing["published_on"].min())
                    print(f"📂 Resuming from {datetime.fromtimestamp(resume_ts, tz=timezone.utc).strftime('%Y-%m-%d')}")
            
            cc_news = fetch_cryptocompare_news(
                days=args.days, resume_from_ts=resume_ts,
                api_key=args.cc_api_key
            )
            all_news.extend(cc_news)
        
        # GDELT (political/macro news) — free, unlimited, no key!
        if args.source in ("political", "all"):
            gdelt_news = fetch_gdelt_political_news(days=args.days)
            all_news.extend(gdelt_news)
        
        # CryptoPanic (optional, better sentiment, needs API key)
        if args.api_key:
            cp_news = fetch_cryptopanic_news(api_key=args.api_key, days=args.days)
            all_news.extend(cp_news)
        
        # ─── Don't save if nothing was fetched (protect existing data!) ───
        if not all_news:
            print("\n⚠️  No news fetched (rate limit?). Existing data preserved.")
            if os.path.exists(RAW_NEWS_PATH):
                raw_df = pd.read_parquet(RAW_NEWS_PATH)
                if len(raw_df) > 0:
                    print(f"   Using existing {len(raw_df):,} items")
                    news_items = raw_df.to_dict("records")
                else:
                    print("❌ No existing data either!")
                    sys.exit(1)
            else:
                print("❌ No data to process!")
                sys.exit(1)
        else:
            # Deduplicate by title + published_on
            seen = set()
            unique_news = []
            for item in all_news:
                key = (item.get("title", ""), item.get("published_on", 0))
                if key not in seen:
                    seen.add(key)
                    unique_news.append(item)
            
            print(f"\n📦 Total unique news: {len(unique_news):,} (deduped from {len(all_news):,})")
            
            # Save raw news (merge with existing if resuming)
            raw_df = pd.DataFrame(unique_news)
            
            if args.resume and os.path.exists(RAW_NEWS_PATH):
                existing = pd.read_parquet(RAW_NEWS_PATH)
                if len(existing) > 0:
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
        if len(raw_df) == 0:
            print("❌ Raw news file is empty!")
            sys.exit(1)
        news_items = raw_df.to_dict("records")
        print(f"📂 Loaded {len(news_items):,} raw news items")
    
    # ─── Score all news with selected scorer ─────────────────────
    news_items = prescore_all_news(news_items, scorer=args.scorer)
    
    # ─── Map to coins ────────────────────────────────────────────
    print("\n🔗 Mapping news to coins...")
    mapped, political_unmapped = map_news_to_coins(news_items)
    
    # ─── Build features ──────────────────────────────────────────
    print("\n🛠️  Building features...")
    features_df = build_news_features(mapped, political_unmapped=political_unmapped)
    
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