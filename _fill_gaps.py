#!/usr/bin/env python3
"""Fill gaps in downloaded news data by fetching missing date ranges."""
import os
import sys
import time
import pandas as pd
import requests
from datetime import datetime, timezone

API_KEY = sys.argv[1] if len(sys.argv) > 1 else None
if not API_KEY:
    print("Usage: python _fill_gaps.py <cc-api-key>")
    sys.exit(1)

RAW_PATH = "data/sentiment/raw_news.parquet"

# Gaps to fill (start from END of gap, paginate backward to START)
GAPS = [
    ("2021-12-20", "2022-12-25"),   # gap 1: 2022-01 -> 2022-11
    ("2023-04-20", "2024-06-15"),   # gap 2: 2023-05 -> 2024-05
    ("2024-08-10", "2025-10-20"),   # gap 3: 2024-09 -> 2025-09
]

def fetch_range(start_date, end_date, api_key):
    """Fetch news between start_date and end_date (both YYYY-MM-DD strings)."""
    base_url = "https://min-api.cryptocompare.com/data/v2/news/"
    
    cutoff_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    lTs = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    
    all_news = []
    page = 0
    consecutive_empty = 0
    
    print(f"\n📡 Fetching {start_date} -> {end_date}...")
    
    while lTs > cutoff_ts:
        try:
            params = {"lang": "EN", "lTs": lTs, "sortOrder": "latest", "api_key": api_key}
            resp = requests.get(base_url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            # Rate limit
            rate_limit = data.get("RateLimit", {})
            calls_made = rate_limit.get("calls_made", {})
            max_calls = rate_limit.get("max_calls", {})
            hour_used = calls_made.get("hour", 0)
            hour_max = max_calls.get("hour", 3000)
            
            if hour_used >= hour_max - 10:
                print(f"   ⏳ Rate limit ({hour_used}/{hour_max}). Waiting 5 min...")
                time.sleep(300)
                continue
            
            if data.get("Type") == 99 or not data.get("Data"):
                consecutive_empty += 1
                if consecutive_empty >= 5:
                    print(f"   ⏳ Rate limited. Waiting 5 min...")
                    time.sleep(300)
                    consecutive_empty = 0
                    continue
                time.sleep(2)
                continue
            
            news_items = data["Data"]
            if not isinstance(news_items, list) or not news_items:
                consecutive_empty += 1
                if consecutive_empty >= 10:
                    break
                time.sleep(1)
                continue
            
            consecutive_empty = 0
            
            for item in news_items:
                all_news.append({
                    "id": item.get("id"),
                    "title": item.get("title", ""),
                    "body": item.get("body", "")[:500],
                    "categories": item.get("categories", ""),
                    "source": item.get("source_info", {}).get("name", ""),
                    "published_on": item.get("published_on", 0),
                    "url": item.get("url", ""),
                    "tags": item.get("tags", ""),
                })
            
            oldest_ts = min(it["published_on"] for it in news_items)
            lTs = oldest_ts
            page += 1
            
            if page % 50 == 0:
                oldest_date = datetime.fromtimestamp(oldest_ts, tz=timezone.utc)
                print(f"   Page {page}: {len(all_news):,} news, "
                      f"oldest: {oldest_date.strftime('%Y-%m-%d')}, "
                      f"rate: {hour_used}/{hour_max}/hr")
            
            time.sleep(0.25)
            
        except requests.exceptions.RequestException as e:
            print(f"   ⚠️  Request error: {e}. Retrying in 5s...")
            time.sleep(5)
        except Exception as e:
            print(f"   ❌ Error: {e}")
            break
    
    print(f"   ✅ Fetched {len(all_news):,} items ({page} pages)")
    return all_news


# Load existing data
print(f"📂 Loading existing data from {RAW_PATH}...")
existing = pd.read_parquet(RAW_PATH)
print(f"   Existing: {len(existing):,} rows")

# Fetch each gap
all_new = []
for start_date, end_date in GAPS:
    gap_news = fetch_range(start_date, end_date, API_KEY)
    all_new.extend(gap_news)
    
    # Save intermediate progress
    if all_new:
        new_df = pd.DataFrame(all_new)
        merged = pd.concat([existing, new_df]).drop_duplicates(
            subset=["title", "published_on"]).reset_index(drop=True)
        merged.to_parquet(RAW_PATH, index=False)
        print(f"   💾 Saved: {len(merged):,} total rows")

# Final stats
final = pd.read_parquet(RAW_PATH)
final["date"] = pd.to_datetime(final["published_on"], unit="s", utc=True)
monthly = final.set_index("date").resample("ME").size()
print(f"\n✅ Final: {len(final):,} rows")
print("Monthly distribution:")
for m, cnt in monthly.items():
    marker = " << LOW!" if cnt < 500 else ""
    print(f"  {m.strftime('%Y-%m')}: {cnt:>6,}{marker}")
