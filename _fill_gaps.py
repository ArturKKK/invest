#!/usr/bin/env python3
"""
Fill gaps in downloaded news data by fetching missing date ranges.
Prioritizes most recent gaps first (most important for model training).
Respects monthly API limits and saves progress incrementally.

Usage:
  python _fill_gaps.py <cc-api-key>
  python _fill_gaps.py <cc-api-key> --max-calls 2500   # limit calls
  python _fill_gaps.py <cc-api-key> --status            # just show gaps
"""
import os
import sys
import time
import argparse
import pandas as pd
import requests
from datetime import datetime, timezone

RAW_PATH = "data/sentiment/raw_news.parquet"

# Gaps to fill — ordered NEWEST FIRST (most valuable for model)
# Status as of 2026-03-09:
#   Have: 2020-09..2021-12, 2022-12..2023-04, 2024-06..2024-08, 2025-10..2026-03
#   Missing: 2024-09..2025-09, 2023-05..2024-05, 2022-01..2022-11, 2020-01..2020-09
GAPS = [
    ("2024-08-10", "2025-10-20"),   # gap 3: 14mo  2024-09 -> 2025-09 (MOST IMPORTANT)
    ("2023-04-20", "2024-06-15"),   # gap 2: 14mo  2023-05 -> 2024-05
    ("2021-12-20", "2022-12-25"),   # gap 1: 11mo  2022-01 -> 2022-11
    ("2020-01-01", "2020-09-16"),   # extend: 8.5mo back to Jan 2020
]


def check_monthly_limit(api_key):
    """Check current monthly usage via a test call."""
    url = "https://min-api.cryptocompare.com/data/v2/news/"
    params = {"lang": "EN", "api_key": api_key}
    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        rl = data.get("RateLimit", {})
        calls = rl.get("calls_made", {})
        limits = rl.get("max_calls", {})

        month_used = calls.get("month", 0)
        month_max = limits.get("month", 11000)
        hour_used = calls.get("hour", 0)
        hour_max = limits.get("hour", 3000)
        day_used = calls.get("day", 0)
        day_max = limits.get("day", 11000)

        return {
            "month_used": month_used, "month_max": month_max,
            "hour_used": hour_used, "hour_max": hour_max,
            "day_used": day_used, "day_max": day_max,
            "remaining": month_max - month_used,
        }
    except Exception as e:
        print(f"Warning: Could not check limits: {e}")
        return None


def fetch_range(start_date, end_date, api_key, max_calls=None, call_counter=None):
    """Fetch news between start_date and end_date. Returns (news_list, calls_made)."""
    base_url = "https://min-api.cryptocompare.com/data/v2/news/"

    cutoff_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    lTs = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())

    all_news = []
    page = 0
    calls = 0
    consecutive_empty = 0

    print(f"\nFetching {start_date} -> {end_date}...")

    while lTs > cutoff_ts:
        # Check call budget
        if max_calls and calls >= max_calls:
            print(f"   Reached call budget ({calls}/{max_calls}). Saving progress.")
            break

        if call_counter and call_counter[0] >= call_counter[1]:
            print(f"   Total call budget exhausted ({call_counter[0]}/{call_counter[1]}). Saving.")
            break

        try:
            params = {"lang": "EN", "lTs": lTs, "sortOrder": "latest", "api_key": api_key}
            resp = requests.get(base_url, params=params, timeout=30)
            calls += 1
            if call_counter:
                call_counter[0] += 1

            # Check HTTP-level rate limit
            if resp.status_code == 429:
                print(f"   HTTP 429 rate limit. Waiting 60s...")
                time.sleep(60)
                continue

            resp.raise_for_status()
            data = resp.json()

            # Check for API error
            if data.get("Type") == 99:
                msg = data.get("Message", "")
                if "rate limit" in msg.lower() or "limit" in msg.lower():
                    print(f"   API rate limit: {msg}")
                    rl = data.get("RateLimit", {})
                    month_used = rl.get("calls_made", {}).get("month", 0)
                    month_max = rl.get("max_calls", {}).get("month", 11000)
                    if month_used >= month_max - 5:
                        print(f"   Monthly limit reached ({month_used}/{month_max}). Stopping.")
                        break
                    print(f"   Waiting 10 min for hourly reset...")
                    time.sleep(600)
                    continue
                consecutive_empty += 1
                if consecutive_empty >= 5:
                    print(f"   Too many errors. Stopping this gap.")
                    break
                time.sleep(2)
                continue

            news_items = data.get("Data", [])
            if not isinstance(news_items, list) or not news_items:
                consecutive_empty += 1
                if consecutive_empty >= 10:
                    print(f"   10 consecutive empty pages. Moving on.")
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

            if page % 25 == 0:
                oldest_date = datetime.fromtimestamp(oldest_ts, tz=timezone.utc)
                budget_str = f" | budget: {calls}" + (f"/{max_calls}" if max_calls else "")
                counter_str = f" | total: {call_counter[0]}/{call_counter[1]}" if call_counter else ""
                print(f"   Page {page}: {len(all_news):,} news, "
                      f"oldest: {oldest_date.strftime('%Y-%m-%d')}"
                      f"{budget_str}{counter_str}")

            time.sleep(0.25)

        except requests.exceptions.RequestException as e:
            print(f"   Request error: {e}. Retrying in 5s...")
            time.sleep(5)
        except Exception as e:
            print(f"   Error: {e}")
            break

    print(f"   Done: {len(all_news):,} items ({page} pages, {calls} API calls)")
    return all_news, calls


def show_status(df):
    """Show data coverage."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['published_on'], unit='s', utc=True)
    monthly = df.set_index('date').resample('ME').size()

    total_months = len(monthly)
    gap_months = len([c for c in monthly if c < 100])
    ok_months = total_months - gap_months

    print(f"\nCoverage: {ok_months}/{total_months} months with data ({gap_months} gaps)")
    print(f"   Total: {len(df):,} news items")
    print(f"   Range: {df['date'].min().strftime('%Y-%m-%d')} -> {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"\n   Monthly distribution:")
    for m, cnt in monthly.items():
        bar = '#' * min(cnt // 300, 30)
        marker = " << GAP!" if cnt < 100 else (" << LOW" if cnt < 500 else "")
        print(f"   {m.strftime('%Y-%m')}: {cnt:>6,} {bar}{marker}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("api_key", nargs="?", help="CryptoCompare API key")
    parser.add_argument("--max-calls", type=int, default=None,
                        help="Max total API calls (default: use all remaining)")
    parser.add_argument("--status", action="store_true",
                        help="Just show current coverage, don't fetch")
    args = parser.parse_args()

    # Load existing data
    print(f"Loading existing data from {RAW_PATH}...")
    existing = pd.read_parquet(RAW_PATH)
    print(f"   Existing: {len(existing):,} rows")

    if args.status:
        show_status(existing)
        return

    if not args.api_key:
        print("Usage: python _fill_gaps.py <cc-api-key> [--max-calls N] [--status]")
        sys.exit(1)

    # Check limits
    print(f"\nChecking API limits...")
    limits = check_monthly_limit(args.api_key)
    if limits:
        print(f"   Monthly: {limits['month_used']}/{limits['month_max']} "
              f"({limits['remaining']} remaining)")
        print(f"   Hourly:  {limits['hour_used']}/{limits['hour_max']}")

        if limits['remaining'] < 50:
            print(f"\nMonthly limit nearly exhausted! Only {limits['remaining']} calls left.")
            print(f"   Wait until next month or upgrade plan.")
            show_status(existing)
            return

        budget = args.max_calls or limits['remaining'] - 10
        print(f"   Budget for this run: {budget} calls")
    else:
        budget = args.max_calls or 2500
        print(f"   Using default budget: {budget}")

    # Fetch gaps (newest first)
    call_counter = [0, budget]  # [current, max]
    total_new = 0

    for start_date, end_date in GAPS:
        if call_counter[0] >= call_counter[1]:
            print(f"\nBudget exhausted. Remaining gaps will need another run.")
            break

        remaining_budget = call_counter[1] - call_counter[0]
        print(f"\n{'=' * 50}")
        print(f"Budget remaining: {remaining_budget} calls")

        gap_news, calls_used = fetch_range(
            start_date, end_date, args.api_key,
            max_calls=remaining_budget,
            call_counter=call_counter
        )

        if gap_news:
            new_df = pd.DataFrame(gap_news)
            existing = pd.concat([existing, new_df]).drop_duplicates(
                subset=["title", "published_on"]).reset_index(drop=True)
            existing.to_parquet(RAW_PATH, index=False)
            total_new += len(gap_news)
            print(f"   Saved: {len(existing):,} total rows (+{len(gap_news):,} new)")

    # Final status
    print(f"\n{'=' * 50}")
    print(f"Done! Added {total_new:,} new items. Total calls: {call_counter[0]}")
    final = pd.read_parquet(RAW_PATH)
    show_status(final)


if __name__ == "__main__":
    main()
