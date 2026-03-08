#!/usr/bin/env python3
"""Calculate API calls needed to fill news gaps."""
import pandas as pd

df = pd.read_parquet('data/sentiment/raw_news.parquet')
df['date'] = pd.to_datetime(df['published_on'], unit='s', utc=True)
monthly = df.set_index('date').resample('ME').size()

gap_months = [m for m, c in monthly.items() if c < 100]
print(f'Gap months: {len(gap_months)}')
print(f'Full months with data: {len(monthly) - len(gap_months)}')

# ~8000 news/month, 50/page = 160 pages/month
estimated_pages = len(gap_months) * 160
print(f'Estimated API calls needed: ~{estimated_pages:,}')
print(f'At 3000/hr: ~{estimated_pages/3000:.0f} hours')

remaining = 11000 - 8315
print(f'\nCurrent month: 8,315 / 11,000 = {remaining} remaining')
print(f'With remaining: ~{remaining*50:,} news items')

total_gap_news = len(gap_months) * 8000
print(f'\nTotal news in gaps: ~{total_gap_news:,}')
print(f'API calls needed: ~{total_gap_news//50:,}')
