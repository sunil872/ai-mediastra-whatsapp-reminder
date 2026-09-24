import pandas as pd
from datetime import date

tx_df = pd.read_parquet('data/refillcare/processed/clean_transactions.parquet')
tx_df['invoice_date'] = pd.to_datetime(tx_df['invoice_date']).dt.date
as_of = date(2026, 9, 24)

sub = tx_df[(tx_df['customerId'] == 'MURLIKRISHNA') & (tx_df['itemId'] == '1148') & (tx_df['invoice_date'] <= as_of)].sort_values('invoice_date')

print(f"Total transactions found: {len(sub)}")

# Aggregate by date
agg = sub.groupby('invoice_date').agg({
    'quantity': 'sum',
    'packing': 'first'
}).reset_index().sort_values('invoice_date')

print(f"\nTotal Distinct Purchase Dates / Visits: {len(agg)}")
for idx, r in agg.iterrows():
    d = r['invoice_date']
    prev_d = agg['invoice_date'].iloc[idx-1] if idx > 0 else None
    gap = f"{(d - prev_d).days}d gap" if prev_d else "Initial"
    packs = int(r['quantity'])
    tabs = packs * 15
    print(f"{idx+1:2d}. {d.strftime('%Y-%m-%d')} | {packs:2d} packs ({tabs:3d} tabs) | Gap: {gap}")

