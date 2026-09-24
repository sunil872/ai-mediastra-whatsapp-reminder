import pandas as pd
import numpy as np
import re
from pathlib import Path

cols = ['customerId', 'itemId', 'invoice_date', 'quantity', 'packing', 'itemName']
df = pd.read_parquet('data/refillcare/processed/purchase_history.parquet', columns=cols)

print('Total rows in purchase_history:', len(df))

def parse_pack(val):
    if pd.isna(val):
        return None, 'NULL'
    s = str(val).strip().upper()
    if not s:
        return None, 'EMPTY'
    
    # 1XN or NXN format (solid units per strip/box)
    m = re.match(r'^(\d+)\s*[X\*]\s*(\d+)$', s)
    if m:
        return int(m.group(1)) * int(m.group(2)), 'SOLID_MULT'
    
    # Pure integer like '1', '10', '15'
    if re.match(r'^\d+$', s):
        return int(s), 'INTEGER'
    
    # N PC, N PCS, N TAB, N CAP
    m = re.match(r'^(\d+)\s*(PC|PCS|TAB|CAP|VIAL|AMP|BOTTLE|STRIP|T)$', s)
    if m:
        return int(m.group(1)), 'UNIT_COUNT'
        
    # Liquids / Ointments / Powders: N ML, N GM, N MG, N L, N KG
    m = re.match(r'^(\d+(\.\d+)?)\s*(ML|GM|GMS|G|MG|L|KG|LTR)$', s)
    if m:
        return float(m.group(1)), 'VOLUME_WEIGHT'
        
    return None, 'UNPARSEABLE'

parsed = [parse_pack(p) for p in df['packing']]
df['pack_units'] = [p[0] for p in parsed]
df['pack_type'] = [p[1] for p in parsed]

print('\n=== Pack Type Breakdown ===')
type_counts = df['pack_type'].value_counts(dropna=False)
type_pcts = df['pack_type'].value_counts(normalize=True) * 100
for t, cnt in type_counts.items():
    print(f'{t:15s}: {cnt:>8,} ({type_pcts[t]:>6.2f}%)')

# Discrete solid units
solid_types = ['SOLID_MULT', 'INTEGER', 'UNIT_COUNT']
df['is_discrete_solid'] = df['pack_type'].isin(solid_types)

# Valid quantity & units
df['has_valid_qty'] = pd.to_numeric(df['quantity'], errors='coerce').notna() & (df['quantity'] > 0)
df['can_calc_units'] = df['pack_units'].notna() & df['has_valid_qty']
df['total_units'] = np.where(df['can_calc_units'], df['quantity'] * df['pack_units'], np.nan)

valid_pack_cnt = int(df['pack_units'].notna().sum())
discrete_cnt = int(df['is_discrete_solid'].sum())
can_calc_cnt = int(df['can_calc_units'].sum())

print('\n=== Usability Stats ===')
print(f'Rows with valid packing: {valid_pack_cnt:,} ({valid_pack_cnt/len(df)*100:.2f}%)')
print(f'Rows with discrete solid units (tablets/caps/pcs): {discrete_cnt:,} ({discrete_cnt/len(df)*100:.2f}%)')
print(f'Rows where Total Units can be calculated (valid pack + positive qty): {can_calc_cnt:,} ({can_calc_cnt/len(df)*100:.2f}%)')

# 5: Customer + Item histories with >= 2 usable purchases
total_pairs = df.groupby(['customerId', 'itemId']).size()
total_ge2 = (total_pairs >= 2).sum()

usable_mask = df['can_calc_units'].values
usable_pairs = df[usable_mask].groupby(['customerId', 'itemId']).size()
usable_ge2 = (usable_pairs >= 2).sum()

# Also discrete solid >= 2
solid_usable_mask = (df['can_calc_units'] & df['is_discrete_solid']).values
solid_pairs = df[solid_usable_mask].groupby(['customerId', 'itemId']).size()
solid_ge2 = (solid_pairs >= 2).sum()

print('\n=== Customer + Item History Depth ===')
print(f'Total unique customerId + itemId pairs in dataset: {len(total_pairs):,}')
print(f'Total pairs with >= 2 purchases: {total_ge2:,} ({total_ge2/len(total_pairs)*100:.2f}%)')
print(f'Pairs with >= 2 usable (all parsed packing & positive qty) purchases: {usable_ge2:,} ({usable_ge2/len(total_pairs)*100:.2f}%)')
print(f'Pairs with >= 2 discrete solid (tablets/caps/pcs) purchases: {solid_ge2:,} ({solid_ge2/len(total_pairs)*100:.2f}%)')

# 6: Examples of 1X15 + quantity 4
ex_1x15_q4 = df[(df['packing'].astype(str).str.upper().isin(['1X15', '1*15'])) & (df['quantity'] == 4)]
print(f'\nRows with 1X15 and quantity 4: {len(ex_1x15_q4):,}')
print(ex_1x15_q4[['customerId', 'itemName', 'packing', 'quantity', 'pack_units', 'total_units', 'invoice_date']].head(8).to_string(index=False))

# 7: Ambiguous / unparseable formats
unparseable = df[df['pack_type'] == 'UNPARSEABLE']['packing'].value_counts()
print(f'\nTop Unparseable formats ({len(unparseable)} distinct strings, {unparseable.sum():,} total rows):')
print(unparseable.head(25))

# 8: ItemId packing consistency
item_pack = df.groupby('itemId')['packing'].apply(lambda s: s.str.upper().str.strip().nunique())
inconsistent_items = item_pack[item_pack > 1]
print(f'\n=== Item Packing Consistency ===')
print(f'Total unique itemIds: {len(item_pack):,}')
print(f'ItemIds with exactly 1 distinct packing: {(item_pack == 1).sum():,} ({(item_pack == 1).mean()*100:.2f}%)')
print(f'ItemIds with >1 distinct packing: {len(inconsistent_items):,} ({len(inconsistent_items)/len(item_pack)*100:.2f}%)')

# Check reasons for inconsistency (case variations vs actual changes)
print('\nSample items with >1 packing representations:')
sample_inconsistent_ids = inconsistent_items.head(10).index
for iid in sample_inconsistent_ids:
    sub = df[df['itemId'] == iid]
    print(f'ItemId {iid} ({sub["itemName"].iloc[0]}): distinct packings = {sub["packing"].unique().tolist()}')
