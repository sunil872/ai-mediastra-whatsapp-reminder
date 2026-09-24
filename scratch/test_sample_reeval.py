import sys
sys.path.insert(0, ".")
import pandas as pd
from datetime import date
from refillcare.engine.unified_engine import UnifiedRefillDecisionEngine
from database.connection import SessionLocal
from refillcare.engine.persistence import RefillPersistenceManager

db = SessionLocal()
pm = RefillPersistenceManager()
queue = pm.get_today_review_queue(db, target_date=date(2026, 9, 24))
db.close()
df = pd.DataFrame(queue)
sample_a = df[df['path'] == 'PATH_A'].sample(n=10, random_state=42)
sample_b = df[df['path'] == 'PATH_B'].sample(n=5, random_state=42)
sample = pd.concat([sample_a, sample_b]).reset_index(drop=True)

tx_df = pd.read_parquet('data/refillcare/processed/clean_transactions.parquet')
tx_df['invoice_date'] = pd.to_datetime(tx_df['invoice_date']).dt.date
as_of = date(2026, 9, 24)

engine = UnifiedRefillDecisionEngine()

for idx, r in sample.iterrows():
    cid = str(r['customer_id'])
    iid = str(r['item_id'])
    sub = tx_df[(tx_df['customerId'] == cid) & (tx_df['itemId'] == iid) & (tx_df['invoice_date'] <= as_of)].sort_values('invoice_date')
    dates = sub['invoice_date'].tolist()
    qtys = sub['quantity'].tolist()
    packs = sub['packing'].tolist()
    mob = sub['mobile'].iloc[-1] if 'mobile' in sub.columns and len(sub) > 0 else None
    
    dec = engine.evaluate_customer_item_trajectory(
        customer_id=cid,
        customer_name=r['customer_name'],
        mobile_no=mob,
        item_id=iid,
        item_name=r['item_name'],
        dates=dates,
        quantities=qtys,
        packings=packs,
        as_of_date=as_of
    )
    print(f"=== #{idx+1} {cid} - {r['item_name']} ({dec.path}) ===")
    print(f"Old DB: refill={r['expected_refill_date']}, offset={r['stage_offset']}, reason={r['decision_reason']}")
    print(f"New Engine: eligible={dec.is_eligible}, interval={dec.predicted_interval_days}d, refill={dec.expected_refill_date}")
    print(f"New Method & Reason: {dec.prediction_method} | {dec.decision_reason}")
    if dec.expected_refill_date:
        diff_from_as_of = (as_of - dec.expected_refill_date).days
        print(f"Days relative to 2026-09-24: {diff_from_as_of:+d}d")
        from datetime import timedelta
        stage_offsets = [-7, -3, -1, 0, 2, 5]
        sched = {off: dec.expected_refill_date + timedelta(days=off) for off in stage_offsets}
        sched_str = ", ".join([f"{off:+d}d: {dt}" for off, dt in sched.items()])
        print(f"New 6-Stage Schedule: {sched_str}")
        matches = [off for off, dt in sched.items() if dt == as_of]
        if matches:
            print(f"Trigger Today (2026-09-24): Stage Offset {matches[0]:+d}")
        else:
            print(f"Trigger Today (2026-09-24): No trigger today (Day {diff_from_as_of:+d} relative to due date)")
    print()
