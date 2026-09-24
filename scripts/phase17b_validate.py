"""
Phase 17B -- RefillCare Identity & History Validation Script
"""
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "refillcare" / "customer_data_fields.csv"
OUT_JSON  = ROOT / "data" / "refillcare" / "processed" / "phase17b_validation.json"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

print(f"[17B] Data file : {DATA_FILE}")
print(f"[17B] Output    : {OUT_JSON}\n")

STRING_COLS = [
    "customerId","itemId","itemCode","customerName","itemName",
    "invoice_number","packing","batchNo","expiryDate",
    "MOBILE_NO","therapeuticCategory","generic_name",
]
NUMERIC_COLS = [
    "quantity","freeQuantity","rate","saleRate","mrp",
    "invoice_total","discountPercent","item_discount",
    "gstAmount","netAmount","costPrice",
]

dtype_map = {c: str for c in STRING_COLS}
print("[17B] Loading CSV ...")
raw = pd.read_csv(DATA_FILE, dtype=dtype_map, low_memory=False)
print(f"[17B] Loaded {len(raw):,} raw rows x {len(raw.columns)} columns")

for col in STRING_COLS:
    if col in raw.columns:
        raw[col] = raw[col].astype(str).str.strip()
        raw[col] = raw[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})

raw["invoice_date"] = pd.to_datetime(raw["invoice_date"], dayfirst=True, errors="coerce")

for col in NUMERIC_COLS:
    if col in raw.columns:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0.0)

for col in ["itemCode","itemId"]:
    if col in raw.columns:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

results = {}

# ---- SECTION 1 ----
print("\n[17B] Section 1: Schema & date range")
results["schema"] = {
    "total_raw_rows": int(len(raw)),
    "columns": list(raw.columns),
    "date_min": str(raw["invoice_date"].min().date()) if raw["invoice_date"].notna().any() else None,
    "date_max": str(raw["invoice_date"].max().date()) if raw["invoice_date"].notna().any() else None,
    "invalid_invoice_dates": int(raw["invoice_date"].isna().sum()),
    "mfgDate_present": "mfgDate" in raw.columns,
    "therapeuticCategory_present": "therapeuticCategory" in raw.columns,
    "generic_name_present": "generic_name" in raw.columns,
}
print(f"  Date range: {results['schema']['date_min']} to {results['schema']['date_max']}")
print(f"  Invalid dates: {results['schema']['invalid_invoice_dates']}")

# ---- SECTION 2 ----
print("\n[17B] Section 2: Customer identity")
missing_customerId   = int(raw["customerId"].isna().sum())
missing_customerName = int(raw["customerName"].isna().sum()) if "customerName" in raw.columns else 0

valid_cust = raw[raw["customerId"].notna()].copy()
unique_customerIds   = int(valid_cust["customerId"].nunique())
unique_customerNames = int(valid_cust["customerName"].nunique()) if "customerName" in valid_cust.columns else 0

cid_to_names    = valid_cust.groupby("customerId")["customerName"].nunique()
cid_multi_names = cid_to_names[cid_to_names > 1]
name_to_cids    = valid_cust.groupby("customerName")["customerId"].nunique()
name_multi_cids = name_to_cids[name_to_cids > 1]

cid_collision_sample = (
    valid_cust[valid_cust["customerId"].isin(cid_multi_names.head(10).index)]
    .groupby("customerId")["customerName"].unique().apply(list).head(10).to_dict()
)
name_collision_sample = (
    valid_cust[valid_cust["customerName"].isin(name_multi_cids.head(10).index)]
    .groupby("customerName")["customerId"].unique().apply(list).head(10).to_dict()
)

if "MOBILE_NO" in raw.columns:
    mob_df = raw[raw["MOBILE_NO"].notna()].copy()
    shared_mobile = mob_df.groupby("MOBILE_NO")["customerId"].nunique()
    shared_mobile_count = int((shared_mobile > 1).sum())
    shared_mobile_cust_total = int(
        mob_df[mob_df["MOBILE_NO"].isin(shared_mobile[shared_mobile > 1].index)]["customerId"].nunique()
    )
    sm_sample_df = (
        mob_df[mob_df["MOBILE_NO"].isin(shared_mobile[shared_mobile > 1].index)]
        .groupby("MOBILE_NO")["customerId"].unique().apply(list)
        .reset_index()
        .sort_values("customerId", key=lambda s: s.apply(len), ascending=False)
        .head(10)
    )
    shared_mobile_sample_dict = {row["MOBILE_NO"]: row["customerId"] for _, row in sm_sample_df.iterrows()}
    missing_mobile = int(raw["MOBILE_NO"].isna().sum())
else:
    shared_mobile_count, shared_mobile_cust_total, missing_mobile = 0, 0, 0
    shared_mobile_sample_dict = {}

results["customer_identity"] = {
    "missing_customerId_rows": missing_customerId,
    "missing_customerName_rows": missing_customerName,
    "unique_customerIds": unique_customerIds,
    "unique_customerNames": unique_customerNames,
    "customerIds_with_multiple_names": int(len(cid_multi_names)),
    "customerNames_with_multiple_ids": int(len(name_multi_cids)),
    "customerId_collision_sample": {str(k): [str(x) for x in v] for k,v in cid_collision_sample.items()},
    "customerName_collision_sample": {str(k): [str(x) for x in v] for k,v in name_collision_sample.items()},
    "missing_MOBILE_NO_rows": missing_mobile,
    "shared_MOBILE_NO_count": shared_mobile_count,
    "customers_on_shared_MOBILE_NO": shared_mobile_cust_total,
    "shared_MOBILE_NO_sample": {str(k): [str(x) for x in v] for k,v in shared_mobile_sample_dict.items()},
}
print(f"  Unique customerIds   : {unique_customerIds:,}")
print(f"  Unique customerNames : {unique_customerNames:,}")
print(f"  Missing customerId rows      : {missing_customerId:,}")
print(f"  customerIds with >1 name     : {len(cid_multi_names):,}")
print(f"  customerNames with >1 id     : {len(name_multi_cids):,}")
print(f"  Shared MOBILE_NO groups      : {shared_mobile_count:,}")
print(f"  Customers on shared MOBILE_NO: {shared_mobile_cust_total:,}")

# ---- SECTION 3 ----
print("\n[17B] Section 3: Medicine identity")
missing_itemId   = int(raw["itemId"].isna().sum())
missing_itemCode = int(raw["itemCode"].isna().sum()) if "itemCode" in raw.columns else 0
missing_itemName = int(raw["itemName"].isna().sum()) if "itemName" in raw.columns else 0

valid_item = raw[raw["itemId"].notna()].copy()
unique_itemIds   = int(valid_item["itemId"].nunique())
unique_itemCodes = int(valid_item["itemCode"].nunique()) if "itemCode" in valid_item.columns else 0

if "itemCode" in valid_item.columns and "itemName" in valid_item.columns:
    code_to_names    = valid_item.groupby("itemCode")["itemName"].nunique()
    code_multi_names = code_to_names[code_to_names > 1]
    code_collision_sample = (
        valid_item[valid_item["itemCode"].isin(code_multi_names.head(10).index)]
        .groupby("itemCode")["itemName"].unique().apply(list).head(10).to_dict()
    )
else:
    code_multi_names = pd.Series(dtype=int)
    code_collision_sample = {}

if "itemCode" in valid_item.columns:
    itemid_to_codes    = valid_item.groupby("itemId")["itemCode"].nunique()
    itemid_multi_codes = itemid_to_codes[itemid_to_codes > 1]
else:
    itemid_multi_codes = pd.Series(dtype=int)

results["medicine_identity"] = {
    "missing_itemId_rows": missing_itemId,
    "missing_itemCode_rows": missing_itemCode,
    "missing_itemName_rows": missing_itemName,
    "unique_itemIds": unique_itemIds,
    "unique_itemCodes": unique_itemCodes,
    "itemCodes_with_multiple_names": int(len(code_multi_names)),
    "itemIds_with_multiple_codes": int(len(itemid_multi_codes)),
    "itemCode_name_collision_sample": {str(k): [str(x) for x in v] for k,v in code_collision_sample.items()},
}
print(f"  Unique itemIds   : {unique_itemIds:,}")
print(f"  Unique itemCodes : {unique_itemCodes:,}")
print(f"  Missing itemId rows          : {missing_itemId:,}")
print(f"  itemCodes with >1 name       : {len(code_multi_names):,}")
print(f"  itemIds with >1 code         : {len(itemid_multi_codes):,}")

# ---- SECTION 4 ----
print("\n[17B] Section 4: Transaction integrity")
exact_dups = int(raw.duplicated().sum())

work = raw[
    raw["customerId"].notna() &
    raw["itemId"].notna() &
    raw["invoice_date"].notna()
].copy()

dup_event_key    = ["invoice_number","customerId","itemId"]
logical_dups     = int(work.duplicated(subset=dup_event_key).sum())
negative_qty     = int((work["quantity"] < 0).sum()) if "quantity" in work.columns else 0
zero_qty         = int((work["quantity"] == 0).sum()) if "quantity" in work.columns else 0
neg_zero_qty     = int((work["quantity"] <= 0).sum()) if "quantity" in work.columns else 0

multi_line_counts   = work.groupby(["invoice_number","customerId","itemId"]).size()
multi_line_invoices = int((multi_line_counts > 1).sum())
multi_line_rows     = int(work.duplicated(subset=["invoice_number","customerId","itemId"], keep=False).sum())

results["transaction_integrity"] = {
    "total_valid_rows": int(len(work)),
    "exact_duplicate_rows": exact_dups,
    "logical_duplicate_events": logical_dups,
    "negative_quantity_rows": negative_qty,
    "zero_quantity_rows": zero_qty,
    "neg_or_zero_quantity_rows": neg_zero_qty,
    "multi_line_invoice_item_combos": multi_line_invoices,
    "multi_line_total_rows": multi_line_rows,
}
print(f"  Valid rows (non-null keys)   : {len(work):,}")
print(f"  Exact duplicate rows         : {exact_dups:,}")
print(f"  Logical duplicate events     : {logical_dups:,}")
print(f"  Negative quantity            : {negative_qty:,}")
print(f"  Zero quantity                : {zero_qty:,}")
print(f"  Multi-line invoice+item      : {multi_line_invoices:,} combos ({multi_line_rows:,} rows)")

# ---- SECTION 5 ----
print("\n[17B] Section 5: Purchase-history distribution")
hist = work.sort_values(["customerId","itemId","invoice_date","invoice_number"], ascending=True).copy()
pair_counts = hist.groupby(["customerId","itemId"]).size()

total_histories = int(len(pair_counts))
hist_1plus  = int((pair_counts >= 1).sum())
hist_2plus  = int((pair_counts >= 2).sum())
hist_3plus  = int((pair_counts >= 3).sum())
hist_5plus  = int((pair_counts >= 5).sum())
hist_10plus = int((pair_counts >= 10).sum())
hist_20plus = int((pair_counts >= 20).sum())

print(f"  Total customer x medicine histories : {total_histories:,}")
print(f"  >=1  purchases: {hist_1plus:,}")
print(f"  >=2  purchases: {hist_2plus:,}")
print(f"  >=3  purchases: {hist_3plus:,}")
print(f"  >=5  purchases: {hist_5plus:,}")
print(f"  >=10 purchases: {hist_10plus:,}")
print(f"  >=20 purchases: {hist_20plus:,}")

hist["previous_purchase_date"] = hist.groupby(["customerId","itemId"])["invoice_date"].shift(1)
hist["days_interval"] = (hist["invoice_date"] - hist["previous_purchase_date"]).dt.days

valid_intervals = hist["days_interval"].dropna()
zero_intervals  = int((valid_intervals == 0).sum())
neg_intervals   = int((valid_intervals < 0).sum())
large_intervals = int((valid_intervals > 365).sum())

interval_pcts = {
    "0d_same_day" : int((valid_intervals == 0).sum()),
    "1_14d"       : int(((valid_intervals >= 1) & (valid_intervals <= 14)).sum()),
    "15_30d"      : int(((valid_intervals >= 15) & (valid_intervals <= 30)).sum()),
    "31_60d"      : int(((valid_intervals >= 31) & (valid_intervals <= 60)).sum()),
    "61_90d"      : int(((valid_intervals >= 61) & (valid_intervals <= 90)).sum()),
    "91_120d"     : int(((valid_intervals >= 91) & (valid_intervals <= 120)).sum()),
    "121_180d"    : int(((valid_intervals >= 121) & (valid_intervals <= 180)).sum()),
    "181_365d"    : int(((valid_intervals >= 181) & (valid_intervals <= 365)).sum()),
    "over_365d"   : int((valid_intervals > 365).sum()),
}
recurring_count = int(((valid_intervals >= 15) & (valid_intervals <= 120)).sum())
total_intervals = int(len(valid_intervals))
recurring_pct   = round(recurring_count / total_intervals * 100, 2) if total_intervals > 0 else 0.0

if len(valid_intervals) > 0:
    interval_stats = {
        "total_intervals": total_intervals,
        "median_days": float(valid_intervals.median()),
        "mean_days": round(float(valid_intervals.mean()), 2),
        "std_days": round(float(valid_intervals.std()), 2),
        "min_days": int(valid_intervals.min()),
        "max_days": int(valid_intervals.max()),
        "zero_day_intervals": zero_intervals,
        "negative_intervals": neg_intervals,
        "large_intervals_over_365d": large_intervals,
        "recurring_15_120d_count": recurring_count,
        "recurring_15_120d_pct": recurring_pct,
        "distribution_buckets": interval_pcts,
    }
else:
    interval_stats = {"total_intervals": 0}

pair_intervals       = hist[hist["days_interval"].notna()].groupby(["customerId","itemId"])["days_interval"]
pair_median_interval = pair_intervals.median()
pair_purchase_cnt    = pair_counts[pair_counts.index.isin(pair_median_interval.index)]
recurring_mask = (
    (pair_purchase_cnt >= 3) &
    (pair_median_interval >= 15) &
    (pair_median_interval <= 120)
)
recurring_histories = int(recurring_mask.sum())
if recurring_histories > 0:
    rec_pairs      = pair_purchase_cnt[recurring_mask].reset_index()[["customerId","itemId"]]
    rec_merged     = hist.merge(rec_pairs, on=["customerId","itemId"], how="inner")
    recurring_customers = int(rec_merged["customerId"].nunique())
else:
    recurring_customers = 0

results["purchase_history"] = {
    "total_customer_medicine_histories": total_histories,
    "histories_1plus_purchases": hist_1plus,
    "histories_2plus_purchases": hist_2plus,
    "histories_3plus_purchases": hist_3plus,
    "histories_5plus_purchases": hist_5plus,
    "histories_10plus_purchases": hist_10plus,
    "histories_20plus_purchases": hist_20plus,
    "interval_statistics": interval_stats,
    "recurring_histories_3plus_15_120d": recurring_histories,
    "recurring_customers": recurring_customers,
}

print(f"\n  Interval stats: median={interval_stats.get('median_days')} d, mean={interval_stats.get('mean_days')} d, std={interval_stats.get('std_days')} d")
print(f"  Zero-day intervals  : {zero_intervals:,}")
print(f"  Negative intervals  : {neg_intervals:,}")
print(f"  >365d intervals     : {large_intervals:,}")
print(f"  Recurring (15-120d) : {recurring_pct}% of intervals")
print(f"  Recurring histories : {recurring_histories:,} pairs")
print(f"  Recurring customers : {recurring_customers:,}")

# ---- SECTION 6 ----
print("\n[17B] Section 6: Pipeline compatibility")
REQUIRED_COLS = [
    "invoice_date","invoice_number","customerName","itemCode",
    "itemName","MOBILE_NO","customerId","itemId","quantity","netAmount",
]
FEATURE_REQUIRED_COLS = [
    "customerId","itemId","invoice_date","invoice_number",
    "quantity","freeQuantity","rate","mrp","discountPercent","netAmount","gstAmount",
]
actual_cols       = set(raw.columns)
missing_required  = [c for c in REQUIRED_COLS if c not in actual_cols]
missing_feat_cols = [c for c in FEATURE_REQUIRED_COLS if c not in actual_cols]

results["pipeline_compatibility"] = {
    "missing_required_cols": missing_required,
    "missing_feature_pipeline_cols": missing_feat_cols,
    "therapeuticCategory_in_dataset": "therapeuticCategory" in actual_cols,
    "generic_name_in_dataset": "generic_name" in actual_cols,
    "cleaning_py_still_references_removed_cols": True,
    "mfgDate_present_pipeline_drops_it": "mfgDate" in actual_cols,
    "all_required_cols_present": len(missing_required) == 0,
}
print(f"  Missing required cols    : {missing_required or 'None'}")
print(f"  Missing feature cols     : {missing_feat_cols or 'None'}")
print(f"  therapeuticCategory present: {'therapeuticCategory' in actual_cols}")
print(f"  generic_name present       : {'generic_name' in actual_cols}")
print(f"  cleaning.py refs removed cols: TRUE (harmless -- guarded by 'if col in df.columns')")

# ---- SECTION 7 ----
print("\n[17B] Section 7: Phase 17C readiness")
blockers = []
warnings_list = []

if missing_required:
    blockers.append(f"Missing required columns: {missing_required}")
if neg_intervals > 0:
    blockers.append(f"{neg_intervals} negative purchase intervals -- chronological ordering broken")
if logical_dups > 0:
    warnings_list.append(f"{logical_dups} logical duplicate events (same invoice_number+customerId+itemId) -- handled by aggregate_invoice_items().")
if len(cid_multi_names) > 0:
    warnings_list.append(f"{len(cid_multi_names)} customerIds map to >1 customerName -- name drift, not a blocker (customerId is canonical).")
if len(name_multi_cids) > 0:
    warnings_list.append(f"{len(name_multi_cids)} customerNames map to >1 customerId -- probable same-name collisions.")
if shared_mobile_count > 0:
    warnings_list.append(f"{shared_mobile_count} MOBILE_NO values shared by >1 customer. MOBILE_NO is NOT identity -- histories strictly isolated by customerId.")
if neg_zero_qty > 0:
    warnings_list.append(f"{neg_zero_qty} rows with zero/negative quantity -- pipeline flags via is_positive_quantity.")
if len(code_multi_names) > 0:
    warnings_list.append(f"{len(code_multi_names)} itemCodes map to >1 itemName -- probable pack-size variants. itemId is canonical.")

suitable_for_17c = len(blockers) == 0
results["phase17c_readiness"] = {
    "suitable_for_phase_17c": suitable_for_17c,
    "blockers": blockers,
    "warnings": warnings_list,
}

verdict = "SUITABLE" if suitable_for_17c else "BLOCKED"
print(f"  Phase 17C readiness: {verdict}")
for b in blockers:
    print(f"  BLOCKER: {b}")
for w in warnings_list:
    print(f"  WARNING: {w}")

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n[17B] JSON results saved: {OUT_JSON}")
print("[17B] Validation script complete.")
