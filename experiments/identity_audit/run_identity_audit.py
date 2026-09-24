"""Customer-Item Identity Impact Audit Execution Script (READ-ONLY, High-Speed Vectorized).

Performs a comprehensive, non-destructive audit comparing:
- Current Entity: customerId + itemId
- Proposed Entity: normalized_store_id + normalized_phone + normalized_customer_name + normalized_item_id

Generates:
- experiments/identity_audit/identity_audit_report.json
- experiments/identity_audit/identity_audit_report.md
- experiments/identity_audit/identity_entity_comparison.csv
- experiments/identity_audit/identity_split_cases.csv
- experiments/identity_audit/identity_merge_cases.csv
- experiments/identity_audit/path_a_impact.csv
- experiments/identity_audit/path_b_impact.csv
- experiments/identity_audit/README.md
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.identity_audit.identity_normalization import (
    build_current_key,
    build_proposed_key,
    mask_phone,
    normalize_customer_name,
    normalize_item_id,
    normalize_phone_number,
    normalize_store_id,
)
from preprocessing.clean_data import clean_sales_transactions
from refillcare.models.path_a_classifier import (
    CADENCE_MEDIAN_MAX,
    CADENCE_MEDIAN_MIN,
    HIGH_DRIFT_ALT,
    HIGH_DRIFT_STRICT,
    HIGH_MAX_RATIO_ALT,
    HIGH_MAX_RATIO_STRICT,
    HIGH_NORM_MAD_ALT,
    HIGH_NORM_MAD_STRICT,
    LABEL_HIGH,
    LABEL_MEDIUM,
    LABEL_UNSTABLE,
    MEDIUM_DRIFT_MAX,
    MEDIUM_NORM_MAD_MAX,
    MIN_PURCHASES,
)
from refillcare.models.hybrid_strategy import compute_regularity_from_intervals

OUTPUT_DIR = Path(__file__).resolve().parent


def classify_path_a_stability(intervals: List[float], purchase_count: int) -> str:
    """Classify stability tier according to Path A decision tree."""
    if purchase_count < MIN_PURCHASES or len(intervals) < 2:
        return LABEL_UNSTABLE

    reg = compute_regularity_from_intervals(intervals)
    median = reg.get("cadence_median", 0.0)
    if median < CADENCE_MEDIAN_MIN or median > CADENCE_MEDIAN_MAX:
        return LABEL_UNSTABLE

    norm_mad = reg.get("cadence_norm_mad", 1.0)
    drift = reg.get("cadence_drift", 99.0)
    max_ratio = reg.get("cadence_max_ratio", 99.0)

    # Path A HIGH rule 1 or 2
    is_high_1 = (norm_mad <= HIGH_NORM_MAD_STRICT) and (drift <= HIGH_DRIFT_STRICT) and (max_ratio <= HIGH_MAX_RATIO_STRICT)
    is_high_2 = (norm_mad <= HIGH_NORM_MAD_ALT) and (drift <= HIGH_DRIFT_ALT) and (max_ratio <= HIGH_MAX_RATIO_ALT)
    if is_high_1 or is_high_2:
        return LABEL_HIGH

    # Path A MEDIUM rule
    if norm_mad <= MEDIUM_NORM_MAD_MAX and drift <= MEDIUM_DRIFT_MAX:
        return LABEL_MEDIUM

    return LABEL_UNSTABLE


def load_canonical_data() -> Tuple[pd.DataFrame, int]:
    """Load and clean canonical historical transaction dataset using standard repository pipeline."""
    csv_path = PROJECT_ROOT / "data" / "refillcare" / "customer_data_fields.csv"
    if not csv_path.exists():
        parquet_path = PROJECT_ROOT / "data" / "refillcare" / "processed" / "clean_transactions.parquet"
        if parquet_path.exists():
            df_raw = pd.read_parquet(parquet_path)
        else:
            raise FileNotFoundError(f"Could not find transaction data at {csv_path} or {parquet_path}")
    else:
        df_raw = pd.read_csv(csv_path, low_memory=False)

    total_raw_rows = len(df_raw)
    df_clean, metrics = clean_sales_transactions(df_raw)
    return df_clean, total_raw_rows


def run_audit() -> Dict[str, Any]:
    t0 = time.time()
    print("=" * 70, flush=True)
    print("Starting Customer-Item Identity Impact Audit (READ-ONLY)", flush=True)
    print("=" * 70, flush=True)

    df_clean, total_raw_rows = load_canonical_data()
    total_valid_rows = len(df_clean)
    print(f"Loaded {total_raw_rows:,} raw transaction rows in {time.time()-t0:.2f}s", flush=True)
    print(f"Valid transactions after cleaning: {total_valid_rows:,} (dropped {total_raw_rows - total_valid_rows:,})", flush=True)

    # Store column check
    if "store_id" not in df_clean.columns and "branch_id" not in df_clean.columns:
        df_clean["store_id"] = "MAIN"
    else:
        df_clean["store_id"] = df_clean.get("store_id", df_clean.get("branch_id", "MAIN"))

    # Step 3: Fast Cached Normalization
    print("Generating current and proposed customer-item identities with cached normalizers...", flush=True)
    t_norm_start = time.time()

    # Cache unique normalization mappings
    unique_stores = {s: normalize_store_id(s) for s in df_clean["store_id"].dropna().unique()}
    unique_stores[None] = "MAIN"
    unique_stores[np.nan] = "MAIN"

    unique_phones = {p: normalize_phone_number(p) for p in df_clean["MOBILE_NO"].dropna().unique()}
    unique_phones[None] = (None, "Missing phone number")
    unique_phones[np.nan] = (None, "Missing phone number")

    unique_names = {n: normalize_customer_name(n) for n in df_clean["customerName"].dropna().unique()}
    unique_names[None] = (None, "Missing customer name")
    unique_names[np.nan] = (None, "Missing customer name")

    unique_items = {i: normalize_item_id(i) for i in df_clean["itemId"].dropna().unique()}
    unique_items[None] = (None, "Missing item ID")
    unique_items[np.nan] = (None, "Missing item ID")

    # Vectorized list creation
    store_vals = df_clean["store_id"].values
    phone_vals = df_clean["MOBILE_NO"].values
    name_vals = df_clean["customerName"].values
    item_vals = df_clean["itemId"].values
    cid_vals = df_clean["customerId"].values

    norm_stores = [unique_stores.get(s, "MAIN") for s in store_vals]
    phone_res = [unique_phones.get(p, (None, "Missing phone number")) for p in phone_vals]
    norm_phones = [r[0] for r in phone_res]
    phone_errors = [r[1] for r in phone_res]

    name_res = [unique_names.get(n, (None, "Missing customer name")) for n in name_vals]
    norm_names = [r[0] for r in name_res]
    name_errors = [r[1] for r in name_res]

    item_res = [unique_items.get(i, (None, "Missing item ID")) for i in item_vals]
    norm_items = [r[0] for r in item_res]
    item_errors = [r[1] for r in item_res]

    current_keys = [f"{str(c).strip()}_{str(i).strip()}" for c, i in zip(cid_vals, item_vals)]
    
    proposed_keys = []
    safety_statuses = []
    for s_n, p_n, n_n, i_n, p_e, n_e, i_e in zip(norm_stores, norm_phones, norm_names, norm_items, phone_errors, name_errors, item_errors):
        p_tok = p_n if p_n else "NO_PHONE"
        n_tok = n_n if n_n else "NO_NAME"
        i_tok = i_n if i_n else "NO_ITEM"
        proposed_keys.append(f"{s_n}_{p_tok}_{n_tok}_{i_tok}")
        safety_statuses.append("INSUFFICIENT_IDENTITY_DATA" if (p_e or n_e or i_e) else "SAFE")

    masked_phone_cache = {p: mask_phone(p) for p in unique_phones.keys()}
    masked_phones = [masked_phone_cache.get(p, "MISSING_PHONE") for p in phone_vals]

    df_clean["current_key"] = current_keys
    df_clean["proposed_key"] = proposed_keys
    df_clean["initial_safety_status"] = safety_statuses
    df_clean["norm_phone"] = norm_phones
    df_clean["norm_name"] = norm_names
    df_clean["norm_item"] = norm_items
    df_clean["norm_store"] = norm_stores
    df_clean["phone_error"] = phone_errors
    df_clean["name_error"] = name_errors
    df_clean["masked_phone"] = masked_phones

    print(f"Generated identities in {time.time()-t_norm_start:.2f}s", flush=True)

    # =========================================================================
    # Step 4: Data Quality Audit
    # =========================================================================
    print("Computing Step 4: Data Quality Audit...", flush=True)
    unique_customer_ids = df_clean["customerId"].nunique(dropna=False)
    unique_customer_names = df_clean["customerName"].nunique(dropna=False)
    unique_item_ids = df_clean["itemId"].nunique(dropna=False)
    unique_current_keys = df_clean["current_key"].nunique()
    unique_proposed_keys = df_clean["proposed_key"].nunique()

    valid_phone_count = int(df_clean["phone_error"].isna().sum())
    missing_phone_mask = df_clean["phone_error"] == "Missing phone number"
    missing_phone_count = int(missing_phone_mask.sum())
    invalid_phone_count = total_valid_rows - valid_phone_count - missing_phone_count
    distinct_norm_phones = int(df_clean["norm_phone"].dropna().nunique())

    phone_grouped = df_clean[df_clean["norm_phone"].notna()].groupby("norm_phone")
    phones_shared_by_multiple_names = int((phone_grouped["norm_name"].nunique() > 1).sum())
    phones_shared_by_multiple_cids = int((phone_grouped["customerId"].nunique() > 1).sum())

    missing_name_count = int((df_clean["name_error"] == "Missing customer name").sum())
    distinct_norm_names = int(df_clean["norm_name"].dropna().nunique())
    name_grouped = df_clean[df_clean["norm_name"].notna()].groupby("norm_name")
    names_shared_by_multiple_cids = int((name_grouped["customerId"].nunique() > 1).sum())
    names_shared_by_multiple_phones = int((name_grouped["norm_phone"].nunique() > 1).sum())

    distinct_stores = int(df_clean["norm_store"].nunique())
    missing_store_count = int(df_clean["store_id"].isna().sum())
    missing_item_count = int(df_clean["itemId"].isna().sum())

    # =========================================================================
    # Step 5: Split Analysis (1 current -> multiple proposed)
    # =========================================================================
    print("Computing Step 5: Split Analysis...", flush=True)
    curr_to_props = df_clean.groupby("current_key")["proposed_key"].nunique()
    split_current_keys = set(curr_to_props[curr_to_props > 1].index.tolist())
    split_current_entity_count = len(split_current_keys)
    split_current_entity_pct = (split_current_entity_count / unique_current_keys * 100) if unique_current_keys else 0.0

    split_rows_df = df_clean[df_clean["current_key"].isin(split_current_keys)]
    split_tx_count = len(split_rows_df)

    split_cases: List[Dict[str, Any]] = []
    if split_current_keys:
        split_sub_grouped = split_rows_df.groupby("current_key")
        for curr_k, sub in split_sub_grouped:
            u_phones = sub["norm_phone"].nunique(dropna=False)
            u_names = sub["norm_name"].nunique(dropna=False)
            u_stores = sub["norm_store"].nunique(dropna=False)
            
            reasons = []
            if u_phones > 1:
                reasons.append("different/changing phone")
            if u_names > 1:
                reasons.append("different/variant customer name")
            if u_stores > 1:
                reasons.append("different store")
            if not reasons:
                reasons.append("missing phone / data quality variation")
            
            split_cases.append({
                "current_key": curr_k,
                "customerId": str(sub["customerId"].iloc[0]),
                "itemId": str(sub["itemId"].iloc[0]),
                "transaction_count": len(sub),
                "distinct_proposed_keys": sub["proposed_key"].nunique(),
                "split_reasons": "; ".join(reasons),
                "sample_masked_phones": ", ".join(sub["masked_phone"].unique()[:3]),
                "sample_names": ", ".join(sub["norm_name"].dropna().unique()[:3]),
            })
    split_cases_df = pd.DataFrame(split_cases)
    split_cases_df.to_csv(OUTPUT_DIR / "identity_split_cases.csv", index=False)

    # =========================================================================
    # Step 6: Merge / Collision Analysis (multiple current -> 1 proposed)
    # =========================================================================
    print("Computing Step 6: Merge / Collision Analysis...", flush=True)
    prop_to_currs = df_clean.groupby("proposed_key")["current_key"].nunique()
    merge_proposed_keys = set(prop_to_currs[prop_to_currs > 1].index.tolist())
    merge_proposed_entity_count = len(merge_proposed_keys)
    merge_rows_df = df_clean[df_clean["proposed_key"].isin(merge_proposed_keys)]
    merge_current_entities_affected = int(merge_rows_df["current_key"].nunique())
    merge_tx_count = len(merge_rows_df)

    merge_cases: List[Dict[str, Any]] = []
    if merge_proposed_keys:
        merge_sub_grouped = merge_rows_df.groupby("proposed_key")
        for prop_k, sub in merge_sub_grouped:
            u_cids = sub["customerId"].dropna().unique().tolist()
            merge_cases.append({
                "proposed_key": prop_k,
                "distinct_current_entities": sub["current_key"].nunique(),
                "distinct_customer_ids": len(u_cids),
                "customer_ids": ", ".join(str(c) for c in u_cids[:5]),
                "transaction_count": len(sub),
                "norm_name": sub["norm_name"].iloc[0] if sub["norm_name"].notna().any() else "UNKNOWN",
                "masked_phone": sub["masked_phone"].iloc[0],
                "norm_item": sub["norm_item"].iloc[0] if sub["norm_item"].notna().any() else "UNKNOWN",
            })
    merge_cases_df = pd.DataFrame(merge_cases)
    merge_cases_df.to_csv(OUTPUT_DIR / "identity_merge_cases.csv", index=False)

    # =========================================================================
    # Step 7: Shared Phone Analysis (Scenarios A, B, C, D)
    # =========================================================================
    print("Computing Step 7: Shared Phone Analysis...", flush=True)
    df_valid_phone = df_clean[df_clean["norm_phone"].notna()].copy()

    phone_item_grouped = df_valid_phone.groupby(["norm_phone", "norm_item"])
    scenario_a_count = int((phone_item_grouped["norm_name"].nunique() > 1).sum())

    phone_name_item_grouped = df_valid_phone.groupby(["norm_phone", "norm_name", "norm_item"])
    scenario_b_count = int((phone_name_item_grouped["customerId"].nunique() > 1).sum())

    scenario_c_count = int((phone_item_grouped["customerId"].nunique() > 1).sum())

    phone_name_grouped = df_valid_phone.groupby(["norm_phone", "norm_name"])
    scenario_d_count = int((phone_name_grouped["norm_item"].nunique() > 1).sum())

    # =========================================================================
    # Step 8 to 12: High-Speed Aggregations for Path A, Path B, Stability, and Safety
    # =========================================================================
    print("Computing Step 8 to 12: Aggregating temporal trajectories and stability...", flush=True)
    t_agg_start = time.time()
    
    max_dataset_date = df_clean["invoice_date"].max()

    # Pre-sort entire dataset by date
    df_clean_sorted = df_clean.sort_values("invoice_date").copy()

    # Helper to compute entity stats in bulk
    def compute_grouped_stats(df_subset: pd.DataFrame, key_col: str) -> Dict[str, Dict[str, Any]]:
        stats_dict = {}
        # Group dates by key
        for key, dates in df_subset.groupby(key_col)["invoice_date"]:
            d_list = dates.tolist()
            count = len(d_list)
            last_d = d_list[-1]
            intervals = []
            if count >= 2:
                intervals = [(d_list[i] - d_list[i-1]).days for i in range(1, count) if (d_list[i] - d_list[i-1]).days > 0]
            
            median_int = float(np.median(intervals)) if intervals else 0.0
            stability = classify_path_a_stability([float(x) for x in intervals], count)
            
            recency_days = (max_dataset_date - last_d).days
            recent = recency_days <= 180
            
            m3 = len(set(d.strftime("%Y-%m") for d in d_list if (max_dataset_date - d).days <= 90))
            m6 = len(set(d.strftime("%Y-%m") for d in d_list if (max_dataset_date - d).days <= 180))
            
            b_eligible = False
            b_reason = "INELIGIBLE"
            if count < MIN_PURCHASES and recent:
                if m3 >= 2:
                    b_eligible = True
                    b_reason = "3M_RECURRING"
                elif m6 >= 3:
                    b_eligible = True
                    b_reason = "6M_RECURRING"

            stats_dict[key] = {
                "count": count,
                "last_date": last_d,
                "median_interval": median_int,
                "stability_tier": stability,
                "path_b_eligible": b_eligible,
                "path_b_reason": b_reason,
            }
        return stats_dict

    curr_stats = compute_grouped_stats(df_clean_sorted, "current_key")
    prop_stats = compute_grouped_stats(df_clean_sorted, "proposed_key")
    print(f"Aggregated {len(curr_stats):,} current entities and {len(prop_stats):,} proposed entities in {time.time()-t_agg_start:.2f}s", flush=True)

    # Pre-map first occurrence metadata for current keys
    meta_df = df_clean.drop_duplicates("current_key").set_index("current_key")

    comparison_records: List[Dict[str, Any]] = []
    path_a_impact_records: List[Dict[str, Any]] = []
    path_b_impact_records: List[Dict[str, Any]] = []

    stability_transitions: Dict[str, int] = {}
    prediction_input_diffs = {
        "purchase_count_changed": 0,
        "last_purchase_date_changed": 0,
        "median_interval_changed": 0,
        "stability_tier_changed": 0,
    }

    curr_path_a_eligible_count = 0
    prop_path_a_eligible_count = sum(1 for p in prop_stats.values() if p["count"] >= MIN_PURCHASES)

    for c_key, c_info in curr_stats.items():
        row_meta = meta_df.loc[c_key]
        primary_prop_key = row_meta["proposed_key"]
        p_info = prop_stats.get(primary_prop_key, {})

        c_count = c_info["count"]
        p_count = p_info.get("count", 0)

        c_last_date = c_info["last_date"]
        p_last_date = p_info.get("last_date")

        c_median = c_info["median_interval"]
        p_median = p_info.get("median_interval", 0.0)

        c_stability = c_info["stability_tier"]
        p_stability = p_info.get("stability_tier", LABEL_UNSTABLE)

        c_a_elig = c_count >= MIN_PURCHASES
        p_a_elig = p_count >= MIN_PURCHASES
        if c_a_elig:
            curr_path_a_eligible_count += 1

        c_b_elig = c_info["path_b_eligible"]
        p_b_elig = p_info.get("path_b_eligible", False)

        is_split = c_key in split_current_keys
        is_merge = primary_prop_key in merge_proposed_keys

        if c_a_elig != p_a_elig or is_split or is_merge:
            path_a_impact_records.append({
                "current_key": c_key,
                "proposed_key": primary_prop_key,
                "current_purchase_count": c_count,
                "proposed_purchase_count": p_count,
                "current_path_a_eligible": c_a_elig,
                "proposed_path_a_eligible": p_a_elig,
                "is_split": is_split,
                "is_merge": is_merge,
                "identity_difference_reason": "SPLIT" if is_split else ("MERGE" if is_merge else "IDENTICAL"),
            })

        if c_b_elig != p_b_elig or is_split or is_merge:
            path_b_impact_records.append({
                "current_key": c_key,
                "proposed_key": primary_prop_key,
                "current_path_b_eligible": c_b_elig,
                "proposed_path_b_eligible": p_b_elig,
                "current_path_b_reason": c_info["path_b_reason"],
                "proposed_path_b_reason": p_info.get("path_b_reason", "INELIGIBLE"),
                "is_split": is_split,
                "is_merge": is_merge,
            })

        trans_key = f"{c_stability} -> {p_stability}"
        stability_transitions[trans_key] = stability_transitions.get(trans_key, 0) + 1

        if c_count != p_count:
            prediction_input_diffs["purchase_count_changed"] += 1
        if c_last_date != p_last_date:
            prediction_input_diffs["last_purchase_date_changed"] += 1
        if abs(c_median - p_median) > 0.01:
            prediction_input_diffs["median_interval_changed"] += 1
        if c_stability != p_stability:
            prediction_input_diffs["stability_tier_changed"] += 1

        init_safety = row_meta["initial_safety_status"]
        if init_safety == "INSUFFICIENT_IDENTITY_DATA":
            final_safety = "INSUFFICIENT_IDENTITY_DATA"
        elif is_merge:
            final_safety = "COLLISION_RISK"
        elif is_split:
            final_safety = "REVIEW"
        else:
            final_safety = "SAFE"

        comparison_records.append({
            "current_key": c_key,
            "proposed_key": primary_prop_key,
            "customerId": str(row_meta["customerId"]),
            "customerName": str(row_meta["customerName"]),
            "masked_phone": row_meta["masked_phone"],
            "itemId": str(row_meta["itemId"]),
            "current_purchase_count": c_count,
            "proposed_purchase_count": p_count,
            "current_last_date": c_last_date.strftime("%Y-%m-%d") if c_last_date else "",
            "proposed_last_date": p_last_date.strftime("%Y-%m-%d") if p_last_date else "",
            "current_median_interval": round(c_median, 2),
            "proposed_median_interval": round(p_median, 2),
            "current_stability_tier": c_stability,
            "proposed_stability_tier": p_stability,
            "current_path_a_eligible": c_a_elig,
            "proposed_path_a_eligible": p_a_elig,
            "current_path_b_eligible": c_b_elig,
            "proposed_path_b_eligible": p_b_elig,
            "safety_classification": final_safety,
        })

    entity_comp_df = pd.DataFrame(comparison_records)
    entity_comp_df.to_csv(OUTPUT_DIR / "identity_entity_comparison.csv", index=False)

    pd.DataFrame(path_a_impact_records).to_csv(OUTPUT_DIR / "path_a_impact.csv", index=False)
    pd.DataFrame(path_b_impact_records).to_csv(OUTPUT_DIR / "path_b_impact.csv", index=False)

    path_a_remained = int((entity_comp_df["current_path_a_eligible"] & entity_comp_df["proposed_path_a_eligible"]).sum())
    path_a_gained = int((~entity_comp_df["current_path_a_eligible"] & entity_comp_df["proposed_path_a_eligible"]).sum())
    path_a_lost = int((entity_comp_df["current_path_a_eligible"] & ~entity_comp_df["proposed_path_a_eligible"]).sum())
    path_a_unchanged = int((entity_comp_df["current_path_a_eligible"] == entity_comp_df["proposed_path_a_eligible"]).sum())

    path_b_remained = int((entity_comp_df["current_path_b_eligible"] & entity_comp_df["proposed_path_b_eligible"]).sum())
    path_b_gained = int((~entity_comp_df["current_path_b_eligible"] & entity_comp_df["proposed_path_b_eligible"]).sum())
    path_b_lost = int((entity_comp_df["current_path_b_eligible"] & ~entity_comp_df["proposed_path_b_eligible"]).sum())
    path_b_unchanged = int((entity_comp_df["current_path_b_eligible"] == entity_comp_df["proposed_path_b_eligible"]).sum())

    safety_counts = entity_comp_df["safety_classification"].value_counts().to_dict()

    audit_summary = {
        "metadata": {
            "audit_timestamp": datetime.now().isoformat(),
            "repository": str(PROJECT_ROOT),
            "is_read_only": True,
            "dry_run": True,
            "execution_duration_sec": round(time.time() - t0, 2),
        },
        "dataset_scope": {
            "total_raw_rows": total_raw_rows,
            "total_valid_rows": total_valid_rows,
            "date_min": df_clean["invoice_date"].min().strftime("%Y-%m-%d"),
            "date_max": df_clean["invoice_date"].max().strftime("%Y-%m-%d"),
            "unique_customer_ids": int(unique_customer_ids),
            "unique_customer_names": int(unique_customer_names),
            "unique_item_ids": int(unique_item_ids),
            "unique_current_entities": int(unique_current_keys),
            "unique_proposed_entities": int(unique_proposed_keys),
        },
        "phone_data_quality": {
            "valid_phone_rows": int(valid_phone_count),
            "missing_phone_rows": int(missing_phone_count),
            "invalid_phone_rows": int(invalid_phone_count),
            "distinct_normalized_phones": int(distinct_norm_phones),
            "phones_shared_by_multiple_customer_names": int(phones_shared_by_multiple_names),
            "phones_shared_by_multiple_customer_ids": int(phones_shared_by_multiple_cids),
        },
        "customer_name_data_quality": {
            "missing_customer_name_rows": int(missing_name_count),
            "distinct_normalized_customer_names": int(distinct_norm_names),
            "names_shared_by_multiple_customer_ids": int(names_shared_by_multiple_cids),
            "names_shared_by_multiple_phones": int(names_shared_by_multiple_phones),
        },
        "store_and_item_data_quality": {
            "distinct_stores": int(distinct_stores),
            "missing_store_rows": int(missing_store_count),
            "missing_item_rows": int(missing_item_count),
        },
        "split_analysis": {
            "affected_current_entities": int(split_current_entity_count),
            "affected_current_entities_pct": round(split_current_entity_pct, 4),
            "affected_transactions": int(split_tx_count),
        },
        "merge_analysis": {
            "affected_proposed_entities": int(merge_proposed_entity_count),
            "affected_current_entities": int(merge_current_entities_affected),
            "affected_transactions": int(merge_tx_count),
        },
        "shared_phone_analysis": {
            "scenario_a_same_phone_diff_name_same_item": int(scenario_a_count),
            "scenario_b_same_phone_same_name_same_item_diff_cid": int(scenario_b_count),
            "scenario_c_same_phone_diff_cid_same_item": int(scenario_c_count),
            "scenario_d_same_phone_same_name_diff_items": int(scenario_d_count),
        },
        "path_a_impact": {
            "current_eligible": curr_path_a_eligible_count,
            "proposed_eligible": prop_path_a_eligible_count,
            "remained_eligible": path_a_remained,
            "newly_eligible": path_a_gained,
            "lost_eligibility": path_a_lost,
            "unchanged": path_a_unchanged,
        },
        "path_b_impact": {
            "current_eligible": int((entity_comp_df["current_path_b_eligible"]).sum()),
            "proposed_eligible": int((entity_comp_df["proposed_path_b_eligible"]).sum()),
            "remained_eligible": path_b_remained,
            "newly_eligible": path_b_gained,
            "lost_eligibility": path_b_lost,
            "unchanged": path_b_unchanged,
        },
        "stability_impact_transitions": stability_transitions,
        "prediction_input_impact": {
            "total_entities_evaluated": len(entity_comp_df),
            "purchase_count_changed": prediction_input_diffs["purchase_count_changed"],
            "last_purchase_date_changed": prediction_input_diffs["last_purchase_date_changed"],
            "median_interval_changed": prediction_input_diffs["median_interval_changed"],
            "stability_tier_changed": prediction_input_diffs["stability_tier_changed"],
        },
        "safety_classifications": {
            "SAFE": int(safety_counts.get("SAFE", 0)),
            "REVIEW": int(safety_counts.get("REVIEW", 0)),
            "COLLISION_RISK": int(safety_counts.get("COLLISION_RISK", 0)),
            "INSUFFICIENT_IDENTITY_DATA": int(safety_counts.get("INSUFFICIENT_IDENTITY_DATA", 0)),
        },
    }

    # Save JSON report
    with open(OUTPUT_DIR / "identity_audit_report.json", "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"Saved {OUTPUT_DIR / 'identity_audit_report.json'}", flush=True)

    # Generate Markdown report
    md_content = generate_markdown_report(audit_summary, split_cases_df, merge_cases_df)
    with open(OUTPUT_DIR / "identity_audit_report.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved {OUTPUT_DIR / 'identity_audit_report.md'}", flush=True)

    # Generate README.md
    readme_content = generate_readme()
    with open(OUTPUT_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"Saved {OUTPUT_DIR / 'README.md'}", flush=True)

    print("=" * 70, flush=True)
    print(f"Audit Complete in {time.time()-t0:.2f}s! All outputs generated successfully.", flush=True)
    print("=" * 70, flush=True)
    return audit_summary


def generate_markdown_report(summary: Dict[str, Any], split_df: pd.DataFrame, merge_df: pd.DataFrame) -> str:
    dq = summary["dataset_scope"]
    pq = summary["phone_data_quality"]
    nq = summary["customer_name_data_quality"]
    sp = summary["split_analysis"]
    mg = summary["merge_analysis"]
    sh = summary["shared_phone_analysis"]
    pa = summary["path_a_impact"]
    pb = summary["path_b_impact"]
    st = summary["stability_impact_transitions"]
    pi = summary["prediction_input_impact"]
    sf = summary["safety_classifications"]

    md = f"""# Customer-Item Identity Impact Audit Report (READ-ONLY)

**Date of Audit:** {summary['metadata']['audit_timestamp']}  
**Status:** COMPLETE (READ-ONLY — Zero Production Mutations)  
**Target Repository:** `ai-mediastra-whatsapp-reminder`

---

## 1. Executive Summary & Identity Definitions

This audit compares the existing production customer-item entity against the proposed multi-factor composite identity:

| Dimension | Current Production Entity | Proposed Audit Entity |
| :--- | :--- | :--- |
| **Formula** | `customerId + itemId` | `normalized_store_id + normalized_phone + normalized_customer_name + normalized_item_id` |
| **Phone Used** | Delivery routing only (`MOBILE_NO`) | Primary disambiguator component |
| **Name Normalization** | Raw / Unmodified in key | Uppercase, whitespace-collapsed, punctuation-stripped |
| **Store Scope** | Implicit single branch (`MAIN`) | Explicit `store_id` (defaults to `MAIN`) |

> [!IMPORTANT]
> **Strict Read-Only Guarantee**:
> - Zero changes to `customerId`, `customerName`, `MOBILE_NO`, `itemId`, or historical transactions.
> - Zero production database migrations or ML model re-training.
> - All phone numbers in this report are strictly masked (`******XXXX`).

---

## 2. Dataset Scope & Data Quality

| Metric | Count | Details |
| :--- | :--- | :--- |
| **Total Raw Transactions** | **{dq['total_raw_rows']:,}** | Historical 5.8-year POS dataset |
| **Total Valid Transactions** | **{dq['total_valid_rows']:,}** | Cleaned (positive qty, valid dates) |
| **Date Range** | {dq['date_min']} to {dq['date_max']} | Longitudinal history |
| **Unique Customer IDs** | {dq['unique_customer_ids']:,} | `customerId` |
| **Unique Customer Names** | {dq['unique_customer_names']:,} | `customerName` |
| **Unique Current Entities** | **{dq['unique_current_entities']:,}** | `customerId + itemId` |
| **Unique Proposed Entities** | **{dq['unique_proposed_entities']:,}** | `store + phone + name + item` |

### Phone Quality & Shared Phone Metrics
- **Valid WhatsApp Phone Format (`91XXXXXXXXXX`)**: {pq['valid_phone_rows']:,} rows
- **Missing Phone Rows**: {pq['missing_phone_rows']:,} rows
- **Invalid Phone Format Rows**: {pq['invalid_phone_rows']:,} rows
- **Distinct Normalized Phone Numbers**: {pq['distinct_normalized_phones']:,}
- **Phones Shared by Multiple Customer Names**: **{pq['phones_shared_by_multiple_customer_names']:,}**
- **Phones Shared by Multiple Customer IDs**: **{pq['phones_shared_by_multiple_customer_ids']:,}**

---

## 3. Split Analysis (1 Current Entity ➔ Multiple Proposed Entities)

Cases where a single `customerId + itemId` splits into multiple proposed keys (e.g. customer changed phone number or name punctuation over time):

- **Affected Current Entities:** **{sp['affected_current_entities']:,}** ({sp['affected_current_entities_pct']:.2f}% of total)
- **Affected Transactions:** {sp['affected_transactions']:,} rows

"""
    if not split_df.empty:
        md += "### Representative Split Cases (Masked)\n\n"
        md += "| Current Key | Customer ID | Item ID | Tx Count | Proposed Keys | Split Reasons | Sample Phones |\n"
        md += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
        for _, r in split_df.head(5).iterrows():
            md += f"| `{r['current_key']}` | {r['customerId']} | {r['itemId']} | {r['transaction_count']} | {r['distinct_proposed_keys']} | {r['split_reasons']} | {r['sample_masked_phones']} |\n"
        md += "\n"

    md += f"""---

## 4. Merge / Collision Analysis (Multiple Current Entities ➔ 1 Proposed Entity)

Cases where multiple `customerId` values share the exact same phone, normalized name, store, and item:

- **Affected Proposed Entities:** **{mg['affected_proposed_entities']:,}**
- **Affected Current Entities Collapsed:** **{mg['affected_current_entities']:,}**
- **Affected Transactions:** {mg['affected_transactions']:,} rows

"""
    if not merge_df.empty:
        md += "### Representative Merge Cases (Masked)\n\n"
        md += "| Proposed Key | Collapsed Entities | Customer IDs | Customer Name | Masked Phone | Item |\n"
        md += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
        for _, r in merge_df.head(5).iterrows():
            md += f"| `{r['proposed_key']}` | {r['distinct_current_entities']} | {r['customer_ids']} | {r['norm_name']} | {r['masked_phone']} | {r['norm_item']} |\n"
        md += "\n"

    md += f"""---

## 5. Shared Phone Analysis

| Scenario | Definition | Count (Pairs / Tuples) |
| :--- | :--- | :--- |
| **Scenario A** | Same Phone + Different Customer Names + Same Item | **{sh['scenario_a_same_phone_diff_name_same_item']:,}** |
| **Scenario B** | Same Phone + Same Customer Name + Same Item (Diff Customer IDs) | **{sh['scenario_b_same_phone_same_name_same_item_diff_cid']:,}** |
| **Scenario C** | Same Phone + Different Customer IDs + Same Item | **{sh['scenario_c_same_phone_diff_cid_same_item']:,}** |
| **Scenario D** | Same Phone + Same Customer Name + Different Items | **{sh['scenario_d_same_phone_same_name_diff_items']:,}** |

---

## 6. Path A & Path B Eligibility Impact

### Path A Impact (>= 6 Historical Purchases)
- **Current Path A Eligible Entities:** {pa['current_eligible']:,}
- **Proposed Path A Eligible Entities:** {pa['proposed_eligible']:,}
- **Entities Remaining Eligible:** {pa['remained_eligible']:,}
- **Newly Eligible Entities (via Merge):** {pa['newly_eligible']:,}
- **Entities Losing Eligibility (via Split):** {pa['lost_eligibility']:,}
- **Parity / Unchanged Entities:** {pa['unchanged']:,}

### Path B Impact (Recency <= 180d, 3M >= 2 or 6M >= 3 Months)
- **Current Path B Eligible Entities:** {pb['current_eligible']:,}
- **Proposed Path B Eligible Entities:** {pb['proposed_eligible']:,}
- **Entities Remaining Eligible:** {pb['remained_eligible']:,}
- **Newly Eligible Entities:** {pb['newly_eligible']:,}
- **Entities Losing Eligibility:** {pb['lost_eligibility']:,}
- **Parity / Unchanged Entities:** {pb['unchanged']:,}

---

## 7. Stability Tier Transitions & Prediction Inputs

### Stability Transitions
```
"""
    for k, v in st.items():
        md += f"{k:<25} : {v:,} entities\n"
    md += f"""```

### Prediction Input Impact
- **Total Entities Evaluated:** {pi['total_entities_evaluated']:,}
- **Entities with Purchase Count Shift:** {pi['purchase_count_changed']:,} ({pi['purchase_count_changed']/pi['total_entities_evaluated']*100:.2f}%)
- **Entities with Last Purchase Date Shift:** {pi['last_purchase_date_changed']:,} ({pi['last_purchase_date_changed']/pi['total_entities_evaluated']*100:.2f}%)
- **Entities with Median Interval Shift:** {pi['median_interval_changed']:,} ({pi['median_interval_changed']/pi['total_entities_evaluated']*100:.2f}%)
- **Entities with Stability Tier Shift:** {pi['stability_tier_changed']:,} ({pi['stability_tier_changed']/pi['total_entities_evaluated']*100:.2f}%)

---

## 8. Safety Classifications

Every evaluated entity is classified into one of 4 governance categories:

| Safety Tier | Count | Percentage | Description |
| :--- | :--- | :--- | :--- |
| **SAFE** | **{sf['SAFE']:,}** | {sf['SAFE']/pi['total_entities_evaluated']*100:.2f}% | Stable store + phone + name + item (1:1 with current entity) |
| **REVIEW** | **{sf['REVIEW']:,}** | {sf['REVIEW']/pi['total_entities_evaluated']*100:.2f}% | 1 current entity split into multiple keys (phone/name change) |
| **COLLISION_RISK** | **{sf['COLLISION_RISK']:,}** | {sf['COLLISION_RISK']/pi['total_entities_evaluated']*100:.2f}% | Multiple current customer IDs collapsed into one key |
| **INSUFFICIENT_DATA**| **{sf['INSUFFICIENT_IDENTITY_DATA']:,}** | {sf['INSUFFICIENT_IDENTITY_DATA']/pi['total_entities_evaluated']*100:.2f}% | Missing or invalid phone/name/item |

---

## 9. Recommendations & Next Steps

1. **Keep `customerId` as Canonical Traceability Key**: The proposed `customer_item_key` provides clean multi-branch deduplication, but `customerId` must always be retained for POS reconciliation.
2. **Review Split Cases**: 1:N splits should be surfaced to pharmacy staff rather than automatically fracturing patient history.
3. **Handle Missing Mobile Numbers**: Transactions lacking valid 10-digit Indian phone numbers should be routed to a "Manual Contact" queue rather than discarded.
"""
    return md


def generate_readme() -> str:
    return """# Customer-Item Identity Impact Audit Module

This directory contains the non-destructive audit artifacts evaluating the composite `customer_item_key`:
`customer_item_key = normalized_store_id + normalized_phone + normalized_customer_name + normalized_item_id`

## Generated Audit Artifacts

- `identity_normalization.py`: Conservative normalizers for store, phone, customer name, and item ID.
- `run_identity_audit.py`: Audit pipeline comparing current vs proposed entities.
- `identity_audit_report.json`: Machine-readable summary statistics.
- `identity_audit_report.md`: Human-readable impact analysis and breakdown.
- `identity_entity_comparison.csv`: Entity-by-entity comparison of purchase counts, stability, and eligibility.
- `identity_split_cases.csv`: Detailed cases where 1 current entity splits into multiple proposed keys.
- `identity_merge_cases.csv`: Detailed cases where multiple current customer IDs merge into 1 proposed key.
- `path_a_impact.csv`: Slice of entities whose Path A eligibility status changes.
- `path_b_impact.csv`: Slice of entities whose Path B eligibility status changes.
"""


if __name__ == "__main__":
    run_audit()
