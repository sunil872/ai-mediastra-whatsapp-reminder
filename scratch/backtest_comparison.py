"""Historical Backtest: Comparing Existing Refill Prediction Approach vs
Historical Consumption + Estimated Days of Supply Approach.

Read-only analysis script.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import math
import pandas as pd
import numpy as np
import joblib

from refillcare.data.packing import parse_pack_units, compute_total_units_purchased
from refillcare.features.consumption import (
    calculate_historical_consumption_rate,
    compute_expanding_consumption_rates,
    calculate_estimated_days_of_supply,
)
from refillcare.models.prediction import generate_batch_predictions
from refillcare.models.hybrid_strategy import (
    evaluate_hybrid_eligibility,
    personal_median_prediction,
    ROUTE_CORE,
    ROUTE_SECONDARY,
    ROUTE_REJECTED,
)

def compute_metrics(y_true, y_pred, total_eligible_cohort_size=None):
    mask = pd.notna(y_true) & pd.notna(y_pred)
    n_eval = int(mask.sum())
    total_cohort = total_eligible_cohort_size if total_eligible_cohort_size is not None else len(y_true)
    coverage = float(n_eval / total_cohort * 100.0) if total_cohort > 0 else 0.0
    
    if n_eval == 0:
        return {
            "evaluated_count": 0,
            "total_cohort_count": total_cohort,
            "coverage_pct": 0.0,
            "mae": None,
            "rmse": None,
            "acc_within_3d": None,
            "acc_within_7d": None,
            "large_error_gt14d_count": 0,
            "large_error_gt14d_pct": 0.0,
            "large_error_gt30d_count": 0,
            "large_error_gt30d_pct": 0.0,
        }
        
    err = np.abs(y_pred[mask] - y_true[mask])
    sq_err = (y_pred[mask] - y_true[mask]) ** 2
    
    mae = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(sq_err)))
    acc3 = float(np.mean(err <= 3.0) * 100.0)
    acc7 = float(np.mean(err <= 7.0) * 100.0)
    large14 = int(np.sum(err > 14.0))
    large14_pct = float(large14 / n_eval * 100.0)
    large30 = int(np.sum(err > 30.0))
    large30_pct = float(large30 / n_eval * 100.0)
    
    return {
        "evaluated_count": n_eval,
        "total_cohort_count": total_cohort,
        "coverage_pct": round(coverage, 2),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "acc_within_3d": round(acc3, 2),
        "acc_within_7d": round(acc7, 2),
        "large_error_gt14d_count": large14,
        "large_error_gt14d_pct": round(large14_pct, 2),
        "large_error_gt30d_count": large30,
        "large_error_gt30d_pct": round(large30_pct, 2),
    }

def run_backtest():
    print("Loading model bundle...", flush=True)
    bundle_path = Path("data/refillcare/processed/models/refill_model.joblib")
    model_bundle = joblib.load(bundle_path)
    
    print("Loading training_dataset.parquet...", flush=True)
    df = pd.read_parquet("data/refillcare/processed/training_dataset.parquet")
    print(f"Total supervised records: {len(df):,}", flush=True)
    
    # Sort chronologically by customerId, itemId, invoice_date, invoice_number
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df = df.sort_values(by=["customerId", "itemId", "invoice_date", "invoice_number"]).reset_index(drop=True)
    
    # 1. Compute total units purchased for each row
    print("Computing total units purchased...", flush=True)
    parsed_units = [compute_total_units_purchased(q, p) for q, p in zip(df["quantity"], df["packing"])]
    df["total_units_purchased"] = parsed_units
    
    # 2. Compute expanding historical consumption rate strictly using past purchases up to each event
    print("Computing expanding historical consumption rate...", flush=True)
    df["historical_consumption_rate"] = compute_expanding_consumption_rates(df)
    
    # 3. Compute Estimated Days of Supply for each row
    print("Computing estimated days of supply...", flush=True)
    estimated_supply_days = []
    estimated_supply_status = []
    
    for u, r in zip(df["total_units_purchased"], df["historical_consumption_rate"]):
        res = calculate_estimated_days_of_supply(u, r)
        estimated_supply_days.append(res["estimated_days_of_supply"])
        estimated_supply_status.append(res["status"])
        
    df["estimated_days_of_supply"] = pd.Series(estimated_supply_days, dtype=np.float64)
    df["supply_status"] = estimated_supply_status
    
    # Raw personal median baseline
    df["raw_personal_median"] = df["historical_interval_median"]
    
    target_col = "target_days_until_next_purchase"
    
    # Splits to evaluate
    splits = {
        "Test Set (Holdout 2026-07 to 2026-08)": df[df["split_set"] == "test"].copy(),
        "Validation Set (Holdout 2026-05 to 2026-06)": df[df["split_set"] == "validation"].copy(),
        "Full Supervised History (All Splits 2020 to 2026)": df.copy(),
    }
    
    results = {}
    
    for split_name, split_df in splits.items():
        print(f"\n=======================================================", flush=True)
        print(f"EVALUATING: {split_name} (Total rows: {len(split_df):,})", flush=True)
        print(f"=======================================================", flush=True)
        
        print("  Generating batch predictions for split...", flush=True)
        pred_df = generate_batch_predictions(model_bundle, split_df)
        split_df["existing_pred_days"] = pred_df["predicted_days_until_refill"]
        split_df["existing_pred_status"] = pred_df["prediction_status"]
        split_df["existing_pred_route"] = pred_df["prediction_route"]
        split_df["existing_rejection_reason"] = pred_df["rejection_reason"]
        
        split_res = {}
        
        # 1. Overall comparison on records where BOTH approaches are available
        both_mask = split_df["existing_pred_days"].notna() & split_df["estimated_days_of_supply"].notna()
        print(f"\n--- 1. OVERLAP SUBSET (Both Existing and Days of Supply Available: {both_mask.sum():,} rows) ---", flush=True)
        existing_metrics_overlap = compute_metrics(split_df.loc[both_mask, target_col], split_df.loc[both_mask, "existing_pred_days"], int(both_mask.sum()))
        supply_metrics_overlap = compute_metrics(split_df.loc[both_mask, target_col], split_df.loc[both_mask, "estimated_days_of_supply"], int(both_mask.sum()))
        
        print(f"Existing Refill Approach (Overlap): MAE={existing_metrics_overlap['mae']}, RMSE={existing_metrics_overlap['rmse']}, ±3d={existing_metrics_overlap['acc_within_3d']}%, ±7d={existing_metrics_overlap['acc_within_7d']}%, >14d={existing_metrics_overlap['large_error_gt14d_pct']}%", flush=True)
        print(f"Estimated Days of Supply (Overlap): MAE={supply_metrics_overlap['mae']}, RMSE={supply_metrics_overlap['rmse']}, ±3d={supply_metrics_overlap['acc_within_3d']}%, ±7d={supply_metrics_overlap['acc_within_7d']}%, >14d={supply_metrics_overlap['large_error_gt14d_pct']}%", flush=True)
        
        split_res["overlap_comparison"] = {
            "existing_refill_approach": existing_metrics_overlap,
            "estimated_days_of_supply": supply_metrics_overlap,
        }
        
        # 2. Standalone Coverage & Accuracy across full cohort
        print(f"\n--- 2. STANDALONE (Full Split Cohort: {len(split_df):,} rows) ---", flush=True)
        existing_metrics_full = compute_metrics(split_df[target_col], split_df["existing_pred_days"], len(split_df))
        supply_metrics_full = compute_metrics(split_df[target_col], split_df["estimated_days_of_supply"], len(split_df))
        median_metrics_full = compute_metrics(split_df[target_col], split_df["raw_personal_median"], len(split_df))
        
        print(f"Existing Refill Approach (Full): Cov={existing_metrics_full['coverage_pct']}%, MAE={existing_metrics_full['mae']}, RMSE={existing_metrics_full['rmse']}, ±3d={existing_metrics_full['acc_within_3d']}%, ±7d={existing_metrics_full['acc_within_7d']}%", flush=True)
        print(f"Estimated Days of Supply (Full): Cov={supply_metrics_full['coverage_pct']}%, MAE={supply_metrics_full['mae']}, RMSE={supply_metrics_full['rmse']}, ±3d={supply_metrics_full['acc_within_3d']}%, ±7d={supply_metrics_full['acc_within_7d']}%", flush=True)
        print(f"Raw Personal Median Baseline (Full): Cov={median_metrics_full['coverage_pct']}%, MAE={median_metrics_full['mae']}, RMSE={median_metrics_full['rmse']}, ±3d={median_metrics_full['acc_within_3d']}%, ±7d={median_metrics_full['acc_within_7d']}%", flush=True)
        
        split_res["full_split_comparison"] = {
            "existing_refill_approach": existing_metrics_full,
            "estimated_days_of_supply": supply_metrics_full,
            "raw_personal_median_baseline": median_metrics_full,
        }
        
        # 3. Sub-cohort Breakdowns:
        # A. Regular Purchase Patterns (is_core_regular or NormMAD <= 0.35 & Drift <= 7)
        # B. Bulk Purchases (quantity >= 1.5 * avg_historical_quantity and quantity > 1)
        # C. Partial / Top-up Purchases (quantity <= 0.65 * avg_historical_quantity and avg_historical_quantity >= 2)
        # D. Records with Missing/Ambiguous Packing
        
        regular_mask = (split_df["existing_pred_status"] == "eligible")
        bulk_mask = (split_df["purchase_count_so_far"] >= 2) & (split_df["quantity_vs_avg_ratio"] >= 1.5) & (split_df["quantity"] > 1)
        partial_mask = (split_df["purchase_count_so_far"] >= 2) & (split_df["quantity_vs_avg_ratio"] <= 0.65)
        missing_pack_mask = split_df["total_units_purchased"].isna() | (split_df["supply_status"] == "unavailable")
        
        cohorts = {
            "regular_purchase_patterns": regular_mask,
            "bulk_purchases": bulk_mask,
            "partial_topup_purchases": partial_mask,
            "missing_ambiguous_packing": missing_pack_mask,
        }
        
        split_res["sub_cohorts"] = {}
        print(f"\n--- 3. SUB-COHORTS ---", flush=True)
        for cname, cmask in cohorts.items():
            sub = split_df[cmask]
            ex_res = compute_metrics(sub[target_col], sub["existing_pred_days"], len(sub))
            sup_res = compute_metrics(sub[target_col], sub["estimated_days_of_supply"], len(sub))
            med_res = compute_metrics(sub[target_col], sub["raw_personal_median"], len(sub))
            
            print(f"Cohort: {cname} (Size: {len(sub):,} rows)", flush=True)
            print(f"  Existing Refill: Cov={ex_res['coverage_pct']}%, MAE={ex_res['mae']}, RMSE={ex_res['rmse']}, ±3d={ex_res['acc_within_3d']}%, ±7d={ex_res['acc_within_7d']}%, >14d={ex_res['large_error_gt14d_pct']}%", flush=True)
            print(f"  Estimated Days of Supply: Cov={sup_res['coverage_pct']}%, MAE={sup_res['mae']}, RMSE={sup_res['rmse']}, ±3d={sup_res['acc_within_3d']}%, ±7d={sup_res['acc_within_7d']}%, >14d={sup_res['large_error_gt14d_pct']}%", flush=True)
            print(f"  Raw Personal Median: Cov={med_res['coverage_pct']}%, MAE={med_res['mae']}, RMSE={med_res['rmse']}, ±3d={med_res['acc_within_3d']}%, ±7d={med_res['acc_within_7d']}%, >14d={med_res['large_error_gt14d_pct']}%", flush=True)
            
            split_res["sub_cohorts"][cname] = {
                "cohort_size": int(len(sub)),
                "existing_refill_approach": ex_res,
                "estimated_days_of_supply": sup_res,
                "raw_personal_median_baseline": med_res,
            }
            
        results[split_name] = split_res
        
    out_json = Path("scratch/backtest_results.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBacktest complete! Results saved to {out_json}", flush=True)

if __name__ == "__main__":
    run_backtest()
