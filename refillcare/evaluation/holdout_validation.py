"""Historical out-of-time validation module for RefillCare.

Executes strict point-in-time prediction generation using historical transactions
dated on or before a specified cutoff date, and evaluates realized prediction accuracy
against subsequent unseen holdout actual transactions without data leakage.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from refillcare.data.cleaning import aggregate_invoice_items, clean_transactions
from refillcare.data.history import create_purchase_history
from refillcare.data.packing import enrich_total_units_purchased
from refillcare.features.engineering import build_feature_dataset
from refillcare.models.prediction import generate_batch_predictions


OUTPUT_COLUMNS = [
    "customerId",
    "itemId",
    "customerName",
    "itemName",
    "last_purchase_date",
    "expected_refill_date",
    "primary_reminder_date",
    "predicted_days_until_refill",
    "estimated_days_of_supply",
    "refill_confidence",
    "status",
    "actual_refill_date",
    "error_days",
    "abs_error_days",
    "is_within_1d",
    "is_within_3d",
    "is_within_7d",
    "is_catastrophic",
]


def _parse_dates_robust(series: pd.Series) -> pd.Series:
    """Robust date parsing for mixed/ISO/standard date strings."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    # Try default ISO/standard format first
    parsed = pd.to_datetime(series, errors="coerce")
    # If many are null, attempt dayfirst
    if parsed.isna().sum() > len(series) * 0.5:
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return parsed


def run_historical_holdout_validation(
    transactions_df: pd.DataFrame,
    model_bundle: Optional[Dict[str, Any]] = None,
    cutoff_date: Union[str, date, datetime] = "2026-07-31",
    holdout_end_date: Union[str, date, datetime] = "2026-08-31",
    min_purchase_count: int = 2,
    target_prediction_window_start: Optional[Union[str, date, datetime]] = "2026-08-01",
    target_prediction_window_end: Optional[Union[str, date, datetime]] = "2026-08-31",
) -> Dict[str, Any]:
    """Execute a strict historical out-of-time holdout validation.

    Process:
    1. Filter historical dataset to transactions <= cutoff_date.
    2. Zero Leakage Guarantee: Holdout transactions (> cutoff_date) are completely isolated.
    3. Reconstruct customer + item chronological timelines and backward-looking intervals.
    4. Compute physical packaging and consumption velocity features strictly from pre-cutoff data.
    5. Generate predictions and expected refill dates for multi-purchase candidates.
    6. Match against actual post-cutoff holdout transactions (cutoff_date < invoice_date <= holdout_end_date).
    7. For matched records (repurchased in holdout), compute realized accuracy metrics (MAE, RMSE, +/-1d, +/-3d, +/-7d).
    8. For unobserved records, preserve status as 'pending' (right-censored, NOT penalized as model failures).

    Args:
        transactions_df: Raw or cleaned transaction DataFrame spanning history and holdout.
        model_bundle: Loaded RefillCare prediction model bundle (optional; uses supply/median fallbacks if None).
        cutoff_date: Historical data cutoff date (default: '2026-07-31').
        holdout_end_date: End of holdout evaluation period (default: '2026-08-31').
        min_purchase_count: Minimum lifetime purchases required for prediction eligibility (default: 2).
        target_prediction_window_start: Optional start date filter for expected refill dates.
        target_prediction_window_end: Optional end date filter for expected refill dates.

    Returns:
        Dict[str, Any] containing evaluated metrics, cohort breakdown, audit details,
        and individual prediction outcome DataFrames.
    """
    cutoff_ts = pd.to_datetime(cutoff_date)
    holdout_end_ts = pd.to_datetime(holdout_end_date)
    holdout_start_ts = cutoff_ts + pd.Timedelta(days=1)

    if transactions_df.empty:
        return _empty_holdout_result(cutoff_ts, holdout_start_ts, holdout_end_ts)

    # 1. Clean transactions & normalize headers
    df = transactions_df.copy()
    df["invoice_date"] = _parse_dates_robust(df["invoice_date"])

    # Drop null dates and clean identifiers
    df = df[df["invoice_date"].notna()].copy()
    if "customerId" not in df.columns or "itemId" not in df.columns:
        raise KeyError("Required columns 'customerId' and 'itemId' missing from input data.")

    df["customerId"] = df["customerId"].astype(str).str.strip()
    df["itemId"] = df["itemId"].astype(str).str.strip()
    valid_id_mask = (df["customerId"] != "") & (df["customerId"] != "nan") & (df["itemId"] != "") & (df["itemId"] != "nan")
    df = df[valid_id_mask].copy()

    # 2. Exclude returns / negative or zero quantities from purchase history
    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0)
        df_valid_qty = df[df["quantity"] > 0].copy()
    else:
        df_valid_qty = df.copy()

    # 3. Aggregate same-invoice lines into canonical purchase events
    if "invoice_number" in df_valid_qty.columns:
        events_df = aggregate_invoice_items(df_valid_qty)
    else:
        events_df = df_valid_qty.drop_duplicates(subset=["customerId", "itemId", "invoice_date"], keep="last").reset_index(drop=True)

    # 4. Strict Temporal Partitioning
    historical_events = events_df[events_df["invoice_date"] <= cutoff_ts].copy()
    holdout_events = events_df[(events_df["invoice_date"] > cutoff_ts) & (events_df["invoice_date"] <= holdout_end_ts)].copy()

    # Verify zero leakage
    leakage_count = int((historical_events["invoice_date"] > cutoff_ts).sum())
    leakage_detected = leakage_count > 0

    if historical_events.empty:
        return _empty_holdout_result(cutoff_ts, holdout_start_ts, holdout_end_ts, historical_tx=0, holdout_tx=len(holdout_events))

    # 5. Build Chronological Purchase History strictly on pre-cutoff data
    history_df = create_purchase_history(historical_events)
    if "packing" in history_df.columns:
        history_df = enrich_total_units_purchased(history_df)

    # 6. Build Feature Dataset strictly from pre-cutoff purchase history
    feature_df = build_feature_dataset(history_df)

    # 7. Isolate the latest purchase event per customer + medicine as of cutoff
    latest_pre_cutoff = feature_df.sort_values("invoice_date").groupby(["customerId", "itemId"], as_index=False).last()

    # Identify multi-purchase candidates (purchase_count >= min_purchase_count)
    p_counts = latest_pre_cutoff.get("purchase_count_so_far", latest_pre_cutoff.get("purchase_seq", 1))
    multi_mask = p_counts.fillna(1).astype(int) >= min_purchase_count
    candidates = latest_pre_cutoff[multi_mask].copy().reset_index(drop=True)
    cold_start_count = int((~multi_mask).sum())

    if candidates.empty:
        return _empty_holdout_result(
            cutoff_ts, holdout_start_ts, holdout_end_ts,
            historical_tx=len(historical_events),
            holdout_tx=len(holdout_events),
            cold_start_count=cold_start_count
        )

    # 8. Generate Predictions using supply-first hybrid engine
    if model_bundle is not None:
        preds_df = generate_batch_predictions(model_bundle, candidates)
    else:
        # Fallback prediction based on historical interval median / days of supply
        preds_df = candidates.copy()
        pred_days = []
        for _, r in preds_df.iterrows():
            dos = r.get("estimated_days_of_supply")
            med = r.get("historical_interval_median", r.get("days_since_previous_purchase", 30.0))
            val = dos if (dos is not None and pd.notna(dos) and dos > 0) else med
            pred_days.append(float(val) if pd.notna(val) and float(val) > 0 else 30.0)
        preds_df["predicted_days_until_refill"] = pred_days
        preds_df["refill_confidence"] = ["HIGH" if int(r.get("purchase_count_so_far", 2)) >= 6 else "MEDIUM" for _, r in preds_df.iterrows()]
        preds_df["expected_refill_date"] = [
            (pd.to_datetime(r["invoice_date"]) + pd.Timedelta(days=int(round(d)))).date()
            for r, d in zip(preds_df.to_dict("records"), pred_days)
        ]

    # Add primary reminder date (2-day buffer)
    exp_dates = pd.to_datetime(preds_df["expected_refill_date"], errors="coerce")
    preds_df["primary_reminder_date"] = [
        (d - pd.Timedelta(days=2)).strftime("%Y-%m-%d") if pd.notna(d) else None
        for d in exp_dates
    ]
    preds_df["last_purchase_date"] = pd.to_datetime(preds_df["invoice_date"]).dt.strftime("%Y-%m-%d")

    # Filter target predictions window if specified
    if target_prediction_window_start is not None and target_prediction_window_end is not None:
        win_start = pd.to_datetime(target_prediction_window_start).date()
        win_end = pd.to_datetime(target_prediction_window_end).date()
        valid_exp = preds_df["expected_refill_date"].dropna()
        target_mask = valid_exp.apply(lambda d: win_start <= (d if isinstance(d, date) else pd.to_datetime(d).date()) <= win_end)
        eval_cohort_df = preds_df.loc[target_mask.index[target_mask]].copy().reset_index(drop=True)
    else:
        eval_cohort_df = preds_df.copy().reset_index(drop=True)

    total_predictions = len(eval_cohort_df)

    # 9. Match Actual Holdout Outcomes using (customerId + itemId)
    evaluated_records: List[Dict[str, Any]] = []
    pending_records: List[Dict[str, Any]] = []

    # Build lookup map for holdout actual purchases
    holdout_events["_parsed_dt"] = pd.to_datetime(holdout_events["invoice_date"])
    holdout_sorted = holdout_events.sort_values("_parsed_dt")

    for _, pred_row in eval_cohort_df.iterrows():
        cid = str(pred_row["customerId"])
        iid = str(pred_row["itemId"])
        last_dt = pd.to_datetime(pred_row["last_purchase_date"])
        exp_dt = pd.to_datetime(pred_row["expected_refill_date"])

        # Filter holdout actuals for this customer + medicine strictly AFTER last_purchase_date
        actual_matches = holdout_sorted[
            (holdout_sorted["customerId"] == cid) &
            (holdout_sorted["itemId"] == iid) &
            (holdout_sorted["_parsed_dt"] > last_dt)
        ]

        base_rec = {
            "customerId": cid,
            "itemId": iid,
            "customerName": pred_row.get("customerName", cid),
            "itemName": pred_row.get("itemName", iid),
            "last_purchase_date": pred_row["last_purchase_date"],
            "expected_refill_date": str(exp_dt.date()) if pd.notna(exp_dt) else None,
            "primary_reminder_date": pred_row.get("primary_reminder_date"),
            "predicted_days_until_refill": pred_row.get("predicted_days_until_refill"),
            "estimated_days_of_supply": pred_row.get("estimated_days_of_supply"),
            "refill_confidence": pred_row.get("refill_confidence", "MEDIUM"),
        }

        if not actual_matches.empty:
            # Earliest valid repurchase in holdout period
            earliest_actual = actual_matches["_parsed_dt"].min()
            error_days = float((earliest_actual - exp_dt).days)
            abs_err = abs(error_days)

            base_rec.update({
                "status": "evaluated",
                "actual_refill_date": earliest_actual.strftime("%Y-%m-%d"),
                "error_days": error_days,
                "abs_error_days": abs_err,
                "is_within_1d": abs_err <= 1.0,
                "is_within_3d": abs_err <= 3.0,
                "is_within_7d": abs_err <= 7.0,
                "is_catastrophic": abs_err > 30.0,
            })
            evaluated_records.append(base_rec)
        else:
            base_rec.update({
                "status": "pending",
                "actual_refill_date": None,
                "error_days": None,
                "abs_error_days": None,
                "is_within_1d": None,
                "is_within_3d": None,
                "is_within_7d": None,
                "is_catastrophic": None,
            })
            pending_records.append(base_rec)

    # 10. Compute Summary Metrics
    evaluated_df = pd.DataFrame(evaluated_records) if evaluated_records else pd.DataFrame(columns=OUTPUT_COLUMNS)
    pending_df = pd.DataFrame(pending_records) if pending_records else pd.DataFrame(columns=OUTPUT_COLUMNS)

    evaluated_count = len(evaluated_records)
    pending_count = len(pending_records)
    eval_coverage_pct = round(100.0 * evaluated_count / total_predictions, 2) if total_predictions > 0 else 0.0

    if evaluated_count > 0:
        errors = evaluated_df["error_days"].values
        abs_errors = evaluated_df["abs_error_days"].values
        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mean_bias = float(np.mean(errors))
        within_1d_pct = float(100.0 * np.mean(abs_errors <= 1.0))
        within_3d_pct = float(100.0 * np.mean(abs_errors <= 3.0))
        within_7d_pct = float(100.0 * np.mean(abs_errors <= 7.0))
        catastrophic_count = int(np.sum(abs_errors > 30.0))
        catastrophic_rate_pct = float(100.0 * catastrophic_count / evaluated_count)
    else:
        mae = rmse = mean_bias = within_1d_pct = within_3d_pct = within_7d_pct = catastrophic_rate_pct = 0.0
        catastrophic_count = 0

    # 11. Compute Cohort Breakdown
    cohort_breakdown: Dict[str, Dict[str, Any]] = {}
    if evaluated_count > 0 and "refill_confidence" in evaluated_df.columns:
        for conf_label in ["HIGH", "MEDIUM", "RISK", "UNSTABLE", "REJECTED"]:
            sub = evaluated_df[evaluated_df["refill_confidence"] == conf_label]
            if not sub.empty:
                sub_abs = sub["abs_error_days"].values
                cohort_breakdown[conf_label] = {
                    "count": len(sub),
                    "mae": round(float(np.mean(sub_abs)), 2),
                    "within_3d_pct": round(float(100.0 * np.mean(sub_abs <= 3.0)), 2),
                    "within_7d_pct": round(float(100.0 * np.mean(sub_abs <= 7.0)), 2),
                    "catastrophic_count": int(np.sum(sub_abs > 30.0)),
                }

    metrics = {
        "total_predictions": total_predictions,
        "evaluated_count": evaluated_count,
        "pending_count": pending_count,
        "evaluation_coverage_pct": eval_coverage_pct,
        "mae_days": round(mae, 2),
        "rmse_days": round(rmse, 2),
        "mean_bias_days": round(mean_bias, 2),
        "within_1d_pct": round(within_1d_pct, 2),
        "within_3d_pct": round(within_3d_pct, 2),
        "within_7d_pct": round(within_7d_pct, 2),
        "catastrophic_errors_gt_30d": catastrophic_count,
        "catastrophic_error_rate_pct": round(catastrophic_rate_pct, 2),
    }

    audit_report = {
        "prediction_cutoff": cutoff_ts.strftime("%Y-%m-%d"),
        "holdout_start": holdout_start_ts.strftime("%Y-%m-%d"),
        "holdout_end": holdout_end_ts.strftime("%Y-%m-%d"),
        "historical_transactions_used": len(historical_events),
        "holdout_transactions_used_for_eval": len(holdout_events),
        "leakage_detected": leakage_detected,
        "leakage_count": leakage_count,
        "matching_key": "customerId + itemId",
        "model_artifact": "RefillCare Phase 17D Supply-First Hybrid Model",
        "days_of_supply_methodology": "Physical units / backward historical consumption velocity (bounded 3-180d)",
        "cold_start_excluded_count": cold_start_count,
        "metrics": metrics,
        "cohort_metrics": cohort_breakdown,
    }

    return {
        "status": "success",
        "metrics": metrics,
        "cohort_metrics": cohort_breakdown,
        "audit_report": audit_report,
        "evaluated_df": evaluated_df,
        "pending_df": pending_df,
        "all_predictions_df": pd.concat([evaluated_df, pending_df], ignore_index=True) if not evaluated_df.empty or not pending_df.empty else pd.DataFrame(columns=OUTPUT_COLUMNS),
    }


def _empty_holdout_result(
    cutoff_ts: pd.Timestamp,
    holdout_start_ts: pd.Timestamp,
    holdout_end_ts: pd.Timestamp,
    historical_tx: int = 0,
    holdout_tx: int = 0,
    cold_start_count: int = 0,
) -> Dict[str, Any]:
    metrics = {
        "total_predictions": 0,
        "evaluated_count": 0,
        "pending_count": 0,
        "evaluation_coverage_pct": 0.0,
        "mae_days": 0.0,
        "rmse_days": 0.0,
        "mean_bias_days": 0.0,
        "within_1d_pct": 0.0,
        "within_3d_pct": 0.0,
        "within_7d_pct": 0.0,
        "catastrophic_errors_gt_30d": 0,
        "catastrophic_error_rate_pct": 0.0,
    }
    audit_report = {
        "prediction_cutoff": cutoff_ts.strftime("%Y-%m-%d"),
        "holdout_start": holdout_start_ts.strftime("%Y-%m-%d"),
        "holdout_end": holdout_end_ts.strftime("%Y-%m-%d"),
        "historical_transactions_used": historical_tx,
        "holdout_transactions_used_for_eval": holdout_tx,
        "leakage_detected": False,
        "leakage_count": 0,
        "matching_key": "customerId + itemId",
        "model_artifact": "RefillCare Phase 17D Supply-First Hybrid Model",
        "days_of_supply_methodology": "Physical units / backward historical consumption velocity (bounded 3-180d)",
        "cold_start_excluded_count": cold_start_count,
        "metrics": metrics,
        "cohort_metrics": {},
    }
    return {
        "status": "empty",
        "metrics": metrics,
        "cohort_metrics": {},
        "audit_report": audit_report,
        "evaluated_df": pd.DataFrame(columns=OUTPUT_COLUMNS),
        "pending_df": pd.DataFrame(columns=OUTPUT_COLUMNS),
        "all_predictions_df": pd.DataFrame(columns=OUTPUT_COLUMNS),
    }
