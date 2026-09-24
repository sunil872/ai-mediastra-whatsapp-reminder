"""Read-only long-interval analysis for RefillCare Phase 17D.

Does NOT modify production data/model. Writes JSON results only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import joblib

from refillcare.models.training import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COL

PROC = ROOT / "data" / "refillcare" / "processed"
MODEL_PATH = PROC / "models" / "refill_model.joblib"

BUCKETS = [
    ("0-14", 0, 14),
    ("15-30", 15, 30),
    ("31-45", 31, 45),
    ("46-60", 46, 60),
    ("61-90", 61, 90),
    ("91-120", 91, 120),
    ("121-180", 121, 180),
    (">180", 181, None),
]


def _bucket_label(days: float) -> str:
    if days > 180:
        return ">180"
    for name, lo, hi in BUCKETS:
        if hi is None:
            continue
        if lo <= days <= hi:
            return name
    return "other"


def main() -> None:
    model_mtime_before = MODEL_PATH.stat().st_mtime

    cols_wanted = list(dict.fromkeys(
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + [
            TARGET_COL,
            "invoice_date",
            "customerId",
            "itemId",
            "purchase_count_so_far",
            "is_recurring_history",
            "historical_interval_median",
            "historical_interval_std",
            "historical_interval_cv",
            "days_since_previous_purchase",
            "split_set",
        ]
    ))

    print("Loading supervised dataset...", flush=True)
    # Prefer full supervised training_dataset; fall back to concat of splits
    path = PROC / "training_dataset.parquet"
    sample = pd.read_parquet(path, columns=None)
    cols = [c for c in cols_wanted if c in sample.columns]
    df = sample[cols].copy()
    del sample

    df = df[df[TARGET_COL].notna() & (df[TARGET_COL] >= 0)].copy()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["bucket"] = df[TARGET_COL].apply(_bucket_label)
    data_max = df["invoice_date"].max()
    data_min = df["invoice_date"].min()
    # Days from this purchase to end of observed data window
    df["days_to_data_end"] = (data_max - df["invoice_date"]).dt.days

    total = len(df)
    print(f"Supervised rows: {total:,} | date span {data_min.date()} -> {data_max.date()}", flush=True)

    # Recurring long-interval pattern: same customer-item has >=2 targets in same bucket
    print("Computing recurring-vs-isolated flags...", flush=True)
    pair_bucket_counts = (
        df.groupby(["customerId", "itemId", "bucket"], sort=False)[TARGET_COL]
        .size()
        .rename("bucket_events_for_pair")
        .reset_index()
    )
    df = df.merge(pair_bucket_counts, on=["customerId", "itemId", "bucket"], how="left")
    df["interval_pattern"] = np.where(
        df["bucket_events_for_pair"] >= 2,
        "recurring_same_bucket",
        "isolated_in_bucket",
    )

    # Production model predictions (read-only)
    print("Scoring with production model (read-only)...", flush=True)
    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipeline"]
    feat_cols = list(bundle.get("features_numeric", NUMERIC_FEATURES)) + list(
        bundle.get("features_categorical", CATEGORICAL_FEATURES)
    )
    feat_cols = [c for c in feat_cols if c in df.columns]
    for c in feat_cols:
        if c not in df.columns:
            df[c] = np.nan
    preds = pipe.predict(df[feat_cols])
    preds = np.clip(np.asarray(preds, dtype=np.float64), 1.0, None)
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    abs_err = np.abs(preds - y)
    signed_err = preds - y
    df["pred_days"] = preds
    df["abs_error"] = abs_err
    df["signed_error"] = signed_err

    # Also baseline: historical median when available else global median
    global_med = float(np.nanmedian(y))
    hist_med = df["historical_interval_median"].to_numpy(dtype=np.float64) if "historical_interval_median" in df.columns else np.full(len(df), np.nan)
    baseline_pred = np.where(np.isnan(hist_med), global_med, hist_med)
    baseline_pred = np.clip(baseline_pred, 1.0, None)
    df["baseline_pred"] = baseline_pred
    df["baseline_abs_error"] = np.abs(baseline_pred - y)

    bucket_reports = {}
    for name, lo, hi in BUCKETS:
        sub = df[df["bucket"] == name]
        n = len(sub)
        if n == 0:
            bucket_reports[name] = {"count": 0, "pct": 0.0}
            continue

        seq = sub["purchase_count_so_far"]
        recurring_flag = sub["is_recurring_history"].fillna(0).astype(int)
        pattern = sub["interval_pattern"]

        # Consistency with prior personal median (when available)
        has_hist = sub["historical_interval_median"].notna()
        if has_hist.any():
            ratio = (sub.loc[has_hist, TARGET_COL] / sub.loc[has_hist, "historical_interval_median"]).replace([np.inf, -np.inf], np.nan)
            consistent_with_hist = float(((ratio >= 0.5) & (ratio <= 2.0)).mean() * 100)
            median_ratio = float(ratio.median()) if len(ratio.dropna()) else None
        else:
            consistent_with_hist = None
            median_ratio = None

        # Right-truncation pressure: share of events where target could not exceed remaining window
        # If target > days_to_data_end, impossible — so for observed targets, days_to_data_end >= target
        # Measure how close purchases sit to the end relative to bucket length
        near_end = sub["days_to_data_end"] <= (hi if hi is not None else 365)
        # For >180 use 365 as probe

        bucket_reports[name] = {
            "range_days": [lo, hi if hi is not None else None],
            "count": int(n),
            "pct": round(100.0 * n / total, 2),
            "target_mean": round(float(sub[TARGET_COL].mean()), 2),
            "target_median": round(float(sub[TARGET_COL].median()), 2),
            "mae_production_model": round(float(sub["abs_error"].mean()), 2),
            "medae_production_model": round(float(sub["abs_error"].median()), 2),
            "mean_signed_error_pred_minus_actual": round(float(sub["signed_error"].mean()), 2),
            "mae_historical_median_baseline": round(float(sub["baseline_abs_error"].mean()), 2),
            "history_depth": {
                "mean_purchase_count_so_far": round(float(seq.mean()), 2),
                "median_purchase_count_so_far": round(float(seq.median()), 2),
                "pct_2_or_fewer": round(float((seq <= 2).mean() * 100), 2),
                "pct_3_to_5": round(float(((seq >= 3) & (seq <= 5)).mean() * 100), 2),
                "pct_gt_5": round(float((seq > 5).mean() * 100), 2),
            },
            "recurring_vs_isolated": {
                "pct_is_recurring_history_flag": round(float(recurring_flag.mean() * 100), 2),
                "pct_recurring_same_bucket_for_customer_item": round(float((pattern == "recurring_same_bucket").mean() * 100), 2),
                "pct_isolated_in_bucket": round(float((pattern == "isolated_in_bucket").mean() * 100), 2),
                "pct_target_within_0.5_2x_personal_hist_median": consistent_with_hist,
                "median_target_over_hist_median_ratio": round(median_ratio, 3) if median_ratio is not None else None,
            },
            "window_position": {
                "mean_days_to_data_end": round(float(sub["days_to_data_end"].mean()), 1),
                "median_days_to_data_end": round(float(sub["days_to_data_end"].median()), 1),
                "pct_invoice_in_last_180d_of_dataset": round(float((sub["days_to_data_end"] <= 180).mean() * 100), 2),
            },
            "pred_distribution": {
                "mean": round(float(sub["pred_days"].mean()), 2),
                "median": round(float(sub["pred_days"].median()), 2),
                "p95": round(float(sub["pred_days"].quantile(0.95)), 2),
            },
        }

    # Cross-bucket interpretation helpers
    long_mask = df["bucket"].isin(["91-120", "121-180", ">180"])
    short_reg = df["bucket"].isin(["15-30", "31-45"])
    long_df = df[long_mask]
    short_df = df[short_reg]

    interpretation = {
        "data_span": {"min": str(data_min.date()), "max": str(data_max.date())},
        "supervised_rows": int(total),
        "long_interval_share_gt_90d_pct": round(float(long_mask.mean() * 100), 2),
        "chronic_like_15_45_share_pct": round(float(short_reg.mean() * 100), 2),
        "long_vs_short_history_depth": {
            "long_mean_purchase_count": round(float(long_df["purchase_count_so_far"].mean()), 2) if len(long_df) else None,
            "short_15_45_mean_purchase_count": round(float(short_df["purchase_count_so_far"].mean()), 2) if len(short_df) else None,
        },
        "long_recurring_same_bucket_pct": round(float((long_df["interval_pattern"] == "recurring_same_bucket").mean() * 100), 2) if len(long_df) else None,
        "long_is_recurring_history_flag_pct": round(float(long_df["is_recurring_history"].fillna(0).mean() * 100), 2) if len(long_df) else None,
        "production_model_mae_long_gt_90": round(float(long_df["abs_error"].mean()), 2) if len(long_df) else None,
        "production_model_mae_15_45": round(float(short_df["abs_error"].mean()), 2) if len(short_df) else None,
        "production_signed_error_long_gt_90": round(float(long_df["signed_error"].mean()), 2) if len(long_df) else None,
        "notes": [
            "Observed targets require both purchases in-window; ultra-long gaps near dataset end are under-represented (right-truncation).",
            "Censoring primarily affects last purchases (no target), not rows that already have a measured next-purchase interval.",
            "Isolated long gaps with shallow history are more likely irregular purchases or stockpiling/travel gaps than stable chronic refill cadence.",
            "Recurring same-bucket long intervals with deeper history can be genuine low-frequency refill behavior (e.g., 90–120 day packs).",
        ],
    }

    # Qualitative classification rates for long buckets
    def classify_row(row) -> str:
        t = row[TARGET_COL]
        hist = row.get("historical_interval_median", np.nan)
        recurring_flag = int(row.get("is_recurring_history", 0) or 0)
        pattern = row.get("interval_pattern")
        depth = row.get("purchase_count_so_far", 1)
        if pattern == "recurring_same_bucket" and depth >= 3 and pd.notna(hist) and hist > 0 and 0.5 <= t / hist <= 2.0:
            return "likely_genuine_low_frequency_refill"
        if depth <= 2 and pattern == "isolated_in_bucket":
            return "likely_irregular_or_cold_start_gap"
        if pd.notna(hist) and hist > 0 and t > 2.5 * hist and depth >= 3:
            return "likely_irregular_break_from_personal_cadence"
        if row.get("days_to_data_end", 9999) < t + 30:
            return "possible_edge_of_window_selection_effect"
        return "mixed_or_unclear"

    if len(long_df):
        classes = long_df.apply(classify_row, axis=1)
        interpretation["long_interval_heuristic_classes_pct"] = {
            k: round(float((classes == k).mean() * 100), 2) for k in sorted(classes.unique())
        }

    out = {
        "experiment": "PHASE_17D_LONG_INTERVAL_ANALYSIS",
        "production_modified": False,
        "model_mtime_unchanged": MODEL_PATH.stat().st_mtime == model_mtime_before,
        "buckets": bucket_reports,
        "interpretation": interpretation,
    }
    out_path = PROC / "phase17d_long_interval_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)
    print(json.dumps({k: {"count": v.get("count"), "pct": v.get("pct"), "mae": v.get("mae_production_model")} for k, v in bucket_reports.items()}, indent=2))


if __name__ == "__main__":
    main()
