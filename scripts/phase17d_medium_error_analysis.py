"""READ-ONLY Phase 17D MEDIUM-branch error analysis for hybrid_routing_v17d.

Analyzes only MEDIUM (secondary) predictions using production XGBoost.
Does NOT modify production code/model/data/reminder/WhatsApp. No retrain.
"""
from __future__ import annotations

import gc
import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from refillcare.models.training import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COL
from refillcare.models.hybrid_strategy import (
    CADENCE_MEDIAN_MIN,
    CADENCE_MEDIAN_MAX,
    CORE_NORM_MAD_MAX,
    CORE_DRIFT_MAX,
    BROAD_NORM_MAD_MAX,
    BROAD_DRIFT_MAX,
    MIN_PURCHASES,
    CONF_MEDIUM,
    resolve_regularity_fields,
)

PROC = ROOT / "data" / "refillcare" / "processed"
MODEL_PATH = PROC / "models" / "refill_model.joblib"
REPORT_PATH = PROC / "phase4_model_report.json"

_spec = importlib.util.spec_from_file_location(
    "phase17d_hybrid_comparison",
    ROOT / "scripts" / "phase17d_hybrid_comparison.py",
)
_hc = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_hc)
_attach_regularity_from_history = _hc._attach_regularity_from_history


def _is_medium(record: dict) -> bool:
    """Eligible secondary (MEDIUM) under production hybrid rules."""
    p = int(record.get("purchase_count_so_far", 1) or 1)
    if p < MIN_PURCHASES:
        return False
    reg = resolve_regularity_fields(record)
    hist = reg["historical_interval_median"]
    nm = reg["norm_mad"]
    dr = reg["cadence_drift"]
    if math.isnan(hist) or hist < CADENCE_MEDIAN_MIN or hist > CADENCE_MEDIAN_MAX:
        return False
    if math.isnan(nm) or math.isnan(dr):
        return False
    if nm > BROAD_NORM_MAD_MAX or dr > BROAD_DRIFT_MAX:
        return False
    is_core = (nm <= CORE_NORM_MAD_MAX) and (dr <= CORE_DRIFT_MAX)
    return not is_core


def _bucket_abs_error(ae: np.ndarray) -> np.ndarray:
    out = np.empty(len(ae), dtype=object)
    out[ae <= 3] = "0-3"
    out[(ae > 3) & (ae <= 7)] = "4-7"
    out[(ae > 7) & (ae <= 14)] = "8-14"
    out[(ae > 14) & (ae <= 30)] = "15-30"
    out[ae > 30] = ">30"
    return out


def _summarize_group(df: pd.DataFrame, key: str) -> dict:
    rows = {}
    for name, g in df.groupby(key, dropna=False):
        ae = g["abs_error"].to_numpy()
        bias = g["bias"].to_numpy()
        rows[str(name)] = {
            "count": int(len(g)),
            "pct": round(100.0 * len(g) / len(df), 2),
            "mae": round(float(ae.mean()), 2),
            "medae": round(float(np.median(ae)), 2),
            "mean_bias": round(float(bias.mean()), 2),
            "median_bias": round(float(np.median(bias)), 2),
            "pct_late_bias": round(100.0 * (bias > 0).mean(), 2),
            "pct_early_bias": round(100.0 * (bias < 0).mean(), 2),
            "within_3_days_pct": round(100.0 * (ae <= 3).mean(), 2),
            "within_7_days_pct": round(100.0 * (ae <= 7).mean(), 2),
            "mean_pred": round(float(g["pred"].mean()), 2),
            "mean_actual": round(float(g["actual"].mean()), 2),
            "mean_hist_median": round(float(g["hist_median"].mean()), 2),
            "mean_norm_mad": round(float(g["norm_mad"].mean()), 2),
            "mean_drift": round(float(g["drift"].mean()), 2),
            "mean_depth": round(float(g["depth"].mean()), 2),
        }
    return rows


def main() -> None:
    model_mtime_before = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    report_mtime_before = REPORT_PATH.stat().st_mtime if REPORT_PATH.exists() else None

    keep = list(
        dict.fromkeys(
            NUMERIC_FEATURES
            + CATEGORICAL_FEATURES
            + [
                TARGET_COL,
                "invoice_date",
                "purchase_count_so_far",
                "customerId",
                "itemId",
                "historical_interval_median",
                "days_since_previous_purchase",
            ]
        )
    )
    print("Loading test set...", flush=True)
    test = pd.read_parquet(PROC / "test.parquet")
    test = test[[c for c in keep if c in test.columns]]
    test = test[test[TARGET_COL] > 0].copy()

    history = pd.read_parquet(PROC / "purchase_history.parquet")
    test = _attach_regularity_from_history(test, history)
    test["historical_interval_norm_mad"] = test["norm_mad"]
    test["recent3_interval_median"] = test["recent3_median"]
    del history
    gc.collect()

    print("Scoring production XGBoost (read-only)...", flush=True)
    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipeline"]
    feats = list(bundle.get("features_numeric", NUMERIC_FEATURES)) + list(
        bundle.get("features_categorical", CATEGORICAL_FEATURES)
    )
    X = test.copy()
    for c in feats:
        if c not in X.columns:
            X[c] = np.nan
    xgb_pred = np.clip(np.asarray(pipe.predict(X[feats]), dtype=np.float64), 1.0, None)
    del bundle, pipe, X
    gc.collect()

    # Filter MEDIUM rows
    medium_mask = np.array([_is_medium(r.to_dict()) for _, r in test.iterrows()], dtype=bool)
    med = test.loc[medium_mask].copy()
    pred = xgb_pred[medium_mask]
    actual = med[TARGET_COL].to_numpy(dtype=np.float64)
    bias = pred - actual
    abs_err = np.abs(bias)

    med = med.reset_index(drop=True)
    med["pred"] = pred
    med["actual"] = actual
    med["bias"] = bias
    med["abs_error"] = abs_err
    med["depth"] = med["purchase_count_so_far"].astype(int)
    med["hist_median"] = med["historical_interval_median"].astype(float)
    med["norm_mad"] = med["norm_mad"].astype(float)
    med["drift"] = med["cadence_drift"].astype(float)
    med["abs_error_bucket"] = _bucket_abs_error(abs_err)

    # Dimension buckets
    med["depth_bucket"] = pd.cut(
        med["depth"],
        bins=[0, 6, 10, 20, 50, 10_000],
        labels=["6", "7-10", "11-20", "21-50", ">50"],
        right=True,
    ).astype(str)
    # Fix: P>=6 medium so depth starts at 6; use clearer labels
    depth_lab = np.full(len(med), "", dtype=object)
    depth_lab[med["depth"] <= 6] = "6"
    depth_lab[(med["depth"] >= 7) & (med["depth"] <= 10)] = "7-10"
    depth_lab[(med["depth"] >= 11) & (med["depth"] <= 20)] = "11-20"
    depth_lab[(med["depth"] >= 21) & (med["depth"] <= 50)] = "21-50"
    depth_lab[med["depth"] > 50] = ">50"
    med["depth_bucket"] = depth_lab

    hist_lab = np.full(len(med), "", dtype=object)
    hm = med["hist_median"].to_numpy()
    hist_lab[hm < 20] = "<20"
    hist_lab[(hm >= 20) & (hm < 30)] = "20-29"
    hist_lab[(hm >= 30) & (hm < 45)] = "30-44"
    hist_lab[(hm >= 45) & (hm < 60)] = "45-59"
    hist_lab[(hm >= 60) & (hm <= 90)] = "60-90"
    hist_lab[hm > 90] = "91-120"
    med["hist_median_bucket"] = hist_lab

    nm = med["norm_mad"].to_numpy()
    nm_lab = np.full(len(med), "", dtype=object)
    # MEDIUM implies not both (nm<=0.35 and drift<=7); nm can still be <=0.35 if drift>7
    nm_lab[nm <= 0.35] = "<=0.35 (drift-driven MEDIUM)"
    nm_lab[(nm > 0.35) & (nm <= 0.50)] = "0.35-0.50"
    nm_lab[nm > 0.50] = ">0.50"  # should be empty if eligibility enforced
    med["norm_mad_bucket"] = nm_lab

    dr = med["drift"].to_numpy()
    dr_lab = np.full(len(med), "", dtype=object)
    dr_lab[dr <= 7] = "<=7 (normMAD-driven MEDIUM)"
    dr_lab[(dr > 7) & (dr <= 10)] = "7-10"
    dr_lab[dr > 10] = ">10"
    med["drift_bucket"] = dr_lab

    pr = med["pred"].to_numpy()
    pr_lab = np.full(len(med), "", dtype=object)
    pr_lab[pr <= 14] = "pred 1-14"
    pr_lab[(pr > 14) & (pr <= 30)] = "pred 15-30"
    pr_lab[(pr > 30) & (pr <= 45)] = "pred 31-45"
    pr_lab[(pr > 45) & (pr <= 60)] = "pred 46-60"
    pr_lab[(pr > 60) & (pr <= 90)] = "pred 61-90"
    pr_lab[pr > 90] = "pred >90"
    med["pred_bucket"] = pr_lab

    ac = med["actual"].to_numpy()
    ac_lab = np.full(len(med), "", dtype=object)
    ac_lab[ac <= 14] = "actual 0-14"
    ac_lab[(ac > 14) & (ac <= 30)] = "actual 15-30"
    ac_lab[(ac > 30) & (ac <= 45)] = "actual 31-45"
    ac_lab[(ac > 45) & (ac <= 60)] = "actual 46-60"
    ac_lab[ac > 60] = "actual >60"
    med["actual_bucket"] = ac_lab

    bias_lab = np.full(len(med), "", dtype=object)
    bias_lab[bias < -14] = "early >14d"
    bias_lab[(bias >= -14) & (bias < -7)] = "early 8-14d"
    bias_lab[(bias >= -7) & (bias < 0)] = "early 1-7d"
    bias_lab[bias == 0] = "exact"
    bias_lab[(bias > 0) & (bias <= 7)] = "late 1-7d"
    bias_lab[(bias > 7) & (bias <= 14)] = "late 8-14d"
    bias_lab[(bias > 14) & (bias <= 30)] = "late 15-30d"
    bias_lab[bias > 30] = "late >30d"
    med["bias_bucket"] = bias_lab

    # Pattern rules (quantify largest error drivers)
    patterns = []

    def add_pattern(name: str, mask: np.ndarray, meaning: str):
        m = mask.astype(bool)
        if m.sum() == 0:
            return
        sub = med.loc[m]
        patterns.append(
            {
                "pattern": name,
                "meaning": meaning,
                "count": int(m.sum()),
                "pct_of_medium": round(100.0 * m.sum() / len(med), 2),
                "mae": round(float(sub["abs_error"].mean()), 2),
                "medae": round(float(sub["abs_error"].median()), 2),
                "mean_bias": round(float(sub["bias"].mean()), 2),
                "share_of_total_abs_error": round(
                    100.0 * sub["abs_error"].sum() / med["abs_error"].sum(), 2
                ),
                "within_7_days_pct": round(100.0 * (sub["abs_error"] <= 7).mean(), 2),
            }
        )

    add_pattern(
        "late_bias_gt_14",
        bias > 14,
        "Predicted interval more than 14 days longer than actual (late reminders)",
    )
    add_pattern(
        "late_bias_gt_30",
        bias > 30,
        "Severe late bias >30 days",
    )
    add_pattern(
        "early_bias_gt_14",
        bias < -14,
        "Predicted interval more than 14 days shorter than actual (early reminders)",
    )
    add_pattern(
        "pred_gt_60",
        pr > 60,
        "Model predicts >60 day interval",
    )
    add_pattern(
        "pred_gt_2x_hist_median",
        pr > (2.0 * hm),
        "Prediction > 2× personal historical median",
    )
    add_pattern(
        "actual_le_14_but_pred_gt_30",
        (ac <= 14) & (pr > 30),
        "Short actual refill (≤14d) but model predicts >30d",
    )
    add_pattern(
        "actual_15_30_but_pred_gt_60",
        (ac >= 15) & (ac <= 30) & (pr > 60),
        "Chronic-like actual (15–30d) but model predicts >60d",
    )
    add_pattern(
        "drift_driven_medium",
        (nm <= CORE_NORM_MAD_MAX) & (dr > CORE_DRIFT_MAX),
        "MEDIUM only because Drift>7 while NormMAD still ≤0.35",
    )
    add_pattern(
        "normmad_driven_medium",
        (nm > CORE_NORM_MAD_MAX) & (dr <= CORE_DRIFT_MAX),
        "MEDIUM only because NormMAD>0.35 while Drift still ≤7",
    )
    add_pattern(
        "both_elevated",
        (nm > CORE_NORM_MAD_MAX) & (dr > CORE_DRIFT_MAX),
        "Both NormMAD and Drift above core thresholds",
    )
    add_pattern(
        "shallow_depth_6_to_10",
        (med["depth"] >= 6) & (med["depth"] <= 10),
        "Shallow-eligible depth 6–10",
    )
    add_pattern(
        "abs_error_gt_14",
        abs_err > 14,
        "Absolute error outside ±14d window",
    )
    add_pattern(
        "abs_error_gt_30",
        abs_err > 30,
        "Absolute error >30 days",
    )

    patterns_sorted = sorted(patterns, key=lambda p: (-p["share_of_total_abs_error"], -p["count"]))

    overall = {
        "count": int(len(med)),
        "mae": round(float(abs_err.mean()), 2),
        "medae": round(float(np.median(abs_err)), 2),
        "rmse": round(float(np.sqrt(np.mean(bias**2))), 2),
        "mean_bias": round(float(bias.mean()), 2),
        "median_bias": round(float(np.median(bias)), 2),
        "pct_late": round(100.0 * (bias > 0).mean(), 2),
        "pct_early": round(100.0 * (bias < 0).mean(), 2),
        "pct_exact": round(100.0 * (bias == 0).mean(), 2),
        "within_3_days_pct": round(100.0 * (abs_err <= 3).mean(), 2),
        "within_7_days_pct": round(100.0 * (abs_err <= 7).mean(), 2),
        "mean_pred": round(float(pred.mean()), 2),
        "mean_actual": round(float(actual.mean()), 2),
        "median_pred": round(float(np.median(pred)), 2),
        "median_actual": round(float(np.median(actual)), 2),
        "pred_p95": round(float(np.percentile(pred, 95)), 2),
        "actual_p95": round(float(np.percentile(actual, 95)), 2),
    }

    results = {
        "experiment": "PHASE_17D_MEDIUM_ERROR_ANALYSIS",
        "read_only": True,
        "confidence_branch": CONF_MEDIUM,
        "controls": {
            "test_rows_universe": int(len(test)),
            "medium_rows": int(len(med)),
            "test_date_min": str(pd.to_datetime(test["invoice_date"]).min().date()),
            "test_date_max": str(pd.to_datetime(test["invoice_date"]).max().date()),
            "predictor": "production refill_model.joblib (reg:squarederror, no retrain)",
            "hybrid_rules": "P>=6, hist median in [15,120], NormMAD<=0.50, Drift<=10, not core",
        },
        "overall": overall,
        "by_absolute_error_bucket": _summarize_group(med, "abs_error_bucket"),
        "by_purchase_history_depth": _summarize_group(med, "depth_bucket"),
        "by_historical_median_interval": _summarize_group(med, "hist_median_bucket"),
        "by_norm_mad": _summarize_group(med, "norm_mad_bucket"),
        "by_drift": _summarize_group(med, "drift_bucket"),
        "by_predicted_interval": _summarize_group(med, "pred_bucket"),
        "by_actual_interval": _summarize_group(med, "actual_bucket"),
        "by_prediction_bias": _summarize_group(med, "bias_bucket"),
        "error_patterns_ranked_by_share_of_total_abs_error": patterns_sorted,
    }

    # Order abs-error buckets
    order = ["0-3", "4-7", "8-14", "15-30", ">30"]
    results["by_absolute_error_bucket"] = {
        k: results["by_absolute_error_bucket"][k]
        for k in order
        if k in results["by_absolute_error_bucket"]
    }

    model_mtime_after = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    report_mtime_after = REPORT_PATH.stat().st_mtime if REPORT_PATH.exists() else None
    results["artifact_safety"] = {
        "refill_model_joblib_mtime_unchanged": model_mtime_before == model_mtime_after,
        "phase4_report_mtime_unchanged": report_mtime_before == report_mtime_after,
        "model_mtime_before": model_mtime_before,
        "model_mtime_after": model_mtime_after,
        "whatsapp_messages_sent": 0,
        "production_code_modified": False,
    }

    out = PROC / "phase17d_medium_error_analysis_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"MEDIUM n={len(med)} MAE={overall['mae']} mean_bias={overall['mean_bias']}", flush=True)
    top = [
        {
            "pattern": p["pattern"],
            "count": p["count"],
            "mae": p["mae"],
            "share_of_total_abs_error": p["share_of_total_abs_error"],
        }
        for p in patterns_sorted[:5]
    ]
    print("Top patterns:", json.dumps(top), flush=True)
    print(f"Wrote {out}", flush=True)
    print("Artifact safety:", results["artifact_safety"], flush=True)


if __name__ == "__main__":
    main()
