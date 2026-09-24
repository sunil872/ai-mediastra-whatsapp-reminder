"""READ-ONLY Phase 17D eligibility-threshold experiment for hybrid_routing_v17d.

Sweeps history-depth gates P>=3,4,5,6,7,10 while holding all other hybrid rules fixed.
Uses existing production refill_model.joblib for MEDIUM branch (no retrain).

Does NOT modify production code, model artifacts, data, reminder logic, or WhatsApp.
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
    CONF_HIGH,
    CONF_MEDIUM,
    CONF_REJECTED,
    ROUTE_CORE,
    ROUTE_SECONDARY,
    ROUTE_REJECTED,
    resolve_regularity_fields,
)

PROC = ROOT / "data" / "refillcare" / "processed"
MODEL_PATH = PROC / "models" / "refill_model.joblib"
REPORT_PATH = PROC / "phase4_model_report.json"

DEPTH_THRESHOLDS = [3, 4, 5, 6, 7, 10]

_spec = importlib.util.spec_from_file_location(
    "phase17d_hybrid_comparison",
    ROOT / "scripts" / "phase17d_hybrid_comparison.py",
)
_hc = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_hc)
_attach_regularity_from_history = _hc._attach_regularity_from_history


def _metrics(y_true, y_pred) -> dict:
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    mask = ~np.isnan(y_t) & ~np.isnan(y_p)
    y_t, y_p = y_t[mask], np.clip(y_p[mask], 1.0, None)
    n = len(y_t)
    if n == 0:
        return {
            "count": 0,
            "mae": None,
            "medae": None,
            "rmse": None,
            "within_3_days_pct": None,
            "within_7_days_pct": None,
        }
    err = y_p - y_t
    abs_err = np.abs(err)
    return {
        "count": int(n),
        "mae": round(float(np.mean(abs_err)), 2),
        "medae": round(float(np.median(abs_err)), 2),
        "rmse": round(float(np.sqrt(np.mean(err**2))), 2),
        "within_3_days_pct": round(float((abs_err <= 3).mean() * 100), 2),
        "within_7_days_pct": round(float((abs_err <= 7).mean() * 100), 2),
    }


def _eligibility_at_depth(record: dict, min_purchases: int) -> dict:
    """Offline copy of hybrid eligibility with parameterized depth gate only."""
    purchase_count = int(record.get("purchase_count_so_far", 1) or 1)
    reg = resolve_regularity_fields(record)
    hist_med = reg["historical_interval_median"]
    norm_mad = reg["norm_mad"]
    drift = reg["cadence_drift"]

    if purchase_count < min_purchases:
        return {
            "is_eligible": False,
            "route": ROUTE_REJECTED,
            "confidence": CONF_REJECTED,
            **reg,
        }
    if math.isnan(hist_med) or hist_med < CADENCE_MEDIAN_MIN or hist_med > CADENCE_MEDIAN_MAX:
        return {
            "is_eligible": False,
            "route": ROUTE_REJECTED,
            "confidence": CONF_REJECTED,
            **reg,
        }
    if math.isnan(norm_mad) or math.isnan(drift):
        return {
            "is_eligible": False,
            "route": ROUTE_REJECTED,
            "confidence": CONF_REJECTED,
            **reg,
        }
    if norm_mad > BROAD_NORM_MAD_MAX or drift > BROAD_DRIFT_MAX:
        return {
            "is_eligible": False,
            "route": ROUTE_REJECTED,
            "confidence": CONF_REJECTED,
            **reg,
        }
    is_core = (norm_mad <= CORE_NORM_MAD_MAX) and (drift <= CORE_DRIFT_MAX)
    if is_core:
        return {
            "is_eligible": True,
            "route": ROUTE_CORE,
            "confidence": CONF_HIGH,
            **reg,
        }
    return {
        "is_eligible": True,
        "route": ROUTE_SECONDARY,
        "confidence": CONF_MEDIUM,
        **reg,
    }


def _score_threshold(
    test: pd.DataFrame,
    y_true: np.ndarray,
    xgb_pred: np.ndarray,
    min_purchases: int,
) -> dict:
    n = len(test)
    pred = np.full(n, np.nan, dtype=np.float64)
    conf = np.full(n, CONF_REJECTED, dtype=object)
    route = np.full(n, ROUTE_REJECTED, dtype=object)

    for i, (_, row) in enumerate(test.iterrows()):
        d = _eligibility_at_depth(row.to_dict(), min_purchases)
        conf[i] = d["confidence"]
        route[i] = d["route"]
        if not d["is_eligible"]:
            continue
        if d["route"] == ROUTE_CORE:
            med = d["historical_interval_median"]
            if med is not None and not math.isnan(med) and med > 0:
                pred[i] = float(med)
            else:
                conf[i] = CONF_REJECTED
                route[i] = ROUTE_REJECTED
        else:
            pred[i] = float(xgb_pred[i])

    accepted = ~np.isnan(pred)
    overall = _metrics(y_true, pred)
    high = conf == CONF_HIGH
    med = conf == CONF_MEDIUM

    p = test["purchase_count_so_far"].to_numpy()
    norm_mad = test["norm_mad"].to_numpy(dtype=np.float64)
    drift = test["cadence_drift"].to_numpy(dtype=np.float64)
    # Regularity classes use fixed Class-1 definition (independent of depth gate)
    regular = (
        (p >= 6)
        & ~np.isnan(norm_mad)
        & ~np.isnan(drift)
        & (norm_mad <= CORE_NORM_MAD_MAX)
        & (drift <= CORE_DRIFT_MAX)
    )
    irregular = (
        (p >= 6)
        & ~np.isnan(norm_mad)
        & ~np.isnan(drift)
        & ((norm_mad > CORE_NORM_MAD_MAX) | (drift > CORE_DRIFT_MAX))
    )
    # Also report regular/irregular among rows that meet *this* depth gate
    depth_ok = p >= min_purchases
    regular_at_depth = (
        depth_ok
        & ~np.isnan(norm_mad)
        & ~np.isnan(drift)
        & (norm_mad <= CORE_NORM_MAD_MAX)
        & (drift <= CORE_DRIFT_MAX)
    )
    irregular_at_depth = (
        depth_ok
        & ~np.isnan(norm_mad)
        & ~np.isnan(drift)
        & ((norm_mad > CORE_NORM_MAD_MAX) | (drift > CORE_DRIFT_MAX))
    )

    def _seg(mask: np.ndarray) -> dict:
        univ = int(mask.sum())
        scored = mask & ~np.isnan(pred)
        m = _metrics(y_true[mask], pred[mask])
        return {
            "cohort_rows": univ,
            "accepted_count": int(scored.sum()),
            "rejected_count": int((mask & np.isnan(pred)).sum()),
            "coverage_pct": round(100.0 * scored.sum() / univ, 2) if univ else None,
            **{k: m[k] for k in ("mae", "medae", "within_3_days_pct", "within_7_days_pct")},
        }

    return {
        "min_purchases": min_purchases,
        "test_row_count": n,
        "accepted_count": int(accepted.sum()),
        "rejected_count": int((~accepted).sum()),
        "coverage_pct": round(100.0 * accepted.sum() / n, 2),
        "mae": overall["mae"],
        "medae": overall["medae"],
        "rmse": overall["rmse"],
        "within_3_days_pct": overall["within_3_days_pct"],
        "within_7_days_pct": overall["within_7_days_pct"],
        "HIGH": {
            "count": int(high.sum()),
            **_metrics(y_true[high], pred[high]),
        },
        "MEDIUM": {
            "count": int(med.sum()),
            **_metrics(y_true[med], pred[med]),
        },
        "regular_vs_irregular": {
            # Among rows with P>=6 Class-1 / Class-2 definitions (same as Hybrid Evaluation)
            "regular_class1_pge6": _seg(regular),
            "irregular_pge6": _seg(irregular),
            # Among rows meeting this experiment's depth gate
            "regular_at_depth_gate": _seg(regular_at_depth),
            "irregular_at_depth_gate": _seg(irregular_at_depth),
        },
        "confidence_counts": {
            "HIGH": int(high.sum()),
            "MEDIUM": int(med.sum()),
            "REJECTED": int((conf == CONF_REJECTED).sum()),
        },
    }


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
    print("Loading test set (read-only)...", flush=True)
    test = pd.read_parquet(PROC / "test.parquet")
    test = test[[c for c in keep if c in test.columns]]
    test = test[test[TARGET_COL] > 0].copy()
    n = len(test)
    y_true = test[TARGET_COL].to_numpy(dtype=np.float64)
    print(f"Test rows: {n:,}", flush=True)

    print("Attaching NormMAD/drift from purchase_history...", flush=True)
    history = pd.read_parquet(PROC / "purchase_history.parquet")
    test = _attach_regularity_from_history(test, history)
    test["historical_interval_norm_mad"] = test["norm_mad"]
    test["recent3_interval_median"] = test["recent3_median"]
    del history
    gc.collect()

    print("Scoring production XGBoost once (read-only)...", flush=True)
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

    results = {
        "experiment": "PHASE_17D_ELIGIBILITY_THRESHOLD_EXPERIMENT",
        "read_only": True,
        "production_model_retrained": False,
        "fixed_rules": {
            "cadence_median_band": [CADENCE_MEDIAN_MIN, CADENCE_MEDIAN_MAX],
            "core_norm_mad_max": CORE_NORM_MAD_MAX,
            "core_drift_max": CORE_DRIFT_MAX,
            "broad_norm_mad_max": BROAD_NORM_MAD_MAX,
            "broad_drift_max": BROAD_DRIFT_MAX,
            "HIGH_predictor": "personal_historical_median",
            "MEDIUM_predictor": "production_refill_model.joblib (no retrain)",
        },
        "controls": {
            "test_rows": n,
            "test_date_min": str(pd.to_datetime(test["invoice_date"]).min().date()),
            "test_date_max": str(pd.to_datetime(test["invoice_date"]).max().date()),
            "depth_thresholds": DEPTH_THRESHOLDS,
            "current_production_min_purchases": 6,
        },
        "thresholds": {},
    }

    for pmin in DEPTH_THRESHOLDS:
        print(f"Evaluating P>={pmin} ...", flush=True)
        block = _score_threshold(test, y_true, xgb_pred, pmin)
        results["thresholds"][f"P_ge_{pmin}"] = block
        print(
            f"  accepted={block['accepted_count']} ({block['coverage_pct']}%) "
            f"MAE={block['mae']} ±7d={block['within_7_days_pct']}% "
            f"HIGH={block['confidence_counts']['HIGH']} MEDIUM={block['confidence_counts']['MEDIUM']}",
            flush=True,
        )

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

    # Ranking helpers
    results["ranking_by_mae"] = sorted(
        [
            {
                "threshold": k,
                "coverage_pct": v["coverage_pct"],
                "mae": v["mae"],
                "within_7_days_pct": v["within_7_days_pct"],
                "accepted_count": v["accepted_count"],
            }
            for k, v in results["thresholds"].items()
            if v["mae"] is not None
        ],
        key=lambda r: (r["mae"], -r["coverage_pct"]),
    )

    out = PROC / "phase17d_eligibility_threshold_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out}", flush=True)
    print("Ranking:", results["ranking_by_mae"], flush=True)
    print("Artifact safety:", results["artifact_safety"], flush=True)


if __name__ == "__main__":
    main()
