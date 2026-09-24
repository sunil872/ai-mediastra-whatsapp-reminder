"""READ-ONLY Phase 17D hybrid_routing_v17d evaluation on untouched test set.

Compares:
1. Current production XGBoost (refill_model.joblib read-only)
2. Historical median baseline
3. Personal cadence (tiered heuristic)
4. hybrid_routing_v17d (implemented prediction layer)

Does NOT retrain, overwrite refill_model.joblib, modify production code, or send WhatsApp.
"""
from __future__ import annotations

import gc
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from refillcare.models.training import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COL
from refillcare.evaluation.baseline import HistoricalMedianBaseline
from refillcare.models.prediction import generate_batch_predictions
from refillcare.models.hybrid_strategy import (
    STRATEGY_NAME,
    CONF_HIGH,
    CONF_MEDIUM,
    CONF_REJECTED,
    CORE_NORM_MAD_MAX,
    CORE_DRIFT_MAX,
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
_tiered_cadence_predict = _hc._tiered_cadence_predict


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
            "within_1_day_pct": None,
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
        "within_1_day_pct": round(float((abs_err <= 1).mean() * 100), 2),
        "within_3_days_pct": round(float((abs_err <= 3).mean() * 100), 2),
        "within_7_days_pct": round(float((abs_err <= 7).mean() * 100), 2),
    }


def _strategy_block(y_true: np.ndarray, y_pred: np.ndarray, universe: int) -> dict:
    scored = ~np.isnan(y_pred)
    accepted = int(scored.sum())
    rejected = int((~scored).sum())
    m = _metrics(y_true, y_pred)
    return {
        "test_row_count": universe,
        "eligible_accepted_count": accepted,
        "rejected_count": rejected,
        "acceptance_pct": round(100.0 * accepted / universe, 2) if universe else None,
        "rejection_pct": round(100.0 * rejected / universe, 2) if universe else None,
        "mae": m["mae"],
        "medae": m["medae"],
        "rmse": m["rmse"],
        "within_1_day_pct": m["within_1_day_pct"],
        "within_3_days_pct": m["within_3_days_pct"],
        "within_7_days_pct": m["within_7_days_pct"],
        "scored_metrics_count": m["count"],
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
                "is_recurring_history",
                "historical_interval_median",
                "itemName",
            ]
        )
    )

    print("Loading train/test (read-only)...", flush=True)
    train = pd.read_parquet(PROC / "train.parquet")
    test = pd.read_parquet(PROC / "test.parquet")
    cols = [c for c in keep if c in train.columns]
    train = train[cols]
    test = test[[c for c in keep if c in test.columns]]
    train = train[train[TARGET_COL] > 0].copy()
    test = test[test[TARGET_COL] > 0].copy()
    n = len(test)
    y_true = test[TARGET_COL].to_numpy(dtype=np.float64)
    print(f"Test rows (target>0): {n:,}", flush=True)

    print("Attaching NormMAD / drift from purchase_history...", flush=True)
    history = pd.read_parquet(PROC / "purchase_history.parquet")
    test = _attach_regularity_from_history(test, history)
    # Align hybrid_strategy expected column names
    test["historical_interval_norm_mad"] = test["norm_mad"]
    test["recent3_interval_median"] = test["recent3_median"]
    del history
    gc.collect()

    # 1) Current production XGBoost
    print("Scoring current production XGBoost (read-only)...", flush=True)
    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipeline"]
    feats = list(bundle.get("features_numeric", NUMERIC_FEATURES)) + list(
        bundle.get("features_categorical", CATEGORICAL_FEATURES)
    )
    X = test.copy()
    for c in feats:
        if c not in X.columns:
            X[c] = np.nan
    current_xgb = np.clip(np.asarray(pipe.predict(X[feats]), dtype=np.float64), 1.0, None)
    del pipe, X
    gc.collect()

    # 2) Historical median baseline
    baseline = HistoricalMedianBaseline()
    baseline.fit(train, target_col=TARGET_COL)
    hist_pred = np.asarray(baseline.predict(test), dtype=np.float64)

    # 3) Personal cadence
    cadence_pred = np.asarray(
        _tiered_cadence_predict(test, baseline.fallback_median), dtype=np.float64
    )

    # 4) hybrid_routing_v17d via implemented prediction layer (uses production bundle for MEDIUM)
    print("Scoring hybrid_routing_v17d via prediction layer...", flush=True)
    hybrid_df = generate_batch_predictions(bundle, test)
    hybrid_pred = hybrid_df["predicted_days_until_refill"].to_numpy(dtype=np.float64)
    hybrid_conf = hybrid_df["refill_confidence"].astype(str).to_numpy()
    hybrid_route = hybrid_df["prediction_route"].astype(str).to_numpy()

    results = {
        "experiment": "PHASE_17D_HYBRID_EVALUATION",
        "read_only": True,
        "production_model_retrained": False,
        "refill_model_overwritten": False,
        "whatsapp_messages_sent": 0,
        "strategy_name": STRATEGY_NAME,
        "controls": {
            "test_rows_target_gt_0": n,
            "test_date_min": str(pd.to_datetime(test["invoice_date"]).min().date()),
            "test_date_max": str(pd.to_datetime(test["invoice_date"]).max().date()),
            "train_rows_for_baseline_fallback_only": int(len(train)),
            "fallback_median_days": baseline.fallback_median,
            "model_path": str(MODEL_PATH),
            "note": (
                "hybrid MEDIUM branch uses existing production refill_model.joblib "
                "(not retrained). HIGH branch uses personal historical median."
            ),
        },
        "strategies": {
            "current_xgboost": {
                "label": "Current production XGBoost",
                "always_on": True,
                **_strategy_block(y_true, current_xgb, n),
            },
            "historical_median_baseline": {
                "label": "Historical median baseline",
                "always_on": True,
                **_strategy_block(y_true, hist_pred, n),
            },
            "personal_cadence_strategy": {
                "label": "Personal cadence (tiered)",
                "always_on": True,
                **_strategy_block(y_true, cadence_pred, n),
            },
            "hybrid_routing_v17d": {
                "label": STRATEGY_NAME,
                "always_on": False,
                **_strategy_block(y_true, hybrid_pred, n),
            },
        },
    }

    # Hybrid extras
    high_mask = hybrid_conf == CONF_HIGH
    med_mask = hybrid_conf == CONF_MEDIUM
    rej_mask = (hybrid_conf == CONF_REJECTED) | np.isnan(hybrid_pred)

    p = test["purchase_count_so_far"].to_numpy()
    norm_mad = test["norm_mad"].to_numpy(dtype=np.float64)
    drift = test["cadence_drift"].to_numpy(dtype=np.float64)
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

    depth_cohorts = {
        "depth_le_2": p <= 2,
        "depth_3_to_5": (p >= 3) & (p <= 5),
        "depth_6_to_10": (p >= 6) & (p <= 10),
        "depth_gt_10": p > 10,
    }

    hybrid_extra = {
        "confidence_counts": {
            "HIGH": int(high_mask.sum()),
            "MEDIUM": int(med_mask.sum()),
            "REJECTED": int(rej_mask.sum()),
        },
        "route_counts": {
            "core_personal_median": int((hybrid_route == "core_personal_median").sum()),
            "secondary_experimental_xgb": int(
                (hybrid_route == "secondary_experimental_xgb").sum()
            ),
            "rejected": int((hybrid_route == "rejected").sum()),
        },
        "HIGH": _metrics(y_true[high_mask], hybrid_pred[high_mask]),
        "MEDIUM": _metrics(y_true[med_mask], hybrid_pred[med_mask]),
        "by_purchase_history_depth": {},
        "by_regularity": {},
    }

    for key, mask in (("regular_class1", regular), ("irregular_ge6", irregular)):
        univ = int(mask.sum())
        scored = ~np.isnan(hybrid_pred[mask])
        m = _metrics(y_true[mask], hybrid_pred[mask])
        hybrid_extra["by_regularity"][key] = {
            "cohort_rows": univ,
            "eligible_accepted_count": int(scored.sum()),
            "rejected_count": int((~scored).sum()),
            "acceptance_pct": round(100.0 * scored.sum() / univ, 2) if univ else None,
            "rejection_pct": round(100.0 * (~scored).sum() / univ, 2) if univ else None,
            "mae": m["mae"],
            "medae": m["medae"],
            "rmse": m["rmse"],
            "within_1_day_pct": m["within_1_day_pct"],
            "within_3_days_pct": m["within_3_days_pct"],
            "within_7_days_pct": m["within_7_days_pct"],
        }

    for name, mask in depth_cohorts.items():
        univ = int(mask.sum())
        scored = ~np.isnan(hybrid_pred[mask])
        m = _metrics(y_true[mask], hybrid_pred[mask])
        hybrid_extra["by_purchase_history_depth"][name] = {
            "cohort_rows": univ,
            "eligible_accepted_count": int(scored.sum()),
            "rejected_count": int((~scored).sum()),
            "acceptance_pct": round(100.0 * scored.sum() / univ, 2) if univ else None,
            "mae": m["mae"],
            "medae": m["medae"],
            "rmse": m["rmse"],
            "within_1_day_pct": m["within_1_day_pct"],
            "within_3_days_pct": m["within_3_days_pct"],
            "within_7_days_pct": m["within_7_days_pct"],
        }

    results["strategies"]["hybrid_routing_v17d"]["details"] = hybrid_extra

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

    out = PROC / "phase17d_hybrid_evaluation_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out}", flush=True)
    for k, v in results["strategies"].items():
        print(
            f"{k}: accepted={v['eligible_accepted_count']} rejected={v['rejected_count']} "
            f"MAE={v['mae']} ±7d={v['within_7_days_pct']}%",
            flush=True,
        )
    print("Hybrid details:", hybrid_extra["confidence_counts"], flush=True)
    print("Artifact safety:", results["artifact_safety"], flush=True)


if __name__ == "__main__":
    main()
