"""READ-ONLY Phase 17D MEDIUM prediction-safety gate experiment.

Compares offline safety strategies on MEDIUM (hybrid secondary) rows only.
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

# Catastrophic = abs error > 30 days (same definition as MEDIUM error analysis)
CATASTROPHIC_ABS_ERROR = 30.0

_spec = importlib.util.spec_from_file_location(
    "phase17d_hybrid_comparison",
    ROOT / "scripts" / "phase17d_hybrid_comparison.py",
)
_hc = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_hc)
_attach_regularity_from_history = _hc._attach_regularity_from_history


def _is_medium(record: dict) -> bool:
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


def _metrics(
    pred: np.ndarray,
    actual: np.ndarray,
    *,
    medium_n: int,
    test_n: int,
    baseline_total_abs: float,
    baseline_cat_abs: float,
) -> dict:
    n_acc = int(len(pred))
    n_rej = int(medium_n - n_acc)
    if n_acc == 0:
        return {
            "accepted": 0,
            "rejected": n_rej,
            "coverage_of_medium_pct": 0.0,
            "coverage_of_test_pct": 0.0,
            "mae": None,
            "medae": None,
            "within_3_days_pct": None,
            "within_7_days_pct": None,
            "total_absolute_error": 0.0,
            "catastrophic_count": 0,
            "catastrophic_abs_error": 0.0,
            "catastrophic_share_of_accepted_abs_error_pct": None,
            "catastrophic_abs_error_vs_baseline_pct": 0.0,
            "total_abs_error_vs_baseline_pct": 0.0,
            "mean_bias": None,
            "mean_pred": None,
            "mean_actual": None,
        }

    abs_err = np.abs(pred - actual)
    bias = pred - actual
    cat_mask = abs_err > CATASTROPHIC_ABS_ERROR
    cat_abs = float(abs_err[cat_mask].sum())
    total_abs = float(abs_err.sum())

    return {
        "accepted": n_acc,
        "rejected": n_rej,
        "coverage_of_medium_pct": round(100.0 * n_acc / medium_n, 2),
        "coverage_of_test_pct": round(100.0 * n_acc / test_n, 2),
        "mae": round(float(abs_err.mean()), 2),
        "medae": round(float(np.median(abs_err)), 2),
        "within_3_days_pct": round(100.0 * float((abs_err <= 3).mean()), 2),
        "within_7_days_pct": round(100.0 * float((abs_err <= 7).mean()), 2),
        "total_absolute_error": round(total_abs, 2),
        "catastrophic_definition": f"abs_error > {CATASTROPHIC_ABS_ERROR:.0f} days",
        "catastrophic_count": int(cat_mask.sum()),
        "catastrophic_pct_of_accepted": round(100.0 * float(cat_mask.mean()), 2),
        "catastrophic_abs_error": round(cat_abs, 2),
        "catastrophic_share_of_accepted_abs_error_pct": round(
            100.0 * cat_abs / total_abs if total_abs > 0 else 0.0, 2
        ),
        "catastrophic_abs_error_vs_baseline_pct": round(
            100.0 * cat_abs / baseline_cat_abs if baseline_cat_abs > 0 else 0.0, 2
        ),
        "total_abs_error_vs_baseline_pct": round(
            100.0 * total_abs / baseline_total_abs if baseline_total_abs > 0 else 0.0, 2
        ),
        "mean_bias": round(float(bias.mean()), 2),
        "mean_pred": round(float(pred.mean()), 2),
        "mean_actual": round(float(actual.mean()), 2),
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
    print("Loading test set...", flush=True)
    test = pd.read_parquet(PROC / "test.parquet")
    test = test[[c for c in keep if c in test.columns]]
    test = test[test[TARGET_COL] > 0].copy()
    test_n = int(len(test))

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

    medium_mask = np.array([_is_medium(r.to_dict()) for _, r in test.iterrows()], dtype=bool)
    med = test.loc[medium_mask].copy().reset_index(drop=True)
    pred_xgb = xgb_pred[medium_mask]
    actual = med[TARGET_COL].to_numpy(dtype=np.float64)
    hist = med["historical_interval_median"].to_numpy(dtype=np.float64)
    medium_n = int(len(med))

    flag_pred_gt_60 = pred_xgb > 60.0
    flag_pred_gt_2x = pred_xgb > (2.0 * hist)

    # Baseline totals for relative catastrophic reporting
    base_abs = np.abs(pred_xgb - actual)
    baseline_total_abs = float(base_abs.sum())
    baseline_cat_abs = float(base_abs[base_abs > CATASTROPHIC_ABS_ERROR].sum())

    strategies: dict[str, dict] = {}

    # A. Current XGBoost (all MEDIUM accepted)
    strategies["A_current_xgb"] = {
        "label": "A. Current XGBoost",
        "rule": "Accept all MEDIUM with production XGBoost prediction",
        "gate_triggered_count": 0,
        "gate_triggered_pct_of_medium": 0.0,
        **_metrics(
            pred_xgb,
            actual,
            medium_n=medium_n,
            test_n=test_n,
            baseline_total_abs=baseline_total_abs,
            baseline_cat_abs=baseline_cat_abs,
        ),
    }

    # B. Reject when predicted interval > 60 days
    keep_b = ~flag_pred_gt_60
    strategies["B_reject_pred_gt_60"] = {
        "label": "B. Reject when predicted interval > 60 days",
        "rule": "Reject MEDIUM if XGB pred > 60; else keep XGB",
        "gate_triggered_count": int(flag_pred_gt_60.sum()),
        "gate_triggered_pct_of_medium": round(100.0 * float(flag_pred_gt_60.mean()), 2),
        **_metrics(
            pred_xgb[keep_b],
            actual[keep_b],
            medium_n=medium_n,
            test_n=test_n,
            baseline_total_abs=baseline_total_abs,
            baseline_cat_abs=baseline_cat_abs,
        ),
    }

    # C. Reject when prediction > 2 × historical median
    keep_c = ~flag_pred_gt_2x
    strategies["C_reject_pred_gt_2x_hist"] = {
        "label": "C. Reject when prediction > 2 × historical median",
        "rule": "Reject MEDIUM if XGB pred > 2 * hist median; else keep XGB",
        "gate_triggered_count": int(flag_pred_gt_2x.sum()),
        "gate_triggered_pct_of_medium": round(100.0 * float(flag_pred_gt_2x.mean()), 2),
        **_metrics(
            pred_xgb[keep_c],
            actual[keep_c],
            medium_n=medium_n,
            test_n=test_n,
            baseline_total_abs=baseline_total_abs,
            baseline_cat_abs=baseline_cat_abs,
        ),
    }

    # D. Use personal historical median when prediction > 2 × historical median
    pred_d = pred_xgb.copy()
    pred_d[flag_pred_gt_2x] = hist[flag_pred_gt_2x]
    strategies["D_fallback_hist_median_when_pred_gt_2x"] = {
        "label": "D. Use personal historical median when prediction > 2 × historical median",
        "rule": "If XGB pred > 2 * hist median, replace with hist median (still accepted); else XGB",
        "gate_triggered_count": int(flag_pred_gt_2x.sum()),
        "gate_triggered_pct_of_medium": round(100.0 * float(flag_pred_gt_2x.mean()), 2),
        "fallback_replaced_count": int(flag_pred_gt_2x.sum()),
        **_metrics(
            pred_d,
            actual,
            medium_n=medium_n,
            test_n=test_n,
            baseline_total_abs=baseline_total_abs,
            baseline_cat_abs=baseline_cat_abs,
        ),
    }

    # E. Combine B + C
    reject_e = flag_pred_gt_60 | flag_pred_gt_2x
    keep_e = ~reject_e
    strategies["E_combine_B_and_C"] = {
        "label": "E. Combine B + C",
        "rule": "Reject MEDIUM if XGB pred > 60 OR XGB pred > 2 * hist median; else keep XGB",
        "gate_triggered_count": int(reject_e.sum()),
        "gate_triggered_pct_of_medium": round(100.0 * float(reject_e.mean()), 2),
        "overlap_both_gates": int((flag_pred_gt_60 & flag_pred_gt_2x).sum()),
        "only_pred_gt_60": int((flag_pred_gt_60 & ~flag_pred_gt_2x).sum()),
        "only_pred_gt_2x": int((~flag_pred_gt_60 & flag_pred_gt_2x).sum()),
        **_metrics(
            pred_xgb[keep_e],
            actual[keep_e],
            medium_n=medium_n,
            test_n=test_n,
            baseline_total_abs=baseline_total_abs,
            baseline_cat_abs=baseline_cat_abs,
        ),
    }

    model_mtime_after = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    report_mtime_after = REPORT_PATH.stat().st_mtime if REPORT_PATH.exists() else None

    results = {
        "experiment": "PHASE_17D_MEDIUM_SAFETY_GATE_EXPERIMENT",
        "read_only": True,
        "confidence_branch": CONF_MEDIUM,
        "controls": {
            "test_rows_universe": test_n,
            "medium_rows": medium_n,
            "test_date_min": str(pd.to_datetime(test["invoice_date"]).min().date()),
            "test_date_max": str(pd.to_datetime(test["invoice_date"]).max().date()),
            "predictor": "production refill_model.joblib (reg:squarederror, no retrain)",
            "hybrid_medium_rules": (
                "P>=6, hist median in [15,120], NormMAD<=0.50, Drift<=10, not core"
            ),
            "catastrophic_definition": f"abs_error > {CATASTROPHIC_ABS_ERROR:.0f} days",
            "metrics_scope": "accepted MEDIUM predictions only (D keeps all MEDIUM accepted)",
        },
        "gate_flag_summary": {
            "pred_gt_60_count": int(flag_pred_gt_60.sum()),
            "pred_gt_60_pct": round(100.0 * float(flag_pred_gt_60.mean()), 2),
            "pred_gt_2x_hist_count": int(flag_pred_gt_2x.sum()),
            "pred_gt_2x_hist_pct": round(100.0 * float(flag_pred_gt_2x.mean()), 2),
            "either_gate_count": int(reject_e.sum()),
            "either_gate_pct": round(100.0 * float(reject_e.mean()), 2),
            "both_gates_count": int((flag_pred_gt_60 & flag_pred_gt_2x).sum()),
        },
        "baseline_error_mass": {
            "total_absolute_error": round(baseline_total_abs, 2),
            "catastrophic_abs_error": round(baseline_cat_abs, 2),
            "catastrophic_count": int((base_abs > CATASTROPHIC_ABS_ERROR).sum()),
        },
        "strategies": strategies,
        "artifact_safety": {
            "refill_model_joblib_mtime_unchanged": model_mtime_before == model_mtime_after,
            "phase4_report_mtime_unchanged": report_mtime_before == report_mtime_after,
            "model_mtime_before": model_mtime_before,
            "model_mtime_after": model_mtime_after,
            "whatsapp_messages_sent": 0,
            "production_code_modified": False,
        },
    }

    out = PROC / "phase17d_medium_safety_gate_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"MEDIUM n={medium_n} / test n={test_n}", flush=True)
    for key, s in strategies.items():
        print(
            f"{key}: accepted={s['accepted']} rejected={s['rejected']} "
            f"MAE={s['mae']} ±7d={s['within_7_days_pct']} "
            f"total_abs={s['total_absolute_error']} cat_abs={s['catastrophic_abs_error']}",
            flush=True,
        )
    print(f"Wrote {out}", flush=True)
    print("Artifact safety:", json.dumps(results["artifact_safety"]), flush=True)


if __name__ == "__main__":
    main()
