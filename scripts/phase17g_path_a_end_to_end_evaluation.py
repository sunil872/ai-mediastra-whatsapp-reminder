"""READ-ONLY Phase 17G Path A end-to-end offline evaluation.

Uses 17F Path A classifier (explicit offline call; feature flag remains OFF).
Does NOT replace production routing, modify model, retrain, touch WhatsApp,
reminder stages, or implement Path B.
"""
from __future__ import annotations

import gc
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from refillcare.models.training import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COL
from refillcare.models.hybrid_strategy import evaluate_hybrid_eligibility, MIN_PURCHASES as HYBRID_MIN_P
from refillcare.models.path_a_classifier import (
    STRATEGY_NAME as PATH_A_STRATEGY,
    BASELINE_STRATEGY_NAME,
    LABEL_HIGH,
    LABEL_MEDIUM,
    LABEL_UNSTABLE,
    classify_path_a,
    classify_path_a_from_intervals,
    is_path_a_classifier_enabled,
)

PROC = ROOT / "data" / "refillcare" / "processed"
MODEL_PATH = PROC / "models" / "refill_model.joblib"
REPORT_PATH = PROC / "phase4_model_report.json"
CATASTROPHIC_ABS_ERROR = 30.0

VARIABLE_CANONICAL = [30, 60, 45, 30, 90, 30, 60, 30, 90]
STABLE_CANONICAL = [30, 30, 31, 29, 30, 32]
UNSTABLE_CANONICAL = [5, 180, 12, 240, 3, 150]

_spec = importlib.util.spec_from_file_location(
    "phase17d_hybrid_comparison",
    ROOT / "scripts" / "phase17d_hybrid_comparison.py",
)
_hc = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_hc)
_attach_regularity_from_history = _hc._attach_regularity_from_history


def _metrics(pred: np.ndarray, actual: np.ndarray) -> Dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(actual) & (actual > 0)
    if mask.sum() == 0:
        return {
            "count": 0,
            "mae": None,
            "medae": None,
            "rmse": None,
            "within_1_day_pct": None,
            "within_3_days_pct": None,
            "within_7_days_pct": None,
            "total_absolute_error": 0.0,
            "catastrophic_count": 0,
            "catastrophic_pct": None,
            "catastrophic_abs_error": 0.0,
            "catastrophic_share_of_abs_error_pct": None,
            "mean_bias": None,
            "mean_pred": None,
            "mean_actual": None,
        }
    p = pred[mask]
    a = actual[mask]
    err = p - a
    ae = np.abs(err)
    cat = ae > CATASTROPHIC_ABS_ERROR
    total_abs = float(ae.sum())
    cat_abs = float(ae[cat].sum())
    return {
        "count": int(mask.sum()),
        "mae": round(float(ae.mean()), 2),
        "medae": round(float(np.median(ae)), 2),
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 2),
        "within_1_day_pct": round(100.0 * float((ae <= 1).mean()), 2),
        "within_3_days_pct": round(100.0 * float((ae <= 3).mean()), 2),
        "within_7_days_pct": round(100.0 * float((ae <= 7).mean()), 2),
        "total_absolute_error": round(total_abs, 2),
        "catastrophic_count": int(cat.sum()),
        "catastrophic_pct": round(100.0 * float(cat.mean()), 2),
        "catastrophic_abs_error": round(cat_abs, 2),
        "catastrophic_share_of_abs_error_pct": round(
            100.0 * cat_abs / total_abs if total_abs > 0 else 0.0, 2
        ),
        "mean_bias": round(float(err.mean()), 2),
        "mean_pred": round(float(p.mean()), 2),
        "mean_actual": round(float(a.mean()), 2),
    }


def _coverage_block(accepted: int, rejected: int, path_a_n: int, test_n: int) -> Dict[str, Any]:
    return {
        "accepted": int(accepted),
        "rejected": int(rejected),
        "coverage_of_path_a_pct": round(100.0 * accepted / path_a_n, 2) if path_a_n else 0.0,
        "coverage_of_test_pct": round(100.0 * accepted / test_n, 2) if test_n else 0.0,
    }


def _hybrid_label(rec: Dict[str, Any]) -> str:
    d = evaluate_hybrid_eligibility(rec)
    conf = d["confidence"]
    if conf == "HIGH":
        return LABEL_HIGH
    if conf == "MEDIUM":
        return LABEL_MEDIUM
    return LABEL_UNSTABLE


def _build_prior_map(history: pd.DataFrame, keys: set) -> Dict[Tuple[str, str, int], List[float]]:
    hist = history[
        ["customerId", "itemId", "invoice_date", "days_since_previous_purchase", "purchase_seq"]
    ].copy()
    hist["invoice_date"] = pd.to_datetime(hist["invoice_date"])
    hist["customerId"] = hist["customerId"].astype(str)
    hist["itemId"] = hist["itemId"].astype(str)
    hist = hist.sort_values(["customerId", "itemId", "invoice_date", "purchase_seq"])
    hist["_key"] = list(zip(hist["customerId"], hist["itemId"]))
    hist = hist[hist["_key"].isin(keys)]
    prior_map: Dict[Tuple[str, str, int], List[float]] = {}
    for (cid, iid), g in hist.groupby(["customerId", "itemId"], sort=False):
        intervals: List[float] = []
        for _, row in g.iterrows():
            seq = int(row["purchase_seq"])
            prior_map[(cid, iid, seq)] = list(intervals)
            d = row["days_since_previous_purchase"]
            if pd.notna(d) and float(d) > 0:
                intervals.append(float(d))
    return prior_map


def _medium_strategy_preds(
    xgb_pred: np.ndarray,
    hist_med: np.ndarray,
    strategy: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (accepted_mask, final_pred_for_all_medium_rows; NaN where rejected)."""
    n = len(xgb_pred)
    pred = np.full(n, np.nan, dtype=np.float64)
    accept = np.ones(n, dtype=bool)
    flag_gt_60 = xgb_pred > 60.0
    flag_gt_2x = xgb_pred > (2.0 * hist_med)

    if strategy == "A":
        pred[:] = xgb_pred
    elif strategy == "B":
        accept = ~flag_gt_60
        pred[accept] = xgb_pred[accept]
    elif strategy == "C":
        accept = ~flag_gt_2x
        pred[accept] = xgb_pred[accept]
    elif strategy == "D":
        pred[:] = xgb_pred
        pred[flag_gt_2x] = hist_med[flag_gt_2x]
        accept[:] = True
    elif strategy == "E":
        accept = ~(flag_gt_60 | flag_gt_2x)
        pred[accept] = xgb_pred[accept]
    else:
        raise ValueError(strategy)
    return accept, pred


def _evaluate_flow(
    *,
    labels: np.ndarray,
    hist_med: np.ndarray,
    xgb_pred: np.ndarray,
    actual: np.ndarray,
    strategy: str,
    path_a_n: int,
    test_n: int,
) -> Dict[str, Any]:
    high_m = labels == LABEL_HIGH
    med_m = labels == LABEL_MEDIUM
    uns_m = labels == LABEL_UNSTABLE

    high_pred = hist_med[high_m]
    high_act = actual[high_m]
    high_metrics = _metrics(high_pred, high_act)

    med_xgb = xgb_pred[med_m]
    med_hist = hist_med[med_m]
    med_act = actual[med_m]
    med_accept, med_pred_all = _medium_strategy_preds(med_xgb, med_hist, strategy)
    med_metrics = _metrics(med_pred_all[med_accept], med_act[med_accept])

    # Overall accepted = HIGH all + MEDIUM accepted
    overall_pred: List[float] = []
    overall_act: List[float] = []
    if high_m.sum():
        overall_pred.extend(high_pred.tolist())
        overall_act.extend(high_act.tolist())
    if med_accept.sum():
        overall_pred.extend(med_pred_all[med_accept].tolist())
        overall_act.extend(med_act[med_accept].tolist())
    overall = _metrics(np.asarray(overall_pred), np.asarray(overall_act))

    accepted = int(high_m.sum() + med_accept.sum())
    rejected = int(path_a_n - accepted)
    cov = _coverage_block(accepted, rejected, path_a_n, test_n)

    return {
        "medium_strategy": strategy,
        "counts": {
            "HIGH": int(high_m.sum()),
            "MEDIUM": int(med_m.sum()),
            "UNSTABLE": int(uns_m.sum()),
            "MEDIUM_accepted": int(med_accept.sum()),
            "MEDIUM_rejected_by_safety": int((~med_accept).sum()),
        },
        **cov,
        "HIGH": high_metrics,
        "MEDIUM": {
            **med_metrics,
            "gate_triggered_count": int((~med_accept).sum()) if strategy != "D" else 0,
            "fallback_replaced_count": int((med_xgb > (2.0 * med_hist)).sum()) if strategy == "D" else 0,
        },
        "UNSTABLE": {
            "count": int(uns_m.sum()),
            "predictions": 0,
            "note": "No automated prediction",
        },
        "overall_path_a_accepted": overall,
    }


def main() -> None:
    model_mtime_before = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    report_mtime_before = REPORT_PATH.stat().st_mtime if REPORT_PATH.exists() else None

    assert is_path_a_classifier_enabled() is False, "Path A feature flag must remain OFF for 17G"

    print("Loading test set...", flush=True)
    test = pd.read_parquet(PROC / "test.parquet")
    test = test[test[TARGET_COL] > 0].copy()
    test_n = int(len(test))

    history = pd.read_parquet(PROC / "purchase_history.parquet")
    test = _attach_regularity_from_history(test, history)
    test["historical_interval_norm_mad"] = test["norm_mad"]
    test["recent3_interval_median"] = test["recent3_median"]

    # Path A universe: established customers only
    path_a_mask = test["purchase_count_so_far"].astype(int) >= HYBRID_MIN_P
    path_a = test.loc[path_a_mask].copy().reset_index(drop=True)
    path_a_n = int(len(path_a))
    path_a["customerId"] = path_a["customerId"].astype(str)
    path_a["itemId"] = path_a["itemId"].astype(str)
    path_a["purchase_seq"] = path_a["purchase_count_so_far"].astype(int)

    keys = set(zip(path_a["customerId"], path_a["itemId"]))
    prior_map = _build_prior_map(history, keys)
    del history
    gc.collect()

    print("Scoring production XGBoost (read-only)...", flush=True)
    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipeline"]
    feats = list(bundle.get("features_numeric", NUMERIC_FEATURES)) + list(
        bundle.get("features_categorical", CATEGORICAL_FEATURES)
    )
    X = path_a.copy()
    for c in feats:
        if c not in X.columns:
            X[c] = np.nan
    xgb_all = np.clip(np.asarray(pipe.predict(X[feats]), dtype=np.float64), 1.0, None)
    del bundle, pipe, X
    gc.collect()

    # Classify each Path A row
    labels_17f: List[str] = []
    labels_v17d: List[str] = []
    hist_vals: List[float] = []
    for _, r in path_a.iterrows():
        key = (str(r["customerId"]), str(r["itemId"]), int(r["purchase_seq"]))
        prior = prior_map.get(key, [])
        hist_med = (
            float(r["historical_interval_median"])
            if pd.notna(r.get("historical_interval_median"))
            else float("nan")
        )
        rec = {
            "purchase_count_so_far": int(r["purchase_count_so_far"]),
            "historical_interval_median": hist_med,
            "historical_interval_norm_mad": float(r["norm_mad"]) if pd.notna(r.get("norm_mad")) else float("nan"),
            "cadence_drift": float(r["cadence_drift"]) if pd.notna(r.get("cadence_drift")) else float("nan"),
            "prior_intervals": prior,
        }
        labels_17f.append(classify_path_a(rec)["path_a_class"])
        labels_v17d.append(_hybrid_label(rec))
        if math.isnan(hist_med) and prior:
            hist_med = float(np.median(np.asarray(prior, dtype=np.float64)))
        hist_vals.append(hist_med)

    labels_17f_a = np.asarray(labels_17f, dtype=object)
    labels_v17d_a = np.asarray(labels_v17d, dtype=object)
    hist_med_a = np.asarray(hist_vals, dtype=np.float64)
    actual_a = path_a[TARGET_COL].to_numpy(dtype=np.float64)

    strategies = ["A", "B", "C", "D", "E"]
    strategy_names = {
        "A": "Current XGBoost",
        "B": "Reject prediction >60 days",
        "C": "Reject prediction >2× historical median",
        "D": "Fallback to historical median when prediction >2× median",
        "E": "Combined B+C",
    }

    path_a_17f = {
        s: _evaluate_flow(
            labels=labels_17f_a,
            hist_med=hist_med_a,
            xgb_pred=xgb_all,
            actual=actual_a,
            strategy=s,
            path_a_n=path_a_n,
            test_n=test_n,
        )
        for s in strategies
    }
    for s, block in path_a_17f.items():
        block["medium_strategy_label"] = strategy_names[s]

    # Baseline hybrid: same MEDIUM safety sweep for fair comparison
    baseline_v17d = {
        s: _evaluate_flow(
            labels=labels_v17d_a,
            hist_med=hist_med_a,
            xgb_pred=xgb_all,
            actual=actual_a,
            strategy=s,
            path_a_n=path_a_n,
            test_n=test_n,
        )
        for s in strategies
    }
    for s, block in baseline_v17d.items():
        block["medium_strategy_label"] = strategy_names[s]

    # Cohort counts (classifier only)
    def _count(labels: np.ndarray) -> Dict[str, int]:
        return {
            LABEL_HIGH: int((labels == LABEL_HIGH).sum()),
            LABEL_MEDIUM: int((labels == LABEL_MEDIUM).sum()),
            LABEL_UNSTABLE: int((labels == LABEL_UNSTABLE).sum()),
        }

    # Canonical pattern inspection
    canonical = {}
    for name, ivs in [
        ("stable", STABLE_CANONICAL),
        ("variable_recurring", VARIABLE_CANONICAL),
        ("unstable_chaotic", UNSTABLE_CANONICAL),
    ]:
        d17f = classify_path_a_from_intervals(ivs)
        d17d = _hybrid_label(
            {
                "purchase_count_so_far": len(ivs) + 1,
                "prior_intervals": ivs,
            }
        )
        hist = float(np.median(ivs))
        # Simulated XGB unknown — report classification + personal-median path only
        canonical[name] = {
            "intervals": ivs,
            "path_a_v17f_class": d17f["path_a_class"],
            "hybrid_v17d_class": d17d,
            "hist_median": hist,
            "norm_mad": d17f.get("norm_mad"),
            "cadence_drift": d17f.get("cadence_drift"),
            "max_over_median": d17f.get("max_over_median"),
            "if_HIGH_would_use": "personal_historical_median",
            "if_MEDIUM_would_compare": list(strategy_names.values()),
            "if_UNSTABLE_would_use": "no_prediction",
            "personal_median_vs_last_interval_note": (
                "Offline classification probe only; no held-out actual for synthetic pattern."
            ),
        }

    # Find real test rows similar to variable recurring (oracle-ish: 17F MEDIUM, Drift>10, NormMAD<=0.55)
    similar_mask = (
        (labels_17f_a == LABEL_MEDIUM)
        & (path_a["cadence_drift"].to_numpy(dtype=np.float64) > 10)
        & (path_a["cadence_drift"].to_numpy(dtype=np.float64) <= 15)
        & (path_a["norm_mad"].to_numpy(dtype=np.float64) <= 0.55)
    )
    similar_idx = np.where(similar_mask)[0][:8]
    variable_examples = []
    for i in similar_idx:
        key = (
            str(path_a.iloc[i]["customerId"]),
            str(path_a.iloc[i]["itemId"]),
            int(path_a.iloc[i]["purchase_seq"]),
        )
        prior = prior_map.get(key, [])[-12:]
        ae_med = abs(hist_med_a[i] - actual_a[i])
        ae_xgb = abs(xgb_all[i] - actual_a[i])
        variable_examples.append(
            {
                "customerId": key[0],
                "itemId": key[1],
                "intervals_tail": prior,
                "hist_median": round(float(hist_med_a[i]), 2),
                "xgb_pred": round(float(xgb_all[i]), 2),
                "actual": float(actual_a[i]),
                "abs_error_personal_median": round(float(ae_med), 2),
                "abs_error_xgb": round(float(ae_xgb), 2),
                "norm_mad": round(float(path_a.iloc[i]["norm_mad"]), 3)
                if pd.notna(path_a.iloc[i]["norm_mad"])
                else None,
                "drift": round(float(path_a.iloc[i]["cadence_drift"]), 2)
                if pd.notna(path_a.iloc[i]["cadence_drift"])
                else None,
                "v17d_label": labels_v17d_a[i],
                "v17f_label": labels_17f_a[i],
            }
        )

    # Comparison summary table helpers
    def _summary_row(flow: Dict[str, Any]) -> Dict[str, Any]:
        o = flow["overall_path_a_accepted"]
        return {
            "accepted": flow["accepted"],
            "rejected": flow["rejected"],
            "coverage_of_path_a_pct": flow["coverage_of_path_a_pct"],
            "coverage_of_test_pct": flow["coverage_of_test_pct"],
            "mae": o["mae"],
            "medae": o["medae"],
            "rmse": o["rmse"],
            "within_1_day_pct": o["within_1_day_pct"],
            "within_3_days_pct": o["within_3_days_pct"],
            "within_7_days_pct": o["within_7_days_pct"],
            "total_absolute_error": o["total_absolute_error"],
            "catastrophic_count": o["catastrophic_count"],
            "catastrophic_abs_error": o["catastrophic_abs_error"],
            "catastrophic_share_of_abs_error_pct": o["catastrophic_share_of_abs_error_pct"],
            "HIGH_mae": flow["HIGH"]["mae"],
            "HIGH_within_7_days_pct": flow["HIGH"]["within_7_days_pct"],
            "MEDIUM_accepted": flow["counts"]["MEDIUM_accepted"],
            "MEDIUM_mae": flow["MEDIUM"]["mae"],
            "MEDIUM_within_7_days_pct": flow["MEDIUM"]["within_7_days_pct"],
        }

    model_mtime_after = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    report_mtime_after = REPORT_PATH.stat().st_mtime if REPORT_PATH.exists() else None

    results = {
        "experiment": "PHASE_17G_PATH_A_END_TO_END_EVALUATION",
        "read_only": True,
        "path_b_implemented": False,
        "strategy_selection": "NONE — evaluate only, do not promote",
        "controls": {
            "test_rows_universe": test_n,
            "path_a_rows": path_a_n,
            "test_date_min": str(pd.to_datetime(test["invoice_date"]).min().date()),
            "test_date_max": str(pd.to_datetime(test["invoice_date"]).max().date()),
            "path_a_classifier": PATH_A_STRATEGY,
            "path_a_feature_flag_enabled": is_path_a_classifier_enabled(),
            "baseline_strategy": BASELINE_STRATEGY_NAME,
            "production_routing_replaced": False,
            "predictor_medium": "production refill_model.joblib (reg:squarederror, read-only)",
            "predictor_high": "personal historical median",
            "unstable": "no prediction",
            "catastrophic_definition": f"abs_error > {CATASTROPHIC_ABS_ERROR:.0f} days",
        },
        "cohort_counts": {
            "path_a_classifier_v17f": _count(labels_17f_a),
            "hybrid_routing_v17d": _count(labels_v17d_a),
        },
        "medium_strategies": strategy_names,
        "path_a_v17f_by_medium_strategy": path_a_17f,
        "baseline_v17d_by_medium_strategy": baseline_v17d,
        "summary_path_a_v17f": {s: _summary_row(path_a_17f[s]) for s in strategies},
        "summary_baseline_v17d": {s: _summary_row(baseline_v17d[s]) for s in strategies},
        "canonical_pattern_inspection": canonical,
        "variable_recurring_test_examples": variable_examples,
        "artifact_safety": {
            "refill_model_joblib_mtime_unchanged": model_mtime_before == model_mtime_after,
            "phase4_report_mtime_unchanged": report_mtime_before == report_mtime_after,
            "model_mtime_before": model_mtime_before,
            "model_mtime_after": model_mtime_after,
            "whatsapp_messages_sent": 0,
            "production_code_modified": False,
            "production_routing_modified": False,
            "path_a_feature_flag_remains_off": is_path_a_classifier_enabled() is False,
            "retrain": False,
            "path_b_implemented": False,
        },
    }

    out = PROC / "phase17g_path_a_evaluation_results.json"
    # Fix invalid walrus leftover if any — write clean JSON
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"Path A n={path_a_n} / test n={test_n}", flush=True)
    print("17F counts:", results["cohort_counts"]["path_a_classifier_v17f"], flush=True)
    print("v17d counts:", results["cohort_counts"]["hybrid_routing_v17d"], flush=True)
    for s in strategies:
        row = results["summary_path_a_v17f"][s]
        print(
            f"17F+{s}: accepted={row['accepted']} MAE={row['mae']} ±7d={row['within_7_days_pct']} "
            f"cat_abs={row['catastrophic_abs_error']}",
            flush=True,
        )
    print(f"Wrote {out}", flush=True)
    print("Artifact safety:", json.dumps(results["artifact_safety"]), flush=True)


if __name__ == "__main__":
    main()
