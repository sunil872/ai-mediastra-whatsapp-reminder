"""READ-ONLY Phase 17K Path A eligibility architecture audit.

Compares three eligibility architectures (all predictors = personal historical median
when accepted; no Path B; no production changes; feature flag remains OFF):

  A) Current 17J: 17F HIGH | (17F MEDIUM AND C6 AND Strategy E using XGB)
  B) Historical-only: 17F HIGH | 17F MEDIUM  (no XGB gate)
  C) History-stricter hybrid (prediction-time history only; thresholds from 17H C2∩C3,
     NOT re-optimized on this test): 17F HIGH | (17F MEDIUM AND NormMAD<=0.40
     AND Drift<=10)

Does NOT select or promote a policy. Does NOT modify production.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from refillcare.models.training import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COL
from refillcare.models.path_a_classifier import (
    LABEL_HIGH,
    LABEL_MEDIUM,
    LABEL_UNSTABLE,
    classify_path_a,
    classify_path_a_from_intervals,
    compute_path_a_interval_diagnostics,
    is_path_a_classifier_enabled,
)

PROC = ROOT / "data" / "refillcare" / "processed"
MODEL_PATH = PROC / "models" / "refill_model.joblib"
REPORT_PATH = PROC / "phase4_model_report.json"
CATASTROPHIC = 30.0
MIN_PURCHASES = 6
VARIABLE_CANONICAL = [30, 60, 45, 30, 90, 30, 60, 30, 90]

# Approach C thresholds — published in Phase 17H (C2 NormMAD<=0.40, C3 Drift<=10).
# Not re-tuned on validation/test in this phase.
C_NORM_MAD_MAX = 0.40
C_DRIFT_MAX = 10.0

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
    if int(mask.sum()) == 0:
        return {
            "count": 0,
            "mae": None,
            "medae": None,
            "rmse": None,
            "within_1_day_pct": None,
            "within_3_days_pct": None,
            "within_7_days_pct": None,
            "catastrophic_count": 0,
            "catastrophic_pct": None,
            "mean_bias": None,
            "total_absolute_error": 0.0,
        }
    p, a = pred[mask], actual[mask]
    err = p - a
    ae = np.abs(err)
    cat = ae > CATASTROPHIC
    return {
        "count": int(mask.sum()),
        "mae": round(float(ae.mean()), 2),
        "medae": round(float(np.median(ae)), 2),
        "rmse": round(float(np.sqrt(np.mean(err**2))), 2),
        "within_1_day_pct": round(100.0 * float((ae <= 1).mean()), 2),
        "within_3_days_pct": round(100.0 * float((ae <= 3).mean()), 2),
        "within_7_days_pct": round(100.0 * float((ae <= 7).mean()), 2),
        "catastrophic_count": int(cat.sum()),
        "catastrophic_pct": round(100.0 * float(cat.mean()), 2),
        "mean_bias": round(float(err.mean()), 2),
        "total_absolute_error": round(float(ae.sum()), 2),
    }


def _build_prior_map(history: pd.DataFrame, keys: set) -> Dict[Tuple[str, str, int], List[float]]:
    pairs = pd.DataFrame(list(keys), columns=["customerId", "itemId"])
    hist = history[
        ["customerId", "itemId", "invoice_date", "days_since_previous_purchase", "purchase_seq"]
    ].copy()
    hist["invoice_date"] = pd.to_datetime(hist["invoice_date"])
    hist["customerId"] = hist["customerId"].astype(str)
    hist["itemId"] = hist["itemId"].astype(str)
    hist = hist.merge(pairs, on=["customerId", "itemId"], how="inner")
    hist = hist.sort_values(["customerId", "itemId", "invoice_date", "purchase_seq"])
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


def _attach_regularity_filtered(frame: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["customerId"] = work["customerId"].astype(str)
    work["itemId"] = work["itemId"].astype(str)
    pairs = work[["customerId", "itemId"]].drop_duplicates()
    hist = history[
        ["customerId", "itemId", "invoice_date", "days_since_previous_purchase", "purchase_seq"]
    ].copy()
    hist["customerId"] = hist["customerId"].astype(str)
    hist["itemId"] = hist["itemId"].astype(str)
    hist = hist.merge(pairs, on=["customerId", "itemId"], how="inner")
    return _attach_regularity_from_history(work, hist)


def _policy_report(
    frame: pd.DataFrame,
    policy_col: str,
    universe_n: int,
    path_a_n: int,
) -> Dict[str, Any]:
    counts = {
        "HIGH": int((frame[policy_col] == LABEL_HIGH).sum()),
        "MEDIUM_SAFE": int((frame[policy_col] == "MEDIUM_SAFE").sum()),
        "MEDIUM_RISK": int((frame[policy_col] == "MEDIUM_RISK").sum()),
        "UNSTABLE": int((frame[policy_col] == LABEL_UNSTABLE).sum()),
    }
    auto = frame[policy_col].isin([LABEL_HIGH, "MEDIUM_SAFE"])
    pred = frame["hist_median"].where(auto)
    overall = _metrics(pred.to_numpy(), frame["actual"].to_numpy())
    by_cohort: Dict[str, Any] = {}
    for name in [LABEL_HIGH, "MEDIUM_SAFE", "MEDIUM_RISK", LABEL_UNSTABLE]:
        sub = frame[frame[policy_col] == name]
        is_auto = name in {LABEL_HIGH, "MEDIUM_SAFE"}
        by_cohort[name] = {
            "count": int(len(sub)),
            "automatic": is_auto,
            "metrics": _metrics(sub["hist_median"].to_numpy(), sub["actual"].to_numpy())
            if is_auto and len(sub)
            else None,
        }
    return {
        "counts": counts,
        "automatic_prediction_coverage": {
            "accepted": int(auto.sum()),
            "rejected": int((~auto).sum()),
            "coverage_of_path_a_pct": round(100.0 * float(auto.mean()), 2) if path_a_n else 0.0,
            "coverage_of_universe_pct": round(100.0 * float(auto.sum()) / universe_n, 2)
            if universe_n
            else 0.0,
        },
        "overall_accepted": overall,
        "by_cohort": by_cohort,
    }


def _evaluate_split(
    split_name: str,
    df_raw: pd.DataFrame,
    history: pd.DataFrame,
    pipe,
    feats: List[str],
) -> Dict[str, Any]:
    df = df_raw[df_raw[TARGET_COL] > 0].copy()
    universe_n = int(len(df))
    df = _attach_regularity_filtered(df, history)
    df["historical_interval_norm_mad"] = df["norm_mad"]

    path_a = df[df["purchase_count_so_far"].astype(int) >= MIN_PURCHASES].copy().reset_index(drop=True)
    path_a_n = int(len(path_a))
    path_a["customerId"] = path_a["customerId"].astype(str)
    path_a["itemId"] = path_a["itemId"].astype(str)
    path_a["purchase_seq"] = path_a["purchase_count_so_far"].astype(int)

    keys = set(zip(path_a["customerId"], path_a["itemId"]))
    prior_map = _build_prior_map(history, keys)

    X = path_a.copy()
    for c in feats:
        if c not in X.columns:
            X[c] = np.nan
    xgb_all = np.clip(np.asarray(pipe.predict(X[feats]), dtype=np.float64), 1.0, None)

    rows = []
    for idx, (_, r) in enumerate(path_a.iterrows()):
        key = (str(r["customerId"]), str(r["itemId"]), int(r["purchase_seq"]))
        prior = prior_map.get(key, [])
        hist_med = (
            float(r["historical_interval_median"])
            if pd.notna(r.get("historical_interval_median"))
            else float("nan")
        )
        nm = float(r["norm_mad"]) if pd.notna(r.get("norm_mad")) else float("nan")
        dr = float(r["cadence_drift"]) if pd.notna(r.get("cadence_drift")) else float("nan")
        rec = {
            "purchase_count_so_far": int(r["purchase_count_so_far"]),
            "historical_interval_median": hist_med,
            "historical_interval_norm_mad": nm,
            "cadence_drift": dr,
            "prior_intervals": prior,
        }
        label_17f = classify_path_a(rec)["path_a_class"]
        diag = compute_path_a_interval_diagnostics(prior)
        if math.isnan(hist_med):
            hist_med = float(diag["hist_median"]) if not math.isnan(diag["hist_median"]) else float("nan")
        if math.isnan(nm):
            nm = float(diag["norm_mad"])
        if math.isnan(dr):
            dr = float(diag["cadence_drift"])

        xgb = float(xgb_all[idx])
        actual = float(r[TARGET_COL])
        pred_over = (xgb / hist_med) if hist_med and hist_med > 0 else float("nan")
        passes_E = bool(np.isfinite(hist_med) and xgb <= 60.0 and xgb <= 2.0 * hist_med)
        passes_C6 = bool(np.isfinite(pred_over) and pred_over <= 1.5)
        passes_C_hist = bool(
            np.isfinite(nm) and np.isfinite(dr) and nm <= C_NORM_MAD_MAX and dr <= C_DRIFT_MAX
        )

        missing_stats = bool(
            math.isnan(hist_med)
            or math.isnan(nm)
            or math.isnan(dr)
            or math.isnan(diag["max_over_median"])
            or len(prior) < 5
        )

        def demote_missing(label: str) -> str:
            if label == LABEL_UNSTABLE:
                return LABEL_UNSTABLE
            return "MEDIUM_RISK"

        # --- Approach A: current 17J (XGB C6 + E) ---
        if missing_stats:
            policy_a = demote_missing(label_17f)
        elif label_17f == LABEL_HIGH:
            policy_a = LABEL_HIGH
        elif label_17f == LABEL_MEDIUM and passes_C6 and passes_E:
            policy_a = "MEDIUM_SAFE"
        elif label_17f == LABEL_MEDIUM:
            policy_a = "MEDIUM_RISK"
        else:
            policy_a = LABEL_UNSTABLE

        # --- Approach B: historical-only (all 17F MEDIUM is SAFE) ---
        if missing_stats:
            policy_b = demote_missing(label_17f)
        elif label_17f == LABEL_HIGH:
            policy_b = LABEL_HIGH
        elif label_17f == LABEL_MEDIUM:
            policy_b = "MEDIUM_SAFE"
        else:
            policy_b = LABEL_UNSTABLE

        # --- Approach C: 17H C2∩C3 history-stricter (no XGB) ---
        if missing_stats:
            policy_c = demote_missing(label_17f)
        elif label_17f == LABEL_HIGH:
            policy_c = LABEL_HIGH
        elif label_17f == LABEL_MEDIUM and passes_C_hist:
            policy_c = "MEDIUM_SAFE"
        elif label_17f == LABEL_MEDIUM:
            policy_c = "MEDIUM_RISK"
        else:
            policy_c = LABEL_UNSTABLE

        rows.append(
            {
                "label_17f": label_17f,
                "policy_a": policy_a,
                "policy_b": policy_b,
                "policy_c": policy_c,
                "hist_median": hist_med,
                "xgb": xgb,
                "actual": actual,
                "missing_stats": missing_stats,
                "passes_C6": passes_C6,
                "passes_E": passes_E,
                "passes_C_hist": passes_C_hist,
                "norm_mad": nm,
                "drift": dr,
                "pred_over_hist": pred_over,
            }
        )

    frame = pd.DataFrame(rows)
    return {
        "split": split_name,
        "date_min": str(pd.to_datetime(df_raw["invoice_date"]).min().date()),
        "date_max": str(pd.to_datetime(df_raw["invoice_date"]).max().date()),
        "universe_n": universe_n,
        "path_a_n": path_a_n,
        "approach_A_17j_xgb_gate": _policy_report(frame, "policy_a", universe_n, path_a_n),
        "approach_B_historical_only": _policy_report(frame, "policy_b", universe_n, path_a_n),
        "approach_C_history_stricter_17h_C2_C3": _policy_report(
            frame, "policy_c", universe_n, path_a_n
        ),
        "missing_stats_demotions": int(frame["missing_stats"].sum()),
        "overlap_notes": {
            "A_MEDIUM_SAFE_also_B": int(
                ((frame["policy_a"] == "MEDIUM_SAFE") & (frame["policy_b"] == "MEDIUM_SAFE")).sum()
            ),
            "B_MEDIUM_SAFE_not_A": int(
                ((frame["policy_b"] == "MEDIUM_SAFE") & (frame["policy_a"] != "MEDIUM_SAFE")).sum()
            ),
            "C_MEDIUM_SAFE_also_A": int(
                ((frame["policy_c"] == "MEDIUM_SAFE") & (frame["policy_a"] == "MEDIUM_SAFE")).sum()
            ),
            "A_MEDIUM_SAFE_not_C": int(
                ((frame["policy_a"] == "MEDIUM_SAFE") & (frame["policy_c"] != "MEDIUM_SAFE")).sum()
            ),
        },
    }


def _canonical_audit() -> Dict[str, Any]:
    diag = compute_path_a_interval_diagnostics(VARIABLE_CANONICAL)
    d17f = classify_path_a_from_intervals(VARIABLE_CANONICAL)
    hist = float(diag["hist_median"])
    nm = float(d17f.get("norm_mad") or diag["norm_mad"])
    dr = float(d17f.get("cadence_drift") or diag["cadence_drift"])
    passes_c_hist = bool(nm <= C_NORM_MAD_MAX and dr <= C_DRIFT_MAX)
    last = float(VARIABLE_CANONICAL[-1])

    def eligibility_for_xgb_ratio(ratio: float) -> Dict[str, Any]:
        xgb = hist * ratio
        passes_c6 = ratio <= 1.5
        passes_e = xgb <= 60.0 and xgb <= 2.0 * hist
        policy_a = (
            "MEDIUM_SAFE"
            if d17f["path_a_class"] == LABEL_MEDIUM and passes_c6 and passes_e
            else "MEDIUM_RISK"
            if d17f["path_a_class"] == LABEL_MEDIUM
            else d17f["path_a_class"]
        )
        policy_b = (
            "MEDIUM_SAFE"
            if d17f["path_a_class"] == LABEL_MEDIUM
            else d17f["path_a_class"]
        )
        policy_c = (
            "MEDIUM_SAFE"
            if d17f["path_a_class"] == LABEL_MEDIUM and passes_c_hist
            else "MEDIUM_RISK"
            if d17f["path_a_class"] == LABEL_MEDIUM
            else d17f["path_a_class"]
        )
        return {
            "xgb": round(xgb, 2),
            "xgb_over_hist": ratio,
            "A": {"policy": policy_a, "auto": policy_a in {LABEL_HIGH, "MEDIUM_SAFE"}},
            "B": {"policy": policy_b, "auto": policy_b in {LABEL_HIGH, "MEDIUM_SAFE"}},
            "C": {"policy": policy_c, "auto": policy_c in {LABEL_HIGH, "MEDIUM_SAFE"}},
        }

    return {
        "intervals": VARIABLE_CANONICAL,
        "path_a_17f_class": d17f["path_a_class"],
        "hist_median": hist,
        "norm_mad": nm,
        "drift": dr,
        "last_interval": last,
        "last_over_hist": round(last / hist, 4) if hist else None,
        "passes_C_hist_17h_C2_C3": passes_c_hist,
        "note_last_interval_gate_would_fail": (
            "A last-interval <=1.5x hist gate would reject this pattern "
            f"(last={last}, hist={hist}, ratio={last/hist:.2f}) even when the next "
            "cycle is calm — another reason not to use last-interval as a SAFE gate."
        ),
        "by_xgb_ratio": {
            "1.1x": eligibility_for_xgb_ratio(1.1),
            "1.5x": eligibility_for_xgb_ratio(1.5),
            "1.6x": eligibility_for_xgb_ratio(1.6),
            "2.1x": eligibility_for_xgb_ratio(2.1),
        },
        "summary": {
            "A_eligible_when_xgb_calm": True,
            "B_always_eligible_as_MEDIUM_SAFE": d17f["path_a_class"] == LABEL_MEDIUM,
            "C_eligible_via_history_stricter": passes_c_hist
            and d17f["path_a_class"] == LABEL_MEDIUM,
        },
    }


def main() -> None:
    assert is_path_a_classifier_enabled() is False
    model_mtime_before = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    report_mtime_before = REPORT_PATH.stat().st_mtime if REPORT_PATH.exists() else None

    print("Loading model + data...", flush=True)
    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipeline"]
    feats = list(bundle.get("features_numeric", NUMERIC_FEATURES)) + list(
        bundle.get("features_categorical", CATEGORICAL_FEATURES)
    )
    history = pd.read_parquet(PROC / "purchase_history.parquet")
    test = pd.read_parquet(PROC / "test.parquet")
    val = pd.read_parquet(PROC / "validation.parquet")

    print("Evaluating VALIDATION...", flush=True)
    val_res = _evaluate_split("validation", val, history, pipe, feats)
    print("Evaluating TEST...", flush=True)
    test_res = _evaluate_split("test", test, history, pipe, feats)

    canonical = _canonical_audit()

    architecture = {
        "question": (
            "MEDIUM-SAFE eligibility in 17J depends on production XGB, while the SAFE "
            "predictor is personal historical median. Is that gate logically safe and "
            "reproducible at real prediction time?"
        ),
        "approach_A": {
            "name": "Current 17J (XGB safety gate -> personal median)",
            "eligibility_rules": {
                "HIGH": "17F HIGH AND valid cadence stats -> personal historical median",
                "MEDIUM_SAFE": (
                    "17F MEDIUM AND XGB<=1.5*hist (C6) AND XGB<=60 AND XGB<=2*hist (E) "
                    "AND valid cadence stats -> personal historical median"
                ),
                "MEDIUM_RISK": "17F MEDIUM failing C6/E OR missing stats demotion -> no auto",
                "UNSTABLE": "17F UNSTABLE OR missing on unstable -> no auto",
            },
            "features_used_for_eligibility": [
                "prior intervals / NormMAD / Drift / max_over_median (17F)",
                "production XGB prediction (C6, Strategy E)",
                "historical_interval_median",
            ],
            "predictor_when_accepted": "personal_historical_median",
            "available_at_prediction_time": True,
            "future_leakage": False,
            "reproducibility_notes": [
                "XGB is available at real prediction time if the production model is scored.",
                "Eligibility is deterministic given the same model artifact + same features.",
                "Coupling: eligibility changes if the model is retrained even when history is unchanged.",
                "Architectural tension: gate uses a model prediction that is discarded for the output.",
            ],
            "logically_safe_assessment": (
                "Operationally reproducible at prediction time (no future leakage). "
                "Logically asymmetric: XGB is used only as a veto on inflation, not as the "
                "predictor. Safe if interpreted as a regime check (model stays near personal "
                "cadence); fragile if the model drifts or if inflation is not a reliable "
                "proxy for median error."
            ),
        },
        "approach_B": {
            "name": "Historical-only eligibility -> personal median",
            "eligibility_rules": {
                "HIGH": "17F HIGH AND valid cadence stats -> personal historical median",
                "MEDIUM_SAFE": "17F MEDIUM AND valid cadence stats -> personal historical median",
                "MEDIUM_RISK": "missing-stats demotion of HIGH/MEDIUM only -> no auto",
                "UNSTABLE": "17F UNSTABLE -> no auto",
            },
            "features_used_for_eligibility": [
                "prior intervals / NormMAD / Drift / band / max_over_median (17F only)",
            ],
            "predictor_when_accepted": "personal_historical_median",
            "available_at_prediction_time": True,
            "future_leakage": False,
            "reproducibility_notes": [
                "Fully aligned: eligibility and predictor both depend only on prior intervals.",
                "Independent of model version / retrain.",
                "Broader coverage: includes MEDIUM rows where XGB would inflate.",
            ],
            "logically_safe_assessment": (
                "Fully coherent and leakage-free. Tradeoff is accuracy: accepting all 17F "
                "MEDIUM without an inflation/regime veto typically worsens MAE / catastrophic "
                "rate versus A (see metrics)."
            ),
        },
        "approach_C": {
            "name": (
                "History-stricter hybrid (17H C2∩C3 NormMAD<=0.40 AND Drift<=10) "
                "-> personal median; no XGB"
            ),
            "eligibility_rules": {
                "HIGH": "17F HIGH AND valid cadence stats -> personal historical median",
                "MEDIUM_SAFE": (
                    f"17F MEDIUM AND NormMAD<={C_NORM_MAD_MAX} AND Drift<={C_DRIFT_MAX} "
                    "AND valid cadence stats -> personal historical median"
                ),
                "MEDIUM_RISK": "17F MEDIUM failing C2∩C3 OR missing demotion -> no auto",
                "UNSTABLE": "17F UNSTABLE -> no auto",
            },
            "features_used_for_eligibility": [
                "17F Path A class (history)",
                "NormMAD",
                "cadence Drift",
            ],
            "predictor_when_accepted": "personal_historical_median",
            "available_at_prediction_time": True,
            "future_leakage": False,
            "threshold_source": (
                "Phase 17H candidate rules C2 (NormMAD<=0.40) and C3 (Drift<=10); "
                "combined here without re-optimizing on val/test."
            ),
            "reproducibility_notes": [
                "No model dependency; eligibility matches median predictor's information set.",
                "Stricter than B; may still include XGB-inflating rows and exclude variable "
                "recurrers with elevated Drift even when XGB is calm.",
            ],
            "logically_safe_assessment": (
                "Coherent history-only SAFE pocket. Justified by prior 17H analysis, not by "
                "new test-set tuning. May conflict with the canonical variable pattern if "
                "Drift exceeds 10."
            ),
        },
        "no_policy_selected": True,
        "no_promotion": True,
    }

    model_mtime_after = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    report_mtime_after = REPORT_PATH.stat().st_mtime if REPORT_PATH.exists() else None

    out = {
        "experiment": "PHASE_17K_PATH_A_ELIGIBILITY_ARCHITECTURE_AUDIT",
        "read_only": True,
        "strategy_promoted": False,
        "policy_selected": False,
        "architecture": architecture,
        "validation": val_res,
        "test": test_res,
        "canonical_variable_pattern": canonical,
        "artifact_safety": {
            "refill_model_joblib_mtime_unchanged": model_mtime_before == model_mtime_after,
            "phase4_report_mtime_unchanged": report_mtime_before == report_mtime_after,
            "model_mtime": model_mtime_after,
            "whatsapp_messages_sent": 0,
            "production_routing_modified": False,
            "feature_flag_remains_off": is_path_a_classifier_enabled() is False,
            "retrain": False,
            "path_b_implemented": False,
        },
    }

    out_path = PROC / "phase17k_path_a_eligibility_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)

    for name, res in [("validation", val_res), ("test", test_res)]:
        a = res["approach_A_17j_xgb_gate"]
        b = res["approach_B_historical_only"]
        c = res["approach_C_history_stricter_17h_C2_C3"]
        print(
            f"{name}: A cov={a['automatic_prediction_coverage']['coverage_of_universe_pct']}% "
            f"MAE={a['overall_accepted']['mae']} +-7={a['overall_accepted']['within_7_days_pct']} "
            f"| B cov={b['automatic_prediction_coverage']['coverage_of_universe_pct']}% "
            f"MAE={b['overall_accepted']['mae']} +-7={b['overall_accepted']['within_7_days_pct']} "
            f"| C cov={c['automatic_prediction_coverage']['coverage_of_universe_pct']}% "
            f"MAE={c['overall_accepted']['mae']} +-7={c['overall_accepted']['within_7_days_pct']}",
            flush=True,
        )
    print(
        "Canonical:",
        f"17F={canonical['path_a_17f_class']}",
        f"C_hist={canonical['passes_C_hist_17h_C2_C3']}",
        f"summary={canonical['summary']}",
        flush=True,
    )
    print("Artifact safety:", json.dumps(out["artifact_safety"]), flush=True)


if __name__ == "__main__":
    main()
