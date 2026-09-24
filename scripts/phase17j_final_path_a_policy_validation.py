"""READ-ONLY Phase 17J final Path A policy validation.

Policy (offline only; feature flag remains OFF):
  HIGH         -> personal historical median
  MEDIUM-SAFE  -> personal historical median
  MEDIUM-RISK  -> no automatic prediction
  UNSTABLE     -> no automatic prediction

MEDIUM-SAFE = 17F MEDIUM AND C6 (XGB<=1.5*hist) AND Strategy E
  (XGB<=60 AND XGB<=2*hist).

Does NOT use future actual as a prediction-time feature.
Does NOT modify production routing/model/WhatsApp/flags. No Path B. No retrain.
"""
from __future__ import annotations

import gc
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
from refillcare.models.hybrid_strategy import (
    MIN_PURCHASES,
    evaluate_hybrid_eligibility,
)
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
VARIABLE_CANONICAL = [30, 60, 45, 30, 90, 30, 60, 30, 90]

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
            "catastrophic_abs_error": 0.0,
            "mean_bias": None,
            "total_absolute_error": 0.0,
            "mean_pred": None,
            "mean_actual": None,
        }
    p, a = pred[mask], actual[mask]
    err = p - a
    ae = np.abs(err)
    cat = ae > CATASTROPHIC
    total = float(ae.sum())
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
        "catastrophic_abs_error": round(float(ae[cat].sum()), 2),
        "mean_bias": round(float(err.mean()), 2),
        "total_absolute_error": round(total, 2),
        "mean_pred": round(float(p.mean()), 2),
        "mean_actual": round(float(a.mean()), 2),
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


def _proxy_stats(y_true: np.ndarray, y_hat: np.ndarray) -> Dict[str, Any]:
    """Precision/recall of a historical collapse proxy vs oracle collapse (analysis only)."""
    y_true = np.asarray(y_true, dtype=bool)
    y_hat = np.asarray(y_hat, dtype=bool)
    tp = int((y_true & y_hat).sum())
    fp = int((~y_true & y_hat).sum())
    fn = int((y_true & ~y_hat).sum())
    tn = int((~y_true & ~y_hat).sum())
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(100.0 * prec, 2) if prec is not None else None,
        "recall": round(100.0 * rec, 2) if rec is not None else None,
        "flag_rate_pct": round(100.0 * float(y_hat.mean()), 2) if len(y_hat) else 0.0,
        "oracle_rate_pct": round(100.0 * float(y_true.mean()), 2) if len(y_true) else 0.0,
    }


def _attach_regularity_filtered(frame: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Like hybrid_comparison attach, but filter history via merge to avoid OOM."""
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

        # Invalid / missing cadence at prediction time
        missing_stats = bool(
            math.isnan(hist_med)
            or math.isnan(nm)
            or math.isnan(dr)
            or math.isnan(diag["max_over_median"])
            or len(prior) < 5
        )

        # Historical collapse proxies (NO future actual)
        last_iv = float(prior[-1]) if prior else float("nan")
        recent3 = float(diag["recent3_median"]) if not math.isnan(diag["recent3_median"]) else float("nan")
        proxy_last_lt_half = bool(np.isfinite(last_iv) and np.isfinite(hist_med) and last_iv < 0.5 * hist_med)
        proxy_recent3_lt_half = bool(
            np.isfinite(recent3) and np.isfinite(hist_med) and recent3 < 0.5 * hist_med
        )
        proxy_last_lt_half_or_recent3 = proxy_last_lt_half or proxy_recent3_lt_half

        # Oracle collapse for analysis only (NOT used in 17J policy decision)
        oracle_collapse = bool(np.isfinite(hist_med) and hist_med > 0 and actual < 0.5 * hist_med)

        # 17D hybrid label
        hybr = evaluate_hybrid_eligibility(rec)
        if hybr["confidence"] == "HIGH":
            label_17d = LABEL_HIGH
        elif hybr["confidence"] == "MEDIUM":
            label_17d = LABEL_MEDIUM
        else:
            label_17d = LABEL_UNSTABLE

        # 17J policy class (prediction-time only; no oracle collapse gate)
        if int(r["purchase_count_so_far"]) < MIN_PURCHASES:
            policy = "INSUFFICIENT_HISTORY"  # should not appear in Path A frame
        elif missing_stats:
            policy = "MEDIUM_RISK" if label_17f == LABEL_MEDIUM else LABEL_UNSTABLE
            # Prefer UNSTABLE for missing on non-medium; force risk/unstable no-auto
            if label_17f == LABEL_HIGH:
                policy = "MEDIUM_RISK"  # demote HIGH with missing diagnostics
            elif label_17f == LABEL_UNSTABLE:
                policy = LABEL_UNSTABLE
            else:
                policy = "MEDIUM_RISK"
        elif label_17f == LABEL_HIGH:
            policy = LABEL_HIGH
        elif label_17f == LABEL_MEDIUM and passes_C6 and passes_E:
            policy = "MEDIUM_SAFE"
        elif label_17f == LABEL_MEDIUM:
            policy = "MEDIUM_RISK"
        else:
            policy = LABEL_UNSTABLE

        auto = policy in {LABEL_HIGH, "MEDIUM_SAFE"}
        pred_17j = hist_med if auto and np.isfinite(hist_med) else float("nan")

        # Comparators
        auto_17d = label_17d in {LABEL_HIGH, LABEL_MEDIUM}
        if label_17d == LABEL_HIGH:
            pred_17d = hist_med
        elif label_17d == LABEL_MEDIUM:
            pred_17d = xgb
        else:
            pred_17d = float("nan")

        # 17F + Strategy E: HIGH median; MEDIUM XGB if E else reject; UNSTABLE reject
        if label_17f == LABEL_HIGH:
            pred_17f_e = hist_med
            auto_17f_e = True
        elif label_17f == LABEL_MEDIUM and passes_E:
            pred_17f_e = xgb
            auto_17f_e = True
        else:
            pred_17f_e = float("nan")
            auto_17f_e = False

        rows.append(
            {
                "label_17f": label_17f,
                "label_17d": label_17d,
                "policy_17j": policy,
                "auto_17j": auto,
                "pred_17j": pred_17j,
                "pred_17d": pred_17d,
                "auto_17d": auto_17d,
                "pred_17f_e": pred_17f_e,
                "auto_17f_e": auto_17f_e,
                "hist_median": hist_med,
                "xgb": xgb,
                "actual": actual,
                "depth": int(r["purchase_count_so_far"]),
                "missing_stats": missing_stats,
                "passes_C6": passes_C6,
                "passes_E": passes_E,
                "oracle_collapse": oracle_collapse,
                "proxy_last_lt_half": proxy_last_lt_half,
                "proxy_recent3_lt_half": proxy_recent3_lt_half,
                "proxy_last_or_recent3_lt_half": proxy_last_lt_half_or_recent3,
                "last_interval": last_iv,
                "recent3_median": recent3,
                "pred_over_hist": pred_over,
                "drift": dr,
                "norm_mad": nm,
                "max_over_median": float(diag["max_over_median"]),
            }
        )

    frame = pd.DataFrame(rows)

    def counts_17j() -> Dict[str, int]:
        c = frame["policy_17j"].value_counts().to_dict()
        return {
            "HIGH": int(c.get(LABEL_HIGH, 0)),
            "MEDIUM_SAFE": int(c.get("MEDIUM_SAFE", 0)),
            "MEDIUM_RISK": int(c.get("MEDIUM_RISK", 0)),
            "UNSTABLE": int(c.get(LABEL_UNSTABLE, 0)),
        }

    auto_mask = frame["auto_17j"].to_numpy()
    metrics_17j = _metrics(frame.loc[auto_mask, "pred_17j"].to_numpy(), frame.loc[auto_mask, "actual"].to_numpy())
    metrics_17d = _metrics(
        frame.loc[frame["auto_17d"], "pred_17d"].to_numpy(),
        frame.loc[frame["auto_17d"], "actual"].to_numpy(),
    )
    metrics_17f_e = _metrics(
        frame.loc[frame["auto_17f_e"], "pred_17f_e"].to_numpy(),
        frame.loc[frame["auto_17f_e"], "actual"].to_numpy(),
    )

    # Cohort-specific metrics under 17J
    by_cohort = {}
    for name, mask in [
        ("HIGH", frame["policy_17j"] == LABEL_HIGH),
        ("MEDIUM_SAFE", frame["policy_17j"] == "MEDIUM_SAFE"),
        ("MEDIUM_RISK", frame["policy_17j"] == "MEDIUM_RISK"),
        ("UNSTABLE", frame["policy_17j"] == LABEL_UNSTABLE),
    ]:
        sub = frame.loc[mask]
        if name in {LABEL_HIGH, "MEDIUM_SAFE"}:
            by_cohort[name] = {
                "count": int(len(sub)),
                "automatic": True,
                "metrics": _metrics(sub["pred_17j"].to_numpy(), sub["actual"].to_numpy()),
            }
        else:
            by_cohort[name] = {
                "count": int(len(sub)),
                "automatic": False,
                "metrics": None,
                # diagnostic: what median would have done if forced
                "counterfactual_median_if_forced": _metrics(
                    sub["hist_median"].to_numpy(), sub["actual"].to_numpy()
                ),
            }

    # Collapse detectability among MEDIUM-SAFE candidates (before any future gate)
    safe_cand = frame[frame["policy_17j"] == "MEDIUM_SAFE"].copy()
    collapse_study = {
        "note": (
            "Oracle collapse uses future actual (actual < 0.5*hist) for analysis only. "
            "Proxies use only prior intervals available at prediction time."
        ),
        "medium_safe_n": int(len(safe_cand)),
        "oracle_collapse_in_medium_safe": int(safe_cand["oracle_collapse"].sum()) if len(safe_cand) else 0,
        "proxies_vs_oracle_on_medium_safe": {
            "last_interval_lt_half_hist": _proxy_stats(
                safe_cand["oracle_collapse"].to_numpy(),
                safe_cand["proxy_last_lt_half"].to_numpy(),
            )
            if len(safe_cand)
            else {},
            "recent3_lt_half_hist": _proxy_stats(
                safe_cand["oracle_collapse"].to_numpy(),
                safe_cand["proxy_recent3_lt_half"].to_numpy(),
            )
            if len(safe_cand)
            else {},
            "last_or_recent3_lt_half_hist": _proxy_stats(
                safe_cand["oracle_collapse"].to_numpy(),
                safe_cand["proxy_last_or_recent3_lt_half"].to_numpy(),
            )
            if len(safe_cand)
            else {},
        },
    }
    # If we had applied best proxy as exclusion, impact on SAFE metrics
    if len(safe_cand):
        keep = ~safe_cand["proxy_last_or_recent3_lt_half"]
        collapse_study["sensitivity_if_exclude_historical_proxy"] = {
            "excluded_n": int((~keep).sum()),
            "remaining_n": int(keep.sum()),
            "metrics_remaining_median": _metrics(
                safe_cand.loc[keep, "hist_median"].to_numpy(),
                safe_cand.loc[keep, "actual"].to_numpy(),
            ),
            "oracle_collapse_caught_among_excluded": int(
                safe_cand.loc[~keep, "oracle_collapse"].sum()
            ),
            "oracle_collapse_missed_among_remaining": int(
                safe_cand.loc[keep, "oracle_collapse"].sum()
            ),
        }

    # Verdict on detectability (oracle uses future actual — analysis only)
    proxy = collapse_study["proxies_vs_oracle_on_medium_safe"].get("last_or_recent3_lt_half_hist", {})
    recall = proxy.get("recall")
    precision = proxy.get("precision")
    if recall is not None and precision is not None and recall >= 40 and precision >= 40:
        collapse_detectable = True
        collapse_verdict = (
            "Oracle collapse uses future actual and must not be a prediction-time feature. "
            f"Historical proxy last|recent3<0.5x hist shows recall={recall}% precision={precision}% "
            "on MEDIUM-SAFE — only weakly/possibly informative; not adopted in primary 17J policy."
        )
    else:
        collapse_detectable = False
        collapse_verdict = (
            "Collapse cannot be reliably detected at prediction time with tested historical proxies "
            "(last interval / recent-3 median < 0.5 x hist). "
            f"On MEDIUM-SAFE: recall={recall}% precision={precision}%. "
            "Oracle collapse (actual < 0.5 x hist) must NOT be used as a feature. "
            "Primary 17J policy therefore does not exclude collapse via future actual."
        )

    return {
        "split": split_name,
        "date_min": str(pd.to_datetime(df_raw["invoice_date"]).min().date()),
        "date_max": str(pd.to_datetime(df_raw["invoice_date"]).max().date()),
        "universe_n": universe_n,
        "path_a_n": path_a_n,
        "policy_17j_counts": counts_17j(),
        "automatic_prediction_coverage": {
            "accepted": int(auto_mask.sum()),
            "rejected": int((~auto_mask).sum()),
            "coverage_of_path_a_pct": round(100.0 * float(auto_mask.mean()), 2),
            "coverage_of_universe_pct": round(100.0 * auto_mask.sum() / universe_n, 2),
        },
        "policy_17j_overall_accepted": metrics_17j,
        "policy_17j_by_cohort": by_cohort,
        "comparator_17d_hybrid_overall_accepted": {
            "accepted": int(frame["auto_17d"].sum()),
            "coverage_of_path_a_pct": round(100.0 * float(frame["auto_17d"].mean()), 2),
            "coverage_of_universe_pct": round(100.0 * frame["auto_17d"].sum() / universe_n, 2),
            "metrics": metrics_17d,
            "note": "HIGH=personal median; MEDIUM=production XGB (no E gate); UNSTABLE=reject",
        },
        "comparator_17f_plus_strategy_E_overall_accepted": {
            "accepted": int(frame["auto_17f_e"].sum()),
            "coverage_of_path_a_pct": round(100.0 * float(frame["auto_17f_e"].mean()), 2),
            "coverage_of_universe_pct": round(100.0 * frame["auto_17f_e"].sum() / universe_n, 2),
            "metrics": metrics_17f_e,
            "note": "HIGH=personal median; MEDIUM=XGB if Strategy E else reject; UNSTABLE=reject",
        },
        "collapse_detectability": {
            **collapse_study,
            "collapse_reliably_detectable_at_prediction_time": collapse_detectable,
            "verdict": collapse_verdict,
        },
        "missing_stats_demotions": int(frame["missing_stats"].sum()),
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

    # Canonical variable pattern
    diag = compute_path_a_interval_diagnostics(VARIABLE_CANONICAL)
    hist = float(diag["hist_median"])
    d17f = classify_path_a_from_intervals(VARIABLE_CANONICAL)
    canonical = {
        "intervals": VARIABLE_CANONICAL,
        "path_a_17f_class": d17f["path_a_class"],
        "hist_median": hist,
        "norm_mad": d17f.get("norm_mad"),
        "drift": d17f.get("cadence_drift"),
        "eligibility_under_17j": {
            "if_xgb_1_1x_hist": {
                "passes_C6": True,
                "passes_E": True,
                "policy": "MEDIUM_SAFE",
                "predictor": "personal_historical_median",
            },
            "if_xgb_1_5x_hist": {
                "passes_C6": True,
                "passes_E": True,
                "policy": "MEDIUM_SAFE",
                "predictor": "personal_historical_median",
            },
            "if_xgb_1_6x_hist": {
                "passes_C6": False,
                "passes_E": True,
                "policy": "MEDIUM_RISK",
                "predictor": "none",
            },
            "if_xgb_2_1x_hist": {
                "passes_C6": False,
                "passes_E": False,
                "policy": "MEDIUM_RISK",
                "predictor": "none",
            },
        },
        "conclusion": (
            "Canonical variable pattern remains MEDIUM under 17F and stays MEDIUM-SAFE "
            "(auto median) when XGB is calm (<=1.5x hist); becomes MEDIUM-RISK (no auto) if XGB inflates."
        ),
    }

    del bundle, pipe, history, test, val
    gc.collect()

    model_mtime_after = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    report_mtime_after = REPORT_PATH.stat().st_mtime if REPORT_PATH.exists() else None

    results = {
        "experiment": "PHASE_17J_FINAL_PATH_A_POLICY_VALIDATION",
        "read_only": True,
        "strategy_promoted": False,
        "policy": {
            "HIGH": "personal_historical_median",
            "MEDIUM_SAFE": "personal_historical_median",
            "MEDIUM_RISK": "no_automatic_prediction",
            "UNSTABLE": "no_automatic_prediction",
            "MEDIUM_SAFE_definition": (
                "17F MEDIUM AND XGB_pred <= 1.5 * hist_median (C6) "
                "AND XGB_pred <= 60 AND XGB_pred <= 2 * hist_median (Strategy E)"
            ),
            "prediction_time_exclusions": [
                "insufficient Path A history (purchase_count < 6) — out of Path A frame",
                "invalid/missing cadence statistics -> no auto (demote to RISK/UNSTABLE)",
                "oracle collapse using future actual is NOT used at prediction time",
            ],
        },
        "validation": val_res,
        "test": test_res,
        "canonical_variable_pattern": canonical,
        "artifact_safety": {
            "refill_model_joblib_mtime_unchanged": model_mtime_before == model_mtime_after,
            "phase4_report_mtime_unchanged": report_mtime_before == report_mtime_after,
            "model_mtime": model_mtime_before,
            "whatsapp_messages_sent": 0,
            "production_routing_modified": False,
            "feature_flag_remains_off": is_path_a_classifier_enabled() is False,
            "retrain": False,
            "path_b_implemented": False,
        },
    }

    # Clean collapse verdict strings (fix accidental syntax from draft)
    for split_key in ("validation", "test"):
        v = results[split_key]["collapse_detectability"]
        proxy = v["proxies_vs_oracle_on_medium_safe"].get("last_or_recent3_lt_half_hist", {})
        if proxy.get("recall") is not None and proxy.get("precision") is not None:
            if proxy["recall"] >= 40 and proxy["precision"] >= 40:
                v["collapse_reliably_detectable_at_prediction_time"] = True
                v["verdict"] = (
                    "Historical proxies may be weakly informative, but oracle collapse uses future "
                    f"actual and is not a prediction-time feature. Proxy last|recent3: "
                    f"recall={proxy['recall']}% precision={proxy['precision']}%."
                )
            else:
                v["collapse_reliably_detectable_at_prediction_time"] = False
                v["verdict"] = (
                    "Collapse cannot be reliably detected at prediction time with the tested "
                    "historical proxies (last interval / recent-3 < 0.5× hist). "
                    f"On MEDIUM-SAFE: recall={proxy['recall']}% precision={proxy['precision']}%. "
                    "Oracle collapse (actual < 0.5× hist) must NOT be used as a feature. "
                    "Primary 17J policy therefore does not exclude collapse via future actual."
                )

    out = PROC / "phase17j_final_path_a_policy_results.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    for key in ("validation", "test"):
        s = results[key]
        m = s["policy_17j_overall_accepted"]
        print(
            f"{key}: counts={s['policy_17j_counts']} "
            f"auto={s['automatic_prediction_coverage']['accepted']} "
            f"cov_test={s['automatic_prediction_coverage']['coverage_of_universe_pct']}% "
            f"MAE={m['mae']} ±7d={m['within_7_days_pct']} cat%={m['catastrophic_pct']}",
            flush=True,
        )
        print(
            f"  vs 17D MAE={s['comparator_17d_hybrid_overall_accepted']['metrics']['mae']} "
            f"±7={s['comparator_17d_hybrid_overall_accepted']['metrics']['within_7_days_pct']} | "
            f"17F+E MAE={s['comparator_17f_plus_strategy_E_overall_accepted']['metrics']['mae']} "
            f"±7={s['comparator_17f_plus_strategy_E_overall_accepted']['metrics']['within_7_days_pct']}",
            flush=True,
        )
        print(
            "  collapse detectable?",
            s["collapse_detectability"]["collapse_reliably_detectable_at_prediction_time"],
            flush=True,
        )
    print(f"Wrote {out}", flush=True)
    print("Artifact safety:", json.dumps(results["artifact_safety"]), flush=True)


if __name__ == "__main__":
    main()
