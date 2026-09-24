"""READ-ONLY Phase 17H MEDIUM subclassification analysis (Path A only).

Splits 17F MEDIUM into candidate MEDIUM-SAFE vs MEDIUM-RISK using simple
rule-based gates. Does NOT promote/implement, retrain, touch WhatsApp,
production routing, or Path B. Feature flag remains OFF.
"""
from __future__ import annotations

import gc
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from refillcare.models.training import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COL
from refillcare.models.hybrid_strategy import MIN_PURCHASES
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
            "total_absolute_error": 0.0,
            "catastrophic_count": 0,
            "catastrophic_pct": None,
            "catastrophic_abs_error": 0.0,
            "catastrophic_share_of_abs_error_pct": None,
            "mean_bias": None,
            "mean_pred": None,
            "mean_actual": None,
        }
    p, a = pred[mask], actual[mask]
    err = p - a
    ae = np.abs(err)
    cat = ae > CATASTROPHIC
    total = float(ae.sum())
    cat_abs = float(ae[cat].sum())
    return {
        "count": int(mask.sum()),
        "mae": round(float(ae.mean()), 2),
        "medae": round(float(np.median(ae)), 2),
        "rmse": round(float(np.sqrt(np.mean(err**2))), 2),
        "within_1_day_pct": round(100.0 * float((ae <= 1).mean()), 2),
        "within_3_days_pct": round(100.0 * float((ae <= 3).mean()), 2),
        "within_7_days_pct": round(100.0 * float((ae <= 7).mean()), 2),
        "total_absolute_error": round(total, 2),
        "catastrophic_count": int(cat.sum()),
        "catastrophic_pct": round(100.0 * float(cat.mean()), 2),
        "catastrophic_abs_error": round(cat_abs, 2),
        "catastrophic_share_of_abs_error_pct": round(100.0 * cat_abs / total if total else 0.0, 2),
        "mean_bias": round(float(err.mean()), 2),
        "mean_pred": round(float(p.mean()), 2),
        "mean_actual": round(float(a.mean()), 2),
    }


def _dist(s: pd.Series) -> Dict[str, float]:
    x = s.dropna().astype(float)
    if len(x) == 0:
        return {}
    qs = x.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "mean": round(float(x.mean()), 2),
        "std": round(float(x.std()), 2),
        "p05": round(float(qs.loc[0.05]), 2),
        "p25": round(float(qs.loc[0.25]), 2),
        "p50": round(float(qs.loc[0.5]), 2),
        "p75": round(float(qs.loc[0.75]), 2),
        "p95": round(float(qs.loc[0.95]), 2),
    }


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


def main() -> None:
    assert is_path_a_classifier_enabled() is False
    model_mtime_before = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    report_mtime_before = REPORT_PATH.stat().st_mtime if REPORT_PATH.exists() else None

    print("Loading data...", flush=True)
    test = pd.read_parquet(PROC / "test.parquet")
    test = test[test[TARGET_COL] > 0].copy()
    test_n = int(len(test))
    history = pd.read_parquet(PROC / "purchase_history.parquet")
    test = _attach_regularity_from_history(test, history)
    test["historical_interval_norm_mad"] = test["norm_mad"]

    path_a = test[test["purchase_count_so_far"].astype(int) >= MIN_PURCHASES].copy().reset_index(drop=True)
    path_a_n = int(len(path_a))
    path_a["customerId"] = path_a["customerId"].astype(str)
    path_a["itemId"] = path_a["itemId"].astype(str)
    path_a["purchase_seq"] = path_a["purchase_count_so_far"].astype(int)

    keys = set(zip(path_a["customerId"], path_a["itemId"]))
    prior_map = _build_prior_map(history, keys)
    del history
    gc.collect()

    print("Scoring production XGB (read-only)...", flush=True)
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

    # Classify + diagnostics
    rows = []
    for idx, (_, r) in enumerate(path_a.iterrows()):
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
        decision = classify_path_a(rec)
        diag = compute_path_a_interval_diagnostics(prior)
        if math.isnan(hist_med):
            hist_med = float(diag["hist_median"]) if not math.isnan(diag["hist_median"]) else float("nan")
        if prior:
            hist_mean = float(np.asarray(prior, dtype=np.float64).mean())
        else:
            hist_mean = float("nan")

        xgb = float(xgb_all[idx])
        actual = float(r[TARGET_COL])
        rows.append(
            {
                "customerId": key[0],
                "itemId": key[1],
                "purchase_count_so_far": int(r["purchase_count_so_far"]),
                "label_17f": decision["path_a_class"],
                "hist_median": hist_med,
                "hist_mean": hist_mean,
                "norm_mad": float(diag["norm_mad"]) if not math.isnan(diag["norm_mad"]) else (
                    float(r["norm_mad"]) if pd.notna(r.get("norm_mad")) else float("nan")
                ),
                "drift": float(diag["cadence_drift"]) if not math.isnan(diag["cadence_drift"]) else (
                    float(r["cadence_drift"]) if pd.notna(r.get("cadence_drift")) else float("nan")
                ),
                "max_over_median": float(diag["max_over_median"]),
                "pct_in_band": float(diag["pct_in_refill_band_15_120"]),
                "pct_within_50": float(diag["pct_within_50pct_of_median"]),
                "pct_extreme": float(diag["pct_extreme_lt10_or_gt180"]),
                "recent3_median": float(diag["recent3_median"]),
                "xgb_pred": xgb,
                "actual": actual,
                "pred_over_hist": (xgb / hist_med) if hist_med and hist_med > 0 else float("nan"),
                "abs_pred_minus_hist": abs(xgb - hist_med) if hist_med == hist_med else float("nan"),
                "bias_xgb": xgb - actual,
                "abs_err_xgb": abs(xgb - actual),
                "abs_err_hist": abs(hist_med - actual) if hist_med == hist_med else float("nan"),
                "passes_E": bool((xgb <= 60.0) and (hist_med == hist_med) and (xgb <= 2.0 * hist_med)),
                "intervals_tail": prior[-12:],
            }
        )

    df = pd.DataFrame(rows)
    high = df[df["label_17f"] == LABEL_HIGH].copy()
    med = df[df["label_17f"] == LABEL_MEDIUM].copy().reset_index(drop=True)
    med_n = int(len(med))

    # Strategy E prediction on MEDIUM
    med["pred_E"] = np.where(med["passes_E"], med["xgb_pred"], np.nan)
    med_E = med[med["passes_E"]].copy()
    baseline_E = _metrics(med_E["pred_E"].to_numpy(), med_E["actual"].to_numpy())
    high_metrics = _metrics(high["hist_median"].to_numpy(), high["actual"].to_numpy())

    # Feature distributions on MEDIUM (+ catastrophic subset)
    cat_mask = med["abs_err_xgb"] > CATASTROPHIC
    profile = {
        "medium_n": med_n,
        "medium_passes_E": int(med["passes_E"].sum()),
        "medium_fails_E": int((~med["passes_E"]).sum()),
        "distributions_all_medium": {
            "depth": _dist(med["purchase_count_so_far"]),
            "hist_median": _dist(med["hist_median"]),
            "hist_mean": _dist(med["hist_mean"]),
            "norm_mad": _dist(med["norm_mad"]),
            "drift": _dist(med["drift"]),
            "max_over_median": _dist(med["max_over_median"]),
            "pct_in_band": _dist(med["pct_in_band"]),
            "pct_within_50": _dist(med["pct_within_50"]),
            "pct_extreme": _dist(med["pct_extreme"]),
            "pred_over_hist": _dist(med["pred_over_hist"]),
            "xgb_pred": _dist(med["xgb_pred"]),
            "actual": _dist(med["actual"]),
            "bias_xgb": _dist(med["bias_xgb"]),
            "abs_err_xgb": _dist(med["abs_err_xgb"]),
        },
        "distributions_catastrophic_xgb": {
            "count": int(cat_mask.sum()),
            "depth": _dist(med.loc[cat_mask, "purchase_count_so_far"]),
            "norm_mad": _dist(med.loc[cat_mask, "norm_mad"]),
            "drift": _dist(med.loc[cat_mask, "drift"]),
            "max_over_median": _dist(med.loc[cat_mask, "max_over_median"]),
            "pred_over_hist": _dist(med.loc[cat_mask, "pred_over_hist"]),
            "pct_extreme": _dist(med.loc[cat_mask, "pct_extreme"]),
            "xgb_pred": _dist(med.loc[cat_mask, "xgb_pred"]),
            "actual": _dist(med.loc[cat_mask, "actual"]),
        },
        "error_pattern_flags": {
            "pred_gt_60": {
                "count": int((med["xgb_pred"] > 60).sum()),
                "pct": round(100.0 * float((med["xgb_pred"] > 60).mean()), 2),
                "mean_abs_err": round(float(med.loc[med["xgb_pred"] > 60, "abs_err_xgb"].mean()), 2)
                if (med["xgb_pred"] > 60).any()
                else None,
            },
            "pred_gt_2x_hist": {
                "count": int((med["pred_over_hist"] > 2.0).sum()),
                "pct": round(100.0 * float((med["pred_over_hist"] > 2.0).mean()), 2),
            },
            "shallow_depth_lt_10": {
                "count": int((med["purchase_count_so_far"] < 10).sum()),
                "pct": round(100.0 * float((med["purchase_count_so_far"] < 10).mean()), 2),
                "mae_xgb": round(float(med.loc[med["purchase_count_so_far"] < 10, "abs_err_xgb"].mean()), 2),
            },
            "drift_gt_10": {
                "count": int((med["drift"] > 10).sum()),
                "pct": round(100.0 * float((med["drift"] > 10).mean()), 2),
                "mae_xgb": round(float(med.loc[med["drift"] > 10, "abs_err_xgb"].mean()), 2),
            },
            "max_over_median_gt_2_5": {
                "count": int((med["max_over_median"] > 2.5).sum()),
                "pct": round(100.0 * float((med["max_over_median"] > 2.5).mean()), 2),
                "mae_xgb": round(float(med.loc[med["max_over_median"] > 2.5, "abs_err_xgb"].mean()), 2),
            },
            "late_bias_gt_14": {
                "count": int((med["bias_xgb"] > 14).sum()),
                "pct": round(100.0 * float((med["bias_xgb"] > 14).mean()), 2),
            },
        },
    }

    # Simple SAFE rules (intentionally coarse — not grid-searched)
    def rule_depth_ge_10(m: pd.DataFrame) -> np.ndarray:
        return m["purchase_count_so_far"].to_numpy() >= 10

    def rule_normmad_le_0_40(m: pd.DataFrame) -> np.ndarray:
        return m["norm_mad"].to_numpy() <= 0.40

    def rule_drift_le_10(m: pd.DataFrame) -> np.ndarray:
        return m["drift"].to_numpy() <= 10.0

    def rule_drift_le_12(m: pd.DataFrame) -> np.ndarray:
        return m["drift"].to_numpy() <= 12.0

    def rule_max_ratio_le_2_5(m: pd.DataFrame) -> np.ndarray:
        return m["max_over_median"].to_numpy() <= 2.5

    def rule_pred_le_1_5x(m: pd.DataFrame) -> np.ndarray:
        return m["pred_over_hist"].to_numpy() <= 1.5

    def rule_within50_ge_60(m: pd.DataFrame) -> np.ndarray:
        return m["pct_within_50"].to_numpy() >= 60.0

    def rule_balanced(m: pd.DataFrame) -> np.ndarray:
        # Keep variable Drift<=15 band partially; require depth + concentration + no crazy pred
        return (
            (m["purchase_count_so_far"] >= 8)
            & (m["norm_mad"] <= 0.50)
            & (m["max_over_median"] <= 2.5)
            & (m["pct_within_50"] >= 55.0)
            & (m["pct_extreme"] < 25.0)
            & (m["pred_over_hist"] <= 1.75)
            & (m["xgb_pred"] <= 50.0)
        ).to_numpy()

    def rule_conservative(m: pd.DataFrame) -> np.ndarray:
        return (
            (m["purchase_count_so_far"] >= 10)
            & (m["norm_mad"] <= 0.40)
            & (m["drift"] <= 12.0)
            & (m["max_over_median"] <= 2.2)
            & (m["pct_within_50"] >= 60.0)
            & (m["pred_over_hist"] <= 1.5)
            & (m["xgb_pred"] <= 45.0)
        ).to_numpy()

    def rule_variable_friendly(m: pd.DataFrame) -> np.ndarray:
        # Designed so canonical 30/60/90-like can stay SAFE if XGB not wild
        return (
            (m["purchase_count_so_far"] >= 8)
            & (m["norm_mad"] <= 0.45)
            & (m["max_over_median"] <= 2.2)
            & (m["pct_in_band"] >= 80.0)
            & (m["pct_extreme"] <= 10.0)
            & (m["pct_within_50"] >= 55.0)
            & (m["pred_over_hist"] <= 1.6)
            & (m["xgb_pred"] <= 55.0)
        ).to_numpy()

    candidates: List[Tuple[str, str, Callable[[pd.DataFrame], np.ndarray]]] = [
        ("C1_depth_ge_10", "SAFE if purchase_count >= 10", rule_depth_ge_10),
        ("C2_normmad_le_0_40", "SAFE if NormMAD <= 0.40", rule_normmad_le_0_40),
        ("C3_drift_le_10", "SAFE if Drift <= 10 (excludes many variable recurrers)", rule_drift_le_10),
        ("C4_drift_le_12", "SAFE if Drift <= 12", rule_drift_le_12),
        ("C5_max_ratio_le_2_5", "SAFE if max/median <= 2.5", rule_max_ratio_le_2_5),
        ("C6_pred_le_1_5x_hist", "SAFE if XGB pred <= 1.5 × hist median", rule_pred_le_1_5x),
        ("C7_within50_ge_60", "SAFE if pct within ±50% of median >= 60", rule_within50_ge_60),
        ("C8_balanced", "SAFE if depth>=8, NM<=0.50, max/med<=2.5, within50>=55, extreme<25, pred<=1.75×hist, pred<=50", rule_balanced),
        ("C9_conservative", "SAFE if depth>=10, NM<=0.40, Drift<=12, max/med<=2.2, within50>=60, pred<=1.5×hist, pred<=45", rule_conservative),
        ("C10_variable_friendly", "SAFE if depth>=8, NM<=0.45, max/med<=2.2, band>=80, extreme<=10, within50>=55, pred<=1.6×hist, pred<=55", rule_variable_friendly),
    ]

    def eval_candidate(cid: str, label: str, safe_mask: np.ndarray) -> Dict[str, Any]:
        safe = med.loc[safe_mask].copy()
        risk = med.loc[~safe_mask].copy()
        # Accepted SAFE under strategy E
        safe_E = safe[safe["passes_E"]]
        risk_E = risk[risk["passes_E"]]  # for residual risk that E would still score
        m_safe_E = _metrics(safe_E["xgb_pred"].to_numpy(), safe_E["actual"].to_numpy())
        m_safe_hist = _metrics(safe["hist_median"].to_numpy(), safe["actual"].to_numpy())
        m_risk_E = _metrics(risk_E["xgb_pred"].to_numpy(), risk_E["actual"].to_numpy())
        m_risk_xgb_all = _metrics(risk["xgb_pred"].to_numpy(), risk["actual"].to_numpy())

        # Compare deltas vs baseline E
        def delta(a, b, key):
            if a.get(key) is None or b.get(key) is None:
                return None
            return round(float(a[key] - b[key]), 2)

        return {
            "id": cid,
            "rule": label,
            "medium_safe_count": int(safe_mask.sum()),
            "medium_risk_count": int((~safe_mask).sum()),
            "medium_safe_pct_of_medium": round(100.0 * float(safe_mask.mean()), 2),
            "medium_safe_coverage_of_path_a_pct": round(100.0 * safe_mask.sum() / path_a_n, 2),
            "medium_safe_coverage_of_test_pct": round(100.0 * safe_mask.sum() / test_n, 2),
            "safe_passing_strategy_E": int(safe["passes_E"].sum()),
            "safe_rejected_by_strategy_E": int((~safe["passes_E"]).sum()),
            "MEDIUM_SAFE_with_strategy_E": m_safe_E,
            "MEDIUM_SAFE_with_personal_median": m_safe_hist,
            "MEDIUM_RISK_with_strategy_E_if_kept": m_risk_E,
            "MEDIUM_RISK_raw_xgb": m_risk_xgb_all,
            "vs_baseline_MEDIUM_E": {
                "baseline_E": baseline_E,
                "delta_mae_safe_E_vs_baseline_E": delta(m_safe_E, baseline_E, "mae"),
                "delta_within_7d_safe_E_vs_baseline_E": delta(m_safe_E, baseline_E, "within_7_days_pct"),
                "delta_cat_abs_safe_E_vs_baseline_E": delta(m_safe_E, baseline_E, "catastrophic_abs_error"),
                "coverage_retained_vs_E_accepted_pct": round(
                    100.0 * m_safe_E["count"] / baseline_E["count"], 2
                )
                if baseline_E["count"]
                else None,
            },
            "vs_HIGH": {
                "HIGH": high_metrics,
                "delta_mae_safe_E_vs_HIGH": delta(m_safe_E, high_metrics, "mae"),
                "delta_within_7d_safe_E_vs_HIGH": delta(m_safe_E, high_metrics, "within_7_days_pct"),
            },
        }

    # Canonical variable features for rule checks
    canon_diag = compute_path_a_interval_diagnostics(VARIABLE_CANONICAL)
    # Synthetic XGB unknown — probe with hist median as "calm" pred and 2x as "wild"
    canon_row_calm = pd.DataFrame(
        [
            {
                "purchase_count_so_far": len(VARIABLE_CANONICAL) + 1,
                "norm_mad": canon_diag["norm_mad"],
                "drift": canon_diag["cadence_drift"],
                "max_over_median": canon_diag["max_over_median"],
                "pct_within_50": canon_diag["pct_within_50pct_of_median"],
                "pct_extreme": canon_diag["pct_extreme_lt10_or_gt180"],
                "pct_in_band": canon_diag["pct_in_refill_band_15_120"],
                "pred_over_hist": 1.1,
                "xgb_pred": 1.1 * canon_diag["hist_median"],
                "hist_median": canon_diag["hist_median"],
            }
        ]
    )
    canon_row_wild = canon_row_calm.copy()
    canon_row_wild["pred_over_hist"] = 2.2
    canon_row_wild["xgb_pred"] = 2.2 * canon_diag["hist_median"]

    candidate_results = []
    for cid, label, fn in candidates:
        safe_mask = fn(med)
        block = eval_candidate(cid, label, safe_mask)
        block["canonical_variable_in_SAFE_if_calm_xgb"] = bool(fn(canon_row_calm)[0])
        block["canonical_variable_in_SAFE_if_wild_xgb"] = bool(fn(canon_row_wild)[0])
        # Examples: correctly accepted (SAFE, abs_err_E <=7) and incorrectly (SAFE, abs_err_E >30)
        safe = med.loc[safe_mask & med["passes_E"]].copy()
        good = safe[safe["abs_err_xgb"] <= 7].sort_values("abs_err_xgb").head(4)
        bad = safe[safe["abs_err_xgb"] > CATASTROPHIC].sort_values("abs_err_xgb", ascending=False).head(4)
        block["examples_correctly_accepted"] = [
            {
                "customerId": str(r.customerId),
                "itemId": str(r.itemId),
                "intervals_tail": r.intervals_tail,
                "hist_median": round(float(r.hist_median), 2),
                "xgb_pred": round(float(r.xgb_pred), 2),
                "actual": float(r.actual),
                "abs_err": round(float(r.abs_err_xgb), 2),
                "norm_mad": round(float(r.norm_mad), 3),
                "drift": round(float(r.drift), 2),
                "max_over_median": round(float(r.max_over_median), 2),
            }
            for r in good.itertuples()
        ]
        block["examples_incorrectly_accepted_catastrophic"] = [
            {
                "customerId": str(r.customerId),
                "itemId": str(r.itemId),
                "intervals_tail": r.intervals_tail,
                "hist_median": round(float(r.hist_median), 2),
                "xgb_pred": round(float(r.xgb_pred), 2),
                "actual": float(r.actual),
                "abs_err": round(float(r.abs_err_xgb), 2),
                "norm_mad": round(float(r.norm_mad), 3),
                "drift": round(float(r.drift), 2),
                "max_over_median": round(float(r.max_over_median), 2),
                "pred_over_hist": round(float(r.pred_over_hist), 2),
            }
            for r in bad.itertuples()
        ]
        candidate_results.append(block)

    # Rank candidates: prefer meaningful coverage (>=25% of MEDIUM SAFE) and MAE improvement, not max reject
    def score(c: Dict[str, Any]):
        m = c["MEDIUM_SAFE_with_strategy_E"]
        if not m["count"] or m["mae"] is None:
            return (-1, 0, 0)
        cov = c["medium_safe_pct_of_medium"]
        # meaningful coverage floor
        cov_ok = 1 if cov >= 25 else 0
        return (
            cov_ok,
            -(m["mae"] or 99),
            m["within_7_days_pct"] or 0,
            cov,
            -(m["catastrophic_pct"] or 100),
        )

    ranked = sorted(candidate_results, key=score, reverse=True)

    # Error driver slices on MEDIUM E-accepted
    slices = {}
    for name, mask in [
        ("depth_6_9", med_E["purchase_count_so_far"] < 10),
        ("depth_ge_10", med_E["purchase_count_so_far"] >= 10),
        ("normmad_le_0_40", med_E["norm_mad"] <= 0.40),
        ("normmad_gt_0_40", med_E["norm_mad"] > 0.40),
        ("drift_le_10", med_E["drift"] <= 10),
        ("drift_10_15", (med_E["drift"] > 10) & (med_E["drift"] <= 15)),
        ("max_ratio_le_2_5", med_E["max_over_median"] <= 2.5),
        ("max_ratio_gt_2_5", med_E["max_over_median"] > 2.5),
        ("pred_le_1_5x", med_E["pred_over_hist"] <= 1.5),
        ("pred_1_5_to_2x", (med_E["pred_over_hist"] > 1.5) & (med_E["pred_over_hist"] <= 2.0)),
    ]:
        sub = med_E.loc[mask]
        slices[name] = _metrics(sub["xgb_pred"].to_numpy(), sub["actual"].to_numpy())

    model_mtime_after = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    report_mtime_after = REPORT_PATH.stat().st_mtime if REPORT_PATH.exists() else None

    # Evidence-based conclusions (filled after viewing ranked — also encoded deterministically)
    best = ranked[0]
    # Find best with coverage >= 30% of MEDIUM and MAE clearly better than baseline E
    meaningful = [
        c
        for c in ranked
        if c["medium_safe_pct_of_medium"] >= 30
        and c["MEDIUM_SAFE_with_strategy_E"]["count"] >= 200
        and (c["MEDIUM_SAFE_with_strategy_E"]["mae"] or 99) <= (baseline_E["mae"] or 0) - 0.5
    ]

    results = {
        "experiment": "PHASE_17H_MEDIUM_SUBCLASSIFICATION_ANALYSIS",
        "read_only": True,
        "strategy_promoted": False,
        "controls": {
            "test_n": test_n,
            "path_a_n": path_a_n,
            "medium_n": med_n,
            "high_n": int(len(high)),
            "classifier": "path_a_classifier_v17f (offline)",
            "feature_flag_enabled": False,
            "medium_baseline_strategy": "E (reject pred>60 OR pred>2×hist)",
            "catastrophic_definition": f"abs_error > {CATASTROPHIC:.0f} days",
        },
        "reference_HIGH": high_metrics,
        "reference_MEDIUM_strategy_E": baseline_E,
        "medium_profile": profile,
        "error_slices_on_MEDIUM_E_accepted": slices,
        "canonical_variable_pattern": {
            "intervals": VARIABLE_CANONICAL,
            "diagnostics": {
                k: (round(v, 4) if isinstance(v, float) and not math.isnan(v) else v)
                for k, v in {**canon_diag, "hist_mean": float(np.mean(VARIABLE_CANONICAL))}.items()
            },
            "path_a_class": classify_path_a_from_intervals(VARIABLE_CANONICAL)["path_a_class"],
        },
        "candidates": candidate_results,
        "candidates_ranked": [c["id"] for c in ranked],
        "meaningful_candidates": [c["id"] for c in meaningful],
        "artifact_safety": {
            "refill_model_joblib_mtime_unchanged": model_mtime_before == model_mtime_after,
            "phase4_report_mtime_unchanged": report_mtime_before == report_mtime_after,
            "model_mtime": model_mtime_before,
            "whatsapp_messages_sent": 0,
            "production_routing_modified": False,
            "feature_flag_remains_off": True,
            "path_b_implemented": False,
            "retrain": False,
        },
    }

    out = PROC / "phase17h_medium_subclassification_results.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"MEDIUM n={med_n} E-accepted={baseline_E['count']} MAE_E={baseline_E['mae']}", flush=True)
    print("Top candidates:", flush=True)
    for c in ranked[:5]:
        m = c["MEDIUM_SAFE_with_strategy_E"]
        print(
            f"  {c['id']}: SAFE={c['medium_safe_count']} ({c['medium_safe_pct_of_medium']}%) "
            f"E_n={m['count']} MAE={m['mae']} ±7d={m['within_7_days_pct']} "
            f"cat%={m['catastrophic_pct']} canon_calm={c['canonical_variable_in_SAFE_if_calm_xgb']}",
            flush=True,
        )
    print(f"Wrote {out}", flush=True)
    print("Artifact safety:", json.dumps(results["artifact_safety"]), flush=True)


if __name__ == "__main__":
    main()
