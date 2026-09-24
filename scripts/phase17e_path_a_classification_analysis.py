"""READ-ONLY Phase 17E Path A classification analysis.

Established customers only (purchase_count >= 6). Evaluates whether current
HIGH / MEDIUM / UNSTABLE (REJECTED) gates separate stable, recurring-variable,
and unpredictable cadence behaviors.

Does NOT modify production code, model, reminder logic, or WhatsApp.
"""
from __future__ import annotations

import gc
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from refillcare.models.hybrid_strategy import (
    MIN_PURCHASES,
    CADENCE_MEDIAN_MIN,
    CADENCE_MEDIAN_MAX,
    CORE_NORM_MAD_MAX,
    CORE_DRIFT_MAX,
    BROAD_NORM_MAD_MAX,
    BROAD_DRIFT_MAX,
    compute_regularity_from_intervals,
    evaluate_hybrid_eligibility,
)
from refillcare.models.training import TARGET_COL

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


# ---------------------------------------------------------------------------
# Business-behavior oracle (independent of current NormMAD/Drift thresholds)
# ---------------------------------------------------------------------------
def interval_behavior_features(intervals: Sequence[float]) -> Dict[str, float]:
    vals = [float(x) for x in intervals if x is not None and float(x) > 0]
    if not vals:
        return {
            "n_intervals": 0,
            "hist_median": float("nan"),
            "hist_mean": float("nan"),
            "hist_std": float("nan"),
            "cv": float("nan"),
            "norm_mad": float("nan"),
            "drift": float("nan"),
            "mad": float("nan"),
            "recent3_median": float("nan"),
            "max_interval": float("nan"),
            "min_interval": float("nan"),
            "max_over_median": float("nan"),
            "pct_within_30pct_median": float("nan"),
            "pct_within_50pct_median": float("nan"),
            "pct_in_refill_band_15_120": float("nan"),
            "pct_extreme_lt10_or_gt180": float("nan"),
            "pct_gt_2x_median": float("nan"),
            "pct_lt_half_median": float("nan"),
            "iqr": float("nan"),
            "iqr_over_median": float("nan"),
        }
    arr = np.asarray(vals, dtype=np.float64)
    med = float(np.median(arr))
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) >= 2 else 0.0
    mad = float(np.median(np.abs(arr - med)))
    norm_mad = float(mad / med) if med > 0 else float("nan")
    r3 = float(np.median(arr[-3:]))
    drift = abs(r3 - med)
    q25, q75 = np.percentile(arr, [25, 75])
    iqr = float(q75 - q25)
    return {
        "n_intervals": float(len(arr)),
        "hist_median": med,
        "hist_mean": mean,
        "hist_std": std,
        "cv": float(std / mean) if mean > 0 else float("nan"),
        "norm_mad": norm_mad,
        "drift": drift,
        "mad": mad,
        "recent3_median": r3,
        "max_interval": float(arr.max()),
        "min_interval": float(arr.min()),
        "max_over_median": float(arr.max() / med) if med > 0 else float("nan"),
        "pct_within_30pct_median": float(np.mean(np.abs(arr - med) <= 0.30 * med) * 100),
        "pct_within_50pct_median": float(np.mean(np.abs(arr - med) <= 0.50 * med) * 100),
        "pct_in_refill_band_15_120": float(np.mean((arr >= 15) & (arr <= 120)) * 100),
        "pct_extreme_lt10_or_gt180": float(np.mean((arr < 10) | (arr > 180)) * 100),
        "pct_gt_2x_median": float(np.mean(arr > 2.0 * med) * 100) if med > 0 else float("nan"),
        "pct_lt_half_median": float(np.mean(arr < 0.5 * med) * 100) if med > 0 else float("nan"),
        "iqr": iqr,
        "iqr_over_median": float(iqr / med) if med > 0 else float("nan"),
    }


def business_oracle_label(feats: Dict[str, float]) -> str:
    """Heuristic business behavior label independent of production thresholds.

    STABLE: tight recurring cadence around a clear center.
    RECURRING_VARIABLE: refill-like median with real variability but still patterned.
    UNSTABLE: no reliable recurring cadence (chaotic / extreme / out-of-band).
    """
    med = feats["hist_median"]
    if math.isnan(med) or feats["n_intervals"] < 5:
        return "UNSTABLE"

    # Unstable: median outside chronic refill band OR many extremes OR chaotic spread
    extreme = feats["pct_extreme_lt10_or_gt180"]
    band = feats["pct_in_refill_band_15_120"]
    max_ratio = feats["max_over_median"]
    within50 = feats["pct_within_50pct_median"]
    cv = feats["cv"]

    if med < 15 or med > 120:
        return "UNSTABLE"
    if extreme >= 40 or band < 50:
        return "UNSTABLE"
    if max_ratio >= 4.0 and within50 < 60:
        return "UNSTABLE"
    if (not math.isnan(cv)) and cv >= 1.2 and within50 < 55:
        return "UNSTABLE"

    # Stable: most intervals near median, no giant outliers
    within30 = feats["pct_within_30pct_median"]
    if within30 >= 70 and max_ratio <= 2.0 and (math.isnan(cv) or cv <= 0.45):
        return "STABLE"
    if within50 >= 80 and max_ratio <= 2.5 and feats["norm_mad"] <= 0.40:
        return "STABLE"

    # Otherwise recurring but variable (still in-band median, enough band support)
    if band >= 60 and extreme < 35:
        return "RECURRING_VARIABLE"

    return "UNSTABLE"


def current_path_a_label(record: Dict[str, Any]) -> str:
    """Map hybrid eligibility to Path A business names (UNSTABLE = REJECTED)."""
    d = evaluate_hybrid_eligibility(record)
    conf = d["confidence"]
    if conf == "HIGH":
        return "HIGH"
    if conf == "MEDIUM":
        return "MEDIUM"
    return "UNSTABLE"


def classify_with_thresholds(
    record: Dict[str, Any],
    *,
    core_nm: float,
    core_dr: float,
    broad_nm: float,
    broad_dr: float,
    cadence_min: float = CADENCE_MEDIAN_MIN,
    cadence_max: float = CADENCE_MEDIAN_MAX,
    min_purchases: int = MIN_PURCHASES,
) -> str:
    p = int(record.get("purchase_count_so_far", 1) or 1)
    if p < min_purchases:
        return "UNSTABLE"
    hist = float(record.get("historical_interval_median", float("nan")))
    nm = float(record.get("norm_mad", float("nan")))
    dr = float(record.get("cadence_drift", float("nan")))
    if math.isnan(hist) or hist < cadence_min or hist > cadence_max:
        return "UNSTABLE"
    if math.isnan(nm) or math.isnan(dr):
        return "UNSTABLE"
    if nm > broad_nm or dr > broad_dr:
        return "UNSTABLE"
    if nm <= core_nm and dr <= core_dr:
        return "HIGH"
    return "MEDIUM"


def _pred_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & (y_true > 0)
    if mask.sum() == 0:
        return {"count": 0, "mae": None, "medae": None, "within_3_days_pct": None, "within_7_days_pct": None}
    yt = y_true[mask]
    yp = y_pred[mask]
    ae = np.abs(yp - yt)
    return {
        "count": int(mask.sum()),
        "mae": round(float(ae.mean()), 2),
        "medae": round(float(np.median(ae)), 2),
        "within_3_days_pct": round(100.0 * float((ae <= 3).mean()), 2),
        "within_7_days_pct": round(100.0 * float((ae <= 7).mean()), 2),
        "mean_bias": round(float((yp - yt).mean()), 2),
    }


def _summarize_slice(df: pd.DataFrame, label_col: str = "path_a_label") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for lab, g in df.groupby(label_col):
        actual = g[TARGET_COL].to_numpy(dtype=np.float64) if TARGET_COL in g.columns else np.array([])
        hist = g["historical_interval_median"].to_numpy(dtype=np.float64)
        m = _pred_metrics(actual, hist) if len(actual) else {"count": int(len(g))}
        out[str(lab)] = {
            "count": int(len(g)),
            "pct": round(100.0 * len(g) / len(df), 2),
            "mean_depth": round(float(g["purchase_count_so_far"].mean()), 2),
            "mean_hist_median": round(float(g["historical_interval_median"].mean()), 2),
            "median_hist_median": round(float(g["historical_interval_median"].median()), 2),
            "mean_norm_mad": round(float(g["norm_mad"].mean()), 2),
            "median_norm_mad": round(float(g["norm_mad"].median()), 2),
            "mean_drift": round(float(g["cadence_drift"].mean()), 2),
            "median_drift": round(float(g["cadence_drift"].median()), 2),
            "mean_cv": round(float(g["cv"].mean()), 2) if "cv" in g else None,
            "mean_max_over_median": round(float(g["max_over_median"].mean()), 2) if "max_over_median" in g else None,
            "mean_pct_within_50pct": round(float(g["pct_within_50pct_median"].mean()), 2)
            if "pct_within_50pct_median" in g
            else None,
            "mean_pct_extreme": round(float(g["pct_extreme_lt10_or_gt180"].mean()), 2)
            if "pct_extreme_lt10_or_gt180" in g
            else None,
            "personal_median_metrics": m,
        }
    return out


def _example_row(row: pd.Series, intervals: Optional[List[float]] = None) -> Dict[str, Any]:
    return {
        "customerId": str(row.get("customerId")),
        "itemId": str(row.get("itemId")),
        "purchase_count_so_far": int(row.get("purchase_count_so_far", 0)),
        "intervals": intervals if intervals is not None else None,
        "hist_median": round(float(row["historical_interval_median"]), 2),
        "hist_mean": round(float(row["hist_mean"]), 2) if "hist_mean" in row and pd.notna(row.get("hist_mean")) else None,
        "norm_mad": round(float(row["norm_mad"]), 3),
        "drift": round(float(row["cadence_drift"]), 2),
        "cv": round(float(row["cv"]), 3) if "cv" in row and pd.notna(row.get("cv")) else None,
        "max_over_median": round(float(row["max_over_median"]), 2)
        if "max_over_median" in row and pd.notna(row.get("max_over_median"))
        else None,
        "pct_within_50pct_median": round(float(row["pct_within_50pct_median"]), 1)
        if "pct_within_50pct_median" in row
        else None,
        "pct_extreme": round(float(row["pct_extreme_lt10_or_gt180"]), 1)
        if "pct_extreme_lt10_or_gt180" in row
        else None,
        "path_a_label": row.get("path_a_label"),
        "oracle_label": row.get("oracle_label"),
        "actual_target": float(row[TARGET_COL]) if TARGET_COL in row and pd.notna(row.get(TARGET_COL)) else None,
        "abs_error_personal_median": (
            round(abs(float(row["historical_interval_median"]) - float(row[TARGET_COL])), 2)
            if TARGET_COL in row and pd.notna(row.get(TARGET_COL))
            else None
        ),
    }


def main() -> None:
    model_mtime_before = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    report_mtime_before = REPORT_PATH.stat().st_mtime if REPORT_PATH.exists() else None

    # ---- Canonical example patterns ----
    examples = {
        "stable_canonical": [30, 30, 31, 29, 30, 32],
        "recurring_variable_canonical": [30, 60, 45, 30, 90, 30, 60, 30, 90],
        "unstable_canonical": [5, 180, 12, 240, 3, 150],
    }
    example_scores = {}
    for name, ivs in examples.items():
        feats = interval_behavior_features(ivs)
        reg = compute_regularity_from_intervals(ivs)
        # Simulate Path A record with P>=6
        record = {
            "purchase_count_so_far": len(ivs) + 1,
            "historical_interval_median": feats["hist_median"],
            "norm_mad": feats["norm_mad"],
            "cadence_drift": feats["drift"],
            "historical_interval_norm_mad": feats["norm_mad"],
        }
        example_scores[name] = {
            "intervals": ivs,
            "features": {k: (round(v, 4) if isinstance(v, float) and not math.isnan(v) else v) for k, v in feats.items()},
            "regularity": {k: (round(v, 4) if isinstance(v, float) and not math.isnan(v) else v) for k, v in reg.items()},
            "oracle_label": business_oracle_label(feats),
            "current_path_a_label": current_path_a_label(record),
        }

    print("Loading test + history...", flush=True)
    test = pd.read_parquet(PROC / "test.parquet")
    test = test[test[TARGET_COL] > 0].copy()
    test_n = int(len(test))
    history = pd.read_parquet(PROC / "purchase_history.parquet")

    # Attach leakage-safe regularity for test
    test = _attach_regularity_from_history(test, history)
    test["historical_interval_norm_mad"] = test["norm_mad"]
    test["recent3_interval_median"] = test["recent3_median"]

    # Build prior interval lists for Path A rows (P>=6) for richer features / examples
    hist = history[
        ["customerId", "itemId", "invoice_date", "days_since_previous_purchase", "purchase_seq"]
    ].copy()
    hist["invoice_date"] = pd.to_datetime(hist["invoice_date"])
    hist["customerId"] = hist["customerId"].astype(str)
    hist["itemId"] = hist["itemId"].astype(str)
    hist = hist.sort_values(["customerId", "itemId", "invoice_date", "purchase_seq"])

    path_a_mask = test["purchase_count_so_far"].astype(int) >= MIN_PURCHASES
    path_a = test.loc[path_a_mask].copy().reset_index(drop=True)
    path_a["customerId"] = path_a["customerId"].astype(str)
    path_a["itemId"] = path_a["itemId"].astype(str)
    path_a["purchase_seq"] = path_a["purchase_count_so_far"].astype(int)

    keys = set(zip(path_a["customerId"], path_a["itemId"]))
    hist["_key"] = list(zip(hist["customerId"], hist["itemId"]))
    hist_sub = hist[hist["_key"].isin(keys)].copy()

    # Map (cid, iid, seq) -> prior intervals (strictly before this purchase)
    prior_map: Dict[Tuple[str, str, int], List[float]] = {}
    for (cid, iid), g in hist_sub.groupby(["customerId", "itemId"], sort=False):
        intervals: List[float] = []
        for _, row in g.iterrows():
            seq = int(row["purchase_seq"])
            prior_map[(cid, iid, seq)] = list(intervals)
            d = row["days_since_previous_purchase"]
            if pd.notna(d) and float(d) > 0:
                intervals.append(float(d))

    feat_rows = []
    for _, row in path_a.iterrows():
        key = (str(row["customerId"]), str(row["itemId"]), int(row["purchase_seq"]))
        ivs = prior_map.get(key, [])
        feats = interval_behavior_features(ivs)
        feat_rows.append(feats)
    feat_df = pd.DataFrame(feat_rows)
    for c in feat_df.columns:
        path_a[c] = feat_df[c].values

    # Prefer attached norm_mad/drift; fill from recomputed if needed
    path_a["norm_mad"] = path_a["norm_mad"].fillna(path_a["norm_mad"])
    # hist_median column already on path_a from features; ensure historical_interval_median present
    if "historical_interval_median" not in path_a.columns or path_a["historical_interval_median"].isna().all():
        path_a["historical_interval_median"] = path_a["hist_median"]
    else:
        # Prefer feature-engineered hist median when present
        path_a["historical_interval_median"] = path_a["historical_interval_median"].fillna(path_a["hist_median"])

    # Labels
    path_a["path_a_label"] = [
        current_path_a_label(
            {
                "purchase_count_so_far": int(r.purchase_count_so_far),
                "historical_interval_median": float(r.historical_interval_median),
                "norm_mad": float(r.norm_mad) if pd.notna(r.norm_mad) else float("nan"),
                "cadence_drift": float(r.cadence_drift) if pd.notna(r.cadence_drift) else float("nan"),
                "historical_interval_norm_mad": float(r.norm_mad) if pd.notna(r.norm_mad) else float("nan"),
            }
        )
        for r in path_a.itertuples(index=False)
    ]
    path_a["oracle_label"] = [
        business_oracle_label(
            {
                "n_intervals": float(r.n_intervals),
                "hist_median": float(r.hist_median),
                "cv": float(r.cv) if pd.notna(r.cv) else float("nan"),
                "norm_mad": float(r.norm_mad) if pd.notna(r.norm_mad) else float("nan"),
                "max_over_median": float(r.max_over_median) if pd.notna(r.max_over_median) else float("nan"),
                "pct_within_30pct_median": float(r.pct_within_30pct_median),
                "pct_within_50pct_median": float(r.pct_within_50pct_median),
                "pct_in_refill_band_15_120": float(r.pct_in_refill_band_15_120),
                "pct_extreme_lt10_or_gt180": float(r.pct_extreme_lt10_or_gt180),
            }
        )
        for r in path_a.itertuples(index=False)
    ]

    # Interval distribution summaries by current label
    def _dist(series: pd.Series) -> Dict[str, Any]:
        s = series.dropna().astype(float)
        if len(s) == 0:
            return {}
        qs = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()
        return {
            "mean": round(float(s.mean()), 2),
            "std": round(float(s.std()), 2),
            "p05": round(float(qs[0.05]), 2),
            "p25": round(float(qs[0.25]), 2),
            "p50": round(float(qs[0.5]), 2),
            "p75": round(float(qs[0.75]), 2),
            "p95": round(float(qs[0.95]), 2),
        }

    interval_dists = {}
    for lab, g in path_a.groupby("path_a_label"):
        interval_dists[str(lab)] = {
            "hist_median": _dist(g["historical_interval_median"]),
            "norm_mad": _dist(g["norm_mad"]),
            "drift": _dist(g["cadence_drift"]),
            "cv": _dist(g["cv"]),
            "max_over_median": _dist(g["max_over_median"]),
            "actual_target": _dist(g[TARGET_COL]),
        }

    # Confusion: current vs oracle
    confusion = (
        path_a.groupby(["path_a_label", "oracle_label"])
        .size()
        .reset_index(name="count")
        .to_dict(orient="records")
    )
    confusion_matrix = {}
    for cur in ["HIGH", "MEDIUM", "UNSTABLE"]:
        confusion_matrix[cur] = {}
        for ora in ["STABLE", "RECURRING_VARIABLE", "UNSTABLE"]:
            n = int(((path_a["path_a_label"] == cur) & (path_a["oracle_label"] == ora)).sum())
            confusion_matrix[cur][ora] = n

    # False classification cases (sampled extremes)
    false_cases = {
        "HIGH_but_oracle_UNSTABLE": [],
        "HIGH_but_oracle_RECURRING_VARIABLE": [],
        "MEDIUM_but_oracle_STABLE": [],
        "MEDIUM_but_oracle_UNSTABLE": [],
        "UNSTABLE_but_oracle_STABLE": [],
        "UNSTABLE_but_oracle_RECURRING_VARIABLE": [],
    }
    for key, (cur, ora) in {
        "HIGH_but_oracle_UNSTABLE": ("HIGH", "UNSTABLE"),
        "HIGH_but_oracle_RECURRING_VARIABLE": ("HIGH", "RECURRING_VARIABLE"),
        "MEDIUM_but_oracle_STABLE": ("MEDIUM", "STABLE"),
        "MEDIUM_but_oracle_UNSTABLE": ("MEDIUM", "UNSTABLE"),
        "UNSTABLE_but_oracle_STABLE": ("UNSTABLE", "STABLE"),
        "UNSTABLE_but_oracle_RECURRING_VARIABLE": ("UNSTABLE", "RECURRING_VARIABLE"),
    }.items():
        sub = path_a[(path_a["path_a_label"] == cur) & (path_a["oracle_label"] == ora)].copy()
        if len(sub) == 0:
            continue
        # pick informative extremes
        if cur == "HIGH" and ora != "STABLE":
            sub = sub.sort_values(["max_over_median", "norm_mad"], ascending=False)
        elif cur == "MEDIUM" and ora == "UNSTABLE":
            sub = sub.sort_values(["pct_extreme_lt10_or_gt180", "max_over_median"], ascending=False)
        elif cur == "MEDIUM" and ora == "STABLE":
            sub = sub.sort_values(["norm_mad", "cadence_drift"], ascending=True)
        elif cur == "UNSTABLE" and ora == "STABLE":
            sub = sub.sort_values(["pct_within_50pct_median"], ascending=False)
        elif cur == "UNSTABLE" and ora == "RECURRING_VARIABLE":
            sub = sub.sort_values(["pct_in_refill_band_15_120"], ascending=False)
        for _, r in sub.head(5).iterrows():
            key_t = (str(r["customerId"]), str(r["itemId"]), int(r["purchase_seq"]))
            false_cases[key].append(_example_row(r, prior_map.get(key_t, [])[-12:]))

    # Real data pattern examples close to canonical types
    pattern_examples = {"stable_like": [], "recurring_variable_like": [], "unstable_like": []}
    stable_pool = path_a[path_a["oracle_label"] == "STABLE"].sort_values("norm_mad")
    var_pool = path_a[path_a["oracle_label"] == "RECURRING_VARIABLE"].sort_values(
        "pct_within_50pct_median", ascending=True
    )
    uns_pool = path_a[path_a["oracle_label"] == "UNSTABLE"].sort_values(
        "pct_extreme_lt10_or_gt180", ascending=False
    )
    for pool, key in [
        (stable_pool, "stable_like"),
        (var_pool, "recurring_variable_like"),
        (uns_pool, "unstable_like"),
    ]:
        for _, r in pool.head(5).iterrows():
            key_t = (str(r["customerId"]), str(r["itemId"]), int(r["purchase_seq"]))
            pattern_examples[key].append(_example_row(r, prior_map.get(key_t, [])[-12:]))

    # Threshold sensitivity (Path A only, P>=6 fixed)
    core_nm_grid = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    core_dr_grid = [5.0, 7.0, 10.0]
    broad_nm_grid = [0.40, 0.50, 0.60, 0.70]
    broad_dr_grid = [7.0, 10.0, 15.0, 20.0]

    sensitivity = []
    records_for_sweep = []
    for r in path_a.itertuples(index=False):
        records_for_sweep.append(
            {
                "purchase_count_so_far": int(r.purchase_count_so_far),
                "historical_interval_median": float(r.historical_interval_median),
                "norm_mad": float(r.norm_mad) if pd.notna(r.norm_mad) else float("nan"),
                "cadence_drift": float(r.cadence_drift) if pd.notna(r.cadence_drift) else float("nan"),
                "oracle": r.oracle_label,
                "actual": float(getattr(r, TARGET_COL)),
                "hist_med": float(r.historical_interval_median),
            }
        )

    def eval_thresholds(core_nm, core_dr, broad_nm, broad_dr) -> Dict[str, Any]:
        labels = []
        for rec in records_for_sweep:
            labels.append(
                classify_with_thresholds(
                    rec,
                    core_nm=core_nm,
                    core_dr=core_dr,
                    broad_nm=broad_nm,
                    broad_dr=broad_dr,
                )
            )
        labels_a = np.array(labels)
        oracle_a = np.array([r["oracle"] for r in records_for_sweep])
        actual_a = np.array([r["actual"] for r in records_for_sweep], dtype=np.float64)
        hist_a = np.array([r["hist_med"] for r in records_for_sweep], dtype=np.float64)

        def agr(pred_lab, ora_lab):
            # Map HIGH->STABLE, MEDIUM->RECURRING_VARIABLE, UNSTABLE->UNSTABLE
            mapping = {"HIGH": "STABLE", "MEDIUM": "RECURRING_VARIABLE", "UNSTABLE": "UNSTABLE"}
            mapped = np.array([mapping[x] for x in pred_lab])
            return float((mapped == ora_lab).mean() * 100)

        counts = {
            "HIGH": int((labels_a == "HIGH").sum()),
            "MEDIUM": int((labels_a == "MEDIUM").sum()),
            "UNSTABLE": int((labels_a == "UNSTABLE").sum()),
        }
        # Personal-median accuracy on HIGH and on HIGH+MEDIUM (accepted)
        high_m = labels_a == "HIGH"
        med_m = labels_a == "MEDIUM"
        acc_m = high_m | med_m
        return {
            "core_norm_mad_max": core_nm,
            "core_drift_max": core_dr,
            "broad_norm_mad_max": broad_nm,
            "broad_drift_max": broad_dr,
            "counts": counts,
            "coverage_high_pct": round(100.0 * counts["HIGH"] / len(labels_a), 2),
            "coverage_accepted_pct": round(100.0 * (counts["HIGH"] + counts["MEDIUM"]) / len(labels_a), 2),
            "oracle_agreement_pct": round(agr(labels_a, oracle_a), 2),
            "high_personal_median": _pred_metrics(actual_a[high_m], hist_a[high_m]),
            "medium_personal_median": _pred_metrics(actual_a[med_m], hist_a[med_m]),
            "accepted_personal_median": _pred_metrics(actual_a[acc_m], hist_a[acc_m]),
            "unstable_personal_median": _pred_metrics(actual_a[~acc_m], hist_a[~acc_m]),
            # Misroutes vs oracle
            "false_high_unstable": int(((labels_a == "HIGH") & (oracle_a == "UNSTABLE")).sum()),
            "false_medium_unstable": int(((labels_a == "MEDIUM") & (oracle_a == "UNSTABLE")).sum()),
            "missed_stable_as_unstable": int(((labels_a == "UNSTABLE") & (oracle_a == "STABLE")).sum()),
            "missed_variable_as_unstable": int(
                ((labels_a == "UNSTABLE") & (oracle_a == "RECURRING_VARIABLE")).sum()
            ),
            "stable_forced_to_medium": int(((labels_a == "MEDIUM") & (oracle_a == "STABLE")).sum()),
        }

    # Current + focused grid
    sensitivity.append(eval_thresholds(0.35, 7.0, 0.50, 10.0))
    for cnm in core_nm_grid:
        for cdr in core_dr_grid:
            for bnm in broad_nm_grid:
                for bdr in broad_dr_grid:
                    if cnm > bnm or cdr > bdr:
                        continue
                    if cnm == 0.35 and cdr == 7.0 and bnm == 0.50 and bdr == 10.0:
                        continue
                    sensitivity.append(eval_thresholds(cnm, cdr, bnm, bdr))

    # Rank candidates: maximize oracle agreement, then HIGH ±7d, then accepted coverage
    def score_cand(c):
        h = c["high_personal_median"]
        a = c["accepted_personal_median"]
        return (
            c["oracle_agreement_pct"],
            h["within_7_days_pct"] if h["within_7_days_pct"] is not None else -1,
            a["within_7_days_pct"] if a["within_7_days_pct"] is not None else -1,
            -c["false_high_unstable"],
            -c["false_medium_unstable"],
            c["coverage_accepted_pct"],
        )

    sensitivity_sorted = sorted(sensitivity, key=score_cand, reverse=True)
    top_candidates = sensitivity_sorted[:12]
    current_metrics = eval_thresholds(0.35, 7.0, 0.50, 10.0)

    # Proposed rules from evidence + top candidates + business constraints
    # Prefer: keep HIGH precise, MEDIUM = recurring variable, cut unstable from MEDIUM
    # Also evaluate a richer rule set using max/median and extreme %
    def classify_proposed(rec_feats: Dict[str, Any]) -> str:
        """Proposed Path A decision tree (offline recommendation)."""
        p = int(rec_feats["purchase_count_so_far"])
        if p < 6:
            return "UNSTABLE"
        med = float(rec_feats["historical_interval_median"])
        nm = float(rec_feats["norm_mad"])
        dr = float(rec_feats["cadence_drift"])
        max_ratio = float(rec_feats.get("max_over_median", float("nan")))
        extreme = float(rec_feats.get("pct_extreme_lt10_or_gt180", float("nan")))
        within50 = float(rec_feats.get("pct_within_50pct_median", float("nan")))
        band = float(rec_feats.get("pct_in_refill_band_15_120", float("nan")))

        if math.isnan(med) or med < 15 or med > 120:
            return "UNSTABLE"
        if math.isnan(nm) or math.isnan(dr):
            return "UNSTABLE"

        # Hard instability guards (beyond NormMAD/Drift alone)
        if (not math.isnan(extreme)) and extreme >= 35:
            return "UNSTABLE"
        if (not math.isnan(band)) and band < 55:
            return "UNSTABLE"
        if (not math.isnan(max_ratio)) and max_ratio >= 3.5 and (math.isnan(within50) or within50 < 65):
            return "UNSTABLE"

        # HIGH: tight regularity
        if nm <= 0.30 and dr <= 7.0 and (math.isnan(max_ratio) or max_ratio <= 2.2):
            return "HIGH"
        # Allow slightly looser NormMAD if very low drift and tight max ratio
        if nm <= 0.35 and dr <= 5.0 and (math.isnan(max_ratio) or max_ratio <= 2.0):
            return "HIGH"

        # MEDIUM: eligible recurring-variable band (Drift<=15 recovers patterns like 30/60/90)
        if nm <= 0.55 and dr <= 15.0:
            return "MEDIUM"
        return "UNSTABLE"

    prop_labels = []
    for r in path_a.itertuples(index=False):
        prop_labels.append(
            classify_proposed(
                {
                    "purchase_count_so_far": int(r.purchase_count_so_far),
                    "historical_interval_median": float(r.historical_interval_median),
                    "norm_mad": float(r.norm_mad) if pd.notna(r.norm_mad) else float("nan"),
                    "cadence_drift": float(r.cadence_drift) if pd.notna(r.cadence_drift) else float("nan"),
                    "max_over_median": float(r.max_over_median) if pd.notna(r.max_over_median) else float("nan"),
                    "pct_extreme_lt10_or_gt180": float(r.pct_extreme_lt10_or_gt180),
                    "pct_within_50pct_median": float(r.pct_within_50pct_median),
                    "pct_in_refill_band_15_120": float(r.pct_in_refill_band_15_120),
                }
            )
        )
    path_a["proposed_label"] = prop_labels

    proposed_vs_oracle = {}
    for cur in ["HIGH", "MEDIUM", "UNSTABLE"]:
        proposed_vs_oracle[cur] = {}
        for ora in ["STABLE", "RECURRING_VARIABLE", "UNSTABLE"]:
            proposed_vs_oracle[cur][ora] = int(
                ((path_a["proposed_label"] == cur) & (path_a["oracle_label"] == ora)).sum()
            )

    def metrics_by_label(label_col: str) -> Dict[str, Any]:
        out = {}
        actual = path_a[TARGET_COL].to_numpy(dtype=np.float64)
        histm = path_a["historical_interval_median"].to_numpy(dtype=np.float64)
        for lab in ["HIGH", "MEDIUM", "UNSTABLE"]:
            m = path_a[label_col] == lab
            out[lab] = {
                "count": int(m.sum()),
                "pct_of_path_a": round(100.0 * m.sum() / len(path_a), 2),
                "pct_of_test": round(100.0 * m.sum() / test_n, 2),
                "personal_median": _pred_metrics(actual[m], histm[m]),
            }
        accepted = path_a[label_col].isin(["HIGH", "MEDIUM"])
        out["ACCEPTED"] = {
            "count": int(accepted.sum()),
            "pct_of_path_a": round(100.0 * accepted.sum() / len(path_a), 2),
            "pct_of_test": round(100.0 * accepted.sum() / test_n, 2),
            "personal_median": _pred_metrics(actual[accepted], histm[accepted]),
        }
        return out

    current_by_label = metrics_by_label("path_a_label")
    proposed_by_label = metrics_by_label("proposed_label")
    oracle_by_label = {}
    actual = path_a[TARGET_COL].to_numpy(dtype=np.float64)
    histm = path_a["historical_interval_median"].to_numpy(dtype=np.float64)
    for lab, mapped in [
        ("STABLE", "STABLE"),
        ("RECURRING_VARIABLE", "RECURRING_VARIABLE"),
        ("UNSTABLE", "UNSTABLE"),
    ]:
        m = path_a["oracle_label"] == lab
        oracle_by_label[lab] = {
            "count": int(m.sum()),
            "pct_of_path_a": round(100.0 * m.sum() / len(path_a), 2),
            "personal_median": _pred_metrics(actual[m], histm[m]),
        }

    # Pair-level established histories from full 2020-2026 purchase_history
    print("Building pair-level Path A cohort from full history...", flush=True)
    pair_stats = []
    for (cid, iid), g in hist.groupby(["customerId", "itemId"], sort=False):
        ivs = [
            float(d)
            for d in g["days_since_previous_purchase"].tolist()
            if pd.notna(d) and float(d) > 0
        ]
        n_purch = int(len(g))
        if n_purch < 6 or len(ivs) < 5:
            continue
        feats = interval_behavior_features(ivs)
        record = {
            "purchase_count_so_far": n_purch,
            "historical_interval_median": feats["hist_median"],
            "norm_mad": feats["norm_mad"],
            "cadence_drift": feats["drift"],
        }
        pair_stats.append(
            {
                "customerId": cid,
                "itemId": iid,
                "n_purchases": n_purch,
                "n_intervals": int(feats["n_intervals"]),
                "oracle_label": business_oracle_label(feats),
                "path_a_label": current_path_a_label(record),
                **{k: feats[k] for k in feats},
            }
        )
    pairs = pd.DataFrame(pair_stats)
    pair_cohort = {
        "established_pairs": int(len(pairs)),
        "by_current_label": pairs["path_a_label"].value_counts().to_dict() if len(pairs) else {},
        "by_oracle_label": pairs["oracle_label"].value_counts().to_dict() if len(pairs) else {},
        "mean_purchases": round(float(pairs["n_purchases"].mean()), 2) if len(pairs) else None,
    }

    # Oracle agreement rates
    mapping = {"HIGH": "STABLE", "MEDIUM": "RECURRING_VARIABLE", "UNSTABLE": "UNSTABLE"}
    cur_mapped = path_a["path_a_label"].map(mapping)
    prop_mapped = path_a["proposed_label"].map(mapping)
    agreement = {
        "current_vs_oracle_pct": round(100.0 * float((cur_mapped == path_a["oracle_label"]).mean()), 2),
        "proposed_vs_oracle_pct": round(100.0 * float((prop_mapped == path_a["oracle_label"]).mean()), 2),
    }

    # Drift-only / NormMAD-only failure modes among MEDIUM
    medium = path_a[path_a["path_a_label"] == "MEDIUM"]
    medium_drivers = {
        "count": int(len(medium)),
        "normmad_driven_drift_le_7": int(((medium["norm_mad"] > 0.35) & (medium["cadence_drift"] <= 7)).sum()),
        "drift_driven_normmad_le_0_35": int(((medium["norm_mad"] <= 0.35) & (medium["cadence_drift"] > 7)).sum()),
        "both_elevated": int(((medium["norm_mad"] > 0.35) & (medium["cadence_drift"] > 7)).sum()),
        "oracle_stable_pct": round(
            100.0 * float((medium["oracle_label"] == "STABLE").mean()) if len(medium) else 0.0, 2
        ),
        "oracle_variable_pct": round(
            100.0 * float((medium["oracle_label"] == "RECURRING_VARIABLE").mean()) if len(medium) else 0.0, 2
        ),
        "oracle_unstable_pct": round(
            100.0 * float((medium["oracle_label"] == "UNSTABLE").mean()) if len(medium) else 0.0, 2
        ),
    }

    model_mtime_after = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    report_mtime_after = REPORT_PATH.stat().st_mtime if REPORT_PATH.exists() else None

    results = {
        "experiment": "PHASE_17E_PATH_A_CLASSIFICATION_ANALYSIS",
        "read_only": True,
        "controls": {
            "path": "A_established_customers",
            "min_purchases": MIN_PURCHASES,
            "test_rows_universe": test_n,
            "path_a_rows": int(len(path_a)),
            "test_date_min": str(pd.to_datetime(test["invoice_date"]).min().date()),
            "test_date_max": str(pd.to_datetime(test["invoice_date"]).max().date()),
            "history_date_min": str(pd.to_datetime(history["invoice_date"]).min().date()),
            "history_date_max": str(pd.to_datetime(history["invoice_date"]).max().date()),
            "current_rules": {
                "min_purchases": MIN_PURCHASES,
                "cadence_median_band": [CADENCE_MEDIAN_MIN, CADENCE_MEDIAN_MAX],
                "high": {"norm_mad_max": CORE_NORM_MAD_MAX, "drift_max": CORE_DRIFT_MAX},
                "medium_eligible": {"norm_mad_max": BROAD_NORM_MAD_MAX, "drift_max": BROAD_DRIFT_MAX},
                "unstable": "fails depth, cadence band, or broad regularity",
            },
        },
        "canonical_examples": example_scores,
        "pair_level_cohort_2020_2026": pair_cohort,
        "current_path_a_cohorts": _summarize_slice(path_a, "path_a_label"),
        "oracle_behavior_cohorts": oracle_by_label,
        "interval_distributions_by_current_label": interval_dists,
        "confusion_current_vs_oracle": confusion_matrix,
        "confusion_records": confusion,
        "medium_drivers": medium_drivers,
        "pattern_examples_from_data": pattern_examples,
        "false_classification_examples": false_cases,
        "threshold_sensitivity_top": top_candidates,
        "current_threshold_metrics": current_metrics,
        "agreement": agreement,
        "current_label_accuracy": current_by_label,
        "proposed_label_accuracy": proposed_by_label,
        "proposed_vs_oracle_confusion": proposed_vs_oracle,
        "proposed_decision_tree": {
            "steps": [
                "1. If purchase_count < 6 -> UNSTABLE (not Path A)",
                "2. If hist_median not in [15,120] -> UNSTABLE",
                "3. If NormMAD or Drift missing -> UNSTABLE",
                "4. If pct_extreme(lt10|gt180) >= 35% -> UNSTABLE",
                "5. If pct_intervals_in_[15,120] < 55% -> UNSTABLE",
                "6. If max/median >= 3.5 AND pct_within_50pct_median < 65% -> UNSTABLE",
                "7. HIGH if (NormMAD<=0.30 AND Drift<=7 AND max/median<=2.2) OR (NormMAD<=0.35 AND Drift<=5 AND max/median<=2.0)",
                "8. MEDIUM if NormMAD<=0.55 AND Drift<=15 (and passed instability guards)",
                "9. Else UNSTABLE",
            ],
            "rationale": (
                "Current NormMAD/Drift alone mislabels chaotic series as MEDIUM (MAD-robust to "
                "extreme gaps) and rejects genuine recurring-variable series when recent-3 drift "
                "exceeds 10d (e.g. 30/60/90 patterns). Add extreme-gap and max/median guards; "
                "tighten HIGH; widen MEDIUM drift to 15 only after those guards."
            ),
        },
        "artifact_safety": {
            "refill_model_joblib_mtime_unchanged": model_mtime_before == model_mtime_after,
            "phase4_report_mtime_unchanged": report_mtime_before == report_mtime_after,
            "model_mtime_before": model_mtime_before,
            "model_mtime_after": model_mtime_after,
            "whatsapp_messages_sent": 0,
            "production_code_modified": False,
        },
    }

    out = PROC / "phase17e_path_a_classification_results.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"Path A n={len(path_a)} / test n={test_n}", flush=True)
    print("Current cohorts:", {k: v["count"] for k, v in current_by_label.items() if k != "ACCEPTED"}, flush=True)
    print("Oracle cohorts:", {k: v["count"] for k, v in oracle_by_label.items()}, flush=True)
    print("Agreement:", agreement, flush=True)
    print(f"Wrote {out}", flush=True)
    print("Artifact safety:", json.dumps(results["artifact_safety"]), flush=True)

    del history, hist, hist_sub, test
    gc.collect()


if __name__ == "__main__":
    main()
