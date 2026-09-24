"""READ-ONLY Phase 17I MEDIUM-SAFE predictor comparison (Path A only).

Compares personal median vs production XGB vs 50/50 blend on the C6-like
MEDIUM-SAFE pocket (pred <= 1.5x hist) under Strategy E gates.
Does NOT train models, promote strategies, touch WhatsApp/routing, or Path B.
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
from refillcare.models.hybrid_strategy import MIN_PURCHASES
from refillcare.models.path_a_classifier import (
    LABEL_HIGH,
    LABEL_MEDIUM,
    classify_path_a,
    classify_path_a_from_intervals,
    compute_path_a_interval_diagnostics,
    is_path_a_classifier_enabled,
)

PROC = ROOT / "data" / "refillcare" / "processed"
MODEL_PATH = PROC / "models" / "refill_model.joblib"
REPORT_PATH = PROC / "phase4_model_report.json"
MODELS_DIR = PROC / "models"
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


def _find_quantile_artifact() -> Optional[str]:
    if not MODELS_DIR.exists():
        return None
    candidates = []
    for p in MODELS_DIR.glob("*.joblib"):
        name = p.name.lower()
        if "quantile" in name or "q50" in name or "q0.5" in name:
            candidates.append(str(p))
    return candidates[0] if candidates else None


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


def _score_split(
    split_name: str,
    df_raw: pd.DataFrame,
    history: pd.DataFrame,
    pipe,
    feats: List[str],
) -> Dict[str, Any]:
    df = df_raw[df_raw[TARGET_COL] > 0].copy()
    n_universe = int(len(df))
    df = _attach_regularity_from_history(df, history)
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
    xgb = np.clip(np.asarray(pipe.predict(X[feats]), dtype=np.float64), 1.0, None)

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
        label = classify_path_a(rec)["path_a_class"]
        diag = compute_path_a_interval_diagnostics(prior)
        if math.isnan(hist_med):
            hist_med = float(diag["hist_median"]) if not math.isnan(diag["hist_median"]) else float("nan")
        xp = float(xgb[idx])
        actual = float(r[TARGET_COL])
        pred_over = (xp / hist_med) if hist_med and hist_med > 0 else float("nan")
        passes_E = bool(np.isfinite(hist_med) and xp <= 60.0 and xp <= 2.0 * hist_med)
        is_c6 = bool(np.isfinite(pred_over) and pred_over <= 1.5)
        rows.append(
            {
                "label": label,
                "depth": int(r["purchase_count_so_far"]),
                "hist_median": hist_med,
                "norm_mad": float(diag["norm_mad"]) if not math.isnan(diag["norm_mad"]) else float("nan"),
                "drift": float(diag["cadence_drift"]) if not math.isnan(diag["cadence_drift"]) else float("nan"),
                "max_over_median": float(diag["max_over_median"]),
                "pct_within_50": float(diag["pct_within_50pct_of_median"]),
                "pct_in_band": float(diag["pct_in_refill_band_15_120"]),
                "pct_extreme": float(diag["pct_extreme_lt10_or_gt180"]),
                "xgb": xp,
                "actual": actual,
                "pred_over_hist": pred_over,
                "passes_E": passes_E,
                "is_c6": is_c6,
                "intervals_tail": prior[-12:],
            }
        )

    frame = pd.DataFrame(rows)
    high = frame[frame["label"] == LABEL_HIGH]
    med = frame[frame["label"] == LABEL_MEDIUM].copy()

    # C6-like MEDIUM-SAFE under Strategy E: MEDIUM & C6 & E
    safe = med[med["is_c6"] & med["passes_E"]].copy()
    # RISK residual: MEDIUM not in that SAFE set
    risk = med[~(med["is_c6"] & med["passes_E"])].copy()

    def preds_block(sub: pd.DataFrame) -> Dict[str, Any]:
        if len(sub) == 0:
            empty = _metrics(np.array([]), np.array([]))
            return {
                "n": 0,
                "coverage_of_split_pct": 0.0,
                "coverage_of_path_a_pct": 0.0,
                "personal_median": empty,
                "production_xgb": empty,
                "blend_50_50": empty,
            }
        hist_p = sub["hist_median"].to_numpy(dtype=np.float64)
        xgb_p = sub["xgb"].to_numpy(dtype=np.float64)
        blend = 0.5 * hist_p + 0.5 * xgb_p
        act = sub["actual"].to_numpy(dtype=np.float64)
        return {
            "n": int(len(sub)),
            "coverage_of_split_pct": round(100.0 * len(sub) / n_universe, 2),
            "coverage_of_path_a_pct": round(100.0 * len(sub) / path_a_n, 2) if path_a_n else 0.0,
            "personal_median": _metrics(hist_p, act),
            "production_xgb": _metrics(xgb_p, act),
            "blend_50_50": _metrics(blend, act),
        }

    # Pattern slices on SAFE rows (same predictors)
    def slice_safe(mask: np.ndarray, name: str) -> Dict[str, Any]:
        sub = safe.loc[mask]
        return {"name": name, "n": int(len(sub)), **{k: v for k, v in preds_block(sub).items() if k != "n"}}

    safe_slices = {}
    if len(safe):
        # variable-like: drift 10-15, max/med<=2.2, extreme low, band high (canonical-ish)
        var_like = (
            (safe["drift"] > 10)
            & (safe["drift"] <= 15)
            & (safe["max_over_median"] <= 2.2)
            & (safe["pct_extreme"] <= 10)
            & (safe["pct_in_band"] >= 80)
        )
        collapse = safe["actual"] < (0.5 * safe["hist_median"])
        inflation_near = (safe["pred_over_hist"] > 1.25) & (safe["pred_over_hist"] <= 1.5)
        shallow = safe["depth"] < 10
        deep = safe["depth"] >= 10
        safe_slices = {
            "variable_like_drift_10_15": slice_safe(var_like.to_numpy(), "variable_like"),
            "shallow_depth_6_9": slice_safe(shallow.to_numpy(), "shallow"),
            "deeper_depth_ge_10": slice_safe(deep.to_numpy(), "deep"),
            "pred_inflation_1_25_to_1_5x": slice_safe(inflation_near.to_numpy(), "inflation_near_cap"),
            "collapse_actual_lt_half_hist": slice_safe(collapse.to_numpy(), "collapse"),
        }

    # Best predictor helper
    def best_of(block: Dict[str, Any]) -> Dict[str, Any]:
        ranking = {}
        for metric, higher_better in [
            ("mae", False),
            ("medae", False),
            ("rmse", False),
            ("within_7_days_pct", True),
            ("catastrophic_pct", False),
            ("total_absolute_error", False),
            ("mean_bias_abs", False),
        ]:
            scores = {}
            for name in ("personal_median", "production_xgb", "blend_50_50"):
                m = block[name]
                if m["count"] == 0 or m.get("mae") is None:
                    continue
                if metric == "mean_bias_abs":
                    scores[name] = abs(m["mean_bias"]) if m["mean_bias"] is not None else None
                else:
                    scores[name] = m.get(metric.replace("_abs", "") if metric != "mean_bias_abs" else metric)
            scores = {k: v for k, v in scores.items() if v is not None}
            if not scores:
                ranking[metric] = None
                continue
            if higher_better:
                ranking[metric] = max(scores, key=scores.get)
            else:
                ranking[metric] = min(scores, key=scores.get)
        return ranking

    high_block = {
        "n": int(len(high)),
        "coverage_of_split_pct": round(100.0 * len(high) / n_universe, 2),
        "personal_median": _metrics(high["hist_median"].to_numpy(), high["actual"].to_numpy()),
        # HIGH reference uses personal median by design; still report XGB for contrast
        "production_xgb": _metrics(high["xgb"].to_numpy(), high["actual"].to_numpy()) if len(high) else _metrics(np.array([]), np.array([])),
        "blend_50_50": _metrics(
            0.5 * high["hist_median"].to_numpy() + 0.5 * high["xgb"].to_numpy(),
            high["actual"].to_numpy(),
        )
        if len(high)
        else _metrics(np.array([]), np.array([])),
    }

    safe_block = preds_block(safe)
    risk_block = preds_block(risk)

    # Median vs XGB head-to-head on identical SAFE rows
    head_to_head = {}
    if len(safe):
        ae_med = np.abs(safe["hist_median"] - safe["actual"])
        ae_xgb = np.abs(safe["xgb"] - safe["actual"])
        head_to_head = {
            "n": int(len(safe)),
            "median_closer_pct": round(100.0 * float((ae_med < ae_xgb).mean()), 2),
            "xgb_closer_pct": round(100.0 * float((ae_xgb < ae_med).mean()), 2),
            "tie_pct": round(100.0 * float((ae_med == ae_xgb).mean()), 2),
            "median_wins_by_ge_3d_pct": round(100.0 * float(((ae_xgb - ae_med) >= 3).mean()), 2),
            "xgb_wins_by_ge_3d_pct": round(100.0 * float(((ae_med - ae_xgb) >= 3).mean()), 2),
            "delta_mae_median_minus_xgb": round(float(ae_med.mean() - ae_xgb.mean()), 2),
            "delta_within_7d_median_minus_xgb_pp": round(
                100.0 * float((ae_med <= 7).mean() - (ae_xgb <= 7).mean()), 2
            ),
        }

    return {
        "split": split_name,
        "universe_n": n_universe,
        "path_a_n": path_a_n,
        "date_min": str(pd.to_datetime(df_raw["invoice_date"]).min().date()),
        "date_max": str(pd.to_datetime(df_raw["invoice_date"]).max().date()),
        "cohort_counts": {
            "HIGH": int(len(high)),
            "MEDIUM": int(len(med)),
            "MEDIUM_SAFE_C6_and_E": int(len(safe)),
            "MEDIUM_RISK_residual": int(len(risk)),
        },
        "HIGH_reference": high_block,
        "MEDIUM_SAFE_C6_and_E": safe_block,
        "MEDIUM_RISK_residual": risk_block,
        "MEDIUM_SAFE_best_predictor_by_metric": best_of(safe_block),
        "MEDIUM_SAFE_median_vs_xgb_head_to_head": head_to_head,
        "MEDIUM_SAFE_slices": safe_slices,
    }


def main() -> None:
    assert is_path_a_classifier_enabled() is False
    model_mtime_before = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    report_mtime_before = REPORT_PATH.stat().st_mtime if REPORT_PATH.exists() else None

    q_art = _find_quantile_artifact()
    print("Loading model + data...", flush=True)
    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipeline"]
    feats = list(bundle.get("features_numeric", NUMERIC_FEATURES)) + list(
        bundle.get("features_categorical", CATEGORICAL_FEATURES)
    )
    history = pd.read_parquet(PROC / "purchase_history.parquet")
    test = pd.read_parquet(PROC / "test.parquet")
    val = pd.read_parquet(PROC / "validation.parquet")

    print("Evaluating TEST holdout...", flush=True)
    test_res = _score_split("test", test, history, pipe, feats)
    print("Evaluating VALIDATION window...", flush=True)
    val_res = _score_split("validation", val, history, pipe, feats)

    # Canonical pattern probe (classification + calm/wild pred implication)
    canon = classify_path_a_from_intervals(VARIABLE_CANONICAL)
    diag = compute_path_a_interval_diagnostics(VARIABLE_CANONICAL)
    hist = float(diag["hist_median"])
    canonical = {
        "intervals": VARIABLE_CANONICAL,
        "path_a_class": canon["path_a_class"],
        "hist_median": hist,
        "diagnostics": {
            k: round(v, 4) if isinstance(v, float) and not math.isnan(v) else v
            for k, v in diag.items()
        },
        "note": (
            "No held-out actual for synthetic series. Under C6+E, membership depends on "
            "XGB pred: calm pred<=1.5x hist stays SAFE; wild pred becomes RISK."
        ),
        "would_be_SAFE_if_xgb_pred": {
            "1.1x_hist": True,  # 1.1 <= 1.5 and <=2 and <60
            "1.5x_hist": True,
            "1.6x_hist": False,  # fails C6
            "2.1x_hist": False,  # fails C6 and E
        },
    }

    del bundle, pipe, history, test, val
    gc.collect()

    model_mtime_after = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    report_mtime_after = REPORT_PATH.stat().st_mtime if REPORT_PATH.exists() else None

    results = {
        "experiment": "PHASE_17I_MEDIUM_SAFE_PREDICTOR_COMPARISON",
        "read_only": True,
        "strategy_promoted": False,
        "controls": {
            "path_a_only": True,
            "safe_definition": "17F MEDIUM AND pred<=1.5*hist_median AND Strategy E (pred<=60 AND pred<=2*hist)",
            "predictors": [
                "personal_historical_median",
                "production_xgb_squarederror",
                "blend_50pct_median_50pct_xgb",
            ],
            "quantile_xgb_artifact_path": q_art,
            "quantile_xgb_evaluated": q_art is not None,
            "quantile_xgb_note": (
                None
                if q_art
                else "No saved quantile-XGB artifact found under data/refillcare/processed/models/; not trained in this phase."
            ),
            "feature_flag_enabled": False,
            "catastrophic_definition": f"abs_error > {CATASTROPHIC:.0f} days",
        },
        "test": test_res,
        "validation": val_res,
        "canonical_variable_pattern": canonical,
        "artifact_safety": {
            "refill_model_joblib_mtime_unchanged": model_mtime_before == model_mtime_after,
            "phase4_report_mtime_unchanged": report_mtime_before == report_mtime_after,
            "model_mtime": model_mtime_before,
            "whatsapp_messages_sent": 0,
            "production_routing_modified": False,
            "retrain": False,
            "path_b_implemented": False,
            "feature_flag_remains_off": True,
        },
    }

    out = PROC / "phase17i_medium_safe_predictor_results.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    for split_key in ("test", "validation"):
        s = results[split_key]
        safe = s["MEDIUM_SAFE_C6_and_E"]
        print(
            f"{split_key}: SAFE n={safe['n']} | "
            f"median MAE={safe['personal_median']['mae']} ±7={safe['personal_median']['within_7_days_pct']} | "
            f"xgb MAE={safe['production_xgb']['mae']} ±7={safe['production_xgb']['within_7_days_pct']} | "
            f"blend MAE={safe['blend_50_50']['mae']} ±7={safe['blend_50_50']['within_7_days_pct']}",
            flush=True,
        )
        print("  best-by-metric:", s["MEDIUM_SAFE_best_predictor_by_metric"], flush=True)
        print("  head-to-head:", s["MEDIUM_SAFE_median_vs_xgb_head_to_head"], flush=True)
    print(f"Wrote {out}", flush=True)
    print("Quantile artifact:", q_art, flush=True)
    print("Artifact safety:", json.dumps(results["artifact_safety"]), flush=True)


if __name__ == "__main__":
    main()
