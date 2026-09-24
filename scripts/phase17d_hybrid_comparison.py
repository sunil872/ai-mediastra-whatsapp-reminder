"""READ-ONLY Phase 17D hybrid prediction comparison.

Compares:
1. current production XGBoost (refill_model.joblib, read-only)
2. personal historical median baseline
3. best personal-cadence strategy (tiered heuristic from cadence experiment)
4. best experimental XGBoost (reg:quantileerror q50, in-memory only)
5. rule-based hybrid routing (thresholds from Phase 17D regularity /
   history-depth / long-interval / cadence evidence)

Does NOT modify production code, model artifacts, reminder logic, or WhatsApp.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from refillcare.models.training import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
)
from refillcare.evaluation.baseline import HistoricalMedianBaseline

PROC = ROOT / "data" / "refillcare" / "processed"
MODEL_PATH = PROC / "models" / "refill_model.joblib"
REPORT_PATH = PROC / "phase4_model_report.json"

# Phase 17D–supported hybrid thresholds
# Regularity Candidate 5 / Candidate 6 (PHASE_17D_REGULARITY_THRESHOLD_SELECTION)
# History depth minimum P>=6 (PHASE_17D_HISTORY_DEPTH_ANALYSIS)
# Cadence band 15–120d (is_recurring_history / long-interval implications)
HYBRID = {
    "min_purchases": 6,
    "core_norm_mad_max": 0.35,
    "core_drift_max": 7.0,
    "broad_norm_mad_max": 0.50,
    "broad_drift_max": 10.0,
    "cadence_median_min": 15.0,
    "cadence_median_max": 120.0,
    "pack_default_days": 30.0,
}


def _metrics(y_true, y_pred) -> dict:
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.clip(np.asarray(y_pred, dtype=np.float64), 1.0, None)
    mask = ~np.isnan(y_t) & ~np.isnan(y_p)
    y_t, y_p = y_t[mask], y_p[mask]
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


def _build_preprocessor(feats: list[str]) -> tuple[ColumnTransformer, list[str], list[str]]:
    num_cols = [c for c in NUMERIC_FEATURES if c in feats]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feats]
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
                        (
                            "ohe",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                                min_frequency=50,
                            ),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )
    return pre, num_cols, cat_cols


def _attach_regularity_from_history(test: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Compute leakage-safe expanding Norm MAD and recent-3 drift for test rows."""
    hist = history[
        ["customerId", "itemId", "invoice_date", "days_since_previous_purchase", "purchase_seq"]
    ].copy()
    hist["invoice_date"] = pd.to_datetime(hist["invoice_date"])
    hist = hist.sort_values(["customerId", "itemId", "invoice_date", "purchase_seq"])

    keys = set(zip(test["customerId"].astype(str), test["itemId"].astype(str)))
    hist["_key"] = list(zip(hist["customerId"].astype(str), hist["itemId"].astype(str)))
    hist = hist[hist["_key"].isin(keys)].copy()

    # Expanding stats keyed by (customerId, itemId, purchase_seq)
    records = []
    for (cid, iid), g in hist.groupby(["customerId", "itemId"], sort=False):
        intervals: list[float] = []
        for _, row in g.iterrows():
            d = row["days_since_previous_purchase"]
            if pd.notna(d) and float(d) > 0:
                intervals.append(float(d))
            seq = int(row["purchase_seq"])
            if len(intervals) == 0:
                mad = np.nan
                norm_mad = np.nan
                recent3 = np.nan
                drift = np.nan
                recent5 = np.nan
            else:
                arr = np.asarray(intervals, dtype=np.float64)
                med = float(np.median(arr))
                mad = float(np.median(np.abs(arr - med)))
                norm_mad = float(mad / med) if med > 0 else np.nan
                recent3 = float(np.median(arr[-3:]))
                recent5 = float(np.median(arr[-5:]))
                drift = abs(recent3 - med) if len(intervals) >= 1 else np.nan
            records.append(
                {
                    "customerId": cid,
                    "itemId": iid,
                    "purchase_seq": seq,
                    "norm_mad": norm_mad,
                    "mad": mad,
                    "recent3_median": recent3,
                    "recent5_median": recent5,
                    "cadence_drift": drift,
                    "n_prior_intervals": len(intervals),
                }
            )

    reg = pd.DataFrame.from_records(records)
    out = test.copy()
    out["customerId"] = out["customerId"].astype(str)
    out["itemId"] = out["itemId"].astype(str)
    # purchase_count_so_far aligns with purchase_seq in feature engineering
    if "purchase_seq" not in out.columns:
        out["purchase_seq"] = out["purchase_count_so_far"]
    out = out.merge(
        reg,
        on=["customerId", "itemId", "purchase_seq"],
        how="left",
        validate="many_to_one",
    )
    return out


def _tiered_cadence_predict(df: pd.DataFrame, fallback: float) -> np.ndarray:
    """Best personal-cadence strategy from PHASE_17D_PERSONAL_CADENCE_EXPERIMENT."""
    p = df["purchase_count_so_far"].to_numpy()
    hist_med = df["historical_interval_median"].to_numpy(dtype=np.float64)
    recent5 = df["recent5_median"].to_numpy(dtype=np.float64)
    recent3 = df["recent3_median"].to_numpy(dtype=np.float64)
    pack = HYBRID["pack_default_days"]
    pred = np.full(len(df), fallback, dtype=np.float64)

    # Tier D: P <= 3 → fixed pack default
    mask_d = p <= 3
    pred[mask_d] = pack

    # Tier C: P 4–5 → recent-3 median (fallback hist / pack)
    mask_c = (p >= 4) & (p <= 5)
    c_vals = np.where(np.isnan(recent3), hist_med, recent3)
    c_vals = np.where(np.isnan(c_vals), pack, c_vals)
    pred[mask_c] = c_vals[mask_c]

    # Tier B: P 6–10 → recent-5 (else hist median)
    mask_b = (p >= 6) & (p <= 10)
    b_vals = np.where(np.isnan(recent5), hist_med, recent5)
    b_vals = np.where(np.isnan(b_vals), fallback, b_vals)
    pred[mask_b] = b_vals[mask_b]

    # Tier A: P > 10 → full historical median
    mask_a = p > 10
    a_vals = np.where(np.isnan(hist_med), fallback, hist_med)
    pred[mask_a] = a_vals[mask_a]

    return pred


def _hybrid_route(df: pd.DataFrame, hist_pred: np.ndarray, exp_xgb_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Rule-based hybrid using only Phase 17D–supported thresholds.

    Returns:
        predictions (NaN where rejected), eligible mask, routing counts
    """
    p = df["purchase_count_so_far"].to_numpy()
    norm_mad = df["norm_mad"].to_numpy(dtype=np.float64)
    drift = df["cadence_drift"].to_numpy(dtype=np.float64)
    hist_med = df["historical_interval_median"].to_numpy(dtype=np.float64)

    cadence_ok = (
        ~np.isnan(hist_med)
        & (hist_med >= HYBRID["cadence_median_min"])
        & (hist_med <= HYBRID["cadence_median_max"])
    )
    depth_ok = p >= HYBRID["min_purchases"]
    broad_ok = (
        depth_ok
        & cadence_ok
        & ~np.isnan(norm_mad)
        & ~np.isnan(drift)
        & (norm_mad <= HYBRID["broad_norm_mad_max"])
        & (drift <= HYBRID["broad_drift_max"])
    )
    core_ok = (
        broad_ok
        & (norm_mad <= HYBRID["core_norm_mad_max"])
        & (drift <= HYBRID["core_drift_max"])
    )
    # Broad-but-not-core: still Candidate-5 eligible, route to experimental XGB
    secondary_ok = broad_ok & ~core_ok

    pred = np.full(len(df), np.nan, dtype=np.float64)
    route = np.full(len(df), "rejected_ineligible", dtype=object)
    pred[core_ok] = hist_pred[core_ok]
    route[core_ok] = "core_personal_median"
    pred[secondary_ok] = exp_xgb_pred[secondary_ok]
    route[secondary_ok] = "secondary_experimental_xgb"

    eligible = broad_ok
    counts = {
        "eligible": int(eligible.sum()),
        "rejected_ineligible": int((~eligible).sum()),
        "core_personal_median": int(core_ok.sum()),
        "secondary_experimental_xgb": int(secondary_ok.sum()),
        "reject_reasons": {
            "depth_lt_6": int((~depth_ok).sum()),
            "cadence_median_out_of_15_120": int((depth_ok & ~cadence_ok).sum()),
            "broad_regularity_fail": int(
                (depth_ok & cadence_ok & ~broad_ok).sum()
            ),
        },
    }
    return pred, eligible, {"route_counts": counts, "route_labels": route.tolist()}


def _cohort_report(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    p = df["purchase_count_so_far"].to_numpy()
    norm_mad = df["norm_mad"].to_numpy(dtype=np.float64)
    drift = df["cadence_drift"].to_numpy(dtype=np.float64)

    cohorts = {
        "overall": np.ones(len(df), dtype=bool),
        "depth_le_2": p <= 2,
        "depth_3_to_5": (p >= 3) & (p <= 5),
        "depth_6_to_10": (p >= 6) & (p <= 10),
        "depth_gt_10": p > 10,
        "depth_ge_6": p >= 6,
        # Regularity classes aligned with PHASE_17D_REGULARITY_ANALYSIS
        "class1_core_regular": (
            (p >= 6)
            & (norm_mad <= 0.35)
            & (drift <= 7)
            & ~np.isnan(norm_mad)
            & ~np.isnan(drift)
        ),
        "class2_high_volume_variable": (
            (p >= 6)
            & ~((norm_mad <= 0.35) & (drift <= 7))
        ),
        "class3_emerging_p4_5": (p >= 4) & (p <= 5),
        "class4_sparse_le_3": p <= 3,
        "is_recurring_history": df["is_recurring_history"].to_numpy() == 1,
    }

    out = {}
    for name, mask in cohorts.items():
        # only score rows with non-NaN predictions inside the cohort
        m = mask & ~np.isnan(y_p) & ~np.isnan(y_t)
        met = _metrics(y_t[m], y_p[m])
        met["cohort_rows"] = int(mask.sum())
        met["scored_rows"] = int(m.sum())
        met["coverage_within_cohort_pct"] = (
            round(100.0 * m.sum() / mask.sum(), 2) if mask.sum() else None
        )
        out[name] = met
    return out


def main() -> None:
    model_mtime_before = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    report_mtime_before = REPORT_PATH.stat().st_mtime if REPORT_PATH.exists() else None

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    keep = list(
        dict.fromkeys(
            feature_cols
            + [
                TARGET_COL,
                "invoice_date",
                "purchase_count_so_far",
                "customerId",
                "itemId",
                "is_recurring_history",
                "historical_interval_median",
            ]
        )
    )

    print("Loading partitions (read-only)...", flush=True)
    train = pd.read_parquet(PROC / "train.parquet")
    test = pd.read_parquet(PROC / "test.parquet")
    train_cols = [c for c in keep if c in train.columns]
    train = train[train_cols].copy()
    test = test[[c for c in keep if c in test.columns]].copy()

    train = train[train[TARGET_COL] > 0].copy()
    test = test[test[TARGET_COL] > 0].copy()
    test_n = len(test)

    print(f"Train/test (target>0): {len(train):,} / {test_n:,}", flush=True)

    print("Computing Norm MAD + recent cadence from purchase_history...", flush=True)
    history = pd.read_parquet(PROC / "purchase_history.parquet")
    test = _attach_regularity_from_history(test, history)
    del history
    gc.collect()

    # Baseline / cadence predictors
    baseline = HistoricalMedianBaseline()
    baseline.fit(train, target_col=TARGET_COL)
    hist_pred = baseline.predict(test)
    cadence_pred = _tiered_cadence_predict(test, baseline.fallback_median)

    # Current production XGB (read-only load)
    print("Scoring current production XGBoost...", flush=True)
    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipeline"]
    feats_prod = bundle.get("features_numeric", NUMERIC_FEATURES) + bundle.get(
        "features_categorical", CATEGORICAL_FEATURES
    )
    X_test_prod = test.copy()
    for c in feats_prod:
        if c not in X_test_prod.columns:
            X_test_prod[c] = np.nan
    current_xgb_pred = np.clip(pipe.predict(X_test_prod[feats_prod]), 1.0, None)
    del bundle, pipe, X_test_prod
    gc.collect()

    # Best experimental XGB (in-memory only; quantileerror q50)
    print("Training in-memory experimental XGBoost (quantileerror q50)...", flush=True)
    feats = [c for c in feature_cols if c in train.columns]
    pre, num_cols, cat_cols = _build_preprocessor(feats)
    X_train = pre.fit_transform(train[feats])
    X_test = pre.transform(test[feats])
    y_train = train[TARGET_COL].to_numpy(dtype=np.float64)
    t0 = time.time()
    exp_model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=0.5,
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=2,
        tree_method="hist",
    )
    exp_model.fit(X_train, y_train)
    exp_train_sec = round(time.time() - t0, 2)
    exp_xgb_pred = np.clip(exp_model.predict(X_test), 1.0, None)
    del exp_model, X_train, X_test, pre
    gc.collect()
    print(f"  experimental XGB trained in {exp_train_sec}s", flush=True)

    # Hybrid routing
    hybrid_pred, eligible, route_meta = _hybrid_route(test, hist_pred, exp_xgb_pred)
    y_true = test[TARGET_COL].to_numpy(dtype=np.float64)

    strategies = {
        "current_xgboost": {
            "description": "Production refill_model.joblib (reg:squarederror)",
            "pred": current_xgb_pred,
            "always_on": True,
        },
        "personal_historical_median": {
            "description": "Personal hist median; train-global fallback when missing",
            "pred": hist_pred,
            "always_on": True,
        },
        "best_personal_cadence": {
            "description": (
                "Tiered cadence (P>10 hist median; P6–10 recent-5; P4–5 recent-3; "
                "P<=3 fixed 30d pack) from PHASE_17D_PERSONAL_CADENCE_EXPERIMENT"
            ),
            "pred": cadence_pred,
            "always_on": True,
        },
        "best_experimental_xgboost": {
            "description": "In-memory reg:quantileerror quantile_alpha=0.5 (PHASE_17D_XGB_OBJECTIVE)",
            "pred": exp_xgb_pred,
            "always_on": True,
        },
        "hybrid_routing": {
            "description": (
                "Eligible if P>=6 & NormMAD<=0.50 & Drift<=10 & hist_median in [15,120]; "
                "core (NormMAD<=0.35 & Drift<=7) → personal median; "
                "else eligible → experimental XGB; otherwise reject"
            ),
            "pred": hybrid_pred,
            "always_on": False,
        },
    }

    results = {
        "experiment": "PHASE_17D_HYBRID_COMPARISON",
        "read_only": True,
        "production_artifacts_modified": False,
        "xgboost_version": xgb.__version__,
        "controls": {
            "test_rows_target_gt_0": test_n,
            "test_date_min": str(pd.to_datetime(test["invoice_date"]).min().date()),
            "test_date_max": str(pd.to_datetime(test["invoice_date"]).max().date()),
            "train_rows_target_gt_0": int(len(train)),
            "fallback_median_days": baseline.fallback_median,
            "experimental_xgb_train_sec": exp_train_sec,
            "hybrid_thresholds": HYBRID,
            "threshold_sources": [
                "docs/PHASE_17D_REGULARITY_THRESHOLD_SELECTION.md (Candidate 5/6)",
                "docs/PHASE_17D_REGULARITY_ANALYSIS.md (Class 1/2)",
                "docs/PHASE_17D_HISTORY_DEPTH_ANALYSIS.md (P>=6 minimum)",
                "docs/PHASE_17D_LONG_INTERVAL_ANALYSIS.md / is_recurring_history (15–120d)",
                "docs/PHASE_17D_PERSONAL_CADENCE_EXPERIMENT.md (tiered cadence)",
                "docs/PHASE_17D_XGB_OBJECTIVE_EXPERIMENT.md (quantileerror q50)",
            ],
        },
        "strategies": {},
    }

    for name, spec in strategies.items():
        pred = np.asarray(spec["pred"], dtype=np.float64)
        scored = ~np.isnan(pred)
        coverage = {
            "eval_universe": test_n,
            "predictions_emitted": int(scored.sum()),
            "coverage_pct": round(100.0 * scored.sum() / test_n, 2),
            "rejected_ineligible": int((~scored).sum()),
            "rejection_ineligible_pct": round(100.0 * (~scored).sum() / test_n, 2),
        }
        overall = _metrics(y_true[scored], pred[scored])
        on_eligible = _metrics(y_true[eligible], pred[eligible]) if spec["always_on"] else _metrics(
            y_true[eligible & scored], pred[eligible & scored]
        )
        entry = {
            "description": spec["description"],
            "always_on": spec["always_on"],
            "coverage": coverage,
            "overall_on_scored_rows": overall,
            "on_hybrid_eligible_subset": on_eligible,
            "cohorts_on_scored_rows": _cohort_report(test, y_true, pred),
        }
        if name == "hybrid_routing":
            entry["routing"] = {
                "route_counts": route_meta["route_counts"],
                # do not dump full route label list into JSON (large); summarize only
            }
        results["strategies"][name] = entry
        print(
            f"{name}: MAE={overall['mae']} MedAE={overall['medae']} "
            f"±7d={overall['within_7_days_pct']}% "
            f"coverage={coverage['coverage_pct']}% reject={coverage['rejection_ineligible_pct']}%",
            flush=True,
        )

    # Ranking helpers
    results["ranking_overall_mae_on_scored"] = sorted(
        [
            {
                "strategy": k,
                "mae": v["overall_on_scored_rows"]["mae"],
                "medae": v["overall_on_scored_rows"]["medae"],
                "within_7_days_pct": v["overall_on_scored_rows"]["within_7_days_pct"],
                "coverage_pct": v["coverage"]["coverage_pct"],
            }
            for k, v in results["strategies"].items()
            if v["overall_on_scored_rows"]["mae"] is not None
        ],
        key=lambda r: (r["mae"], -r["coverage_pct"]),
    )
    results["ranking_on_hybrid_eligible_mae"] = sorted(
        [
            {
                "strategy": k,
                "mae": v["on_hybrid_eligible_subset"]["mae"],
                "medae": v["on_hybrid_eligible_subset"]["medae"],
                "within_7_days_pct": v["on_hybrid_eligible_subset"]["within_7_days_pct"],
                "count": v["on_hybrid_eligible_subset"]["count"],
            }
            for k, v in results["strategies"].items()
            if v["on_hybrid_eligible_subset"]["mae"] is not None
        ],
        key=lambda r: r["mae"],
    )

    model_mtime_after = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    report_mtime_after = REPORT_PATH.stat().st_mtime if REPORT_PATH.exists() else None
    results["artifact_safety"] = {
        "refill_model_joblib_mtime_unchanged": model_mtime_before == model_mtime_after,
        "phase4_report_mtime_unchanged": report_mtime_before == report_mtime_after,
        "model_mtime_before": model_mtime_before,
        "model_mtime_after": model_mtime_after,
        "production_code_modified": False,
        "reminder_logic_modified": False,
        "whatsapp_modified": False,
    }

    out = PROC / "phase17d_hybrid_comparison_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out}", flush=True)
    print("Ranking overall:", results["ranking_overall_mae_on_scored"], flush=True)
    print("Ranking eligible:", results["ranking_on_hybrid_eligible_mae"], flush=True)
    print("Artifact safety:", results["artifact_safety"], flush=True)


if __name__ == "__main__":
    main()
