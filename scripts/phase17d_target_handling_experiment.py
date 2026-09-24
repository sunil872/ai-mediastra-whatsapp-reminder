"""OFFLINE Phase 17D experiment: target-handling strategies.

Evidence from PHASE_17D_LONG_INTERVAL_ANALYSIS:
- >180d gaps (~12%) are mostly irregular/churn, not refill cadence
- >120d is arguably non-core for reminder training
- Heavy right tail pulls squared-error fits

Strategies (training-target only; evaluate vs original targets):
1. unchanged
2. cap_120  — min(y, 120)
3. cap_180  — min(y, 180)
4. log1p    — train on log1p(y), invert with expm1 at predict

Identical temporal val/test; production artifacts untouched.
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
from refillcare.models.evaluation import evaluate_regression_model

PROC = ROOT / "data" / "refillcare" / "processed"
MODEL_PATH = PROC / "models" / "refill_model.joblib"
REPORT_PATH = PROC / "phase4_model_report.json"

COMMON_XGB = dict(
    objective="reg:squarederror",
    n_estimators=150,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=2,
    tree_method="hist",
)

STRATEGIES = [
    {
        "name": "unchanged",
        "kind": "identity",
        "evidence": "Control: Phase 4 / production target as-is",
    },
    {
        "name": "cap_120",
        "kind": "cap",
        "cap": 120.0,
        "evidence": (
            "Long-interval analysis: >120d arguably non-core refill signal; "
            "91–120 still mixed but >120 increasingly irregular"
        ),
    },
    {
        "name": "cap_180",
        "kind": "cap",
        "cap": 180.0,
        "evidence": (
            "Long-interval analysis: >180d = 12.3%, median depth 2, "
            "~65% irregular/cold-start/break; dominate overall MAE"
        ),
    },
    {
        "name": "log1p",
        "kind": "log1p",
        "evidence": (
            "Robust transform for heavy-tailed refill intervals "
            "(ultra-long gaps inflate squared error on raw days scale)"
        ),
    },
]


def _transform_train_target(y: np.ndarray, strategy: dict) -> np.ndarray:
    kind = strategy["kind"]
    y = np.asarray(y, dtype=np.float64)
    if kind == "identity":
        return y.copy()
    if kind == "cap":
        return np.minimum(y, float(strategy["cap"]))
    if kind == "log1p":
        return np.log1p(y)
    raise ValueError(f"Unknown strategy kind: {kind}")


def _invert_predictions(y_pred: np.ndarray, strategy: dict) -> np.ndarray:
    kind = strategy["kind"]
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if kind == "log1p":
        return np.expm1(y_pred)
    # Cap strategies train on capped labels but emit days-scale predictions as-is
    return y_pred


def _metrics(y_true, y_pred) -> dict:
    m = evaluate_regression_model(y_true, y_pred)
    return {
        "count": m["count"],
        "mae": m["mae"],
        "medae": m["median_abs_error"],
        "rmse": m["rmse"],
        "within_3_days_pct": m["within_3_days_pct"],
        "within_7_days_pct": m["within_7_days_pct"],
        "actual_target": m.get("actual_target"),
        "predicted_target": m.get("predicted_target"),
    }


def _cohort_metrics(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.clip(np.asarray(y_pred, dtype=np.float64), 1.0, None)
    seq = df["purchase_count_so_far"].to_numpy()

    cohorts = {
        "overall": np.ones(len(y_t), dtype=bool),
        "depth_2": seq <= 2,
        "depth_3_to_5": (seq >= 3) & (seq <= 5),
        "depth_gt_5": seq > 5,
        "actual_le_90d": y_t <= 90,
        "actual_le_120d": y_t <= 120,
        "actual_le_180d": y_t <= 180,
        "actual_gt_180d": y_t > 180,
        "interval_0_14": y_t <= 14,
        "interval_15_45": (y_t >= 15) & (y_t <= 45),
        "interval_46_90": (y_t >= 46) & (y_t <= 90),
        "interval_91_180": (y_t >= 91) & (y_t <= 180),
        "interval_gt_180": y_t > 180,
    }

    out = {}
    for name, mask in cohorts.items():
        if not np.any(mask):
            out[name] = {
                "count": 0,
                "mae": None,
                "medae": None,
                "rmse": None,
                "within_3_days_pct": None,
                "within_7_days_pct": None,
            }
            continue
        m = _metrics(y_t[mask], y_p[mask])
        out[name] = {
            k: m[k]
            for k in (
                "count",
                "mae",
                "medae",
                "rmse",
                "within_3_days_pct",
                "within_7_days_pct",
            )
        }
    return out


def _build_preprocessor(feats: list[str]) -> ColumnTransformer:
    num_cols = [c for c in NUMERIC_FEATURES if c in feats]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feats]
    return ColumnTransformer(
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
    ), num_cols, cat_cols


def main() -> None:
    model_mtime_before = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    report_mtime_before = REPORT_PATH.stat().st_mtime if REPORT_PATH.exists() else None

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    keep = list(
        dict.fromkeys(feature_cols + [TARGET_COL, "invoice_date", "purchase_count_so_far"])
    )

    print("Loading identical Phase 4 partitions...", flush=True)
    probe = pd.read_parquet(PROC / "train.parquet")
    train_cols = [c for c in keep if c in probe.columns]
    del probe
    gc.collect()

    train = pd.read_parquet(PROC / "train.parquet", columns=train_cols)
    val = pd.read_parquet(PROC / "validation.parquet", columns=train_cols)
    test = pd.read_parquet(PROC / "test.parquet", columns=train_cols)

    train = train[train[TARGET_COL] > 0].copy()
    val = val[val[TARGET_COL] > 0].copy()
    test = test[test[TARGET_COL] > 0].copy()

    feats = [c for c in feature_cols if c in train.columns]
    print(f"Features used ({len(feats)}): {feats}", flush=True)
    print(
        f"Rows train/val/test: {len(train):,} / {len(val):,} / {len(test):,}",
        flush=True,
    )

    y_train_raw = train[TARGET_COL].to_numpy(dtype=np.float64)
    y_val_raw = val[TARGET_COL].to_numpy(dtype=np.float64)
    y_test_raw = test[TARGET_COL].to_numpy(dtype=np.float64)

    print("Fitting identical preprocessor once...", flush=True)
    pre, num_cols, cat_cols = _build_preprocessor(feats)
    X_train = pre.fit_transform(train[feats])
    X_val = pre.transform(val[feats])
    X_test = pre.transform(test[feats])
    del pre
    gc.collect()
    print(f"Transformed feature matrix shape: {X_train.shape}", flush=True)

    # Training-label diagnostics (evidence context)
    train_label_stats = {
        "raw": {
            "mean": round(float(np.mean(y_train_raw)), 2),
            "median": round(float(np.median(y_train_raw)), 2),
            "p95": round(float(np.percentile(y_train_raw, 95)), 2),
            "max": round(float(np.max(y_train_raw)), 2),
            "pct_gt_120": round(float((y_train_raw > 120).mean() * 100), 2),
            "pct_gt_180": round(float((y_train_raw > 180).mean() * 100), 2),
        }
    }

    results = {
        "experiment": "PHASE_17D_TARGET_HANDLING_EXPERIMENT",
        "xgboost_version": xgb.__version__,
        "read_only": True,
        "production_model_replaced": False,
        "evaluation_note": (
            "All strategies evaluated against ORIGINAL (uncapped, untransformed) "
            "validation/test targets for operational fairness."
        ),
        "evidence_basis": "docs/PHASE_17D_LONG_INTERVAL_ANALYSIS.md",
        "controls": {
            "features_numeric": num_cols,
            "features_categorical": cat_cols,
            "hyperparameters": COMMON_XGB,
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "test_rows": int(len(test)),
            "train_date_min": str(pd.to_datetime(train["invoice_date"]).min().date()),
            "train_date_max": str(pd.to_datetime(train["invoice_date"]).max().date()),
            "val_date_min": str(pd.to_datetime(val["invoice_date"]).min().date()),
            "val_date_max": str(pd.to_datetime(val["invoice_date"]).max().date()),
            "test_date_min": str(pd.to_datetime(test["invoice_date"]).min().date()),
            "test_date_max": str(pd.to_datetime(test["invoice_date"]).max().date()),
            "preprocessing": "Single fit ColumnTransformer (median impute + OHE min_frequency=50)",
            "target_filter": "target_days_until_next_purchase > 0",
            "fixed_objective": "reg:squarederror (production default; isolates target handling)",
        },
        "train_label_stats": train_label_stats,
        "strategies": {},
    }

    for strategy in STRATEGIES:
        name = strategy["name"]
        print(f"\nTraining strategy={name} ...", flush=True)
        y_train = _transform_train_target(y_train_raw, strategy)
        train_label_stats[name] = {
            "mean": round(float(np.mean(y_train)), 2),
            "median": round(float(np.median(y_train)), 2),
            "p95": round(float(np.percentile(y_train, 95)), 2),
            "max": round(float(np.max(y_train)), 2),
            "n_changed_vs_raw": int(np.sum(~np.isclose(y_train, y_train_raw))),
        }

        t0 = time.time()
        model = xgb.XGBRegressor(**COMMON_XGB)
        model.fit(X_train, y_train)
        train_sec = round(time.time() - t0, 2)

        val_pred = _invert_predictions(model.predict(X_val), strategy)
        test_pred = _invert_predictions(model.predict(X_test), strategy)

        results["strategies"][name] = {
            "kind": strategy["kind"],
            "cap": strategy.get("cap"),
            "evidence": strategy["evidence"],
            "training_time_sec": train_sec,
            "train_target_summary": train_label_stats[name],
            "validation": _metrics(y_val_raw, val_pred),
            "test": _metrics(y_test_raw, test_pred),
            "validation_cohorts": _cohort_metrics(val, y_val_raw, val_pred),
            "test_cohorts": _cohort_metrics(test, y_test_raw, test_pred),
        }
        print(
            f"  done in {train_sec}s | Val MAE={results['strategies'][name]['validation']['mae']} "
            f"| Test MAE={results['strategies'][name]['test']['mae']} "
            f"| Test ±7d={results['strategies'][name]['test']['within_7_days_pct']}%",
            flush=True,
        )
        del model
        gc.collect()

    ranking = sorted(
        results["strategies"].items(),
        key=lambda kv: (
            kv[1]["validation"]["mae"] if kv[1]["validation"]["mae"] is not None else 1e9,
            kv[1]["validation"]["medae"] if kv[1]["validation"]["medae"] is not None else 1e9,
        ),
    )
    results["ranking_by_validation_mae"] = [
        {
            "strategy": k,
            "val_mae": v["validation"]["mae"],
            "val_medae": v["validation"]["medae"],
            "val_within_7d": v["validation"]["within_7_days_pct"],
            "test_mae": v["test"]["mae"],
            "test_medae": v["test"]["medae"],
            "test_within_7d": v["test"]["within_7_days_pct"],
            "test_depth_gt5_mae": v["test_cohorts"]["depth_gt_5"]["mae"],
            "test_actual_le_180_mae": v["test_cohorts"]["actual_le_180d"]["mae"],
        }
        for k, v in ranking
    ]

    model_mtime_after = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    report_mtime_after = REPORT_PATH.stat().st_mtime if REPORT_PATH.exists() else None
    results["artifact_safety"] = {
        "refill_model_joblib_mtime_unchanged": model_mtime_before == model_mtime_after,
        "phase4_report_mtime_unchanged": report_mtime_before == report_mtime_after,
        "model_mtime_before": model_mtime_before,
        "model_mtime_after": model_mtime_after,
    }
    results["train_label_stats"] = train_label_stats

    out = PROC / "phase17d_target_handling_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out}", flush=True)
    print("Ranking:", results["ranking_by_validation_mae"], flush=True)
    print("Artifact safety:", results["artifact_safety"], flush=True)


if __name__ == "__main__":
    main()
