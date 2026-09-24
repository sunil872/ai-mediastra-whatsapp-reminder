"""OFFLINE Phase 17D experiment: XGBoost objective comparison.

Identical features, training data, val/test windows, and preprocessing.
Does NOT overwrite production model artifacts.
Writes JSON results under data/refillcare/processed/ for report generation.
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

from refillcare.models.training import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
    build_preprocessor,
)
from refillcare.models.evaluation import evaluate_regression_model

PROC = ROOT / "data" / "refillcare" / "processed"
MODEL_PATH = PROC / "models" / "refill_model.joblib"
REPORT_PATH = PROC / "phase4_model_report.json"

# Same hyperparams as Phase 4 XGBoost (objective is the only intentional change)
COMMON_XGB = dict(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=2,
    tree_method="hist",
)

OBJECTIVES = [
    {"name": "reg:squarederror", "objective": "reg:squarederror", "extra": {}},
    {"name": "reg:absoluteerror", "objective": "reg:absoluteerror", "extra": {}},
    {"name": "reg:pseudohubererror", "objective": "reg:pseudohubererror", "extra": {}},
    {
        "name": "reg:quantileerror_q50",
        "objective": "reg:quantileerror",
        "extra": {"quantile_alpha": 0.5},
    },
]


def _pred_distribution(y_pred: np.ndarray) -> dict:
    y = np.clip(np.asarray(y_pred, dtype=np.float64), 1.0, None)
    return {
        "mean": round(float(np.mean(y)), 2),
        "median": round(float(np.median(y)), 2),
        "std": round(float(np.std(y)), 2),
        "p5": round(float(np.percentile(y, 5)), 2),
        "p25": round(float(np.percentile(y, 25)), 2),
        "p75": round(float(np.percentile(y, 75)), 2),
        "p95": round(float(np.percentile(y, 95)), 2),
        "min": round(float(np.min(y)), 2),
        "max": round(float(np.max(y)), 2),
        "pct_clipped_to_1": round(float((y <= 1.0).mean() * 100.0), 2),
    }


def _metrics(y_true, y_pred) -> dict:
    m = evaluate_regression_model(y_true, y_pred)
    return {
        "count": m["count"],
        "mae": m["mae"],
        "medae": m["median_abs_error"],
        "rmse": m["rmse"],
        "within_3_days_pct": m["within_3_days_pct"],
        "within_7_days_pct": m["within_7_days_pct"],
        "prediction_distribution": _pred_distribution(y_pred),
        "actual_target_summary": m.get("actual_target"),
    }


def _history_depth(df: pd.DataFrame, y_pred: np.ndarray) -> dict:
    y_t = df[TARGET_COL].to_numpy(dtype=np.float64)
    y_p = np.clip(np.asarray(y_pred, dtype=np.float64), 1.0, None)
    seq = df["purchase_count_so_far"].to_numpy()
    out = {}
    for name, mask in {
        "2_purchases": seq <= 2,
        "3_to_5_purchases": (seq >= 3) & (seq <= 5),
        "gt_5_purchases": seq > 5,
    }.items():
        if not np.any(mask):
            out[name] = {
                "count": 0, "mae": None, "medae": None, "rmse": None,
                "within_3_days_pct": None, "within_7_days_pct": None,
            }
            continue
        out[name] = {
            k: v for k, v in _metrics(y_t[mask], y_p[mask]).items()
            if k not in ("prediction_distribution", "actual_target_summary")
        }
    return out


def main() -> None:
    model_mtime_before = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    report_mtime_before = REPORT_PATH.stat().st_mtime if REPORT_PATH.exists() else None

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    keep = list(dict.fromkeys(feature_cols + [TARGET_COL, "invoice_date", "purchase_count_so_far"]))

    print("Loading identical Phase 4 partitions...", flush=True)
    train = pd.read_parquet(PROC / "train.parquet", columns=[c for c in keep if True])
    # categorical may be missing in columns list if not present — filter
    available = [c for c in keep if c in train.columns or c in ("invoice_date",)]
    # re-read safely with intersection
    train_cols = [c for c in keep if c in pd.read_parquet(PROC / "train.parquet", columns=[]).columns]
    # parquet columns=[] may not work — use previous train.columns
    train_cols = [c for c in keep if c in train.columns]
    # ensure we load all needed from each file
    all_needed = set(train_cols)
    # reload with only existing
    train = pd.read_parquet(PROC / "train.parquet", columns=train_cols)
    val = pd.read_parquet(PROC / "validation.parquet", columns=train_cols)
    test = pd.read_parquet(PROC / "test.parquet", columns=train_cols)

    # Identical Phase 4 filter: positive targets only
    train = train[train[TARGET_COL] > 0].copy()
    val = val[val[TARGET_COL] > 0].copy()
    test = test[test[TARGET_COL] > 0].copy()

    feats = [c for c in feature_cols if c in train.columns]
    missing = [c for c in feature_cols if c not in train.columns]
    print(f"Features used ({len(feats)}): {feats}", flush=True)
    if missing:
        print(f"Missing features skipped: {missing}", flush=True)

    X_train_raw = train[feats]
    y_train = train[TARGET_COL].to_numpy(dtype=np.float64)
    X_val_raw = val[feats]
    y_val = val[TARGET_COL].to_numpy(dtype=np.float64)
    X_test_raw = test[feats]
    y_test = test[TARGET_COL].to_numpy(dtype=np.float64)

    print(
        f"Rows train/val/test: {len(train):,} / {len(val):,} / {len(test):,}",
        flush=True,
    )
    print(
        f"Dates train {pd.to_datetime(train['invoice_date']).min().date()}->"
        f"{pd.to_datetime(train['invoice_date']).max().date()} | "
        f"val {pd.to_datetime(val['invoice_date']).min().date()}->"
        f"{pd.to_datetime(val['invoice_date']).max().date()} | "
        f"test {pd.to_datetime(test['invoice_date']).min().date()}->"
        f"{pd.to_datetime(test['invoice_date']).max().date()}",
        flush=True,
    )

    print("Fitting identical preprocessor once...", flush=True)
    pre = build_preprocessor()
    # ColumnTransformer expects the declared columns; rebuild with available only
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder

    num_cols = [c for c in NUMERIC_FEATURES if c in feats]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feats]
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=50)),
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    X_train = pre.fit_transform(X_train_raw)
    X_val = pre.transform(X_val_raw)
    X_test = pre.transform(X_test_raw)
    # free raw frames
    del X_train_raw, X_val_raw, X_test_raw, pre
    gc.collect()
    print(f"Transformed feature matrix shape: {X_train.shape}", flush=True)

    results = {
        "experiment": "PHASE_17D_XGB_OBJECTIVE_EXPERIMENT",
        "xgboost_version": xgb.__version__,
        "read_only": True,
        "production_model_replaced": False,
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
        },
        "objectives": {},
    }

    for spec in OBJECTIVES:
        name = spec["name"]
        print(f"\nTraining objective={name} ...", flush=True)
        t0 = time.time()
        model = xgb.XGBRegressor(
            objective=spec["objective"],
            **COMMON_XGB,
            **spec["extra"],
        )
        model.fit(X_train, y_train)
        train_sec = round(time.time() - t0, 2)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        results["objectives"][name] = {
            "objective": spec["objective"],
            "extra_params": spec["extra"],
            "training_time_sec": train_sec,
            "validation": _metrics(y_val, val_pred),
            "test": _metrics(y_test, test_pred),
            "test_errors_by_history_depth": _history_depth(test, test_pred),
        }
        print(
            f"  done in {train_sec}s | Val MAE={results['objectives'][name]['validation']['mae']} "
            f"| Test MAE={results['objectives'][name]['test']['mae']}",
            flush=True,
        )
        del model
        gc.collect()

    # ranking by validation MAE then MedAE
    ranking = sorted(
        results["objectives"].items(),
        key=lambda kv: (
            kv[1]["validation"]["mae"] if kv[1]["validation"]["mae"] is not None else 1e9,
            kv[1]["validation"]["medae"] if kv[1]["validation"]["medae"] is not None else 1e9,
        ),
    )
    results["ranking_by_validation_mae"] = [
        {"objective": k, "val_mae": v["validation"]["mae"], "val_medae": v["validation"]["medae"],
         "test_mae": v["test"]["mae"], "test_within_7d": v["test"]["within_7_days_pct"]}
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

    out = PROC / "phase17d_xgb_objective_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out}", flush=True)
    print("Ranking:", results["ranking_by_validation_mae"], flush=True)
    print("Artifact safety:", results["artifact_safety"], flush=True)


if __name__ == "__main__":
    main()
