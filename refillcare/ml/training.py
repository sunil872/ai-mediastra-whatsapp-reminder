"""Train refill-interval models on the SAME point-in-time features used at serving time.

The legacy ``scripts/train_human_ml_model.py`` approximated features (constant purchase_count,
typical units = latest units, ...) differently from ``compute_point_in_time_features_single``
used by the engine - a train/serve skew. Here a single event builder + feature function is
shared by training, evaluation and serving.

Models are written to the models directory and registered as *candidates* with real
training metadata. They reach production only through the promotion gate.
"""
from __future__ import annotations

import hashlib
import logging
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from refillcare.features.point_in_time import POINT_IN_TIME_FEATURE_COLS, compute_point_in_time_features_single
from refillcare.ml import registry as reg
from refillcare.ml.evaluation import build_evaluation_events, prepare_purchase_events

logger = logging.getLogger("refillcare.ml.training")

PATH_A_MIN_HISTORY = 6  # Path A uses the model only for customer-items with >= 6 purchases


def build_training_matrix(
    history_df: pd.DataFrame,
    train_start: date,
    train_end: date,
    min_history: int = PATH_A_MIN_HISTORY,
    max_events: Optional[int] = 150_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Feature matrix for purchases dated in [train_start, train_end] whose next purchase is known.

    Leakage guard: the *target* (next purchase) must also be on or before ``train_end``, so nothing
    after the cutoff is used.
    """
    pe = prepare_purchase_events(history_df)
    pe = pe[pe["date"] <= pd.Timestamp(train_end)]  # hide post-cutoff purchases entirely
    sel = build_evaluation_events(pe, train_start, train_end, min_prior_purchases=min_history - 1, max_events=max_events, seed=seed)
    rows: List[Dict[str, Any]] = []
    for ev in sel["events"]:
        feats = compute_point_in_time_features_single(
            dates=ev["dates"],
            quantities=ev["quantities"],
            packings=ev["packings"],
            item_name=ev["item_name"],
        )
        row = {c: float(feats.get(c) or 0.0) if feats.get(c) is not None else np.nan for c in POINT_IN_TIME_FEATURE_COLS}
        row["target_days"] = ev["actual_days"]
        row["event_date"] = ev["dates"][-1]
        rows.append(row)
    return pd.DataFrame(rows)


def _dataset_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256(pd.util.hash_pandas_object(df.fillna(-1), index=False).values.tobytes()).hexdigest()
    return h[:16]


def train_interval_model(
    db: Session,
    history_df: pd.DataFrame,
    train_end: date,
    train_start: Optional[date] = None,
    version_id: Optional[str] = None,
    max_events: Optional[int] = 150_000,
    hyperparameters: Optional[Dict[str, Any]] = None,
    created_by: str = "system",
    seed: int = 42,
) -> Dict[str, Any]:
    """Train a HistGradientBoosting interval model and register it as a candidate."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    import joblib

    train_start = train_start or date(2000, 1, 1)
    t0 = time.time()
    mat = build_training_matrix(history_df, train_start, train_end, max_events=max_events, seed=seed)
    if len(mat) < 500:
        raise ValueError(f"Only {len(mat)} training events before {train_end}; need >= 500.")
    mat = mat[(mat["target_days"] >= 5) & (mat["target_days"] <= 180)]
    X = mat[POINT_IN_TIME_FEATURE_COLS]
    y = mat["target_days"]
    params = {"loss": "absolute_error", "max_iter": 300, "learning_rate": 0.05, "min_samples_leaf": 40, "random_state": seed}
    params.update(hyperparameters or {})
    model = HistGradientBoostingRegressor(**params)
    model.fit(X, y)
    train_pred = model.predict(X)
    train_metrics = {
        "train_events": int(len(mat)),
        "train_mae_days": round(float(np.abs(train_pred - y).mean()), 2),
        "train_within_7_days_pct": round(float((np.abs(train_pred - y) <= 7).mean() * 100), 2),
        "note": "In-sample training fit only - NOT a performance claim. Use an evaluation run.",
    }

    version_id = version_id or f"hgb-{train_end.strftime('%Y%m%d')}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    models_dir = reg.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    rel = f"{version_id}.joblib"
    bundle = {
        "model": model,
        "feature_cols": list(POINT_IN_TIME_FEATURE_COLS),
        "metadata": {
            "model_type": "HistGradientBoostingRegressor",
            "version_id": version_id,
            "train_start": train_start.isoformat(),
            "train_cutoff": train_end.isoformat(),
            "train_samples": int(len(mat)),
            "postprocess": "safety_only",
            "feature_source": "refillcare.features.point_in_time.compute_point_in_time_features_single",
            "trained_at": datetime.utcnow().isoformat(),
            **{k: v for k, v in train_metrics.items() if k != "note"},
        },
    }
    joblib.dump(bundle, models_dir / rel)
    v = reg.register_version(
        db,
        version_id=version_id,
        model_family=reg.FAMILY_PATH_A_INTERVAL,
        model_type=reg.TYPE_HIST_GB,
        artifact_uri=rel,
        feature_cols=list(POINT_IN_TIME_FEATURE_COLS),
        train_window_start=train_start,
        train_window_end=train_end,
        training_sample_count=int(len(mat)),
        dataset_hash=_dataset_hash(mat),
        hyperparameters=params,
        train_metrics=train_metrics,
        notes="Trained on serving-identical point-in-time features; safety-only post-processing.",
        created_by=created_by,
    )
    logger.info("Trained %s on %d events in %.1fs", version_id, len(mat), time.time() - t0)
    return {"version": reg.to_dict(v), "artifact_path": str(models_dir / rel), "seconds": round(time.time() - t0, 1)}
