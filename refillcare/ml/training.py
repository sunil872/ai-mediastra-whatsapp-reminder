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
import os
import time
from datetime import date, datetime, timedelta
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

# The model learns the patient's deviation from their own median, log(interval / median), instead
# of raw days: one model then fits 30-day and 90-day patients alike. predict_with_bundle inverts it.
TARGET_RAW = "days"
TARGET_LOG_RATIO = "log_ratio_to_personal_median"
TARGET_TRANSFORMS = (TARGET_RAW, TARGET_LOG_RATIO)


def build_training_matrix(
    history_df: pd.DataFrame,
    train_start: date,
    train_end: date,
    min_history: int = PATH_A_MIN_HISTORY,
    max_events: Optional[int] = 150_000,
    seed: int = 42,
    label_horizon_days: Optional[int] = None,
    salt_stats: Optional[Any] = None,
) -> pd.DataFrame:
    """Feature matrix for purchases dated in [train_start, train_end - label_horizon] whose next
    purchase is known by ``train_end``.

    Leakage guard: the *target* (next purchase) must also be on or before ``train_end``, so nothing
    after the cutoff is used.

    Label-bias guard: purchases in the last ``label_horizon_days`` before the cutoff are not used
    as training events. For them only *short* refills are already observed, which taught the model
    that refills come sooner than they do (env ML_TRAIN_LABEL_HORIZON_DAYS, default 90).
    """
    from refillcare.features.context import PopulationStats, VisitIndex, build_context

    horizon = label_horizon_days if label_horizon_days is not None else int(os.getenv("ML_TRAIN_LABEL_HORIZON_DAYS", "90"))
    pe = prepare_purchase_events(history_df)
    pe = pe[pe["date"] <= pd.Timestamp(train_end)]  # hide post-cutoff purchases entirely
    event_end = train_end - timedelta(days=max(0, horizon))
    sel = build_evaluation_events(pe, train_start, event_end, min_prior_purchases=min_history - 1, max_events=max_events, seed=seed)
    visit_index = VisitIndex.from_events(pe)
    stats = salt_stats if salt_stats is not None else PopulationStats.from_events(pe, as_of=train_end)
    rows: List[Dict[str, Any]] = []
    for ev in sel["events"]:
        feats = compute_point_in_time_features_single(
            dates=ev["dates"],
            quantities=ev["quantities"],
            packings=ev["packings"],
            item_name=ev["item_name"],
            context=build_context(visit_index, stats, ev["customer_id"], ev["item_id"], ev["dates"][-1], ev.get("salt"), ev["item_name"]),
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
    target_transform: Optional[str] = None,
    label_horizon_days: Optional[int] = None,
) -> Dict[str, Any]:
    """Train a HistGradientBoosting interval model and register it as a candidate.

    ``target_transform``: ``log_ratio_to_personal_median`` (default, env ML_TARGET_TRANSFORM) or ``days``.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    import joblib

    train_start = train_start or date(2000, 1, 1)
    t0 = time.time()
    transform = (target_transform or os.getenv("ML_TARGET_TRANSFORM", TARGET_LOG_RATIO)).strip()
    if transform not in TARGET_TRANSFORMS:
        raise ValueError(f"Unknown target_transform '{transform}'; use one of {TARGET_TRANSFORMS}.")
    mat = build_training_matrix(history_df, train_start, train_end, max_events=max_events, seed=seed, label_horizon_days=label_horizon_days)
    if len(mat) < 500:
        raise ValueError(f"Only {len(mat)} training events before {train_end}; need >= 500.")
    mat = mat[(mat["target_days"] >= 5) & (mat["target_days"] <= 180)]
    X = mat[POINT_IN_TIME_FEATURE_COLS]
    y = mat["target_days"]
    personal_median = mat["historical_median_interval"].clip(lower=5.0)
    y_fit = np.log(y / personal_median) if transform == TARGET_LOG_RATIO else y
    params = {
        "loss": "absolute_error", "max_iter": 600, "learning_rate": 0.05, "min_samples_leaf": 40,
        "early_stopping": True, "validation_fraction": 0.1, "n_iter_no_change": 25, "random_state": seed,
    }
    params.update(hyperparameters or {})
    model = HistGradientBoostingRegressor(**params)
    model.fit(X, y_fit)
    raw = model.predict(X)
    train_pred = personal_median.to_numpy() * np.exp(raw) if transform == TARGET_LOG_RATIO else raw
    train_metrics = {
        "iterations_used": int(getattr(model, "n_iter_", params["max_iter"])),
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
            "target_transform": transform,
            "label_horizon_days": label_horizon_days if label_horizon_days is not None else int(os.getenv("ML_TRAIN_LABEL_HORIZON_DAYS", "90")),
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
