"""Model Serialization and Versioning Module supporting .pkl bundles."""

from __future__ import annotations

import pickle
import joblib
from pathlib import Path
from datetime import datetime, date
from typing import Dict, Any, Optional, Union

from config.config import MODELS_DIR, DEFAULT_PKL_MODEL_PATH, DEFAULT_JOBLIB_MODEL_PATH


def save_model_bundle_pkl(
    model: Any,
    version_id: str,
    cutoff_date: date,
    hyperparameters: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    encoders: Optional[Any] = None,
    target_path: Optional[Path] = None,
) -> Path:
    """Serialize model bundle to versioned .pkl file with metadata and encoder lineage.

    Args:
        model: Trained model artifact (LightGBM, XGBoost, etc.).
        version_id: Semantic version identifier (e.g. 'v1.0.0').
        cutoff_date: Strict historical training cutoff date.
        hyperparameters: Dictionary of training hyperparameters.
        metrics: Dictionary of validation benchmark metrics (MAE, ±3d, ±7d).
        encoders: Fitted feature encoders.
        target_path: Optional output file path.

    Returns:
        Path: Path to saved .pkl model artifact.
    """
    if target_path is None:
        target_path = MODELS_DIR / f"refill_model_{version_id}.pkl"

    bundle = {
        "version_id": version_id,
        "model": model,
        "model_type": type(model).__name__,
        "dataset_cutoff_date": cutoff_date.isoformat() if isinstance(cutoff_date, date) else str(cutoff_date),
        "hyperparameters": hyperparameters or {},
        "metrics": metrics or {},
        "encoders": encoders,
        "created_at": datetime.utcnow().isoformat(),
    }

    with open(target_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Also update default symlink / pointer
    with open(DEFAULT_PKL_MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    return target_path


def load_model_bundle(
    model_path: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """Load model bundle from .pkl or fallback to .joblib for compatibility.

    Args:
        model_path: Optional path to .pkl or .joblib file.

    Returns:
        Optional[Dict[str, Any]]: Unserialized model bundle dictionary.
    """
    candidates = []
    if model_path:
        candidates.append(Path(model_path))
    candidates.extend([DEFAULT_PKL_MODEL_PATH, DEFAULT_JOBLIB_MODEL_PATH])

    for p in candidates:
        if p.exists():
            try:
                if str(p).endswith(".pkl"):
                    with open(p, "rb") as f:
                        return pickle.load(f)
                else:
                    return joblib.load(p)
            except Exception as e:
                continue

    return None
