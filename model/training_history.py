"""Training history and model evaluation experiment tracker."""

from __future__ import annotations

import json
from datetime import datetime, date
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd

from config.config import PROCESSED_DATA_DIR

EXPERIMENT_LOG_PATH = PROCESSED_DATA_DIR / "training_experiments.json"


class TrainingHistoryTracker:
    """Logs and retrieves model training experiments and validation metrics."""

    @staticmethod
    def log_experiment(
        version_id: str,
        model_type: str,
        cutoff_date: date,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, Any],
        notes: str = "",
    ) -> None:
        """Append training run record to persistent log."""
        record = {
            "version_id": version_id,
            "model_type": model_type,
            "cutoff_date": cutoff_date.isoformat() if isinstance(cutoff_date, date) else str(cutoff_date),
            "hyperparameters": hyperparameters,
            "metrics": metrics,
            "notes": notes,
            "timestamp": datetime.utcnow().isoformat(),
        }

        records = TrainingHistoryTracker.get_all_experiments()
        records.append(record)

        with open(EXPERIMENT_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    @staticmethod
    def get_all_experiments() -> List[Dict[str, Any]]:
        """Retrieve all recorded training runs."""
        if not EXPERIMENT_LOG_PATH.exists():
            return []
        try:
            with open(EXPERIMENT_LOG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
