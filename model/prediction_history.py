"""Prediction History and Outcome Evaluation Module."""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import pandas as pd
from reminder.storage import RefillCareStorage, DEFAULT_DB_PATH
from refillcare.data.monthly_ingestion import evaluate_prediction_outcomes


class PredictionHistoryManager:
    """Manages immutable prediction snapshots and real-world outcome matching."""

    def __init__(self, storage: Optional[RefillCareStorage] = None):
        self.storage = storage or RefillCareStorage(db_path=DEFAULT_DB_PATH)

    def save_snapshots(self, snapshots_df: pd.DataFrame, import_batch_id: Optional[str] = None) -> int:
        """Batch save prediction snapshots into database without overwriting past snapshots."""
        if snapshots_df.empty:
            return 0
        df = snapshots_df.copy()
        if import_batch_id:
            df["import_batch_id"] = import_batch_id
        return self.storage.save_prediction_snapshots(df)

    def get_snapshots(
        self,
        customer_id: Optional[str] = None,
        item_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve stored prediction snapshots."""
        return self.storage.get_prediction_snapshots(customer_id=customer_id, item_id=item_id, status=status)

    def evaluate_with_actuals(self, actual_sales_df: pd.DataFrame) -> Dict[str, Any]:
        """Match actual sales against pending prediction snapshots to compute realization metrics."""
        return evaluate_prediction_outcomes(history_df=actual_sales_df, storage=self.storage)
