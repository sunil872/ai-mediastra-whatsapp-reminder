"""Prediction Summary & KPI Reporting Module."""

from __future__ import annotations

from typing import Dict, Any
import pandas as pd
import numpy as np


def generate_prediction_summary_report(
    eligible_df: pd.DataFrame,
    ineligible_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Generate executive operational KPI summary for pharmacy operations."""
    total_eligible = len(eligible_df)
    total_ineligible = len(ineligible_df)
    total_evaluated = total_eligible + total_ineligible

    valid_phone_cnt = 0
    if not eligible_df.empty and "mobile_status" in eligible_df.columns:
        valid_phone_cnt = int((eligible_df["mobile_status"] == "Valid").sum())
    missing_phone_cnt = total_eligible - valid_phone_cnt

    dos_count = 0
    path_a_count = 0
    if not eligible_df.empty and "prediction_source" in eligible_df.columns:
        dos_count = int(eligible_df["prediction_source"].str.contains("Days of Supply", na=False).sum())
        path_a_count = int(eligible_df["prediction_source"].str.contains("Path A", na=False).sum())

    return {
        "customers_evaluated": total_evaluated,
        "eligible_refill_predictions": total_eligible,
        "review_queue_customers": total_ineligible,
        "valid_whatsapp_delivery_ready": valid_phone_cnt,
        "missing_mobile_records": missing_phone_cnt,
        "pack_aware_days_of_supply_predictions": dos_count,
        "path_a_median_predictions": path_a_count,
    }
