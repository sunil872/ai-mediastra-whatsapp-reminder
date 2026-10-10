"""Backward compatibility wrapper for refillcare.data.sales_ingestion.

All ingestion functionality is dynamically provided in `sales_ingestion.py`.
This module preserves compatibility for any existing code.
"""

from refillcare.data.sales_ingestion import *  # noqa: F401, F403
from refillcare.data.sales_ingestion import (
    normalize_sales_dataframe,
    validate_monthly_sales_data,
    determine_mobile_status,
    format_display_phone_10digits,
    process_monthly_sales_data,
    rollback_monthly_sales_import,
    generate_updated_predictions,
    ingest_monthly_sales_pipeline,
    evaluate_prediction_outcomes,
)

__all__ = [
    "normalize_sales_dataframe",
    "validate_monthly_sales_data",
    "determine_mobile_status",
    "format_display_phone_10digits",
    "process_monthly_sales_data",
    "rollback_monthly_sales_import",
    "generate_updated_predictions",
    "ingest_monthly_sales_pipeline",
    "evaluate_prediction_outcomes",
]
