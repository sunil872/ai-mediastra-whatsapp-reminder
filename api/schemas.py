"""Pydantic schemas for RefillCare API."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field


# ------------------------------------------------------------------------------
# Sales Schemas
# ------------------------------------------------------------------------------
class DateDiagnosticsSchema(BaseModel):
    source_format: str
    date_range_formatted: str
    min_date_formatted: str
    max_date_formatted: str
    sample_dates_formatted: List[str]
    valid_count: int
    invalid_count: int
    is_ambiguous: bool
    warning: Optional[str] = None


class SalesPreviewResponse(BaseModel):
    source_filename: str
    file_date_range: str
    records_received: int
    valid_records: int
    records_requiring_review: int
    new_customers: int
    existing_customers: int
    unique_customers: int
    medicines_count: int
    missing_customer_info: int
    missing_mobile_numbers: int
    duplicate_records: int
    invalid_quantities: int
    date_diagnostics: DateDiagnosticsSchema
    preview_records: List[Dict[str, Any]]
    can_process: bool


class SalesIngestionResponse(BaseModel):
    status: str
    import_batch_id: str
    source_filename: str
    records_inserted: int
    records_skipped: int
    customers_updated: int
    new_customers: int
    new_medicines: int
    latest_sales_date: str
    date_range: str
    records_requiring_review: int
    message: str


class ImportBatchResponse(BaseModel):
    import_batch_id: str
    source_filename: str
    upload_timestamp: Union[datetime, str]
    detected_date_range: Optional[str] = "N/A"
    source_format: Optional[str] = "Auto"
    record_count: int = 0
    records_inserted: int = 0
    records_skipped: int = 0
    records_requiring_review: int = 0
    processing_status: str = "ACTIVE"
    is_active: Optional[bool] = True


class RollbackResponse(BaseModel):
    status: str
    rolled_back_batch_id: str
    records_removed: int
    restored_latest_sales_date: str
    invalidated_snapshots: int
    message: str


# ------------------------------------------------------------------------------
# Prediction Schemas
# ------------------------------------------------------------------------------
class GeneratePredictionsRequest(BaseModel):
    prediction_date: Optional[date] = None
    force_recalculate: bool = False


class PredictionGenerationSummary(BaseModel):
    status: str
    prediction_date: str
    customers_evaluated: int
    predictions_generated: int
    customers_requiring_review: int
    customers_not_eligible: int
    missing_mobile_predictions: int
    message: str


class PredictionSnapshotItem(BaseModel):
    snapshot_id: str
    customer_id: str
    customer_name: Optional[str] = "Patient"
    item_id: str
    item_name: Optional[str] = "Medication"
    phone_number: Optional[str] = ""
    mobile_status: Optional[str] = "Missing"
    last_purchase_date: Optional[Union[date, str]] = None
    estimated_days_of_supply: Optional[float] = 30.0
    expected_refill_date: Optional[Union[date, str]] = None
    reminder_date: Optional[Union[date, str]] = None
    prediction_source: Optional[str] = "Path A (ML)"
    pilot_tier: Optional[str] = "Tier A (Strong Pilot)"
    status: Optional[str] = "pending"
    actual_next_purchase_date: Optional[Union[date, str]] = None
    outcome_error_days: Optional[float] = None


class OutcomeEvaluationSummary(BaseModel):
    predictions_evaluated: int
    within_3_days_count: int
    within_3_days_pct: float
    within_7_days_count: int
    within_7_days_pct: float
    mean_absolute_error: float
    median_absolute_error: float
    early_refills_count: int
    late_refills_count: int
    on_time_count: int


# ------------------------------------------------------------------------------
# Reminder Schemas
# ------------------------------------------------------------------------------
class ReminderItem(BaseModel):
    reminder_id: str
    customer_id: str
    customer_name: Optional[str]
    phone_number: str
    mobile_status: str
    item_id: str
    item_name: str
    last_purchase_date: str
    estimated_days_of_supply: float
    expected_refill_date: str
    reminder_date: str
    reminder_stage: str
    delivery_status: str


class UpdateReminderStatusRequest(BaseModel):
    delivery_status: str = Field(..., description="SCHEDULED, SENT, DELIVERED, FAILED, SUPPRESSED, CALLED")
    note: Optional[str] = None


# ------------------------------------------------------------------------------
# Model Registry Schemas
# ------------------------------------------------------------------------------
class ModelVersionItem(BaseModel):
    version_id: str
    model_name: str
    model_type: str
    artifact_path: str
    dataset_cutoff_date: date
    training_sample_count: int
    hyperparameters: Optional[Dict[str, Any]]
    metrics_train: Optional[Dict[str, Any]]
    metrics_val: Optional[Dict[str, Any]]
    is_active_production: bool
    created_at: datetime
    notes: Optional[str]


class TrainModelRequest(BaseModel):
    version_id: str = Field(..., description="Semantic version string, e.g. v1.1.0")
    cutoff_date: Optional[date] = None
    notes: Optional[str] = None


# ------------------------------------------------------------------------------
# WhatsApp Schemas
# ------------------------------------------------------------------------------
class WhatsAppBatchDispatchRequest(BaseModel):
    reminder_date: date
    dry_run: bool = True
    batch_size: int = 50


class WhatsAppDispatchResponse(BaseModel):
    total_eligible: int
    dispatched_count: int
    success_count: int
    failed_count: int
    is_dry_run: bool
    delivery_summary: List[Dict[str, Any]]


# ------------------------------------------------------------------------------
# Analytics & Operations KPI Schemas
# ------------------------------------------------------------------------------
class OperationsKPISummary(BaseModel):
    total_customers_monitored: int
    total_sales_transactions: int
    latest_sales_date: str
    sales_coverage_date_range: str
    upcoming_reminders_count: int
    reminders_due_today: int
    customers_requiring_review: int
    active_model_version: str
    active_batch_id: Optional[str]


# ------------------------------------------------------------------------------
# V1 Refill Decision & Pharmacist Review Queue Schemas
# ------------------------------------------------------------------------------
class RefillDecisionItem(BaseModel):
    decision_id: str
    cycle_id: str
    customer_id: str
    customer_name: Optional[str]
    mobile_no: Optional[str]
    item_id: str
    item_name: Optional[str]
    customer_item_key: str
    path: str
    purchase_count: int
    is_eligible: bool
    stability_tier: str
    cadence_median: Optional[float]
    prediction_method: str
    predicted_interval_days: Optional[int]
    last_purchase_date: date
    expected_refill_date: Optional[date]
    decision_reason: Optional[str]
    created_at: datetime


class ReminderQueueItem(BaseModel):
    reminder_id: str
    cycle_id: str
    customer_item_key: str
    customer_id: str
    customer_name: str
    phone_number: Optional[str]
    masked_phone: str
    item_id: str
    item_name: str
    stage_offset: int
    target_send_date: str
    expected_refill_date: str
    status: str
    path: str
    stability_tier: str
    prediction_method: str
    historical_median_days: Optional[float]
    predicted_interval_days: Optional[int]
    decision_reason: str
    message_text: Optional[str]


class ReviewActionRequest(BaseModel):
    reason: Optional[str] = "Pharmacist decision"
    is_dry_run: bool = True


class ReviewActionResponse(BaseModel):
    reminder_id: str
    status: str
    success: bool
    message: Optional[str] = None
    is_dry_run: Optional[bool] = None
    provider_msg_id: Optional[str] = None


class CustomerRefillHistoryResponse(BaseModel):
    customer_id: str
    customer_name: Optional[str]
    total_cycles: int
    cycles: List[Dict[str, Any]]

