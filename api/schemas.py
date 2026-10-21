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
    online_sync: Dict[str, Any] = Field(default_factory=dict)
    ml_data_freshness: Optional[Dict[str, Any]] = None
    ml_retrain_queued: bool = False
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
    refill_function: Optional[str] = "DUE_REFILL"
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
    dataset_cutoff_date: Optional[date] = None
    training_sample_count: int = 0
    hyperparameters: Optional[Dict[str, Any]] = None
    metrics_train: Optional[Dict[str, Any]] = Field(None, description="In-sample training fit (not a performance claim)")
    metrics_val: Optional[Dict[str, Any]] = Field(None, description="Latest measured out-of-time evaluation summary")
    is_active_production: bool
    created_at: datetime
    notes: Optional[str] = None
    status: Optional[str] = None
    is_validated: bool = False
    latest_evaluation_run_id: Optional[str] = None


class TrainModelRequest(BaseModel):
    version_id: str = Field(..., description="Semantic version string, e.g. v1.1.0")
    cutoff_date: Optional[date] = None
    notes: Optional[str] = None


# ------------------------------------------------------------------------------
# WhatsApp Schemas
# ------------------------------------------------------------------------------
class WhatsAppBatchDispatchRequest(BaseModel):
    reminder_date: Optional[date] = None
    target_month: Optional[str] = Field(None, description="YYYY-MM (alternative to reminder_date)")
    tier_filter: Optional[str] = Field("ALL", description="ALL, DUE, FOLLOWUP, LAPSED")
    dry_run: bool = True
    batch_size: int = Field(50, ge=1, le=500)


class WhatsAppDispatchResponse(BaseModel):
    total_eligible: int
    total_candidates: Optional[int] = None
    valid_candidates: Optional[int] = None
    dispatched_count: int
    success_count: int
    failed_count: int
    is_dry_run: bool
    skipped_already_sent: Optional[int] = None
    requires_approval: Optional[bool] = None
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
    model_is_validated: bool = False
    measured_backtest_within_7_days_pct: Optional[float] = Field(None, description="From the latest evaluation run of the production model; null if never evaluated")
    measured_backtest_mae_days: Optional[float] = None
    measured_backtest_run_id: Optional[str] = None
    live_within_7_days_pct: Optional[float] = Field(None, description="Realised accuracy of past predictions (refill_decision_outcomes)")
    live_resolved_predictions: Optional[int] = None
    channel_summary: Optional[Dict[str, Any]] = None
    target_prediction_month: Optional[str] = None
    target_month_reminders_count: Optional[int] = None
    monthly_stage_distribution: Optional[Dict[str, int]] = None
    adherence_rate_pct: Optional[float] = Field(None, description="Measured refill rate (resolved / (resolved + no refill)); null until outcomes exist")
    chronic_medicines_count: Optional[int] = None
    out_of_stock_or_critical: Optional[int] = None


class DosageRegimenItem(BaseModel):
    regimen: str
    daily_rate: float
    prevalence_pct: float
    sample_count: int
    description: str


class ArchetypeItem(BaseModel):
    archetype: str
    description: str
    example: str


class BenchmarkComparisonItem(BaseModel):
    method: str
    within_7_days_pct: float
    mae_days: float
    status: str


class ConsensusRegimenAnalyticsSummary(BaseModel):
    engine_version: str
    total_transitions_trained: int = 0
    regimens: List[DosageRegimenItem]
    archetypes: List[ArchetypeItem]
    benchmarks: List[BenchmarkComparisonItem]


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
    archetype: Optional[str] = "Consensus"
    dosage_regimen: Optional[str] = "1.0/d (OD)"
    model_version: Optional[str] = None
    prediction_source: Optional[str] = None
    raw_ml_prediction_days: Optional[float] = None


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
    refill_function: Optional[str] = "DUE_REFILL"
    prediction_method: str
    historical_median_days: Optional[float]
    predicted_interval_days: Optional[int]
    decision_reason: str
    message_text: Optional[str]
    archetype: Optional[str] = "Consensus"
    dosage_regimen: Optional[str] = "1.0/d (OD)"


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


class TransactionChannelSummary(BaseModel):
    total_transactions: int
    customer_sales: int
    b2b_inter_store: int
    unknown: int
    excluded_from_refillcare: int
    refillcare_eligible: int
    raw_total_transactions: Optional[int] = None
    cleaned_transactions: Optional[int] = None
    purged_duplicates: Optional[int] = None


# ------------------------------------------------------------------------------
# Med-Sync & Quantile Uncertainty Schemas
# ------------------------------------------------------------------------------
class SyncedMedicationItemSchema(BaseModel):
    item_id: str
    item_name: str
    last_purchase_date: str
    expected_refill_date: str
    predicted_interval_days: Optional[int] = None
    stability_tier: str
    dos_days: Optional[float] = None
    quantile_p10_date: Optional[str] = None
    quantile_p90_date: Optional[str] = None
    is_anchor: bool = False


class MedSyncBundleSchema(BaseModel):
    bundle_id: str
    customer_id: str
    customer_name: str
    mobile_no: Optional[str] = None
    raw_mobile_no: Optional[str] = None
    anchor_item_id: str
    anchor_item_name: str
    anchor_refill_date: str
    window_start_date: str
    window_end_date: str
    lifecycle_tier: Optional[str] = "DUE"
    refill_function: Optional[str] = "DUE_REFILL"
    synced_items: List[SyncedMedicationItemSchema] = []
    total_items_count: int
    message_reduction_count: int
    bundled_message_text: str
    earliest_p10_date: Optional[str] = None
    latest_p90_date: Optional[str] = None
    status: str = "PENDING"
    created_at: Optional[str] = None


class MedSyncImpactSummarySchema(BaseModel):
    total_patients_analyzed: int
    total_prescriptions_synced: int
    total_dispatches_generated: int
    multi_item_bundles_count: int
    single_item_bundles_count: int
    multi_item_bundle_rate_pct: float
    individual_messages_saved: int
    message_reduction_rate_pct: float
    max_items_in_single_bundle: int
    avg_items_per_bundle: float


class MedSyncBundlesResponse(BaseModel):
    sync_window_days: int
    impact_summary: MedSyncImpactSummarySchema
    bundles: List[MedSyncBundleSchema]


# ------------------------------------------------------------------------------
# Medication Classification Rules & Live Pharmacist Catalog Schemas
# ------------------------------------------------------------------------------
class MedicationRuleItem(BaseModel):
    id: Optional[Union[int, str]] = None
    pattern: str
    rule_type: str = Field(..., description="CHRONIC or EXCLUSION")
    category: str = Field(..., description="Clinical therapy or exclusion category")
    brand_name: Optional[str] = None
    notes: Optional[str] = None
    created_by: Optional[str] = "Pharmacist"
    is_active: bool = True
    created_at: Optional[Union[datetime, str]] = None


class AddMedicationRuleRequest(BaseModel):
    pattern: str = Field(..., description="Brand name, molecule, salt or keyword")
    rule_type: str = Field(..., description="CHRONIC or EXCLUSION")
    category: str = Field(..., description="Clinical therapy or exclusion class")
    brand_name: Optional[str] = None
    notes: Optional[str] = None
    created_by: Optional[str] = "Pharmacist"


class MedicationRuleListResponse(BaseModel):
    total_rules: int
    rules: List[MedicationRuleItem]
    source: str = "Supabase PostgreSQL"


class ClassifyMedicationRequest(BaseModel):
    item_name: str
    packing: Optional[str] = None


class ClassifyMedicationResponse(BaseModel):
    item_name: str
    is_chronic_eligible: bool
    category: str
    matched_rule: Optional[str] = None
    exclusion_reason: Optional[str] = None
    source: Optional[str] = "Supabase / Classifier Engine"


# ------------------------------------------------------------------------------
# Clinical Review Queue Schemas
# ------------------------------------------------------------------------------
class ClinicalReviewItem(BaseModel):
    customer_id: str
    customer_name: str
    mobile_number: str
    mobile_status: str
    medication: str
    last_purchase_date: str
    expected_refill_date: str
    review_category: str
    review_reason: str


class ClinicalReviewSummaryResponse(BaseModel):
    target_month: str
    total_in_review_queue: int
    missing_mobile_numbers: int
    stability_variance_risk: int
    cold_start_single_purchase: int
    available_months: List[str]


# ------------------------------------------------------------------------------
# WhatsApp Gateway Schemas
# ------------------------------------------------------------------------------
class WhatsAppCandidateItem(BaseModel):
    reminder_id: str
    customer_name: str
    phone_number: str
    mobile_display: str
    mobile_status: str
    medication: str
    expected_refill_date: str
    target_send_date: str
    stage_offset: int
    tier: str
    refill_function: str
    status: str
    items_count: int = 1


class WhatsAppCandidatesResponse(BaseModel):
    target_date: Optional[str] = None
    target_month: Optional[str] = None
    target_label: str
    total_scheduled: int
    delivery_ready: int
    missing_mobile: int
    candidates: List[WhatsAppCandidateItem]


class WhatsAppPreviewRequest(BaseModel):
    customer_name: str
    medications: Union[str, List[str]]
    expected_refill_date: str
    tier: Optional[str] = "DUE"
    stage_offset: Optional[int] = 0


class WhatsAppPreviewResponse(BaseModel):
    tier: str
    template_name: str
    customer_name: str
    dynamic_body_text: str
    parameters: List[Any]
    visual_preview: str


class WhatsAppSingleSendRequest(BaseModel):
    phone_number: str
    customer_name: str
    medications: Union[str, List[str]]
    expected_refill_date: str
    tier: Optional[str] = "DUE"
    stage_offset: Optional[int] = 0
    dry_run: bool = True


class WhatsAppSingleSendResponse(BaseModel):
    success: bool
    message: str
    status_code: int
    response_payload: Optional[Dict[str, Any]] = None


class CustomerOptInOutResponse(BaseModel):
    customer_id: str
    opt_in_whatsapp: bool
    status: str
    message: str


class WhatsAppWebhookResponse(BaseModel):
    status: str
    action: str
    message: str
    customer_id: Optional[str] = None
    phone_number: Optional[str] = None

