"""Enterprise SQLAlchemy Models for RefillCare & Mediastra.

Includes complete entities for:
- Import Batches & Sales Transactions
- Customers & Medicines
- Model Registry & Training History (.pkl / .joblib serialization lineage)
- Prediction Snapshots & Outcome Realization Tracking
- Reminder Schedules & Delivery Audit Logs
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Date,
    Text,
    ForeignKey,
    Index,
    JSON,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from database.connection import Base


class ImportBatchModel(Base):
    """Tracks every file upload and data ingestion batch."""
    __tablename__ = "import_batches"

    import_batch_id = Column(String(64), primary_key=True, index=True)
    source_filename = Column(String(255), nullable=False)
    upload_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    detected_date_range = Column(String(64), nullable=True)
    source_format = Column(String(32), nullable=True)
    record_count = Column(Integer, default=0, nullable=False)
    records_inserted = Column(Integer, default=0, nullable=False)
    records_skipped = Column(Integer, default=0, nullable=False)
    records_requiring_review = Column(Integer, default=0, nullable=False)
    processing_status = Column(String(32), default="ACTIVE", nullable=False)  # ACTIVE, ROLLED_BACK, FAILED
    checksum_sha256 = Column(String(64), nullable=True, index=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    transactions = relationship("SalesTransactionModel", back_populates="batch", cascade="all, delete-orphan")
    snapshots = relationship("PredictionSnapshotModel", back_populates="batch")


class SalesTransactionModel(Base):
    """Individual pharmacy sales and dispensing records."""
    __tablename__ = "sales_transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    import_batch_id = Column(String(64), ForeignKey("import_batches.import_batch_id"), nullable=False, index=True)
    invoice_number = Column(String(64), nullable=True, index=True)
    invoice_date = Column(DateTime, nullable=False, index=True)
    customer_id = Column(String(64), nullable=False, index=True)
    customer_name = Column(String(255), nullable=True)
    item_id = Column(String(64), nullable=False, index=True)
    item_name = Column(String(255), nullable=True)
    quantity = Column(Float, nullable=False)
    packing = Column(String(64), nullable=True)
    mobile_no = Column(String(32), nullable=True, index=True)
    salt_composition = Column(String(255), nullable=True)
    net_amount = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    batch = relationship("ImportBatchModel", back_populates="transactions")

    __table_args__ = (
        Index("idx_sales_cust_item", "customer_id", "item_id"),
        Index("idx_sales_date_cust", "invoice_date", "customer_id"),
    )


class CustomerModel(Base):
    """Customer profile, phone validation, and consent flags."""
    __tablename__ = "customers"

    customer_id = Column(String(64), primary_key=True, index=True)
    customer_name = Column(String(255), nullable=True)
    mobile_number = Column(String(32), nullable=True, index=True)
    mobile_status = Column(String(32), default="Missing", nullable=False)  # Valid, Missing, Invalid format
    opt_in_whatsapp = Column(Boolean, default=True, nullable=False)
    risk_category = Column(String(32), default="Chronic", nullable=False)  # Chronic, Acute, High-Variance
    total_lifetime_purchases = Column(Integer, default=0, nullable=False)
    first_seen_date = Column(Date, nullable=True)
    last_seen_date = Column(Date, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class MedicineModel(Base):
    """Medicine metadata, packaging units, and dosage benchmarks."""
    __tablename__ = "medicines"

    item_id = Column(String(64), primary_key=True, index=True)
    item_name = Column(String(255), nullable=False, index=True)
    packing = Column(String(64), nullable=True)
    units_per_pack = Column(Integer, default=10, nullable=False)
    salt_composition = Column(String(255), nullable=True)
    default_daily_dose = Column(Float, default=1.0, nullable=False)
    is_chronic_drug = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ModelRegistryModel(Base):
    """ML Model versioning, hyperparameter registry, and artifact lineage."""
    __tablename__ = "model_registry"

    version_id = Column(String(64), primary_key=True, index=True)  # e.g., "v1.0.0", "v1.1.0"
    model_name = Column(String(128), default="RefillCare-Hybrid-PathAB", nullable=False)
    model_type = Column(String(64), default="LightGBM_Heuristic_Hybrid", nullable=False)
    artifact_path = Column(String(512), nullable=False)  # path to .pkl / .joblib bundle
    dataset_cutoff_date = Column(Date, nullable=False)
    training_sample_count = Column(Integer, default=0, nullable=False)
    hyperparameters = Column(JSON, nullable=True)
    metrics_train = Column(JSON, nullable=True)  # MAE, within_3d, within_7d
    metrics_val = Column(JSON, nullable=True)
    is_active_production = Column(Boolean, default=False, nullable=False, index=True)
    created_by = Column(String(64), default="system", nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    training_runs = relationship("TrainingRunModel", back_populates="model_version_rel")


class TrainingRunModel(Base):
    """History of all model training experiments and benchmark results."""
    __tablename__ = "training_runs"

    run_id = Column(String(64), primary_key=True, index=True)
    version_id = Column(String(64), ForeignKey("model_registry.version_id"), nullable=False, index=True)
    status = Column(String(32), default="COMPLETED", nullable=False)  # RUNNING, COMPLETED, FAILED
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    training_duration_seconds = Column(Float, nullable=True)
    mae_days = Column(Float, nullable=True)
    pct_within_3_days = Column(Float, nullable=True)
    pct_within_7_days = Column(Float, nullable=True)
    logs = Column(Text, nullable=True)

    model_version_rel = relationship("ModelRegistryModel", back_populates="training_runs")


class PredictionSnapshotModel(Base):
    """Immutable prediction snapshots generated from specific data cutoffs."""
    __tablename__ = "prediction_snapshots"

    snapshot_id = Column(String(64), primary_key=True, index=True)
    import_batch_id = Column(String(64), ForeignKey("import_batches.import_batch_id"), nullable=True, index=True)
    model_version = Column(String(64), default="v1.0.0", nullable=False, index=True)
    customer_id = Column(String(64), nullable=False, index=True)
    customer_name = Column(String(255), nullable=True)
    item_id = Column(String(64), nullable=False, index=True)
    item_name = Column(String(255), nullable=True)
    phone_number = Column(String(32), nullable=True)
    mobile_status = Column(String(32), default="Missing", nullable=False)
    prediction_date = Column(Date, nullable=False, index=True)
    last_purchase_date = Column(Date, nullable=False)
    estimated_days_of_supply = Column(Float, nullable=False)
    expected_refill_date = Column(Date, nullable=False, index=True)
    reminder_date = Column(Date, nullable=False, index=True)
    prediction_source = Column(String(64), default="Path A (ML)", nullable=False)
    history_quality = Column(String(64), default="high_history", nullable=False)
    pilot_tier = Column(String(64), default="Tier A (Strong Pilot)", nullable=False)
    status = Column(String(32), default="PENDING_EVALUATION", nullable=False, index=True)  # PENDING_EVALUATION, EVALUATED, INVALIDATED
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    batch = relationship("ImportBatchModel", back_populates="snapshots")
    outcome = relationship("PredictionOutcomeModel", back_populates="snapshot", uselist=False, cascade="all, delete-orphan")
    reminders = relationship("ReminderScheduleModel", back_populates="snapshot", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_snap_cust_item_pred", "customer_id", "item_id", "prediction_date"),
        Index("idx_snap_remind_status", "reminder_date", "status"),
    )


class PredictionOutcomeModel(Base):
    """Realized accuracy metrics matching predictions against subsequent purchases."""
    __tablename__ = "prediction_outcomes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_id = Column(String(64), ForeignKey("prediction_snapshots.snapshot_id"), unique=True, nullable=False, index=True)
    actual_next_purchase_date = Column(Date, nullable=True)
    outcome_error_days = Column(Float, nullable=True)
    is_within_3_days = Column(Boolean, default=False, nullable=False)
    is_within_7_days = Column(Boolean, default=False, nullable=False)
    error_category = Column(String(32), nullable=True)  # ON_TIME, EARLY, LATE, NO_PURCHASE
    evaluated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    snapshot = relationship("PredictionSnapshotModel", back_populates="outcome")


class ReminderScheduleModel(Base):
    """Daily reminder delivery schedules and operational dispatch states."""
    __tablename__ = "reminder_schedules"

    reminder_id = Column(String(64), primary_key=True, index=True)
    snapshot_id = Column(String(64), ForeignKey("prediction_snapshots.snapshot_id"), nullable=True, index=True)
    customer_id = Column(String(64), nullable=False, index=True)
    customer_name = Column(String(255), nullable=True)
    phone_number = Column(String(32), nullable=False, index=True)
    item_id = Column(String(64), nullable=False)
    item_name = Column(String(255), nullable=False)
    reminder_date = Column(Date, nullable=False, index=True)
    expected_refill_date = Column(Date, nullable=False)
    reminder_stage = Column(String(32), default="3_days_before", nullable=False)
    channel = Column(String(32), default="WHATSAPP", nullable=False)  # WHATSAPP, MANUAL_CALL
    delivery_status = Column(String(32), default="SCHEDULED", nullable=False, index=True)  # SCHEDULED, SENT, DELIVERED, FAILED, SUPPRESSED
    xinno_message_id = Column(String(128), nullable=True, index=True)
    message_content = Column(Text, nullable=True)
    sent_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    snapshot = relationship("PredictionSnapshotModel", back_populates="reminders")


class WhatsAppDeliveryLogModel(Base):
    """Audit trail of all WhatsApp messages sent via Xinno gateway."""
    __tablename__ = "whatsapp_delivery_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    reminder_id = Column(String(64), nullable=True, index=True)
    phone_number = Column(String(32), nullable=False, index=True)
    customer_name = Column(String(255), nullable=True)
    template_name = Column(String(64), nullable=False)
    xinno_message_id = Column(String(128), nullable=True, index=True)
    is_dry_run = Column(Boolean, default=True, nullable=False)
    status = Column(String(32), default="PENDING", nullable=False)  # SUCCESS, FAILED, DRY_RUN_SUCCESS
    http_status_code = Column(Integer, nullable=True)
    response_payload = Column(Text, nullable=True)
    error_detail = Column(Text, nullable=True)
    dispatched_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class RefillDecisionModel(Base):
    """Unified single-source-of-truth RefillDecision entity."""
    __tablename__ = "refill_decisions"

    decision_id = Column(String(64), primary_key=True, index=True)
    cycle_id = Column(String(64), nullable=False, index=True)
    customer_id = Column(String(64), nullable=False, index=True)
    customer_name = Column(String(255), nullable=True)
    mobile_no = Column(String(32), nullable=True, index=True)
    item_id = Column(String(64), nullable=False, index=True)
    item_name = Column(String(255), nullable=True)
    customer_item_key = Column(String(128), nullable=False, index=True)

    path = Column(String(32), nullable=False)  # PATH_A, PATH_B, INELIGIBLE
    purchase_count = Column(Integer, default=0, nullable=False)
    is_eligible = Column(Boolean, default=False, nullable=False, index=True)

    stability_tier = Column(String(32), default="UNSTABLE", nullable=False, index=True)
    cadence_median = Column(Float, nullable=True)
    cadence_norm_mad = Column(Float, nullable=True)
    cadence_drift = Column(Float, nullable=True)
    dos_days = Column(Float, nullable=True)
    units_purchased = Column(Float, nullable=True)

    prediction_method = Column(String(64), default="NONE", nullable=False)
    predicted_interval_days = Column(Integer, nullable=True)
    last_purchase_date = Column(Date, nullable=False, index=True)
    expected_refill_date = Column(Date, nullable=True, index=True)
    decision_reason = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    cycles = relationship("ReminderCycleModel", back_populates="decision")


class ReminderCycleModel(Base):
    """Lifecycle state of an active or historical reminder cycle for a customer-item pair."""
    __tablename__ = "reminder_cycles"

    cycle_id = Column(String(64), primary_key=True, index=True)
    customer_item_key = Column(String(128), nullable=False, index=True)
    customer_id = Column(String(64), nullable=False, index=True)
    item_id = Column(String(64), nullable=False, index=True)
    decision_id = Column(String(64), ForeignKey("refill_decisions.decision_id"), nullable=True, index=True)

    last_purchase_date = Column(Date, nullable=False, index=True)
    expected_refill_date = Column(Date, nullable=False, index=True)
    is_active = Column(Boolean, default=True, nullable=False, index=True)

    superseded_at = Column(DateTime, nullable=True)
    superseded_by_purchase_date = Column(Date, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    decision = relationship("RefillDecisionModel", back_populates="cycles")
    stages = relationship("ReminderStageModel", back_populates="cycle", cascade="all, delete-orphan")


class ReminderStageModel(Base):
    """Individual stage within a 6-stage reminder cycle (-7d, -3d, -1d, 0d, +2d, +5d)."""
    __tablename__ = "reminder_stages"

    reminder_id = Column(String(64), primary_key=True, index=True)
    cycle_id = Column(String(64), ForeignKey("reminder_cycles.cycle_id"), nullable=False, index=True)
    customer_id = Column(String(64), nullable=False, index=True)
    item_id = Column(String(64), nullable=False, index=True)
    customer_item_key = Column(String(128), nullable=False, index=True)

    stage_offset = Column(Integer, nullable=False)  # -7, -3, -1, 0, 2, 5
    target_send_date = Column(Date, nullable=False, index=True)
    expected_refill_date = Column(Date, nullable=False, index=True)

    status = Column(String(32), default="PENDING", nullable=False, index=True)  # PENDING, APPROVED, REJECTED, SENT, FAILED, SUPERSEDED_BY_PURCHASE, CANCELLED
    message_text = Column(Text, nullable=True)

    sent_at = Column(DateTime, nullable=True)
    provider_msg_id = Column(String(128), nullable=True, index=True)
    failure_reason = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    cycle = relationship("ReminderCycleModel", back_populates="stages")

    __table_args__ = (
        UniqueConstraint("cycle_id", "stage_offset", name="uq_cycle_stage_offset"),
        Index("idx_stage_target_status", "target_send_date", "status"),
        Index("idx_stage_cust_item", "customer_id", "item_id"),
    )

