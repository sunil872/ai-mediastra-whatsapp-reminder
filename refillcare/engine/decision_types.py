"""Dataclasses and constants for Unified Refill Decision Engine & Lifecycle."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional

# Path Constants
PATH_A = "PATH_A"
PATH_B = "PATH_B"
PATH_INELIGIBLE = "INELIGIBLE"

# Stability Constants
STABILITY_HIGH = "HIGH"
STABILITY_MEDIUM_SAFE = "MEDIUM-SAFE"
STABILITY_MEDIUM_RISK = "MEDIUM-RISK"
STABILITY_UNSTABLE = "UNSTABLE"

# Prediction Method Constants
PRED_PATH_A_HISTORICAL_MEDIAN = "PATH_A_PERSONAL_HISTORICAL_MEDIAN"
PRED_PATH_B_DOS_EDS = "PATH_B_DOS_EDS"
PRED_HISTORICAL_MEDIAN = "PATH_A_PERSONAL_HISTORICAL_MEDIAN"
PRED_PATH_B_RECENCY = "PATH_B_DOS_EDS"
PRED_DAYS_OF_SUPPLY = "PATH_B_DOS_EDS"
PRED_NONE = "NONE"
PRED_REVIEW_REQUIRED = "REVIEW_REQUIRED"

# Reminder Stage Offsets (relative to expected_refill_date)
STAGE_MINUS_7 = -7
STAGE_MINUS_3 = -3
STAGE_MINUS_1 = -1
STAGE_DAY_0 = 0
STAGE_PLUS_2 = 2
STAGE_PLUS_5 = 5
STAGE_PLUS_40 = 40

STAGE_OFFSETS = [
    STAGE_MINUS_7,
    STAGE_MINUS_3,
    STAGE_MINUS_1,
    STAGE_DAY_0,
    STAGE_PLUS_2,
    STAGE_PLUS_5,
    STAGE_PLUS_40,
]

# Lifecycle Stage & Recency Constants
CHURN_CUTOFF_DAYS = 75
STATUS_CHURNED_INACTIVE = "CHURNED_INACTIVE"

# Lifecycle Stage Statuses
STATUS_PENDING = "PENDING"
STATUS_APPROVED = "APPROVED"
STATUS_REJECTED = "REJECTED"
STATUS_DUE = "DUE"
STATUS_SENT = "SENT"
STATUS_FAILED = "FAILED"
STATUS_SUPERSEDED = "SUPERSEDED_BY_PURCHASE"
STATUS_CANCELLED = "CANCELLED"


@dataclass
class RefillDecision:
    """Unified single-source-of-truth decision for a customer-item pair."""

    customer_id: str
    customer_name: str
    mobile_no: Optional[str]
    item_id: str
    item_name: str
    customer_item_key: str
    
    # Qualification & Routing
    path: str  # PATH_A | PATH_B | INELIGIBLE
    purchase_count: int
    is_eligible: bool
    
    # Stability
    stability_tier: str  # HIGH | MEDIUM-SAFE | MEDIUM-RISK | UNSTABLE
    cadence_median: Optional[float]
    cadence_norm_mad: Optional[float]
    cadence_drift: Optional[float]
    
    # DOS / Packaging
    dos_days: Optional[float]
    units_purchased: Optional[float]
    
    # Prediction Outputs
    prediction_method: str
    predicted_interval_days: Optional[int]
    last_purchase_date: date
    expected_refill_date: Optional[date]
    decision_reason: str
    
    # Provenance & Lifecycle
    cycle_id: str
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.last_purchase_date:
            d["last_purchase_date"] = self.last_purchase_date.strftime("%Y-%m-%d")
        if self.expected_refill_date:
            d["expected_refill_date"] = self.expected_refill_date.strftime("%Y-%m-%d")
        if self.created_at:
            d["created_at"] = self.created_at.isoformat()
        return d


@dataclass
class ReminderStage:
    """Individual stage within a 6-stage reminder cycle."""

    cycle_id: str
    customer_id: str
    item_id: str
    customer_item_key: str
    stage_offset: int  # -7, -3, -1, 0, 2, 5
    target_send_date: date
    expected_refill_date: date
    status: str = STATUS_PENDING  # PENDING | DUE | SENT | FAILED | SUPERSEDED_BY_PURCHASE
    message_text: Optional[str] = None
    sent_at: Optional[datetime] = None
    provider_msg_id: Optional[str] = None
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.target_send_date:
            d["target_send_date"] = self.target_send_date.strftime("%Y-%m-%d")
        if self.expected_refill_date:
            d["expected_refill_date"] = self.expected_refill_date.strftime("%Y-%m-%d")
        if self.sent_at:
            d["sent_at"] = self.sent_at.isoformat()
        return d


@dataclass
class ReminderCycleState:
    """State of an active or historical reminder cycle for a customer-item pair."""

    cycle_id: str
    customer_item_key: str
    customer_id: str
    item_id: str
    last_purchase_date: date
    expected_refill_date: date
    stages: List[ReminderStage] = field(default_factory=list)
    is_active: bool = True
    superseded_at: Optional[datetime] = None
    superseded_by_purchase_date: Optional[date] = None

    def cancel_pending_stages(self, purchase_date: date) -> int:
        """Cancel all unsent/pending stages when a customer repurchases."""
        cancelled_count = 0
        now = datetime.now()
        self.is_active = False
        self.superseded_at = now
        self.superseded_by_purchase_date = purchase_date
        for st in self.stages:
            if st.status in (STATUS_PENDING, STATUS_DUE):
                st.status = STATUS_SUPERSEDED
                st.failure_reason = f"Customer repurchased on {purchase_date.strftime('%Y-%m-%d')}"
                cancelled_count += 1
        return cancelled_count
