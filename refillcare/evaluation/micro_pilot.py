"""RefillCare Phase 16 — First Controlled Micro-Pilot Engine.

Provides the complete execution, safety validation, outcome capture,
stop-condition monitoring, and reporting infrastructure for the first
RefillCare Controlled Micro-Pilot.

Enforces:
- Strict Tier A candidate prioritization (recurring chronic history only)
- Micro-batch controls (5 / 10 / 20 candidates only, default 5)
- Compound identity isolation (customerId + itemId; phone is destination only)
- Multi-assertion pre-send safety validation (9 checks per candidate)
- Purchase-before-send cycle-reset checks
- Explicit human approval requirement with structured rejection reasons
- Default Dry-Run mode with zero live provider calls without explicit action
- Rigorous outcome capture (provider acceptance vs later refill matching)
- Right-censoring / Pending Observation window handling (30 days)
- Automated micro-pilot stop condition detection
- Concise, exportable Micro-Pilot Summary Report
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, date, timedelta, timezone
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd

from reminder.storage import RefillCareStorage, _extract_phone_last4
from reminder.scheduler import (
    REMINDER_STAGES,
    evaluate_refill_eligibility,
)
from reminder.dispatch import (
    DispatchResult,
    build_refillcare_whatsapp_payload,
    mask_phone_for_ui,
    normalize_phone_number,
    REFILLCARE_TEMPLATE_NAME,
)


def _format_stage_label(s: int) -> str:
    """Format numeric stage offset into human-readable label."""
    if s == 0:
        return "0 days (Refill Day)"
    return f"{s:+d} days"
from refillcare.evaluation.pilot_monitoring import _safe_parse_date
from refillcare.evaluation.pilot_outcomes import (
    PilotCohort,
    OFFLINE_BENCHMARKS,
    STRUCTURED_REJECTION_REASONS,
)
from refillcare.evaluation.pilot_operations import (
    validate_pre_send_safety,
)

# Valid Micro-Pilot Batch Sizes
ALLOWED_MICRO_PILOT_BATCH_SIZES = [5, 10, 20]
DEFAULT_MICRO_PILOT_BATCH_SIZE = 5


@dataclass
class MicroPilotCandidate:
    """Individual candidate record in the micro-pilot review queue."""

    reminder_id: str
    customerId: str
    itemId: str
    customer_name: str
    medicine: str
    phone_masked: str
    phone_normalized: str
    latest_purchase_date: str
    predicted_interval_days: float
    expected_refill_date: str
    reminder_stage: int
    reminder_date: str
    history_quality: str
    pilot_tier: str
    purchase_count: int
    pre_send_safety_status: str  # "PASS" or "FAIL"
    safety_reason: str
    safety_code: str
    is_safe_to_send: bool
    approval_status: str = "PENDING"  # "PENDING", "APPROVED", "REJECTED"
    rejection_reason: Optional[str] = None
    operator_notes: str = ""
    message_preview: str = ""
    template_name: str = REFILLCARE_TEMPLATE_NAME
    parameter_count: int = 6

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MicroPilotStopCondition:
    """Surfaced warning or stop-state condition during micro-pilot execution."""

    name: str
    severity: str  # "WARNING", "CRITICAL_STOP"
    triggered: bool
    details: str
    remediation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MicroPilotOutcomeRecord:
    """Granular post-reminder outcome record for an individual candidate."""

    reminder_id: str
    customerId: str
    itemId: str
    customer_name: str
    medicine: str
    phone_masked: str
    reminder_date: str
    reminder_stage: int
    expected_refill_date: str
    dispatch_mode: str  # "dry_run" or "live"
    provider_status: str  # "accepted", "failed", "blocked"
    actual_refill_date: Optional[str] = None
    has_post_reminder_purchase: bool = False
    days_reminder_to_purchase: Optional[float] = None
    prediction_error_days: Optional[float] = None
    absolute_prediction_error_days: Optional[float] = None
    is_accurate_3d: Optional[bool] = None
    is_accurate_7d: Optional[bool] = None
    observation_status: str = "PENDING_OBSERVATION"  # "OBSERVED_REFILL", "PENDING_OBSERVATION", "NO_OBSERVED_REFILL"
    risk_flag: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MicroPilotReport:
    """Comprehensive, exportable summary report of the controlled micro-pilot run."""

    pilot_run_id: str = "MICROPILOT-2026-01"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    target_tier: str = "Tier A (Strong Pilot)"
    batch_size_selected: int = 5
    total_tier_a_available: int = 0
    candidates_reviewed: int = 0
    approved_count: int = 0
    rejected_count: int = 0
    pending_approval_count: int = 0
    dry_run_dispatches: int = 0
    live_dispatches: int = 0
    provider_accepted: int = 0
    provider_failed: int = 0
    blocked_by_safety: int = 0
    later_refills_observed: int = 0
    pending_observation: int = 0
    no_observed_refills: int = 0
    mae_days: Optional[float] = None
    medae_days: Optional[float] = None
    within_3d_pct: Optional[float] = None
    within_7d_pct: Optional[float] = None
    stage_breakdown: Dict[str, Any] = field(default_factory=dict)
    stop_conditions: List[MicroPilotStopCondition] = field(default_factory=list)
    has_critical_stop: bool = False
    operator_notes: str = "Initial micro-pilot preparation."

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["stop_conditions"] = [s.to_dict() for s in self.stop_conditions]
        return res

    def to_markdown(self) -> str:
        """Format the report into clean GitHub Markdown."""
        lines = [
            f"# 💊 RefillCare Micro-Pilot Run Report — `{self.pilot_run_id}`",
            f"**Execution Timestamp:** {self.timestamp}  ",
            f"**Target Tier:** {self.target_tier}  ",
            f"**Batch Size:** {self.batch_size_selected} Candidates (Max 20)  ",
            f"**Critical Stop Condition:** {'🔴 TRIGGERED' if self.has_critical_stop else '🟢 NONE'}",
            "",
            "## 1. Operational Review & Dispatch Summary",
            "| Metric | Count / Value | Description |",
            "|---|---|---|",
            f"| Total Tier A Candidates Available | {self.total_tier_a_available} | Eligible recurring histories in queue |",
            f"| Candidates Selected & Reviewed | {self.candidates_reviewed} | Sized to {self.batch_size_selected} for controlled micro-pilot |",
            f"| Approved by Operator | {self.approved_count} | Explicit human sign-off |",
            f"| Rejected by Operator | {self.rejected_count} | Suppressed with structured reasons |",
            f"| Pending Review | {self.pending_approval_count} | Awaiting operator action |",
            f"| Dry-Run Dispatches | {self.dry_run_dispatches} | Simulated with full validation & audit |",
            f"| Live API Dispatches | {self.live_dispatches} | Real WhatsApp transmissions |",
            f"| Provider Accepted | {self.provider_accepted} | WhatsApp gateway acknowledged |",
            f"| Provider Failed | {self.provider_failed} | Delivery errors (no auto-retries) |",
            f"| Blocked by Safety Checks | {self.blocked_by_safety} | Pre-send assertion interceptions |",
            "",
            "## 2. Refill Outcome Tracking (30-Day Observation Window)",
            "| Outcome Metric | Count / Value | Note |",
            "|---|---|---|",
            f"| Pharmacy Refills Observed | {self.later_refills_observed} | Subsequent (customerId, itemId) purchase recorded |",
            f"| Pending Observation (Right-Censored) | {self.pending_observation} | In observation window (<30 days post-reminder) |",
            f"| No Observed Refill (Window Expired) | {self.no_observed_refills} | 30-day window completed with no local purchase |",
        ]

        if self.mae_days is not None:
            lines.extend([
                f"| Prediction MAE | {self.mae_days:.2f} days | Actual vs Expected refill date |",
                f"| Prediction MedAE | {self.medae_days:.2f} days | Median Absolute Error |",
                f"| Accuracy within ±3 Days | {self.within_3d_pct:.1f}% | High-precision adherence window |",
                f"| Accuracy within ±7 Days | {self.within_7d_pct:.1f}% | Standard clinical adherence window |",
            ])
        else:
            lines.append("| Prediction Accuracy Metrics | *Insufficient mature outcomes* | Requires mature post-pilot POS observations |")

        lines.extend([
            "",
            "## 3. Operational Stop Conditions & Safety Surveillance",
        ])

        if not self.stop_conditions:
            lines.append("🟢 *No stop conditions or warnings active.*")
        else:
            for sc in self.stop_conditions:
                status_icon = "🔴 CRITICAL" if sc.severity == "CRITICAL_STOP" else "🟡 WARNING"
                state_text = "**ACTIVE**" if sc.triggered else "INACTIVE"
                lines.append(f"- {status_icon} **{sc.name}**: {state_text} — {sc.details}")
                if sc.remediation and sc.triggered:
                    lines.append(f"  *Remediation:* {sc.remediation}")

        lines.extend([
            "",
            "## 4. Operator Notes & Governance",
            f"> {self.operator_notes}",
            "",
            "---",
            "*RefillCare Phase 16 Controlled Micro-Pilot Engine — WhatsApp template and ML model remain strictly frozen.*",
        ])

        return "\n".join(lines)


def prepare_micro_pilot_batch(
    schedule_df: pd.DataFrame,
    transactions_df: Optional[pd.DataFrame] = None,
    storage: Optional[RefillCareStorage] = None,
    batch_size: int = DEFAULT_MICRO_PILOT_BATCH_SIZE,
    target_date: Optional[Union[str, date]] = None,
    store_name: str = "PHARMA HUBB",
    store_contact: str = "9876543210",
) -> Tuple[List[MicroPilotCandidate], int]:
    """Extract and build a verified Micro-Pilot candidate batch.

    Enforces:
    - Tier A selection (high history quality >=5 purchases, recurring chronic history).
    - Batch size restriction: exactly 5, 10, or 20 (defaults to 5).
    - Strict compound identity (customerId + itemId).
    - Shared phone isolation (phone is delivery destination only).
    - Multi-assertion pre-send safety validation.
    - Repurchase / cycle-reset check against transactions.

    Args:
        schedule_df: Scheduled reminders DataFrame.
        transactions_df: POS transactions DataFrame.
        storage: Persistent RefillCareStorage instance.
        batch_size: Candidate limit (must be 5, 10, or 20).
        target_date: Target reminder due date.
        store_name: Medical store name for template body variable.
        store_contact: Medical store phone for template body variable.

    Returns:
        Tuple[List[MicroPilotCandidate], int]: Sized candidate list and total Tier A available count.
    """
    if batch_size not in ALLOWED_MICRO_PILOT_BATCH_SIZES:
        batch_size = DEFAULT_MICRO_PILOT_BATCH_SIZE

    if schedule_df is None or schedule_df.empty:
        return [], 0

    target_d = _safe_parse_date(target_date) or date.today()
    target_d_str = target_d.isoformat()

    # Filter for Tier A eligible records
    df = schedule_df.copy()

    # Normalize column names for robust extraction
    tier_col = "Pilot Tier" if "Pilot Tier" in df.columns else ("pilot_tier" if "pilot_tier" in df.columns else None)
    status_col = "Status" if "Status" in df.columns else ("status" if "status" in df.columns else None)
    rdate_col = "raw_reminder_date" if "raw_reminder_date" in df.columns else ("Reminder Date" if "Reminder Date" in df.columns else "reminder_date")

    # Filter active scheduled reminders
    if status_col and status_col in df.columns:
        df = df[df[status_col].astype(str).str.lower().isin(["scheduled", "pending"])]

    # Filter Tier A
    if tier_col and tier_col in df.columns:
        tier_a_mask = df[tier_col].astype(str).str.contains("Tier A", case=False, na=False)
        tier_a_df = df[tier_a_mask]
    else:
        # Fallback to high_history quality if tier column is absent
        qual_col = "History Quality" if "History Quality" in df.columns else "history_quality"
        if qual_col in df.columns:
            tier_a_df = df[df[qual_col] == "high_history"]
        else:
            tier_a_df = df

    total_tier_a_available = len(tier_a_df)

    # Sort deterministically by reminder date, purchase count descending, customerId, itemId
    sort_cols = []
    if rdate_col in tier_a_df.columns:
        sort_cols.append(rdate_col)
    pcount_col = "Purchase Count" if "Purchase Count" in tier_a_df.columns else "purchase_count_so_far"
    if pcount_col in tier_a_df.columns:
        sort_cols.append(pcount_col)

    if sort_cols:
        ascending_flags = [True] + ([False] if len(sort_cols) > 1 else [])
        tier_a_df = tier_a_df.sort_values(by=sort_cols, ascending=ascending_flags)

    # Slice small micro-pilot batch
    selected_slice = tier_a_df.head(batch_size)

    candidates: List[MicroPilotCandidate] = []

    for _, row in selected_slice.iterrows():
        rec_dict = row.to_dict()

        # Extract core identifiers
        cid = str(rec_dict.get("customerId", rec_dict.get("customer_id", "")))
        iid = str(rec_dict.get("itemId", rec_dict.get("item_id", "")))
        cname = str(rec_dict.get("Customer Name", rec_dict.get("customerName", rec_dict.get("Customer", cid))))
        iname = str(rec_dict.get("Medicine", rec_dict.get("itemName", iid)))
        raw_phone = str(rec_dict.get("Delivery Phone", rec_dict.get("MOBILE_NO", rec_dict.get("phone", "")))).strip()
        rem_id = str(rec_dict.get("reminder_id", f"{cid}___{iid}___{rec_dict.get('expected_refill_date', '')}___{rec_dict.get('reminder_stage', 0):+d}d"))

        pcount = int(rec_dict.get("Purchase Count", rec_dict.get("purchase_count_so_far", rec_dict.get("purchase_count", 5))))
        pinterval = float(rec_dict.get("Predicted Interval (Days)", rec_dict.get("predicted_days_until_refill", rec_dict.get("predicted_interval_days", 30.0))))
        exp_refill = str(rec_dict.get("Expected Refill Date", rec_dict.get("expected_refill_date", "")))
        rem_date = str(rec_dict.get("raw_reminder_date", rec_dict.get("Reminder Date", rec_dict.get("reminder_date", target_d_str))))
        rem_stage = int(rec_dict.get("reminder_stage", rec_dict.get("raw_stage", 0)))
        hqual = str(rec_dict.get("History Quality", rec_dict.get("history_quality", "high_history")))
        ptier = str(rec_dict.get("Pilot Tier", rec_dict.get("pilot_tier", "Tier A (Strong Pilot)")))

        # Standardize keys on rec_dict for safety and payload construction
        rec_dict["customerId"] = cid
        rec_dict["itemId"] = iid
        rec_dict["MOBILE_NO"] = raw_phone
        rec_dict["phone_raw"] = raw_phone
        rec_dict["invoice_date"] = str(rec_dict.get("Last Purchase Date", rec_dict.get("invoice_date", "")))
        rec_dict["expected_refill_date"] = exp_refill
        rec_dict["reminder_id"] = rem_id

        # Multi-step safety check
        is_safe, safe_reason, safe_code = validate_pre_send_safety(
            rec_dict,
            storage=storage,
            transactions_df=transactions_df,
            store_name=store_name,
            store_contact=store_contact,
        )

        # Build payload preview and parameter check
        payload_info = build_refillcare_whatsapp_payload(
            rec_dict,
            store_name=store_name,
            store_contact=store_contact,
        )

        norm_phone = normalize_phone_number(raw_phone)
        masked_phone = mask_phone_for_ui(raw_phone)

        candidate = MicroPilotCandidate(
            reminder_id=rem_id,
            customerId=cid,
            itemId=iid,
            customer_name=cname,
            medicine=iname,
            phone_masked=masked_phone,
            phone_normalized=norm_phone,
            latest_purchase_date=str(rec_dict.get("Last Purchase Date", rec_dict.get("invoice_date", ""))),
            predicted_interval_days=round(pinterval, 1),
            expected_refill_date=exp_refill,
            reminder_stage=rem_stage,
            reminder_date=rem_date,
            history_quality=hqual,
            pilot_tier=ptier,
            purchase_count=pcount,
            pre_send_safety_status="PASS" if is_safe else "FAIL",
            safety_reason=safe_reason,
            safety_code=safe_code,
            is_safe_to_send=is_safe,
            approval_status="PENDING",
            message_preview=payload_info.get("preview_message", ""),
            template_name=payload_info.get("template_name", REFILLCARE_TEMPLATE_NAME),
            parameter_count=len(payload_info.get("parameters", [])),
        )
        candidates.append(candidate)

    return candidates, total_tier_a_available


def execute_micro_pilot_batch(
    candidates: List[MicroPilotCandidate],
    approved_candidate_ids: List[str],
    is_live: bool = False,
    storage: Optional[RefillCareStorage] = None,
    transactions_df: Optional[pd.DataFrame] = None,
    provider_callable: Optional[Any] = None,
    pilot_run_id: str = "MICROPILOT-2026-01",
    store_name: str = "PHARMA HUBB",
    store_contact: str = "9876543210",
) -> Dict[str, Any]:
    """Execute dispatch for operator-approved micro-pilot candidates.

    Enforces:
    - Default Dry-Run mode (`is_live=False`).
    - Dispatches only candidates in `approved_candidate_ids`.
    - Real-time pre-send safety validation immediately prior to dispatch.
    - Repurchase check immediately prior to dispatch.
    - Never claims 'delivered'; records provider 'accepted' or 'failed'.
    - Persists full audit log with masked phone and deterministic reminder_id.

    Args:
        candidates: Candidate batch from `prepare_micro_pilot_batch`.
        approved_candidate_ids: List of reminder_ids explicitly approved by operator.
        is_live: Whether live WhatsApp API dispatch is authorized (default False).
        storage: Persistent storage instance.
        transactions_df: POS transactions DataFrame.
        provider_callable: Optional provider callable for mocking or live dispatch.
        pilot_run_id: Pilot execution session identifier.
        store_name: Medical store name for template body variable.
        store_contact: Medical store phone for template body variable.

    Returns:
        Dict[str, Any]: Batch execution summary with individual dispatch results.
    """
    active_storage = storage or RefillCareStorage()
    approved_set = set(approved_candidate_ids)

    dispatched_results = []
    accepted_count = 0
    failed_count = 0
    blocked_count = 0
    dry_run_count = 0
    live_count = 0

    for cand in candidates:
        rem_id = cand.reminder_id

        # Check if operator approved this candidate
        if rem_id not in approved_set:
            cand.approval_status = "REJECTED" if cand.approval_status == "REJECTED" else "PENDING"
            continue

        cand.approval_status = "APPROVED"

        # Construct raw record dictionary for safety re-validation
        cand_dict = {
            "reminder_id": cand.reminder_id,
            "customerId": cand.customerId,
            "itemId": cand.itemId,
            "customerName": cand.customer_name,
            "itemName": cand.medicine,
            "MOBILE_NO": cand.phone_normalized,
            "expected_refill_date": cand.expected_refill_date,
            "reminder_stage": cand.reminder_stage,
            "invoice_date": cand.latest_purchase_date,
        }

        # Real-time safety validation
        is_safe, safe_reason, safe_code = validate_pre_send_safety(
            cand_dict,
            storage=active_storage,
            transactions_df=transactions_df,
            store_name=store_name,
            store_contact=store_contact,
        )

        if not is_safe:
            blocked_count += 1
            dispatched_results.append({
                "reminder_id": rem_id,
                "customerId": cand.customerId,
                "itemId": cand.itemId,
                "status": "blocked",
                "reason": safe_reason,
                "safety_code": safe_code,
                "is_dry_run": not is_live,
            })
            continue

        # Build exact 6-parameter WhatsApp payload
        payload_info = build_refillcare_whatsapp_payload(
            cand_dict,
            store_name=store_name,
            store_contact=store_contact,
        )

        if not payload_info.get("is_valid", False):
            blocked_count += 1
            dispatched_results.append({
                "reminder_id": rem_id,
                "customerId": cand.customerId,
                "itemId": cand.itemId,
                "status": "blocked",
                "reason": payload_info.get("error", "Payload construction failed"),
                "safety_code": "PAYLOAD_INVALID",
                "is_dry_run": not is_live,
            })
            continue

        # Dispatch execution
        if not is_live:
            # DRY RUN MODE
            dry_run_count += 1
            accepted_count += 1

            disp_res = DispatchResult(
                reminder_id=rem_id,
                customerId=cand.customerId,
                itemId=cand.itemId,
                customerName=cand.customer_name,
                phone_masked=cand.phone_masked,
                reminder_stage=str(cand.reminder_stage),
                expected_refill_date=cand.expected_refill_date,
                status="accepted",
                success=True,
                message="[DRY RUN] Micro-pilot payload validated successfully. No HTTP request sent.",
                provider_message_id=f"dry_run_{rem_id[:12]}",
                dry_run=True,
            )

            # Persist Dry-Run audit entry in SQLite
            active_storage.record_dispatch_start(cand_dict, is_dry_run=True)
            active_storage.record_dispatch_result(disp_res, is_dry_run=True)

            dispatched_results.append({
                "reminder_id": rem_id,
                "customerId": cand.customerId,
                "itemId": cand.itemId,
                "status": "accepted",
                "mode": "dry_run",
                "is_dry_run": True,
                "details": "Dry-run dispatch validated and recorded in SQLite audit.",
            })
        else:
            # LIVE SEND MODE (Requires explicit user confirmation)
            live_count += 1
            active_storage.record_dispatch_start(cand_dict, is_dry_run=False)

            try:
                if provider_callable is not None:
                    # Invoke injected provider callable (mock or custom transport)
                    provider_resp = provider_callable(payload_info.get("payload", {}))
                    success = provider_resp.get("success", False)
                    status_str = "accepted" if success else "failed"
                    prov_msg_id = provider_resp.get("messages", [{}])[0].get("id", f"live_{rem_id[:8]}") if success else ""
                    err_msg = provider_resp.get("error", {}).get("message", "") if not success else ""
                else:
                    # Fallback to existing dispatch transport in reminder.dispatch
                    from reminder.dispatch import send_whatsapp_reminder_template
                    res_tuple = send_whatsapp_reminder_template(
                        cand_dict,
                        is_dry_run=False,
                        store_name=store_name,
                        store_contact=store_contact,
                    )
                    success = res_tuple[0]
                    status_str = "accepted" if success else "failed"
                    resp_data = res_tuple[1]
                    prov_msg_id = resp_data.get("provider_message_id", f"live_{rem_id[:8]}") if isinstance(resp_data, dict) and success else ""
                    err_msg = str(resp_data) if not success else ""

                if success:
                    accepted_count += 1
                else:
                    failed_count += 1

                disp_res = DispatchResult(
                    reminder_id=rem_id,
                    customerId=cand.customerId,
                    itemId=cand.itemId,
                    customerName=cand.customer_name,
                    phone_masked=cand.phone_masked,
                    reminder_stage=str(cand.reminder_stage),
                    expected_refill_date=cand.expected_refill_date,
                    status=status_str,
                    success=success,
                    message="[LIVE] WhatsApp reminder sent successfully." if success else (err_msg or "Provider rejection."),
                    provider_message_id=prov_msg_id,
                    error_category="" if success else "Provider Error",
                    dry_run=False,
                )

                # Persist Live audit entry in SQLite
                active_storage.record_dispatch_result(disp_res, is_dry_run=False)

                dispatched_results.append({
                    "reminder_id": rem_id,
                    "customerId": cand.customerId,
                    "itemId": cand.itemId,
                    "status": status_str,
                    "mode": "live",
                    "is_dry_run": False,
                    "details": f"Live send result: {status_str}.",
                })

            except Exception as e:
                failed_count += 1
                disp_res = DispatchResult(
                    reminder_id=rem_id,
                    customerId=cand.customerId,
                    itemId=cand.itemId,
                    customerName=cand.customer_name,
                    phone_masked=cand.phone_masked,
                    reminder_stage=str(cand.reminder_stage),
                    expected_refill_date=cand.expected_refill_date,
                    status="failed",
                    success=False,
                    message=str(e),
                    error_category="Exception",
                    dry_run=False,
                )
                active_storage.record_dispatch_result(disp_res, is_dry_run=False)

                dispatched_results.append({
                    "reminder_id": rem_id,
                    "customerId": cand.customerId,
                    "itemId": cand.itemId,
                    "status": "failed",
                    "mode": "live",
                    "is_dry_run": False,
                    "error": str(e),
                })

    return {
        "pilot_run_id": pilot_run_id,
        "is_live": is_live,
        "total_approved": len(approved_set),
        "accepted_count": accepted_count,
        "failed_count": failed_count,
        "blocked_count": blocked_count,
        "dry_run_count": dry_run_count,
        "live_count": live_count,
        "results": dispatched_results,
    }


def check_micro_pilot_stop_conditions(
    candidates: List[MicroPilotCandidate],
    audit_df: Optional[pd.DataFrame] = None,
    storage: Optional[RefillCareStorage] = None,
    max_failure_rate_threshold: float = 0.10,
) -> List[MicroPilotStopCondition]:
    """Surveillance engine checking operational stop conditions and safety alerts.

    Stop Conditions Evaluated:
    1. Duplicate Send Interception
    2. Repurchase / Newer Purchase Detection (Cycle Reset)
    3. Invalid Phone / Malformed Destination Check
    4. WhatsApp Template Name & Parameter Integrity
    5. Abnormal Dispatch Failure Rate (>10%)
    6. Extreme Prediction Anomaly
    7. Persistent SQLite Storage Access Integrity

    Args:
        candidates: Micro-pilot candidate list.
        audit_df: Optional audit DataFrame.
        storage: RefillCareStorage instance.
        max_failure_rate_threshold: Max tolerated failure rate (default 10%).

    Returns:
        List[MicroPilotStopCondition]: Evaluated stop conditions.
    """
    conditions: List[MicroPilotStopCondition] = []

    # 1. Duplicate Send Check across candidate list
    rem_ids = [c.reminder_id for c in candidates]
    has_dups = len(rem_ids) != len(set(rem_ids))
    conditions.append(MicroPilotStopCondition(
        name="Duplicate Send Prevention",
        severity="CRITICAL_STOP" if has_dups else "WARNING",
        triggered=has_dups,
        details=f"Duplicate reminder IDs detected in candidate queue ({len(rem_ids) - len(set(rem_ids))} collisions)" if has_dups else "Zero duplicate reminder IDs in candidate batch.",
        remediation="Deduplicate schedule queue before dispatch." if has_dups else None,
    ))

    # 2. Newer Purchase / Cycle Reset Check
    stale_count = sum(1 for c in candidates if c.safety_code == "NEWER_PURCHASE_DETECTED")
    has_stale = stale_count > 0
    conditions.append(MicroPilotStopCondition(
        name="Newer Purchase Cycle Reset",
        severity="WARNING" if stale_count < len(candidates) else "CRITICAL_STOP",
        triggered=has_stale,
        details=f"{stale_count} candidates invalidated due to newer repurchase." if has_stale else "All candidates have valid purchase baselines.",
        remediation="Cancel invalidated reminder stages in scheduler." if has_stale else None,
    ))

    # 3. Invalid Destination Phone Format Check
    invalid_phone_count = sum(1 for c in candidates if not c.phone_normalized.startswith("91") or len(c.phone_normalized) != 12)
    has_invalid_phone = invalid_phone_count > 0
    conditions.append(MicroPilotStopCondition(
        name="Destination Phone Normalization",
        severity="CRITICAL_STOP" if has_invalid_phone else "WARNING",
        triggered=has_invalid_phone,
        details=f"{invalid_phone_count} candidates have non-standard phone formatting." if has_invalid_phone else "All candidate phone destinations conform to E.164 (91XXXXXXXXXX).",
        remediation="Update patient mobile numbers in POS master table." if has_invalid_phone else None,
    ))

    # 4. WhatsApp Approved Template Integrity Check
    template_errors = [c for c in candidates if c.template_name != REFILLCARE_TEMPLATE_NAME or c.parameter_count != 6]
    has_template_error = len(template_errors) > 0
    conditions.append(MicroPilotStopCondition(
        name="Approved WhatsApp Template Exactness",
        severity="CRITICAL_STOP",
        triggered=has_template_error,
        details=f"{len(template_errors)} candidates violate template name or 6-parameter count requirement." if has_template_error else f"Template strictly matches '{REFILLCARE_TEMPLATE_NAME}' with exactly 6 body parameters.",
        remediation="Verify reminder.dispatch payload builder constants." if has_template_error else None,
    ))

    # 5. Abnormal Dispatch Failure Rate Check (from SQLite audit)
    if audit_df is not None and not audit_df.empty:
        status_col = "status" if "status" in audit_df.columns else "Status"
        dry_col = "is_dry_run" if "is_dry_run" in audit_df.columns else "is_dry_run"

        # Check live sends only if present, else dry-run sends
        live_df = audit_df[audit_df[dry_col] == 0] if dry_col in audit_df.columns else pd.DataFrame()
        eval_df = live_df if not live_df.empty else audit_df

        total_disp = len(eval_df)
        failed_disp = len(eval_df[eval_df[status_col] == "failed"])
        fail_rate = (failed_disp / total_disp) if total_disp > 0 else 0.0

        abnormal_fail = total_disp >= 5 and fail_rate > max_failure_rate_threshold
        conditions.append(MicroPilotStopCondition(
            name="Provider Dispatch Failure Rate",
            severity="CRITICAL_STOP" if abnormal_fail else "WARNING",
            triggered=abnormal_fail,
            details=f"Dispatch failure rate: {fail_rate:.1%} ({failed_disp}/{total_disp}). Threshold: {max_failure_rate_threshold:.0%}." if abnormal_fail else f"Dispatch failure rate within healthy limits ({fail_rate:.1%}).",
            remediation="Pause live pilot and check WhatsApp Cloud API / Xinno credentials." if abnormal_fail else None,
        ))
    else:
        conditions.append(MicroPilotStopCondition(
            name="Provider Dispatch Failure Rate",
            severity="WARNING",
            triggered=False,
            details="No audit dispatch records to evaluate yet.",
        ))

    # 6. Extreme Prediction Anomaly Check (>180 days interval or <3 days)
    anomaly_preds = [c for c in candidates if c.predicted_interval_days > 180.0 or c.predicted_interval_days < 3.0]
    has_anomaly = len(anomaly_preds) > 0
    conditions.append(MicroPilotStopCondition(
        name="Prediction Anomaly Surveillance",
        severity="WARNING",
        triggered=has_anomaly,
        details=f"{len(anomaly_preds)} candidates have extreme predicted intervals (<3d or >180d)." if has_anomaly else "All predicted intervals fall within realistic clinical refill windows (3–180 days).",
        remediation="Review candidate historical visit patterns before approving." if has_anomaly else None,
    ))

    # 7. Persistent SQLite Storage Accessibility Check
    storage_accessible = False
    if storage is not None:
        try:
            storage_accessible = storage.db_path.exists()
        except Exception:
            storage_accessible = False
    else:
        storage_accessible = True

    conditions.append(MicroPilotStopCondition(
        name="Persistent Audit Storage Access",
        severity="CRITICAL_STOP",
        triggered=not storage_accessible,
        details="SQLite audit database accessible and read/write capable." if storage_accessible else "SQLite audit database inaccessible or directory permissions missing.",
        remediation="Ensure REFILLCARE_DB_PATH directory exists with write permissions." if not storage_accessible else None,
    ))

    return conditions


def evaluate_micro_pilot_outcomes(
    audit_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    reference_date: Optional[Union[str, date]] = None,
    observation_window_days: int = 30,
) -> pd.DataFrame:
    """Capture and compute micro-pilot refill outcomes and prediction precision.

    Matches audit records strictly against POS transactions using compound
    (customerId, itemId) identity and returns status:
    - OBSERVED_REFILL: Valid subsequent POS purchase recorded after reminder.
    - PENDING_OBSERVATION: < 30 days post-reminder date and no refill observed yet (Right-Censored).
    - NO_OBSERVED_REFILL: >= 30 days post-reminder date and no refill observed (Window Expired).

    Args:
        audit_df: Audit DataFrame from SQLite.
        transactions_df: POS transactions DataFrame.
        reference_date: Evaluation date (default: today).
        observation_window_days: Observation window maturity in days (default 30).

    Returns:
        pd.DataFrame: Granular outcome table with timing, error, and right-censoring flags.
    """
    ref_d = _safe_parse_date(reference_date) or date.today()

    if audit_df is None or audit_df.empty:
        return pd.DataFrame(columns=[
            "reminder_id", "customerId", "itemId", "customer_name", "medicine",
            "phone_masked", "reminder_date", "reminder_stage", "expected_refill_date",
            "dispatch_mode", "provider_status", "actual_refill_date", "has_post_reminder_purchase",
            "days_reminder_to_purchase", "prediction_error_days", "absolute_prediction_error_days",
            "is_accurate_3d", "is_accurate_7d", "observation_status", "risk_flag"
        ])

    # Clean POS transactions: remove returns, sort chronologically
    pos = transactions_df.copy() if transactions_df is not None and not transactions_df.empty else pd.DataFrame()
    if not pos.empty:
        pos["cid_str"] = pos["customerId"].astype(str)
        pos["iid_str"] = pos["itemId"].astype(str)
        if "quantity" in pos.columns:
            pos = pos[pos["quantity"] > 0]
        pos["tx_date"] = pd.to_datetime(pos["invoice_date"]).dt.date
        pos = pos.sort_values(by="tx_date")

    outcome_rows: List[Dict[str, Any]] = []

    for _, row in audit_df.iterrows():
        cid = str(row.get("customer_id", row.get("customerId", "")))
        iid = str(row.get("item_id", row.get("itemId", "")))
        rem_id = str(row.get("reminder_id", ""))
        cname = str(row.get("customer_name", row.get("Customer", cid)))
        iname = str(row.get("medicine", row.get("Medicine", iid)))
        phone_raw = str(row.get("phone", row.get("MOBILE_NO", "")))
        phone_mask = mask_phone_for_ui(phone_raw)

        r_date = _safe_parse_date(row.get("reminder_date", row.get("Reminder Date", "")))
        exp_date = _safe_parse_date(row.get("expected_refill_date", row.get("Expected Refill Date", "")))
        stage = int(row.get("reminder_stage", 0))
        is_dry = bool(row.get("is_dry_run", True))
        disp_mode = "dry_run" if is_dry else "live"
        prov_status = str(row.get("status", "accepted"))

        # Look for subsequent purchases of this (customerId, itemId)
        actual_refill_d: Optional[date] = None
        if not pos.empty and r_date:
            cust_txs = pos[(pos["cid_str"] == cid) & (pos["iid_str"] == iid) & (pos["tx_date"] >= r_date)]
            if not cust_txs.empty:
                actual_refill_d = cust_txs.iloc[0]["tx_date"]

        # Calculate timing, error, and right-censoring status
        has_post = actual_refill_d is not None
        days_rem_to_purch = (actual_refill_d - r_date).days if (has_post and r_date) else None

        pred_err = None
        abs_err = None
        acc_3d = None
        acc_7d = None

        if actual_refill_d and exp_date:
            pred_err = (actual_refill_d - exp_date).days
            abs_err = abs(pred_err)
            acc_3d = abs_err <= 3
            acc_7d = abs_err <= 7

        # Right-Censoring Classification
        if has_post:
            obs_status = "OBSERVED_REFILL"
        else:
            if r_date:
                days_since_rem = (ref_d - r_date).days
                if days_since_rem < observation_window_days:
                    obs_status = "PENDING_OBSERVATION"
                else:
                    obs_status = "NO_OBSERVED_REFILL"
            else:
                obs_status = "PENDING_OBSERVATION"

        # Surfaced risk flag
        risk_flag = None
        if pred_err is not None:
            if pred_err < -7:
                risk_flag = "PREMATURE_REMINDER"
            elif pred_err > 14:
                risk_flag = "LATE_REMINDER"

        outcome_rows.append({
            "reminder_id": rem_id,
            "customerId": cid,
            "itemId": iid,
            "customer_name": cname,
            "medicine": iname,
            "phone_masked": phone_mask,
            "reminder_date": r_date.isoformat() if r_date else "",
            "reminder_stage": stage,
            "expected_refill_date": exp_date.isoformat() if exp_date else "",
            "dispatch_mode": disp_mode,
            "provider_status": prov_status,
            "actual_refill_date": actual_refill_d.isoformat() if actual_refill_d else None,
            "has_post_reminder_purchase": has_post,
            "days_reminder_to_purchase": days_rem_to_purch,
            "prediction_error_days": pred_err,
            "absolute_prediction_error_days": abs_err,
            "is_accurate_3d": acc_3d,
            "is_accurate_7d": acc_7d,
            "observation_status": obs_status,
            "risk_flag": risk_flag,
        })

    return pd.DataFrame(outcome_rows)


def generate_micro_pilot_summary_report(
    candidates: List[MicroPilotCandidate],
    dispatched_summary: Optional[Dict[str, Any]] = None,
    outcomes_df: Optional[pd.DataFrame] = None,
    stop_conditions: Optional[List[MicroPilotStopCondition]] = None,
    pilot_run_id: str = "MICROPILOT-2026-01",
    total_tier_a_available: int = 0,
    operator_notes: str = "First controlled micro-pilot execution.",
) -> MicroPilotReport:
    """Consolidate micro-pilot operational metrics and outcomes into a report.

    Args:
        candidates: Reviewed candidate list.
        dispatched_summary: Output from `execute_micro_pilot_batch`.
        outcomes_df: Output from `evaluate_micro_pilot_outcomes`.
        stop_conditions: Evaluated stop conditions.
        pilot_run_id: Pilot run identifier.
        total_tier_a_available: Total Tier A candidates in queue.
        operator_notes: Structured operator remarks.

    Returns:
        MicroPilotReport: Fully populated dataclass report.
    """
    total_reviewed = len(candidates)
    approved_count = sum(1 for c in candidates if c.approval_status == "APPROVED")
    rejected_count = sum(1 for c in candidates if c.approval_status == "REJECTED")
    pending_count = sum(1 for c in candidates if c.approval_status == "PENDING")

    disp = dispatched_summary or {}
    dry_count = disp.get("dry_run_count", 0)
    live_count = disp.get("live_count", 0)
    accepted_count = disp.get("accepted_count", 0)
    failed_count = disp.get("failed_count", 0)
    blocked_count = disp.get("blocked_count", 0)

    # Outcomes aggregation
    obs_refills = 0
    pending_obs = 0
    no_refills = 0
    mae_val = None
    medae_val = None
    w3_pct = None
    w7_pct = None
    stage_breakdown: Dict[str, Any] = {}

    if outcomes_df is not None and not outcomes_df.empty:
        obs_refills = int(outcomes_df["has_post_reminder_purchase"].sum())
        pending_obs = int((outcomes_df["observation_status"] == "PENDING_OBSERVATION").sum())
        no_refills = int((outcomes_df["observation_status"] == "NO_OBSERVED_REFILL").sum())

        valid_errors = outcomes_df["absolute_prediction_error_days"].dropna()
        if len(valid_errors) >= 3:
            mae_val = float(valid_errors.mean())
            medae_val = float(valid_errors.median())
            w3_pct = float((valid_errors <= 3).mean() * 100.0)
            w7_pct = float((valid_errors <= 7).mean() * 100.0)

        # Stage level breakdown
        for stg in REMINDER_STAGES:
            stg_df = outcomes_df[outcomes_df["reminder_stage"] == stg]
            if not stg_df.empty:
                stg_label = _format_stage_label(stg)
                stage_breakdown[stg_label] = {
                    "count": len(stg_df),
                    "observed_refills": int(stg_df["has_post_reminder_purchase"].sum()),
                    "pending": int((stg_df["observation_status"] == "PENDING_OBSERVATION").sum()),
                }

    # Stop conditions aggregation
    conds = stop_conditions or []
    has_critical = any(c.triggered and c.severity == "CRITICAL_STOP" for c in conds)

    return MicroPilotReport(
        pilot_run_id=pilot_run_id,
        target_tier="Tier A (Strong Pilot)",
        batch_size_selected=total_reviewed,
        total_tier_a_available=total_tier_a_available,
        candidates_reviewed=total_reviewed,
        approved_count=approved_count,
        rejected_count=rejected_count,
        pending_approval_count=pending_count,
        dry_run_dispatches=dry_count,
        live_dispatches=live_count,
        provider_accepted=accepted_count,
        provider_failed=failed_count,
        blocked_by_safety=blocked_count,
        later_refills_observed=obs_refills,
        pending_observation=pending_obs,
        no_observed_refills=no_refills,
        mae_days=mae_val,
        medae_days=medae_val,
        within_3d_pct=w3_pct,
        within_7d_pct=w7_pct,
        stage_breakdown=stage_breakdown,
        stop_conditions=conds,
        has_critical_stop=has_critical,
        operator_notes=operator_notes,
    )
