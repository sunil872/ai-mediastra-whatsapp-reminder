"""RefillCare Phase 12 — Controlled Pilot Execution & Monitoring Engine.

Provides operational funnel tracking, post-reminder refill matching,
right-censoring handling, prediction accuracy measurement, reminder fatigue
analytics, audit reconciliation, and daily pilot reporting.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd

from reminder.storage import _extract_phone_last4


@dataclass
class PilotConfiguration:
    """Standardized configuration for the RefillCare 30-Day Controlled Pilot."""

    pilot_name: str = "RefillCare 30-Day Controlled Pilot"
    version: str = "1.0.0"
    duration_days: int = 30
    mode: str = "controlled_manual"  # Strictly operator-controlled
    auto_send_enabled: bool = False  # NEVER True in pilot
    auto_retry_enabled: bool = False  # NEVER True in pilot
    observation_window_days: int = 30
    approved_stages: List[int] = field(default_factory=lambda: [-7, -3, -1, 0, 2, 5])
    primary_tier: str = "Tier A (Strong Pilot)"
    max_reminders_per_cycle: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert pilot configuration to standard dictionary."""
        return asdict(self)


def _safe_parse_date(d: Any) -> Optional[date]:
    """Safely convert string, Timestamp, or datetime to date object."""
    if d is None or pd.isna(d) or str(d).strip() in ("", "NaT", "None", "-"):
        return None
    if isinstance(d, (datetime, pd.Timestamp)):
        return d.date()
    if isinstance(d, date):
        return d
    try:
        s = str(d).strip()
        if len(s) >= 10 and s[4] in ("-", "/") and s[:4].isdigit():
            return pd.to_datetime(s, dayfirst=False).date()
        else:
            return pd.to_datetime(s, dayfirst=True).date()
    except Exception:
        return None


def compute_pilot_funnel(
    audit_df: pd.DataFrame,
    eligible_df: pd.DataFrame,
    ineligible_df: Optional[pd.DataFrame] = None,
    matched_refills_df: Optional[pd.DataFrame] = None,
    current_date: Optional[Union[str, date]] = None,
) -> Dict[str, Any]:
    """Calculate the operational end-to-end pilot funnel metrics.

    Funnel Stages:
    1. Total Identified Candidates (Eligible + Cold-Start)
    2. Eligible Regimens (Tier A + Tier B + Tier C)
    3. Reminders Due for Dispatch on or before current date
    4. Reviewed by Operator
    5. Approved by Operator
    6. Dispatch Attempted (Recorded in SQLite audit)
    7. WhatsApp Provider Accepted (Live or Dry-Run)
    8. Provider Failed / Rejected
    9. Duplicates Intercepted & Prevented
    10. Observed Post-Reminder Refills (Supported by actual purchase data)
    """
    curr_d = _safe_parse_date(current_date) or date.today()

    total_eligible = len(eligible_df) if eligible_df is not None else 0
    total_ineligible = len(ineligible_df) if ineligible_df is not None else 0
    total_candidates = total_eligible + total_ineligible

    tier_a_count = 0
    tier_b_count = 0
    tier_c_count = 0
    if eligible_df is not None and not eligible_df.empty and "Pilot Tier" in eligible_df.columns:
        t_counts = eligible_df["Pilot Tier"].value_counts().to_dict()
        tier_a_count = t_counts.get("Tier A (Strong Pilot)", 0)
        tier_b_count = t_counts.get("Tier B (Review Required)", 0)
        tier_c_count = t_counts.get("Tier C (Suppressed / Low History)", 0)

    # Analyze audit records
    dispatch_attempted = 0
    provider_accepted = 0
    provider_accepted_live = 0
    provider_accepted_dry = 0
    provider_failed = 0
    duplicates_prevented = 0
    cancelled_count = 0
    due_today_count = 0

    if audit_df is not None and not audit_df.empty:
        status_col = "Status" if "Status" in audit_df.columns else "status"
        dry_col = "is_dry_run" if "is_dry_run" in audit_df.columns else "Is Dry Run"
        r_date_col = "Reminder Date" if "Reminder Date" in audit_df.columns else "reminder_date"

        dispatch_attempted = len(audit_df[audit_df[status_col].isin(["accepted", "sending", "failed"])])
        
        accepted_mask = audit_df[status_col].str.lower() == "accepted"
        provider_accepted = int(accepted_mask.sum())

        if dry_col in audit_df.columns:
            dry_mask = audit_df[dry_col].astype(str).isin(["1", "True", "true", "1.0"])
            provider_accepted_dry = int((accepted_mask & dry_mask).sum())
            provider_accepted_live = int((accepted_mask & ~dry_mask).sum())
        else:
            provider_accepted_dry = provider_accepted
            provider_accepted_live = 0

        provider_failed = int((audit_df[status_col].str.lower() == "failed").sum())
        duplicates_prevented = int((audit_df[status_col].str.lower() == "duplicate_prevented").sum())
        cancelled_count = int((audit_df[status_col].str.lower() == "cancelled").sum())

        # Due count
        if r_date_col in audit_df.columns:
            parsed_dates = audit_df[r_date_col].apply(_safe_parse_date)
            due_today_count = int((parsed_dates == curr_d).sum())

    # Observed Refill metrics
    observed_refills = 0
    pending_observation = 0
    no_refill_closed = 0
    post_reminder_refill_rate = 0.0

    if matched_refills_df is not None and not matched_refills_df.empty:
        stat_col = "refill_status" if "refill_status" in matched_refills_df.columns else "Status"
        refill_counts = matched_refills_df[stat_col].value_counts().to_dict()
        observed_refills = refill_counts.get("Observed Refill", 0)
        pending_observation = refill_counts.get("Pending Observation (Window Open)", 0)
        no_refill_closed = refill_counts.get("No Observed Refill (Window Closed)", 0)

        denom = observed_refills + no_refill_closed
        if denom > 0:
            post_reminder_refill_rate = round((observed_refills / denom) * 100.0, 2)

    return {
        "report_date": curr_d.isoformat(),
        "total_candidates": total_candidates,
        "eligible_candidates": total_eligible,
        "cold_start_excluded": total_ineligible,
        "tier_a_count": tier_a_count,
        "tier_b_count": tier_b_count,
        "tier_c_count": tier_c_count,
        "reminders_due_today": due_today_count,
        "dispatch_attempted": dispatch_attempted,
        "provider_accepted": provider_accepted,
        "provider_accepted_live": provider_accepted_live,
        "provider_accepted_dry": provider_accepted_dry,
        "provider_failed": provider_failed,
        "duplicates_prevented": duplicates_prevented,
        "cancelled_count": cancelled_count,
        "observed_refills": observed_refills,
        "pending_observation": pending_observation,
        "no_refill_closed": no_refill_closed,
        "post_reminder_refill_rate_pct": post_reminder_refill_rate,
    }


def match_post_reminder_refills(
    audit_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    observation_window_days: int = 30,
    reference_date: Optional[Union[str, date]] = None,
) -> pd.DataFrame:
    """Match accepted reminder dispatches to subsequent actual purchases for the exact (customerId, itemId).

    Rules:
    1. Identity Match: Strictly uses customerId + itemId. Shared phone numbers do NOT merge purchases.
    2. Temporal Order: Looks for transactions with invoice_date strictly AFTER reminder_date (or expected_refill_date).
    3. Observation Window: Refill is counted if it occurs within [1, observation_window_days] after reminder_date.
    4. Right-Censoring: If no purchase is observed and (ref_date - reminder_date) <= observation_window_days,
       the record is flagged as 'Pending Observation (Window Open)' rather than a failure.
    5. Non-Causal Tracking: Labeled as 'Observed Post-Reminder Refill' (association only, not causal attribution).

    Args:
        audit_df: DataFrame of dispatched reminders from SQLite audit.
        transactions_df: DataFrame of actual purchase transactions.
        observation_window_days: Window in days to monitor for a repurchase (default: 30 days).
        reference_date: Current evaluation date (default: today).

    Returns:
        pd.DataFrame: Audit records enriched with actual refill dates, errors, and right-censoring flags.
    """
    if audit_df.empty:
        return pd.DataFrame(columns=[
            "reminder_id", "customerId", "itemId", "customerName", "itemName",
            "phone_masked", "reminder_date", "expected_refill_date", "reminder_stage",
            "status", "is_dry_run", "actual_refill_date", "days_to_refill",
            "prediction_error_days", "absolute_error_days", "refill_status",
            "is_accurate_7d", "is_accurate_3d", "is_accurate_1d"
        ])

    ref_d = _safe_parse_date(reference_date) or date.today()

    # Standardize column names
    cid_col = "customer_id" if "customer_id" in audit_df.columns else "customerId"
    iid_col = "item_id" if "item_id" in audit_df.columns else "itemId"
    r_date_col = "reminder_date" if "reminder_date" in audit_df.columns else "Reminder Date"
    exp_date_col = "expected_refill_date" if "expected_refill_date" in audit_df.columns else "Expected Refill Date"
    status_col = "status" if "status" in audit_df.columns else "Status"
    dry_col = "is_dry_run" if "is_dry_run" in audit_df.columns else "Is Dry Run"

    # Pre-index transactions by (customerId, itemId)
    tx_df = transactions_df.copy() if transactions_df is not None else pd.DataFrame()
    tx_lookup: Dict[str, List[date]] = {}

    if not tx_df.empty:
        t_cid = "customerId" if "customerId" in tx_df.columns else "customer_id"
        t_iid = "itemId" if "itemId" in tx_df.columns else "item_id"
        t_date = "invoice_date" if "invoice_date" in tx_df.columns else "Invoice Date"

        tx_clean = tx_df[[t_cid, t_iid, t_date]].dropna().copy()
        tx_clean["parsed_date"] = tx_clean[t_date].apply(_safe_parse_date)
        tx_clean = tx_clean.dropna(subset=["parsed_date"]).sort_values("parsed_date")

        for _, row in tx_clean.iterrows():
            k = f"{str(row[t_cid]).strip()}:::{str(row[t_iid]).strip()}"
            if k not in tx_lookup:
                tx_lookup[k] = []
            tx_lookup[k].append(row["parsed_date"])

    matched_records: List[Dict[str, Any]] = []

    for _, row in audit_df.iterrows():
        st_val = str(row.get(status_col, "")).lower()
        if st_val not in ("accepted", "sent"):
            continue

        cid = str(row.get(cid_col, "")).strip()
        iid = str(row.get(iid_col, "")).strip()
        rem_d = _safe_parse_date(row.get(r_date_col))
        exp_d = _safe_parse_date(row.get(exp_date_col))

        if not rem_d or not exp_d or not cid or not iid:
            continue

        key = f"{cid}:::{iid}"
        subsequent_purchases = [p for p in tx_lookup.get(key, []) if p > rem_d]

        rec = {
            "reminder_id": str(row.get("reminder_id", "")),
            "customerId": cid,
            "itemId": iid,
            "customerName": str(row.get("customer_name", row.get("Customer", cid))),
            "itemName": str(row.get("item_name", row.get("Medicine", iid))),
            "phone_masked": str(row.get("phone_last4", row.get("Phone Last-4", ""))),
            "reminder_date": rem_d.isoformat(),
            "expected_refill_date": exp_d.isoformat(),
            "reminder_stage": str(row.get("reminder_stage", row.get("Reminder Stage", "0"))),
            "status": st_val,
            "is_dry_run": int(row.get(dry_col, 0) in (1, "1", True, "True")),
            "actual_refill_date": None,
            "days_to_refill": None,
            "prediction_error_days": None,
            "absolute_error_days": None,
            "refill_status": "No Observed Refill",
            "is_accurate_7d": False,
            "is_accurate_3d": False,
            "is_accurate_1d": False,
        }

        if subsequent_purchases:
            # First subsequent purchase after reminder date
            actual_d = subsequent_purchases[0]
            days_from_rem = (actual_d - rem_d).days

            if 0 < days_from_rem <= observation_window_days:
                pred_err = (actual_d - exp_d).days
                abs_err = abs(pred_err)

                rec["actual_refill_date"] = actual_d.isoformat()
                rec["days_to_refill"] = days_from_rem
                rec["prediction_error_days"] = pred_err
                rec["absolute_error_days"] = abs_err
                rec["refill_status"] = "Observed Refill"
                rec["is_accurate_7d"] = bool(abs_err <= 7)
                rec["is_accurate_3d"] = bool(abs_err <= 3)
                rec["is_accurate_1d"] = bool(abs_err <= 1)
            else:
                rec["refill_status"] = "No Observed Refill (Window Closed)"
        else:
            # Handle Right-Censoring
            days_elapsed = (ref_d - rem_d).days
            if days_elapsed <= observation_window_days:
                rec["refill_status"] = "Pending Observation (Window Open)"
            else:
                rec["refill_status"] = "No Observed Refill (Window Closed)"

        matched_records.append(rec)

    return pd.DataFrame(matched_records)


def evaluate_pilot_prediction_accuracy(matched_refills_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate empirical prediction accuracy metrics on observed post-reminder refills.

    Only evaluates records where an actual subsequent purchase date was observed.
    """
    if matched_refills_df.empty:
        return {
            "evaluated_refill_count": 0,
            "mae_days": None,
            "rmse_days": None,
            "median_absolute_error_days": None,
            "within_1_day_pct": None,
            "within_3_days_pct": None,
            "within_7_days_pct": None,
        }

    observed = matched_refills_df[matched_refills_df["refill_status"] == "Observed Refill"].copy()
    n = len(observed)

    if n == 0:
        return {
            "evaluated_refill_count": 0,
            "mae_days": None,
            "rmse_days": None,
            "median_absolute_error_days": None,
            "within_1_day_pct": None,
            "within_3_days_pct": None,
            "within_7_days_pct": None,
        }

    abs_errors = observed["absolute_error_days"].dropna().astype(float).values
    errors = observed["prediction_error_days"].dropna().astype(float).values

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    med_ae = float(np.median(abs_errors))

    within_1 = float((abs_errors <= 1.0).sum() / n * 100.0)
    within_3 = float((abs_errors <= 3.0).sum() / n * 100.0)
    within_7 = float((abs_errors <= 7.0).sum() / n * 100.0)

    return {
        "evaluated_refill_count": n,
        "mae_days": round(mae, 2),
        "rmse_days": round(rmse, 2),
        "median_absolute_error_days": round(med_ae, 2),
        "within_1_day_pct": round(within_1, 2),
        "within_3_days_pct": round(within_3, 2),
        "within_7_days_pct": round(within_7, 2),
    }


def analyze_reminder_fatigue(audit_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze notification distribution, multi-stage exposure, and potential customer fatigue."""
    if audit_df.empty:
        return {
            "total_dispatches": 0,
            "unique_customers": 0,
            "avg_reminders_per_customer": 0.0,
            "max_reminders_single_customer": 0,
            "stage_distribution": {},
            "customers_with_3_plus_reminders": 0,
            "followup_stage_count": 0,  # +2d and +5d
        }

    cid_col = "customer_id" if "customer_id" in audit_df.columns else "customerId"
    stage_col = "reminder_stage" if "reminder_stage" in audit_df.columns else "Reminder Stage"

    per_cust = audit_df.groupby(cid_col).size()
    total_disp = len(audit_df)
    unique_c = len(per_cust)
    avg_rems = float(per_cust.mean()) if unique_c > 0 else 0.0
    max_rems = int(per_cust.max()) if unique_c > 0 else 0
    c_3plus = int((per_cust >= 3).sum())

    stage_dist = audit_df[stage_col].astype(str).value_counts().to_dict()

    followup_count = sum(
        v for k, v in stage_dist.items() if any(tag in k for tag in ["+2", "+5", "2d", "5d", "2 days", "5 days"])
    )

    return {
        "total_dispatches": total_disp,
        "unique_customers": unique_c,
        "avg_reminders_per_customer": round(avg_rems, 2),
        "max_reminders_single_customer": max_rems,
        "stage_distribution": stage_dist,
        "customers_with_3_plus_reminders": c_3plus,
        "followup_stage_count": followup_count,
    }


def reconcile_audit_state(
    scheduler_reminders: List[Dict[str, Any]],
    storage: Any,
) -> Dict[str, Any]:
    """Reconcile in-memory scheduler state against persistent SQLite audit state.

    Detects:
    - Missing audit records
    - In-sync records
    - In-flight or stale sending records
    - Duplicate prevention integrity
    """
    total_scheduled_in_memory = len(scheduler_reminders)
    in_sync_count = 0
    missing_in_db = 0
    stale_sending_count = 0

    all_db_rems = storage.get_all_reminders() if hasattr(storage, "get_all_reminders") else []
    db_lookup = {r["reminder_id"]: r for r in all_db_rems}

    for rem in scheduler_reminders:
        r_id = rem.get("reminder_id", "")
        if r_id in db_lookup:
            in_sync_count += 1
            if db_lookup[r_id].get("status") == "sending":
                stale_sending_count += 1
        else:
            missing_in_db += 1

    return {
        "scheduler_count": total_scheduled_in_memory,
        "database_audit_count": len(all_db_rems),
        "in_sync_count": in_sync_count,
        "missing_in_db": missing_in_db,
        "stale_sending_count": stale_sending_count,
        "audit_health": "HEALTHY" if missing_in_db == 0 and stale_sending_count == 0 else "ATTENTION_REQUIRED",
    }


def generate_daily_pilot_summary(
    audit_df: pd.DataFrame,
    eligible_df: pd.DataFrame,
    ineligible_df: Optional[pd.DataFrame] = None,
    matched_refills_df: Optional[pd.DataFrame] = None,
    report_date: Optional[Union[str, date]] = None,
) -> pd.DataFrame:
    """Generate structured daily aggregated pilot summary row for export and trend tracking."""
    funnel = compute_pilot_funnel(
        audit_df=audit_df,
        eligible_df=eligible_df,
        ineligible_df=ineligible_df,
        matched_refills_df=matched_refills_df,
        current_date=report_date,
    )

    acc = evaluate_pilot_prediction_accuracy(matched_refills_df if matched_refills_df is not None else pd.DataFrame())

    summary_data = {
        "Report Date": [funnel["report_date"]],
        "Total Candidates": [funnel["total_candidates"]],
        "Eligible Regimens": [funnel["eligible_candidates"]],
        "Tier A (Strong Pilot)": [funnel["tier_a_count"]],
        "Tier B (Review Required)": [funnel["tier_b_count"]],
        "Tier C (Suppressed)": [funnel["tier_c_count"]],
        "Cold Start Excluded": [funnel["cold_start_excluded"]],
        "Due Today": [funnel["reminders_due_today"]],
        "Dispatch Attempted": [funnel["dispatch_attempted"]],
        "Provider Accepted (Live)": [funnel["provider_accepted_live"]],
        "Provider Accepted (Dry-Run)": [funnel["provider_accepted_dry"]],
        "Provider Failed": [funnel["provider_failed"]],
        "Duplicates Blocked": [funnel["duplicates_prevented"]],
        "Cancelled Cycles": [funnel["cancelled_count"]],
        "Observed Refills": [funnel["observed_refills"]],
        "Pending Observation (Open)": [funnel["pending_observation"]],
        "No Refill (Closed)": [funnel["no_refill_closed"]],
        "Post-Reminder Refill Rate (%)": [funnel["post_reminder_refill_rate_pct"]],
        "Evaluated Refills": [acc["evaluated_refill_count"]],
        "Refill MAE (Days)": [acc["mae_days"]],
        "Refill ±7d Accuracy (%)": [acc["within_7_days_pct"]],
    }

    return pd.DataFrame(summary_data)
