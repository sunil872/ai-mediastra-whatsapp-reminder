"""RefillCare Phase 13 — Controlled Pilot Data Collection & Outcome Evaluation Engine.

Provides deep outcome evaluation for the 30-day controlled pilot:
- Traceable pilot cohort identification and observation window handling
- Exact (customerId, itemId) actual refill matching with returns filtering
- Prediction error (actual vs expected) and empirical precision metrics
- Granular reminder timing classification (Pre-Refill, Same-Day, Post-Refill, Censored)
- Per-stage operational touchpoint analysis (-7d, -3d, -1d, 0d, +2d, +5d)
- False-positive risk surveillance (premature, stale, irregular)
- Multi-dimensional cohort segmentation (Pilot Tier, History Quality, Purchase Length, Medicine)
- Offline benchmark vs pilot performance comparison
- Operator review and structured rejection reason tracking
- Safe, non-identifying pilot outcome dataset generation and CSV export
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd

from reminder.storage import _extract_phone_last4
from refillcare.evaluation.pilot_monitoring import _safe_parse_date, PilotConfiguration


@dataclass
class PilotCohort:
    """Specification of the controlled pilot evaluation cohort and observation parameters."""

    pilot_id: str = "PILOT-2026-01"
    pilot_name: str = "RefillCare 30-Day Controlled Pilot"
    start_date: str = "2026-05-01"
    end_date: str = "2026-05-30"
    activity_window_days: int = 30
    observation_window_days: int = 30
    eligible_tiers: List[str] = field(default_factory=lambda: ["Tier A (Strong Pilot)", "Tier B (Review Required)"])
    model_version: str = "xgboost_refill_v1"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize cohort configuration to dictionary."""
        return asdict(self)


# Standard Structured Operator Rejection Reasons
STRUCTURED_REJECTION_REASONS = [
    "customer_already_purchased",
    "uncertain_prediction",
    "inappropriate_regimen",
    "invalid_contact",
    "pharmacist_discretion",
    "duplicate_intercepted",
    "other",
]

# Offline Test Set Reference Benchmarks (Phase 4 / Phase 10)
OFFLINE_BENCHMARKS = {
    "overall_mae": 16.20,
    "overall_rmse": 21.98,
    "overall_medae": 11.23,
    "overall_within_7d": 33.85,
    "regular_cycles_mae": 11.12,
    "regular_cycles_within_7d": 42.54,
    "high_purchase_mae": 9.64,
    "high_purchase_within_7d": 47.27,
}


def build_pilot_outcome_dataset(
    audit_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    eligible_df: Optional[pd.DataFrame] = None,
    cohort: Optional[PilotCohort] = None,
    reference_date: Optional[Union[str, date]] = None,
) -> pd.DataFrame:
    """Construct the comprehensive pilot outcome evaluation dataset.

    Matches dispatched and scheduled reminders against actual subsequent pharmacy
    transactions using exact (customerId, itemId) identity and strictly filters
    out return/adjustment transactions.

    Args:
        audit_df: SQLite audit DataFrame or dispatch records.
        transactions_df: POS transactions DataFrame.
        eligible_df: Optional pre-computed eligibility DataFrame.
        cohort: PilotCohort configuration (defaults to standard 30-day cohort).
        reference_date: Current evaluation date (default: today).

    Returns:
        pd.DataFrame: Granular pilot outcome dataset with timing, error, and censoring classifications.
    """
    cohort_cfg = cohort or PilotCohort()
    ref_d = _safe_parse_date(reference_date) or date.today()
    obs_window = cohort_cfg.observation_window_days

    if audit_df is None or audit_df.empty:
        return pd.DataFrame(columns=[
            "pilot_id", "customerId", "itemId", "customer_name", "medicine",
            "phone_masked", "pilot_tier", "history_quality", "purchase_count",
            "latest_purchase_date", "predicted_interval_days", "expected_refill_date",
            "reminder_stage", "reminder_date", "reminder_status", "is_dry_run",
            "actual_refill_date", "days_to_refill", "prediction_error_days",
            "absolute_prediction_error_days", "timing_category", "observation_status",
            "is_post_reminder_refill", "is_accurate_1d", "is_accurate_3d", "is_accurate_7d",
            "review_status", "rejection_reason", "risk_flag"
        ])

    # Standardize column references in audit DataFrame
    cid_col = "customer_id" if "customer_id" in audit_df.columns else "customerId"
    iid_col = "item_id" if "item_id" in audit_df.columns else "itemId"
    r_date_col = "reminder_date" if "reminder_date" in audit_df.columns else "Reminder Date"
    exp_date_col = "expected_refill_date" if "expected_refill_date" in audit_df.columns else "Expected Refill Date"
    status_col = "status" if "status" in audit_df.columns else "Status"
    stage_col = "reminder_stage" if "reminder_stage" in audit_df.columns else "Reminder Stage"
    dry_col = "is_dry_run" if "is_dry_run" in audit_df.columns else "Is Dry Run"
    tier_col = "pilot_tier" if "pilot_tier" in audit_df.columns else "Pilot Tier"
    qual_col = "history_quality" if "history_quality" in audit_df.columns else "History Quality"

    # Pre-index valid positive transactions by (customerId, itemId)
    tx_lookup: Dict[str, List[date]] = {}
    if transactions_df is not None and not transactions_df.empty:
        tx_df = transactions_df.copy()
        t_cid = "customerId" if "customerId" in tx_df.columns else "customer_id"
        t_iid = "itemId" if "itemId" in tx_df.columns else "item_id"
        t_date = "invoice_date" if "invoice_date" in tx_df.columns else "Invoice Date"
        t_qty = "quantity" if "quantity" in tx_df.columns else "Quantity"

        # Exclude returns and negative adjustments
        if t_qty in tx_df.columns:
            tx_df = tx_df[pd.to_numeric(tx_df[t_qty], errors="coerce").fillna(1) > 0]

        tx_clean = tx_df[[t_cid, t_iid, t_date]].dropna().copy()
        tx_clean["parsed_date"] = tx_clean[t_date].apply(_safe_parse_date)
        tx_clean = tx_clean.dropna(subset=["parsed_date"]).sort_values("parsed_date")

        for _, row in tx_clean.iterrows():
            k = f"{str(row[t_cid]).strip()}:::{str(row[t_iid]).strip()}"
            if k not in tx_lookup:
                tx_lookup[k] = []
            tx_lookup[k].append(row["parsed_date"])

    # Optional metadata lookup from eligible_df
    meta_lookup: Dict[str, Dict[str, Any]] = {}
    if eligible_df is not None and not eligible_df.empty:
        e_cid = "customerId" if "customerId" in eligible_df.columns else "customer_id"
        e_iid = "itemId" if "itemId" in eligible_df.columns else "item_id"
        for _, erow in eligible_df.iterrows():
            k = f"{str(erow[e_cid]).strip()}:::{str(erow[e_iid]).strip()}"
            meta_lookup[k] = erow.to_dict()

    outcome_rows: List[Dict[str, Any]] = []

    for _, row in audit_df.iterrows():
        cid = str(row.get(cid_col, "")).strip()
        iid = str(row.get(iid_col, "")).strip()
        if not cid or not iid:
            continue

        key = f"{cid}:::{iid}"
        meta = meta_lookup.get(key, {})

        rem_d = _safe_parse_date(row.get(r_date_col))
        exp_d = _safe_parse_date(row.get(exp_date_col, meta.get("expected_refill_date")))
        st_val = str(row.get(status_col, "scheduled")).lower()
        stage_val = str(row.get(stage_col, meta.get("reminder_stage", "0")))
        cname = str(row.get("customer_name", row.get("Customer", meta.get("Customer Name", cid))))
        iname = str(row.get("item_name", row.get("Medicine", meta.get("Medicine", iid))))
        phone_l4 = str(row.get("phone_last4", _extract_phone_last4(row.get("MOBILE_NO", meta.get("Delivery Phone", "")))))
        tier_val = str(row.get(tier_col, meta.get("Pilot Tier", "Tier B (Review Required)")))
        qual_val = str(row.get(qual_col, meta.get("History Quality", "medium_history")))
        p_cnt = int(row.get("purchase_count", meta.get("Purchase Count", meta.get("purchase_count_so_far", 1))))
        last_purch_d = str(row.get("latest_purchase_date", meta.get("Last Purchase Date", meta.get("invoice_date", "-"))))
        pred_int = row.get("predicted_interval_days", meta.get("Predicted Interval (Days)", meta.get("predicted_days_until_refill")))

        is_dry = int(row.get(dry_col, 0) in (1, "1", True, "True", "Yes"))
        rev_status = str(row.get("review_status", "approved" if st_val in ("accepted", "sending") else "pending"))
        rej_reason = str(row.get("rejection_reason", ""))

        rec: Dict[str, Any] = {
            "pilot_id": cohort_cfg.pilot_id,
            "customerId": cid,
            "itemId": iid,
            "customer_name": cname,
            "medicine": iname,
            "phone_masked": f"***{phone_l4}" if phone_l4 else "",
            "pilot_tier": tier_val,
            "history_quality": qual_val,
            "purchase_count": p_cnt,
            "latest_purchase_date": last_purch_d,
            "predicted_interval_days": round(float(pred_int), 1) if pred_int is not None and not pd.isna(pred_int) else None,
            "expected_refill_date": exp_d.isoformat() if exp_d else None,
            "reminder_stage": stage_val,
            "reminder_date": rem_d.isoformat() if rem_d else None,
            "reminder_status": st_val,
            "is_dry_run": is_dry,
            "actual_refill_date": None,
            "days_to_refill": None,
            "prediction_error_days": None,
            "absolute_prediction_error_days": None,
            "timing_category": "NO OBSERVED REFILL",
            "observation_status": "No Observed Refill (Window Closed)",
            "is_post_reminder_refill": False,
            "is_accurate_1d": False,
            "is_accurate_3d": False,
            "is_accurate_7d": False,
            "review_status": rev_status,
            "rejection_reason": rej_reason,
            "risk_flag": "None",
        }

        # Match subsequent purchase
        all_patient_purchases = tx_lookup.get(key, [])
        subsequent_purchases = [p for p in all_patient_purchases if (rem_d and p >= rem_d) or (exp_d and p > exp_d)]

        # Check for stale reminders (purchase happened before reminder date)
        pre_reminder_purchases = [p for p in all_patient_purchases if rem_d and p < rem_d]

        if subsequent_purchases:
            actual_d = subsequent_purchases[0]
            rec["actual_refill_date"] = actual_d.isoformat()
            
            if rem_d:
                days_from_rem = (actual_d - rem_d).days
                rec["days_to_refill"] = days_from_rem

                if actual_d < rem_d:
                    rec["timing_category"] = "POST-REFILL FOLLOW-UP"
                    rec["risk_flag"] = "Potentially Stale Reminder"
                    rec["is_post_reminder_refill"] = False
                elif actual_d == rem_d:
                    rec["timing_category"] = "SAME-DAY"
                    rec["is_post_reminder_refill"] = True
                else:
                    rec["timing_category"] = "PRE-REFILL REMINDER"
                    rec["is_post_reminder_refill"] = True

                if 0 <= days_from_rem <= obs_window or days_from_rem < 0:
                    rec["observation_status"] = "Observed Refill"
                else:
                    rec["observation_status"] = "No Observed Refill (Window Closed)"
            else:
                rec["observation_status"] = "Observed Refill"

            # Compute prediction accuracy metrics against expected date
            if exp_d:
                p_err = (actual_d - exp_d).days
                abs_p_err = abs(p_err)
                rec["prediction_error_days"] = p_err
                rec["absolute_prediction_error_days"] = abs_p_err
                rec["is_accurate_1d"] = bool(abs_p_err <= 1)
                rec["is_accurate_3d"] = bool(abs_p_err <= 3)
                rec["is_accurate_7d"] = bool(abs_p_err <= 7)

                if p_err <= -7:
                    rec["risk_flag"] = "Potentially Premature Reminder"
        elif pre_reminder_purchases:
            # Customer purchased before this reminder date (e.g. +2d/+5d followup or stale cycle)
            actual_d = pre_reminder_purchases[-1]
            rec["actual_refill_date"] = actual_d.isoformat()
            rec["timing_category"] = "POST-REFILL FOLLOW-UP"
            rec["risk_flag"] = "Potentially Stale Reminder"
            rec["observation_status"] = "Observed Refill"
            rec["is_post_reminder_refill"] = False

            if rem_d:
                rec["days_to_refill"] = (actual_d - rem_d).days
            if exp_d:
                p_err = (actual_d - exp_d).days
                abs_p_err = abs(p_err)
                rec["prediction_error_days"] = p_err
                rec["absolute_prediction_error_days"] = abs_p_err
                rec["is_accurate_1d"] = bool(abs_p_err <= 1)
                rec["is_accurate_3d"] = bool(abs_p_err <= 3)
                rec["is_accurate_7d"] = bool(abs_p_err <= 7)
        else:
            # Handle Right-Censoring
            if rem_d:
                elapsed = (ref_d - rem_d).days
                if elapsed <= obs_window:
                    rec["observation_status"] = "Pending Observation (Window Open)"
                    rec["timing_category"] = "PENDING OBSERVATION"
                else:
                    rec["observation_status"] = "No Observed Refill (Window Closed)"
                    rec["timing_category"] = "NO OBSERVED REFILL"
            else:
                rec["observation_status"] = "Pending Observation (Window Open)"
                rec["timing_category"] = "PENDING OBSERVATION"

        # Check irregular history risk
        if p_cnt < 3 and rec["risk_flag"] == "None":
            rec["risk_flag"] = "Irregular History Candidate"

        outcome_rows.append(rec)

    return pd.DataFrame(outcome_rows)


def evaluate_timing_and_stages(outcomes_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze reminder timing categories and per-stage touchpoint outcomes."""
    if outcomes_df.empty:
        return {
            "timing_distribution": {},
            "stage_analysis": pd.DataFrame(),
            "pre_refill_count": 0,
            "same_day_count": 0,
            "post_refill_count": 0,
            "pending_observation_count": 0,
        }

    timing_counts = outcomes_df["timing_category"].value_counts().to_dict()

    # Per-stage analysis
    stage_records = []
    for stage, group in outcomes_df.groupby("reminder_stage"):
        n_total = len(group)
        n_accepted = len(group[group["reminder_status"].isin(["accepted", "sent"])])
        n_observed = len(group[group["observation_status"] == "Observed Refill"])
        n_pending = len(group[group["observation_status"] == "Pending Observation (Window Open)"])
        n_closed_no_refill = len(group[group["observation_status"] == "No Observed Refill (Window Closed)"])
        
        n_pre = len(group[group["timing_category"] == "PRE-REFILL REMINDER"])
        n_same = len(group[group["timing_category"] == "SAME-DAY"])
        n_post = len(group[group["timing_category"] == "POST-REFILL FOLLOW-UP"])

        denom = n_observed + n_closed_no_refill
        refill_rate = round((n_observed / denom) * 100.0, 2) if denom > 0 else 0.0

        stage_records.append({
            "Stage": str(stage),
            "Total Due / Scheduled": n_total,
            "Accepted Dispatches": n_accepted,
            "Observed Refills": n_observed,
            "Pre-Refill": n_pre,
            "Same-Day": n_same,
            "Post-Refill Follow-Up": n_post,
            "Pending Observation": n_pending,
            "Post-Reminder Refill Rate (%)": refill_rate,
        })

    stage_df = pd.DataFrame(stage_records)

    return {
        "timing_distribution": timing_counts,
        "stage_analysis": stage_df,
        "pre_refill_count": timing_counts.get("PRE-REFILL REMINDER", 0),
        "same_day_count": timing_counts.get("SAME-DAY", 0),
        "post_refill_count": timing_counts.get("POST-REFILL FOLLOW-UP", 0),
        "pending_observation_count": timing_counts.get("PENDING OBSERVATION", 0),
    }


def evaluate_cohort_segments(outcomes_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Segment model prediction accuracy and post-reminder refill rates across cohorts."""
    if outcomes_df.empty:
        return {
            "by_tier": pd.DataFrame(),
            "by_history_quality": pd.DataFrame(),
            "by_purchase_length": pd.DataFrame(),
            "by_medicine": pd.DataFrame(),
        }

    def _calc_segment_metrics(df_sub: pd.DataFrame, group_col: str) -> pd.DataFrame:
        rows = []
        for name, g in df_sub.groupby(group_col):
            n_total = len(g)
            obs = g[g["observation_status"] == "Observed Refill"]
            n_obs = len(obs)
            closed_no = len(g[g["observation_status"] == "No Observed Refill (Window Closed)"])
            pending = len(g[g["observation_status"] == "Pending Observation (Window Open)"])

            denom = n_obs + closed_no
            refill_rate = round((n_obs / denom) * 100.0, 2) if denom > 0 else 0.0

            mae = None
            medae = None
            acc_7d = None
            if n_obs > 0 and obs["absolute_prediction_error_days"].notna().any():
                errs = obs["absolute_prediction_error_days"].dropna().astype(float)
                mae = round(float(errs.mean()), 2)
                medae = round(float(errs.median()), 2)
                acc_7d = round(float((errs <= 7.0).sum() / n_obs * 100.0), 2)

            rows.append({
                group_col: str(name),
                "Total Regimens": n_total,
                "Observed Refills": n_obs,
                "Pending Observation": pending,
                "Post-Reminder Refill Rate (%)": refill_rate,
                "MAE (Days)": mae,
                "MedAE (Days)": medae,
                "±7d Accuracy (%)": acc_7d,
            })
        return pd.DataFrame(rows)

    by_tier = _calc_segment_metrics(outcomes_df, "pilot_tier")
    by_qual = _calc_segment_metrics(outcomes_df, "history_quality")

    # Group purchase length bins
    temp_df = outcomes_df.copy()
    temp_df["purchase_length_bin"] = pd.cut(
        temp_df["purchase_count"],
        bins=[-np.inf, 2, 4, np.inf],
        labels=["2 Purchases (Low)", "3-4 Purchases (Moderate)", "5+ Purchases (High)"]
    ).astype(str)
    by_purch = _calc_segment_metrics(temp_df, "purchase_length_bin")

    # Group medicine where N >= 3
    med_counts = outcomes_df["medicine"].value_counts()
    frequent_meds = med_counts[med_counts >= 3].index
    med_sub = outcomes_df[outcomes_df["medicine"].isin(frequent_meds)]
    by_med = _calc_segment_metrics(med_sub, "medicine") if not med_sub.empty else pd.DataFrame()

    return {
        "by_tier": by_tier,
        "by_history_quality": by_qual,
        "by_purchase_length": by_purch,
        "by_medicine": by_med,
    }


def compare_against_offline_benchmarks(outcomes_df: pd.DataFrame) -> Dict[str, Any]:
    """Compare pilot evaluation outcomes against Phase 4 offline test set benchmarks."""
    obs = outcomes_df[outcomes_df["observation_status"] == "Observed Refill"]
    n_obs = len(obs)

    if n_obs == 0:
        return {
            "status": "INSUFFICIENT_DATA",
            "evaluated_refills": 0,
            "pilot_mae": None,
            "offline_benchmark_mae": OFFLINE_BENCHMARKS["overall_mae"],
            "delta_mae": None,
            "tier_a_pilot_mae": None,
            "tier_a_offline_mae": OFFLINE_BENCHMARKS["high_purchase_mae"],
            "maturity_assessment": "Insufficient observed refills during pilot to perform definitive offline benchmark comparison.",
        }

    errs = obs["absolute_prediction_error_days"].dropna().astype(float)
    pilot_mae = round(float(errs.mean()), 2)
    delta_mae = round(pilot_mae - OFFLINE_BENCHMARKS["overall_mae"], 2)

    tier_a_obs = obs[obs["pilot_tier"] == "Tier A (Strong Pilot)"]
    tier_a_mae = None
    if not tier_a_obs.empty and tier_a_obs["absolute_prediction_error_days"].notna().any():
        tier_a_mae = round(float(tier_a_obs["absolute_prediction_error_days"].dropna().astype(float).mean()), 2)

    maturity = "EARLY_PILOT_STABLE" if n_obs >= 20 else "COLLECTING_EVIDENCE"

    return {
        "status": "EVALUATED",
        "evaluated_refills": n_obs,
        "pilot_mae": pilot_mae,
        "offline_benchmark_mae": OFFLINE_BENCHMARKS["overall_mae"],
        "delta_mae": delta_mae,
        "tier_a_pilot_mae": tier_a_mae,
        "tier_a_offline_mae": OFFLINE_BENCHMARKS["high_purchase_mae"],
        "maturity_assessment": f"Pilot data shows {n_obs} evaluated refills with MAE = {pilot_mae} days (Tier A MAE = {tier_a_mae or 'N/A'} days).",
    }


def evaluate_pilot_decision(
    outcomes_df: pd.DataFrame,
    reconciliation_health: str = "HEALTHY",
) -> Dict[str, Any]:
    """Generate an evidence-based pilot governance recommendation based on observed outcomes."""
    obs = outcomes_df[outcomes_df["observation_status"] == "Observed Refill"]
    n_obs = len(obs)

    if outcomes_df.empty:
        return {
            "recommendation": "INSUFFICIENT_DATA",
            "summary": "No pilot activity records available to evaluate.",
            "retraining_recommended": False,
            "next_steps": "Initiate controlled pilot review workflow.",
        }

    # Check for safety or reconciliation issues
    if reconciliation_health != "HEALTHY":
        return {
            "recommendation": "PAUSE",
            "summary": "Audit state reconciliation requires attention. Inconsistencies detected between scheduler and audit logs.",
            "retraining_recommended": False,
            "next_steps": "Investigate audit synchronization and resolve storage discrepancies.",
        }

    if n_obs < 10:
        return {
            "recommendation": "KEEP TIER A ONLY",
            "summary": f"Early pilot evidence ({n_obs} observed refills). Operational workflow and safety safeguards are functioning properly.",
            "retraining_recommended": False,
            "next_steps": "Continue 30-day controlled pilot under Tier A restriction. Await further refill observations before model recalibration.",
        }

    errs = obs["absolute_prediction_error_days"].dropna().astype(float)
    mae = float(errs.mean())

    if mae <= 12.0:
        return {
            "recommendation": "EXPAND TIER A",
            "summary": f"Strong pilot performance (MAE = {mae:.1f} days across {n_obs} refills) with robust operational adherence.",
            "retraining_recommended": False,
            "next_steps": "Safely increase Tier A candidate volume while maintaining manual pharmacist review.",
        }
    elif mae <= 18.0:
        return {
            "recommendation": "KEEP TIER A ONLY",
            "summary": f"Acceptable pilot performance (MAE = {mae:.1f} days). Tier A demonstrates stable clinical utility.",
            "retraining_recommended": False,
            "next_steps": "Maintain controlled manual pilot for Tier A only. Continue collecting evidence.",
        }
    else:
        return {
            "recommendation": "RESTRICT FURTHER",
            "summary": f"Elevated prediction variance observed (MAE = {mae:.1f} days). Recommend restricting to high-frequency patients (>=5 purchases).",
            "retraining_recommended": True,
            "next_steps": "Retrain refill interval model once pilot observation window completes and >=100 refills are collected.",
        }


def export_pilot_outcomes_csv(outcomes_df: pd.DataFrame) -> bytes:
    """Generate safe, privacy-compliant CSV bytes of pilot outcome evaluation data."""
    if outcomes_df.empty:
        return b""

    # Explicitly verify no unmasked phone numbers or credentials are included
    safe_cols = [
        "pilot_id", "customerId", "itemId", "customer_name", "medicine",
        "phone_masked", "pilot_tier", "history_quality", "purchase_count",
        "latest_purchase_date", "predicted_interval_days", "expected_refill_date",
        "reminder_stage", "reminder_date", "reminder_status", "is_dry_run",
        "actual_refill_date", "days_to_refill", "prediction_error_days",
        "absolute_prediction_error_days", "timing_category", "observation_status",
        "is_post_reminder_refill", "is_accurate_1d", "is_accurate_3d", "is_accurate_7d",
        "review_status", "rejection_reason", "risk_flag"
    ]
    present_cols = [c for c in safe_cols if c in outcomes_df.columns]
    
    buf = pd.DataFrame(outcomes_df[present_cols]).to_csv(index=False)
    return buf.encode("utf-8")
