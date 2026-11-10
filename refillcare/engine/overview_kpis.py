"""Pharmacy Overview numbers for ONE store and THIS month (shared by the Streamlit app and the API).

Everything is counted from the store's own database at request time - no fixed numbers:

* customers_monitored            - distinct patients with a customer sale in this store
                                   (``history_store.store_summary`` - same number in every tab and the API)
* first_sales_date / latest_sales_date / sales_lines_stored - the store's sales data on file
* reminder_messages_month        - reminder messages (stages) scheduled this month (send rule applied:
                                   rules-chronic or pharmacist-added medicines); one patient-medicine
                                   has up to 7 messages per cycle, so this is larger than the patient count
* reminders_due_today            - messages scheduled for today
* patients_needing_review_today  - patients not getting reminders (chronic / not classified medicine)
                                   whose refill is due today (same list as "Patients Needing Review")
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

OPEN_STAGE_STATUSES = ("PENDING", "DUE", "APPROVED")


def month_bounds(today: date) -> tuple:
    start = today.replace(day=1)
    nxt = date(start.year + (start.month == 12), start.month % 12 + 1, 1)
    return start, nxt


def overview_kpis(db: Session, today: Optional[date] = None) -> Dict[str, Any]:
    from database.models import RefillDecisionModel, ReminderCycleModel, ReminderStageModel
    from refillcare.engine.pharmacist_overrides import review_candidates

    today = today or date.today()
    m_start, m_next = month_bounds(today)

    from refillcare.data.history_store import store_summary

    summary = store_summary(db)

    from refillcare.engine.persistence import medicine_can_be_sent

    rows = (
        db.query(ReminderStageModel.customer_id, ReminderStageModel.target_send_date,
                 RefillDecisionModel.item_name, RefillDecisionModel.path)
        .join(ReminderCycleModel, ReminderStageModel.cycle_id == ReminderCycleModel.cycle_id)
        .join(RefillDecisionModel, ReminderCycleModel.decision_id == RefillDecisionModel.decision_id)
        .filter(
            ReminderStageModel.status.in_(OPEN_STAGE_STATUSES),
            ReminderCycleModel.is_active.is_(True),
            RefillDecisionModel.is_eligible.is_(True),
            ReminderStageModel.target_send_date >= min(m_start, today),
            ReminderStageModel.target_send_date < m_next,
        )
        .all()
    )
    sendable: Dict[tuple, bool] = {}
    messages_month, due_today = 0, 0
    for cust, send_date, item, path in rows:
        ok = sendable.get((item, path))
        if ok is None:
            ok = sendable[(item, path)] = medicine_can_be_sent(item, path)  # same send rule as the queue
        if not ok:
            continue
        if send_date >= m_start:
            messages_month += 1
        if send_date == today:
            due_today += 1

    review = review_candidates(db, today, today)
    review_patients = len({r["customer_id"] for r in review})

    return {
        "month": m_start.strftime("%Y-%m"),
        "month_label": m_start.strftime("%B %Y"),
        "customers_monitored": summary["customers_monitored"],
        "all_time_customers": summary.get("all_time_customers", summary["customers_monitored"]),
        "active_patients": summary.get("active_patients", summary["customers_monitored"]),
        "active_patients_label": summary.get("active_patients_label", "Active patients (last 12 months)"),
        "active_patients_subtext": summary.get("active_patients_subtext", f"All-time: {summary['customers_monitored']:,}"),
        "has_12m_history": summary.get("has_12m_history", True),
        "history_span_months": summary.get("history_span_months", 12),
        "history_span_days": summary.get("history_span_days", 365),
        "first_sales_date": summary["first_sales_date"],
        "latest_sales_date": summary["latest_sales_date"],
        "sales_lines_stored": summary["sales_lines_stored"],
        "reminder_messages_month": messages_month,
        "reminders_due_today": due_today,
        "patients_needing_review_today": review_patients,
    }
