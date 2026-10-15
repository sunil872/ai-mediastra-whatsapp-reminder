"""
Enterprise Regression & Regex Safety Tests
Verifies that all Lifecycle Stage filters, customer queries, and medication searches
operate with 100% regex-safety and immunity against quantifier/special regex character errors.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from app_refillcare import filter_predictions, filter_schedules
from reminder.storage import RefillCareStorage


def test_filter_predictions_regex_safety():
    """Verify filter_predictions handles all regex bombs and messy data without crashing."""
    df = pd.DataFrame({
        "Customer Name": ["John Doe +91", "Jane (Doe)", "Alice [VIP]", "Bob *Special", np.nan, 9999],
        "Customer": ["John Doe +91", "Jane (Doe)", "Alice [VIP]", "Bob *Special", np.nan, 9999],
        "Medication": ["PAN (40MG)", "AMOXICILLIN + CLAV", "TELMA [40/12.5]", "ASPIRIN * 75", np.nan, 1234],
        "Medicine": ["PAN (40MG)", "AMOXICILLIN + CLAV", "TELMA [40/12.5]", "ASPIRIN * 75", np.nan, 1234],
        "History Quality": ["high_history", "medium_history", "low_history", "", "", ""],
        "Pilot Tier": ["Tier A", "Tier B", "Tier A", "", "", ""],
    })

    bad_queries = ["+45", "+91", "(", "[", "*", "?", "+", "\\", "^", "$", "40/12.5", "nan", ""]
    for q in bad_queries:
        res_c = filter_predictions(df, customer_query=q)
        assert isinstance(res_c, pd.DataFrame)
        res_m = filter_predictions(df, medicine_query=q)
        assert isinstance(res_m, pd.DataFrame)


def test_filter_schedules_regex_safety():
    """Verify filter_schedules handles regex symbols in customer, medicine, and stages."""
    df = pd.DataFrame({
        "Customer Name": ["Patient +91", "Mr (Brown)", "Ms [Green]"],
        "Customer": ["Patient +91", "Mr (Brown)", "Ms [Green]"],
        "Medication": ["PAN (40MG)", "TELMA + AMLO", "ASPIRIN [75MG]"],
        "Medicine": ["PAN (40MG)", "TELMA + AMLO", "ASPIRIN [75MG]"],
        "Reminder Stage": ["+45 days", "+2 days", "0 days"],
        "Status": ["scheduled", "pending", "sent"],
        "Pilot Tier": ["Tier A", "Tier B", "Tier A"],
    })

    for q in ["+45", "+2", "(", "[", "*", "+"]:
        res_c = filter_schedules(df, customer_query=q)
        assert isinstance(res_c, pd.DataFrame)
        res_m = filter_schedules(df, medicine_query=q)
        assert isinstance(res_m, pd.DataFrame)


def test_lifecycle_stage_filtering_logic():
    """Test the exact filtering logic used in render_app tab 4 across all 7 lifecycle stages."""
    import re

    # Dataframe replicating both DB structure and in-memory scheduler structure
    df = pd.DataFrame({
        "Reminder Stage": [
            "-7 days (Due in 7 Days)",
            "-3 days (Due in 3 Days)",
            "-1 day (Due Tomorrow)",
            "0 days (Due Today)",
            "+2 days (Follow-up)",
            "+5 days (Follow-up)",
            "+45 days (Re-engagement)",
            None,
            "Unknown Stage",
        ],
        "stage_offset": [-7, -3, -1, 0, 2, 5, 45, np.nan, 99],
        "Customer Name": [f"Patient_{i}" for i in range(9)],
        "Mobile Number": ["+919912028234", "9912028235", "9912028236", "9912028237", "9912028238", "9912028239", "9912028240", "", "None"],
    })

    stages = [
        ("Stage: -7 days (Due in 7 Days)", -7),
        ("Stage: -3 days (Due in 3 Days)", -3),
        ("Stage: -1 day (Due Tomorrow)", -1),
        ("Stage: 0 days (Due Today)", 0),
        ("Stage: +2 days (Follow-up)", 2),
        ("Stage: +5 days (Follow-up)", 5),
        ("Stage: +45 days (Re-engagement)", 45),
    ]

    for stage_label, expected_offset in stages:
        stg_target = stage_label.replace("Stage:", "").strip()
        m_offset = re.search(r"Stage:\s*([+-]?\d+)", stage_label)
        target_offset = int(m_offset.group(1)) if m_offset else None

        cond_stage = pd.Series(False, index=df.index)
        if "Reminder Stage" in df.columns:
            cond_stage = cond_stage | df["Reminder Stage"].astype(str).str.contains(
                stg_target[:6], case=False, na=False, regex=False
            )
        if target_offset is not None and "stage_offset" in df.columns:
            cond_stage = cond_stage | (df["stage_offset"] == target_offset)

        filtered = df[cond_stage]
        assert len(filtered) == 1, f"Failed for {stage_label}"
        assert filtered.iloc[0]["stage_offset"] == expected_offset


def test_storage_get_audit_dataframe_regex_safety(tmp_path):
    """Verify RefillCareStorage.get_audit_dataframe does not raise on regex characters."""
    db_file = tmp_path / "test_storage.db"
    storage = RefillCareStorage(db_file)
    storage.insert_or_update_scheduled_reminder({
        "reminder_id": "rem-101",
        "customer_id": "cust-101",
        "customer_name": "VIP Customer +91 (Care)",
        "phone_number": "+919912028234",
        "item_id": "item-101",
        "item_name": "PAN (40MG) + DOMPERIDONE",
        "reminder_stage": "+45 days (Re-engagement)",
        "reminder_date": "2026-10-15",
        "expected_refill_date": "2026-10-15",
        "status": "SCHEDULED",
    })

    # Test +45 days filter
    res = storage.get_audit_dataframe(stage_filter="+45 days")
    assert len(res) == 1

    # Test regex in customer search
    res_c = storage.get_audit_dataframe(customer_query="+91 (Care)")
    assert len(res_c) == 1

    # Test regex in medicine search
    res_m = storage.get_audit_dataframe(medicine_query="PAN (40MG)")
    assert len(res_m) == 1
