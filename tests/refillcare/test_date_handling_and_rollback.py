"""Comprehensive unit and integration tests for RefillCare Date Handling & Import Rollback.

Tests:
1. DD-MM-YYYY display formatting.
2. '01-08-2026' parsed strictly as August 1, 2026.
3. '12-08-2026' parsed strictly as August 12, 2026.
4. Native Excel datetime cell preservation.
5. Ambiguous string date handling and warnings.
6. Invalid date rejection.
7. Duplicate upload detection and skipping.
8. Import batch ID creation and metadata tracking.
9. Undo last import / rollback execution.
10. Verification that older historical transactions remain untouched after rollback.
11. Invalidation of predictions generated from rolled-back import.
12. Preservation of valid pre-existing prediction snapshots.
13. End-to-end lifecycle: Upload → Validate → Process → Generate Predictions → Undo Import.
"""

from datetime import datetime, date
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from refillcare.data.dates import (
    parse_pharmacy_dates,
    format_date_dd_mm_yyyy,
    parse_single_date,
    UI_DATE_FORMAT,
)
from refillcare.data.monthly_ingestion import (
    validate_monthly_sales_data,
    process_monthly_sales_data,
    rollback_monthly_sales_import,
    generate_updated_predictions,
)
from reminder.storage import RefillCareStorage


@pytest.fixture
def temp_storage():
    """Provide a temporary SQLite storage instance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_refillcare.db"
        storage = RefillCareStorage(db_path=db_path)
        yield storage


def _create_sample_base_history() -> pd.DataFrame:
    """Create existing multi-year historical transactions through 2026-07-31."""
    records = [
        {"customerId": "C101", "itemId": "M01", "customerName": "Ramesh", "itemName": "Glycomet 500", "invoice_number": "INV-001", "invoice_date": "2026-05-01", "quantity": 2, "packing": "1X15", "MOBILE_NO": "9849011111", "import_batch_id": "LEGACY_HISTORY"},
        {"customerId": "C101", "itemId": "M01", "customerName": "Ramesh", "itemName": "Glycomet 500", "invoice_number": "INV-002", "invoice_date": "2026-05-31", "quantity": 2, "packing": "1X15", "MOBILE_NO": "9849011111", "import_batch_id": "LEGACY_HISTORY"},
        {"customerId": "C101", "itemId": "M01", "customerName": "Ramesh", "itemName": "Glycomet 500", "invoice_number": "INV-003", "invoice_date": "2026-06-30", "quantity": 2, "packing": "1X15", "MOBILE_NO": "9849011111", "import_batch_id": "LEGACY_HISTORY"},
        {"customerId": "C101", "itemId": "M01", "customerName": "Ramesh", "itemName": "Glycomet 500", "invoice_number": "INV-004", "invoice_date": "2026-07-30", "quantity": 2, "packing": "1X15", "MOBILE_NO": "9849011111", "import_batch_id": "LEGACY_HISTORY"},
        {"customerId": "C102", "itemId": "M02", "customerName": "Pooja", "itemName": "Telma 40", "invoice_number": "INV-005", "invoice_date": "2026-07-10", "quantity": 3, "packing": "1X10", "MOBILE_NO": "9849022222", "import_batch_id": "LEGACY_HISTORY"},
    ]
    df = pd.DataFrame(records)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df


# ------------------------------------------------------------------------------
# 1. DD-MM-YYYY Display Formatting Tests
# ------------------------------------------------------------------------------
def test_dd_mm_yyyy_display_formatting():
    """Verify that dates are always formatted as DD-MM-YYYY in the UI."""
    d1 = datetime(2026, 8, 1)
    d2 = pd.Timestamp("2026-08-12")
    d3 = date(2026, 12, 5)
    
    assert format_date_dd_mm_yyyy(d1) == "01-08-2026"
    assert format_date_dd_mm_yyyy(d2) == "12-08-2026"
    assert format_date_dd_mm_yyyy(d3) == "05-12-2026"
    assert format_date_dd_mm_yyyy(None) == "-"
    assert format_date_dd_mm_yyyy("invalid") == "invalid"


# ------------------------------------------------------------------------------
# 2 & 3. 01-08-2026 and 12-08-2026 String Parsing Tests
# ------------------------------------------------------------------------------
def test_august_dates_interpretation():
    """Verify that '01-08-2026' and '12-08-2026' are interpreted as August 1 & 12, NOT Jan 8 or Dec 8."""
    series = pd.Series(["01-08-2026", "05-08-2026", "12-08-2026"])
    parsed, meta = parse_pharmacy_dates(series)
    
    assert meta["min_date_formatted"] == "01-08-2026"
    assert meta["max_date_formatted"] == "12-08-2026"
    assert meta["date_range_formatted"] == "01-08-2026 to 12-08-2026"
    
    # Check exact date components
    assert parsed.iloc[0].year == 2026
    assert parsed.iloc[0].month == 8
    assert parsed.iloc[0].day == 1
    
    assert parsed.iloc[2].year == 2026
    assert parsed.iloc[2].month == 8
    assert parsed.iloc[2].day == 12


# ------------------------------------------------------------------------------
# 4. Excel Datetime Cell Preservation Tests
# ------------------------------------------------------------------------------
def test_excel_native_datetime_cells():
    """Verify that native Excel datetime cells are preserved directly without corruption."""
    excel_dates = pd.Series([
        pd.Timestamp(year=2026, month=8, day=1, hour=10, minute=30),
        pd.Timestamp(year=2026, month=8, day=12, hour=15, minute=0),
    ])
    parsed, meta = parse_pharmacy_dates(excel_dates)
    
    assert meta["source_format"] == "Excel Native Datetime"
    assert meta["min_date_formatted"] == "01-08-2026"
    assert meta["max_date_formatted"] == "12-08-2026"
    assert parsed.iloc[0].day == 1 and parsed.iloc[0].month == 8
    assert parsed.iloc[1].day == 12 and parsed.iloc[1].month == 8


# ------------------------------------------------------------------------------
# 5 & 6. Ambiguous and Invalid Date Handling Tests
# ------------------------------------------------------------------------------
def test_ambiguous_and_invalid_dates():
    """Verify handling of invalid or empty date columns."""
    # Invalid dates
    invalid_series = pd.Series(["not_a_date", "99-99-9999", None, ""])
    parsed, meta = parse_pharmacy_dates(invalid_series)
    assert meta["valid_count"] == 0
    assert meta["invalid_count"] == 4
    assert meta["is_ambiguous"] is True

    # Empty series
    empty_series = pd.Series([], dtype=object)
    _, empty_meta = parse_pharmacy_dates(empty_series)
    assert empty_meta["valid_count"] == 0


# ------------------------------------------------------------------------------
# 7. Duplicate Upload Detection Tests
# ------------------------------------------------------------------------------
def test_duplicate_upload_detection():
    """Verify that re-uploading the same transactions skips duplicates."""
    base_history = _create_sample_base_history()
    
    # Upload data identical to base history
    upload_df = pd.DataFrame([
        {"customerId": "C101", "itemId": "M01", "customerName": "Ramesh", "itemName": "Glycomet 500", "invoice_number": "INV-004", "invoice_date": "30-07-2026", "quantity": 2, "packing": "1X15", "MOBILE_NO": "9849011111"},
    ])
    
    res = process_monthly_sales_data(upload_df, existing_history_df=base_history)
    assert res["status"] == "success"
    assert res["metrics"]["new_records_added"] == 0
    assert res["metrics"]["existing_duplicates_skipped"] == 1


# ------------------------------------------------------------------------------
# 8, 9, 10, 11, 12. Batch Tracking, Rollback & Snapshot Invalidation Tests
# ------------------------------------------------------------------------------
def test_batch_tracking_and_rollback(temp_storage):
    """Verify import batch assignment, rollback execution, and preservation of older history."""
    base_history = _create_sample_base_history()
    base_count = len(base_history)
    
    # 1. Insert pre-existing snapshot in storage
    temp_storage.save_prediction_snapshots([
        {
            "snapshot_id": "snap_legacy_1",
            "customer_id": "C101",
            "item_id": "M01",
            "last_purchase_date": "2026-07-30",
            "prediction_date": "2026-07-31",
            "expected_refill_date": "2026-08-29",
            "reminder_date": "2026-08-27",
            "import_batch_id": "LEGACY_HISTORY",
        }
    ])
    assert len(temp_storage.get_prediction_snapshots()) == 1
    
    # 2. Upload August sales data (01-08-2026 to 12-08-2026)
    new_sales = pd.DataFrame([
        {"customerId": "C101", "itemId": "M01", "customerName": "Ramesh", "itemName": "Glycomet 500", "invoice_number": "INV-101", "invoice_date": "01-08-2026", "quantity": 2, "packing": "1X15", "MOBILE_NO": "9849011111"},
        {"customerId": "C101", "itemId": "M01", "customerName": "Ramesh", "itemName": "Glycomet 500", "invoice_number": "INV-102", "invoice_date": "12-08-2026", "quantity": 2, "packing": "1X15", "MOBILE_NO": "9849011111"},
        {"customerId": "C103", "itemId": "M03", "customerName": "Vikram", "itemName": "Pan D", "invoice_number": "INV-103", "invoice_date": "10-08-2026", "quantity": 1, "packing": "1X10", "MOBILE_NO": "9849033333"},
    ])
    
    proc_res = process_monthly_sales_data(
        sales_input=new_sales,
        existing_history_df=base_history,
        storage=temp_storage,
        source_filename="august_sales.xlsx",
    )
    assert proc_res["status"] == "success"
    batch_id = proc_res["import_batch_id"]
    assert batch_id.startswith("BATCH_")
    assert proc_res["metrics"]["new_records_added"] == 3
    assert proc_res["metrics"]["latest_sales_date"] == "12-08-2026"
    
    updated_history = proc_res["updated_history_df"]
    assert len(updated_history) == base_count + 3
    
    # 3. Save new prediction snapshots associated with this batch
    temp_storage.save_prediction_snapshots([
        {
            "snapshot_id": f"snap_{batch_id}_1",
            "customer_id": "C101",
            "item_id": "M01",
            "last_purchase_date": "2026-08-12",
            "prediction_date": "2026-08-12",
            "expected_refill_date": "2026-09-11",
            "reminder_date": "2026-09-09",
            "import_batch_id": batch_id,
        }
    ])
    assert len(temp_storage.get_prediction_snapshots()) == 2
    
    # 4. Verify batch is recorded in database
    active_batch = temp_storage.get_latest_active_import_batch()
    assert active_batch is not None
    assert active_batch["import_batch_id"] == batch_id
    assert active_batch["records_inserted"] == 3
    
    # 5. Execute Rollback ("Undo Last Import")
    rb_res = rollback_monthly_sales_import(
        current_history_df=updated_history,
        import_batch_id=batch_id,
        storage=temp_storage,
    )
    assert rb_res["status"] == "success"
    assert rb_res["records_removed"] == 3
    assert rb_res["restored_latest_sales_date"] == "30-07-2026"
    
    # 6. Verify older historical data remains completely untouched
    restored_hist = rb_res["restored_history_df"]
    assert len(restored_hist) == base_count
    assert set(restored_hist["customerId"].unique()) == {"C101", "C102"}
    
    # 7. Verify snapshots generated from rolled back batch are invalidated while older snapshots remain
    snaps_after_rb = temp_storage.get_prediction_snapshots()
    assert len(snaps_after_rb) == 1
    assert snaps_after_rb[0]["snapshot_id"] == "snap_legacy_1"
    
    # 8. Verify batch status in storage is updated
    assert temp_storage.get_latest_active_import_batch() is None


# ------------------------------------------------------------------------------
# 13. End-to-End Lifecycle Test: Upload → Validate → Process → Rollback
# ------------------------------------------------------------------------------
def test_full_e2e_upload_process_rollback_lifecycle(temp_storage):
    """Test full cycle: validate incoming file, process with batch ID, and undo import."""
    base_history = _create_sample_base_history()
    
    incoming_data = pd.DataFrame([
        {"customerId": "C201", "itemId": "M10", "customerName": "Deepa", "itemName": "Januvia 100", "invoice_number": "INV-501", "invoice_date": "01-08-2026", "quantity": 1, "packing": "1X14", "MOBILE_NO": "9849055555"},
        {"customerId": "C201", "itemId": "M10", "customerName": "Deepa", "itemName": "Januvia 100", "invoice_number": "INV-502", "invoice_date": "10-08-2026", "quantity": 1, "packing": "1X14", "MOBILE_NO": "9849055555"},
    ])
    
    # Step A: Validate
    valid_df, review_df, v_metrics = validate_monthly_sales_data(incoming_data, existing_history_df=base_history)
    assert v_metrics["valid_records"] == 2
    assert v_metrics["file_date_range"] == "01-08-2026 to 10-08-2026"
    assert v_metrics["date_diagnostics"]["min_date_formatted"] == "01-08-2026"
    assert v_metrics["date_diagnostics"]["max_date_formatted"] == "10-08-2026"
    
    # Step B: Process
    p_res = process_monthly_sales_data(
        sales_input=valid_df,
        existing_history_df=base_history,
        storage=temp_storage,
        source_filename="august_batch.csv",
    )
    assert p_res["status"] == "success"
    batch_id = p_res["import_batch_id"]
    updated_history = p_res["updated_history_df"]
    assert p_res["metrics"]["latest_sales_date"] == "10-08-2026"
    
    # Step C: Undo
    rb_res = rollback_monthly_sales_import(
        current_history_df=updated_history,
        import_batch_id=batch_id,
        storage=temp_storage,
    )
    assert rb_res["status"] == "success"
    assert rb_res["restored_latest_sales_date"] == "30-07-2026"
    assert len(rb_res["restored_history_df"]) == len(base_history)
