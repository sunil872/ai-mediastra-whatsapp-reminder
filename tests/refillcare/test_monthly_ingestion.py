"""Unit and integration tests for the RefillCare Monthly Sales Ingestion Production Workflow."""

import io
from pathlib import Path
from datetime import date, datetime, timedelta
import pytest
import pandas as pd
import numpy as np

from refillcare.data.monthly_ingestion import (
    generate_updated_predictions,
    validate_monthly_sales_data,
    ingest_monthly_sales_pipeline,
    process_monthly_sales_data,
    normalize_sales_dataframe,
    determine_mobile_status,
    evaluate_prediction_outcomes,
)
from app_refillcare import load_refill_model, load_purchase_history, build_export_csv
from reminder.storage import RefillCareStorage


@pytest.fixture
def temp_storage(tmp_path):
    """Provide an isolated, temporary SQLite database for testing snapshots and outcomes."""
    db_file = tmp_path / "test_refillcare_ingestion.db"
    return RefillCareStorage(db_path=db_file)


def test_normalize_sales_dataframe():
    """Verify standardizing different column naming conventions."""
    raw_df = pd.DataFrame({
        "Patient Code": ["C001"],
        "Patient Name": ["Rahul Verma"],
        "Drug ID": ["I101"],
        "Drug Name": ["Metformin 500mg"],
        "Bill Date": ["2026-08-01"],
        "Qty Ordered": [2],
        "Pack Size": ["1X15"],
        "Contact": ["9876543210"],
    })
    norm = normalize_sales_dataframe(raw_df)

    assert "customerId" in norm.columns
    assert "customerName" in norm.columns
    assert "itemId" in norm.columns
    assert "itemName" in norm.columns
    assert "invoice_date" in norm.columns
    assert "quantity" in norm.columns
    assert "packing" in norm.columns
    assert "MOBILE_NO" in norm.columns
    assert "invoice_number" in norm.columns


def test_validate_monthly_sales_data_partitioning():
    """Verify splitting into valid records vs review records and computing business metrics."""
    data = {
        "invoice_date": ["2026-08-01", "2026-08-05", "invalid_date", "2026-08-10", "2026-08-12"],
        "customerId": ["C1", "C2", "C3", "", "C5"],
        "customerName": ["Alice", "Bob", "Charlie", "David", "Eva"],
        "itemId": ["I1", "I2", "I3", "I4", ""],
        "itemName": ["Drug A", "Drug B", "Drug C", "Drug D", "Drug E"],
        "quantity": [2, 1, 3, 2, -1],
        "MOBILE_NO": ["9876543210", "", "12345", "919876543210", "8888888888"],
    }
    df = pd.DataFrame(data)

    existing_history = pd.DataFrame({
        "customerId": ["C1"],
        "customerName": ["Alice"],
    })

    valid_df, review_df, metrics = validate_monthly_sales_data(df, existing_history_df=existing_history)

    assert len(valid_df) == 2  # C1 and C2 are valid
    assert len(review_df) == 3  # C3 (bad date), C4 (missing customerId), C5 (missing itemId & negative qty)
    assert metrics["records_received"] == 5
    assert metrics["valid_records"] == 2
    assert metrics["records_requiring_review"] == 3
    assert metrics["new_customers"] >= 1  # C2 is new
    assert metrics["existing_customers"] == 1  # C1 exists
    assert "01-08-2026 to 12-08-2026" in metrics["file_date_range"]


def test_history_retention_and_deduplication(temp_storage):
    """Verify that existing multi-year history is NEVER replaced and incoming duplicates are handled."""
    model_bundle = load_refill_model()
    assert model_bundle is not None

    # Construct realistic historical base (representing 5.8-year dataset)
    existing_history = pd.DataFrame([
        {
            "customerId": "CUST_001",
            "customerName": "Ramesh Kumar",
            "itemId": "ITEM_001",
            "itemName": "Telmisartan 40mg",
            "invoice_date": pd.Timestamp("2026-01-10"),
            "invoice_number": "INV_0001",
            "quantity": 2,
            "packing": "1X15",
            "MOBILE_NO": "9876543210",
        },
        {
            "customerId": "CUST_001",
            "customerName": "Ramesh Kumar",
            "itemId": "ITEM_001",
            "itemName": "Telmisartan 40mg",
            "invoice_date": pd.Timestamp("2026-02-10"),
            "invoice_number": "INV_0002",
            "quantity": 2,
            "packing": "1X15",
            "MOBILE_NO": "9876543210",
        },
        {
            "customerId": "CUST_001",
            "customerName": "Ramesh Kumar",
            "itemId": "ITEM_001",
            "itemName": "Telmisartan 40mg",
            "invoice_date": pd.Timestamp("2026-03-12"),
            "invoice_number": "INV_0003",
            "quantity": 2,
            "packing": "1X15",
            "MOBILE_NO": "9876543210",
        },
    ])

    # Construct new month's sales file with 1 duplicate of previous and 1 new purchase
    new_sales_month1 = pd.DataFrame([
        # Duplicate of previous transaction (should be deduplicated)
        {
            "customerId": "CUST_001",
            "customerName": "Ramesh Kumar",
            "itemId": "ITEM_001",
            "itemName": "Telmisartan 40mg",
            "invoice_date": "2026-03-12",
            "invoice_number": "INV_0003",
            "quantity": 2,
            "packing": "1X15",
            "MOBILE_NO": "9876543210",
        },
        # New transaction for April
        {
            "customerId": "CUST_001",
            "customerName": "Ramesh Kumar",
            "itemId": "ITEM_001",
            "itemName": "Telmisartan 40mg",
            "invoice_date": "2026-04-14",
            "invoice_number": "INV_0004",
            "quantity": 2,
            "packing": "1X15",
            "MOBILE_NO": "9876543210",
        },
    ])

    result = ingest_monthly_sales_pipeline(
        new_sales=new_sales_month1,
        existing_history_df=existing_history,
        model_bundle=model_bundle,
        storage=temp_storage,
        prediction_date="2026-04-15",
    )

    assert result["status"] == "success"
    combined_history = result["combined_history_df"]

    # Original 3 history records + 1 new (deduplicated) = 4 records
    assert len(combined_history) == 4
    # Verify chronological sequence
    assert list(combined_history["purchase_seq"]) == [1, 2, 3, 4]
    assert combined_history["total_purchases"].iloc[-1] == 4

    # Verify that eligible refill list includes updated prediction
    el_df = result["eligible_df"]
    assert not el_df.empty
    cust_row = el_df[el_df["customerId"] == "CUST_001"].iloc[0]
    assert cust_row["Last Purchase Date"] == "2026-04-14"
    assert pd.notna(cust_row["Estimated Days of Supply"])
    assert cust_row["Expected Refill Date"] > "2026-04-14"
    assert cust_row["Reminder Date"] < cust_row["Expected Refill Date"]


def test_prediction_snapshots_non_overwriting_and_unseen_data(temp_storage):
    """Verify that prediction snapshots are recorded without overwriting past history, and treated as unseen."""
    model_bundle = load_refill_model()

    base_history = pd.DataFrame([
        {
            "customerId": "CUST_A",
            "customerName": "Anil Shah",
            "itemId": "ITEM_A",
            "itemName": "Atorvastatin 10mg",
            "invoice_date": pd.Timestamp("2026-01-01"),
            "invoice_number": "INV_A1",
            "quantity": 2,
            "packing": "1X15",
            "MOBILE_NO": "9876543210",
        },
        {
            "customerId": "CUST_A",
            "customerName": "Anil Shah",
            "itemId": "ITEM_A",
            "itemName": "Atorvastatin 10mg",
            "invoice_date": pd.Timestamp("2026-02-01"),
            "invoice_number": "INV_A2",
            "quantity": 2,
            "packing": "1X15",
            "MOBILE_NO": "9876543210",
        },
    ])

    # Month 1 Ingestion
    month1_sales = pd.DataFrame([
        {
            "customerId": "CUST_A",
            "customerName": "Anil Shah",
            "itemId": "ITEM_A",
            "itemName": "Atorvastatin 10mg",
            "invoice_date": "2026-03-03",
            "invoice_number": "INV_A3",
            "quantity": 2,
            "packing": "1X15",
            "MOBILE_NO": "9876543210",
        },
    ])

    res1 = ingest_monthly_sales_pipeline(
        new_sales=month1_sales,
        existing_history_df=base_history,
        model_bundle=model_bundle,
        storage=temp_storage,
        prediction_date="2026-03-05",
    )

    snaps_m1 = temp_storage.get_prediction_snapshots(customer_id="CUST_A")
    assert len(snaps_m1) == 1
    assert snaps_m1[0]["prediction_date"] == "2026-03-05"
    assert snaps_m1[0]["last_purchase_date"] == "2026-03-03"
    assert snaps_m1[0]["status"] == "pending"
    assert snaps_m1[0]["actual_next_purchase_date"] is None

    # Month 2 Ingestion: provides subsequent actual purchase for April
    month2_sales = pd.DataFrame([
        {
            "customerId": "CUST_A",
            "customerName": "Anil Shah",
            "itemId": "ITEM_A",
            "itemName": "Atorvastatin 10mg",
            "invoice_date": "2026-04-05",
            "invoice_number": "INV_A4",
            "quantity": 2,
            "packing": "1X15",
            "MOBILE_NO": "9876543210",
        },
    ])

    res2 = ingest_monthly_sales_pipeline(
        new_sales=month2_sales,
        existing_history_df=res1["combined_history_df"],
        model_bundle=model_bundle,
        storage=temp_storage,
        prediction_date="2026-04-06",
    )

    all_snaps = temp_storage.get_prediction_snapshots(customer_id="CUST_A")
    assert len(all_snaps) == 2  # Both historical snapshots preserved

    # Verify Month 1 snapshot was matched against actual April purchase
    m1_updated = [s for s in all_snaps if s["prediction_date"] == "2026-03-05"][0]
    assert m1_updated["status"] == "evaluated"
    assert m1_updated["actual_next_purchase_date"] == "2026-04-05"
    assert m1_updated["outcome_error_days"] is not None

    # Verify Month 2 snapshot is newly pending (unseen production data)
    m2_snap = [s for s in all_snaps if s["prediction_date"] == "2026-04-06"][0]
    assert m2_snap["status"] == "pending"
    assert m2_snap["actual_next_purchase_date"] is None


def test_csv_export_from_updated_reminder_list(temp_storage):
    """Verify CSV export after monthly ingestion contains the exact 10 columns."""
    model_bundle = load_refill_model()
    raw_data = load_purchase_history()
    assert raw_data is not None

    if "purchase_seq" in raw_data.columns:
        multi = raw_data[raw_data["purchase_seq"] >= 3]
        sample_history = multi.head(60) if not multi.empty else raw_data.head(60)
    else:
        sample_history = raw_data.head(60)

    # New sales row for the selected customer-medication
    first_row = sample_history.iloc[0]
    new_sales = pd.DataFrame([
        {
            "customerId": str(first_row["customerId"]),
            "customerName": str(first_row.get("customerName", "Sample Customer")),
            "itemId": str(first_row["itemId"]),
            "itemName": str(first_row.get("itemName", "Sample Medicine")),
            "invoice_date": "2026-09-01",
            "quantity": 2,
            "packing": "1X15",
            "MOBILE_NO": "9876543210",
        }
    ])

    res = ingest_monthly_sales_pipeline(
        new_sales=new_sales,
        existing_history_df=sample_history,
        model_bundle=model_bundle,
        storage=temp_storage,
        prediction_date="2026-09-02",
    )

    sched_df = res["reminder_schedule_df"]
    assert not sched_df.empty

    csv_bytes = build_export_csv(sched_df)
    assert isinstance(csv_bytes, bytes)

    df_out = pd.read_csv(io.BytesIO(csv_bytes))
    expected_cols = [
        "customer_id",
        "customer_name",
        "phone_number",
        "mobile_status",
        "item_id",
        "medication_name",
        "last_purchase_date",
        "estimated_days_of_supply",
        "expected_refill_date",
        "reminder_date",
    ]
    assert list(df_out.columns) == expected_cols
    assert len(df_out) == len(sched_df)


def test_process_monthly_sales_data_valid_upload():
    """Verify processing valid sales transactions without generating reminders."""
    existing_history = pd.DataFrame([
        {
            "customerId": "CUST_1",
            "customerName": "Ramesh Kumar",
            "itemId": "ITEM_1",
            "itemName": "Telmisartan 40mg",
            "invoice_date": pd.Timestamp("2026-08-01"),
            "quantity": 2,
            "packing": "1X15",
            "MOBILE_NO": "9876543210",
        }
    ])

    new_sales = pd.DataFrame([
        {
            "DATE": "2026-09-01",
            "TRN NO.": "TRN-901",
            "PARTY NAME": "Ramesh Kumar",
            "customerId": "CUST_1",
            "ITEM CODE": "ITEM_1",
            "itemName": "Telmisartan 40mg",
            "quantity": 2,
            "packing": "1X15",
            "MOBILE NO.": "9876543210",
        },
        {
            "DATE": "2026-09-05",
            "TRN NO.": "TRN-902",
            "PARTY NAME": "New Patient",
            "customerId": "CUST_2",
            "ITEM CODE": "ITEM_2",
            "itemName": "Atorvastatin 20mg",
            "quantity": 1,
            "packing": "1X10",
            "MOBILE NO.": "9123456780",
        },
    ])

    result = process_monthly_sales_data(new_sales, existing_history_df=existing_history)

    assert result["status"] == "success"
    assert result["message"] == "Sales Data Updated Successfully"
    metrics = result["metrics"]
    assert metrics["new_records_added"] == 2
    assert metrics["existing_duplicates_skipped"] == 0
    assert metrics["customers_updated"] == 1  # CUST_1
    assert metrics["new_customers"] == 1      # CUST_2
    assert metrics["new_medicines"] == 1      # ITEM_2
    assert metrics["latest_sales_date"] == "05-09-2026"
    assert metrics["records_requiring_review"] == 0
    assert len(result["updated_history_df"]) == 3


def test_process_monthly_sales_data_duplicate_handling():
    """Verify duplicate transactions (both within upload and against existing history) are skipped."""
    existing_history = pd.DataFrame([
        {
            "customerId": "CUST_1",
            "customerName": "Ramesh Kumar",
            "itemId": "ITEM_1",
            "itemName": "Telmisartan 40mg",
            "invoice_date": pd.Timestamp("2026-08-01"),
            "quantity": 2,
            "packing": "1X15",
            "MOBILE_NO": "9876543210",
        }
    ])

    incoming_sales = pd.DataFrame([
        # Exact duplicate of existing history record
        {
            "customerId": "CUST_1",
            "customerName": "Ramesh Kumar",
            "itemId": "ITEM_1",
            "itemName": "Telmisartan 40mg",
            "invoice_date": "2026-08-01",
            "quantity": 2,
            "packing": "1X15",
            "MOBILE_NO": "9876543210",
        },
        # New valid record
        {
            "customerId": "CUST_1",
            "customerName": "Ramesh Kumar",
            "itemId": "ITEM_1",
            "itemName": "Telmisartan 40mg",
            "invoice_date": "2026-09-01",
            "quantity": 2,
            "packing": "1X15",
            "MOBILE_NO": "9876543210",
        },
        # Intra-file duplicate of the new record
        {
            "customerId": "CUST_1",
            "customerName": "Ramesh Kumar",
            "itemId": "ITEM_1",
            "itemName": "Telmisartan 40mg",
            "invoice_date": "2026-09-01",
            "quantity": 2,
            "packing": "1X15",
            "MOBILE_NO": "9876543210",
        },
    ])

    result = process_monthly_sales_data(incoming_sales, existing_history_df=existing_history)

    assert result["status"] == "success"
    metrics = result["metrics"]
    assert metrics["new_records_added"] == 1
    assert metrics["existing_duplicates_skipped"] == 2  # 1 from existing history match, 1 intra-file
    assert len(result["updated_history_df"]) == 2


def test_process_monthly_sales_data_missing_mobile_number():
    """Verify records with missing or empty mobile numbers are preserved in the history dataset."""
    new_sales = pd.DataFrame([
        {
            "customerId": "CUST_NO_PHONE",
            "customerName": "Patient Without Phone",
            "itemId": "ITEM_1",
            "itemName": "Metformin 500mg",
            "invoice_date": "2026-09-01",
            "quantity": 2,
            "packing": "1X15",
            "MOBILE_NO": "",
        }
    ])

    result = process_monthly_sales_data(new_sales, existing_history_df=pd.DataFrame())

    assert result["status"] == "success"
    assert result["metrics"]["new_records_added"] == 1
    updated_df = result["updated_history_df"]
    assert len(updated_df) == 1
    assert updated_df.iloc[0]["customerId"] == "CUST_NO_PHONE"


def test_process_monthly_sales_data_existing_history_preservation():
    """Verify that existing 5.8-year multi-year history records are completely preserved."""
    existing_history = pd.DataFrame([
        {
            "customerId": f"HIST_CUST_{i}",
            "customerName": f"Patient {i}",
            "itemId": "ITEM_1",
            "itemName": "Amlodipine 5mg",
            "invoice_date": pd.Timestamp(f"2025-0{i+1}-10"),
            "quantity": 1,
            "packing": "1X10",
            "MOBILE_NO": "9876543210",
        }
        for i in range(5)
    ])

    new_sales = pd.DataFrame([
        {
            "customerId": "NEW_CUST_99",
            "customerName": "Brand New Patient",
            "itemId": "ITEM_1",
            "itemName": "Amlodipine 5mg",
            "invoice_date": "2026-09-10",
            "quantity": 1,
            "packing": "1X10",
            "MOBILE_NO": "9999999999",
        }
    ])

    result = process_monthly_sales_data(new_sales, existing_history_df=existing_history)

    assert result["status"] == "success"
    updated_df = result["updated_history_df"]
    # All 5 original history rows + 1 new row = 6 rows
    assert len(updated_df) == 6
    for i in range(5):
        assert f"HIST_CUST_{i}" in updated_df["customerId"].values


def test_process_monthly_sales_data_no_model_modification():
    """Verify that processing sales data does NOT retrain or alter model bundle or files."""
    model_bundle = load_refill_model()
    assert model_bundle is not None

    # Snapshot model bundle state
    model_obj = model_bundle.get("model")
    model_features = model_bundle.get("feature_names")

    new_sales = pd.DataFrame([
        {
            "customerId": "CUST_X",
            "customerName": "Test Customer",
            "itemId": "ITEM_X",
            "itemName": "Test Medicine",
            "invoice_date": "2026-09-15",
            "quantity": 3,
            "packing": "1X10",
            "MOBILE_NO": "9876543210",
        }
    ])

    result = process_monthly_sales_data(new_sales)
    assert result["status"] == "success"

    # Reload model bundle and ensure unchanged
    model_bundle_after = load_refill_model()
    assert model_bundle_after is not None
    assert model_bundle_after.get("feature_names") == model_features
    # Check that the model object is intact
    assert type(model_bundle_after.get("model")) == type(model_obj)


def test_generate_updated_predictions_workflow():
    """Verify generate_updated_predictions executes the production flow and returns the 5 summary metrics."""
    model_bundle = load_refill_model()
    assert model_bundle is not None

    sample_history = pd.DataFrame([
        # Multi-purchase customer 1 (Valid phone)
        {
            "customerId": "CUST_ELIG_1",
            "customerName": "Ramesh Kumar",
            "itemId": "ITEM_1",
            "itemName": "Telmisartan 40mg",
            "invoice_date": pd.Timestamp("2026-06-01"),
            "quantity": 2,
            "packing": "1X15",
            "MOBILE_NO": "9876543210",
        },
        {
            "customerId": "CUST_ELIG_1",
            "customerName": "Ramesh Kumar",
            "itemId": "ITEM_1",
            "itemName": "Telmisartan 40mg",
            "invoice_date": pd.Timestamp("2026-07-01"),
            "quantity": 2,
            "packing": "1X15",
            "MOBILE_NO": "9876543210",
        },
        {
            "customerId": "CUST_ELIG_1",
            "customerName": "Ramesh Kumar",
            "itemId": "ITEM_1",
            "itemName": "Telmisartan 40mg",
            "invoice_date": pd.Timestamp("2026-08-01"),
            "quantity": 2,
            "packing": "1X15",
            "MOBILE_NO": "9876543210",
        },
        # Multi-purchase customer 2 (Missing phone)
        {
            "customerId": "CUST_ELIG_2",
            "customerName": "Anita Desai",
            "itemId": "ITEM_2",
            "itemName": "Atorvastatin 10mg",
            "invoice_date": pd.Timestamp("2026-07-10"),
            "quantity": 1,
            "packing": "1X10",
            "MOBILE_NO": "",
        },
        {
            "customerId": "CUST_ELIG_2",
            "customerName": "Anita Desai",
            "itemId": "ITEM_2",
            "itemName": "Atorvastatin 10mg",
            "invoice_date": pd.Timestamp("2026-08-10"),
            "quantity": 1,
            "packing": "1X10",
            "MOBILE_NO": "",
        },
        # Single-purchase customer (Cold-start / not eligible)
        {
            "customerId": "CUST_COLD_1",
            "customerName": "Single Purchase Patient",
            "itemId": "ITEM_3",
            "itemName": "Metformin 500mg",
            "invoice_date": pd.Timestamp("2026-08-15"),
            "quantity": 2,
            "packing": "1X15",
            "MOBILE_NO": "9123456780",
        },
    ])

    result = generate_updated_predictions(sample_history, model_bundle)

    assert result["status"] == "success"
    eligible_df = result["eligible_df"]
    ineligible_df = result["ineligible_df"]
    metrics = result["metrics"]

    assert metrics["customers_evaluated"] == 3  # CUST_ELIG_1, CUST_ELIG_2, CUST_COLD_1
    assert metrics["predictions_generated"] == 2  # CUST_ELIG_1 and CUST_ELIG_2
    assert metrics["customers_not_eligible"] == 1  # CUST_COLD_1
    assert metrics["missing_mobile_predictions"] == 1  # CUST_ELIG_2 has missing phone

    # Verify expected refill date and 2-day reminder date exist
    assert "Expected Refill Date" in eligible_df.columns
    assert "Reminder Date" in eligible_df.columns
    assert all(d != "-" for d in eligible_df["Expected Refill Date"])
    assert all(d != "-" for d in eligible_df["Reminder Date"])


def test_evaluate_prediction_outcomes_accuracy_and_metrics():
    """Verify evaluate_prediction_outcomes calculates exact errors, ±1/±3/±7 day metrics, and MAE."""
    predictions = pd.DataFrame([
        # Pred 1: Exp 2026-08-30 -> Actual 2026-08-30 (Diff: 0 days, within 1, 3, 7)
        {
            "customerId": "CUST_1",
            "customerName": "Ramesh Kumar",
            "itemId": "ITEM_1",
            "itemName": "Telmisartan 40mg",
            "last_purchase_date": "2026-08-01",
            "expected_refill_date": "2026-08-30",
        },
        # Pred 2: Exp 2026-08-30 -> Actual 2026-09-01 (Diff: +2 days, within 3, 7)
        {
            "customerId": "CUST_2",
            "customerName": "Anita Desai",
            "itemId": "ITEM_2",
            "itemName": "Atorvastatin 10mg",
            "last_purchase_date": "2026-08-01",
            "expected_refill_date": "2026-08-30",
        },
        # Pred 3: Exp 2026-08-30 -> Actual 2026-09-05 (Diff: +6 days, within 7)
        {
            "customerId": "CUST_3",
            "customerName": "Sunil Sharma",
            "itemId": "ITEM_3",
            "itemName": "Metformin 500mg",
            "last_purchase_date": "2026-08-01",
            "expected_refill_date": "2026-08-30",
        },
        # Pred 4: Exp 2026-08-30 -> Actual 2026-09-12 (Diff: +13 days, > 7)
        {
            "customerId": "CUST_4",
            "customerName": "Pooja Gupta",
            "itemId": "ITEM_4",
            "itemName": "Amlodipine 5mg",
            "last_purchase_date": "2026-08-01",
            "expected_refill_date": "2026-08-30",
        },
        # Pred 5: Exp 2026-08-30 -> NO subsequent purchase (Pending customer -> MUST NOT be marked failure)
        {
            "customerId": "CUST_5",
            "customerName": "Pending Patient",
            "itemId": "ITEM_5",
            "itemName": "Glimepiride 2mg",
            "last_purchase_date": "2026-08-01",
            "expected_refill_date": "2026-08-30",
        },
    ])

    actual_sales = pd.DataFrame([
        {"customerId": "CUST_1", "itemId": "ITEM_1", "invoice_date": "2026-08-30"},
        {"customerId": "CUST_2", "itemId": "ITEM_2", "invoice_date": "2026-09-01"},
        {"customerId": "CUST_3", "itemId": "ITEM_3", "invoice_date": "2026-09-05"},
        {"customerId": "CUST_4", "itemId": "ITEM_4", "invoice_date": "2026-09-12"},
        # An unrelated customer purchase
        {"customerId": "CUST_UNRELATED", "itemId": "ITEM_1", "invoice_date": "2026-09-02"},
    ])

    results = evaluate_prediction_outcomes(predictions, actual_sales)

    assert results["status"] == "success"
    # Only 4 predictions had actual subsequent purchases
    assert results["predictions_evaluated"] == 4
    # Pending customer is NOT a failure, incremented pending_count
    assert results["pending_count"] == 1

    # Within ±1 day: CUST_1 (0 days diff) -> 1/4 = 25.0%
    assert results["within_1_day_count"] == 1
    assert results["within_1_day_pct"] == 25.0

    # Within ±3 days: CUST_1 (0 days), CUST_2 (+2 days) -> 2/4 = 50.0%
    assert results["within_3_days_count"] == 2
    assert results["within_3_days_pct"] == 50.0

    # Within ±7 days: CUST_1 (0), CUST_2 (+2), CUST_3 (+6) -> 3/4 = 75.0%
    assert results["within_7_days_count"] == 3
    assert results["within_7_days_pct"] == 75.0

    # MAE = (0 + 2 + 6 + 13) / 4 = 21 / 4 = 5.25 -> 5.2 or 5.3
    assert abs(results["mean_absolute_error"] - 5.25) < 0.1

    # Evaluated table has exact 4 records
    df_eval = results["evaluated_df"]
    assert len(df_eval) == 4
    assert list(df_eval["Customer Name"].values) == ["Ramesh Kumar", "Anita Desai", "Sunil Sharma", "Pooja Gupta"]


def test_evaluate_prediction_outcomes_empty_and_graceful():
    """Verify evaluate_prediction_outcomes handles empty inputs gracefully without exceptions."""
    res_empty_all = evaluate_prediction_outcomes(pd.DataFrame(), pd.DataFrame())
    assert res_empty_all["status"] == "no_evaluated_records"
    assert res_empty_all["predictions_evaluated"] == 0
    assert res_empty_all["mean_absolute_error"] == 0.0

    res_none = evaluate_prediction_outcomes(None, None)
    assert res_none["status"] == "no_evaluated_records"


def test_storage_get_evaluated_prediction_results_lifecycle(temp_storage):
    """Verify saving snapshots, matching actual transactions, and retrieving evaluated results via storage."""
    snapshots = [
        {
            "snapshot_id": "snap_101",
            "customer_id": "C_101",
            "item_id": "I_101",
            "customer_name": "Test Patient 1",
            "item_name": "Medicine A",
            "last_purchase_date": "2026-08-01",
            "prediction_date": "2026-08-01",
            "estimated_days_of_supply": 30.0,
            "expected_refill_date": "2026-08-31",
            "reminder_date": "2026-08-29",
            "prediction_source": "hybrid_supply",
            "status": "pending",
        },
        {
            "snapshot_id": "snap_102",
            "customer_id": "C_102",
            "item_id": "I_102",
            "customer_name": "Test Patient 2",
            "item_name": "Medicine B",
            "last_purchase_date": "2026-08-01",
            "prediction_date": "2026-08-01",
            "estimated_days_of_supply": 30.0,
            "expected_refill_date": "2026-08-31",
            "reminder_date": "2026-08-29",
            "prediction_source": "hybrid_supply",
            "status": "pending",
        },
    ]

    saved = temp_storage.save_prediction_snapshots(snapshots)
    assert saved == 2

    # Before new sales arrive: 0 evaluated
    pre_results = temp_storage.get_evaluated_prediction_results()
    assert pre_results["predictions_evaluated"] == 0

    # New sales arrive: C_101 repurchased on 2026-08-31 (0 days diff); C_102 has not repurchased yet
    new_sales = pd.DataFrame([
        {"customerId": "C_101", "itemId": "I_101", "invoice_date": "2026-08-31"},
    ])

    match_metrics = temp_storage.match_and_update_prediction_outcomes(new_sales)
    assert match_metrics["evaluated_count"] == 1
    assert match_metrics["pending_count"] == 1

    post_results = temp_storage.get_evaluated_prediction_results()
    assert post_results["status"] == "success"
    assert post_results["predictions_evaluated"] == 1
    assert post_results["within_1_day_count"] == 1
    assert post_results["within_3_days_count"] == 1
    assert post_results["within_7_days_count"] == 1
    assert post_results["mean_absolute_error"] == 0.0
    assert len(post_results["evaluated_df"]) == 1
    assert post_results["evaluated_df"]["Customer Name"].iloc[0] == "Test Patient 1"



