"""Unit tests for RefillCare Streamlit Application."""

import io
from pathlib import Path
from datetime import date
import pytest
import pandas as pd
import numpy as np

import app_refillcare as app


def test_app_imports_and_constants():
    """Verify core constants and module configurations."""
    assert app.DEFAULT_MODEL_PATH.endswith("refill_model.joblib")
    assert app.DEFAULT_TEST_DATA_PATH.endswith("test.parquet")
    assert app.DEFAULT_HISTORY_DATA_PATH.endswith("purchase_history.parquet")


def test_find_artifact_path():
    """Verify artifact path resolution works and resolves existing files."""
    model_path = app.find_artifact_path(app.DEFAULT_MODEL_PATH)
    assert isinstance(model_path, Path)
    assert model_path.exists()

    non_existent = app.find_artifact_path("non_existent_file.xyz")
    assert isinstance(non_existent, Path)
    assert not non_existent.exists()


def test_load_refill_model():
    """Verify model bundle loading and missing artifact handling."""
    bundle = app.load_refill_model()
    assert bundle is not None
    assert "pipeline" in bundle
    assert "features_numeric" in bundle

    missing_bundle = app.load_refill_model("nonexistent_path/model.joblib")
    assert missing_bundle is None


def test_load_refill_data_and_history():
    """Verify test dataset and canonical history loading."""
    df = app.load_refill_data()
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    hist = app.load_purchase_history()
    assert hist is not None
    assert isinstance(hist, pd.DataFrame)
    assert not hist.empty

    assert app.load_refill_data("nonexistent_dir/data.parquet") is None
    assert app.load_purchase_history("nonexistent_dir/history.parquet") is None


def test_determine_mobile_status():
    """Verify classification of mobile numbers into Valid, Missing, and Invalid format."""
    # Valid formats
    assert app.determine_mobile_status("9876543210") == "Valid"
    assert app.determine_mobile_status("8123456789") == "Valid"
    assert app.determine_mobile_status("+919876543210") == "Valid"
    assert app.determine_mobile_status("919876543210") == "Valid"
    assert app.determine_mobile_status("09876543210") == "Valid"

    # Missing formats
    assert app.determine_mobile_status(None) == "Missing"
    assert app.determine_mobile_status("") == "Missing"
    assert app.determine_mobile_status("   ") == "Missing"
    assert app.determine_mobile_status(np.nan) == "Missing"
    assert app.determine_mobile_status("-") == "Missing"
    assert app.determine_mobile_status("None") == "Missing"
    assert app.determine_mobile_status("nan") == "Missing"

    # Invalid formats
    assert app.determine_mobile_status("12345") == "Invalid format"
    assert app.determine_mobile_status("98765") == "Invalid format"
    assert app.determine_mobile_status("abc") == "Invalid format"
    assert app.determine_mobile_status("1234567890") == "Invalid format"  # Starts with 1
    assert app.determine_mobile_status("0000000000") == "Invalid format"


def test_prepare_prediction_overview_with_sample():
    """Verify splitting of eligible vs ineligible records and metrics calculation."""
    bundle = app.load_refill_model()
    raw_data = app.load_refill_data()
    assert bundle is not None
    assert raw_data is not None

    sample_df = raw_data.head(30)
    el_df, inel_df, metrics = app.prepare_prediction_overview(sample_df, bundle)

    assert isinstance(el_df, pd.DataFrame)
    assert isinstance(inel_df, pd.DataFrame)
    assert isinstance(metrics, dict)

    assert metrics["total_histories"] == len(el_df) + len(inel_df)
    assert metrics["eligible_count"] == len(el_df)
    assert metrics["ineligible_count"] == len(inel_df)

    if not el_df.empty:
        assert "Customer Name" in el_df.columns
        assert "Mobile Number" in el_df.columns
        assert "Mobile Status" in el_df.columns
        assert "Medication" in el_df.columns
        assert "Last Purchase Date" in el_df.columns
        assert "Estimated Days of Supply" in el_df.columns
        assert "Expected Refill Date" in el_df.columns
        assert "Reminder Date" in el_df.columns
        assert "History Quality" in el_df.columns
        assert all(el_df["Purchase Count"] >= 6)

    if not inel_df.empty:
        assert "Reason for Ineligibility" in inel_df.columns
        assert "Purchase Count" in inel_df.columns


def test_prepare_prediction_overview_empty_dataframe():
    """Verify graceful handling of an empty dataframe without crashing."""
    bundle = app.load_refill_model()
    empty_df = pd.DataFrame()
    el_df, inel_df, metrics = app.prepare_prediction_overview(empty_df, bundle)

    assert el_df.empty
    assert inel_df.empty
    assert metrics["total_histories"] == 0
    assert metrics["eligible_count"] == 0


def test_generate_reminder_schedule_table():
    """Verify that scheduler generates all stages with expected labels and new fields."""
    bundle = app.load_refill_model()
    raw_data = app.load_refill_data()
    if "purchase_count_so_far" in raw_data.columns:
        deep = raw_data[raw_data["purchase_count_so_far"] >= 6]
        sample_df = deep.head(80) if not deep.empty else raw_data.head(80)
    else:
        sample_df = raw_data.head(80)
    el_df, _, _ = app.prepare_prediction_overview(sample_df, bundle)

    assert not el_df.empty, "Expected at least one hybrid-eligible row in test sample"
    sched_df = app.generate_reminder_schedule_table(el_df, max_records=2)

    assert not sched_df.empty
    assert "Customer" in sched_df.columns
    assert "Medicine" in sched_df.columns
    assert "Customer Name" in sched_df.columns
    assert "Mobile Number" in sched_df.columns
    assert "Mobile Status" in sched_df.columns
    assert "Estimated Days of Supply" in sched_df.columns
    assert "Reminder Stage" in sched_df.columns
    assert "Reminder Message" in sched_df.columns
    assert "Status" in sched_df.columns

    stages = sched_df["Reminder Stage"].unique().tolist()
    expected_possible = ["-7 days", "-3 days", "-1 day", "0 days", "+2 days", "+5 days"]
    for s in stages:
        assert s in expected_possible

    empty_sched = app.generate_reminder_schedule_table(pd.DataFrame())
    assert empty_sched.empty


def test_filter_predictions():
    """Verify prediction filtering by customer, medicine, and history quality."""
    bundle = app.load_refill_model()
    raw_data = app.load_refill_data()
    el_df, _, _ = app.prepare_prediction_overview(raw_data.head(50), bundle)

    if not el_df.empty:
        cust_sample = str(el_df["Customer Name"].iloc[0])[:4]
        filtered_cust = app.filter_predictions(el_df, customer_query=cust_sample)
        assert len(filtered_cust) > 0
        assert all(cust_sample.lower() in str(c).lower() for c in filtered_cust["Customer Name"])

        filtered_qual = app.filter_predictions(el_df, quality_filter="high_history")
        assert all(q == "high_history" for q in filtered_qual["History Quality"])


def test_filter_schedules():
    """Verify schedule filtering by stage and status."""
    bundle = app.load_refill_model()
    raw_data = app.load_refill_data()
    el_df, _, _ = app.prepare_prediction_overview(raw_data.head(20), bundle)
    sched_df = app.generate_reminder_schedule_table(el_df, max_records=3)

    if not sched_df.empty:
        filtered_stage = app.filter_schedules(sched_df, stage_filter="-7 days")
        assert all(s == "-7 days" for s in filtered_stage["Reminder Stage"])

        filtered_status = app.filter_schedules(sched_df, status_filter="scheduled")
        assert all(st == "scheduled" for st in filtered_status["Status"])


def test_get_customer_history_summary():
    """Verify patient history timeline extraction and interval calculations."""
    hist_df = app.load_purchase_history()
    assert hist_df is not None

    sample_row = hist_df[hist_df["purchase_seq"] > 2].iloc[0]
    cid = sample_row["customerId"]
    iid = sample_row["itemId"]

    summary = app.get_customer_history_summary(hist_df, cid, iid)
    assert summary["total_purchases"] >= 2
    assert "visits" in summary
    assert isinstance(summary["visits"], pd.DataFrame)
    assert not summary["visits"].empty
    assert summary["median_interval"] is not None


def test_csv_export_exact_10_columns():
    """Verify CSV export contains the exact 10 required client columns."""
    sample_data = pd.DataFrame([
        {
            "customerId": "CUST101",
            "Customer Name": "Rajesh Kumar",
            "Mobile Number": "9876543210",
            "itemId": "ITEM202",
            "Medication": "Metformin 500mg",
            "Last Purchase Date": "2026-08-01",
            "Estimated Days of Supply": 30.0,
            "Expected Refill Date": "2026-08-31",
            "Reminder Date": "2026-08-29",
            "Mobile Status": "Valid",
        },
        {
            "customerId": "CUST102",
            "Customer Name": "Sunita Sharma",
            "Mobile Number": "",
            "itemId": "ITEM303",
            "Medication": "Amlodipine 5mg",
            "Last Purchase Date": "2026-08-05",
            "Estimated Days of Supply": 15.0,
            "Expected Refill Date": "2026-08-20",
            "Reminder Date": "2026-08-18",
            "Mobile Status": "Missing",
        },
    ])

    csv_bytes = app.build_export_csv(sample_data)
    assert isinstance(csv_bytes, bytes)

    df_exported = pd.read_csv(io.BytesIO(csv_bytes))

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

    assert list(df_exported.columns) == expected_cols
    assert len(df_exported) == 2
    # Verify both records retained even with missing phone
    assert df_exported["mobile_status"].tolist() == ["Valid", "Missing"]
    assert df_exported["customer_name"].tolist() == ["Rajesh Kumar", "Sunita Sharma"]
    assert df_exported["estimated_days_of_supply"].tolist() == [30.0, 15.0]


def test_build_export_csv_empty():
    """Verify build_export_csv handles empty dataframe gracefully with correct header."""
    csv_bytes = app.build_export_csv(pd.DataFrame())
    df_exported = pd.read_csv(io.BytesIO(csv_bytes))
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
    assert list(df_exported.columns) == expected_cols
    assert len(df_exported) == 0


def test_parse_and_validate_uploaded_sales_file():
    """Verify Data Update upload validation and all 8 business/hygiene metrics calculation."""
    csv_content = """invoice_date,customerId,customerName,itemId,itemName,quantity,packing,MOBILE_NO
2026-08-01,C001,Amit Patel,I001,Telmisartan 40mg,2,1X15,9876543210
2026-08-05,C002,Priya Singh,I002,Atorvastatin 10mg,1,1X10,9123456780
2026-08-10,C003,New Customer,I001,Telmisartan 40mg,4,1X15,
,C004,Invalid Date Cust,I003,Drug X,1,1X10,9876543210
2026-08-15,C005,Return Cust,I004,Drug Y,-1,1X10,9876543210
"""
    file_obj = io.BytesIO(csv_content.encode("utf-8"))
    file_obj.name = "monthly_sales_aug2026.csv"

    existing_hist = pd.DataFrame({
        "customerId": ["C001", "C002"],
        "customerName": ["Amit Patel", "Priya Singh"],
    })

    parsed_df, metrics = app.parse_and_validate_uploaded_sales_file(file_obj, existing_history_df=existing_hist)

    assert parsed_df is not None
    assert metrics["records_received"] == 5
    assert metrics["unique_customers"] == 5
    assert metrics["medicines_count"] == 4  # I001, I002, I003, I004
    assert metrics["valid_records"] == 3  # C001, C002, C003 have valid date, customer, item, quantity > 0
    assert metrics["records_requiring_review"] == 2  # missing date (C004) and negative quantity (C005)
    assert metrics["missing_customer_info"] == 0
    assert metrics["missing_mobile_numbers"] == 1  # C003 has empty mobile
    assert metrics["duplicate_records"] == 0
    assert metrics["invalid_quantities"] == 1  # C005 has -1
    assert "01-08-2026 to 15-08-2026" in metrics["file_date_range"]
    assert metrics["new_customers"] >= 1  # C003, C004, C005 are new
    assert metrics["existing_customers"] == 2  # C001 and C002
    # Verify that record with missing mobile number is NOT deleted
    assert len(parsed_df) == 5
    assert "C003" in parsed_df["customerId"].values


def test_parse_and_validate_custom_project_column_headers():
    """Verify parsing and validation with legacy pharmacy column header names (DATE, TRN NO., PARTY NAME, ITEM CODE, MOBILE NO.)."""
    csv_content = """DATE,TRN NO.,PARTY NAME,ITEM CODE,itemName,packing,quantity,MOBILE NO.
2026-09-01,TRN-101,Ramesh Sharma,DRUG-001,Metformin 500mg,1X15,2,9876543210
2026-09-02,TRN-102,Anita Desai,DRUG-002,Amlodipine 5mg,1X10,1,
2026-09-03,TRN-103,,DRUG-003,Telmisartan 40mg,1X15,2,9123456780
2026-09-04,TRN-104,Vijay Singh,DRUG-001,Metformin 500mg,1X15,0,9876543210
2026-09-01,TRN-101,Ramesh Sharma,DRUG-001,Metformin 500mg,1X15,2,9876543210
"""
    file_obj = io.BytesIO(csv_content.encode("utf-8"))
    file_obj.name = "pharmacy_sept2026.csv"

    parsed_df, metrics = app.parse_and_validate_uploaded_sales_file(file_obj)

    assert parsed_df is not None
    assert "invoice_date" in parsed_df.columns
    assert "invoice_number" in parsed_df.columns
    assert "customerName" in parsed_df.columns
    assert "customerId" in parsed_df.columns
    assert "itemId" in parsed_df.columns
    assert "MOBILE_NO" in parsed_df.columns
    assert "packing" in parsed_df.columns
    assert "quantity" in parsed_df.columns

    assert metrics["records_received"] == 5
    assert metrics["medicines_count"] == 3  # DRUG-001, DRUG-002, DRUG-003
    assert metrics["missing_customer_info"] == 1  # Row 3 has empty party name
    assert metrics["missing_mobile_numbers"] == 1  # Row 2 has empty mobile
    assert metrics["duplicate_records"] == 1  # Row 5 is duplicate of row 1
    assert metrics["invalid_quantities"] == 1  # Row 4 has quantity 0
    assert metrics["valid_records"] == 3  # Row 1, 2, 5
    # Confirm rows with missing mobile are NOT removed from dataframe
    assert len(parsed_df) == 5


def test_build_reminder_list_csv_exact_operational_fields_and_mobile_filtering():
    """Verify build_reminder_list_csv contains only the exact operational fields and filters out invalid mobiles."""
    input_data = pd.DataFrame([
        {
            "customerId": "C1001",
            "Customer Name": "Rahul Verma",
            "Mobile Number": "9876543210",
            "Mobile Status": "Valid",
            "itemId": "MED201",
            "Medication": "Telmisartan 40mg",
            "Last Purchase Date": "2026-08-01",
            "Expected Refill Date": "2026-08-31",
            "Reminder Date": "2026-08-29",
            "Estimated Days of Supply": 30.0,
            # ML / diagnostic / technical internal fields that must NOT appear in output CSV:
            "feature_mean_interval": 28.5,
            "xgboost_prediction": 29.8,
            "rmse_score": 1.2,
            "train_split": "train",
        },
        {
            "customerId": "C1002",
            "Customer Name": "Meera Sharma",
            "Mobile Number": "",  # Missing mobile
            "Mobile Status": "Missing",
            "itemId": "MED202",
            "Medication": "Metformin 500mg",
            "Last Purchase Date": "2026-08-02",
            "Expected Refill Date": "2026-09-01",
            "Reminder Date": "2026-08-30",
            "Estimated Days of Supply": 30.0,
            "xgboost_prediction": 30.0,
        },
        {
            "customerId": "C1003",
            "Customer Name": "Vikram Patel",
            "Mobile Number": "12345",  # Invalid format
            "Mobile Status": "Invalid format",
            "itemId": "MED203",
            "Medication": "Amlodipine 5mg",
            "Last Purchase Date": "2026-08-03",
            "Expected Refill Date": "2026-09-02",
            "Reminder Date": "2026-08-31",
            "Estimated Days of Supply": 30.0,
        },
        {
            "customerId": "C1004",
            "Customer Name": "Pooja Gupta",
            "Mobile Number": "+919123456780",
            "Mobile Status": "Valid",
            "itemId": "MED204",
            "Medication": "Atorvastatin 10mg",
            "Last Purchase Date": "2026-08-04",
            "Expected Refill Date": "2026-09-03",
            "Reminder Date": "2026-09-01",
            "Estimated Days of Supply": 30.0,
        },
    ])

    csv_bytes = app.build_reminder_list_csv(input_data)
    assert isinstance(csv_bytes, bytes)

    df_exported = pd.read_csv(io.BytesIO(csv_bytes))

    exact_operational_cols = [
        "customerId",
        "customerName",
        "MOBILE_NO",
        "itemId",
        "itemName",
        "last_purchase_date",
        "expected_refill_date",
        "reminder_date",
        "predicted_days_until_refill",
    ]

    assert list(df_exported.columns) == exact_operational_cols

    # Verify only valid mobile numbers are included in the downloadable delivery CSV
    assert len(df_exported) == 2
    assert df_exported["customerId"].tolist() == ["C1001", "C1004"]
    assert df_exported["customerName"].tolist() == ["Rahul Verma", "Pooja Gupta"]

    # Verify NO internal ML / diagnostics features are present
    for forbidden in ["feature_mean_interval", "xgboost_prediction", "rmse_score", "train_split", "Mobile Status"]:
        assert forbidden not in df_exported.columns

    # Verify original dataframe and customers with missing/invalid phones were NOT deleted from customer system
    assert len(input_data) == 4
    assert "C1002" in input_data["customerId"].values
    assert "C1003" in input_data["customerId"].values


def test_build_reminder_list_csv_empty():
    """Verify build_reminder_list_csv handles empty dataframe gracefully with correct header."""
    csv_bytes = app.build_reminder_list_csv(pd.DataFrame())
    df_exported = pd.read_csv(io.BytesIO(csv_bytes))
    exact_operational_cols = [
        "customerId",
        "customerName",
        "MOBILE_NO",
        "itemId",
        "itemName",
        "last_purchase_date",
        "expected_refill_date",
        "reminder_date",
        "predicted_days_until_refill",
    ]
    assert list(df_exported.columns) == exact_operational_cols
    assert len(df_exported) == 0


def test_multi_stage_schedule_filtering_and_active_date():
    """Verify multi-stage schedule generation, stage breakdown, and 2026-09-21 trigger matching."""
    bundle = app.load_refill_model()
    raw_data = app.load_refill_data()
    el_df, _, _ = app.prepare_prediction_overview(raw_data, bundle)

    assert not el_df.empty
    assert "Reminder Date" in el_df.columns

    # Full 6-stage schedule
    full_sched = app.generate_reminder_schedule_table(el_df)
    assert not full_sched.empty
    assert "Reminder Stage" in full_sched.columns

    # Check 2026-09-21 multi-stage due records
    due_21 = full_sched[full_sched["Reminder Date"] == "2026-09-21"]
    assert len(due_21) > 0

    # Check CSV export from multi-stage view
    csv_bytes = app.build_reminder_list_csv(due_21)
    df_exported = pd.read_csv(io.BytesIO(csv_bytes))
    assert len(df_exported) == len(due_21)
    assert "customerId" in df_exported.columns
    assert "MOBILE_NO" in df_exported.columns



