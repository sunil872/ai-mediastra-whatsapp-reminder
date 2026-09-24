"""Unit tests for RefillCare transaction cleaning and invoice aggregation.

Uses synthetic test fixtures only (no real customer data).
"""

import pandas as pd
import numpy as np
import pytest
from refillcare.data.cleaning import (
    load_raw_transactions,
    clean_transactions,
    aggregate_invoice_items,
)


@pytest.fixture
def synthetic_raw_csv(tmp_path):
    """Create a temporary synthetic CSV file with various edge cases."""
    csv_file = tmp_path / "synthetic_transactions.csv"
    data = """invoice_date,invoice_number,customerName,itemCode,itemName,packing,batchNo,mfgDate,expiryDate,quantity,freeQuantity,rate,saleRate,mrp,invoice_total,discountPercent,item_discount,gstAmount,netAmount,costPrice,MOBILE_NO,therapeuticCategory,generic_name,customerId,itemId
01/06/2025,INV/001,Alice Smith,101,Medicine Alpha,10 Tab,B001,,12/26,2,0,10.0,10.0,12.0,20.0,0,0,2.4,22.4,8.0,9876543210,Cardio,Alpha 10mg,ALICE_9876543210,101
01/06/2025,INV/001,Alice Smith,101,Medicine Alpha,10 Tab,B002,,12/26,3,1,10.0,10.0,12.0,30.0,0,0,3.6,33.6,8.0,9876543210,Cardio,Alpha 10mg,ALICE_9876543210,101
01/06/2025,INV/001,Alice Smith,101,Medicine Alpha,10 Tab,B001,,12/26,2,0,10.0,10.0,12.0,20.0,0,0,2.4,22.4,8.0,9876543210,Cardio,Alpha 10mg,ALICE_9876543210,101
15/07/2025,INV/002,Alice Smith,101,Medicine Alpha,10 Tab,B003,,12/26,2,0,10.0,10.0,12.0,20.0,0,0,2.4,22.4,8.0,9876543210,Cardio,Alpha 10mg,ALICE_9876543210,101
10/08/2025,INV/003,Bob Jones,102,Medicine Beta,20 Tab,B004,,06/26,1,0,50.0,50.0,60.0,50.0,0,0,6.0,56.0,40.0,9123456789,Derma,Beta 20mg,BOB_9123456789,102
12/08/2025,INV/004,,103,Medicine Gamma,15 Tab,B005,,08/26,1,0,30.0,30.0,35.0,30.0,0,0,3.6,33.6,25.0,9999999999,Gastro,Gamma 15mg,,103
14/08/2025,INV/005,Charlie Brown,104,Medicine Delta,10 Tab,B006,,09/26,1,0,15.0,15.0,20.0,15.0,0,0,1.8,16.8,10.0,9888888888,Neuro,Delta 10mg,CHARLIE_9888888888,
"""
    csv_file.write_text(data.strip(), encoding="utf-8")
    return csv_file


def test_load_raw_transactions_date_parsing(synthetic_raw_csv):
    """Verify DD/MM/YYYY date format parsing with dayfirst=True."""
    df = load_raw_transactions(synthetic_raw_csv)
    # Check row 1 date: 01/06/2025 -> June 1, 2025
    assert df["invoice_date"].iloc[0] == pd.Timestamp("2025-06-01")
    # Check row 4 date: 15/07/2025 -> July 15, 2025
    assert df["invoice_date"].iloc[3] == pd.Timestamp("2025-07-15")


def test_clean_transactions_drops_invalid_and_duplicates(synthetic_raw_csv):
    """Verify clean_transactions drops missing customerId, missing itemId, mfgDate, and exact duplicate rows."""
    raw_df = load_raw_transactions(synthetic_raw_csv)
    assert len(raw_df) == 7
    assert "mfgDate" in raw_df.columns

    cleaned = clean_transactions(raw_df)

    # mfgDate should be removed
    assert "mfgDate" not in cleaned.columns

    # Row 3 was exact duplicate of Row 1 -> dropped
    # Row 6 had missing customerId -> dropped
    # Row 7 had missing itemId -> dropped
    # Remaining rows: Row 1, Row 2, Row 4, Row 5 = 4 rows
    assert len(cleaned) == 4

    # All remaining records must have non-empty customerId and itemId
    assert cleaned["customerId"].notna().all()
    assert (cleaned["customerId"] != "").all()
    assert cleaned["itemId"].notna().all()
    assert (cleaned["itemId"] != "").all()


def test_aggregate_invoice_items():
    """Verify aggregation of duplicate invoice lines for same customer, invoice, and item."""
    df = pd.DataFrame([
        {
            "customerId": "CUST_001",
            "invoice_number": "INV_100",
            "invoice_date": pd.Timestamp("2025-06-01"),
            "itemId": "ITEM_A",
            "itemCode": "ITEM_A",
            "quantity": 2,
            "freeQuantity": 0,
            "netAmount": 20.0,
            "gstAmount": 2.4,
            "item_discount": 0.0,
            "rate": 10.0,
            "saleRate": 10.0,
            "mrp": 12.0,
            "batchNo": "BATCH_1",
            "customerName": "Customer One",
            "itemName": "Medicine Alpha",
            "MOBILE_NO": "9876543210",
        },
        {
            "customerId": "CUST_001",
            "invoice_number": "INV_100",
            "invoice_date": pd.Timestamp("2025-06-01"),
            "itemId": "ITEM_A",
            "itemCode": "ITEM_A",
            "quantity": 3,
            "freeQuantity": 1,
            "netAmount": 30.0,
            "gstAmount": 3.6,
            "item_discount": 5.0,
            "rate": 10.0,
            "saleRate": 10.0,
            "mrp": 12.0,
            "batchNo": "BATCH_2",
            "customerName": "Customer One",
            "itemName": "Medicine Alpha",
            "MOBILE_NO": "9876543210",
        },
    ])

    agg = aggregate_invoice_items(df)

    # Must result in exactly 1 purchase event
    assert len(agg) == 1
    row = agg.iloc[0]

    # Quantities and amounts must be summed
    assert row["quantity"] == 5
    assert row["freeQuantity"] == 1
    assert row["netAmount"] == 50.0
    assert row["gstAmount"] == 6.0
    assert row["item_discount"] == 5.0

    # Deterministic metadata preserved
    assert row["customerId"] == "CUST_001"
    assert row["itemId"] == "ITEM_A"
    assert row["batchNo"] == "BATCH_1"
    assert row["MOBILE_NO"] == "9876543210"


def test_legacy_phone1_header_renamed_to_mobile_no(tmp_path):
    """Old CSV exports with phone1/Phone1 must load as MOBILE_NO."""
    csv_file = tmp_path / "legacy_phone1.csv"
    csv_file.write_text(
        "invoice_date,invoice_number,customerName,itemCode,itemName,packing,batchNo,"
        "mfgDate,expiryDate,quantity,freeQuantity,rate,saleRate,mrp,invoice_total,"
        "discountPercent,item_discount,gstAmount,netAmount,costPrice,phone1,"
        "therapeuticCategory,generic_name,customerId,itemId\n"
        "01/06/2025,INV/001,Alice,101,Med A,10 Tab,B001,,12/26,1,0,10,10,12,10,0,0,1,11,8,"
        "9876543210,Cardio,Alpha,CUST_A,101\n",
        encoding="utf-8",
    )
    df = load_raw_transactions(csv_file)
    assert "MOBILE_NO" in df.columns
    assert "phone1" not in df.columns
    assert df.loc[0, "MOBILE_NO"] == "9876543210"
