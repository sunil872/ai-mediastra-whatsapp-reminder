"""Unit tests for Medication Switch Detection in RefillCare.

Tests:
1. X -> Y switch (e.g., Glycomet GP 0.5 -> Glycomet GP 1)
2. Unrelated X -> Y medicine
3. First Y purchase (P1: 100% X consumption velocity + Y units)
4. Second Y purchase (P2: 50/50 progressive blending)
5. Multiple concurrent medicines (multi-therapy protection against false positive switches)
6. Same customer with multiple medicines (switching one medicine while keeping another active)
7. Brand stem and dosage extraction
8. Relationship classification
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pytest

from refillcare.features.medication_switch import (
    extract_brand_stem,
    extract_medicine_strength,
    classify_medicine_relationship,
    detect_medication_switches_for_customer,
    calculate_switched_medication_refill,
    CLIENT_LABEL_MEDICATION_CHANGE,
)


def test_extract_brand_stem_and_strength():
    """Verify brand stem normalization and strength extraction across common formulations."""
    # Glycomet GP 0.5 vs Glycomet GP 1
    assert extract_brand_stem("GLYCOMET GP 0.5 TAB") == "glycomet gp"
    assert extract_brand_stem("GLYCOMET GP 1") == "glycomet gp"
    assert extract_medicine_strength("GLYCOMET GP 0.5 TAB") == "0.5"
    assert extract_medicine_strength("GLYCOMET GP 1") == "1"

    # Telma 40 vs Telma 80
    assert extract_brand_stem("TELMA 40MG TAB") == "telma"
    assert extract_brand_stem("TELMA 80 MG") == "telma"
    assert extract_medicine_strength("TELMA 40MG TAB") == "40"
    assert extract_medicine_strength("TELMA 80 MG") == "80"

    # Janumet 50/500 vs Janumet 50/1000
    assert extract_brand_stem("JANUMET 50/500 TABLETS") == "janumet"
    assert extract_brand_stem("JANUMET 50/1000 ER") == "janumet"
    assert extract_medicine_strength("JANUMET 50/500 TABLETS") == "50/500"
    assert extract_medicine_strength("JANUMET 50/1000 ER") == "50/1000"


def test_classify_medicine_relationship():
    """Verify classification of same-brand, same-salt, same-category, and unrelated drugs."""
    # 1. Same brand family + dosage change
    item_x = {"itemName": "GLYCOMET GP 0.5", "salt_composition": "METFORMIN+GLIMEPIRIDE"}
    item_y = {"itemName": "GLYCOMET GP 1", "salt_composition": "METFORMIN+GLIMEPIRIDE"}
    rel1 = classify_medicine_relationship(item_x, item_y)
    assert rel1["relationship_type"] == "same_brand_family"
    assert rel1["is_related"] is True
    assert rel1["is_same_brand"] is True
    assert rel1["is_dosage_change"] is True

    # 2. Same salt composition, different brand
    item_a = {"itemName": "TELMA 40", "salt_composition": "TELMISARTAN", "salt_category": "CARDIAC"}
    item_b = {"itemName": "TELMIKIND 40", "salt_composition": "TELMISARTAN", "salt_category": "CARDIAC"}
    rel2 = classify_medicine_relationship(item_a, item_b)
    assert rel2["relationship_type"] == "same_salt_composition"
    assert rel2["is_related"] is True
    assert rel2["is_same_salt"] is True

    # 3. Same therapeutic category
    item_c = {"itemName": "TELMA 40", "salt_composition": "TELMISARTAN", "salt_category": "ANTI-HYPERTENSIVE"}
    item_d = {"itemName": "AMLONG 5", "salt_composition": "AMLODIPINE", "salt_category": "ANTI-HYPERTENSIVE"}
    rel3 = classify_medicine_relationship(item_c, item_d)
    assert rel3["relationship_type"] == "same_salt_category"
    assert rel3["is_related"] is True

    # 4. Completely unrelated medicines
    item_e = {"itemName": "PAN 40", "salt_composition": "PANTOPRAZOLE", "salt_category": "GASTRO"}
    item_f = {"itemName": "AUGMENTIN 625", "salt_composition": "AMOXICILLIN+CLAV", "salt_category": "ANTIBIOTIC"}
    rel4 = classify_medicine_relationship(item_e, item_f)
    assert rel4["relationship_type"] == "unrelated"
    assert rel4["is_related"] is False


def test_detect_medication_switch_x_to_y():
    """Verify detection of a true medication switch (Glycomet GP 0.5 -> Glycomet GP 1)."""
    # Customer buys Glycomet GP 0.5 on Jan 1, Feb 1, Mar 1 (3 purchases), then switches to Glycomet GP 1 on Apr 1
    records = [
        {"customerId": "CUST_001", "itemId": "ITEM_G05", "itemName": "GLYCOMET GP 0.5", "invoice_date": "2026-01-01", "quantity": 2, "packing": "1X15", "invoice_number": "INV1"},
        {"customerId": "CUST_001", "itemId": "ITEM_G05", "itemName": "GLYCOMET GP 0.5", "invoice_date": "2026-02-01", "quantity": 2, "packing": "1X15", "invoice_number": "INV2"},
        {"customerId": "CUST_001", "itemId": "ITEM_G05", "itemName": "GLYCOMET GP 0.5", "invoice_date": "2026-03-01", "quantity": 2, "packing": "1X15", "invoice_number": "INV3"},
        {"customerId": "CUST_001", "itemId": "ITEM_G1", "itemName": "GLYCOMET GP 1", "invoice_date": "2026-04-01", "quantity": 2, "packing": "1X15", "invoice_number": "INV4"},
    ]
    df = pd.DataFrame(records)

    switches = detect_medication_switches_for_customer(df)
    assert len(switches) == 1
    sw = switches[0]

    assert sw["customerId"] == "CUST_001"
    assert sw["previous_item_id"] == "ITEM_G05"
    assert sw["new_item_id"] == "ITEM_G1"
    assert sw["switch_date"] == "2026-04-01"
    assert sw["relationship_type"] == "same_brand_family"
    assert sw["is_same_brand"] is True
    assert sw["is_dosage_change"] is True
    assert sw["client_status_label"] == CLIENT_LABEL_MEDICATION_CHANGE
    assert sw["previous_purchase_count"] == 3
    assert sw["previous_historical_consumption_rate"] is not None
    assert sw["new_total_units"] == 30.0


def test_unrelated_x_to_y_medicine_not_switched():
    """Verify that unrelated acute medication (e.g. antibiotic) is not treated as a switch."""
    # Customer buys Glycomet GP 0.5 regularly, and buys Augmentin 625 once for acute infection
    records = [
        {"customerId": "CUST_002", "itemId": "ITEM_G05", "itemName": "GLYCOMET GP 0.5", "invoice_date": "2026-01-01", "quantity": 2, "packing": "1X15", "invoice_number": "INV1", "salt_category": "DIABETES"},
        {"customerId": "CUST_002", "itemId": "ITEM_G05", "itemName": "GLYCOMET GP 0.5", "invoice_date": "2026-02-01", "quantity": 2, "packing": "1X15", "invoice_number": "INV2", "salt_category": "DIABETES"},
        {"customerId": "CUST_002", "itemId": "ITEM_AUG", "itemName": "AUGMENTIN 625", "invoice_date": "2026-02-15", "quantity": 1, "packing": "1X10", "invoice_number": "INV3", "salt_category": "ANTIBIOTIC"},
    ]
    df = pd.DataFrame(records)

    switches = detect_medication_switches_for_customer(df)
    # Augmentin is unrelated -> no switch detected
    assert len(switches) == 0


def test_first_y_purchase_uses_x_velocity_and_y_units():
    """Verify that 1st Y purchase uses Y's own units and X's consumption velocity as supporting context."""
    # X history: 60 tablets consumed across 60 days -> rate = 1.0 tablet/day
    switch_info = {
        "customerId": "CUST_003",
        "previous_item_id": "ITEM_G05",
        "previous_item_name": "GLYCOMET GP 0.5",
        "new_item_id": "ITEM_G1",
        "new_item_name": "GLYCOMET GP 1",
        "relationship_type": "same_brand_family",
        "previous_historical_consumption_rate": 1.0,
    }

    # First purchase on Y: buys 4 packs of 1X15 = 60 tablets
    current_y_record = {
        "customerId": "CUST_003",
        "itemId": "ITEM_G1",
        "itemName": "GLYCOMET GP 1",
        "invoice_date": "2026-04-01",
        "quantity": 4,
        "packing": "1X15",
    }

    res = calculate_switched_medication_refill(switch_info, current_y_record, y_purchase_seq=1)

    assert res["prediction_status"] == "eligible"
    assert res["total_units_purchased"] == 60.0
    assert res["estimated_daily_consumption"] == 1.0
    assert res["estimated_days_of_supply"] == 60.0  # 60 units / 1.0 per day = 60 days
    assert res["predicted_days_until_refill"] == 60.0
    assert res["expected_refill_date"] == "2026-05-31"  # 2026-04-01 + 60 days
    assert res["reminder_date"] == "2026-05-29"  # 60 - 2 buffer = 58 days offset -> 2026-05-29
    assert res["client_status_label"] == CLIENT_LABEL_MEDICATION_CHANGE
    assert res["switch_detected"] is True
    assert res["prediction_strategy"] == "switch_p1_x_prior"


def test_second_y_purchase_progressive_blending():
    """Verify that 2nd Y purchase blends Y's newly observed interval with X's supporting context."""
    switch_info = {
        "customerId": "CUST_004",
        "previous_item_id": "ITEM_T40",
        "previous_item_name": "TELMA 40",
        "new_item_id": "ITEM_T80",
        "new_item_name": "TELMA 80",
        "relationship_type": "same_brand_family",
        "previous_historical_consumption_rate": 1.0,  # X rate = 1.0 unit/day
    }

    # Y purchase history:
    # Purchase 1: 2026-04-01, 30 units (1X30, qty 1)
    # Purchase 2: 2026-04-21 (20 days later), 30 units
    # Y observed rate over elapsed 20 days = 30 units / 20 days = 1.5 units/day
    y_history = [
        {"customerId": "CUST_004", "itemId": "ITEM_T80", "invoice_date": "2026-04-01", "quantity": 1, "packing": "1X30"},
        {"customerId": "CUST_004", "itemId": "ITEM_T80", "invoice_date": "2026-04-21", "quantity": 1, "packing": "1X30"},
    ]
    y_df = pd.DataFrame(y_history)

    current_y_record = y_history[1]

    res = calculate_switched_medication_refill(
        switch_info=switch_info,
        current_y_record=current_y_record,
        y_purchase_seq=2,
        y_history_df=y_df,
    )

    # Blended rate: 0.5 * 1.5 (Y) + 0.5 * 1.0 (X) = 1.25 units/day
    assert res["prediction_status"] == "eligible"
    assert res["estimated_daily_consumption"] == 1.25
    # Days of supply: 30 units / 1.25 = 24.0 days
    assert res["estimated_days_of_supply"] == 24.0
    assert res["predicted_days_until_refill"] == 24.0
    assert res["expected_refill_date"] == "2026-05-15"  # 2026-04-21 + 24 days
    assert res["reminder_date"] == "2026-05-13"  # 24 - 2 buffer = 22 days -> 2026-05-13
    assert res["client_status_label"] == CLIENT_LABEL_MEDICATION_CHANGE
    assert res["prediction_strategy"] == "switch_p2_blend_50_50"


def test_multiple_concurrent_medicines_not_switches():
    """Verify that patients purchasing multiple chronic medications concurrently are recognized as multi-therapy, not false switches."""
    # Customer regularly buys Metformin and Telmisartan together on every visit
    records = [
        {"customerId": "CUST_005", "itemId": "METFORMIN", "itemName": "GLYCOMET 500", "invoice_date": "2026-01-01", "quantity": 2, "packing": "1X15", "invoice_number": "INV1"},
        {"customerId": "CUST_005", "itemId": "TELMI", "itemName": "TELMA 40", "invoice_date": "2026-01-01", "quantity": 1, "packing": "1X30", "invoice_number": "INV1"},
        {"customerId": "CUST_005", "itemId": "METFORMIN", "itemName": "GLYCOMET 500", "invoice_date": "2026-02-01", "quantity": 2, "packing": "1X15", "invoice_number": "INV2"},
        {"customerId": "CUST_005", "itemId": "TELMI", "itemName": "TELMA 40", "invoice_date": "2026-02-01", "quantity": 1, "packing": "1X30", "invoice_number": "INV2"},
        {"customerId": "CUST_005", "itemId": "METFORMIN", "itemName": "GLYCOMET 500", "invoice_date": "2026-03-01", "quantity": 2, "packing": "1X15", "invoice_number": "INV3"},
        {"customerId": "CUST_005", "itemId": "TELMI", "itemName": "TELMA 40", "invoice_date": "2026-03-01", "quantity": 1, "packing": "1X30", "invoice_number": "INV3"},
    ]
    df = pd.DataFrame(records)

    switches = detect_medication_switches_for_customer(df)
    # Both items are concurrent co-prescriptions -> 0 switches detected
    assert len(switches) == 0


def test_same_customer_multi_medicine_with_single_switch():
    """Verify customer on Drug A and Drug B where Drug A is switched to A2 while Drug B continues."""
    records = [
        # Drug A (Glycomet GP 0.5) bought Jan, Feb, Mar
        {"customerId": "CUST_006", "itemId": "ITEM_A1", "itemName": "GLYCOMET GP 0.5", "invoice_date": "2026-01-01", "quantity": 2, "packing": "1X15", "invoice_number": "INV1"},
        {"customerId": "CUST_006", "itemId": "ITEM_B", "itemName": "TELMA 40", "invoice_date": "2026-01-01", "quantity": 1, "packing": "1X30", "invoice_number": "INV1"},
        {"customerId": "CUST_006", "itemId": "ITEM_A1", "itemName": "GLYCOMET GP 0.5", "invoice_date": "2026-02-01", "quantity": 2, "packing": "1X15", "invoice_number": "INV2"},
        {"customerId": "CUST_006", "itemId": "ITEM_B", "itemName": "TELMA 40", "invoice_date": "2026-02-01", "quantity": 1, "packing": "1X30", "invoice_number": "INV2"},
        {"customerId": "CUST_006", "itemId": "ITEM_A1", "itemName": "GLYCOMET GP 0.5", "invoice_date": "2026-03-01", "quantity": 2, "packing": "1X15", "invoice_number": "INV3"},
        {"customerId": "CUST_006", "itemId": "ITEM_B", "itemName": "TELMA 40", "invoice_date": "2026-03-01", "quantity": 1, "packing": "1X30", "invoice_number": "INV3"},
        # On Apr 1, Drug A switches to Glycomet GP 1, while Drug B (Telma 40) continues as normal
        {"customerId": "CUST_006", "itemId": "ITEM_A2", "itemName": "GLYCOMET GP 1", "invoice_date": "2026-04-01", "quantity": 2, "packing": "1X15", "invoice_number": "INV4"},
        {"customerId": "CUST_006", "itemId": "ITEM_B", "itemName": "TELMA 40", "invoice_date": "2026-04-01", "quantity": 1, "packing": "1X30", "invoice_number": "INV4"},
    ]
    df = pd.DataFrame(records)

    switches = detect_medication_switches_for_customer(df)
    # Exactly 1 switch: ITEM_A1 -> ITEM_A2. Drug B (TELMA 40) is NOT switched
    assert len(switches) == 1
    sw = switches[0]
    assert sw["previous_item_id"] == "ITEM_A1"
    assert sw["new_item_id"] == "ITEM_A2"
    assert sw["relationship_type"] == "same_brand_family"
    assert sw["client_status_label"] == CLIENT_LABEL_MEDICATION_CHANGE
