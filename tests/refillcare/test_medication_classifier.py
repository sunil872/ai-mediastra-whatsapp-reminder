"""Unit Tests for Medication Classifier & Chronic Therapy Gate."""

from datetime import date
from typing import List, Optional
import pandas as pd
import pytest

from refillcare.data.medication_classifier import (
    classify_medication,
    is_chronic_medication,
    filter_chronic_transactions,
)
from refillcare.engine.unified_engine import UnifiedRefillDecisionEngine, STABILITY_UNSTABLE, PATH_INELIGIBLE


def test_generic_otc_fmcg_exclusions():
    """Verify that all common generic, OTC, FMCG, acute, and surgical items are correctly excluded."""
    non_chronic_items = [
        ("VICKS INHALER", "1X1"),
        ("DOLO 650", "1X15"),
        ("BORNOL CREAM", "20GM"),
        ("BOROLINE", "20GM"),
        ("ORS SACHET", "1SAC"),
        ("SHAMPOO 200ML", "200ML"),
        ("BANDAGE 5CM", "1X1"),
        ("LSDEW SOAP", "75GM"),
        ("HIMALAYA BABY SOAP", "125GM"),
        ("OMNIGEL 30GM", "30GM"),
        ("VOLINI GEL", "30GM"),
        ("CETAPHIL REST MOST LOTION", "295ML"),
        ("HIMALAYA BABY LOTION", "100ML"),
        ("FACE WASH 100ML", "100ML"),
        ("PONDS FACE CREAM", "50GM"),
        ("PROTEIN POWDER 500GM", "500GM"),
        ("THREPTIN BIS 1KG", "1KG"),
        ("SIMILAC IQ 1", "400GM"),
        ("PEDIASURE CHOC 1KG", "1KG"),
        ("CANDID POWDER 120GM", "120GM"),
        ("COTTON BUDS 100S", "100S"),
        ("COTTON ROLL 500GM", "500GM"),
        ("STERILE WATER 5ML", "1X5ML"),
        ("COLGATE TOOTH PASTE", "100GM"),
        ("ORAL B TOOTH BRUSH", "1X1"),
        ("PARACHUTE HAIR OIL 100ML", "100ML"),
        ("PONDS COLD CREAM 100ML", "100ML"),
        ("SURGICAL GLOVES 7.5", "1PAIR"),
        ("VACCINE INJ", "1VIAL"),
        ("FACE SERUM 30ML", "30ML"),
        ("ENO FRUIT SALT", "1SAC"),
        ("BABY WIPES 80S", "80S"),
        ("CETAPHIL GENTLE CLEANSER", "125ML"),
        ("MAMY POKO L 50", "1X50"),
        ("WHISPER CHOICE XL", "1X6"),
        ("PAMPERS M4", "1X4"),
    ]

    for item, pack in non_chronic_items:
        res = classify_medication(item, packing=pack)
        assert not res["is_chronic_eligible"], f"Expected {item} to be NON-CHRONIC, but got eligible=True ({res})"
        assert is_chronic_medication(item, packing=pack) is False


def test_chronic_medication_inclusions():
    """Verify that genuine chronic maintenance therapies are correctly classified as eligible."""
    chronic_items = [
        ("CONCOR 5MG TAB", "1X10"),
        ("GLUCORYL M2 FORTE", "1X15"),
        ("ROZAT 5MG TAB", "1X15"),
        ("MIGRABETA PLUS TAB", "1X10"),
        ("TELMA 40MG TAB", "1X15"),
        ("THYRONORM 50MCG", "1X100"),
        ("INTAGLIP M FORTE TAB", "1X10"),
        ("AMARYL M2 TAB", "1X20"),
        ("PROTHIADEN 75MG TAB", "1X15"),
        ("ECOSPRIN 75MG TAB", "1X14"),
        ("AMLONG 5MG TAB", "1X15"),
        ("CILACAR 10MG TAB", "1X15"),
        ("GLYCIPHAGE 500MG TAB", "1X20"),
        ("REVLAMER 400MG TAB", "1X10"),
        ("STARPRESS XL 25 TAB", "1X15"),
        ("MET-XL 50 TAB", "1X15"),
    ]

    for item, pack in chronic_items:
        res = classify_medication(item, packing=pack)
        assert res["is_chronic_eligible"], f"Expected {item} to be CHRONIC, but got eligible=False ({res})"
        assert is_chronic_medication(item, packing=pack) is True


def test_filter_chronic_transactions():
    """Verify DataFrame filtering retains only chronic rows."""
    df = pd.DataFrame({
        "customerId": ["C1", "C2", "C3", "C4"],
        "itemName": ["CONCOR 5MG TAB", "DOLO 650", "GLUCORYL M2 FORTE", "LSDEW SOAP"],
        "packing": ["1X10", "1X15", "1X15", "75GM"],
        "quantity": [2, 1, 3, 2],
    })

    filtered = filter_chronic_transactions(df, item_col="itemName", packing_col="packing")
    assert len(filtered) == 2
    assert set(filtered["itemName"]) == {"CONCOR 5MG TAB", "GLUCORYL M2 FORTE"}


def test_unified_engine_excludes_non_chronic_trajectory():
    """Verify that unified decision engine marks non-chronic trajectories as ineligible."""
    engine = UnifiedRefillDecisionEngine()

    # Case 1: Repeat buyer of Dolo 650 (10 purchases)
    dates = [
        date(2026, 1, 1), date(2026, 1, 20), date(2026, 2, 15),
        date(2026, 3, 10), date(2026, 4, 5), date(2026, 5, 1),
        date(2026, 5, 25), date(2026, 6, 20), date(2026, 7, 15), date(2026, 8, 10)
    ]
    quantities = [1.0] * 10
    packings: List[Optional[str]] = ["1X15"] * 10

    dec = engine.evaluate_customer_item_trajectory(
        customer_id="CUST_101",
        customer_name="John Doe",
        mobile_no="9876543210",
        item_id="ITEM_DOLO",
        item_name="DOLO 650",
        dates=dates,
        quantities=quantities,
        packings=packings,
    )

    assert not dec.is_eligible
    assert dec.path == PATH_INELIGIBLE
    assert "Excluded:" in dec.decision_reason
    assert "ACUTE" in dec.decision_reason or "Non-chronic" in dec.decision_reason

    # Case 2: Repeat buyer of Concor 5mg (10 purchases) -> Should be eligible Path A
    dec_chronic = engine.evaluate_customer_item_trajectory(
        customer_id="CUST_102",
        customer_name="Jane Doe",
        mobile_no="9876543210",
        item_id="ITEM_CONCOR",
        item_name="CONCOR 5MG TAB",
        dates=dates,
        quantities=quantities,
        packings=packings,
    )

    assert dec_chronic.is_eligible
    assert dec_chronic.path == "PATH_A"
    assert dec_chronic.expected_refill_date is not None


def test_behavioral_and_population_consensus():
    """Verify personal recurrence and population consensus verification."""
    from refillcare.data.medication_classifier import (
        compute_item_population_chronic_stats,
        verify_medication_chronic_status,
    )

    # Mock transaction dataframe
    df = pd.DataFrame({
        "customerId": ["C1", "C1", "C2", "C2", "C3", "C3", "C4", "C4", "C5", "C5", "C6"],
        "itemName": [
            "NEW_GENERIC_BP_TAB", "NEW_GENERIC_BP_TAB",
            "NEW_GENERIC_BP_TAB", "NEW_GENERIC_BP_TAB",
            "NEW_GENERIC_BP_TAB", "NEW_GENERIC_BP_TAB",
            "NEW_GENERIC_BP_TAB", "NEW_GENERIC_BP_TAB",
            "NEW_GENERIC_BP_TAB", "NEW_GENERIC_BP_TAB",
            "ONE_TIME_ACUTE_SYP"
        ],
        "invoice_date": [
            "2026-01-01", "2026-02-01",
            "2026-01-10", "2026-02-12",
            "2026-01-15", "2026-02-14",
            "2026-01-20", "2026-02-18",
            "2026-01-25", "2026-02-27",
            "2026-01-05"
        ]
    })

    stats = compute_item_population_chronic_stats(df)
    assert "NEW_GENERIC_BP_TAB" in stats.index
    assert stats.loc["NEW_GENERIC_BP_TAB", "is_population_chronic_consensus"] == True
    assert stats.loc["ONE_TIME_ACUTE_SYP", "is_population_chronic_consensus"] == False

    # Verify multi-tiered verification helper
    ver_res = verify_medication_chronic_status(
        item_name="NEW_GENERIC_BP_TAB",
        patient_history_count=4,
        patient_median_interval=30.0,
        population_repeat_rate=0.80
    )
    assert ver_res["is_chronic_eligible"] == True
    assert "CHRONIC_BY_PERSONAL_RECURRENCE" in ver_res["category"]
