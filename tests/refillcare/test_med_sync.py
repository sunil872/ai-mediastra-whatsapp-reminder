"""Unit tests for Med-Sync (Multi-Prescription Synchronization) and Quantile Uncertainty Envelopes."""

from datetime import date, timedelta
import pytest
import numpy as np
import pandas as pd

from refillcare.engine.med_sync import MedSyncEngine, MedSyncBundle, SyncedMedicationItem
from refillcare.engine.decision_types import RefillDecision


def test_med_sync_clustering_and_anchoring():
    """Test clustering multiple prescriptions for a single patient within a 7-day window."""
    engine = MedSyncEngine(sync_window_days=7)

    today = date(2026, 9, 28)
    
    # 3 active prescriptions for Rajesh:
    # 1. Telmisartan due in 10 days (HIGH stability)
    # 2. Metformin due in 12 days (MEDIUM-SAFE stability)
    # 3. Atorvastatin due in 14 days (HIGH stability)
    # 4. Antibiotic / acute eye drop due in 45 days (should be in a separate second cluster)
    decisions = [
        RefillDecision(
            customer_id="CUST001",
            customer_name="Rajesh Kumar",
            mobile_no="919876543210",
            item_id="MED_TELMI_40",
            item_name="Telmisartan 40mg",
            customer_item_key="CUST001_MED_TELMI_40",
            path="PATH_A",
            purchase_count=8,
            is_eligible=True,
            stability_tier="HIGH",
            cadence_median=30.0,
            cadence_norm_mad=0.15,
            cadence_drift=2.0,
            dos_days=30.0,
            units_purchased=30.0,
            prediction_method="PATH_A_PERSONAL_HISTORICAL_MEDIAN",
            predicted_interval_days=30,
            last_purchase_date=today - timedelta(days=20),
            expected_refill_date=today + timedelta(days=10),
            decision_reason="Path A High Stability",
            cycle_id="CYC_001",
            quantile_p10_date=today + timedelta(days=5),
            quantile_p90_date=today + timedelta(days=15),
        ),
        RefillDecision(
            customer_id="CUST001",
            customer_name="Rajesh Kumar",
            mobile_no="919876543210",
            item_id="MED_MET_500",
            item_name="Metformin 500mg",
            customer_item_key="CUST001_MED_MET_500",
            path="PATH_A",
            purchase_count=5,
            is_eligible=True,
            stability_tier="MEDIUM-SAFE",
            cadence_median=28.0,
            cadence_norm_mad=0.25,
            cadence_drift=3.0,
            dos_days=30.0,
            units_purchased=30.0,
            prediction_method="PATH_A_PERSONAL_HISTORICAL_MEDIAN",
            predicted_interval_days=28,
            last_purchase_date=today - timedelta(days=16),
            expected_refill_date=today + timedelta(days=12),
            decision_reason="Path A Medium Safe",
            cycle_id="CYC_002",
            quantile_p10_date=today + timedelta(days=7),
            quantile_p90_date=today + timedelta(days=17),
        ),
        RefillDecision(
            customer_id="CUST001",
            customer_name="Rajesh Kumar",
            mobile_no="919876543210",
            item_id="MED_ATORVA_10",
            item_name="Atorvastatin 10mg",
            customer_item_key="CUST001_MED_ATORVA_10",
            path="PATH_A",
            purchase_count=6,
            is_eligible=True,
            stability_tier="HIGH",
            cadence_median=30.0,
            cadence_norm_mad=0.10,
            cadence_drift=1.5,
            dos_days=30.0,
            units_purchased=30.0,
            prediction_method="PATH_A_PERSONAL_HISTORICAL_MEDIAN",
            predicted_interval_days=30,
            last_purchase_date=today - timedelta(days=16),
            expected_refill_date=today + timedelta(days=14),
            decision_reason="Path A High Stability",
            cycle_id="CYC_003",
            quantile_p10_date=today + timedelta(days=9),
            quantile_p90_date=today + timedelta(days=19),
        ),
        RefillDecision(
            customer_id="CUST001",
            customer_name="Rajesh Kumar",
            mobile_no="919876543210",
            item_id="MED_EYE_DROP",
            item_name="Refresh Tears Eye Drops",
            customer_item_key="CUST001_MED_EYE_DROP",
            path="PATH_B",
            purchase_count=2,
            is_eligible=True,
            stability_tier="UNSTABLE",
            cadence_median=45.0,
            cadence_norm_mad=0.60,
            cadence_drift=12.0,
            dos_days=45.0,
            units_purchased=1.0,
            prediction_method="PATH_B_DOS_EDS",
            predicted_interval_days=45,
            last_purchase_date=today,
            expected_refill_date=today + timedelta(days=45),
            decision_reason="Path B DOS",
            cycle_id="CYC_004",
            quantile_p10_date=today + timedelta(days=35),
            quantile_p90_date=today + timedelta(days=55),
        ),
    ]

    bundles = engine.cluster_patient_decisions(decisions, sync_window_days=7)

    # Should form 2 bundles: 1 multi-prescription bundle of 3 meds, and 1 single-med bundle
    assert len(bundles) == 2

    b1 = bundles[0]
    assert b1.customer_id == "CUST001"
    assert b1.total_items_count == 3
    assert b1.message_reduction_count == 2
    assert b1.anchor_item_name == "Telmisartan 40mg"
    assert b1.anchor_refill_date == today + timedelta(days=10)
    assert b1.earliest_p10_date == today + timedelta(days=5)
    assert b1.latest_p90_date == today + timedelta(days=19)
    assert "Synchronized Prescription Refill Notice" in b1.bundled_message_text
    assert "Telmisartan 40mg" in b1.bundled_message_text
    assert "Metformin 500mg" in b1.bundled_message_text
    assert "Atorvastatin 10mg" in b1.bundled_message_text

    b2 = bundles[1]
    assert b2.total_items_count == 1
    assert b2.message_reduction_count == 0
    assert b2.anchor_item_name == "Refresh Tears Eye Drops"

    # Impact summary
    impact = engine.summarize_sync_impact(bundles)
    assert impact["total_patients_analyzed"] == 1
    assert impact["total_prescriptions_synced"] == 4
    assert impact["total_dispatches_generated"] == 2
    assert impact["individual_messages_saved"] == 2
    assert impact["message_reduction_rate_pct"] == 50.0


def test_quantile_uncertainty_model_bundle_integrity():
    """Verify serialized chronic multi-quantile model bundle contains P10, P50, P90 regressors."""
    import joblib
    from pathlib import Path

    model_path = Path("data/refillcare/processed/models/chronic_refill_model.joblib")
    if not model_path.exists():
        pytest.skip("Model not yet trained in test environment.")

    bundle = joblib.load(model_path)
    assert "model_p10" in bundle
    assert "model_p50" in bundle
    assert "model_p90" in bundle
    assert "quantile_envelope" in bundle

    features = bundle["feature_cols"]
    sample_x = np.ones((5, len(features)), dtype=np.float32)

    p10 = bundle["model_p10"].predict(sample_x)
    p50 = bundle["model_p50"].predict(sample_x)
    p90 = bundle["model_p90"].predict(sample_x)

    assert len(p10) == 5
    assert len(p50) == 5
    assert len(p90) == 5


def test_med_sync_stage_tier_isolation():
    """Verify that DUE (-1d, 0d), FOLLOWUP (+2d, +5d), and LAPSED (+45d) items are strictly isolated into separate bundles."""
    engine = MedSyncEngine(sync_window_days=7)
    target = date(2026, 9, 30)

    # 1 customer with:
    # 2 Due items (Stage 0, -1)
    # 1 Follow-up item (Stage +5)
    # 1 Lapsed item (Stage +45)
    records = [
        {"customer_id": "C_TEST", "customer_name": "Kiran Kumar", "mobile_no": "9849950003", "item_id": "MED1", "item_name": "Telma 40", "expected_refill_date": target, "stage_offset": 0},
        {"customer_id": "C_TEST", "customer_name": "Kiran Kumar", "mobile_no": "9849950003", "item_id": "MED2", "item_name": "Concor 5", "expected_refill_date": target, "stage_offset": -1},
        {"customer_id": "C_TEST", "customer_name": "Kiran Kumar", "mobile_no": "9849950003", "item_id": "MED3", "item_name": "Ecosprin 75", "expected_refill_date": target, "stage_offset": 5},
        {"customer_id": "C_TEST", "customer_name": "Kiran Kumar", "mobile_no": "9849950003", "item_id": "MED4", "item_name": "Thyronorm 50", "expected_refill_date": target, "stage_offset": 45},
    ]

    bundles = engine.cluster_patient_decisions(records, sync_window_days=7)

    # Must produce 3 distinct bundles (DUE, FOLLOWUP, LAPSED) — NEVER MERGED TOGETHER
    assert len(bundles) == 3
    tier_map = {b.lifecycle_tier: b for b in bundles}
    assert "DUE" in tier_map
    assert "FOLLOWUP" in tier_map
    assert "LAPSED" in tier_map

    # Check DUE bundle
    b_due = tier_map["DUE"]
    assert b_due.total_items_count == 2
    assert "Synchronized Prescription Refill Notice" in b_due.bundled_message_text
    assert "Telma 40" in b_due.bundled_message_text
    assert "Concor 5" in b_due.bundled_message_text

    # Check FOLLOWUP bundle
    b_fup = tier_map["FOLLOWUP"]
    assert b_fup.total_items_count == 1
    assert "follow-up" in b_fup.bundled_message_text.lower()
    assert "Ecosprin 75" in b_fup.bundled_message_text

    # Check LAPSED bundle
    b_lapsed = tier_map["LAPSED"]
    assert b_lapsed.total_items_count == 1
    assert "care check-in" in b_lapsed.bundled_message_text.lower()
    assert "Thyronorm 50" in b_lapsed.bundled_message_text


def test_med_sync_exact_same_day_window_zero():
    """Verify that sync_window_days=0 performs exact same-date grouping."""
    engine = MedSyncEngine(sync_window_days=0)
    target = date(2026, 9, 30)

    # Kiran with 9 medications scheduled on 2026-09-30, and 1 on 2026-10-05
    records = [
        {"customer_id": "C_KIRAN", "customer_name": "Kiran", "mobile_no": "9849950003", "item_id": f"MED_{i}", "item_name": f"Medication {i}", "expected_refill_date": target, "stage_offset": 0}
        for i in range(1, 10)
    ]
    records.append({
        "customer_id": "C_KIRAN", "customer_name": "Kiran", "mobile_no": "9849950003", "item_id": "MED_LATER", "item_name": "Medication Later", "expected_refill_date": date(2026, 10, 5), "stage_offset": 0
    })

    bundles = engine.cluster_patient_decisions(records, sync_window_days=0)

    # Should form 2 bundles: 1 bundle on 2026-09-30 (9 meds), 1 bundle on 2026-10-05 (1 med)
    assert len(bundles) == 2
    b_today = [b for b in bundles if b.anchor_refill_date == target][0]
    assert b_today.total_items_count == 9
    assert b_today.message_reduction_count == 8
    assert b_today.customer_name == "Kiran"


def test_med_sync_customer_phone_composite_key():
    """Verify that patients with identical names but different mobile numbers are not merged, and blank numbers work."""
    engine = MedSyncEngine(sync_window_days=7)
    target = date(2026, 9, 28)

    records = [
        # Ramesh A (mobile 9705606266)
        {"customer_id": "RAMESH", "customer_name": "RAMESH", "mobile_no": "9705606266", "item_id": "M1", "item_name": "EGLUCENT MIX 25 CART", "expected_refill_date": target},
        # Ramesh B (mobile 9133928333)
        {"customer_id": "RAMESH", "customer_name": "RAMESH", "mobile_no": "9133928333", "item_id": "M2", "item_name": "VILDAMAC M 50/ 500MG TAB", "expected_refill_date": target},
        # D Arun (mobile - / blank)
        {"customer_id": "D ARUN", "customer_name": "D ARUN", "mobile_no": "-", "item_id": "M3", "item_name": "NEUROBION FORTE TAB", "expected_refill_date": target},
        # Kiran with 2 meds under same mobile
        {"customer_id": "KIRAN", "customer_name": "KIRAN", "mobile_no": "9849950003", "item_id": "M4", "item_name": "CONCOR COR 2.5MG TAB", "expected_refill_date": target},
        {"customer_id": "KIRAN", "customer_name": "KIRAN", "mobile_no": "9849950003", "item_id": "M5", "item_name": "IVABRAD 5MG TAB", "expected_refill_date": target},
    ]

    bundles = engine.cluster_patient_decisions(records, sync_window_days=7)

    # 4 distinct patients: Ramesh A, Ramesh B, D Arun, Kiran (2 meds) -> 4 bundles total
    assert len(bundles) == 4

    ramesh_bundles = [b for b in bundles if b.customer_name == "RAMESH"]
    assert len(ramesh_bundles) == 2
    ramesh_mobiles = {b.mobile_no for b in ramesh_bundles}
    assert ramesh_mobiles == {"9705606266", "9133928333"}

    arun_b = [b for b in bundles if b.customer_name == "D ARUN"][0]
    assert arun_b.mobile_no == "-"
    assert arun_b.total_items_count == 1

    kiran_b = [b for b in bundles if b.customer_name == "KIRAN"][0]
    assert kiran_b.mobile_no == "9849950003"
    assert kiran_b.total_items_count == 2
    assert kiran_b.message_reduction_count == 1

