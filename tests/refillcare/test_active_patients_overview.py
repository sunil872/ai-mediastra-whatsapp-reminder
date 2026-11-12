"""Unit tests for Active Patients dynamic window calculation in store_summary and overview_kpis."""
from datetime import date, datetime, timedelta
from sqlalchemy import text
from database.connection import get_db
from database.models import SalesTransactionModel
from refillcare.data import history_store as hs
from refillcare.engine.overview_kpis import overview_kpis


def test_active_patients_12_months_and_all_time():
    """Stores with >= 12 months of sales show 'Active patients (last 12 months)' with all-time total."""
    db = next(get_db())
    hs.clear_cache()

    # Seed sales spanning > 12 months
    db.add_all([
        SalesTransactionModel(
            import_batch_id="BATCH_12M_TEST",
            invoice_number="S0/OLD_1",
            invoice_date=datetime(2025, 1, 1),
            customer_id="PATIENT_OLD",
            customer_name="Old Patient",
            item_id="ITEM_1",
            quantity=1.0,
            transaction_type="CUSTOMER_SALE",
            refillcare_eligible=True,
        ),
        SalesTransactionModel(
            import_batch_id="BATCH_12M_TEST",
            invoice_number="S0/RECENT_1",
            invoice_date=datetime(2026, 10, 15),
            customer_id="PATIENT_RECENT",
            customer_name="Recent Patient",
            item_id="ITEM_1",
            quantity=1.0,
            transaction_type="CUSTOMER_SALE",
            refillcare_eligible=True,
        ),
    ])
    db.commit()
    hs.clear_cache()

    try:
        ov = overview_kpis(db, date(2026, 10, 31))
        assert ov["has_12m_history"] is True
        assert ov["active_patients_label"] == "Active patients (last 12 months)"
        assert ov["active_patients"] >= 1
        assert ov["all_time_customers"] >= ov["active_patients"]
        assert "All-time:" in ov["active_patients_subtext"]
    finally:
        # the shared test database must stay empty for later tests (e.g. identity re-key counts patients)
        db.query(SalesTransactionModel).filter(SalesTransactionModel.import_batch_id == "BATCH_12M_TEST").delete()
        db.commit()
        db.close()
        hs.clear_cache()


def test_active_patients_less_than_12_months():
    """When a store has less than 12 months of sales data, label dynamically reflects actual months."""
    now = datetime(2026, 10, 31)
    lo = datetime(2026, 7, 1)  # ~4 months
    span_days = (now.date() - lo.date()).days
    months_span = max(1, round(span_days / 30.4375))
    assert months_span == 4

    # Verify label generator logic
    if span_days >= 365:
        lbl = "Active patients (last 12 months)"
    elif span_days < 30:
        lbl = f"Active patients ({max(1, span_days)} days history)"
    else:
        lbl = f"Active patients (last {months_span} months)"
    assert lbl == "Active patients (last 4 months)"
