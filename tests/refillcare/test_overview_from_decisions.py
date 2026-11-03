"""The Streamlit overview reads the engine's stored decisions (fast path) with the legacy columns."""
from datetime import date

import app_refillcare as app
from database.connection import SessionLocal, init_store
from database.models import RefillDecisionModel
from refillcare.tenancy import current_store_id


def _decision(i, key, last, expected, eligible, reason, path="PATH_A"):
    return RefillDecisionModel(
        decision_id=f"d{i}", cycle_id=f"c{i}", customer_id="C1", item_id=key.split("|")[1], customer_item_key=key,
        customer_name="Asha", item_name="ATORLIP 20MG TAB", mobile_no="9848012345", path=path, purchase_count=7,
        is_eligible=eligible, stability_tier="HIGH", predicted_interval_days=30.0, dos_days=30.0,
        last_purchase_date=last, expected_refill_date=expected, decision_reason=reason,
    )


def test_overview_uses_latest_decision_per_medicine():
    sid = current_store_id()
    init_store(sid)
    with SessionLocal(sid) as db:
        db.query(RefillDecisionModel).delete()
        db.add_all([
            _decision(1, "C1|I1", date(2026, 8, 1), date(2026, 8, 31), False, "older cycle"),
            _decision(2, "C1|I1", date(2026, 9, 1), date(2026, 10, 1), True, "Path A median"),
            _decision(3, "C1|I2", date(2026, 9, 5), None, False, "Excluded: Non-chronic item"),
        ])
        db.commit()
    version = app._decisions_version.__wrapped__(sid)
    el, inel, metrics = app.prepare_overview_from_decisions.__wrapped__(sid, version)

    assert metrics["total_histories"] == 2 and metrics["eligible_count"] == 1 and metrics["ineligible_count"] == 1
    row = el.iloc[0]
    assert row["Expected Refill Date"] == "2026-10-01" and row["Reminder Date"] == "2026-09-29"
    assert row["Last Purchase Date"] == "2026-09-01" and row["Pilot Tier"].startswith("Tier A")
    assert row["Mobile Status"] == app.determine_mobile_status("9848012345")
    # Same columns the tabs read from the legacy on-the-fly overview
    for col in ("Customer Name", "Medicine", "Delivery Phone", "Predicted Interval (Days)", "reminder_date", "itemName"):
        assert col in el.columns
    assert inel.iloc[0]["Reason for Ineligibility"].startswith("Excluded")
