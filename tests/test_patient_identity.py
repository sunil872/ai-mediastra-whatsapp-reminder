"""Patient identity (name + mobile), salt-group therapy keys, and the reminder re-key cut-over."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from refillcare.data.identity import (
    apply_patient_identity,
    apply_therapy_grouping,
    mobile10,
    normalise_trajectory_keys,
    salt_molecules,
    therapy_item_id,
)
from refillcare.engine.unified_engine import UnifiedRefillDecisionEngine


def _rows(name, mobile, item, salt, start, n, gap=30, item_name="TELMA 40"):
    return [{"customerId": name, "customerName": name, "MOBILE_NO": mobile, "itemId": item, "itemName": item_name,
             "salt_composition": salt, "invoice_date": pd.Timestamp(start + timedelta(days=gap * k)),
             "quantity": 2, "packing": "1X15", "refillcare_eligible": True} for k in range(n)]


def test_mobile10_accepts_indian_mobiles_only():
    assert mobile10("+91 98480-12345") == "9848012345"
    assert mobile10("040 2345 6789") is None  # landline
    assert mobile10(None) is None and mobile10("12345") is None


def test_same_name_different_phones_are_different_patients():
    df = pd.DataFrame(_rows("LAXMI", "9848012345", "T40", "TELMISARTAN-40Mg [00]", date(2026, 1, 1), 3)
                      + _rows("LAXMI", "9701112222", "T40", "TELMISARTAN-40Mg [00]", date(2026, 1, 10), 3))
    out = apply_patient_identity(df, "name_mobile")
    assert set(out["customerId"]) == {"LAXMI_9848012345", "LAXMI_9701112222"}
    assert set(out["source_customer_id"]) == {"LAXMI"}
    assert apply_patient_identity(df, "name")["customerId"].eq("LAXMI").all()  # legacy mode unchanged


def test_missing_mobile_borrows_only_an_unambiguous_phone():
    one = _rows("RAVI", "9848012345", "T40", None, date(2026, 1, 1), 2) + _rows("RAVI", None, "T40", None, date(2026, 3, 1), 1)
    two = (_rows("SITA", "9848012345", "T40", None, date(2026, 1, 1), 1) + _rows("SITA", "9701112222", "T40", None, date(2026, 1, 5), 1)
           + _rows("SITA", None, "T40", None, date(2026, 3, 1), 1))
    out = apply_patient_identity(pd.DataFrame(one + two), "name_mobile")
    ravi = out[out["source_customer_id"] == "RAVI"]
    assert ravi["customerId"].eq("RAVI_9848012345").all()
    sita_no_phone = out[(out["source_customer_id"] == "SITA") & out["MOBILE_NO"].isna()]
    assert sita_no_phone["customerId"].tolist() == ["SITA"] and sita_no_phone["identity_source"].tolist() == ["name_only"]


def test_salt_molecules_ignore_brand_and_strength_but_keep_combinations():
    assert salt_molecules("TELMISARTAN-40Mg [00]") == salt_molecules("TELMISARTAN-80MG") == "TELMISARTAN"
    assert salt_molecules("METFORMIN-500Mg-SR [00]") == "METFORMIN"
    assert salt_molecules("AMLODIPINE-5Mg+TELMISARTAN-40Mg [00]") == "AMLODIPINE+TELMISARTAN"
    assert salt_molecules(None) is None


def test_sequential_brand_switch_becomes_one_trajectory():
    df = pd.DataFrame(_rows("P", "9848012345", "TELMA40", "TELMISARTAN-40Mg [00]", date(2026, 1, 1), 4)
                      + _rows("P", "9848012345", "TELMIKIND80", "TELMISARTAN-80MG", date(2026, 5, 1), 3, item_name="TELMIKIND 80"))
    out = apply_therapy_grouping(df, "salt")
    assert out["itemId"].nunique() == 1 and out["itemId"].iloc[0] == therapy_item_id("TELMISARTAN")
    assert set(out["brand_item_id"]) == {"TELMA40", "TELMIKIND80"}


def test_brands_taken_side_by_side_stay_separate():
    df = pd.DataFrame(_rows("P", "9848012345", "MET500", "METFORMIN-500Mg [00]", date(2026, 1, 1), 5)
                      + _rows("P", "9848012345", "MET1000", "METFORMIN-1000Mg [00]", date(2026, 1, 15), 5))
    out = apply_therapy_grouping(df, "salt")
    assert set(out["itemId"]) == {"MET500", "MET1000"}


def test_engine_continues_cycle_across_brand_switch(monkeypatch):
    monkeypatch.setenv("REFILLCARE_PATIENT_IDENTITY", "name_mobile")
    monkeypatch.setenv("REFILLCARE_ITEM_GROUPING", "salt")
    df = pd.DataFrame(_rows("P", "9848012345", "TELMA40", "TELMISARTAN-40Mg [00]", date(2026, 1, 1), 4)
                      + _rows("P", "9848012345", "TELMIKIND80", "TELMISARTAN-80MG", date(2026, 5, 1), 3, item_name="TELMIKIND 80"))
    _, decs = UnifiedRefillDecisionEngine(identity_guard=False).evaluate_all_customer_items(df, as_of_date=date(2026, 7, 2))
    assert len(decs) == 1
    d = decs[0]
    assert d.customer_id == "P_9848012345" and d.purchase_count == 7 and d.item_name == "TELMIKIND 80"
    assert d.is_eligible and d.predicted_interval_days == 30


def test_normalise_is_idempotent(monkeypatch):
    monkeypatch.setenv("REFILLCARE_PATIENT_IDENTITY", "name_mobile")
    monkeypatch.setenv("REFILLCARE_ITEM_GROUPING", "salt")
    df = pd.DataFrame(_rows("P", "9848012345", "TELMA40", "TELMISARTAN-40Mg [00]", date(2026, 1, 1), 3))
    once = normalise_trajectory_keys(df)
    twice = normalise_trajectory_keys(once)
    assert once["customerId"].tolist() == twice["customerId"].tolist() == ["P_9848012345"] * 3


def test_rekey_supersedes_old_cycles_and_never_resends(monkeypatch, tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from database.connection import Base
    from database.models import ReminderCycleModel, ReminderStageModel
    from refillcare.engine.decision_types import STATUS_CANCELLED, STATUS_PENDING, STATUS_SENT, STATUS_SUPERSEDED
    from refillcare.engine.persistence import RefillPersistenceManager
    from refillcare.engine.rekey import rekey_reminder_cycles

    eng = create_engine(f"sqlite:///{(tmp_path / 't.db').as_posix()}")
    Base.metadata.create_all(eng)
    db = sessionmaker(bind=eng)()
    rows = _rows("LAXMI", "9848012345", "T40", "TELMISARTAN-40Mg [00]", date(2026, 1, 1), 7)
    hist = pd.DataFrame(rows)
    as_of = date(2026, 7, 21)

    # Legacy run: name-only key, then the first stage (-7) is sent
    monkeypatch.setenv("REFILLCARE_PATIENT_IDENTITY", "name")
    monkeypatch.setenv("REFILLCARE_ITEM_GROUPING", "item")
    _, decs = UnifiedRefillDecisionEngine(identity_guard=False).evaluate_all_customer_items(hist, as_of_date=as_of)
    pm = RefillPersistenceManager()
    pm.save_refill_decisions(db, decs)
    pm.sync_reminder_cycles(db, decs)
    old = db.query(ReminderCycleModel).one()
    first = min(old.stages, key=lambda s: s.stage_offset)
    first.status = STATUS_SENT
    db.commit()

    monkeypatch.setenv("REFILLCARE_PATIENT_IDENTITY", "name_mobile")
    dry = rekey_reminder_cycles(db, hist, apply=False, as_of=as_of)
    assert dry["active_cycles_with_old_keys"] == 1 and dry["new_cycles_with_already_sent_stages"] == 1
    assert db.query(ReminderCycleModel).filter_by(is_active=True).count() == 1  # dry run wrote nothing

    res = rekey_reminder_cycles(db, hist, apply=True, as_of=as_of)
    assert res["applied"] and res["new_stages_cancelled_already_sent"] == 1
    active = db.query(ReminderCycleModel).filter_by(is_active=True).all()
    assert [c.customer_id for c in active] == ["LAXMI_9848012345"]
    old_open = [s for s in db.get(ReminderCycleModel, old.cycle_id).stages if s.status not in (STATUS_SENT,)]
    assert old_open and all(s.status == STATUS_SUPERSEDED for s in old_open)
    new_stages = db.query(ReminderStageModel).filter_by(cycle_id=active[0].cycle_id).all()
    by_off = {s.stage_offset: s.status for s in new_stages}
    assert by_off[first.stage_offset] == STATUS_CANCELLED  # already sent under the old key
    assert all(st == STATUS_PENDING for off, st in by_off.items() if off != first.stage_offset)
    db.close()


def test_check_identity_setting_warning(tmp_path, monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from database.models import Base, ReminderCycleModel
    from refillcare.data.identity import check_identity_setting_warning

    eng = create_engine(f"sqlite:///{(tmp_path / 'warn.db').as_posix()}")
    Base.metadata.create_all(eng)
    db = sessionmaker(bind=eng)()

    # Empty DB: returns None
    assert check_identity_setting_warning(db) is None

    # Populate 25 cycles with name_mobile keys
    for i in range(25):
        db.add(ReminderCycleModel(
            cycle_id=f"RC-LAXMI_{9848000000+i}-T40-20260101",
            customer_item_key=f"LAXMI_{9848000000+i}_T40",
            customer_id=f"LAXMI_{9848000000+i}",
            item_id="T40",
            is_active=True,
            last_purchase_date=date(2026, 1, 1),
            expected_refill_date=date(2026, 2, 1),
        ))
    db.commit()

    # In name_mobile mode: consistent -> None
    monkeypatch.setenv("REFILLCARE_PATIENT_IDENTITY", "name_mobile")
    assert check_identity_setting_warning(db) is None

    # In name mode: mismatch -> warning
    monkeypatch.setenv("REFILLCARE_PATIENT_IDENTITY", "name")
    warn = check_identity_setting_warning(db)
    assert warn is not None
    assert "REFILLCARE_PATIENT_IDENTITY is 'name', but active reminder cycles use 'name_mobile' keys" in warn

    # Replace cycles with name-only keys
    db.query(ReminderCycleModel).delete()
    for i in range(25):
        db.add(ReminderCycleModel(
            cycle_id=f"RC-LAXMI{i}-T40-20260101",
            customer_item_key=f"LAXMI{i}_T40",
            customer_id=f"LAXMI{i}",
            item_id="T40",
            is_active=True,
            last_purchase_date=date(2026, 1, 1),
            expected_refill_date=date(2026, 2, 1),
        ))
    db.commit()

    # In name mode: consistent -> None
    monkeypatch.setenv("REFILLCARE_PATIENT_IDENTITY", "name")
    assert check_identity_setting_warning(db) is None

    # In name_mobile mode: mismatch -> warning
    monkeypatch.setenv("REFILLCARE_PATIENT_IDENTITY", "name_mobile")
    warn2 = check_identity_setting_warning(db)
    assert warn2 is not None
    assert "REFILLCARE_PATIENT_IDENTITY is 'name_mobile', but active reminder cycles use legacy 'name' keys" in warn2

    db.close()

