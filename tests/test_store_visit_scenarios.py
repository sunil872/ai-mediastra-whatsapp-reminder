"""Store-visit scenarios through the real path: upload -> re-score -> reminder stages -> Med-Sync -> WhatsApp send.

Fixed on 10 Nov 2026 after scenario testing on a real patient (CHANDRA MOHAN, partial refill):
A  every stage of a cycle is sent (the bundle id carried only the refill date, so -1 / 0 / +2 / +5 looked
   "already delivered" after the first message);
B  a patient is name + mobile: another person with the same name never cancels / takes over reminders;
C  a return-only upload re-scores and brings the earlier schedule back;
D  a regular patient buying clearly fewer / more strips than usual gets a matching date;
E  separate bills of the same medicine on the same day are one purchase.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

MEDS = {"TELMA 40 TAB": "4060", "GLYCOMET GP 1 TAB": "5011", "ECOSPRIN AV 75 CAP": "6022"}
RAVI, RAVI_MOB, OTHER_MOB = "RAVI KUMAR", "9848012345", "9848099999"
PKEY = f"{RAVI}_{RAVI_MOB}"
START = date(2026, 3, 2)
LAST = START + timedelta(days=30 * 7)   # 8 monthly purchases of 2 strips
TODAY = LAST + timedelta(days=27)       # -3 stage day of the next refill


class FakeClient:
    template_name = "refillcare_medicine_reminder"
    api_key, waba_number = "test", "test"

    def __init__(self):
        self.sent = []

    def send_refill_reminder(self, phone_number, **kw):
        meds = kw.get("medications")
        meds = meds if isinstance(meds, list) else [m.strip() for m in str(meds).split(",")]
        self.sent.append({"meds": sorted(meds), "tier": kw.get("tier")})
        return {"success": True, "status_code": 200, "message": "ok", "message_id": f"M{len(self.sent)}"}


def _rows(day, items, mobile=RAVI_MOB, name=RAVI, inv="S0/1"):
    out = []
    for it in items:
        med, strips = it[0], it[1]
        out.append({"Bill No.": it[2] if len(it) > 2 else inv, "Bill Date": day.strftime("%d/%m/%Y"), "Party Name": name,
                    "Mobile": mobile, "Product": med, "Prod Code": MEDS[med], "Pack": "1X15", "Qty": str(strips), "MRP": "100"})
    return pd.DataFrame(out)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    import uuid

    from database.connection import SessionLocal
    from refillcare.data.incremental_ingest import ingest_sales_batch
    from refillcare.stores import clear_cache
    from refillcare.tenancy import use_store

    monkeypatch.setenv("REFILLCARE_DATA_DIR", str(tmp_path / "data"))
    with use_store(f"v-{uuid.uuid4().hex[:8]}"):
        db = SessionLocal()
        hist = pd.concat([_rows(START + timedelta(days=30 * k), [(m, 2) for m in MEDS], inv=f"S0/{100 + k}")
                          for k in range(8)], ignore_index=True)
        ingest_sales_batch(db, hist, "history.csv")
        yield db
        db.close()
    clear_cache()


def _upload(db, day, items, **kw):
    from refillcare.data.incremental_ingest import ingest_sales_batch

    return ingest_sales_batch(db, _rows(day, items, **kw), f"{day}-{kw.get('inv', 'x')}.csv", force=True)


def _send(db, day):
    from api.services import EnterpriseServices
    from refillcare.whatsapp.dispatcher import RefillWhatsAppDispatcher

    bundles, _, _, _ = EnterpriseServices(db)._outreach_bundles(target_date=day)
    mine = [b for b in bundles if b.customer_id == PKEY]
    fake = FakeClient()
    RefillWhatsAppDispatcher(client=fake).dispatch_medsync_batch(db, mine, dry_run=False, batch_limit=None,
                                                                 approve_before_send=True, approved_by="test")
    return fake.sent


def _stages(db, med):
    from database.models import ReminderCycleModel as C, ReminderStageModel as S

    return (db.query(S).join(C, S.cycle_id == C.cycle_id)
            .filter(S.customer_item_key == f"{PKEY}_{MEDS[med]}", C.is_active.is_(True),
                    S.status.in_(("PENDING", "DUE", "APPROVED"))).order_by(S.target_send_date).all())


def _decision(db, med, key=PKEY):
    from database.models import RefillDecisionModel as R

    return (db.query(R).filter(R.customer_item_key == f"{key}_{MEDS[med]}")
            .order_by(R.last_purchase_date.desc(), R.updated_at.desc()).first())


def test_a_every_stage_is_sent_once_and_partial_refill_reminds_only_the_rest(store):
    first = _send(store, TODAY)
    assert len(first) == 1 and len(first[0]["meds"]) == 3 and _send(store, TODAY) == []  # one message, sent once
    _upload(store, TODAY, [("TELMA 40 TAB", 2), ("GLYCOMET GP 1 TAB", 2)])         # buys 2 of the 3
    nxt = min(s.target_send_date for s in _stages(store, "ECOSPRIN AV 75 CAP") if s.target_send_date > TODAY)
    later = _send(store, nxt)
    assert later and later[0]["meds"] == ["ECOSPRIN AV 75 CAP"], "next stage is sent, only for the unbought medicine"


def test_b_same_name_other_mobile_is_another_patient(store):
    def future():  # stages after the upload day (stages up to the newest sales date expire as usual)
        return [(s.target_send_date, s.stage_offset) for s in _stages(store, "ECOSPRIN AV 75 CAP") if s.target_send_date > TODAY]

    before = future()
    _upload(store, TODAY, [("ECOSPRIN AV 75 CAP", 2)], mobile=OTHER_MOB, inv="S0/900")
    assert before and future() == before, "our patient's ECOSPRIN schedule is untouched"
    assert _decision(store, "ECOSPRIN AV 75 CAP").mobile_no.endswith(RAVI_MOB)
    assert _decision(store, "ECOSPRIN AV 75 CAP", key=f"{RAVI}_{OTHER_MOB}") is not None


def test_c_return_only_upload_restores_the_earlier_schedule(store):
    _upload(store, TODAY, [("TELMA 40 TAB", 2)], inv="S0/901")
    assert _stages(store, "TELMA 40 TAB")[0].target_send_date > TODAY + timedelta(days=14)
    res = _upload(store, TODAY + timedelta(days=1), [("TELMA 40 TAB", -2)], inv="S0/902")
    assert res["online_sync"]["customer_items_rescored"] > 0
    assert _stages(store, "TELMA 40 TAB")[0].target_send_date <= TODAY + timedelta(days=5), "due again now"


def test_d_one_strip_instead_of_two_halves_the_interval(store):
    _upload(store, TODAY, [("TELMA 40 TAB", 1)], inv="S0/903")
    d = _decision(store, "TELMA 40 TAB")
    assert d.expected_refill_date <= TODAY + timedelta(days=18), d.decision_reason  # ~15 days, not ~30


def test_e_two_bills_same_day_are_one_purchase(store):
    _upload(store, TODAY, [("TELMA 40 TAB", 1, "S0/904"), ("TELMA 40 TAB", 1, "S0/905")])
    d = _decision(store, "TELMA 40 TAB")
    assert d.units_purchased == 30 and d.purchase_count == 9  # 8 earlier + 1 (not 2) purchases of 2 strips
