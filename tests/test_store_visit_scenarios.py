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


# --------------------------------------------------------------------------- returns (11 Nov 2026)
def _review(db, day):
    from refillcare.engine.pharmacist_overrides import review_candidates

    return [r for r in review_candidates(db, day, day) if r["customer_item_key"] == f"{PKEY}_{MEDS['TELMA 40 TAB']}"]


def test_f_full_return_after_15_days_pauses_reminders_and_asks_pharmacist(store):
    """Lakshmi Devi case: bought, returned ALL strips 15 days later -> doctor stopped / changed the medicine?"""
    _upload(store, TODAY, [("TELMA 40 TAB", 2)], inv="S0/910")
    ret_day = TODAY + timedelta(days=15)  # before today (future dates are rejected by the upload)
    _upload(store, ret_day, [("TELMA 40 TAB", -2)], inv="S0/911")
    assert _stages(store, "TELMA 40 TAB") == [], "no reminders (not even +45 re-engagement) for a returned medicine"
    rev = _review(store, ret_day)
    assert len(rev) == 1 and "Returned all strips" in rev[0]["reason_not_in_reminders"]
    # pharmacist confirms the patient still takes it -> back in reminders
    from refillcare.engine.pharmacist_overrides import add_to_reminders

    add_to_reminders(store, rev[0]["customer_item_key"], next_refill_date=ret_day + timedelta(days=3), today=ret_day)
    assert _stages(store, "TELMA 40 TAB"), "pharmacist addition overrides the pause"


def test_g_full_return_within_7_days_is_a_billing_correction(store):
    _upload(store, TODAY, [("TELMA 40 TAB", 2)], inv="S0/912")
    _upload(store, TODAY + timedelta(days=2), [("TELMA 40 TAB", -2)], inv="S0/913")
    assert _stages(store, "TELMA 40 TAB")[0].target_send_date <= TODAY + timedelta(days=5), "earlier schedule resumes"
    assert _review(store, TODAY + timedelta(days=2)) == []


def test_h_buying_again_after_a_late_return_resumes_normal_reminders(store):
    _upload(store, TODAY, [("TELMA 40 TAB", 2)], inv="S0/914")
    _upload(store, TODAY + timedelta(days=10), [("TELMA 40 TAB", -2)], inv="S0/915")   # (dates stay before today:
    assert _stages(store, "TELMA 40 TAB") == []                                         #  future dates are rejected)
    _upload(store, TODAY + timedelta(days=15), [("TELMA 40 TAB", 2)], inv="S0/916")
    assert _stages(store, "TELMA 40 TAB"), "new purchase -> reminders again"


def test_i_partial_return_late_only_reduces_the_strips(store):
    _upload(store, TODAY, [("TELMA 40 TAB", 2)], inv="S0/917")
    res = _upload(store, TODAY + timedelta(days=10), [("TELMA 40 TAB", -1)], inv="S0/918")
    assert res["data_warnings"]["returns_netted"]["returns_netted"] == 1
    assert res["data_warnings"]["returns_netted"]["sales_fully_returned"] == 0
    assert _stages(store, "TELMA 40 TAB"), "still takes it: reminders continue, earlier (1 strip left)"


def test_j_sales_lines_card_breaks_down_returns_and_invalid_rows(store):
    from refillcare.data import history_store as hs

    _upload(store, TODAY, [("TELMA 40 TAB", 2), ("GLYCOMET GP 1 TAB", 2)], inv="S0/920")
    _upload(store, TODAY + timedelta(days=2), [("GLYCOMET GP 1 TAB", -2)], inv="S0/921")   # correction (2 days)
    _upload(store, TODAY + timedelta(days=15), [("TELMA 40 TAB", -2)], inv="S0/922")       # late (15 days)
    c = hs.channel_counts(store)
    assert (c["return_lines"], c["returns_netted"], c["returns_unmatched"]) == (2, 2, 0)
    assert (c["sales_fully_returned"], c["returned_corrections"], c["returned_late"]) == (2, 1, 1)
    # 12 Nov 2026: return lines are stored as rows (every uploaded line accounted for)
    assert c["return_lines_stored"] == 2 and c["not_stored"] == c["invalid_rows"] == 0
    assert c["stored"] == c["sales_lines"] + c["return_lines_stored"] and c["return_window_days"] == 90


def test_k_identity_switch_leaves_one_row_per_patient_medicine(store, monkeypatch):
    """11 Nov 2026: after name -> name + mobile, old name-only decisions stayed and the review list showed every
    patient-medicine twice (A SUDHEER KUMAR_1678 and A SUDHEER KUMAR_8555914289_1678)."""
    from database.models import RefillDecisionModel as R
    from refillcare.data.incremental_ingest import rescore_customers
    from refillcare.engine.pharmacist_overrides import review_candidates
    from refillcare.engine.rekey import retire_stale_decisions
    from refillcare.whatsapp.pharmacist_views import search_patients

    monkeypatch.setenv("REFILLCARE_PATIENT_IDENTITY", "name")                    # history scored the old way
    rescore_customers(store, [RAVI])
    assert store.query(R).filter(R.customer_item_key == f"{RAVI}_{MEDS['TELMA 40 TAB']}").count() > 0
    monkeypatch.setenv("REFILLCARE_PATIENT_IDENTITY", "name_mobile")             # switch: the identity re-key
    from refillcare.data import history_store as hs
    from refillcare.engine.rekey import rekey_reminder_cycles

    rep = rekey_reminder_cycles(store, hs.load_all(store), apply=True)            # retires old cycles + decisions
    assert rep["stale_decisions"]["deleted"] == 3 and rep["stale_decisions"]["kept_active_cycle"] == 0
    assert retire_stale_decisions(store)["stale"] == 0                            # nothing left
    assert store.query(R).filter(R.customer_item_key == f"{RAVI}_{MEDS['TELMA 40 TAB']}").count() == 0
    keys = [r["customer_item_key"] for r in review_candidates(store)]
    assert len(keys) == len(set(keys)) and not any(k.startswith(f"{RAVI}_") and RAVI_MOB not in k for k in keys)
    hits = search_patients(store, RAVI)
    assert len(hits) == 1 and hits[0]["customer_id"] == PKEY                        # one row per person


def test_l_rolled_back_or_returned_purchase_leaves_no_stale_decision(store):
    from database.models import RefillDecisionModel as R

    _upload(store, TODAY, [("TELMA 40 TAB", 2)], inv="S0/930")
    _upload(store, TODAY + timedelta(days=15), [("TELMA 40 TAB", -2)], inv="S0/931")   # late full return
    key = f"{PKEY}_{MEDS['TELMA 40 TAB']}"
    assert store.query(R).filter(R.customer_item_key == key, R.last_purchase_date == TODAY).count() == 0
    latest = store.query(R).filter(R.customer_item_key == key).order_by(R.last_purchase_date.desc()).first()
    assert not latest.is_eligible and "Returned all strips" in latest.decision_reason   # still paused + reviewable


def test_m_control_group_is_off_by_default(monkeypatch):
    """11 Nov 2026: the client wants every patient reminded - no no-reminder control group unless configured."""
    from refillcare.whatsapp.holdout import holdout_pct, is_holdout

    monkeypatch.delenv("REFILLCARE_HOLDOUT_PCT", raising=False)
    assert holdout_pct() == 0
    assert not any(is_holdout(f"C{i}", f"98480{i:05d}") for i in range(500))


# ---- 12 Nov 2026: return lines are kept as rows and re-applied automatically ---------------------------

def _return_rows(db):
    from database.models import SalesTransactionModel as Line
    from refillcare.data import history_store as hs

    return db.query(Line).filter(Line.transaction_type == hs.RETURN_TYPE).all()


def test_n_return_lines_are_stored_never_counted_as_purchases(store):
    from refillcare.data import history_store as hs

    _upload(store, TODAY, [("TELMA 40 TAB", 2)], inv="S0/940")
    res = _upload(store, TODAY + timedelta(days=3), [("TELMA 40 TAB", -1)], inv="S0/941")
    m = res["metrics"]
    assert (m["return_lines"], m["return_lines_saved"], m["returns_applied"], m["returns_waiting"]) == (1, 1, 1, 0)
    rows = _return_rows(store)
    assert len(rows) == 1 and rows[0].quantity == -1 and not rows[0].refillcare_eligible
    assert rows[0].exclusion_reason.startswith(hs.RETURN_APPLIED + ":")
    hist = hs.load_all(store)
    assert (pd.to_numeric(hist["quantity"]) >= 0).all(), "loaders never return a return line"
    assert hs.store_summary(store)["sales_lines_stored"] == hs.sales_line_count(store)


def test_o_same_return_uploaded_twice_is_subtracted_once(store):
    from database.models import SalesTransactionModel as Line

    _upload(store, TODAY, [("TELMA 40 TAB", 3)], inv="S0/942")
    _upload(store, TODAY + timedelta(days=3), [("TELMA 40 TAB", -1)], inv="S0/943")
    again = _upload(store, TODAY + timedelta(days=3), [("TELMA 40 TAB", -1)], inv="S0/943")
    assert again["data_warnings"]["returns_netted"]["already_stored"] == 1
    sale = store.query(Line).filter(Line.invoice_number == "S0/942").one()
    assert sale.quantity == 2, "re-uploading a file must not subtract the return again"
    assert len(_return_rows(store)) == 1


def test_p_return_uploaded_before_its_sale_is_applied_when_the_sale_arrives(store):
    from database.models import SalesTransactionModel as Line
    from refillcare.data import history_store as hs

    # missed day: the return file (day +3) is uploaded first, the sale (day 0) afterwards
    first = _upload(store, TODAY + timedelta(days=3), [("ECOSPRIN AV 75 CAP", -2)], inv="S0/945", name="SITA DEVI", mobile=OTHER_MOB)
    assert first["metrics"]["returns_waiting"] == 1
    assert _return_rows(store)[0].exclusion_reason == hs.RETURN_WAITING
    later = _upload(store, TODAY, [("ECOSPRIN AV 75 CAP", 2)], inv="S0/944", name="SITA DEVI", mobile=OTHER_MOB)
    assert later["metrics"]["earlier_returns_applied_now"] == 1
    sale = store.query(Line).filter(Line.invoice_number == "S0/944").one()
    assert sale.quantity == 0 and not sale.refillcare_eligible
    assert _return_rows(store)[0].exclusion_reason.startswith(hs.RETURN_APPLIED)
    assert hs.channel_counts(store)["returns_unmatched"] == 0


def test_q_wider_return_window_applies_waiting_returns_without_re_upload(store, monkeypatch):
    from refillcare.data import history_store as hs
    from refillcare.data.incremental_ingest import recheck_returns

    monkeypatch.setenv("REFILLCARE_RETURN_WINDOW_DAYS", "5")
    _upload(store, TODAY, [("TELMA 40 TAB", 2)], inv="S0/946")
    _upload(store, TODAY + timedelta(days=10), [("TELMA 40 TAB", -1)], inv="S0/947")
    assert _return_rows(store)[0].exclusion_reason == hs.RETURN_WAITING           # outside the 5-day window
    monkeypatch.setenv("REFILLCARE_RETURN_WINDOW_DAYS", "90")
    out = recheck_returns(store)
    assert out["returns_applied"] == 1 and out["returns_still_waiting"] == 0
    assert _return_rows(store)[0].exclusion_reason.startswith(hs.RETURN_APPLIED)


def test_r_rolling_back_a_return_upload_gives_the_strips_back(store):
    from database.models import SalesTransactionModel as Line
    from refillcare.data.incremental_ingest import rollback_batch

    _upload(store, TODAY, [("TELMA 40 TAB", 2)], inv="S0/948")
    ret = _upload(store, TODAY + timedelta(days=2), [("TELMA 40 TAB", -2)], inv="S0/949")
    sale = store.query(Line).filter(Line.invoice_number == "S0/948").one()
    assert sale.quantity == 0 and not sale.refillcare_eligible
    res = rollback_batch(store, ret["import_batch_id"])
    store.expire_all()
    sale = store.query(Line).filter(Line.invoice_number == "S0/948").one()
    assert sale.quantity == 2 and sale.refillcare_eligible and sale.exclusion_reason is None
    assert res["reminder_state_reverted"]["returns_undone"]["sales_restored"] == 1
    assert _return_rows(store) == []
    assert _stages(store, "TELMA 40 TAB"), "the purchase counts again -> reminders again"


def test_s_rolling_back_the_sale_puts_its_return_back_to_waiting(store):
    from refillcare.data import history_store as hs
    from refillcare.data.incremental_ingest import rollback_batch

    sale = _upload(store, TODAY, [("TELMA 40 TAB", 2)], inv="S0/950")
    _upload(store, TODAY + timedelta(days=2), [("TELMA 40 TAB", -1)], inv="S0/951")
    rollback_batch(store, sale["import_batch_id"])
    store.expire_all()
    assert _return_rows(store)[0].exclusion_reason == hs.RETURN_WAITING


def test_t_skipped_rows_are_listed_with_a_plain_reason(store):
    bad = _rows(TODAY, [("TELMA 40 TAB", 2)], inv="S0/952")
    bad.loc[0, "Party Name"] = ""
    bad.loc[0, "Mobile"] = ""
    good = _rows(TODAY, [("ECOSPRIN AV 75 CAP", 2)], inv="S0/953")
    from refillcare.data.incremental_ingest import ingest_sales_batch

    res = ingest_sales_batch(store, pd.concat([good, bad], ignore_index=True), "mixed.csv", force=True)
    sk = res["skipped_rows_df"]
    assert len(sk) == 1 and sk.iloc[0]["medicine"] and sk.iloc[0]["reason"]
