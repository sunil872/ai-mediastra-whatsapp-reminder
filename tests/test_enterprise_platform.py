"""Multi-store, incremental uploads, jobs, model artifacts, gate and logging - the enterprise layer."""
from __future__ import annotations

import io
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture()
def store_db(tmp_path, monkeypatch):
    """A fresh store with its own database file and data folder."""
    import uuid

    from database.connection import SessionLocal, reset_engines_for_tests
    from refillcare.stores import clear_cache
    from refillcare.tenancy import use_store

    monkeypatch.setenv("REFILLCARE_DATA_DIR", str(tmp_path / "data"))
    sid = f"t-{uuid.uuid4().hex[:8]}"
    with use_store(sid):
        db = SessionLocal()
        yield db
        db.close()
    clear_cache()


def _day_file(day: date, rows=(("RAVI", "9848012345", "4060", "TELMA 40 TAB", 2),), prefix="S0/", start_no=1, mrp=True):
    data = []
    for k, (name, mob, code, item, qty) in enumerate(rows):
        r = {"Bill No.": f"{prefix}{start_no + k}", "Bill Date": day.strftime("%d/%m/%Y"), "Party Name": name, "Mobile": mob,
             "Product": item, "Prod Code": code, "Pack": "1X15", "Qty": str(qty)}
        if mrp:
            r["MRP"] = "120"
        data.append(r)
    return pd.DataFrame(data)


def _history_rows(n_days=8, gap=30):
    """A patient with a steady 30-day cadence, as a store file."""
    start = date(2026, 1, 5)
    return pd.concat([_day_file(start + timedelta(days=gap * k), start_no=100 + k) for k in range(n_days)], ignore_index=True)


# ----------------------------------------------------------------------------- incremental uploads
def test_one_day_upload_is_incremental_idempotent_and_backfills(store_db):
    from refillcare.data import history_store as hs
    from refillcare.data.incremental_ingest import ingest_sales_batch

    r1 = ingest_sales_batch(store_db, _history_rows(), "history.csv")
    assert r1["metrics"]["new_records_added"] == 8 and r1["metrics"]["ignored_columns"] == ["MRP"]
    day = date(2026, 9, 30)
    r2 = ingest_sales_batch(store_db, _day_file(day, start_no=500), "day.csv")
    assert r2["metrics"]["new_records_added"] == 1
    assert r2["online_sync"]["customer_items_rescored"] >= 1
    # Same lines again (no checksum given): nothing added
    r3 = ingest_sales_batch(store_db, _day_file(day, start_no=500), "day-again.csv")
    assert r3["metrics"]["new_records_added"] == 0 and r3["metrics"]["existing_duplicates_skipped"] == 1
    # Partial day then the full day: only the missing line is added (a second invoice the same day is kept)
    full = _day_file(day, rows=(("RAVI", "9848012345", "4060", "TELMA 40 TAB", 2), ("SITA", "9701112222", "5977", "GLYCOMET 500", 1)), start_no=500)
    r4 = ingest_sales_batch(store_db, full, "day-full.csv")
    assert r4["metrics"]["new_records_added"] == 1
    # A back-dated day arriving late is inserted in place
    r5 = ingest_sales_batch(store_db, _day_file(date(2026, 9, 10), start_no=700), "late.csv")
    assert r5["metrics"]["new_records_added"] == 1 and r5["data_warnings"].get("backdated_file")
    assert hs.history_count(store_db) == 11
    assert hs.date_bounds(store_db) == (date(2026, 1, 5), day)


def test_same_file_twice_is_refused_by_checksum(store_db):
    from refillcare.data.incremental_ingest import DuplicateUpload, ingest_sales_batch

    ingest_sales_batch(store_db, _history_rows(2), "a.csv", checksum="abc")
    with pytest.raises(DuplicateUpload):
        ingest_sales_batch(store_db, _history_rows(2), "a.csv", checksum="abc")
    assert ingest_sales_batch(store_db, _history_rows(2), "a.csv", checksum="abc", force=True)["metrics"]["new_records_added"] == 0


def test_gap_warning_and_data_status(store_db, monkeypatch):
    from refillcare.data.incremental_ingest import data_status, ingest_sales_batch

    ingest_sales_batch(store_db, _day_file(date(2026, 9, 1)), "d1.csv")
    r = ingest_sales_batch(store_db, _day_file(date(2026, 9, 5), start_no=9), "d5.csv")
    assert r["data_warnings"]["gap_before_file_days"] == 3
    st = data_status(store_db, today=date(2026, 9, 6))
    assert st["ok_to_send"] and st["days_behind"] == 1
    assert not data_status(store_db, today=date(2026, 9, 20))["ok_to_send"]


def test_implausible_dates_and_credit_notes_never_enter_history(store_db):
    from refillcare.data import history_store as hs
    from refillcare.data.incremental_ingest import ingest_sales_batch

    df = pd.concat([_day_file(date(2026, 9, 1)), _day_file(date(2062, 9, 1), start_no=2),
                    _day_file(date(2026, 9, 2), prefix="CN/", start_no=3)], ignore_index=True)
    r = ingest_sales_batch(store_db, df, "mixed.csv")
    assert r["metrics"]["new_records_added"] == 1
    assert hs.date_bounds(store_db)[1] == date(2026, 9, 1)


def test_returns_are_netted_against_the_sale(store_db):
    from database.models import SalesTransactionModel
    from refillcare.data.incremental_ingest import ingest_sales_batch

    ingest_sales_batch(store_db, _day_file(date(2026, 9, 1), rows=(("RAVI", "9848012345", "4060", "TELMA 40 TAB", 2),)), "sale.csv")
    ret = _day_file(date(2026, 9, 3), rows=(("RAVI", "9848012345", "4060", "TELMA 40 TAB", -2),), start_no=50)
    r = ingest_sales_batch(store_db, ret, "return.csv")
    assert r["data_warnings"]["returns_netted"]["sales_fully_returned"] == 1
    line = store_db.query(SalesTransactionModel).one()
    assert line.quantity == 0 and line.refillcare_eligible is False


def test_rollback_deletes_only_the_batch_and_rescores(store_db):
    from refillcare.data import history_store as hs
    from refillcare.data.incremental_ingest import ingest_sales_batch, rollback_batch

    ingest_sales_batch(store_db, _history_rows(), "history.csv")
    r = ingest_sales_batch(store_db, _day_file(date(2026, 9, 30), start_no=500), "day.csv")
    res = rollback_batch(store_db, r["import_batch_id"])
    assert res["records_removed"] == 1 and hs.history_count(store_db) == 8
    assert res["restored_latest_sales_date"] == "03-08-2026"
    with pytest.raises(ValueError):
        rollback_batch(store_db, r["import_batch_id"])


def test_failed_sync_is_recorded_and_retryable(store_db, monkeypatch):
    import refillcare.data.incremental_ingest as ii
    from database.models import ImportBatchModel

    def boom(*a, **k):
        raise RuntimeError("scoring exploded")

    monkeypatch.setattr(ii, "rescore_customers", boom)
    r = ii.ingest_sales_batch(store_db, _history_rows(2), "h.csv")
    b = store_db.get(ImportBatchModel, r["import_batch_id"])
    assert b.sync_status == "SYNC_FAILED" and "exploded" in b.sync_error
    assert not ii.data_status(store_db, today=date(2026, 2, 5))["ok_to_send"]
    monkeypatch.undo()
    assert "customer_items_rescored" in ii.sync_batch(store_db, r["import_batch_id"])
    assert store_db.get(ImportBatchModel, r["import_batch_id"]).sync_status == "SYNCED"


def test_live_send_blocked_on_stale_data(store_db, monkeypatch):
    from refillcare.data.incremental_ingest import ingest_sales_batch
    from refillcare.whatsapp.dispatcher import StaleSalesData, check_live_send_allowed

    monkeypatch.setenv("REFILLCARE_STALE_DATA_GUARD", "true")
    ingest_sales_batch(store_db, _day_file(date(2026, 1, 1)), "old.csv")
    with pytest.raises(StaleSalesData):
        check_live_send_allowed(store_db)
    assert check_live_send_allowed(store_db, allow_stale=True)["stale"]


# ----------------------------------------------------------------------------- multi-store isolation
def test_two_stores_never_see_each_others_data(tmp_path, monkeypatch):
    from database.connection import SessionLocal
    from refillcare.data import history_store as hs
    from refillcare.data.incremental_ingest import ingest_sales_batch
    from refillcare.stores import register_store, store_setting
    from refillcare.tenancy import use_store

    monkeypatch.setenv("REFILLCARE_DATA_DIR", str(tmp_path / "data"))
    register_store("iso-a", tenant_id="t1", store_name="Store A", settings={"REFILLCARE_PATIENT_IDENTITY": "name_mobile"})
    register_store("iso-b", tenant_id="t1", store_name="Store B")
    with use_store("iso-a"), SessionLocal() as db:
        ingest_sales_batch(db, _history_rows(3), "a.csv")
        assert store_setting("REFILLCARE_PATIENT_IDENTITY") == "name_mobile"
    with use_store("iso-b"), SessionLocal() as db:
        assert hs.history_count(db) == 0
        assert store_setting("REFILLCARE_PATIENT_IDENTITY", "name") == "name"
    with use_store("iso-a"), SessionLocal() as db:
        assert hs.history_count(db) == 3


def test_store_ids_are_validated():
    from refillcare.tenancy import InvalidStoreId, normalise_store_id, schema_for_store

    assert schema_for_store("NGOs-Colony") == "store_ngos_colony"
    for bad in ("", "a b", "x;drop", "../etc", "a" * 41):
        with pytest.raises(InvalidStoreId):
            normalise_store_id(bad)


def test_api_routes_requests_to_the_header_store(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from api.main import app

    monkeypatch.setenv("REFILLCARE_DATA_DIR", str(tmp_path / "data"))
    c = TestClient(app)
    csv = _history_rows(3).to_csv(index=False).encode()
    r = c.post("/api/v1/sales/ingest", files={"file": ("h.csv", io.BytesIO(csv), "text/csv")}, headers={"X-Store-Id": "api-a"})
    assert r.status_code == 200, r.text
    assert r.json()["records_inserted"] == 3
    dup = c.post("/api/v1/sales/ingest", files={"file": ("h.csv", io.BytesIO(csv), "text/csv")}, headers={"X-Store-Id": "api-a"})
    assert dup.status_code == 409
    st_a = c.get("/api/v1/sales/data-status", headers={"X-Store-Id": "api-a"}).json()
    st_b = c.get("/api/v1/sales/data-status", headers={"X-Store-Id": "api-b"}).json()
    assert st_a["latest_sales_date"] == "2026-03-06" and st_b["latest_sales_date"] is None
    batches = c.get("/api/v1/sales/batches", headers={"X-Store-Id": "api-a"}).json()
    assert batches[0]["sync_status"] == "SYNCED"


# ----------------------------------------------------------------------------- job queue
def test_job_queue_dedupes_claims_once_and_records_results(monkeypatch):
    from refillcare.jobs import queue
    from refillcare.jobs.worker import run_job
    from refillcare.tenancy import use_store

    with use_store("jobs-a"):
        j1 = queue.enqueue("expire_stages", {"today": "2026-09-01"}, dedupe_key="exp:1")
        j2 = queue.enqueue("expire_stages", {"today": "2026-09-01"}, dedupe_key="exp:1")
        assert j2["deduplicated"] and j2["job_id"] == j1["job_id"]
        while True:  # other tests may have left jobs queued in the shared control plane
            claimed = queue.claim_next()
            if claimed is None or claimed["job_id"] == j1["job_id"]:
                break
            queue.finish(claimed["job_id"], result={"skipped": True})
        assert claimed["job_id"] == j1["job_id"] and claimed["status"] == "running"
        run_job(claimed)
        done = queue.get_job(j1["job_id"])
        assert done["status"] == "done" and "stages_expired" in done["result"]
        with pytest.raises(ValueError):
            queue.enqueue("rm -rf", {})


def test_scheduler_queues_one_retrain_per_month_not_every_hour(monkeypatch):
    import refillcare.ml.service as svc
    import refillcare.stores as stores_mod
    from refillcare.jobs import queue
    from refillcare.jobs.worker import schedule_store_jobs

    sid = "jobs-retrain-loop"
    monkeypatch.setattr(stores_mod, "list_stores", lambda: [{"store_id": sid, "status": "active"}])
    monkeypatch.setattr(svc, "load_history", lambda: None)
    monkeypatch.setattr(svc, "data_freshness", lambda db, hist: {"retrain_recommended": True})
    day = date(2026, 12, 2)
    key = f"retrain:{day:%Y-%m}"

    schedule_store_jobs(today=day)
    assert queue.count_jobs(key, ("queued",), store_id=sid) == 1
    # The retrain finishes but is not promoted -> freshness still says "retrain" ...
    while True:
        c = queue.claim_next()
        if c is None:
            break
        queue.finish(c["job_id"], result={"promoted": False})
    # ... the next hourly runs must not queue it again this month
    schedule_store_jobs(today=day)
    schedule_store_jobs(today=day)
    assert queue.count_jobs(key, ("queued", "running"), store_id=sid) == 0
    assert queue.count_jobs(key, ("done",), store_id=sid) == 1


def test_failed_job_is_recorded():
    from refillcare.jobs import queue
    from refillcare.jobs.worker import run_job
    from refillcare.tenancy import use_store

    with use_store("jobs-b"):
        j = queue.enqueue("rules_ab", {"version_id": "does-not-exist"})
        while True:
            c = queue.claim_next()
            if c is None or c["job_id"] == j["job_id"]:
                break
            queue.finish(c["job_id"], result={"skipped": True})
        run_job(c)
        assert queue.get_job(j["job_id"])["status"] == "failed"


# ----------------------------------------------------------------------------- model artifacts
def test_artifact_checksum_is_verified_and_old_models_pruned(store_db, tmp_path, monkeypatch):
    from database.models import MLModelVersionModel
    from refillcare.ml import artifact_store as art
    from refillcare.ml import registry as reg

    monkeypatch.setattr(reg, "MODELS_DIR", tmp_path / "models")
    uri, digest = art.save_bundle({"model": None, "feature_cols": ["a"]}, "m1.joblib")
    assert art.local_path_for(uri, digest).is_file()
    (art.models_dir() / "m1.joblib").write_bytes(b"tampered")
    with pytest.raises(art.ArtifactIntegrityError):
        art.local_path_for(uri, digest)

    for k in range(5):
        u, d = art.save_bundle({"model": None, "feature_cols": ["a"]}, f"v{k}.joblib")
        store_db.add(MLModelVersionModel(version_id=f"hgb-v{k}", model_family=reg.FAMILY_PATH_A_INTERVAL, model_type=reg.TYPE_HIST_GB,
                                         status=reg.STATUS_PRODUCTION if k == 0 else reg.STATUS_CANDIDATE, artifact_uri=u,
                                         artifact_sha256=d, created_at=datetime(2026, 1, 1 + k)))
    store_db.commit()
    assert art.prune_artifacts(store_db, keep=2) == 2
    kept = {v.version_id for v in store_db.query(MLModelVersionModel).filter(MLModelVersionModel.artifact_uri.isnot(None))}
    assert kept == {"hgb-v0", "hgb-v4", "hgb-v3"}


def test_model_rollback_restores_previous_validated_version(store_db):
    from refillcare.ml import registry as reg

    reg.ensure_bootstrap_versions(store_db)
    reg.register_version(store_db, version_id="hgb-old", model_family=reg.FAMILY_PATH_A_INTERVAL, model_type=reg.TYPE_HIST_GB,
                         artifact_uri="hgb-old.joblib")
    reg.register_version(store_db, version_id="hgb-new", model_family=reg.FAMILY_PATH_A_INTERVAL, model_type=reg.TYPE_HIST_GB,
                         artifact_uri="hgb-new.joblib")
    reg.promote_version(store_db, "hgb-old", "t", {"approved": True})
    reg.promote_version(store_db, "hgb-new", "t", {"approved": True})
    assert reg.rollback_production(store_db, "tester").version_id == "hgb-old"
    assert reg.get_by_status(store_db, reg.FAMILY_PATH_A_INTERVAL, reg.STATUS_PRODUCTION).version_id == "hgb-old"


def test_model_rollback_skips_version_whose_artifact_was_deleted(store_db):
    from refillcare.ml import registry as reg

    reg.ensure_bootstrap_versions(store_db)
    reg.register_version(store_db, version_id="hgb-old", model_family=reg.FAMILY_PATH_A_INTERVAL, model_type=reg.TYPE_HIST_GB,
                         artifact_uri="hgb-old.joblib")
    reg.register_version(store_db, version_id="hgb-new", model_family=reg.FAMILY_PATH_A_INTERVAL, model_type=reg.TYPE_HIST_GB,
                         artifact_uri="hgb-new.joblib")
    reg.promote_version(store_db, "hgb-old", "t", {"approved": True})
    reg.promote_version(store_db, "hgb-new", "t", {"approved": True})
    store_db.get(reg.MLModelVersionModel, "hgb-old").artifact_uri = None  # file pruned / deleted
    store_db.commit()
    assert reg.rollback_production(store_db, "tester").version_id == reg.BASELINE_VERSION_ID


# ----------------------------------------------------------------------------- evaluation & gate
def test_patient_bootstrap_widens_intervals_for_clustered_events():
    from refillcare.ml.evaluation import _metrics

    rng = np.random.default_rng(0)
    actual = np.full(400, 30.0)
    # 20 patients x 20 events, each patient either always right or always wrong
    pred = np.repeat(np.where(rng.random(20) < 0.6, 30.0, 50.0), 20)
    groups = np.repeat(np.arange(20).astype(str), 20)
    naive = _metrics(pred, actual, np.random.default_rng(1))["within_7_days_ci95"]
    clustered = _metrics(pred, actual, np.random.default_rng(1), groups=groups)["within_7_days_ci95"]
    assert (clustered[1] - clustered[0]) > 2 * (naive[1] - naive[0])


def test_gate_checks_late_reminders_and_coverage():
    from refillcare.ml.evaluation import BASELINE_METHOD
    from refillcare.ml.promotion import evaluate_gate

    m = {"within_7_days_pct": 70, "mae_days": 6, "early_pct": 10, "late_pct": 25}
    rep = {"paired_comparison": {"n_events": 500, "methods": {"cand": m, BASELINE_METHOD: {**m, "within_7_days_pct": 60, "late_pct": 10}, "prod": m},
                                 "diffs": {f"cand vs {BASELINE_METHOD}": {"within_7_days_diff_pp": 10, "within_7_days_diff_ci95": [5, 15], "mae_diff_days": -1}}},
           "coverage": {"cand": {"coverage_pct": 40}, "prod": {"coverage_pct": 60}}, "segments": {}}
    dec = evaluate_gate(rep, "cand", candidate_is_baseline=False, production_method="prod")
    assert "no_extra_late_reminders" in dec["failed_checks"] and "coverage_not_lower" in dec["failed_checks"]


# ----------------------------------------------------------------------------- logging & dates
def test_phones_are_masked_in_logs(caplog):
    import logging

    from refillcare.logging_setup import PHIFilter, mask_phones

    assert mask_phones("send to 9848012345 and +91 97011 12222") == "send to 98******45 and 97******22"
    rec = logging.LogRecord("x", logging.INFO, __file__, 1, "patient %s", ("9848012345",), None)
    PHIFilter().filter(rec)
    assert "9848012345" not in rec.getMessage()


def test_dates_after_the_first_100_rows_are_still_parsed():
    from refillcare.data.dates import parse_pharmacy_dates

    vals = ["01-08-2026"] * 120 + ["2026-08-15", "15 Aug 2026"]
    parsed, meta = parse_pharmacy_dates(pd.Series(vals))
    assert parsed.notna().all() and meta["invalid_count"] == 0
    bad = pd.Series(["01-08-2026"] * 20 + ["garbage"] * 5)
    _, meta2 = parse_pharmacy_dates(bad)
    assert meta2["is_ambiguous"]


def test_upload_keeps_every_bill_line_stores_b2b_and_cleans_name_ids_only(store_db):
    """9 Nov 2026: batches / repeat scans of one medicine on one bill are all kept and added up; SB/ lines are
    stored as not eligible; 'AKHILA .R' -> 'AKHILA R' (same as the original history) but code ids stay as billed."""
    from database.models import RefillDecisionModel, SalesTransactionModel as Line
    from refillcare.data import history_store as hs
    from refillcare.data.incremental_ingest import ingest_sales_batch

    day = date(2026, 10, 31)
    f = _day_file(day, rows=(("AKHILA .R", "9848012345", "1162", "GLIMY MV1 TAB", 1),
                             ("AKHILA .R", "9848012345", "1162", "GLIMY MV1 TAB", 2),
                             ("AKHILA .R", "9848012345", "1162", "GLIMY MV1 TAB", 1)), start_no=36940)
    f["Bill No."] = "S0/36940"  # one bill, three lines (two identical)
    f = pd.concat([f, _day_file(day, rows=(("CUST_001", "9848099999", "4060", "TELMA 40 TAB", 2),), start_no=7),
                   _day_file(day, rows=(("BRANCH 2", "", "4060", "TELMA 40 TAB", 50),), prefix="SB/", start_no=1)], ignore_index=True)
    res = ingest_sales_batch(store_db, f, "day.csv")
    lines = store_db.query(Line).all()
    glimy = [r for r in lines if r.item_id == "1162"]
    assert len(glimy) == 3 and {r.customer_id for r in glimy} == {"AKHILA R"}
    assert {r.customer_id for r in lines if r.item_id == "4060" and r.transaction_type == "CUSTOMER_SALE"} == {"CUST_001"}
    b2b = [r for r in lines if r.transaction_type == "B2B_INTER_STORE"]
    assert len(b2b) == 1 and b2b[0].refillcare_eligible is False
    assert hs.channel_counts(store_db)["b2b_inter_store"] == 1
    dec = store_db.query(RefillDecisionModel).filter(RefillDecisionModel.customer_item_key == "AKHILA R_9848012345_1162").one()  # patient = name + mobile
    assert dec.purchase_count == 1 and dec.units_purchased == 60  # 1 + 2 + 1 strips of 15 (1X15) = one purchase of 60 tablets
    assert store_db.query(RefillDecisionModel).filter(RefillDecisionModel.customer_id == "BRANCH 2").count() == 0
    # re-uploading the same file adds nothing
    assert ingest_sales_batch(store_db, f, "day-again.csv", force=True)["metrics"]["new_records_added"] == 0
