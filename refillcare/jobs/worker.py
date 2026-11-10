"""Background worker + scheduler: ``python -m refillcare.jobs.worker``.

Runs queued jobs (one at a time, each inside its store's context) and, once an hour, schedules
for every active store:
- ``expire_stages`` daily (unsent reminder stages past their send date never go out late),
- ``retrain`` when the data has moved a month past the serving model (promotes only if the gate approves
  and ``ML_AUTO_PROMOTE=true``),
- ``refresh_salt_priors`` monthly.
"""
from __future__ import annotations

import logging
import os
import signal
import time
from datetime import date, datetime
from typing import Any, Callable, Dict

from refillcare.jobs import queue

logger = logging.getLogger("refillcare.worker")

MAX_SCHEDULED_RETRAIN_FAILURES = 3  # scheduled retrain retries per store per month
_stop = False


def _flag(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() in ("1", "true", "yes", "on")


# ----------------------------------------------------------------------------- job handlers
def _retrain(db, p: Dict[str, Any]) -> Dict[str, Any]:
    from refillcare.ml.service import retrain_cycle

    res = retrain_cycle(db, promote_best=bool(p.get("promote_best")), created_by=p.get("created_by", "worker"),
                        max_eval_events=p.get("max_eval_events"))
    return {k: res[k] for k in ("candidate_version", "selected_on_validation", "promoted", "gate", "test") if k in res}


def _rules_ab(db, p):
    from refillcare.ml.service import compare_rules

    return compare_rules(db, version_id=p.get("version_id"), max_events=int(p.get("max_events", 3000)))


def _rolling(db, p):
    from refillcare.ml.service import rolling_evaluation

    return rolling_evaluation(db, n_folds=p.get("n_folds"), created_by="worker",
                              max_eval_events=p.get("max_eval_events"), max_train_events=int(p.get("max_train_events", 150000)))


def _evaluate(db, p):
    from refillcare.ml.service import run_evaluation

    run = run_evaluation(db, version_ids=p.get("version_ids"), candidate_version_id=p.get("candidate_version_id"),
                         eval_start=_d(p.get("eval_start")), eval_end=_d(p.get("eval_end")), max_events=int(p.get("max_events", 3000)),
                         seed=int(p.get("seed", 42)), created_by=p.get("created_by", "worker"), run_id=p.get("run_id"))
    return {"run_id": run.run_id, "status": run.status, "error": run.error}


def _train(db, p):
    from refillcare.ml.service import load_history, plan_time_splits
    from refillcare.ml.training import train_interval_model

    hist = load_history()
    end = _d(p.get("train_end")) or plan_time_splits(hist)["train"]["end"]
    res = train_interval_model(db, hist, train_end=end, train_start=_d(p.get("train_start")),
                               max_events=int(p.get("max_events", 150000)), created_by=p.get("created_by", "worker"))
    return {"version": res["version"]["version_id"]}


def _rekey(db, p):
    from refillcare.engine.rekey import backup_sqlite_db, rekey_reminder_cycles
    from refillcare.ml.service import load_history

    backup = backup_sqlite_db()
    res = rekey_reminder_cycles(db, load_history(), apply=True)
    res["database_backup"] = backup
    return res


def _sync_batch(db, p):
    from refillcare.data.incremental_ingest import sync_batch

    res = sync_batch(db, p["batch_id"])
    if res.get("sync_status") == "SYNC_FAILED":
        raise RuntimeError(res.get("error", "sync failed"))
    return res


def _expire(db, p):
    from refillcare.engine.persistence import RefillPersistenceManager

    return {"stages_expired": RefillPersistenceManager().expire_past_due_stages(db, today=_d(p.get("today")) or date.today())}


def _salt_priors(db, p):
    from refillcare.data.incremental_ingest import get_population_stats

    return {"salts": len(get_population_stats(db, force_refresh=True).by_salt)}


HANDLERS: Dict[str, Callable] = {
    "retrain": _retrain, "rules_ab": _rules_ab, "rolling": _rolling, "evaluate": _evaluate, "train": _train,
    "rekey": _rekey, "sync_batch": _sync_batch, "expire_stages": _expire, "refresh_salt_priors": _salt_priors,
}


def _d(v):
    if v in (None, ""):
        return None
    return v if isinstance(v, date) else datetime.fromisoformat(str(v)).date()


def run_job(job: Dict[str, Any]) -> None:
    from database.connection import SessionLocal
    from refillcare.tenancy import use_store

    handler = HANDLERS.get(job["kind"])
    try:
        if handler is None:
            raise ValueError(f"No handler for job kind {job['kind']}")
        with use_store(job["store_id"]):
            db = SessionLocal()
            try:
                result = handler(db, job.get("params") or {})
            finally:
                db.close()
        queue.finish(job["job_id"], result=_jsonable(result))
        logger.info("Job %s (%s, %s) done", job["job_id"], job["kind"], job["store_id"])
    except Exception as e:  # recorded on the job; the worker keeps running
        logger.exception("Job %s (%s, %s) failed", job["job_id"], job["kind"], job["store_id"])
        queue.finish(job["job_id"], error=repr(e))


def _jsonable(obj):
    import json

    return json.loads(json.dumps(obj, default=str)) if obj is not None else None


# ----------------------------------------------------------------------------- scheduler
def schedule_store_jobs(today: date | None = None) -> int:
    """Queue the periodic jobs of every active store (idempotent per day / month)."""
    from database.connection import SessionLocal
    from refillcare.ml.service import data_freshness, load_history
    from refillcare.stores import list_stores
    from refillcare.tenancy import use_store

    today = today or date.today()
    queued = 0
    for st in list_stores():
        if st.get("status") != "active":
            continue
        sid = st["store_id"]
        queue.enqueue("expire_stages", {"today": today.isoformat()}, store_id=sid, dedupe_key=f"expire:{today.isoformat()}")
        queued += 1
        if today.day == 1:
            queue.enqueue("refresh_salt_priors", {}, store_id=sid, dedupe_key=f"salt:{today:%Y-%m}")
        try:
            with use_store(sid):
                db = SessionLocal()
                try:
                    fresh = data_freshness(db, load_history())
                finally:
                    db.close()
            retrain_key = f"retrain:{today:%Y-%m}"
            # One scheduled retrain per store per month: a finished retrain that was not promoted
            # (gate rejected, or ML_AUTO_PROMOTE off) still reports "retrain recommended", so without
            # this check the hourly scheduler re-queued the same retrain every hour. Failures retry up to 3x.
            already_tried = (queue.count_jobs(retrain_key, ("done",), store_id=sid) > 0
                             or queue.count_jobs(retrain_key, ("failed",), store_id=sid) >= MAX_SCHEDULED_RETRAIN_FAILURES)
            if fresh.get("retrain_recommended") and not already_tried:
                queue.enqueue("retrain", {"promote_best": _flag("ML_AUTO_PROMOTE"), "created_by": "scheduler"},
                              store_id=sid, dedupe_key=retrain_key)
                queued += 1
        except Exception:
            logger.exception("Freshness check failed for store %s", sid)
    return queued


def main() -> int:  # pragma: no cover - long-running process
    from refillcare.logging_setup import configure_logging

    configure_logging()
    poll = float(os.getenv("REFILLCARE_WORKER_POLL_SECONDS", "5"))
    schedule_every = float(os.getenv("REFILLCARE_SCHEDULE_EVERY_SECONDS", "3600"))
    last_schedule = 0.0

    def _sig(*_):
        global _stop
        _stop = True

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    logger.info("RefillCare worker started (poll %ss)", poll)
    while not _stop:
        if not _flag("REFILLCARE_DISABLE_SCHEDULER") and time.monotonic() - last_schedule > schedule_every:
            try:
                schedule_store_jobs()
            except Exception:
                logger.exception("Scheduling failed")
            last_schedule = time.monotonic()
        job = queue.claim_next()
        if job is None:
            time.sleep(poll)
            continue
        run_job(job)
    logger.info("RefillCare worker stopped")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
