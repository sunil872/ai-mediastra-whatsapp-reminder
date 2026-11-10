"""Database-backed job queue (no Redis / Celery to pay for or operate).

``enqueue`` records a job for the current store; a worker (``python -m refillcare.jobs.worker``)
claims queued jobs atomically, so a job runs once even with several API / worker containers, and
jobs survive restarts. ``dedupe_key`` keeps one pending job per key (e.g. one retrain per store).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import update

from database.connection import ControlSession
from database.control_models import JobModel
from refillcare.tenancy import current_store_id

JOB_KINDS = ("retrain", "rules_ab", "rolling", "evaluate", "train", "rekey", "sync_batch", "expire_stages", "refresh_salt_priors")
MAX_ATTEMPTS = 3
STALE_RUNNING_MINUTES = 180  # a job "running" this long was orphaned by a crashed worker


def enqueue(kind: str, params: Optional[Dict[str, Any]] = None, store_id: Optional[str] = None,
            dedupe_key: Optional[str] = None, created_by: str = "system") -> Dict[str, Any]:
    if kind not in JOB_KINDS:
        raise ValueError(f"Unknown job kind '{kind}'")
    sid = store_id or current_store_id()
    with ControlSession() as s:
        if dedupe_key:
            pending = (s.query(JobModel).filter(JobModel.store_id == sid, JobModel.dedupe_key == dedupe_key,
                                                JobModel.status.in_(("queued", "running"))).first())
            if pending is not None:
                return to_dict(pending) | {"deduplicated": True}
        job = JobModel(store_id=sid, kind=kind, params=params or {}, dedupe_key=dedupe_key, created_by=created_by)
        s.add(job)
        s.commit()
        return to_dict(job)


def count_jobs(dedupe_key: str, statuses: tuple, store_id: Optional[str] = None) -> int:
    """How many jobs with this dedupe key are in the given statuses (e.g. ("done",))."""
    sid = store_id or current_store_id()
    with ControlSession() as s:
        return (s.query(JobModel)
                .filter(JobModel.store_id == sid, JobModel.dedupe_key == dedupe_key, JobModel.status.in_(statuses))
                .count())


def claim_next() -> Optional[Dict[str, Any]]:
    """Atomically move the oldest queued job to running and return it (None when idle)."""
    with ControlSession() as s:
        stale = datetime.utcnow() - timedelta(minutes=STALE_RUNNING_MINUTES)
        s.execute(update(JobModel).where(JobModel.status == "running", JobModel.started_at < stale)
                  .values(status="queued"))
        s.commit()
        for _ in range(5):
            job = (s.query(JobModel).filter(JobModel.status == "queued").order_by(JobModel.created_at, JobModel.job_id).first())
            if job is None:
                return None
            res = s.execute(update(JobModel).where(JobModel.job_id == job.job_id, JobModel.status == "queued")
                            .values(status="running", started_at=datetime.utcnow(), attempts=JobModel.attempts + 1))
            s.commit()
            if res.rowcount == 1:  # we won the claim
                s.refresh(job)
                return to_dict(job)
        return None


def finish(job_id: int, result: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> None:
    with ControlSession() as s:
        job = s.get(JobModel, job_id)
        if job is None:
            return
        if error and job.attempts < MAX_ATTEMPTS and job.kind in ("sync_batch", "expire_stages"):
            job.status, job.error = "queued", error[:4000]  # transient: retry
        else:
            job.status = "failed" if error else "done"
            job.error = error[:4000] if error else None
            job.result = result
        job.finished_at = datetime.utcnow()
        s.commit()


def get_job(job_id: int) -> Optional[Dict[str, Any]]:
    with ControlSession() as s:
        job = s.get(JobModel, job_id)
        return to_dict(job) if job else None


def list_jobs(store_id: Optional[str] = None, limit: int = 20, kind: Optional[str] = None) -> List[Dict[str, Any]]:
    with ControlSession() as s:
        q = s.query(JobModel).filter(JobModel.store_id == (store_id or current_store_id()))
        if kind:
            q = q.filter(JobModel.kind == kind)
        return [to_dict(j) for j in q.order_by(JobModel.created_at.desc()).limit(limit).all()]


def latest_result(kind: str, store_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    with ControlSession() as s:
        j = (s.query(JobModel).filter(JobModel.store_id == (store_id or current_store_id()), JobModel.kind == kind,
                                      JobModel.status == "done").order_by(JobModel.finished_at.desc()).first())
        return to_dict(j) if j else None


def to_dict(j: JobModel) -> Dict[str, Any]:
    return {"job_id": j.job_id, "store_id": j.store_id, "kind": j.kind, "params": j.params, "status": j.status,
            "attempts": j.attempts, "error": j.error, "result": j.result, "created_by": j.created_by,
            "created_at": j.created_at, "started_at": j.started_at, "finished_at": j.finished_at}
