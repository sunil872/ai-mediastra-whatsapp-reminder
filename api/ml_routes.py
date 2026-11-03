"""ML lifecycle API: registry, evaluations, promotion gate, outcomes and ML health."""
from __future__ import annotations

import logging
import uuid
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.security import require_api_key
from database.connection import SessionLocal, get_db
from database.models import MLEvaluationRunModel
from refillcare.ml import registry as reg

logger = logging.getLogger("refillcare.api.ml")

router = APIRouter(prefix="/api/v1/ml", tags=["ML Lifecycle (Registry, Evaluation, Promotion)"])


class EvaluationRequest(BaseModel):
    version_ids: Optional[List[str]] = Field(None, description="Versions to backtest (default: production, shadow and candidates)")
    candidate_version_id: Optional[str] = Field(None, description="Version under test for the promotion gate")
    eval_start: Optional[date] = Field(None, description="Window start (default: planned test window, see /api/v1/ml/plan)")
    eval_end: Optional[date] = Field(None, description="Window end (default: last sale - censor buffer)")
    max_events: int = Field(3000, ge=200, le=50000)
    seed: int = 42


class PromotionRequest(BaseModel):
    evaluation_run_id: Optional[str] = None
    force: bool = Field(False, description="Override a failed gate (requires force_reason; recorded)")
    force_reason: Optional[str] = None


class RetrainRequest(BaseModel):
    promote_best: bool = Field(False, description="Promote the validation-selected model if the test-window gate approves")
    max_eval_events: int = Field(3000, ge=200, le=50000)


class StatusRequest(BaseModel):
    status: str = Field(..., description="candidate | shadow | retired")


class TrainRequest(BaseModel):
    train_end: Optional[date] = Field(None, description="Training cutoff (default: end of the planned train window)")
    train_start: Optional[date] = None
    max_events: int = Field(150000, ge=1000, le=1000000)


def _queue(kind: str, params: Dict[str, Any], actor: str, dedupe_key: Optional[str] = None) -> Dict[str, Any]:
    """Durable background job for the request's store (survives restarts, runs once)."""
    from refillcare.jobs.queue import enqueue

    job = enqueue(kind, params, dedupe_key=dedupe_key, created_by=actor)
    return {"status": "QUEUED", "job_id": job["job_id"], "deduplicated": bool(job.get("deduplicated")),
            "poll": f"/api/v1/jobs/{job['job_id']}"}


@router.get("/plan")
def current_split_plan(db: Session = Depends(get_db)):
    """Train / validation / test windows derived from the latest sale (they move with every new month)."""
    from refillcare.ml.service import data_freshness

    return data_freshness(db)


@router.post("/retrain", status_code=status.HTTP_202_ACCEPTED)
def start_retrain(req: RetrainRequest, actor: str = Depends(require_api_key)):
    """Queue a retrain cycle (one pending per store); evaluations appear in /api/v1/ml/evaluations."""
    res = _queue("retrain", {"promote_best": req.promote_best, "max_eval_events": req.max_eval_events, "created_by": actor}, actor, dedupe_key="retrain")
    res["message"] = "Retrain cycle queued; results appear in /api/v1/ml/evaluations (VAL-*/TEST-* runs)."
    return res


@router.get("/models")
def list_model_versions(family: Optional[str] = Query(reg.FAMILY_PATH_A_INTERVAL), db: Session = Depends(get_db)):
    reg.ensure_bootstrap_versions(db)
    return [reg.to_dict(v, db) for v in reg.list_versions(db, family)]


@router.post("/models/{version_id}/status")
def set_model_status(version_id: str, req: StatusRequest, actor: str = Depends(require_api_key), db: Session = Depends(get_db)):
    try:
        return reg.to_dict(reg.set_status(db, version_id, req.status), db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/models/{version_id}/gate")
def check_promotion_gate(version_id: str, evaluation_run_id: Optional[str] = None, db: Session = Depends(get_db)):
    from refillcare.ml.service import gate_for_version

    try:
        return gate_for_version(db, version_id, evaluation_run_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/models/{version_id}/promote")
def promote_model_version(version_id: str, req: PromotionRequest, actor: str = Depends(require_api_key), db: Session = Depends(get_db)):
    from refillcare.ml.service import promote

    try:
        res = promote(db, version_id, promoted_by=actor, evaluation_run_id=req.evaluation_run_id, force=req.force, force_reason=req.force_reason)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not res.get("promoted"):
        raise HTTPException(status_code=409, detail={"message": "Promotion gate rejected this version.", "decision": res["decision"]})
    return res


@router.post("/evaluations", status_code=status.HTTP_202_ACCEPTED)
def start_evaluation(req: EvaluationRequest, actor: str = Depends(require_api_key)):
    run_id = f"EVAL-{datetime.utcnow().strftime('%Y%m%d%H%M%S') + '-' + uuid.uuid4().hex[:4]}"
    params = req.model_dump(mode="json") | {"run_id": run_id, "created_by": actor}
    res = _queue("evaluate", params, actor)
    return res | {"run_id": run_id, "poll_run": f"/api/v1/ml/evaluations/{run_id}"}


@router.post("/models/rollback")
def rollback_production_model(actor: str = Depends(require_api_key), db: Session = Depends(get_db)):
    """Put the previous validated production model back (the baseline when there is none)."""
    try:
        v = reg.rollback_production(db, rolled_back_by=actor)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"production_version": v.version_id, "rolled_back": True}


@router.get("/evaluations")
def list_evaluations(limit: int = Query(20, ge=1, le=200), db: Session = Depends(get_db)):
    runs = db.query(MLEvaluationRunModel).order_by(MLEvaluationRunModel.started_at.desc()).limit(limit).all()
    return [
        {
            "run_id": r.run_id,
            "status": r.status,
            "candidate_model_version": r.model_version,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "completed_at": r.completed_at.isoformat() if r.completed_at else None,
            "error": r.error,
            "summary": reg.summarize_report_for_version(r.report, r.model_version or "") if r.report else None,
        }
        for r in runs
    ]


@router.get("/evaluations/{run_id}")
def get_evaluation(run_id: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    r = db.get(MLEvaluationRunModel, run_id)
    if r is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found (it may still be queued).")
    return {"run_id": r.run_id, "status": r.status, "error": r.error, "params": r.params, "report": r.report}


@router.post("/train", status_code=status.HTTP_202_ACCEPTED)
def start_training(req: TrainRequest, actor: str = Depends(require_api_key)):
    res = _queue("train", req.model_dump(mode="json") | {"created_by": actor}, actor)
    res["message"] = "Training queued; the new version appears in /api/v1/ml/models as a candidate."
    return res


@router.post("/outcomes/refresh")
def refresh_live_outcomes(actor: str = Depends(require_api_key), db: Session = Depends(get_db)):
    from refillcare.ml.outcomes import refresh_outcomes
    from refillcare.ml.service import load_history

    return refresh_outcomes(db, load_history())


@router.get("/outcomes/summary")
def live_outcome_summary(window_days: int = Query(90, ge=7, le=730), db: Session = Depends(get_db)):
    from refillcare.ml.outcomes import outcome_summary

    return outcome_summary(db, window_days=window_days)


@router.get("/health")
def ml_health_status(window_days: int = Query(30, ge=1, le=365), db: Session = Depends(get_db)):
    from refillcare.ml.service import ml_health

    return ml_health(db, since_days=window_days)


# ----------------------------------------------------------------------------- settings, rule A/B, rolling test, identity cut-over
class RulesAbRequest(BaseModel):
    version_id: Optional[str] = Field(None, description="Model version (default: production)")
    max_events: int = Field(3000, ge=200, le=50000)


class RollingRequest(BaseModel):
    n_folds: Optional[int] = Field(None, ge=1, le=12, description="Back-to-back test months (default ML_ROLLING_FOLDS or 3)")
    max_eval_events: int = Field(3000, ge=200, le=50000)
    max_train_events: int = Field(150000, ge=1000, le=1000000)


class RekeyRequest(BaseModel):
    confirm: bool = Field(False, description="Must be true: supersedes old-key reminder cycles and re-scores every patient")


@router.get("/settings")
def prediction_settings_view():
    """Prediction rules, patient identity and therapy grouping the engine is running with (from .env)."""
    from refillcare.ml.service import prediction_settings

    return prediction_settings()


@router.post("/rules-ab", status_code=status.HTTP_202_ACCEPTED)
def start_rules_ab(req: RulesAbRequest, actor: str = Depends(require_api_key)):
    """Compare every prediction rule on the same unseen events (read-only; result via /reports/rules_ab)."""
    return _queue("rules_ab", req.model_dump(mode="json"), actor, dedupe_key="rules_ab") | {"result": "/api/v1/ml/reports/rules_ab"}


@router.post("/rolling", status_code=status.HTTP_202_ACCEPTED)
def start_rolling(req: RollingRequest, actor: str = Depends(require_api_key)):
    """Score model + current rules on several back-to-back unseen months (trains one model per month)."""
    return _queue("rolling", req.model_dump(mode="json"), actor, dedupe_key="rolling") | {"result": "/api/v1/ml/reports/rolling"}


@router.get("/reports/{kind}")
def latest_ml_report(kind: str):
    """Latest saved rules A/B (``rules_ab``) or rolling evaluation (``rolling``) result."""
    from refillcare.ml.service import latest_report

    try:
        rep = latest_report(kind)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if rep is None:
        raise HTTPException(status_code=404, detail=f"No {kind} report yet.")
    return rep


@router.get("/identity/rekey")
def rekey_dry_run(db: Session = Depends(get_db)):
    """Dry run of the patient-identity / salt-grouping cut-over: what would change (nothing is written)."""
    from refillcare.engine.rekey import rekey_reminder_cycles
    from refillcare.ml.service import load_history

    return rekey_reminder_cycles(db, load_history(), apply=False)


@router.post("/identity/rekey", status_code=status.HTTP_202_ACCEPTED)
def rekey_apply(req: RekeyRequest, actor: str = Depends(require_api_key)):
    """Queue the cut-over after switching REFILLCARE_PATIENT_IDENTITY / REFILLCARE_ITEM_GROUPING.

    The job backs up the database first (SQLite online backup; on RDS rely on its snapshots) and never
    re-sends a reminder stage already sent under the old key.
    """
    if not req.confirm:
        raise HTTPException(status_code=400, detail="Set confirm=true to apply the re-key.")
    return _queue("rekey", {}, actor, dedupe_key="rekey")

