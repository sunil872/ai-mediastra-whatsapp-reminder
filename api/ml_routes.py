"""ML lifecycle API: registry, evaluations, promotion gate, outcomes and ML health."""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
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


def _run_evaluation_background(run_id: str, req: EvaluationRequest, user: str) -> None:
    from refillcare.ml.service import run_evaluation

    db = SessionLocal()
    try:
        run_evaluation(db, version_ids=req.version_ids, candidate_version_id=req.candidate_version_id,
                       eval_start=req.eval_start, eval_end=req.eval_end, max_events=req.max_events,
                       seed=req.seed, created_by=user, run_id=run_id)
    finally:
        db.close()


def _run_training_background(req: TrainRequest, user: str) -> None:
    from refillcare.ml.service import load_history, plan_time_splits
    from refillcare.ml.training import train_interval_model

    db = SessionLocal()
    try:
        hist = load_history()
        train_end = req.train_end or plan_time_splits(hist)["train"]["end"]
        train_interval_model(db, hist, train_end=train_end, train_start=req.train_start,
                             max_events=req.max_events, created_by=user)
    finally:
        db.close()


def _run_retrain_background(req: RetrainRequest, user: str) -> None:
    from refillcare.ml.service import retrain_cycle

    db = SessionLocal()
    try:
        retrain_cycle(db, promote_best=req.promote_best, created_by=user, max_eval_events=req.max_eval_events)
    except Exception:
        logger.exception("Retrain cycle failed")
    finally:
        db.close()


@router.get("/plan")
def current_split_plan(db: Session = Depends(get_db)):
    """Train / validation / test windows derived from the latest sale (they move with every new month)."""
    from refillcare.ml.service import data_freshness

    return data_freshness(db)


@router.post("/retrain", status_code=status.HTTP_202_ACCEPTED)
def start_retrain(req: RetrainRequest, background: BackgroundTasks, actor: str = Depends(require_api_key)):
    background.add_task(_run_retrain_background, req, actor)
    return {"status": "QUEUED", "message": "Retrain cycle started; results appear in /api/v1/ml/evaluations (VAL-*/TEST-* runs)."}


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
def start_evaluation(req: EvaluationRequest, background: BackgroundTasks, actor: str = Depends(require_api_key)):
    run_id = f"EVAL-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    background.add_task(_run_evaluation_background, run_id, req, actor)
    return {"run_id": run_id, "status": "QUEUED", "poll": f"/api/v1/ml/evaluations/{run_id}"}


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
def start_training(req: TrainRequest, background: BackgroundTasks, actor: str = Depends(require_api_key)):
    background.add_task(_run_training_background, req, actor)
    return {"status": "QUEUED", "message": "Training started; the new version appears in /api/v1/ml/models as a candidate."}


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
