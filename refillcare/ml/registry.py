"""Model registry (``ml_model_versions``) for the store.

- Lifecycle: candidate -> shadow -> production -> retired.
- Exactly one production version per model_family; promotion is atomic.
- Artifacts are stored relative to the models directory, never as machine-specific absolute paths.
- Metrics are never typed in: they come from evaluation runs (``latest_evaluation_run_id``).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from config.config import MODELS_DIR, PROJECT_ROOT
from database.models import MLEvaluationRunModel, MLModelVersionModel

FAMILY_PATH_A_INTERVAL = "path_a_interval"
TYPE_BASELINE_MEDIAN = "baseline_historical_median"
TYPE_HIST_GB = "hist_gradient_boosting"

STATUS_CANDIDATE = "candidate"
STATUS_SHADOW = "shadow"
STATUS_PRODUCTION = "production"
STATUS_RETIRED = "retired"
VALID_STATUSES = {STATUS_CANDIDATE, STATUS_SHADOW, STATUS_PRODUCTION, STATUS_RETIRED}

BASELINE_VERSION_ID = "baseline-median-v1"
LEGACY_VERSION_ID = "legacy-human-hgb"
LEGACY_HUMAN_MODEL_FILE = "human_ml_refill_model.joblib"


def feature_schema_hash(feature_cols: List[str]) -> str:
    return hashlib.sha256(json.dumps(list(feature_cols)).encode("utf-8")).hexdigest()[:16]


def current_code_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def resolve_artifact_path(artifact_uri: Optional[str], models_dir: Optional[Path] = None) -> Optional[Path]:
    """Resolve a relative artifact URI against the models directory."""
    if not artifact_uri:
        return None
    p = Path(artifact_uri)
    if p.is_absolute():
        return p if p.is_file() else None
    for base in (models_dir or MODELS_DIR, PROJECT_ROOT):
        c = Path(base) / p
        if c.is_file():
            return c
    return None


def summarize_report_for_version(report: Dict[str, Any], version_id: str) -> Optional[Dict[str, Any]]:
    """Compact paired comparison (model vs baselines) from an evaluation report."""
    paired = (report or {}).get("paired_comparison") or {}
    methods = paired.get("methods") or {}
    if not methods:
        return None
    return {
        "run_id": report.get("run_id"),
        "evaluated_window": report.get("evaluation_window"),
        "n_paired_events": paired.get("n_events"),
        "methods": {
            name: {k: m.get(k) for k in ("mae_days", "within_7_days_pct", "within_3_days_pct", "early_pct", "late_pct", "within_7_days_ci95")}
            for name, m in methods.items()
        },
        "candidate_method": report.get("candidate_method"),
        "candidate_model_version": report.get("candidate_model_version"),
    }


def to_dict(v: MLModelVersionModel, db: Optional[Session] = None) -> Dict[str, Any]:
    d = {
        "version_id": v.version_id,
        "model_family": v.model_family,
        "model_type": v.model_type,
        "status": v.status,
        "artifact_uri": v.artifact_uri,
        "feature_schema_hash": v.feature_schema_hash,
        "train_window_start": v.train_window_start.isoformat() if v.train_window_start else None,
        "train_window_end": v.train_window_end.isoformat() if v.train_window_end else None,
        "training_sample_count": v.training_sample_count,
        "dataset_hash": v.dataset_hash,
        "code_commit": v.code_commit,
        "hyperparameters": v.hyperparameters or {},
        "train_metrics": v.train_metrics or {},
        "is_validated": bool(v.is_validated),
        "latest_evaluation_run_id": v.latest_evaluation_run_id,
        "notes": v.notes or "",
        "created_by": v.created_by,
        "created_at": v.created_at.isoformat() if v.created_at else None,
        "promoted_at": v.promoted_at.isoformat() if v.promoted_at else None,
        "promoted_by": v.promoted_by,
        "promotion_decision": v.promotion_decision,
        "evaluation_summary": None,
    }
    if db is not None and v.latest_evaluation_run_id:
        run = db.get(MLEvaluationRunModel, v.latest_evaluation_run_id)
        if run is not None and run.report:
            d["evaluation_summary"] = summarize_report_for_version(run.report, v.version_id)
    return d


def list_versions(db: Session, family: Optional[str] = None) -> List[MLModelVersionModel]:
    q = db.query(MLModelVersionModel)
    if family:
        q = q.filter(MLModelVersionModel.model_family == family)
    return q.order_by(MLModelVersionModel.created_at.desc()).all()


def get_version(db: Session, version_id: str) -> Optional[MLModelVersionModel]:
    return db.get(MLModelVersionModel, version_id)


def get_by_status(db: Session, family: str, status: str) -> Optional[MLModelVersionModel]:
    return (
        db.query(MLModelVersionModel)
        .filter(MLModelVersionModel.model_family == family, MLModelVersionModel.status == status)
        .order_by(MLModelVersionModel.created_at.desc())
        .first()
    )


def register_version(
    db: Session,
    *,
    version_id: str,
    model_family: str,
    model_type: str,
    artifact_uri: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
    status: str = STATUS_CANDIDATE,
    train_window_start: Optional[date] = None,
    train_window_end: Optional[date] = None,
    training_sample_count: Optional[int] = None,
    dataset_hash: Optional[str] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    train_metrics: Optional[Dict[str, Any]] = None,
    notes: Optional[str] = None,
    created_by: str = "system",
) -> MLModelVersionModel:
    if status not in (STATUS_CANDIDATE, STATUS_SHADOW):
        raise ValueError("New versions start as 'candidate' or 'shadow'; use promote_version for production.")
    if db.get(MLModelVersionModel, version_id) is not None:
        raise ValueError(f"Model version '{version_id}' already exists.")
    v = MLModelVersionModel(
        version_id=version_id,
        model_family=model_family,
        model_type=model_type,
        status=status,
        artifact_uri=artifact_uri,
        feature_cols=list(feature_cols) if feature_cols else None,
        feature_schema_hash=feature_schema_hash(feature_cols) if feature_cols else None,
        train_window_start=train_window_start,
        train_window_end=train_window_end,
        training_sample_count=training_sample_count,
        dataset_hash=dataset_hash,
        code_commit=current_code_commit(),
        hyperparameters=hyperparameters,
        train_metrics=train_metrics,
        notes=notes,
        created_by=created_by,
    )
    db.add(v)
    db.commit()
    return v


def set_status(db: Session, version_id: str, status: str) -> MLModelVersionModel:
    """Move a version to candidate / shadow / retired (production goes through promote_version)."""
    if status not in VALID_STATUSES or status == STATUS_PRODUCTION:
        raise ValueError("Use promote_version to set production; valid here: candidate, shadow, retired.")
    v = db.get(MLModelVersionModel, version_id)
    if v is None:
        raise ValueError(f"Model version '{version_id}' not found.")
    if status == STATUS_SHADOW:
        # one shadow per family keeps shadow comparisons unambiguous
        for other in db.query(MLModelVersionModel).filter(
            MLModelVersionModel.model_family == v.model_family,
            MLModelVersionModel.status == STATUS_SHADOW,
            MLModelVersionModel.version_id != v.version_id,
        ):
            other.status = STATUS_CANDIDATE
    v.status = status
    db.commit()
    return v


def promote_version(db: Session, version_id: str, promoted_by: str, decision: Dict[str, Any]) -> MLModelVersionModel:
    """Atomically make ``version_id`` the production model of its family."""
    v = db.get(MLModelVersionModel, version_id)
    if v is None:
        raise ValueError(f"Model version '{version_id}' not found.")
    for c in db.query(MLModelVersionModel).filter(
        MLModelVersionModel.model_family == v.model_family,
        MLModelVersionModel.status == STATUS_PRODUCTION,
        MLModelVersionModel.version_id != v.version_id,
    ).all():
        c.status = STATUS_RETIRED
    db.flush()
    v.status = STATUS_PRODUCTION
    v.promoted_at = datetime.utcnow()
    v.promoted_by = promoted_by
    v.promotion_decision = decision
    v.is_validated = bool(decision.get("approved")) and not decision.get("forced", False)
    db.commit()
    return v


def rollback_production(db: Session, rolled_back_by: str, family: str = FAMILY_PATH_A_INTERVAL) -> MLModelVersionModel:
    """Put the previous validated production model back (one step; the current one is retired)."""
    current = get_by_status(db, family, STATUS_PRODUCTION)
    q = (db.query(MLModelVersionModel)
         .filter(MLModelVersionModel.model_family == family, MLModelVersionModel.status == STATUS_RETIRED,
                 MLModelVersionModel.promoted_at.isnot(None), MLModelVersionModel.is_validated.is_(True)))
    if current is not None:
        q = q.filter(MLModelVersionModel.version_id != current.version_id)
    previous = q.order_by(MLModelVersionModel.promoted_at.desc()).first()
    if previous is None:
        previous = db.get(MLModelVersionModel, BASELINE_VERSION_ID)
        if previous is None:
            raise ValueError("No previous validated model to roll back to.")
    decision = {"approved": True, "rollback": True, "from_version": current.version_id if current else None,
                "by": rolled_back_by, "at": datetime.utcnow().isoformat()}
    return promote_version(db, previous.version_id, promoted_by=rolled_back_by, decision=decision | {"forced": False})


def ensure_bootstrap_versions(db: Session) -> None:
    """Make the registry truthful on first use (idempotent).

    - ``baseline-median-v1``: the personal historical-median baseline (no artifact).
    - ``legacy-human-hgb``: the existing human_ml_refill_model.joblib with its *real* bundle
      metadata - registered as production (it is what currently serves predictions) but
      ``is_validated=False`` until an evaluation + promotion gate confirms it.
    """
    if db.get(MLModelVersionModel, BASELINE_VERSION_ID) is None:
        db.add(MLModelVersionModel(
            version_id=BASELINE_VERSION_ID,
            model_family=FAMILY_PATH_A_INTERVAL,
            model_type=TYPE_BASELINE_MEDIAN,
            status=STATUS_CANDIDATE,
            artifact_uri=None,
            notes="Personal historical median interval (no ML). Reference baseline for every promotion gate.",
            created_by="bootstrap",
            code_commit=current_code_commit(),
        ))
        db.commit()

    if db.get(MLModelVersionModel, LEGACY_VERSION_ID) is None:
        meta: Dict[str, Any] = {}
        feature_cols: Optional[List[str]] = None
        legacy_path = MODELS_DIR / LEGACY_HUMAN_MODEL_FILE
        if legacy_path.is_file():
            try:
                import joblib

                bundle = joblib.load(legacy_path)
                meta = dict(bundle.get("metadata") or {})
                feature_cols = list(bundle.get("feature_cols") or []) or None
            except Exception:
                meta = {}
        has_prod = get_by_status(db, FAMILY_PATH_A_INTERVAL, STATUS_PRODUCTION) is not None
        cutoff = meta.get("train_cutoff")
        db.add(MLModelVersionModel(
            version_id=LEGACY_VERSION_ID,
            model_family=FAMILY_PATH_A_INTERVAL,
            model_type=TYPE_HIST_GB,
            status=STATUS_CANDIDATE if (has_prod or not legacy_path.is_file()) else STATUS_PRODUCTION,
            artifact_uri=LEGACY_HUMAN_MODEL_FILE if legacy_path.is_file() else None,
            feature_cols=feature_cols,
            feature_schema_hash=feature_schema_hash(feature_cols) if feature_cols else None,
            train_window_end=date.fromisoformat(cutoff) if cutoff else None,
            training_sample_count=meta.get("train_samples"),
            hyperparameters={"loss": meta.get("loss")} if meta.get("loss") else None,
            train_metrics={"train_mae_days": meta.get("train_mae")} if meta.get("train_mae") is not None else None,
            is_validated=False,
            notes=(
                "Pre-registry model (scripts/train_human_ml_model.py). Grandfathered as production; NOT validated. "
                "Known issue: training features were approximated differently from serving features (train/serve skew)."
            ),
            created_by="bootstrap",
            code_commit=current_code_commit(),
        ))
        db.commit()
