"""FastAPI Enterprise Application for RefillCare & Mediastra.

Production-grade RESTful API with automated OpenAPI docs, CORS middleware,
database session dependency injection, and static frontend mounting.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from pathlib import Path
from datetime import date, datetime
from typing import Optional, List, Dict, Any

from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    UploadFile,
    File,
    Form,
    Query,
    status,
    Response,
    BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi import Request
from sqlalchemy.orm import Session
from sqlalchemy import text

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.connection import get_db, init_db
from api.schemas import (
    PharmacistAddRequest,
    PharmacistBulkAddRequest,
    WhatsAppSendTodayRequest,
    PharmacistRemoveRequest,
    ColumnMappingSaveRequest,
    StoreRegisterRequest,
    SalesPreviewResponse,
    SalesIngestionResponse,
    ImportBatchResponse,
    RollbackResponse,
    GeneratePredictionsRequest,
    PredictionGenerationSummary,
    PredictionSnapshotItem,
    OutcomeEvaluationSummary,
    ReminderItem,
    ModelVersionItem,
    WhatsAppBatchDispatchRequest,
    WhatsAppDispatchResponse,
    OperationsKPISummary,
    RefillDecisionItem,
    ReminderQueueItem,
    ReviewActionRequest,
    ReviewActionResponse,
    CustomerRefillHistoryResponse,
    TransactionChannelSummary,
    ConsensusRegimenAnalyticsSummary,
    MedicationRuleListResponse,
    PharmacyOverviewResponse,
    AddMedicationRuleRequest,
    ClassifyMedicationRequest,
    ClassifyMedicationResponse,
    ClinicalReviewItem,
    ClinicalReviewSummaryResponse,
    WhatsAppCandidateItem,
    WhatsAppCandidatesResponse,
    WhatsAppPreviewRequest,
    WhatsAppPreviewResponse,
    WhatsAppSingleSendRequest,
    WhatsAppSingleSendResponse,
    CustomerOptInOutResponse,
    WhatsAppWebhookResponse,
)
from api.services import EnterpriseServices
from api.ml_routes import router as ml_router
from refillcare.engine.persistence import RefillPersistenceManager
from api.security import check_api_key, rate_limiter, require_api_key, verify_webhook
from refillcare.logging_setup import configure_logging, request_id_var
from refillcare.tenancy import InvalidStoreId, current_store_id, is_production, set_current_store
from refillcare.data.incremental_ingest import DuplicateUpload
from refillcare.whatsapp.dispatcher import StaleSalesData
from database.models import RefillDecisionModel, ReminderCycleModel, ReminderStageModel

# The database schema is created / additively migrated lazily on first use
# (database.connection.get_sessionmaker), so importing the app never touches a live database.

# Create FastAPI app
configure_logging()
logger = logging.getLogger("refillcare.api")
_DOCS = not is_production() or os.getenv("REFILLCARE_ENABLE_DOCS", "false").lower() in ("1", "true", "yes", "on")
app = FastAPI(
    title="RefillCare & Mediastra Enterprise API",
    description="Enterprise Medication Refill Prediction, Prescription Tracking, and WhatsApp Gateway API. "
                "Every /api call needs X-API-Key and X-Store-Id.",
    version="3.0.0",
    docs_url="/docs" if _DOCS else None,
    redoc_url="/redoc" if _DOCS else None,
    openapi_url="/openapi.json" if _DOCS else None,
)

# Warm the pharmacist classification rules (Supabase) in the background at start-up, so the first
# reminder / KPI request does not pay the network fetch (~2-4 s). Skipped in tests.
if os.getenv("REFILLCARE_ENV", "").lower() != "test" and os.getenv("REFILLCARE_WARM_RULES", "true").lower() in ("1", "true", "yes", "on"):
    import threading as _threading

    def _warm_rules() -> None:
        try:
            from refillcare.data.medication_classifier import get_dynamic_classification_rules

            get_dynamic_classification_rules()
        except Exception:  # pragma: no cover - warm-up is best effort
            logger.warning("Classification rule warm-up failed; rules load on first use.")

    _threading.Thread(target=_warm_rules, name="refillcare-rules-warmup", daemon=True).start()

# Startup sanity check: verify patient identity setting matches stored database keys
if os.getenv("REFILLCARE_ENV", "").lower() != "test":
    try:
        from refillcare.data.identity import check_identity_setting_warning

        _ident_warn = check_identity_setting_warning()
        if _ident_warn:
            logger.warning("Configuration Warning: %s", _ident_warn)
    except Exception as _e:
        logger.debug("Identity check at startup skipped: %s", _e)

# CORS: explicit allow-list (comma-separated CORS_ALLOW_ORIGINS); never "*" with credentials
_cors_origins = [o.strip() for o in os.getenv(
    "CORS_ALLOW_ORIGINS",
    "http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:8501,http://localhost:8501",
).split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key", "X-Store-Id", "X-Tenant-Id", "X-Actor", "X-Request-Id"],
)

_UPLOAD_PATHS = ("/api/v1/sales/ingest", "/api/v1/sales/preview", "/api/v1/sales/column-mapping/detect")
_WEBHOOK_PATH = "/api/v1/whatsapp/webhook"


@app.middleware("http")
async def refillcare_request_context(request: Request, call_next):
    """Request id, rate limit, API key, and the store of this request (X-Store-Id)."""
    from refillcare.stores import UnknownStore, require_known_store

    rid = (request.headers.get("X-Request-Id") or uuid.uuid4().hex[:16])[:64]
    request_id_var.set(rid)
    path = request.url.path
    if path.startswith("/api/"):
        client = request.client.host if request.client else "-"
        bucket = f"{request.headers.get('X-API-Key', '')[:6]}|{client}"
        if not rate_limiter.allow(bucket, cost=10.0 if path in _UPLOAD_PATHS else 1.0):
            return JSONResponse(status_code=429, content={"detail": "Too many requests; slow down.", "request_id": rid})
        if not path.startswith(_WEBHOOK_PATH):
            try:
                request.state.actor = check_api_key(request)
            except HTTPException as e:
                return JSONResponse(status_code=e.status_code, content={"detail": e.detail, "request_id": rid})
        raw_store = request.headers.get("X-Store-Id") or request.query_params.get("store") or request.query_params.get("store_id")
        if not raw_store and is_production() and not path.startswith("/api/v1/stores"):
            return JSONResponse(status_code=400, content={"detail": "X-Store-Id header is required.", "request_id": rid})
        try:
            set_current_store(require_known_store(raw_store) if raw_store else None, request.headers.get("X-Tenant-Id"))
        except InvalidStoreId as e:
            return JSONResponse(status_code=400, content={"detail": str(e), "request_id": rid})
        except UnknownStore as e:
            return JSONResponse(status_code=404, content={"detail": str(e), "request_id": rid})
    else:
        set_current_store(None)
    response = await call_next(request)
    response.headers["X-Request-Id"] = rid
    return response


def _client_error(e: Exception) -> str:
    """Message safe to return to the caller: validation errors verbatim, anything else generic."""
    if isinstance(e, (ValueError, LookupError, PermissionError)):
        return str(e)
    logger.exception("Request failed")
    return f"Internal error (request id {request_id_var.get()})."

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Invalid user input (bad month, reversed range, ...) is a 400, not a silent default."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    """Never leak internals: log with the request id, return a generic message."""
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal error.", "request_id": request_id_var.get()})


app.include_router(ml_router)


def _pm() -> RefillPersistenceManager:
    # Store name / contact come from .env (MEDICAL_STORE_NAME / MEDICAL_STORE_CONTACT)
    return RefillPersistenceManager()


async def _read_upload(file: UploadFile) -> bytes:
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")
    return contents

FRONTEND_DIR = PROJECT_ROOT / "frontend"


# ------------------------------------------------------------------------------
# 1. SYSTEM HEALTH & METRICS
# ------------------------------------------------------------------------------
@app.get("/livez", tags=["System"], include_in_schema=False)
def liveness():
    """Process is up (load balancer / ECS liveness)."""
    return {"status": "alive"}


@app.get("/readyz", tags=["System"], include_in_schema=False)
def readiness():
    """Ready to serve: control-plane database reachable."""
    from database.connection import ControlSession

    try:
        with ControlSession() as c:
            c.execute(text("SELECT 1"))
    except Exception:
        return JSONResponse(status_code=503, content={"status": "not-ready"})
    return {"status": "ready"}


@app.get("/health", tags=["System"])
def health_check(db: Session = Depends(get_db)):
    """Readiness probe: database ping, classification-rule source and serving model."""
    db_ok = True
    db_error = None
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:  # pragma: no cover
        db_ok, db_error = False, str(e)
    try:
        from refillcare.data.medication_classifier import get_rules_status

        rules = get_rules_status()
    except Exception as e:  # pragma: no cover
        rules = {"error": str(e)}
    try:
        from refillcare.ml.model_provider import ModelProvider

        serving = ModelProvider.from_registry(db).describe()
    except Exception as e:  # pragma: no cover
        serving = {"error": str(e)}
    return {
        "status": "healthy" if db_ok and not (isinstance(serving, dict) and serving.get("degraded")) else "degraded",
        "service": "RefillCare Enterprise API",
        "version": "2.1.0",
        "timestamp": datetime.utcnow().isoformat(),
        "database": "connected" if db_ok else f"error: {db_error}",
        "write_protection": "api-key" if os.getenv("REFILLCARE_API_KEY") else "local-machine-only",
        "classification_rules": rules,
        "serving_model": serving,
    }


@app.get("/api/v1/analytics/kpi", response_model=OperationsKPISummary, tags=["Analytics"])
def get_kpi_summary(db: Session = Depends(get_db)):
    """Retrieve executive operations summary KPIs."""
    srv = EnterpriseServices(db)
    return srv.get_operations_kpi()


@app.get("/api/v1/analytics/transaction-types", response_model=TransactionChannelSummary, tags=["Analytics"])
def get_transaction_types_summary(db: Session = Depends(get_db)):
    """Retrieve dynamic transaction-channel classification breakdown (Customer vs B2B vs Unknown)."""
    srv = EnterpriseServices(db)
    return srv.get_transaction_channel_kpis()


@app.get("/api/v1/analytics/consensus-regimen", response_model=ConsensusRegimenAnalyticsSummary, tags=["Analytics"])
def get_consensus_regimen_analytics(db: Session = Depends(get_db)):
    """Retrieve clinical dosage regimen prevalence (OD/BD/QOD/TID) and human consensus benchmarks."""
    srv = EnterpriseServices(db)
    return srv.get_consensus_regimen_analytics()





# ------------------------------------------------------------------------------
# 2. SALES DATA INGESTION & BATCH APIS
# ------------------------------------------------------------------------------
def _mapping_form(column_mapping: Optional[str]) -> Optional[Dict[str, Optional[str]]]:
    """``column_mapping`` form field: JSON {RefillCare field: store header or null}."""
    if not column_mapping:
        return None
    try:
        m = json.loads(column_mapping)
    except ValueError:
        raise HTTPException(status_code=400, detail="column_mapping must be JSON: {field: header}")
    if not isinstance(m, dict):
        raise HTTPException(status_code=400, detail="column_mapping must be a JSON object")
    return {str(k): (str(v) if v else None) for k, v in m.items()}


@app.post("/api/v1/sales/column-mapping/detect", tags=["Sales Ingestion"])
async def detect_column_mapping(
    file: UploadFile = File(...),
    column_mapping: Optional[str] = Form(None),
    actor: str = Depends(require_api_key),
    db: Session = Depends(get_db),
):
    """How this file's headers map to RefillCare fields: saved store profile, else auto-detected.

    ``needs_confirmation`` is true only when a required field is missing; ``fields`` carries the
    chosen column, confidence, source and sample values for the confirm screen.
    """
    srv = EnterpriseServices(db)
    contents = await _read_upload(file)
    try:
        return srv.detect_column_mapping(contents, file.filename or "uploaded_sales.csv", _mapping_form(column_mapping))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_client_error(e))


@app.post("/api/v1/sales/column-mapping", tags=["Sales Ingestion"])
def save_column_mapping(req: ColumnMappingSaveRequest, actor: str = Depends(require_api_key), db: Session = Depends(get_db)):
    """Save the confirmed mapping for this store and header layout (used automatically next time)."""
    srv = EnterpriseServices(db)
    try:
        return srv.save_column_mapping(req.headers, req.mapping, actor=actor)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_client_error(e))


@app.get("/api/v1/sales/column-mapping/profiles", tags=["Sales Ingestion"])
def list_column_mappings(db: Session = Depends(get_db)):
    """Saved column-mapping profiles of this store."""
    return EnterpriseServices(db).list_column_mappings()


@app.delete("/api/v1/sales/column-mapping/profiles/{profile_id}", tags=["Sales Ingestion"])
def delete_column_mapping(profile_id: int, actor: str = Depends(require_api_key), db: Session = Depends(get_db)):
    """Forget a saved mapping (the next upload with that layout is auto-detected again)."""
    if not EnterpriseServices(db).delete_column_mapping(profile_id):
        raise HTTPException(status_code=404, detail="Mapping profile not found")
    return {"deleted": profile_id}


@app.post("/api/v1/sales/preview", response_model=SalesPreviewResponse, tags=["Sales Ingestion"])
async def preview_sales_file(
    file: UploadFile = File(...),
    column_mapping: Optional[str] = Form(None),
    actor: str = Depends(require_api_key),
    db: Session = Depends(get_db),
):
    """Validate and preview an uploaded sales file with strict DD-MM-YYYY date diagnostics.

    Optional ``column_mapping`` (JSON) overrides the saved / auto-detected column mapping.
    """
    srv = EnterpriseServices(db)
    contents = await _read_upload(file)
    try:
        res = srv.preview_sales_file(contents, file.filename or "uploaded_sales.xlsx", _mapping_form(column_mapping))
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_client_error(e))


@app.post("/api/v1/sales/ingest", response_model=SalesIngestionResponse, tags=["Sales Ingestion"])
async def ingest_sales_file(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    column_mapping: Optional[str] = Form(None),
    force: bool = Form(False),
    actor: str = Depends(require_api_key),
    db: Session = Depends(get_db),
):
    """Ingest validated sales records into persistent storage with an immutable batch ID.

    Append-only: lines already stored are skipped, only the patients in the file are re-scored.
    The same file twice returns 409 unless ``force=true``.

    The response re-plans the ML train/validation/test windows on the new data
    (``ml_data_freshness``). With ``ML_AUTO_RETRAIN_ON_INGEST=true`` a due retrain cycle is queued
    in the background; it only promotes when ``ML_AUTO_PROMOTE=true`` and the gate approves.
    """
    srv = EnterpriseServices(db)
    contents = await _read_upload(file)
    try:
        res = srv.ingest_sales_file(contents, file.filename or "uploaded_sales.xlsx", _mapping_form(column_mapping), actor=actor, force=force)
    except HTTPException:
        raise
    except DuplicateUpload as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_client_error(e))
    fresh = res.get("ml_data_freshness") or {}
    if fresh.get("retrain_recommended") and _env_flag("ML_AUTO_RETRAIN_ON_INGEST"):
        from refillcare.jobs.queue import enqueue

        enqueue("retrain", {"promote_best": _env_flag("ML_AUTO_PROMOTE"), "created_by": actor}, dedupe_key="retrain", created_by=actor)
        res["ml_retrain_queued"] = True
    return res


def _env_flag(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() in ("1", "true", "yes", "on")


@app.get("/api/v1/stores", tags=["Stores"])
def list_registered_stores():
    """Stores served by this deployment."""
    from refillcare.stores import list_stores

    return list_stores()


@app.post("/api/v1/stores", tags=["Stores"])
def register_store_endpoint(req: StoreRegisterRequest, actor: str = Depends(require_api_key)):
    """Register / update a store and create its database schema (call once per new Mediastra store)."""
    from refillcare.stores import register_store

    try:
        return register_store(req.store_id, tenant_id=req.tenant_id, store_name=req.store_name,
                              store_contact=req.store_contact, settings=req.settings)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/stores/current", tags=["Stores"])
def current_store():
    """The store of this request (X-Store-Id) and its settings."""
    from refillcare.stores import get_store

    return get_store(current_store_id()) or {"store_id": current_store_id()}


@app.get("/api/v1/sales/data-status", tags=["Sales Ingestion"])
def sales_data_status(db: Session = Depends(get_db)):
    """How current the store's sales are; live sends are blocked when stale or a batch failed to sync."""
    from refillcare.data.incremental_ingest import data_status

    return data_status(db)


@app.post("/api/v1/sales/batches/{batch_id}/retry-sync", tags=["Sales Ingestion"])
def retry_batch_sync(batch_id: str, actor: str = Depends(require_api_key), db: Session = Depends(get_db)):
    """Re-run the post-upload re-scoring of a batch whose sync failed."""
    from refillcare.data.incremental_ingest import sync_batch

    try:
        return sync_batch(db, batch_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/v1/jobs", tags=["Jobs"])
def list_store_jobs(limit: int = Query(20, ge=1, le=200), kind: Optional[str] = None):
    """Background jobs of this store (retrain, A/B, rolling test, re-key, sync retries, expiry)."""
    from refillcare.jobs.queue import list_jobs

    return list_jobs(limit=limit, kind=kind)


@app.get("/api/v1/jobs/{job_id}", tags=["Jobs"])
def get_store_job(job_id: int):
    from refillcare.jobs.queue import get_job

    job = get_job(job_id)
    if job is None or job["store_id"] != current_store_id():
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/v1/sales/batches", response_model=List[ImportBatchResponse], tags=["Sales Ingestion"])
def list_batches(db: Session = Depends(get_db)):
    """Retrieve all historical import batches."""
    srv = EnterpriseServices(db)
    return srv.list_import_batches()


@app.post("/api/v1/sales/batches/{batch_id}/rollback", response_model=RollbackResponse, tags=["Sales Ingestion"])
def rollback_import_batch(batch_id: str, actor: str = Depends(require_api_key),
    db: Session = Depends(get_db)):
    """Safely undo an import batch, remove its records, and restore previous latest date."""
    srv = EnterpriseServices(db)
    try:
        res = srv.rollback_batch(batch_id)
        return res
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_client_error(e))


# ------------------------------------------------------------------------------
# 3. PREDICTION & OUTCOME APIS
# ------------------------------------------------------------------------------
@app.post("/api/v1/predictions/generate", response_model=PredictionGenerationSummary, tags=["Predictions"])
def generate_predictions(
    req: Optional[GeneratePredictionsRequest] = None,
    actor: str = Depends(require_api_key),
    db: Session = Depends(get_db),
):
    """Generate updated refill predictions and schedule reminder cycles."""
    srv = EnterpriseServices(db)
    try:
        pred_dt = req.prediction_date if req else None
        res = srv.generate_predictions(prediction_date=pred_dt)
        return res
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_client_error(e))


@app.get("/api/v1/predictions/snapshots", response_model=List[PredictionSnapshotItem], tags=["Predictions"])
def get_prediction_snapshots(
    tier: Optional[str] = Query(None, description="Filter by pilot tier"),
    mobile_status: Optional[str] = Query(None, description="Valid, Missing, Invalid"),
    limit: int = Query(200, ge=1, le=2000),
    db: Session = Depends(get_db),
):
    """Query active refill prediction snapshots."""
    srv = EnterpriseServices(db)
    return srv.get_prediction_snapshots(tier=tier, mobile_status=mobile_status, limit=limit)


@app.post("/api/v1/predictions/evaluate", response_model=OutcomeEvaluationSummary, tags=["Predictions"])
def evaluate_outcomes(actor: str = Depends(require_api_key),
    db: Session = Depends(get_db)):
    """Evaluate accuracy of pending prediction snapshots against actual realized purchases."""
    srv = EnterpriseServices(db)
    try:
        res = srv.evaluate_outcomes()
        return res
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_client_error(e))


# ------------------------------------------------------------------------------
# 4. REMINDER DELIVERY & EXPORT APIS
# ------------------------------------------------------------------------------
@app.get("/api/v1/reminders/daily", tags=["Reminders"])
def get_daily_reminders(
    target_date: Optional[date] = Query(None, description="Target reminder date (defaults to today if month not set)"),
    target_month: Optional[str] = Query(None, description="Target reminder month in YYYY-MM format (e.g. 2026-09)"),
    start_date: Optional[date] = Query(None, description="Start date for custom date range or rolling horizon"),
    end_date: Optional[date] = Query(None, description="End date for custom date range or rolling horizon"),
    mobile_status: str = Query("All", description="All, Valid, Missing"),
    stage: Optional[str] = Query(None, description="Filter by reminder stage"),
    refill_function: Optional[str] = Query(None, description="Filter by refill function / pathway"),
    path: Optional[str] = Query(None, description="Filter by decision path"),
    customer_query: Optional[str] = Query(None, description="Search by customer name or phone"),
    medication_query: Optional[str] = Query(None, description="Search by medication name"),
    limit: Optional[int] = Query(None, ge=1, le=5000, description="Max records to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
):
    """Get scheduled reminders for a specific date, full target month, or rolling date range with clinical filtering."""
    import json
    srv = EnterpriseServices(db)
    items = srv.get_daily_reminders(
        target_date=target_date,
        target_month=target_month,
        start_date=start_date,
        end_date=end_date,
        mobile_filter=mobile_status,
        stage=stage,
        refill_function=refill_function,
        path=path,
        customer_query=customer_query,
        medication_query=medication_query,
        limit=limit,
        offset=offset,
    )
    return Response(content=json.dumps(items), media_type="application/json")


@app.get("/api/v1/review/summary", response_model=ClinicalReviewSummaryResponse, tags=["Clinical Review Queue"])
def get_clinical_review_summary(
    target_month: Optional[str] = Query(None, description="Target forecast month YYYY-MM (default: active forecast month)"),
    db: Session = Depends(get_db),
):
    """Retrieve aggregate summary counts for the Clinical Review Queue (synced with Streamlit)."""
    srv = EnterpriseServices(db)
    return srv.get_clinical_review_summary(target_month=target_month)


@app.get("/api/v1/review/queue", response_model=List[ClinicalReviewItem], tags=["Clinical Review Queue"])
def get_clinical_review_queue(
    target_month: Optional[str] = Query(None, description="Target forecast month YYYY-MM (default: active forecast month)"),
    category: Optional[str] = Query(None, description="All, Missing Mobile, Stability Risk, Cold-Start"),
    missing_mobile_only: bool = Query(False, description="Filter for missing mobile numbers only"),
    search: Optional[str] = Query(None, description="Search by customer name, phone, or reason keyword"),
    limit: int = Query(200, ge=1, le=5000, description="Max candidate items to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    db: Session = Depends(get_db),
):
    """Retrieve detailed clinical review queue records with reason classification and patient context."""
    srv = EnterpriseServices(db)
    return srv.get_clinical_review_queue(
        target_month=target_month,
        category=category,
        missing_mobile_only=missing_mobile_only,
        search=search,
        limit=limit,
        offset=offset,
    )


@app.get("/api/v1/reminders/export-csv", tags=["Reminders"])
def export_reminder_csv(
    target_date: Optional[date] = Query(None, description="Target reminder date"),
    target_month: Optional[str] = Query(None, description="Target reminder month in YYYY-MM format (e.g. 2026-09)"),
    db: Session = Depends(get_db),
):
    """Download delivery-ready CSV with 10 standard columns for valid mobile numbers."""
    srv = EnterpriseServices(db)
    csv_bytes = srv.export_reminder_csv(target_date=target_date, target_month=target_month)
    label = target_month.replace("-", "") if target_month else (target_date or date.today()).strftime("%Y%m%d")
    filename = f"refill_reminders_{label}.csv"
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/api/v1/reminders/export-json", tags=["Reminders"])
def export_reminder_json(
    target_date: Optional[date] = Query(None, description="Target reminder date"),
    target_month: Optional[str] = Query(None, description="Target reminder month in YYYY-MM format (e.g. 2026-09)"),
    mobile_status: str = Query("All", description="All, Valid, Missing"),
    db: Session = Depends(get_db),
):
    """Download full structured JSON for scheduled reminders on a date or full month."""
    import json
    srv = EnterpriseServices(db)
    data = srv.export_reminder_json(target_date=target_date, target_month=target_month, mobile_filter=mobile_status)
    label = target_month.replace("-", "") if target_month else (target_date or date.today()).strftime("%Y%m%d")
    filename = f"refill_reminders_{label}.json"
    return Response(
        content=json.dumps(data, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/api/v1/reminders/months", response_model=List[str], tags=["Reminders"])
def list_reminder_months(db: Session = Depends(get_db)):
    """List distinct available scheduled reminder months."""
    srv = EnterpriseServices(db)
    return srv.list_available_reminder_months()



# ------------------------------------------------------------------------------
# 5. MODEL REGISTRY & VERSIONING APIS
# ------------------------------------------------------------------------------
@app.get("/api/v1/models", response_model=List[ModelVersionItem], tags=["Model Registry"])
def list_models(db: Session = Depends(get_db)):
    """List registered model versions and performance benchmarks."""
    srv = EnterpriseServices(db)
    return srv.list_models()


@app.post("/api/v1/models/{version_id}/activate", tags=["Model Registry"])
def activate_model(version_id: str, actor: str = Depends(require_api_key),
    db: Session = Depends(get_db)):
    """Set active production model version."""
    srv = EnterpriseServices(db)
    try:
        return srv.activate_model(version_id, promoted_by=actor)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=_client_error(e))


# ------------------------------------------------------------------------------
# 6. WHATSAPP GATEWAY DISPATCH
# ------------------------------------------------------------------------------
@app.get("/api/v1/whatsapp/candidates", response_model=WhatsAppCandidatesResponse, tags=["WhatsApp Gateway"])
def get_whatsapp_candidates(
    target_date: Optional[date] = Query(None, description="Target outreach date (e.g. 2026-10-13)"),
    target_month: Optional[str] = Query(None, description="Target outreach month (e.g. 2026-10)"),
    tier: str = Query("ALL", description="ALL, DUE, FOLLOWUP, LAPSED"),
    sync_window: int = Query(0, ge=0, le=14, description="Sync window in days"),
    mobile_status: str = Query("All", description="All, Valid, Missing"),
    search: Optional[str] = Query(None, description="Search customer name, phone, or medication"),
    db: Session = Depends(get_db),
):
    """Retrieve bundled patient candidate records for WhatsApp outreach matching Streamlit gateway."""
    srv = EnterpriseServices(db)
    return srv.get_whatsapp_outreach_candidates(
        target_date=target_date,
        target_month=target_month,
        tier=tier,
        sync_window=sync_window,
        mobile_status=mobile_status,
        search=search,
    )


@app.post("/api/v1/whatsapp/preview", response_model=WhatsAppPreviewResponse, tags=["WhatsApp Gateway"])
def preview_whatsapp_message(
    req: WhatsAppPreviewRequest,
    db: Session = Depends(get_db),
):
    """Generate high-contrast visual WhatsApp template preview and parameters."""
    srv = EnterpriseServices(db)
    return srv.preview_whatsapp_message(
        customer_name=req.customer_name,
        medications=req.medications,
        expected_refill_date=req.expected_refill_date,
        tier=req.tier,
        stage_offset=req.stage_offset,
    )


@app.post("/api/v1/whatsapp/send-single", response_model=WhatsAppSingleSendResponse, tags=["WhatsApp Gateway"])
def send_single_whatsapp(
    req: WhatsAppSingleSendRequest,
    actor: str = Depends(require_api_key),
    db: Session = Depends(get_db),
):
    """Send or simulate single patient WhatsApp reminder."""
    srv = EnterpriseServices(db)
    return srv.send_single_whatsapp_reminder(
        phone_number=req.phone_number,
        customer_name=req.customer_name,
        medications=req.medications,
        expected_refill_date=req.expected_refill_date,
        tier=req.tier,
        stage_offset=req.stage_offset,
        dry_run=req.dry_run,
    )


@app.post("/api/v1/whatsapp/dispatch", response_model=WhatsAppDispatchResponse, tags=["WhatsApp Gateway"])
def dispatch_whatsapp(
    req: WhatsAppBatchDispatchRequest,
    actor: str = Depends(require_api_key),
    db: Session = Depends(get_db),
):
    """Dispatch WhatsApp reminders via Xinno API (supports dry-run mode)."""
    srv = EnterpriseServices(db)
    try:
        return srv.dispatch_whatsapp_batch(
            reminder_date=req.reminder_date,
            target_month=getattr(req, "target_month", None),
            tier_filter=getattr(req, "tier_filter", None) or "ALL",
            dry_run=req.dry_run,
            batch_size=req.batch_size,
            approve_before_send=req.approve_before_send,
            override_stale_data=req.override_stale_data,
            override_reason=req.override_reason,
            actor=actor,
        )
    except StaleSalesData as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_client_error(e))


@app.get("/api/v1/whatsapp/send-summary", tags=["WhatsApp Gateway"])
def whatsapp_send_summary(
    target_date: Optional[date] = Query(None, description="Day to send (default today; only today can be sent)"),
    db: Session = Depends(get_db),
):
    """Counts for the "Send WhatsApp reminders" Yes/No confirmation: to_send, medicines, no_valid_mobile,
    opted_out, already_sent, holdout (control group), plus can_send / whatsapp_configured / sales_data_stale."""
    return EnterpriseServices(db).get_whatsapp_send_summary(target_date)


@app.post("/api/v1/whatsapp/send-today", tags=["WhatsApp Gateway"])
def whatsapp_send_today(req: WhatsAppSendTodayRequest, actor: str = Depends(require_api_key), db: Session = Depends(get_db)):
    """The pharmacist confirmed "Yes, send": approve and send ALL of today's reminders (live, no dry run).

    Opt-outs, invalid mobiles, the control group and already-sent reminders are skipped. Returns 409 when
    the sales data is stale unless ``override_reason`` is given (logged)."""
    try:
        return EnterpriseServices(db).send_whatsapp_today(override_reason=req.override_reason, actor=actor)
    except StaleSalesData as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/api/v1/whatsapp/webhook", response_model=WhatsAppWebhookResponse, tags=["WhatsApp Gateway"])
async def receive_whatsapp_webhook(
    request: Request,
    db: Session = Depends(get_db),
):
    """Receive incoming WhatsApp replies/callbacks from Xinno or Meta cloud API.
    Automatically honours STOP / UNSUBSCRIBE messages to opt out patients.
    """
    raw = await request.body()
    verify_webhook(request, raw)
    srv = EnterpriseServices(db)
    try:
        body = json.loads(raw.decode("utf-8") or "{}")
    except ValueError:
        body = {}
    return srv.handle_inbound_whatsapp_webhook(body)


# ------------------------------------------------------------------------------
# 7. V1 REFILL DECISION & PHARMACIST REVIEW QUEUE APIS
# ------------------------------------------------------------------------------
@app.get("/api/refill-decisions", response_model=List[RefillDecisionItem], tags=["V1 Refill Decisions"])
def list_refill_decisions(
    path: Optional[str] = Query(None, description="PATH_A, PATH_B, INELIGIBLE"),
    stability: Optional[str] = Query(None, description="HIGH, MEDIUM-SAFE, MEDIUM-RISK, UNSTABLE"),
    eligible_only: bool = Query(False),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """Query persistent RefillDecisions generated by UnifiedRefillDecisionEngine."""
    query = db.query(RefillDecisionModel)
    if path:
        query = query.filter(RefillDecisionModel.path == path)
    if stability:
        query = query.filter(RefillDecisionModel.stability_tier == stability)
    if eligible_only:
        query = query.filter(RefillDecisionModel.is_eligible == True)
    return query.limit(limit).all()


@app.get("/api/refill-decisions/{decision_id}", response_model=RefillDecisionItem, tags=["V1 Refill Decisions"])
def get_refill_decision(decision_id: str, db: Session = Depends(get_db)):
    """Retrieve full details of a specific RefillDecision."""
    dec = db.query(RefillDecisionModel).filter(RefillDecisionModel.decision_id == decision_id).first()
    if not dec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Decision '{decision_id}' not found.")
    return dec


@app.get("/api/v1/reminders/review-candidates", tags=["V1 Pharmacist Review Queue"])
def pharmacist_review_candidates(
    start: Optional[date] = Query(None, description="Estimated refill due from (default: today)"),
    end: Optional[date] = Query(None, description="Estimated refill due until (default: same as start)"),
    all_dates: bool = Query(False, description="Ignore the date window"),
    db: Session = Depends(get_db),
):
    """Patients NOT getting reminders whose medicine is chronic or not yet classified (never known non-chronic)."""
    from refillcare.engine.pharmacist_overrides import review_candidates

    if all_dates:
        return review_candidates(db)
    day_from = start or date.today()
    return review_candidates(db, day_from, end or day_from)


@app.get("/api/v1/reminders/review-other", tags=["V1 Pharmacist Review Queue"])
def pharmacist_review_other(
    category: str = Query(..., description="missing_mobile | unstable | b2b"),
    day: Optional[date] = Query(None, description="Expected refill date (default today; ignored for b2b)"),
    db: Session = Depends(get_db),
):
    """Other Patients Needing Review lists (same rows as the Streamlit Review tab)."""
    from refillcare.engine.review_lists import review_other

    try:
        return review_other(db, category, day or date.today())
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/api/v1/overview", response_model=PharmacyOverviewResponse, tags=["Analytics"])
def pharmacy_overview(db: Session = Depends(get_db)):
    """Pharmacy Overview for this store and this calendar month (same numbers as the Streamlit overview)."""
    from refillcare.engine.overview_kpis import overview_kpis

    return overview_kpis(db)


@app.post("/api/v1/reminders/pharmacist-add", tags=["V1 Pharmacist Review Queue"])
def pharmacist_add_to_reminders(req: PharmacistAddRequest, actor: str = Depends(require_api_key), db: Session = Depends(get_db)):
    """Add a left-out patient-medicine to reminders (kept through every re-scoring)."""
    from refillcare.engine.pharmacist_overrides import add_to_reminders

    try:
        return add_to_reminders(db, req.customer_item_key, interval_days=req.interval_days,
                                next_refill_date=req.next_refill_date, added_by=actor, note=req.note)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/api/v1/reminders/pharmacist-add-bulk", tags=["V1 Pharmacist Review Queue"])
def pharmacist_bulk_add_to_reminders(req: PharmacistBulkAddRequest, actor: str = Depends(require_api_key), db: Session = Depends(get_db)):
    """Add many left-out patient-medicines at once (each keeps its own usual gap and due date)."""
    from refillcare.engine.pharmacist_overrides import add_many_to_reminders

    return add_many_to_reminders(db, req.customer_item_keys, added_by=actor, note=req.note)


@app.post("/api/v1/reminders/pharmacist-remove", tags=["V1 Pharmacist Review Queue"])
def pharmacist_remove_from_reminders(req: PharmacistRemoveRequest, actor: str = Depends(require_api_key), db: Session = Depends(get_db)):
    """Undo a pharmacist addition (cancels its pending reminders)."""
    from refillcare.engine.pharmacist_overrides import remove_from_reminders

    try:
        return remove_from_reminders(db, req.customer_item_key, removed_by=actor)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/api/v1/reminders/pharmacist-added", tags=["V1 Pharmacist Review Queue"])
def pharmacist_added_reminders(db: Session = Depends(get_db)):
    """Patient-medicines currently in reminders because a pharmacist added them."""
    from refillcare.engine.pharmacist_overrides import list_pharmacist_added

    return list_pharmacist_added(db)


@app.get("/api/reminders/today", response_model=List[ReminderQueueItem], tags=["V1 Pharmacist Review Queue"])
def get_today_reminder_queue(
    target_date: Optional[date] = Query(None, description="Queue date (defaults to today)"),
    path: Optional[str] = Query(None),
    stability: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """Retrieve today's pharmacist review queue with decision provenance."""
    eval_date = target_date or EnterpriseServices(db).forecast_as_of()
    pm = _pm()
    return pm.get_today_review_queue(db, target_date=eval_date, path_filter=path, stability_filter=stability)


@app.get("/api/reminders/{reminder_id}", response_model=ReminderQueueItem, tags=["V1 Pharmacist Review Queue"])
def get_reminder_stage_detail(reminder_id: str, db: Session = Depends(get_db)):
    """Get single reminder stage detail."""
    stage = db.query(ReminderStageModel).filter(ReminderStageModel.reminder_id == reminder_id).first()
    if not stage:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Reminder stage '{reminder_id}' not found.")
    pm = _pm()
    queue = pm.get_today_review_queue(db, target_date=stage.target_send_date)
    for item in queue:
        if item["reminder_id"] == reminder_id:
            return item
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Reminder details not available.")


@app.post("/api/reminders/{reminder_id}/approve", response_model=ReviewActionResponse, tags=["V1 Pharmacist Review Queue"])
def approve_reminder(reminder_id: str, actor: str = Depends(require_api_key),
    db: Session = Depends(get_db)):
    """Approve a reminder stage for dispatch."""
    pm = _pm()
    try:
        res = pm.approve_reminder_stage(db, reminder_id)
        return ReviewActionResponse(reminder_id=reminder_id, status=res["status"], success=True, message="Approved successfully")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_client_error(e))


@app.post("/api/reminders/{reminder_id}/reject", response_model=ReviewActionResponse, tags=["V1 Pharmacist Review Queue"])
def reject_reminder(reminder_id: str, req: ReviewActionRequest, actor: str = Depends(require_api_key),
    db: Session = Depends(get_db)):
    """Reject a reminder stage with a pharmacist reason."""
    pm = _pm()
    try:
        res = pm.reject_reminder_stage(db, reminder_id, reason=req.reason or "Pharmacist rejected")
        return ReviewActionResponse(reminder_id=reminder_id, status=res["status"], success=True, message=res["reason"])
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_client_error(e))


@app.post("/api/reminders/{reminder_id}/dispatch", response_model=ReviewActionResponse, tags=["V1 Pharmacist Review Queue"])
def dispatch_single_reminder(reminder_id: str, req: ReviewActionRequest, actor: str = Depends(require_api_key),
    db: Session = Depends(get_db)):
    """Controlled dispatch of a single reminder (DRY-RUN by default)."""
    pm = _pm()
    try:
        res = pm.dispatch_reminder_stage(db, reminder_id, is_dry_run=req.is_dry_run)
        return ReviewActionResponse(
            reminder_id=reminder_id,
            status=res["status"],
            success=res.get("success", False),
            is_dry_run=res.get("is_dry_run", True),
            provider_msg_id=res.get("provider_msg_id"),
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_client_error(e))


@app.get("/api/customers/{customer_id}/refill-history", response_model=CustomerRefillHistoryResponse, tags=["V1 Refill Decisions"])
def get_customer_refill_history(customer_id: str, db: Session = Depends(get_db)):
    """Get complete historical refill cycles and reminder stages for a patient."""
    cycles = db.query(ReminderCycleModel).filter(ReminderCycleModel.customer_id == customer_id).all()
    if not cycles:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No refill cycles for customer '{customer_id}'.")
    c_name = cycles[0].decision.customer_name if (cycles and cycles[0].decision) else None

    cycles_list = []
    for c in cycles:
        stages_list = [
            {
                "reminder_id": st.reminder_id,
                "stage_offset": st.stage_offset,
                "target_send_date": st.target_send_date.strftime("%Y-%m-%d"),
                "status": st.status,
                "sent_at": st.sent_at.isoformat() if st.sent_at else None,
            }
            for st in c.stages
        ]
        cycles_list.append({
            "cycle_id": c.cycle_id,
            "item_id": c.item_id,
            "last_purchase_date": c.last_purchase_date.strftime("%Y-%m-%d") if c.last_purchase_date else "-",
            "expected_refill_date": c.expected_refill_date.strftime("%Y-%m-%d") if c.expected_refill_date else None,
            "is_active": c.is_active,
            "superseded_by_purchase_date": c.superseded_by_purchase_date.strftime("%Y-%m-%d") if c.superseded_by_purchase_date else None,
            "stages": stages_list,
        })

    return CustomerRefillHistoryResponse(
        customer_id=customer_id,
        customer_name=c_name,
        total_cycles=len(cycles_list),
        cycles=cycles_list,
    )


@app.get("/api/customers/opt-outs", tags=["V1 Refill Decisions"])
def list_opted_out_customers(db: Session = Depends(get_db)):
    """Patients who opted out of WhatsApp reminders (via STOP reply or pharmacist action)."""
    from refillcare.whatsapp.consent import list_opted_out

    return list_opted_out(db)


@app.post("/api/customers/{customer_id}/opt-out", response_model=CustomerOptInOutResponse, tags=["V1 Refill Decisions"])
def opt_out_customer(
    customer_id: str,
    actor: str = Depends(require_api_key),
    db: Session = Depends(get_db),
):
    """Manually opt a patient out of automated WhatsApp reminders."""
    srv = EnterpriseServices(db)
    return srv.set_customer_opt_in(customer_id=customer_id, opt_in=False)


@app.post("/api/customers/{customer_id}/opt-in", response_model=CustomerOptInOutResponse, tags=["V1 Refill Decisions"])
def opt_in_customer(
    customer_id: str,
    actor: str = Depends(require_api_key),
    db: Session = Depends(get_db),
):
    """Manually re-enable automated WhatsApp reminders for a patient."""
    srv = EnterpriseServices(db)
    return srv.set_customer_opt_in(customer_id=customer_id, opt_in=True)


@app.get("/api/customers/search", tags=["V1 Refill Decisions"])
def search_customers_for_consent(
    q: str = Query(..., min_length=2, description="Patient name or mobile number (at least 4 digits)"),
    limit: int = Query(25, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Find patients by name or mobile, with whether they receive WhatsApp reminders (pharmacist screen)."""
    from refillcare.whatsapp.pharmacist_views import search_patients

    return search_patients(db, q, limit=limit)


@app.post("/api/customers/whatsapp-messages", tags=["V1 Refill Decisions"])
def set_customer_whatsapp_messages(
    allowed: bool = Query(..., description="false = stop reminders, true = resume"),
    mobile_number: Optional[str] = Query(None, description="Preferred: covers every record sharing this number"),
    customer_id: Optional[str] = Query(None, description="Used when the patient has no mobile number"),
    actor: str = Depends(require_api_key),
    db: Session = Depends(get_db),
):
    """Stop or resume WhatsApp reminders for a patient (same logic as the Streamlit consent section)."""
    from refillcare.whatsapp.pharmacist_views import set_messages_allowed

    try:
        ids = set_messages_allowed(db, allowed=allowed, mobile_number=mobile_number, customer_id=customer_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return {"allowed": allowed, "customer_ids": ids}


@app.get("/api/v1/whatsapp/delivery-log", tags=["WhatsApp Gateway"])
def whatsapp_delivery_log(
    day: Optional[date] = Query(None, description="India-time day (default: today)"),
    status_filter: str = Query("ALL", alias="status", description="ALL, SENT or FAILED"),
    db: Session = Depends(get_db),
):
    """Live messages sent on one day: time, patient, mobile, Sent/Failed and a plain failure reason."""
    from refillcare.whatsapp.pharmacist_views import delivery_log_rows

    return delivery_log_rows(db, day=day, status=status_filter)


# ------------------------------------------------------------------------------
# 8. MED-SYNC & QUANTILE UNCERTAINTY ENDPOINTS
# ------------------------------------------------------------------------------
@app.get("/api/v2/med-sync/bundles", tags=["Med-Sync Multi-Prescription Bundling"])
def get_med_sync_bundles(
    sync_window_days: int = Query(0, ge=0, le=30, description="Max days gap between prescriptions to cluster (0 = Exact Same-Date Group-By)"),
    pharmacy_name: str = Query("Mediastra Pharmacy", description="Pharmacy name for templated messages"),
    customer_id: Optional[str] = Query(None, description="Optional filter by patient ID"),
    target_month: Optional[str] = Query(None, description="Optional prediction target month filter, e.g. '2026-09' or 'ALL'"),
    target_date: Optional[date] = Query(None, description="Optional prediction target date filter, e.g. '2026-09-24'"),
    lifecycle_tier: Optional[str] = Query(None, description="Optional filter by lifecycle tier: DUE, FOLLOWUP, LAPSED, ALL"),
    db: Session = Depends(get_db),
):
    """Retrieve synchronized multi-prescription reminder bundles and message reduction stats for a target date or month."""
    services = EnterpriseServices(db)
    return services.get_med_sync_bundles(
        sync_window_days=sync_window_days,
        pharmacy_name=pharmacy_name,
        customer_id=customer_id,
        target_month=target_month,
        target_date=target_date,
        lifecycle_tier=lifecycle_tier,
    )


@app.get("/api/v2/models/quantiles", tags=["Quantile Uncertainty Envelopes"])
def get_model_quantile_envelope(db: Session = Depends(get_db)):
    """Retrieve 3-head Quantile uncertainty envelope (P10, P50, P90) metrics and coverage report."""
    services = EnterpriseServices(db)
    return services.get_quantile_uncertainty_metrics()


# ------------------------------------------------------------------------------
# 9. PHARMACIST MEDICATION RULES & LIVE CLASSIFIER APIS (SUPABASE)
# ------------------------------------------------------------------------------
@app.get("/api/v1/medications/rules", response_model=MedicationRuleListResponse, tags=["Medication Classifier"])
def list_medication_classification_rules(
    rule_type: Optional[str] = Query(None, description="Filter by CHRONIC or EXCLUSION"),
    search: Optional[str] = Query(None, description="Search pattern keyword, brand, or salt"),
    db: Session = Depends(get_db),
):
    """Retrieve active medication classification rules from Supabase (or local cache)."""
    srv = EnterpriseServices(db)
    return srv.list_medication_rules(rule_type=rule_type, search=search)


@app.post("/api/v1/medications/rules", tags=["Medication Classifier"])
def add_medication_classification_rule(
    req: AddMedicationRuleRequest,
    actor: str = Depends(require_api_key),
    db: Session = Depends(get_db),
):
    """Add or update a medication classification rule in Supabase."""
    srv = EnterpriseServices(db)
    res = srv.add_medication_rule(
        pattern=req.pattern,
        rule_type=req.rule_type,
        category=req.category,
        brand_name=req.brand_name,
        notes=req.notes,
        created_by=req.created_by or "Pharmacist",
    )
    if res.get("status") == "error":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=res.get("message"))
    return res


@app.delete("/api/v1/medications/rules/{rule_id}", tags=["Medication Classifier"])
def delete_medication_classification_rule(
    rule_id: str,
    actor: str = Depends(require_api_key),
    db: Session = Depends(get_db),
):
    """Delete a medication classification rule from Supabase by its ID."""
    srv = EnterpriseServices(db)
    res = srv.delete_medication_rule(rule_id=rule_id)
    if res.get("status") == "error":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=res.get("message"))
    return res


@app.post("/api/v1/medications/classify", response_model=ClassifyMedicationResponse, tags=["Medication Classifier"])
def test_classify_medication(
    req: ClassifyMedicationRequest,
    db: Session = Depends(get_db),
):
    """Test classify a medication item using active Supabase rules and clinical parsing."""
    srv = EnterpriseServices(db)
    return srv.classify_medication_item(item_name=req.item_name, packing=req.packing)


# ------------------------------------------------------------------------------
# 10. FRONTEND STATIC ASSETS & SPA ROUTING
# ------------------------------------------------------------------------------
ASSETS_DIR = PROJECT_ROOT / "assets"
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse, tags=["Web UI"])
    def serve_frontend_index():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
        return HTMLResponse("<h2>RefillCare Enterprise API Online. Frontend loading...</h2>")


@app.get("/favicon.ico", include_in_schema=False)
def serve_favicon():
    favicon_ico = FRONTEND_DIR / "favicon.ico"
    if not favicon_ico.exists():
        favicon_ico = ASSETS_DIR / "images" / "favicon.ico"
    if favicon_ico.exists():
        return FileResponse(str(favicon_ico), media_type="image/x-icon")
    return Response(status_code=404)
