"""FastAPI Enterprise Application for RefillCare & Mediastra.

Production-grade RESTful API with automated OpenAPI docs, CORS middleware,
database session dependency injection, and static frontend mounting.
"""

from __future__ import annotations

import json
import os
import sys
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
    ColumnMappingSaveRequest,
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
from api.security import require_api_key
from database.models import RefillDecisionModel, ReminderCycleModel, ReminderStageModel

# The database schema is created / additively migrated lazily on first use
# (database.connection.get_sessionmaker), so importing the app never touches a live database.

# Create FastAPI app
app = FastAPI(
    title="RefillCare & Mediastra Enterprise API",
    description="Enterprise Medication Refill Prediction, Prescription Tracking, and WhatsApp Gateway API.",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

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
    allow_headers=["Content-Type", "X-API-Key"],
)

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Invalid user input (bad month, reversed range, ...) is a 400, not a silent default."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


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
        "status": "healthy" if db_ok else "degraded",
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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/api/v1/sales/column-mapping", tags=["Sales Ingestion"])
def save_column_mapping(req: ColumnMappingSaveRequest, actor: str = Depends(require_api_key), db: Session = Depends(get_db)):
    """Save the confirmed mapping for this store and header layout (used automatically next time)."""
    srv = EnterpriseServices(db)
    try:
        return srv.save_column_mapping(req.headers, req.mapping, actor=actor)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/api/v1/sales/ingest", response_model=SalesIngestionResponse, tags=["Sales Ingestion"])
async def ingest_sales_file(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    column_mapping: Optional[str] = Form(None),
    actor: str = Depends(require_api_key),
    db: Session = Depends(get_db),
):
    """Ingest validated sales records into persistent storage with an immutable batch ID.

    The response re-plans the ML train/validation/test windows on the new data
    (``ml_data_freshness``). With ``ML_AUTO_RETRAIN_ON_INGEST=true`` a due retrain cycle is queued
    in the background; it only promotes when ``ML_AUTO_PROMOTE=true`` and the gate approves.
    """
    srv = EnterpriseServices(db)
    contents = await _read_upload(file)
    try:
        res = srv.ingest_sales_file(contents, file.filename or "uploaded_sales.xlsx", _mapping_form(column_mapping), actor=actor)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    fresh = res.get("ml_data_freshness") or {}
    if fresh.get("retrain_recommended") and _env_flag("ML_AUTO_RETRAIN_ON_INGEST"):
        from api.ml_routes import RetrainRequest, _run_retrain_background

        background.add_task(_run_retrain_background, RetrainRequest(promote_best=_env_flag("ML_AUTO_PROMOTE")), actor)
        res["ml_retrain_queued"] = True
    return res


def _env_flag(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() in ("1", "true", "yes", "on")


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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


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
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


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
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/api/v1/whatsapp/webhook", response_model=WhatsAppWebhookResponse, tags=["WhatsApp Gateway"])
async def receive_whatsapp_webhook(
    request: Request,
    db: Session = Depends(get_db),
):
    """Receive incoming WhatsApp replies/callbacks from Xinno or Meta cloud API.
    Automatically honours STOP / UNSUBSCRIBE messages to opt out patients.
    """
    srv = EnterpriseServices(db)
    try:
        body = await request.json()
    except Exception:
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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/api/reminders/{reminder_id}/reject", response_model=ReviewActionResponse, tags=["V1 Pharmacist Review Queue"])
def reject_reminder(reminder_id: str, req: ReviewActionRequest, actor: str = Depends(require_api_key),
    db: Session = Depends(get_db)):
    """Reject a reminder stage with a pharmacist reason."""
    pm = _pm()
    try:
        res = pm.reject_reminder_stage(db, reminder_id, reason=req.reason or "Pharmacist rejected")
        return ReviewActionResponse(reminder_id=reminder_id, status=res["status"], success=True, message=res["reason"])
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


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
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse, tags=["Web UI"])
    def serve_frontend_index():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
        return HTMLResponse("<h2>RefillCare Enterprise API Online. Frontend loading...</h2>")
