"""FastAPI Enterprise Application for RefillCare & Mediastra.

Production-grade RESTful API with automated OpenAPI docs, CORS middleware,
database session dependency injection, and static frontend mounting.
"""

from __future__ import annotations

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
    Query,
    status,
    Response,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from sqlalchemy.orm import Session

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.connection import get_db, init_db
from api.schemas import (
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
)
from api.services import EnterpriseServices
from refillcare.engine.persistence import RefillPersistenceManager
from database.models import RefillDecisionModel, ReminderCycleModel, ReminderStageModel

# Initialize database schema
init_db()

# Create FastAPI app
app = FastAPI(
    title="RefillCare & Mediastra Enterprise API",
    description="Enterprise Medication Refill Prediction, Prescription Tracking, and WhatsApp Gateway API.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for cross-origin frontend support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = PROJECT_ROOT / "frontend"


# ------------------------------------------------------------------------------
# 1. SYSTEM HEALTH & METRICS
# ------------------------------------------------------------------------------
@app.get("/health", tags=["System"])
def health_check():
    """System health and readiness probe."""
    return {
        "status": "healthy",
        "service": "RefillCare Enterprise API",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "database": "connected",
    }


@app.get("/api/v1/analytics/kpi", response_model=OperationsKPISummary, tags=["Analytics"])
def get_kpi_summary(db: Session = Depends(get_db)):
    """Retrieve executive operations summary KPIs."""
    srv = EnterpriseServices(db)
    return srv.get_operations_kpi()


# ------------------------------------------------------------------------------
# 2. SALES DATA INGESTION & BATCH APIS
# ------------------------------------------------------------------------------
@app.post("/api/v1/sales/preview", response_model=SalesPreviewResponse, tags=["Sales Ingestion"])
async def preview_sales_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Validate and preview an uploaded sales file with strict DD-MM-YYYY date diagnostics."""
    srv = EnterpriseServices(db)
    contents = await file.read()
    try:
        res = srv.preview_sales_file(contents, file.filename or "uploaded_sales.xlsx")
        return res
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/api/v1/sales/ingest", response_model=SalesIngestionResponse, tags=["Sales Ingestion"])
async def ingest_sales_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Ingest validated sales records into persistent storage with an immutable batch ID."""
    srv = EnterpriseServices(db)
    contents = await file.read()
    try:
        res = srv.ingest_sales_file(contents, file.filename or "uploaded_sales.xlsx")
        return res
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/api/v1/sales/batches", response_model=List[ImportBatchResponse], tags=["Sales Ingestion"])
def list_batches(db: Session = Depends(get_db)):
    """Retrieve all historical import batches."""
    srv = EnterpriseServices(db)
    return srv.list_import_batches()


@app.post("/api/v1/sales/batches/{batch_id}/rollback", response_model=RollbackResponse, tags=["Sales Ingestion"])
def rollback_import_batch(batch_id: str, db: Session = Depends(get_db)):
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
def evaluate_outcomes(db: Session = Depends(get_db)):
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
@app.get("/api/v1/reminders/daily", response_model=List[ReminderItem], tags=["Reminders"])
def get_daily_reminders(
    target_date: Optional[date] = Query(None, description="Target reminder date (defaults to today)"),
    mobile_status: str = Query("All", description="All, Valid, Missing"),
    db: Session = Depends(get_db),
):
    """Get scheduled reminders for a specific date."""
    srv = EnterpriseServices(db)
    dt = target_date or date.today()
    return srv.get_daily_reminders(target_date=dt, mobile_filter=mobile_status)


@app.get("/api/v1/reminders/export-csv", tags=["Reminders"])
def export_reminder_csv(
    target_date: Optional[date] = Query(None, description="Target reminder date"),
    db: Session = Depends(get_db),
):
    """Download delivery-ready CSV with 10 standard columns for valid mobile numbers."""
    srv = EnterpriseServices(db)
    dt = target_date or date.today()
    csv_bytes = srv.export_reminder_csv(target_date=dt)
    filename = f"refill_reminders_{dt.strftime('%Y%m%d')}.csv"
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ------------------------------------------------------------------------------
# 5. MODEL REGISTRY & VERSIONING APIS
# ------------------------------------------------------------------------------
@app.get("/api/v1/models", response_model=List[ModelVersionItem], tags=["Model Registry"])
def list_models(db: Session = Depends(get_db)):
    """List registered model versions and performance benchmarks."""
    srv = EnterpriseServices(db)
    return srv.list_models()


@app.post("/api/v1/models/{version_id}/activate", tags=["Model Registry"])
def activate_model(version_id: str, db: Session = Depends(get_db)):
    """Set active production model version."""
    srv = EnterpriseServices(db)
    try:
        return srv.activate_model(version_id)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


# ------------------------------------------------------------------------------
# 6. WHATSAPP GATEWAY DISPATCH
# ------------------------------------------------------------------------------
@app.post("/api/v1/whatsapp/dispatch", response_model=WhatsAppDispatchResponse, tags=["WhatsApp Gateway"])
def dispatch_whatsapp(
    req: WhatsAppBatchDispatchRequest,
    db: Session = Depends(get_db),
):
    """Dispatch WhatsApp reminders via Xinno API (supports dry-run mode)."""
    srv = EnterpriseServices(db)
    try:
        return srv.dispatch_whatsapp_batch(
            reminder_date=req.reminder_date,
            dry_run=req.dry_run,
            batch_size=req.batch_size,
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


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
    eval_date = target_date or date.today()
    pm = RefillPersistenceManager()
    return pm.get_today_review_queue(db, target_date=eval_date, path_filter=path, stability_filter=stability)


@app.get("/api/reminders/{reminder_id}", response_model=ReminderQueueItem, tags=["V1 Pharmacist Review Queue"])
def get_reminder_stage_detail(reminder_id: str, db: Session = Depends(get_db)):
    """Get single reminder stage detail."""
    stage = db.query(ReminderStageModel).filter(ReminderStageModel.reminder_id == reminder_id).first()
    if not stage:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Reminder stage '{reminder_id}' not found.")
    pm = RefillPersistenceManager()
    queue = pm.get_today_review_queue(db, target_date=stage.target_send_date)
    for item in queue:
        if item["reminder_id"] == reminder_id:
            return item
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Reminder details not available.")


@app.post("/api/reminders/{reminder_id}/approve", response_model=ReviewActionResponse, tags=["V1 Pharmacist Review Queue"])
def approve_reminder(reminder_id: str, db: Session = Depends(get_db)):
    """Approve a reminder stage for dispatch."""
    pm = RefillPersistenceManager()
    try:
        res = pm.approve_reminder_stage(db, reminder_id)
        return ReviewActionResponse(reminder_id=reminder_id, status=res["status"], success=True, message="Approved successfully")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/api/reminders/{reminder_id}/reject", response_model=ReviewActionResponse, tags=["V1 Pharmacist Review Queue"])
def reject_reminder(reminder_id: str, req: ReviewActionRequest, db: Session = Depends(get_db)):
    """Reject a reminder stage with a pharmacist reason."""
    pm = RefillPersistenceManager()
    try:
        res = pm.reject_reminder_stage(db, reminder_id, reason=req.reason or "Pharmacist rejected")
        return ReviewActionResponse(reminder_id=reminder_id, status=res["status"], success=True, message=res["reason"])
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/api/reminders/{reminder_id}/dispatch", response_model=ReviewActionResponse, tags=["V1 Pharmacist Review Queue"])
def dispatch_single_reminder(reminder_id: str, req: ReviewActionRequest, db: Session = Depends(get_db)):
    """Controlled dispatch of a single reminder (DRY-RUN by default)."""
    pm = RefillPersistenceManager()
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
            "last_purchase_date": c.last_purchase_date.strftime("%Y-%m-%d"),
            "expected_refill_date": c.expected_refill_date.strftime("%Y-%m-%d"),
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


# ------------------------------------------------------------------------------
# 8. FRONTEND STATIC ASSETS & SPA ROUTING
# ------------------------------------------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse, tags=["Web UI"])
    def serve_frontend_index():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
        return HTMLResponse("<h2>RefillCare Enterprise API Online. Frontend loading...</h2>")
