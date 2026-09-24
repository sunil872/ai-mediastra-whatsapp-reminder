# RefillCare & Mediastra — Remaining Phases & Next Steps Implementation Plan

**Document Version:** 1.0.0  
**Date:** 2026-09-23  
**Status:** Approved Roadmap & Implementation Plan  
**Target Repository:** `ai-mediastra-whatsapp-reminder`

---

## Executive Summary & Current State

RefillCare & Mediastra has completed **Phases 1 through 17K**, establishing:

1. **Machine Learning Core**: Cadence feature engineering, XGBoost training, historical median predictors, and the Phase 17K Path A eligibility architecture audit.
2. **Enterprise API & Data Layer**: FastAPI REST API (`api/main.py`), SQLAlchemy ORM models (`database/models.py`), SQLite/PostgreSQL connection engine (`database/connection.py`).
3. **Frontend Dashboard**: Single-page modern UI (`frontend/index.html`, `js/app.js`, `css/styles.css`).
4. **WhatsApp Communication Gateway**: Xinno CPaaS integration for text and rich-media templates with dry-run safety gates.

The following **Phases 18 through 24** represent the remaining work required to bring RefillCare to full production scale, autonomous scheduling, and closed-loop clinical efficacy.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             COMPLETED (Phases 1-18)                              │
│  Data Ingestion ➔ Feature Eng ➔ Model R&D ➔ FastAPI Layer ➔ UI ➔ Xinno Client    │
│  ➔ Phase 18: Path B Multi-Month Recency Architecture                             │
└─────────────────────────────────────────┬────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         REMAINING PHASES (Phases 19-24)                          │
├────────────────────────────────┬─────────────────────────────────────────────────┤
│ Phase 19: Full-Stack SPA Sync  │ End-to-end UI drag-and-drop & API synchronization│
├────────────────────────────────┼─────────────────────────────────────────────────┤
│ Phase 20: Automated Scheduler  │ Daily automated cron jobs & batch queue engine  │
├────────────────────────────────┼─────────────────────────────────────────────────┤
│ Phase 21: Production Database  │ PostgreSQL migration, Alembic & Multi-Branch    │
├────────────────────────────────┼─────────────────────────────────────────────────┤
│ Phase 22: WhatsApp Webhooks    │ Two-way delivery receipts & patient replies     │
├────────────────────────────────┼─────────────────────────────────────────────────┤
│ Phase 23: Model Promotion & CI │ Path A activation flag & automated retraining   │
├────────────────────────────────┼─────────────────────────────────────────────────┤
│ Phase 24: 30-Day Live Pilot    │ Controlled real-world pharmacy pilot execution  │
└────────────────────────────────┴─────────────────────────────────────────────────┘
```

---

## Detailed Phases & Status

### 📌 Phase 18: Path B Prediction Engine (Multi-Month Recency & Recurrence Architecture) — [COMPLETED]

- **Objective**: Extend automated refill predictions to patients with fewer than 6 historical purchases (`purchase_count < 6`) using explicit recency and multi-month recurrence evaluation:
  ```
                   PATH B
                      │
                      ▼
            Latest-purchase recency (<=180 days)
                      │
                      ▼
            ┌─────────────────────┐
            │ 3-month evaluation  │
            │ ≥2 distinct months  │
            └─────────────────────┘
                      │
                if not satisfied
                      ▼
            ┌─────────────────────┐
            │ 6-month evaluation  │
            │ ≥3 distinct months  │
            └─────────────────────┘
                      │
                      ▼
                 Eligible
  ```
- **Implementation Highlights**:
  1. `refillcare/models/path_b_classifier.py`: Evaluates latest purchase recency ($\le 180$ days), 3-month window ($\ge 2$ distinct calendar months), and 6-month fallback window ($\ge 3$ distinct calendar months). Strict zero-future data leakage.
  2. `refillcare/models/path_b_predictor.py`: Generates `predicted_cycle_days`, `next_refill_date`, and `reminder_date` (applying the 3-day lead buffer).
  3. `preprocessing/feature_table.py` & `model/predictor.py`: Integrated into core feature engineering and prediction pipelines.
  4. `tests/refillcare/test_path_b_predictor.py`: 9 comprehensive unit and integration tests (100% passing).
- **Deliverables**:
  - `refillcare/models/path_b_classifier.py`
  - `refillcare/models/path_b_predictor.py`
  - `tests/refillcare/test_path_b_predictor.py`

---

### 📌 Phase 19: Full-Stack Single-Page App (SPA) Complete Integration

- **Objective**: Connect all dashboard views in `frontend/index.html` and `frontend/js/app.js` with live backend API endpoints.
- **Key Tasks**:
  1. **Batch Ingestion UI**: Implement drag-and-drop Excel/CSV upload with real-time schema validation and column alias preview.
  2. **Interactive Reminder Queue**: Add server-side pagination, search by patient name/phone, filtering by branch and mobile status, and single-click reminder status toggles.
  3. **Model Management Panel**: Live table of model versions, metrics comparison (MAE, Coverage, ±3d Accuracy), and one-click model activation.
  4. **Analytics Charts**: Integrate Chart.js/ApexCharts for daily dispatch volume, conversion trends, and cohort retention.
- **Deliverables**:
  - Enhanced `frontend/js/app.js` & `frontend/js/api.js`
  - Complete toast feedback & error boundary handling
  - `docs/PHASE_19_SPA_API_INTEGRATION.md`

---

### 📌 Phase 20: Automated Background Worker & Daily Job Scheduler

- **Objective**: Automate daily operations so manual script execution is no longer required.
- **Key Tasks**:
  1. **Daily Job Orchestration**: Set up APScheduler / Celery / Cron worker running at **08:00 AM IST** daily.
  2. **Workflow Pipeline**:
     - Pull latest sales records & ingest.
     - Compute daily feature vector & generate predictions.
     - Build daily reminder candidates (`reminder_date == today`).
     - Trigger dry-run or live WhatsApp dispatch based on branch settings.
  3. **Operational Failure Alerts**: Send automated email/Slack alerts to administrators if daily ingestion or dispatch encounters errors.
- **Deliverables**:
  - `services/scheduler.py` & `scripts/run_daily_cron.py`
  - Background task management in FastAPI startup event
  - `docs/PHASE_20_AUTOMATED_SCHEDULER.md`

---

### 📌 Phase 21: Production PostgreSQL Database & Multi-Tenant Migration

- **Objective**: Move from SQLite development database to production-grade PostgreSQL with schema versioning and branch isolation.
- **Key Tasks**:
  1. **Alembic Database Migrations**: Initialize Alembic migration scripts for database schema evolution without data loss.
  2. **PostgreSQL Connection Pool**: Configure SQLAlchemy connection pooling with health check probes (`pool_pre_ping=True`).
  3. **Multi-Branch Isolation**: Ensure all sales, predictions, reminders, and audit records carry strict `branch_id` and role-based access control.
- **Deliverables**:
  - `alembic/` directory and migration scripts
  - PostgreSQL deployment configuration in Docker / Kubernetes
  - `docs/PHASE_21_PRODUCTION_DATABASE_MIGRATION.md`

---

### 📌 Phase 22: Bi-Directional WhatsApp Webhooks & Inbound Response Handling

- **Objective**: Capture real-time WhatsApp message delivery states and automate patient replies.
- **Key Tasks**:
  1. **Webhook Endpoint**: Create `POST /api/v1/whatsapp/webhook` to receive Xinno/Meta callback events.
  2. **Delivery Status Tracking**: Update reminder records in real-time (`SENT` ➔ `DELIVERED` ➔ `READ` ➔ `FAILED`).
  3. **Inbound Reply Parser**: Handle patient responses:
     - `"REFILL"` / `"YES"` ➔ Mark customer as **Confirmed Refill**, alert pharmacist.
     - `"STOP"` / `"UNSUBSCRIBE"` ➔ Add phone to suppression list (compliance).
     - `"CALL ME"` ➔ Create pharmacist follow-up task.
- **Deliverables**:
  - `api/routers/webhook.py` & `services/webhook_handler.py`
  - Suppression list table in database
  - `docs/PHASE_22_WHATSAPP_WEBHOOKS.md`

---

### 📌 Phase 23: Phase 17K Promotion & Automated CI/CD Model Retraining

- **Objective**: Finalize the Phase 17K audit policy decision and enable the feature flag.
- **Key Tasks**:
  1. **Policy Selection**: Formally select between **Approach A** (17J XGB safety gate ➔ median), **Approach B** (Historical-only ➔ median), or **Approach C** (History-stricter hybrid).
  2. **Enable Feature Flag**: Set `USE_PATH_A_CLASSIFIER_V17F=True` in production configuration.
  3. **Automated Monthly Retraining Pipeline**: Train new XGBoost models on monthly rolling sales windows, compare metrics against active production baseline, and register new model artifacts in database.
- **Deliverables**:
  - `services/retraining_pipeline.py`
  - Model promotion audit log
  - `docs/PHASE_23_MODEL_PROMOTION_AND_CICD.md`

---

### 📌 Phase 24: 30-Day Live Controlled Pharmacy Pilot & Outcome Evaluation

- **Objective**: Execute a real-world, controlled clinical pilot across selected pharmacy branches.
- **Key Tasks**:
  1. **Control vs. Treatment Split**: 50% treatment group (receives WhatsApp reminder), 50% control group (no reminder sent).
  2. **30-Day Execution**: Run automated daily dispatches under pharmacist supervision.
  3. **Outcome Metrics**:
     - **Refill Conversion Lift**: Increase in on-time refill purchases (within ±3 days) vs control.
     - **Revenue Impact**: Incremental medication sales generated.
     - **Patient Opt-Out Rate**: Unsubscribe / spam report percentage (< 0.5% target).
- **Deliverables**:
  - `scripts/evaluate_pilot_outcomes.py`
  - Executive outcome dashboard
  - `docs/PHASE_24_PILOT_OUTCOME_REPORT.md`

---

## Suggested Implementation Roadmap & Priority

| Phase                                         | Priority            | Estimated Complexity | Dependencies  |
| :-------------------------------------------- | :------------------ | :------------------- | :------------ |
| **Phase 19: Full-Stack SPA Integration**      | 🔴 High (Immediate) | Medium               | Phase 17      |
| **Phase 20: Automated Daily Scheduler**       | 🔴 High             | Medium               | Phase 19      |
| **Phase 22: WhatsApp Webhooks & Receipts**    | 🟡 Medium           | Medium               | Phase 20      |
| **Phase 18: Path B Low-History Predictor**    | 🟡 Medium           | High                 | Phase 17K     |
| **Phase 21: Production PostgreSQL & Alembic** | 🟡 Medium           | Medium               | Phase 19      |
| **Phase 23: Model Promotion & CI Retraining** | 🟢 Normal           | Medium               | Phase 17K, 18 |
| **Phase 24: 30-Day Controlled Pilot**         | 🚀 Milestone        | High                 | Phases 19–23  |
