# RefillCare Enterprise — Clinical AI Refill Prediction & Med-Sync Platform

> **Architecture Tier:** Enterprise Production Architecture  
> **System Status:** Production-Ready Unified Platform  
> **Automated Test Suite:** 100% Passing (Comprehensive Regression & Unit Test Coverage)  

RefillCare Enterprise is an end-to-end clinical intelligence, predictive refill scheduling, and multi-prescription synchronization platform engineered specifically for retail pharmacy chains and omnichannel healthcare networks.

---

## 1. Executive Summary & Business Value

### The Retail Pharmacy Challenge
Patients managing chronic conditions (Hypertension, Type-2 Diabetes, Cardiovascular Disease, Thyroid Disorders, and Respiratory Illnesses) frequently experience therapy gaps due to unintentional refill delays, leading to medication non-adherence, adverse health outcomes, and significant pharmacy revenue leakage:
- **Flawed Fixed 30-Day Refill Cadences:** Patients consume varying daily dosages (OD, BD, TID, QOD), purchase varying pack sizes (10-tablet trial strips vs 30/60/90-tablet boxes), and purchase before their current supply runs out (accumulating home inventory).
- **Noisy POS/ERP Transaction Data:** Sales streams frequently contain wholesale/inter-store transfers, acute/PRN purchases (analgesics, antibiotics, cold preparations), missing phone numbers, and ambiguous date formats.
- **Patient Notification Fatigue:** Sending uncoordinated individual WhatsApp reminders for 3 different medications across different days creates communication friction, leading to patient dissatisfaction and high message opt-out rates.

### The RefillCare Solution
RefillCare automates the complete operational lifecycle from raw ERP sales ingestion to unified, synchronized patient outreach:
- **Zero-Contamination Wholesale Isolation:** Automatically separates B2B inter-store transfers (`SB/...`) from retail customer sales (`S0/...`) upstream before modeling.
- **Behavioral & Clinical Chronic Classifier:** Filters acute PRN purchases using longitudinal purchase frequency and consensus clinical classification.
- **Dual-Path Clinical Decision Routing:**
  - **Path A (Chronic Adherence, $\ge 6$ Purchases):** Evaluated with Median Absolute Deviation (MAD) stability filtering and 3-Head Quantile XGBoost Regression ($P_{10}, P_{50}, P_{90}$).
  - **Path B (Developing Adherence, $< 6$ Purchases):** Governed by Days-of-Supply (DOS) calculated from verified quantity sold and daily consumption rates with physical safety bounds.
- **5 Clinical Behavioral Archetypes:** Models multi-pack scaling, early top-up carryover ($R_{inv}$), partial 10-strip clamping, post-lapse reset, and consensus physical bounding ($[0.65, 1.50] \times D_{supply}$).
- **Med-Sync (Multi-Prescription Synchronization Engine):** Clusters multiple active chronic prescriptions due within an 8-day synchronization window into a single unified appointment reminder, reducing messaging noise by 50%+.
- **Multi-Stage Lifecycle Management:** Schedules proactive outreach (`-7d`, `-3d`, `-1d`, `0d`, `+2d`, `+5d`, `+40d`) with automatic lifecycle invalidation (`SUPERSEDED_BY_PURCHASE`) upon repurchase.
- **Pharmacist Review Queue & Dual-Format Exports:** Routes ambiguous or missing-phone records to a clinical review queue, and generates delivery-ready 10-column CSVs and structured JSON payloads on demand.

---

## 2. End-to-End System Architecture

```text
  [ Raw POS / ERP Sales Stream (CSV / Excel) ]
                       │
                       ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. INGESTION, CLEANSING & B2B ISOLATION GATE                │
  │    (refillcare/data/monthly_ingestion.py & dates.py)        │
  │    • Strict Date Normalization & Multi-Format Detection     │
  │    • Wholesale / Inter-Store (SB/...) Upstream Isolation    │
  │    • 10-Digit Mobile Display & E.164 Delivery Sanitization  │
  │    • Traceable Import Batches with 1-Click Rollback         │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 2. CLINICAL CHRONIC CLASSIFIER & DUAL-PATH ENGINE           │
  │    (refillcare/engine/unified_engine.py & models/)          │
  │    • PRN / Analgesic Purging (Dolo, Paracetamol, etc.)      │
  │    • Path A (>= 6 buys): 3-Head Quantile Regressors         │
  │      (P10 Early Warning, P50 Median, P90 Lapsed Bound)      │
  │    • Path B (< 6 buys): Days-of-Supply Physical Model       │
  │    • 5 Behavioral Archetypes & Partial Strip Clamping       │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 3. MED-SYNC APPOINTMENT SYNCHRONIZATION ENGINE              │
  │    (refillcare/engine/med_sync.py)                          │
  │    • Temporal clustering within default 8-day sync window   │
  │    • Clinical anchor designation (highest stability tier)   │
  │    • Consolidated multi-medication WhatsApp copy generation │
  │    • Proactive 30-day box upsell recommendations            │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 4. ENTERPRISE PERSISTENCE & LIFECYCLE CONTROLLER            │
  │    (refillcare/engine/persistence.py & database/)           │
  │    • Persistent SQLite Database (enterprise.db)             │
  │    • RefillDecisionModel (Audited Rationale & Bounds)       │
  │    • ReminderCycleModel & 6-Stage Lifecycle Tracking        │
  │    • Repurchase Auto-Reset (SUPERSEDED_BY_PURCHASE)         │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 5. MULTI-CHANNEL DELIVERY & PHARMACIST INTERFACES           │
  │    • Streamlit Dashboard (app_refillcare.py): Operations,   │
  │      Review Queue, Date/Month Filters, Dual-Format Exports  │
  │    • FastAPI Application (api/main.py): REST Endpoints      │
  │    • Vanilla HTML/JS Frontend: Real-Time SPA Interface      │
  │    • WhatsApp CPaaS Integration: Template Message Dispatch  │
  └─────────────────────────────────────────────────────────────┘
```

---

## 3. Core Business Logic & Algorithmic Foundations

### A. Dual-Path Clinical Decision Routing
1. **Path A (Established Chronic Cadence, $\ge 6$ Purchases):**
   - Analyzes personal purchase intervals using Median Absolute Deviation (MAD) stability filtering.
   - For high-stability cohorts, predictions leverage personal cadence coupled with 3-Head Quantile XGBoost Regressors.
   - Quantile predictions yield an uncertainty envelope:
     $$P_{10} \le P_{50} \le P_{90}$$
     where $P_{10}$ flags early refill risk, $P_{50}$ represents the expected refill date, and $P_{90}$ sets the lapsed adherence threshold.
2. **Path B (Developing Adherence / New Chronic Patients, $< 6$ Purchases):**
   - Calculates baseline supply duration from Days-of-Supply (DOS):
     $$D_{supply} = \frac{\text{Verified Units Purchased}}{\text{Consensus Daily Consumption Rate}}$$
   - Applies recurrence validation across multi-month observation windows.

### B. 5 Clinical Behavioral Archetypes
- **Multi-Pack Scaled:** Automatically scales supply days when patients purchase multi-packs (e.g. 60 or 90 units instead of standard 30 units).
- **Early Top-Up Carryover ($R_{inv}$):** When a patient refills prior to running out, remaining pill inventory is credited forward as carryover supply.
- **Partial Purchase Scaled (10-Strip Clamping):** Prevents premature or delayed notifications when patients buy short 10-capsule strips, clamping supply to 10–15 days per pack.
- **Post-Lapse Reset:** Resets baseline cadence strictly based on newly purchased quantity following prolonged treatment gaps ($>75$ days).
- **Consensus Physical Bounding:** Enforces physical boundaries $[0.65, 1.50] \times D_{supply}$ to prevent statistical model divergence on noisy transaction sequences.

### C. Med-Sync Appointment Bundling Algorithm
- Iterates over active patient prescriptions and performs temporal clustering using a configurable synchronization window (default: **8 days**).
- Designates an anchor prescription based on clinical stability tier and earliest refill date.
- Compiles a consolidated multi-item WhatsApp notification with interactive response options (`1` to Confirm All, `2` to Customize).
- Operates in dual target modes:
  - **Filter by Month:** Displays all consolidated bundles across an entire target prediction month.
  - **Filter by Date:** Pinpoints patient bundles whose primary anchor appointment falls on a specific date.

### D. Multi-Stage Lifecycle & Repurchase Supersession
- Schedules patient engagement across 6 strategic lifecycle touchpoints:
  - **Day -7:** Advance refill planning notification.
  - **Day -3:** Refill preparation reminder.
  - **Day -1:** Refill due tomorrow reminder.
  - **Day 0 (Due):** Refill due today alert.
  - **Day +2:** First adherence follow-up.
  - **Day +5:** Urgency follow-up.
  - **Day +40:** Lapsed patient re-engagement.
- **Repurchase Auto-Reset:** When a patient purchases their medication on or before a scheduled reminder date, all remaining pending stages for that prescription cycle are automatically transitioned to `SUPERSEDED_BY_PURCHASE` with full audit provenance.

---

## 4. Repository Structure

```text
ai-mediastra-whatsapp-reminder/
├── refillcare/             # Core Clinical Decision Engine & Domain Logic
│   ├── engine/             # Decision routing, Med-Sync bundler, persistence manager
│   │   ├── decision_types.py   # RefillDecision dataclass, constants, stages
│   │   ├── med_sync.py         # Multi-prescription bundling & sync engine (default 8d)
│   │   ├── persistence.py      # SQLAlchemy persistence & lifecycle manager
│   │   ├── reminder_lifecycle.py # WhatsApp clinical template copies
│   │   └── unified_engine.py   # Unified decision engine & stability routing
│   ├── models/             # Stability classifier, 3-head quantile ML models
│   ├── features/           # Consumption velocity, Days-of-Supply (DOS)
│   ├── data/               # Ingestion, date parsing, chronic classifier
│   │   ├── medication_classifier.py # Behavioral chronic & PRN exclusion classifier
│   │   ├── monthly_ingestion.py     # Batch ingestion, rollback, 10-digit phone formatter
│   │   └── dates.py                 # Multi-format date normalization
│   ├── evaluation/         # Holdout validation, backtesting, benchmark suites
│   └── config/             # Path thresholds, stability limits, system constants
│
├── database/               # Database & Persistence Layer
│   ├── connection.py       # SQLAlchemy engine & SessionLocal (enterprise.db)
│   ├── models.py           # ORM schemas: RefillDecisionModel, ReminderCycleModel, ReminderStageModel
│   └── fetch_sales.py      # Historical transaction access queries
│
├── reminder/               # Reminder Delivery & Scheduling Utilities
│   ├── reminder_engine.py  # 10-column delivery CSV generation with phone sanitization
│   ├── send_message.py     # Single/batch WhatsApp dispatch with safety dry-run guard
│   ├── storage.py          # SQLite snapshot store for review queues
│   ├── scheduler.py        # Multi-stage schedule utilities (-7d to +40d)
│   └── message_logs.py     # Audit logging of sent and simulated messages
│
├── services/               # External Communication Integrations
│   ├── xinno_whatsapp.py   # Xinno CPaaS WhatsApp Business API client
│   ├── xinno_image_template.py # WhatsApp image template API client
│   └── cloudinary_image.py # Cloudinary image hosting for media campaigns
│
├── utils/                  # Cross-Cutting Engineering Utilities
│   ├── validators.py       # Canonical phone normalization (10-digit display & E.164 delivery)
│   ├── column_aliases.py   # Header mapping for 15+ pharmacy ERP formats
│   ├── bulk_send.py        # Throttled batch dispatch with rate limiting
│   └── audit.py            # Phone masking & compliance audit logging
│
├── api/                    # Enterprise FastAPI Application
│   ├── main.py             # REST API entry point, CORS, OpenAPI docs, static routes
│   ├── schemas.py          # Pydantic request and response schemas
│   └── services.py         # Business service layer (Med-Sync, Quantiles, KPIs, Queues)
│
├── frontend/               # Web Application Interface
│   ├── index.html          # Dashboard interface with Date/Month toggles & search
│   ├── style.css           # Design system tokens and styles
│   ├── js/app.js           # Client application logic and event handlers
│   └── js/api.js           # REST API client
│
├── data/                   # Data Directory
│   └── refillcare/processed/
│       ├── enterprise.db   # Production SQLite database
│       ├── clean_transactions.parquet # Cleaned, B2B-purged transaction records
│       ├── purchase_history.parquet   # Canonical longitudinal purchase dataset
│       └── models/         # Trained serialized machine learning models
│
├── scripts/                # Production Engineering & Diagnostic Scripts
│   ├── train_chronic_specialized_model.py # 3-Head Quantile XGBoost training
│   ├── run_offline_validation.py         # Holdout evaluation & metrics
│   ├── verify_system_health.py           # End-to-end system health check
│   └── build_refillcare_notebooks.py     # Reproducible walkthrough notebooks
│
├── tests/                  # Comprehensive Automated Test Suite
│   ├── api/                # FastAPI endpoint integration tests
│   └── refillcare/         # Engine, Med-Sync, Classifier, Lifecycle tests
│
├── app_refillcare.py       # Main Operations Dashboard (Streamlit)
├── requirements.txt        # Python package dependencies
├── pytest.ini              # Pytest configuration
└── README.md               # Master System Documentation
```

---

## 5. Operations & Execution Guide

### A. CLI Commands

```bash
# 1. Run Full Batch Prediction & Persistence Pipeline
python run_prediction.py

# 2. Query and Export Due Refill Reminders
python run_reminder.py

# 3. Simulate or Dispatch WhatsApp Messages (Safety Dry-Run Enabled by Default)
python run_message.py

# 4. Retrain Specialized 3-Head Quantile Regressors
python scripts/train_chronic_specialized_model.py

# 5. Run System Health Diagnostic Suite
python scripts/verify_system_health.py
```

### B. Starting User Interfaces & REST API

```bash
# 1. Start Streamlit Clinical Operations Dashboard
streamlit run app_refillcare.py

# 2. Start Enterprise FastAPI Backend Server (Interactive Docs: http://127.0.0.1:8000/docs)
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

### C. REST API Endpoints Reference

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/v2/med-sync/bundles` | `GET` | Retrieve synchronized patient bundles (supports `sync_window_days`, `target_date`, and `target_month`). |
| `/api/v2/models/quantiles` | `GET` | Retrieve 3-Head Quantile uncertainty envelope metrics ($P_{10}, P_{50}, P_{90}$). |
| `/api/reminders/daily` | `GET` | Fetch scheduled reminder queue for a target date or full target month. |
| `/api/reminders/monthly` | `GET` | Fetch monthly aggregated reminder delivery queue. |
| `/api/reminders/{id}/approve` | `POST` | Pharmacist manual approval of a pending reminder stage. |
| `/api/reminders/{id}/reject` | `POST` | Pharmacist rejection of a pending reminder stage with reason code. |
| `/api/sales/monthly-upload` | `POST` | Ingest and validate new monthly sales file with batch traceability. |
| `/api/sales/rollback` | `POST` | 1-Click rollback of an ingested sales batch. |

---

## 6. Testing & Quality Assurance

The test suite validates data ingestion, wholesale isolation, clinical classification, 3-head quantile prediction, Med-Sync clustering, persistence lifecycle auto-reset, and REST endpoints.

```bash
# Execute entire automated test suite
pytest tests/ -q

# Execute Med-Sync and API integration tests specifically
pytest tests/refillcare/test_med_sync.py tests/api/test_enterprise_api.py -v
```
