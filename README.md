# RefillCare Enterprise — Clinical AI Refill Prediction & Med-Sync Platform

> **Document Version:** 2.0.0 (Enterprise Production Architecture)  
> **Last Updated:** 2026-09-28  
> **Status:** Production-Ready V2 Deployed  
> **Automated Test Suite:** 751 / 751 Passed (100% Green)  

RefillCare is an enterprise-grade clinical intelligence, predictive refill scheduling, and multi-prescription synchronization platform designed specifically for retail pharmacies.

---

## 1. Problem Statement & Value Proposition

### The Retail Pharmacy Challenge
Chronic patients (Hypertension, Diabetes, Cardiology, Thyroid, Respiratory) frequently forget or delay refilling their ongoing maintenance therapies.
- **Flawed Fixed 30-Day Intervals:** Patients take different daily dosages (OD, BD, TID, QOD), purchase varying pack sizes, and refill before running out (accumulating home inventory).
- **Messy POS/ERP Transaction Data:** Sales logs contain wholesale/inter-store transfers (`SB/` prefixes), acute/one-time buyers (cold remedies, eye drops, pain relievers), missing mobile numbers, and ambiguous date formats (`DD-MM-YYYY` vs `MM-DD-YYYY`).
- **Patient Notification Fatigue:** Sending 3 separate WhatsApp reminders for 3 different medications on Monday, Wednesday, and Friday irritates patients and leads to message opt-outs.

### The RefillCare Solution
RefillCare automates the entire journey from raw POS ingestion to unified patient outreach:
- **Zero-Contamination Wholesale Isolation:** Automatically separates B2B inter-store transfers (`SB/...`) from retail sales (`S0/...`).
- **Behavioral & Clinical Chronic Classifier:** Filters acute non-chronic purchases using longitudinal purchase frequency and cross-patient consensus.
- **3-Head Quantile Uncertainty Envelopes ($P_{10}, P_{50}, P_{90}$):** Predicts median point estimates ($P_{50}$) alongside early refill risk ($P_{10}$) and critical lapse bounds ($P_{90}$) with 80.7% empirical test coverage.
- **Med-Sync (Multi-Prescription Synchronization Engine):** Clusters multiple active chronic prescriptions due within a $\le 7$-day window into a single consolidated WhatsApp reminder.
- **Multi-Stage Reminder Lifecycle:** Schedules timely touches (`-7d`, `-3d`, `-1d`, `0d`, `+2d`, `+5d`) and automatically invalidates remaining stages upon repurchase (`SUPERSEDED_BY_PURCHASE`).
- **Pharmacist Review Queue & Delivery Exports:** Routes ambiguous or missing-phone records to a pharmacist queue, and exports delivery-ready 10-column CSVs daily.

---

## 2. System Architecture & Workflow

```text
  [ Raw POS Sales File (CSV / Excel) ]
                    │
                    ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. INGESTION, CLEANSING & B2B ISOLATION GATE                │
  │    (refillcare/data/monthly_ingestion.py & dates.py)        │
  │    • Strict DD-MM-YYYY Date Normalization & Ambiguity Checks│
  │    • Wholesale / Inter-Store (SB/...) Exclusion             │
  │    • Mobile Number E.164 Sanitation & Audit                 │
  │    • 1-Click Batch Rollback (Undo Upload)                   │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 2. CLINICAL AI & 3-HEAD QUANTILE REGRESSION ENGINE          │
  │    (refillcare/engine/unified_engine.py & models/)          │
  │    • Path A (>= 6 purchases): Median-loss Quantile XGBoost  │
  │      (P10 Lower Bound, P50 Median, P90 Upper Bound)         │
  │    • Path B (< 6 purchases): Pack Days-of-Supply (DOS)      │
  │      + 3-Month / 6-Month Recurrence Verification            │
  │    • Stability Tiering: HIGH, MEDIUM-SAFE, MEDIUM-RISK,     │
  │      and UNSTABLE                                           │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 3. MED-SYNC MULTI-PRESCRIPTION BUNDLING                     │
  │    (refillcare/engine/med_sync.py)                          │
  │    • Groups patient chronic meds due within <= 7 days       │
  │    • Designates primary high-stability anchor date          │
  │    • Consolidates multi-item WhatsApp reminder copy         │
  │    • Reduces patient notification fatigue by 50%+           │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 4. ENTERPRISE PERSISTENCE & DATABASE LAYER                  │
  │    (refillcare/engine/persistence.py & database/)           │
  │    • SQLite Enterprise DB: enterprise.db                    │
  │    • RefillDecisionModel (Audited Rationale & Bounds)       │
  │    • ReminderCycleModel & ReminderStageModel Lifecycle      │
  │    • Idempotent (cycle_id, stage_offset) dispatch tracking  │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 5. OPERATIONAL DASHBOARD & REST API                         │
  │    • Streamlit UI (app_refillcare.py): Pharmacist Review,   │
  │      Date-Filtered Lists, Ingestion Upload, CSV Export      │
  │    • FastAPI Backend (api/main.py): REST Endpoints for      │
  │      Med-Sync bundles, Quantiles, Decisions, and Webhook    │
  └─────────────────────────────────────────────────────────────┘
```

---

## 3. Verified System Metrics & Performance

- **Automated Test Suite:** `751/751 tests passing` (`pytest tests/`).
- **Chronic Specialized XGBoost Model MAE:** **9.73 days** (vs 25.71 days Baseline, **-62.2% Error**).
- **High-Stability Chronic Patients (>10 Purchases):** **7.36 days MAE**, **84.95% accuracy within $\pm 14$ days**.
- **Quantile Uncertainty Envelope Coverage ($P_{10} \to P_{90}$):** **80.66%** of actual refills fall within predicted interval bounds.
- **B2B Wholesale Contamination:** `0.00%` (100% upstream isolation).

---

## 4. Repository Directory Structure

```text
ai-mediastra-whatsapp-reminder/
├── refillcare/             # Core Clinical Decision Engine & Domain Logic
│   ├── engine/             # Decision routing, Med-Sync bundler, persistence manager
│   │   ├── decision_types.py   # RefillDecision dataclass, constants, stages
│   │   ├── med_sync.py         # Multi-prescription bundling & sync engine
│   │   ├── persistence.py      # SQLAlchemy persistence & lifecycle manager
│   │   ├── reminder_lifecycle.py # WhatsApp template messages
│   │   └── unified_engine.py   # Unified decision engine & stability routing
│   ├── models/             # Stability classifier, hybrid ML models, prediction
│   ├── features/           # Consumption velocity, Days-of-Supply (DOS)
│   ├── data/               # Packing units parser, dates, monthly ingestion, classifier
│   │   └── medication_classifier.py # Behavioral & population chronic classifier
│   ├── evaluation/         # Holdout validation, backtesting, benchmark suites
│   └── config/             # Path thresholds, stability limits, system constants
│
├── database/               # Database Layer
│   ├── connection.py       # SQLAlchemy engine & SessionLocal (enterprise.db)
│   ├── models.py           # ORM schemas: RefillDecisionModel, ReminderCycleModel, etc.
│   └── fetch_sales.py      # Transaction queries
│
├── reminder/               # Reminder Delivery & Scheduling
│   ├── reminder_engine.py  # 10-column delivery CSV generation with phone sanitization
│   ├── send_message.py     # Single/batch WhatsApp dispatch with safety dry-run guard
│   ├── storage.py          # Fast SQLite snapshot store for review queues
│   ├── scheduler.py        # 6-stage schedule utilities (-7d, -3d, -1d, 0d, +2d, +5d)
│   └── message_logs.py     # Local audit logging of sent/simulated messages
│
├── services/               # External Integrations
│   ├── xinno_whatsapp.py   # Xinno CPaaS WhatsApp Business API client (text templates)
│   ├── xinno_image_template.py # WhatsApp image template API client
│   └── cloudinary_image.py # Cloudinary image hosting for rich media campaigns
│
├── utils/                  # Cross-Cutting Utilities
│   ├── validators.py       # Canonical Indian phone normalization (E.164 without '+')
│   ├── column_aliases.py   # Robust CSV column header mapping (15+ ERP formats)
│   ├── bulk_send.py        # Safe sequential batch dispatch with throttling
│   └── audit.py            # Phone masking & compliance audit logging
│
├── api/                    # Enterprise FastAPI Application
│   ├── main.py             # REST API entry point, CORS, OpenAPI docs, static routes
│   ├── schemas.py          # Pydantic response/request validation schemas
│   └── services.py         # Service layer (Med-Sync, Quantiles, KPIs, Ingestion)
│
├── frontend/               # Single Page Application Static Assets
│   ├── index.html          # Web dashboard interface
│   ├── style.css           # Modern styles
│   └── app.js              # Live API integration & table rendering
│
├── exports/                # Output Directory for Daily Operational CSVs
│   └── reminder_list_*.csv # Canonical 10-column CSVs for store delivery
│
├── data/                   # Data Directory
│   └── refillcare/processed/
│       ├── enterprise.db   # Production SQLite database
│       ├── refillcare.db   # Auxiliary snapshot database
│       ├── clean_transactions.parquet # Cleaned, B2B-purged transaction history
│       ├── purchase_history.parquet   # Canonical longitudinal purchase dataset
│       ├── train.parquet, validation.parquet, test.parquet # ML partitions
│       ├── chronic_model_benchmark_report.json # Benchmark & quantile report
│       └── models/
│           ├── chronic_refill_model.joblib # 3-Head Quantile XGBoost Regressor
│           ├── refill_model.joblib         # Production model mirror
│           └── human_ml_refill_model.joblib# Human consensus model
│
├── scripts/                # Production Engineering & Operational Scripts
│   ├── train_chronic_specialized_model.py # Trains 3-head quantile model (P10, P50, P90)
│   ├── train_human_ml_model.py           # Trains human consensus ML model
│   ├── run_offline_validation.py         # Evaluates test holdout metrics
│   ├── backtest_human_scenarios.py       # Backtests 5 behavioral archetypes
│   ├── backtest_consumption_dos_vs_hybrid.py # Evaluates DOS vs Hybrid cadence
│   ├── audit_channel_impact.py           # Audits B2B vs Retail transaction isolation
│   ├── verify_column_aliases.py          # Validates header normalization
│   ├── execute_single_live_test.py       # Dispatches single-recipient test message
│   ├── execute_bulk_live_test.py         # Batch WhatsApp dispatch runner
│   ├── verify_image_bulk_dry_run.py      # Dry-run test for promotional images
│   ├── verify_system_health.py           # System-wide diagnostics
│   └── build_refillcare_notebooks.py     # Generates reproducible Jupyter notebooks
│
├── notebooks/              # Interactive Analysis & Master Walkthrough
│   ├── RefillCare_End_to_End_Master_Walkthrough.ipynb # Master 5-phase interactive guide
│   └── refillcare/         # Modular Phase 1 to Phase 5 Notebooks
│
├── whatsapp_campaigns/     # Standalone WhatsApp Broadcast Campaign Apps
│   ├── app.py              # Text Campaign Streamlit App
│   ├── app_text_campaign.py# Text Campaign Implementation
│   ├── app_image_campaign.py # Image + Text Campaign Streamlit App
│   └── sample_data/        # Sample customer CSVs
│
├── tests/                  # Complete Automated Test Suite (751 passing tests)
│   ├── api/                # FastAPI endpoint integration tests
│   └── refillcare/         # Engine, Med-Sync, Classifier, Lifecycle tests
│
├── app_refillcare.py       # Main RefillCare Operations Dashboard (Streamlit)
├── run_prediction.py       # CLI Runner: Batch decision evaluation
├── run_reminder.py         # CLI Runner: Reminder schedule query & CSV export
├── run_message.py          # CLI Runner: WhatsApp dispatch simulation
├── run_train.py            # CLI Runner: Model retraining
├── requirements.txt        # Python package dependencies
├── pytest.ini              # Pytest configuration
└── README.md               # Master Documentation
```

---

## 5. Operational Commands & API Endpoints

### A. Daily CLI Workflows

```bash
# 1. Run Full Prediction & Persistence Pipeline
python run_prediction.py

# 2. Export Today's Due Refill Reminder List (10-column CSV)
python run_reminder.py

# 3. Simulate / Dispatch WhatsApp Messages (Default: DRY-RUN)
python run_message.py

# 4. Retrain 3-Head Chronic Quantile Regressors (P10, P50, P90)
python scripts/train_chronic_specialized_model.py
```

### B. User Interfaces & REST API

```bash
# 1. Start Main RefillCare Operations Dashboard
streamlit run app_refillcare.py

# 2. Start FastAPI Backend (Swagger Docs: http://127.0.0.1:8000/docs)
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

# 3. Access Med-Sync Bundles API
curl http://127.0.0.1:8000/api/v2/med-sync/bundles?sync_window_days=7

# 4. Access Quantile Uncertainty Bounds API
curl http://127.0.0.1:8000/api/v2/models/quantiles
```

### C. WhatsApp Broadcast Apps

```bash
# 1. Text Campaign App
streamlit run whatsapp_campaigns/app.py

# 2. Image Campaign App
streamlit run whatsapp_campaigns/app_image_campaign.py
```

---

## 6. Testing & Quality Assurance

```bash
# Run full test suite (751 tests)
pytest -q

# Run Med-Sync & API tests specifically
pytest tests/refillcare/test_med_sync.py tests/api/test_enterprise_api.py -v
```
