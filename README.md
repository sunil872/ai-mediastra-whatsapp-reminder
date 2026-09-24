# AI Mediastra / PHARMA HUBB — RefillCare & WhatsApp Reminder Platform

An enterprise-grade medication refill prediction, reminder lifecycle management, and WhatsApp dispatch platform for retail pharmacy operations.

> 📖 **Comprehensive Implementation & Roadmap Guide:** See [REFILLCARE_SYSTEM_END_TO_END_IMPLEMENTATION.md](file:///c:/Users/sunil/ai-mediastra-whatsapp-reminder/ai-mediastra-whatsapp-reminder/REFILLCARE_SYSTEM_END_TO_END_IMPLEMENTATION.md) for the complete end-to-end breakdown of Version 1.0 (Phases 1–17) and the feature roadmap for Version 2.0 and 3.0.

---

## 1. System Architecture & End-to-End Flow

```text
  [ Raw ERP Transactions / Parquet / CSV ]
                     │
                     ▼
  ┌─────────────────────────────────────────────────────────────┐
  │         1. REFILLCARE UNIFIED DECISION ENGINE               │
  │            (refillcare/engine/unified_engine.py)             │
  │                                                             │
  │   Trajectory Evaluation (Customer ID + Item ID)             │
  │   ├── Path A (>= 6 purchases):                              │
  │   │   ├── Phase 17J Stability Classification (HIGH/MED)     │
  │   │   ├── Quantity-Aware Scaling (Ratio = U_latest/U_typ)   │
  │   │   │   ├── Partial Purchase (<0.8): Scaled & Capped at U │
  │   │   │   └── Multi-Pack (>1.3): Scaled up to 180 days      │
  │   │   ├── Post-Lapse Reset (>1.5x cadence gap)              │
  │   │   └── Corroboration Guardrails (DOS vs Cadence)         │
  │   │                                                         │
  │   └── Path B (< 6 purchases):                               │
  │       ├── 3-Month Recurrence (>= 2 distinct calendar months)│
  │       ├── 6-Month Recurrence (>= 3 distinct calendar months)│
  │       ├── Pack DOS / Consumption Velocity Prediction        │
  │       └── 75-Day Churn Inactivity Gate                      │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │         2. ENTERPRISE PERSISTENCE & LIFECYCLE SYNC          │
  │            (refillcare/engine/persistence.py)               │
  │                                                             │
  │   SQLite Enterprise Database: data/.../enterprise.db        │
  │   ├── RefillDecisionModel: Historical records & rationale   │
  │   ├── ReminderCycleModel: Active/superseded cycle states    │
  │   └── ReminderStageModel: 6-Stage Schedules (-7,-3,-1,0,+2,+5)
  │       * Auto-Reset: Prior stages superseded upon repurchase │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │         3. DAILY EXECUTION, REVIEW & DISPATCH               │
  │                                                             │
  │   [ run_reminder.py ]  ──> Exports/reminder_list_YYYY-MM-DD.csv
  │   [ app_refillcare.py] ──> Pharmacist Review & Operations UI│
  │   [ run_message.py ]   ──> WhatsApp Gateway (DRY-RUN / Live) │
  │   [ services/xinno ]   ──> Meta Cloud / WABA Delivery       │
  └─────────────────────────────────────────────────────────────┘
```

---

## 2. Repository Directory Organization

The codebase is organized into modular layers to maintain high separation of concerns:

```text
ai-mediastra-whatsapp-reminder/
├── refillcare/             # Core Decision Engine & Domain Logic
│   ├── engine/             # Unified decision engine, persistence manager, cycle models
│   ├── models/             # Stability classifier, hybrid ML models, prediction pipelines
│   ├── features/           # Consumption velocity, Days-of-Supply (DOS), feature engineering
│   ├── data/               # Packing units parser, date parsing, monthly ingestion pipelines
│   ├── evaluation/         # Holdout validation, backtesting, benchmark suites
│   └── config/             # Path thresholds, stability limits, system constants
│
├── database/               # Database Layer
│   ├── connection.py       # SQLAlchemy engine & SessionLocal (SQLite enterprise.db)
│   ├── models.py           # ORM schemas: RefillDecisionModel, ReminderCycleModel, ReminderStageModel
│   └── fetch_sales.py      # Transaction extraction from clean parquet / SQL datasets
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
│   ├── column_aliases.py   # Robust CSV column header mapping (15+ ERP header formats)
│   ├── bulk_send.py        # Safe sequential batch dispatch with throttling
│   └── audit.py            # Phone masking & compliance audit logging
│
├── exports/                # Output Directory for Daily Operational CSVs
│   └── reminder_list_*.csv # Canonical 10-column CSVs for store delivery (gitignored)
│
├── api/ & app/             # REST Endpoints
│   ├── api/main.py         # FastAPI backend (health check, customer trajectory API)
│   └── app/main.py         # Reverse-proxy entry point for uvicorn deployments
│
├── notebooks/              # Documentation & Interactive Analysis
│   ├── RefillCare_End_to_End_Master_Walkthrough.ipynb # Master 5-phase interactive tutorial
│   └── refillcare/         # Modular Phase-by-Phase Notebooks
│       ├── 01_phase_1_data_analysis.ipynb
│       ├── 02_phase_2_data_pipeline.ipynb
│       ├── 03_phase_3_feature_engineering.ipynb
│       ├── 04_phase_4_model_training.ipynb
│       └── 05_phase_5_unified_engine_and_scheduling.ipynb
│
├── scripts/                # Offline Batch Scripts & Research Experiments
│   ├── phase17j_final_path_a_policy_validation.py # Formal policy validation runner
│   └── verify_image_bulk_dry_run.py               # Image campaign integration dry-run
│
├── tests/                  # Automated Test Suite (720+ tests, 100% passing)
│   ├── refillcare/         # Tests for unified engine, quantity scaling, and persistence
│   ├── test_phase*.py      # Phase 1 through Phase 17 milestone test suites
│   └── test_phone_normalization.py # Strict phone sanitization test cases
│
├── whatsapp_campaigns/     # [ISOLATED FOLDER] Standalone WhatsApp Broadcast Apps
│   ├── app.py              # Text Campaign Streamlit App (Default Runner)
│   ├── app_text_campaign.py # Text Campaign Implementation
│   ├── app_image_campaign.py # Image + Text Campaign Streamlit App
│   ├── sample_data/        # Sample customer CSVs for campaigns
│   └── README.md           # Standalone campaign documentation & guide
│
├── app_refillcare.py       # [ONLY UI IN ROOT] Main RefillCare Operations Dashboard
│
├── run_prediction.py       # CLI Runner: Batch decision evaluation & database persistence
├── run_reminder.py         # CLI Runner: Today's reminder schedule query & CSV export
├── run_message.py          # CLI Runner: WhatsApp dispatch simulation (dry-run safe)
├── run_train.py            # CLI Runner: Model retraining & versioning
│
├── requirements.txt        # Python package dependencies
├── pytest.ini              # Pytest configuration
└── README.md               # Master System Architecture & Operations Guide
```

---

## 3. Operational Workflows & Commands

### A. The 4 Daily CLI Pipelines

Run these commands from the repository root:

```bash
# 1. Run Full Evaluation & Database Persistence
# Reads purchase history, evaluates Path A/B logic, updates cycles, supersedes repurchased stages
python run_prediction.py

# 2. Export Today's Due Reminder Delivery List
# Queries active stages due for target date and generates exports/reminder_list_YYYY-MM-DD.csv
python run_reminder.py

# 3. Simulate / Dispatch WhatsApp Messages (Default: DRY-RUN)
# Reads today's queue and simulates template dispatch without hitting live network
python run_message.py

# 4. Retrain Machine Learning Models (Offline)
# Retrains LightGBM/heuristic hybrid models and saves versioned bundle
python run_train.py
```

### B. User Interfaces (Web Apps)

```bash
# 1. Main RefillCare Operations Dashboard (Pharmacist Review & Patient Search)
streamlit run app_refillcare.py

# 2. Standalone Text Campaign Uploader (Isolated Folder)
streamlit run whatsapp_campaigns/app.py

# 3. Standalone Image + Text Campaign Uploader (Isolated Folder)
streamlit run whatsapp_campaigns/app_image_campaign.py

# 4. FastAPI REST Backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 4. Clinical Decision Logic (Path A vs. Path B)

The prediction pipeline operates under strict clinical safety rules:

### Path A: Regular Patients ($\ge 6$ Lifetime Purchases)
* **Anchor:** Customer's personal historical cadence median.
* **Stability Tiers:** Classified via Phase 17J rules into `HIGH`, `MEDIUM-SAFE`, `MEDIUM-RISK`, or `UNSTABLE`.
* **Quantity-Aware Scaling:**
  * **Partial Purchase ($\text{Ratio} < 0.8$):** When a patient buys fewer units than usual (e.g. 1 strip of 10 instead of 20 tabs), interval scales down proportionally and is **hard-capped at the physical units purchased**.
  * **Multi-Pack Purchase ($\text{Ratio} > 1.3$):** When a patient buys bulk packs (e.g. 60 tabs instead of 30), interval scales up proportionally (up to 180 days) to prevent premature harassment.
* **Post-Lapse Gap Reset:** If a purchase occurs after a gap $> 1.5 \times \text{Cadence}$, the engine resets the cycle anchor and clamps to the immediate physical supply.
* **Corroboration Guardrail:** Divergent predictions between cadence and Days-of-Supply (DOS) are automatically routed to the **Pharmacist Review Queue**.

### Path B: Developing Patients ($< 6$ Lifetime Purchases)
* **Qualification:**
  * **3-Month Rule:** Purchases in $\ge 2$ distinct calendar months within the last 90 days.
  * **6-Month Rule:** Purchases in $\ge 3$ distinct calendar months within the last 180 days.
* **Authoritative Metric:** Calculated Days-of-Supply (DOS) based on pack unit size and active consumption velocity.
* **Lifecycle Gate:** Lapsed patients with expected refill $> 75$ days in the past are marked churned/dormant.

### The 6-Stage Reminder Lifecycle
Every active customer-item receives a standard 6-stage reminder sequence:
* **Day -7:** Early Refill Advance Notice
* **Day -3:** Primary Reminder Notice
* **Day -1:** Urgent Refill Alert
* **Day 0:** Exact Due Date Notification
* **Day +2:** Post-Due Follow-up Notice
* **Day +5:** Final Follow-up Alert
* **Repurchase Auto-Reset:** When a customer returns to the pharmacy and repurchases the item, all pending future stages of the previous cycle are automatically marked `SUPERSEDED_BY_PURCHASE`.

---

## 5. Setup & Environment Configuration

### 1. Requirements
* Python 3.10+
* Virtual environment (`.venv`)

### 2. Installation
```bash
# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables (`.env`)
Create a `.env` file based on `.env.example`:
```env
# Xinno CPaaS Gateway
XINNO_API_URL=https://whatsapp.xinno.in/REST/directApi/message
XINNO_API_KEY=your_xinno_api_key_here
XINNO_WABA_NUMBER=919515473474
WHATSAPP_TEMPLATE_LANGUAGE=en
WHATSAPP_TEMPLATE_NAME=reminder_refill_followup_v3

# Image Campaign Template
XINNO_IMAGE_TEMPLATE_NAME=refill_reminder_image
XINNO_IMAGE_URL=https://res.cloudinary.com/your_cloud/image/upload/your_template.png

# Cloudinary Integration
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret

# Store Branding
MEDICAL_STORE_NAME=PHARMA HUBB
```

---

## 6. Testing & Quality Assurance

The repository includes a comprehensive automated test suite covering all data transformations, stability classifications, quantity scaling, and WhatsApp mock dispatches:

```bash
# Run complete test suite (720+ tests)
pytest -q

# Run unified engine tests specifically
pytest tests/refillcare/test_v1_unified_engine.py -v

# Run phone normalization tests
pytest tests/test_phone_normalization.py -v
```
