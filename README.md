# RefillCare Enterprise — AI Refill Prediction & Med-Sync Platform

RefillCare Enterprise is an end-to-end clinical AI and patient engagement system built for retail pharmacy chains. It forecasts chronic medication refill dates, bundles multiple prescriptions into synchronized monthly appointments, and delivers personalized WhatsApp refill reminders.

---

## 1. What RefillCare Solves

- **No More Fixed 30-Day Guesses:** Patients take medicines at different daily frequencies (OD, BD, TID) and buy different pack sizes (10s, 30s, 60s). RefillCare computes personalized consumption intervals and Days-of-Supply (DOS).
- **No More Message Spam:** Instead of sending 3 separate WhatsApp messages on 3 different days for 3 medicines, Med-Sync groups them into **one unified appointment reminder**.
- **No More Obsolete Reminders:** If a patient buys their medicine early or on their due date, previous reminders are instantly marked `STATUS_SUPERSEDED`.
- **Clean Clinical Isolation:** Wholesale/inter-store transfers (`SB/...`) and acute products (syrups, drops, painkillers) are filtered out upstream so only valid chronic patients receive refill outreach.

---

## 2. Core Features

### 🔄 Dynamic Continuous-Observation Pipeline
- **Arbitrary Sales Uploads:** Ingest 1-day, 7-day, 13-day, or monthly POS sales files.
- **Dynamic As-Of Date ($T_{\text{eval}} = T_{\max} + 1\text{d}$):** Forecasting automatically begins the day after the latest transaction date.
- **Rolling 30-Day Outreach Horizon:** View all upcoming reminders for the active 30-day window ($T_{\text{eval}} \to T_{\text{eval}} + 29\text{d}$) or select a complete calendar month.
- **Safety Pre-Image Backups:** Automatically creates a backup snapshot before appending new records to `data/refillcare/customer_data_fields.csv`, with 1-click rollback support.

### 🧠 Dual-Path AI Engine & 5 Behavioral Archetypes
- **Path A ($\ge 6$ buys):** Evaluates longitudinal cadence using Median Absolute Deviation (MAD) stability filtering and 3-Head Quantile XGBoost Regressors ($P_{10}, P_{50}, P_{90}$).
- **Path B ($< 6$ buys):** Governed by verified Days-of-Supply (DOS) with daily consumption bounds.
- **5 Archetypes:** Multi-pack scaling (60/90 units), early carryover ($R_{inv}$), partial 10-strip clamping, post-lapse reset, and physical consensus bounding ($[0.65, 1.50] \times \text{DOS}$).

### 📦 Med-Sync: Multi-Prescription Synchronization
- Groups active chronic medications for each patient (matched by Name + Mobile) into a single delivery or pickup appointment.
- Configurable sync window (0–14 days, default 0d exact date or 8d range).
- 3-tier lifecycle isolation: **Due/Advance** (-7d to 0d), **Follow-up** (+2d, +5d), and **Lapsed Re-engagement** (+45d).

### 💬 WhatsApp & Xinno Messaging Gateway
- 6-variable dynamic template (`refillcare_medicine_reminder`) with custom pharmacy store name, contact number, and medicine list.
- Live high-contrast preview card, single test dispatches, and batch outreach with missing-mobile safety guards.

---

## 3. Project Structure

```text
ai-mediastra-whatsapp-reminder/
├── refillcare/             # Core Clinical AI Engine & Domain Logic
│   ├── engine/             # Decision routing, Med-Sync bundler, persistence manager
│   │   ├── persistence.py      # Database cycle sync & date-range review queues
│   │   ├── unified_engine.py   # Stability evaluation & dual-path decision engine
│   │   ├── med_sync.py         # Multi-prescription patient bundling
│   │   └── decision_types.py   # Data structures & lifecycle constants
│   ├── data/               # Ingestion, date parsing, chronic classifier
│   │   ├── sales_ingestion.py  # Master append with backup, deduplication, rollback
│   │   ├── medication_classifier.py # Supabase rules & chronic safety gate
│   │   └── dates.py            # Pharmacy date normalizer
│   └── models/             # 3-Head Quantile Regressors & feature extractors
│
├── database/               # SQLite Persistence (enterprise.db)
│   ├── models.py           # RefillDecisionModel, ReminderCycleModel, ReminderStageModel
│   └── connection.py       # SQLAlchemy SessionLocal
│
├── api/                    # FastAPI Enterprise Backend
│   ├── main.py             # REST API routes (/api/v1/reminders/daily, etc.)
│   ├── services.py         # Business logic layer with online inference sync
│   └── schemas.py          # Pydantic request/response schemas
│
├── frontend/               # Web Application Interface
│   ├── index.html          # Web SPA with Date, Rolling 30d, and Month views
│   └── js/app.js           # Frontend logic & API client
│
├── data/refillcare/        # Master Data & Storage
│   ├── customer_data_fields.csv   # Master POS transaction dataset (909k+ rows)
│   ├── backups/                   # Timestamped pre-image safety backups
│   └── processed/                 # Parquet tables & enterprise.db
│
├── app_refillcare.py       # Main Operations Dashboard (Streamlit)
├── scripts/                # Execution & Validation Scripts
│   └── run_sales_pipeline.py # End-to-end sales ingestion & decision pipeline
└── tests/                  # Automated Test Suite (pytest)
```

---

## 4. Quickstart: How to Run

### Step 1: Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Launch Streamlit Dashboard
```bash
streamlit run app_refillcare.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.
- **Tab 1: Dashboard Overview** — View live KPIs, retail vs B2B transaction breakdowns, and channel audit metrics.
- **Tab 2: Data Update** — Upload new weekly or monthly sales files; automatically appends to master CSV with backup and syncs decision cycles.
- **Tab 4: Reminder List** — Filter reminders by **Specific Date**, **Rolling 30 Days (Active Horizon)**, or **Complete Month**.
- **Tab 5: Med-Sync Bundles** — Inspect consolidated patient appointments and messages saved.
- **Tab 6: WhatsApp Gateway** — Preview templates and test or batch-dispatch messages.

### Step 3: Launch FastAPI Backend & Web UI
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
- Web Application: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Interactive API Documentation (Swagger): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Step 4: Run CLI Ingestion Pipeline
To ingest new sales from the command line and update decisions:
```bash
python scripts/run_sales_pipeline.py --new-sales path/to/sales.xlsx
```

---

## 5. Key REST API Endpoints

| Endpoint | Method | Purpose |
| :--- | :---: | :--- |
| `/api/v1/reminders/daily` | `GET` | Retrieve reminders by `target_date`, `target_month`, or `start_date` + `end_date` (rolling horizon). |
| `/api/v1/medsync/bundles` | `GET` | Get patient-grouped Med-Sync bundles and messages-saved statistics. |
| `/api/v1/dashboard/kpis` | `GET` | Live operations KPIs and monthly forecast distribution. |
| `/api/v1/whatsapp/candidates` | `GET` | Clustered WhatsApp outreach candidates ready for dispatch. |
| `/api/v1/sales/upload` | `POST` | Ingest new sales records with batch ID and online decision sync. |
| `/api/v1/sales/rollback/{batch_id}` | `POST` | Safely roll back an uploaded batch and restore master CSV. |

---

## 6. Running Tests

```bash
# Run unit tests for chronic medication classifier
pytest tests/refillcare/test_medication_classifier.py -v

# Run the dynamic continuous-observation integration test
python scratch/test_rolling_pipeline.py
```
*Current test suite status: 100% passing.*
