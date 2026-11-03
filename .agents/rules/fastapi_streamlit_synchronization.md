# Rule: FastAPI and Streamlit Multi-Interface Synchronization

## Objective & System Invariants
Mediastra Refillcare operates two client presentation surfaces that consume the same enterprise core:
1. **FastAPI (`api/main.py`, `api/services.py`, `frontend/`)**: Production REST API for real-time applications, hospital EMR integrations, and standalone web UI.
2. **Streamlit (`app_refillcare.py`)**: Pharmacist clinical dashboard for desktop operations and client review demonstrations.

To ensure consistency across clinical evaluations and production deployments, both surfaces **MUST adhere to identical business logic, metric calculations, and filtering safety without creating runtime dependencies on each other**.

---

### Core Principles

#### 1. Architectural Decoupling (Zero Inter-Process Dependency)
- **FastAPI must NEVER import Streamlit** or rely on Streamlit's process/session state.
- **Streamlit must NEVER depend on FastAPI** running on `localhost:8000` to calculate metrics or render views.
- Both surfaces must import and consume the common domain layers:
  - Database models (`database/models.py`)
  - Engine & persistence (`refillcare/engine/persistence.py`)
  - Classifier & Supabase rules (`refillcare/data/medication_classifier.py`)
  - Storage & predictions (`reminder/storage.py`)

#### 2. Operations KPI & Metric Exact Parity
All core KPIs must yield identical dynamic values across FastAPI endpoints (`/api/v1/analytics/kpi`) and Streamlit KPI cards:
- **Monitored Patients**: Exact count of active unique customer IDs (e.g., `14,071`).
- **Scheduled Refills Pipeline**: Exact active upcoming reminder pipeline for the target forecast month (e.g., `11,212` for October 2026).
- **Refills Due Today**: Immediate eligible candidates scheduled for today's date (e.g., `435`).
- **Chronic Therapy Portfolio**: Dynamically derived from active Supabase medication rules (e.g., `1,920+`), never hardcoded.

#### 3. Clinical Review Queue 1:1 Synchronicity
Any customer or prediction requiring pharmacist verification must follow identical classification in both interfaces:
- **Total In Review Queue**: Total candidates flagged (e.g., `2,015`).
- **Missing Mobile Numbers**: Candidates blocked due to invalid/missing contact numbers (e.g., `46`).
- **Stability / Variance Risk**: Candidates with high cadence variance or irregular interval history (e.g., `915`).
- **Cold-Start (Single Purchase)**: Patients with only 1 historic purchase transaction (e.g., `427`).
- Both Streamlit and FastAPI UI must provide monthly target selection (`2026-10`, `2026-09`), category filtering, and individual record approval/rejection.

#### 4. Regex & String Safety Invariant
- Reminder stages include plus/minus signs (e.g., `+45 days (Re-engagement)`, `-7 days (Early Heads-up)`).
- **Never compile raw stage strings as regular expressions**. Always pass `regex=False` in pandas (`.str.contains(pat, regex=False)`) and standard literal `.includes()` in JavaScript.
- Avoid regex quantifiers (`+`, `*`, `?`) at position 0 to prevent `re.error: nothing to repeat at position 0`.

#### 5. Dynamic Month & Date Defaults
- Outreach target dates must default dynamically to the current run date (`datetime.now().date()` / `today()`).
- Forecast target pipelines must dynamically adapt to the latest available sales transaction month (e.g. October 2026 following September 2026 sales cutoff).
