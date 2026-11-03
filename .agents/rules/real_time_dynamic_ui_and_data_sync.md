# Rule: Real-Time Dynamic UI and Full Data Synchronization

## Core Mandate: Real-Time Reactivity, Zero Stale Metrics & Instant Cache Invalidation
Whenever data rows, medication rules, transactions, or model parameters are added, edited, superseded, or deleted anywhere in the system (Supabase, SQLite `enterprise.db`, or parquet artifacts), the changes **MUST reflect immediately in real time across the UI, API, and Frontend without requiring manual restarts or page reloads**.

### Mandatory Implementation Requirements:

1. **Zero Hardcoded Metric Fallbacks:**
   - Never insert hardcoded fallback integers (e.g. `19233`, `15552`, `15547`, `592`, `631`) in UI scripts, API services, or frontend code.
   - All metrics, pipeline counts, and stage distributions must be dynamically computed in real time from the underlying database or live data source.

2. **Active & Eligible SQL Filter Invariants:**
   - Any query calculating scheduled reminders, stages, forecast pipelines, or operational queues must join and filter strictly by active eligible records:
     - `ReminderStageModel.status.in_(["PENDING", "DUE", "APPROVED"])`
     - `ReminderCycleModel.is_active == True`
     - `RefillDecisionModel.is_eligible == True`
   - Never run raw, unfiltered counts (`SELECT COUNT(*) FROM reminder_stages`) that include superseded, cancelled, or inactive historical records.

3. **Immediate Cache Clearing on Data Mutations:**
   - Any mutation action (adding a pharmacist rule, deleting a rule, ingesting sales, or resetting cycles) must immediately invalidate all caches:
     - In Streamlit: `st.cache_data.clear()` and `st.cache_resource.clear()`.
     - In Python memory: reset cache timestamp variables (`_CACHED_RULES_TIMESTAMP = 0.0`).
     - In Frontend Web App: call `loadInitialData()` or `fetchKPIs()` immediately after any add/delete action completes.

4. **Multi-Layer Parity Across All Surfaces:**
   - Any filtering, validation rule, or clinical safety gate applied in the engine (`refillcare/engine/persistence.py`) must simultaneously be implemented in:
     - Streamlit UI (`app_refillcare.py`)
     - FastAPI REST API (`api/services.py`, `api/main.py`)
     - Web Frontend (`frontend/js/app.js`)
     - CLI Scripts (`scripts/sync_enterprise_db_chronic.py`, `scripts/retrain_production.py`)

5. **Formulation Safety Gate Absolute Precedence:**
   - All liquid syrups, suspensions, drops, eye drops, ear drops, nasal sprays, topicals, soaps, and acute OTC formulations must be strictly suppressed from automated refill reminder modeling before chronic checking, with zero leakage.
