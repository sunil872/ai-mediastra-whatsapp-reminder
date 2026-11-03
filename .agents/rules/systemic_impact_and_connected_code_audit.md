# Rule: Systemic Impact and Connected Code Audit

## Core Mandate: Zero Orphaned Changes & Full Multi-Layer Propagation
Whenever any data model, utility function, configuration, classifier, calculation, rule, or pipeline is updated, modified, or fixed in one file, the agent **MUST systematically audit and update ALL connected files, callers, APIs, and UI layers** across the entire codebase. Never leave a change localized to one file when downstream or peer files rely on it.

### Strict Workflow:
1. **Dependency & Caller Tracing (Mandatory Search):**
   - Before completing any task, execute a project-wide search (`grep_search`) for every modified function name, variable, or data schema.
   - Trace all consumers:
     - **Core Engines & Libraries:** (`refillcare/...`, `database/...`, `refillcare/data/medication_classifier.py`, `refillcare/engine/unified_engine.py`, etc.)
     - **APIs & Backend Services:** (`api/services.py`, `api/routes/...`)
     - **Interactive UIs & Dashboards:** (`app_refillcare.py`, Streamlit views, Web frontends)
     - **Background Jobs & Scripts:** (`scripts/...`, seeders, migration utilities)
     - **Automated Test Suites:** (`tests/...`)

2. **Dual-Key Medication & Salt Classification Standard:**
   - Any medication classification or gating logic must evaluate **both Item/Brand Name and Active Salt/Chemical Composition**.
   - If a caller does not pass `salt_name`, the classifier must utilize catalog lookup (e.g., from `SALT ITEMS LIST.csv`) as an automatic fallback.
   - Formulation safety rules (e.g., suppressing Syrups, Drops, Eye Drops, Ointments, Injections, Sachets) must strictly take precedence over chronic matching, preventing any non-solid items from leaking into refill reminder predictions.

3. **No Arbitrary Limits, Caching Bugs, or Disconnected Logic:**
   - If a backend service or data provider supports complete datasets (e.g., Supabase rules, transaction records, forecasts), ensure callers in UI/API do not introduce hardcoded, arbitrary caps (e.g., hardcoded `limit=200` or `limit=300`) without explicit user-controlled pagination.
   - Any export or download button in the UI must guarantee 100% data integrity and download the complete up-to-date dataset.
   - Invalidate or refresh in-memory caches whenever database records or rules are updated.

4. **Synchronous Propagation Across All Surfaces:**
   - If a parameter, default value, or return contract is enhanced (such as adding `salt_name` or `packing`), all calling locations must be updated to take advantage of the improvement.
   - If an operational feature is added to the backend, verify that both the REST API and the visual UI expose the updated capability accurately.

5. **Verify All Touchpoints with Tests:**
   - Run tests and sanity checks across all touched files to guarantee seamless end-to-end integration without broken imports or silent truncations.

