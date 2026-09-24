# Rule: Single Source of Truth for Operational UIs and Review Queues

1. **Database-First Queue Rendering:**
   - Any user-facing interface (Streamlit, Web frontend, or REST API endpoint) displaying "Today's Reminders", "Review Queues", or "Due Alerts" MUST query the authoritative persistence store (`RefillPersistenceManager.get_today_review_queue()` backed by `enterprise.db`).
   - Never generate real-time operational reminder views by running ad-hoc batch predictions on sample holdout slices (`test.parquet`).

2. **Accurate Date & Stage Provenance:**
   - Operational tables must reflect all 23 database attributes (Patient ID, Medication, Path A/B classification, Days of Supply, Stage Offset, Status, Prediction Rationale).
   - Explanatory notes and UI footers must reflect the up-to-date clinical rules (e.g. 6-stage lifecycle, multi-pack scaling, repurchase supersession) rather than legacy single-interval heuristics.
