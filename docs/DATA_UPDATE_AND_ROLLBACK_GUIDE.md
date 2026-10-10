# RefillCare Data Pipeline: September Upload & August Rollback Guide

## 📌 Context & Decision Summary
- **Current Objective**: Meet immediate deadline to showcase October 2026 refill predictions generated from September 2026 sales data.
- **Chosen Approach**: Directly merged September 2026 sales (`14,094` transactions) into the canonical master CSV `data/refillcare/customer_data_fields.csv` (Total: `909,651` transactions from Dec 2020 to Sep 2026).
- **Post-Deadline Roadmap**: Fix and streamline the **Tab 2: Data Update** in-app monthly CSV upload engine in the backend so future monthly files can be uploaded directly via the Streamlit UI without requiring manual file merges.

---

## ↺ How to Rollback to August 2026 Baseline (Whenever Needed)

To revoke all September data and restore the system back to the clean August 31, 2026 baseline:

### Option 1: One-Click Script
Run the automated restore script:
```bash
python scripts/switch_to_august_baseline.py
```
This automatically:
1. Restores `customer_data_fields.csv` from the pre-September backup (`data/refillcare/backups/customer_data_fields_backup_pre_sep.csv`).
2. Re-runs the baseline pipeline to restore `purchase_history.parquet` to 815k records (up to Aug 31, 2026).
3. Restores `enterprise.db` back to the September reminder schedule baseline.

### Option 2: Git Restore
```bash
git checkout -- data/refillcare/customer_data_fields.csv data/refillcare/processed/
```

---

## 🔮 How to Switch Back to September Data (Predicting October 2026)

To re-apply September data and generate October 2026 predictions:
```bash
python scripts/switch_to_september_data.py
```

---

## 🛠️ Next Steps / Technical Debt for Data Update Tab
1. Normalize item code data types in `read_sales_file_low_memory` (`clean_item_identifier`) so incremental monthly uploads automatically join against multi-year item codes without float `.0` mismatches.
2. In `process_monthly_sales_data`, optimize in-memory DataFrame merges to avoid memory errors on low-memory environments.
3. Add a dedicated "Predict Target Month" dropdown in the UI to seamlessly toggle between reviewing September and October reminder pipelines.
