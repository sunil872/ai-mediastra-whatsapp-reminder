# RefillCare Dynamic Monthly Ingestion & Prediction Pipeline Guide

## 📌 Architecture & Master Dataset Overview
- **Master Dataset**: `data/refillcare/customer_data_fields.csv`
- **Current Data Horizon**: Multi-year transactions up to **September 30, 2026** (`909,651` records).
- **Default Prediction Target**: **October 2026** (and ongoing monthly horizons as new data is ingested).
- **Dynamic Date Inference**: The system automatically determines the latest sales date from the database or master dataset. Separate static month-switch scripts are deprecated and replaced by a unified production pipeline.

---

## 🚀 Running the Production Pipeline Dynamically

### 1. Evaluate Current Master Dataset (e.g. September Data → October 2026 Predictions)
To run the end-to-end data processing, feature engineering, and Unified Refill Decision Engine on the current master dataset:
```bash
python scripts/run_monthly_pipeline.py
```
This automatically:
1. Detects the maximum transaction date in `data/refillcare/customer_data_fields.csv` (`2026-09-30`).
2. Calculates the dynamic prediction evaluation date (`2026-10-01`) targeting **October 2026**.
3. Executes `run_refillcare_data_pipeline()` to generate updated Parquet artifacts.
4. Evaluates `UnifiedRefillDecisionEngine` across all patient-medication regimens.
5. Persists decisions and synchronizes multi-stage reminder cycles in `enterprise.db`.

---

### 2. Ingesting New Monthly Sales (e.g. October, November, etc.)
When a new monthly sales export (CSV or Excel) arrives from the pharmacy POS:
```bash
python scripts/run_monthly_pipeline.py --new-sales "path/to/new_month_sales.csv"
```
This automatically:
1. Validates incoming transactions and column mappings.
2. Creates an automated safety backup of `data/refillcare/customer_data_fields.csv` in `data/refillcare/backups/`.
3. Normalizes dates, item codes, and phone numbers.
4. Appends the validated transactions to `data/refillcare/customer_data_fields.csv`.
5. Re-runs the full pipeline and updates predictions for the next target horizon.

---

### 3. Pipeline Options & Overrides
| Flag | Description | Default |
|------|-------------|---------|
| `--master-csv` | Path to canonical transactions dataset | `data/refillcare/customer_data_fields.csv` |
| `--new-sales` | Path to incremental monthly sales file | `None` (processes current master) |
| `--as-of-date` | Evaluation cutoff date (`YYYY-MM-DD`) | 1st of the month following latest transaction |
| `--target-month` | Target reminder month (`YYYY-MM`) | Derived from `as-of-date` |
| `--dry-run` | Compute predictions without database write | `False` |

---

## 🛡️ Enterprise Data Security & Hygiene
- **Local Persistence Only**: Master CSV files, backups, SQLite databases, and Parquets are tracked in `.gitignore` and strictly prevented from committing to GitHub.
- **Deduplication**: Transactions are deduplicated chronologically per invoice and patient-medication pair.
- **Zero Loss Guarantee**: Incoming monthly files are appended—never overwriting historical transaction logs.
