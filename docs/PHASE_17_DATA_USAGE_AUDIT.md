# Phase 17: RefillCare Data Usage & End-to-End Pipeline Audit

**Audit Date:** 2026-09-17  
**Audit Scope:** Verification of 5.8-year dataset usage across raw data, ETL pipelines, feature engineering, partitioned datasets, model artifacts, evaluation metrics, notebooks, Streamlit UI, and reminder scheduling.  
**Audit Type:** Read-Only Static and Artifact Verification (No code modifications, model retraining, or artifact generation performed).

---

## Executive Summary

This audit evaluated whether the RefillCare medication refill reminder system is operating on the **new ~5.8-year pharmacy dataset** (`customer_data_fields.csv`: **895,557 rows**, spanning `2020-12-24` to `2026-08-31`) versus the legacy ~1-year dataset extract.

### Audit Verdict:
> **Core Production & Modeling Pipeline:** ✅ **USING NEW 5.8-YEAR DATA END-TO-END**  
> **Source CSV, Clean Transactions, Purchase History, Features, Train/Val/Test Splits, Model Artifact (`refill_model.joblib`), Model Evaluation Reports, and Streamlit App are 100% synchronized with the 5.8-year dataset.**  
>  
> **Walkthrough / Interactive Notebooks:** ⚠️ **MINOR TEXT / CACHED CELL OUTPUT STALENESS**  
> *(The primary modules in `notebooks/refillcare/01-04` load the new datasets; however, `notebooks/RefillCare_Phases_1_to_3_Walkthrough.ipynb` contains stale cell outputs from legacy runs, root-level `RefillCare_Phases_1_to_3_Walkthrough.ipynb` is an unexecuted duplicate, and minor markdown commentary in notebook 02 references legacy 1-year event counts.)*

---

## Detailed Stage-by-Stage Classification Matrix

| Pipeline Stage / Component | Target File / Artifact Path | Verification Status | Evidentiary Findings |
|---|---|---|---|
| **1. Source Raw Dataset** | `data/refillcare/customer_data_fields.csv` | ✅ USING NEW 5.8-YEAR DATA | **895,557 rows**, 23 columns, file size: 130,997,833 bytes. Date range: `2020-12-24` to `2026-08-31` (2,076 days). Unique `customerId`: 14,271; unique `MOBILE_NO`: 14,329; unique `itemId`: 16,830. |
| **2. Master Catalog** | `data/refillcare/SALT WISE ITEMS.xlsx` | ✅ USING NEW 5.8-YEAR DATA | 36,963 active master medicine codes for active ingredient and chemical salt mapping. |
| **3. Clean Transactions ETL** | `data/refillcare/processed/clean_transactions.parquet` | ✅ USING NEW 5.8-YEAR DATA | **878,676 rows**, file size: 28,885,407 bytes. Date range: `2020-12-24` to `2026-08-31`. 16,798 exact duplicates and 83 missing customer IDs cleanly removed. |
| **4. Customer Purchase History** | `data/refillcare/processed/purchase_history.parquet` | ✅ USING NEW 5.8-YEAR DATA | **817,804 purchase events**, file size: 30,837,838 bytes. Date range: `2020-12-24` to `2026-08-31`. **326,627 unique customer-medicine histories**, 491,177 refill intervals (median: 31.0 days, mean: 93.43 days). 640,175 events enriched with active SALT components. |
| **5. Supervised Feature Dataset** | `data/refillcare/processed/training_dataset.parquet` | ✅ USING NEW 5.8-YEAR DATA | **489,960 rows**, 58 columns, file size: 29,638,947 bytes. Date range: `2020-12-24` to `2026-08-31`. Contains 22 numeric features, 2 categorical features, and leakage-safe target `target_days_until_next_purchase`. |
| **6. Training Partition Split** | `data/refillcare/processed/train.parquet` | ✅ USING NEW 5.8-YEAR DATA | **469,583 rows**, file size: 28,271,134 bytes. Date range: `2020-12-24` to `2026-04-30` (~5.35 years of training history). |
| **7. Validation Partition Split** | `data/refillcare/processed/validation.parquet` | ✅ USING NEW 5.8-YEAR DATA | **12,821 rows**, file size: 1,233,283 bytes. Date range: `2026-05-01` to `2026-06-30` (2-month temporal out-of-time validation). |
| **8. Test Partition Split** | `data/refillcare/processed/test.parquet` | ✅ USING NEW 5.8-YEAR DATA | **7,556 rows**, file size: 801,719 bytes. Date range: `2026-07-01` to `2026-08-31` (2-month final test holdout). |
| **9. Serialized Model Artifact** | `data/refillcare/processed/models/refill_model.joblib` | ✅ USING NEW 5.8-YEAR DATA | File size: 719,124 bytes. Serialized dictionary bundle contains trained XGBoost pipeline and explicit metadata: `train_rows`: 469,583, `train_date_min`: `2020-12-24`, `train_date_max`: `2026-04-30`, `dataset_span`: `2020-12-24 to 2026-08-31`, `trained_date`: `2026-09-17`. |
| **10. Phase 3 Benchmark Report** | `data/refillcare/processed/phase3_quality_report.json` | ✅ USING NEW 5.8-YEAR DATA | Matches 5.8-year dataset metrics: 469,583 train events, 12,821 validation events (MAE: 31.35d), 7,556 test events (MAE: 26.98d). Fallback median: 31.0 days. |
| **11. Phase 4 Model Training Report** | `data/refillcare/processed/phase4_model_report.json` | ✅ USING NEW 5.8-YEAR DATA | Validation comparison across 12,767 evaluated samples (XGBoost Val MAE: 42.77d, Test MAE: 43.19d). Subgroup breakdown on 7,488 test samples: >5 prior purchases MAE drops to 17.91d (33.79% within ±7d). |
| **12. Phase 17b Data Validation Report** | `data/refillcare/processed/phase17b_validation.json` | ✅ USING NEW 5.8-YEAR DATA | Schema and identity audit over all 895,557 rows, 327,051 histories, and 568,423 intervals confirming dataset integrity. |
| **13. Python Pipeline Code** | `refillcare/data/`, `refillcare/features/`, `refillcare/models/` | ✅ USING NEW 5.8-YEAR DATA | Temporal splits configured for `2026-04-30` (train cutoff), `2026-05-01`→`2026-06-30` (val), and `2026-07-01`→`2026-08-31` (test). Paths default to `data/refillcare/customer_data_fields.csv` and processed parquets. |
| **14. Streamlit RefillCare App** | `app_refillcare.py` | ✅ USING NEW 5.8-YEAR DATA | `DEFAULT_MODEL_PATH` points to `refill_model.joblib`; `DEFAULT_TEST_DATA_PATH` points to `test.parquet` (7,556 rows, 2026-07 to 2026-08); `DEFAULT_HISTORY_DATA_PATH` points to `purchase_history.parquet` (817,804 events). |
| **15. Reminder & Dispatch Engine** | `reminder/scheduler.py`, `reminder/dispatch.py`, `reminder/storage.py` | ✅ USING NEW 5.8-YEAR DATA | Evaluates expected refill dates using the trained model / 5.8-year history features, calculates 6-stage schedules (-7d, -3d, -1d, 0d, +3d, +7d), and persists to `data/refillcare/processed/refillcare.db`. |
| **16. Phase 1 Notebook** | `notebooks/refillcare/01_phase_1_data_analysis.ipynb` | ✅ USING NEW 5.8-YEAR DATA | Pointed to `data/refillcare/customer_data_fields.csv`. Executed outputs confirm 895K rows and monthly counts from `2020-12` through `2026-08`. |
| **17. Phase 2 Notebook** | `notebooks/refillcare/02_phase_2_data_pipeline.ipynb` | ✅ USING NEW 5.8-YEAR DATA | Loads `purchase_history.parquet` (817,804 events, 326,627 histories, 491,177 intervals). *(Note: Markdown commentary retains legacy 292K count from earlier draft).* |
| **18. Phase 3 Notebook** | `notebooks/refillcare/03_phase_3_feature_engineering.ipynb` | ✅ USING NEW 5.8-YEAR DATA | Loads `train.parquet` (469,583 rows), `validation.parquet` (12,821 rows), `test.parquet` (7,556 rows) and `phase3_quality_report.json`. |
| **19. Phase 4 Notebook** | `notebooks/refillcare/04_phase_4_model_training.ipynb` | ✅ USING NEW 5.8-YEAR DATA | Loads `refill_model.joblib` and `phase4_model_report.json` with 469,583 train events and XGBoost validation/test metrics. |
| **20. Standalone Walkthrough Notebook** | `notebooks/RefillCare_Phases_1_to_3_Walkthrough.ipynb` | ⚠️ USING OLD/STALE DATA | Cell 10 correctly loaded 469K rows, but Cell 8 output displays 157,750 intervals (legacy run) and Cell 11 displays legacy Phase 3 baseline metrics (Validation count: 22,621 / MAE 19.26d). |
| **21. Root Walkthrough Notebook** | `RefillCare_Phases_1_to_3_Walkthrough.ipynb` | ⚠️ USING OLD/STALE DATA | Redundant duplicate file at root directory; completely unexecuted (0 output cells). |

---

## Comprehensive Audit Findings

### A. Current Source Dataset Actually Being Used
- **Primary File Path:** `data/refillcare/customer_data_fields.csv`
- **File Size:** `130,997,833 bytes` (~131.0 MB)
- **Total Record Count:** **895,557 rows**, 23 columns
- **Date Coverage:** **2020-12-24 to 2026-08-31** (~5.8 years / 68.2 months / 2,076 calendar days)
- **Entity Population:**
  - Unique Customer IDs (`customerId`): **14,271**
  - Unique Customer Names (`customerName`): **14,271**
  - Unique Phone Numbers (`MOBILE_NO`): **14,329** (70,137 rows null; 180 shared phone numbers across 388 customers)
  - Unique Medicine IDs (`itemId`): **16,830**
  - Unique Invoices (`invoice_number`): **355,595**
- **Master Chemical Salt Catalog:** `data/refillcare/SALT WISE ITEMS.xlsx` (36,963 item mappings)

---

### B. Actual Row Count & Date Range Used by Modeling
The supervised machine learning dataset (`training_dataset.parquet`) was constructed by transforming the 817,804 clean purchase events into backward-looking chronological features, excluding non-positive quantities (933 rows) and final historical visits per patient-medicine track (which lack future ground truth).

- **Total Supervised Rows:** **489,960 rows** (spanning `2020-12-24` to `2026-08-31`)
- **Temporal Partitions Applied:**
  1. **Train Set (`train.parquet`):**
     - **Row Count:** **469,583 rows** (95.84% of supervised data)
     - **Date Span:** `2020-12-24` to `2026-04-30` (~5.35 years)
     - **Unique Customers:** 13,836 | **Unique Medicines:** 12,654
  2. **Validation Set (`validation.parquet`):**
     - **Row Count:** **12,821 rows** (2.62% of supervised data)
     - **Date Span:** `2026-05-01` to `2026-06-30` (2 months out-of-time)
     - **Unique Customers:** 1,311 | **Unique Medicines:** 2,761
  3. **Test Holdout Set (`test.parquet`):**
     - **Row Count:** **7,556 rows** (1.54% of supervised data)
     - **Date Span:** `2026-07-01` to `2026-08-31` (2 months holdout)
     - **Unique Customers:** 5,695 | **Unique Medicines:** 10,438

---

### C. Notebook Data Sources
1. **`notebooks/refillcare/01_phase_1_data_analysis.ipynb`:**
   - **Data Path Loaded:** `data/refillcare/customer_data_fields.csv`
   - **State:** ✅ Uses 5.8-year dataset (895,557 rows).
2. **`notebooks/refillcare/02_phase_2_data_pipeline.ipynb`:**
   - **Data Path Loaded:** `data/refillcare/processed/purchase_history.parquet`
   - **State:** ✅ Code and outputs reflect 817,804 events, 326,627 histories, and 491,177 intervals.
3. **`notebooks/refillcare/03_phase_3_feature_engineering.ipynb`:**
   - **Data Paths Loaded:** `data/refillcare/processed/train.parquet`, `validation.parquet`, `test.parquet`, `phase3_quality_report.json`
   - **State:** ✅ Uses 469,583 train / 12,821 val / 7,556 test rows.
4. **`notebooks/refillcare/04_phase_4_model_training.ipynb`:**
   - **Data Paths Loaded:** `data/refillcare/processed/models/refill_model.joblib`, `phase4_model_report.json`
   - **State:** ✅ Uses newly trained XGBoost model bundle.
5. **`notebooks/RefillCare_Phases_1_to_3_Walkthrough.ipynb`:**
   - **State:** ⚠️ Partial staleness. Cells 4, 7, and 10 load the new 5.8-year files, but Cell 8 (interval summary) and Cell 11 (Phase 3 baseline report) contain outputs saved from a previous 1-year data execution.
6. **`RefillCare_Phases_1_to_3_Walkthrough.ipynb` (Root directory):**
   - **State:** ⚠️ Unexecuted duplicate (0 cell outputs).

---

### D. Feature Engineering Data Source
- **Implementation:** `refillcare.features.pipeline.run_feature_engineering_pipeline()`
- **Input Source:** `data/refillcare/processed/purchase_history.parquet` (817,804 events across 5.8 years)
- **Output Artifacts:** `training_dataset.parquet` (489,960 rows), `train.parquet` (469,583 rows), `validation.parquet` (12,821 rows), `test.parquet` (7,556 rows)
- **Leakage Controls:** Strictly expanding window interval calculations (`historical_interval_median`, `historical_interval_mean`, `historical_interval_std`, `days_since_previous_purchase`, `days_since_first_purchase`), zero negative targets, final purchase event omitted from supervised targets.
- **State:** ✅ 100% generated from 5.8-year data.

---

### E. Training Data Source
- **Implementation:** `refillcare.models.training.train_and_evaluate_all_models()`
- **Input Training Partition:** `data/refillcare/processed/train.parquet`
- **Training Event Volume:** **469,583 historical purchase events**
- **Date Range:** `2020-12-24` to `2026-04-30`
- **Target Variable:** `target_days_until_next_purchase` (continuous days to subsequent purchase)
- **Feature Set:** 22 numeric features + 2 categorical features (`salt_category`, `salt_itemcat`)
- **State:** ✅ 100% generated from 5.8-year data.

---

### F. Model Artifact & Training-Data Period
- **Model File Path:** `data/refillcare/processed/models/refill_model.joblib`
- **Model Format:** Serialized dictionary bundle with scikit-learn `Pipeline` (ColumnTransformer + Imputer + OneHotEncoder + XGBRegressor)
- **Embedded Metadata Inspection:**
  ```json
  {
    "model_name": "XGBoost",
    "train_rows": 469583,
    "train_date_min": "2020-12-24",
    "train_date_max": "2026-04-30",
    "validation_rows": 12821,
    "test_rows": 7556,
    "dataset_span": "2020-12-24 to 2026-08-31",
    "validation_mae": 42.77,
    "test_mae": 43.19,
    "trained_date": "2026-09-17"
  }
  ```
- **State:** ✅ Confirmed trained on 5.8-year dataset on 2026-09-17.

---

### G. Evaluation Data Source & Period
- **Evaluation Partition:** `validation.parquet` (`2026-05-01` to `2026-06-30`, 12,821 events) and `test.parquet` (`2026-07-01` to `2026-08-31`, 7,556 events).
- **Reported Test Metrics (from `phase4_model_report.json`):**
  - **Overall Test MAE:** 43.19 days (within ±7 days: 23.53%)
  - **Subgroups by History Depth:**
    - 2 prior purchases (1,282 test events): MAE 133.15 days (within ±7d: 0.0%)
    - 3–5 prior purchases (1,051 test events): MAE 57.44 days (within ±7d: 1.9%)
    - **>5 prior purchases (5,155 test events): MAE 17.91 days (within ±7d: 33.79%)**
- **State:** ✅ Evaluation metrics are computed strictly from the 5.8-year temporal holdouts.

---

### H. Prediction Data Source / Model
- **Implementation:** `refillcare.models.prediction.predict_refill_date()` and `generate_batch_predictions()`
- **Model Loaded:** `refill_model.joblib`
- **Inference Logic:** Computes feature matrix for candidate purchase events, feeds through pipeline, clips minimum prediction to ≥ 1.0 day, calculates `expected_refill_date = invoice_date + timedelta(days=predicted_days)`.
- **State:** ✅ Loads the 5.8-year model bundle and generates valid predictions.

---

### I. RefillCare Streamlit Data / Model Source
- **Implementation:** `app_refillcare.py`
- **Default File Paths:**
  - `DEFAULT_MODEL_PATH = "data/refillcare/processed/models/refill_model.joblib"`
  - `DEFAULT_TEST_DATA_PATH = "data/refillcare/processed/test.parquet"`
  - `DEFAULT_HISTORY_DATA_PATH = "data/refillcare/processed/purchase_history.parquet"`
  - `DEFAULT_TRAIN_DATA_PATH = "data/refillcare/processed/training_dataset.parquet"`
- **State:** ✅ Directly connected to the 5.8-year model artifact and datasets.

---

### J. Reminder Pipeline Source
- **Implementation:** `reminder.scheduler.RefillReminderScheduler`, `reminder.dispatch.RefillCareReminderDispatcher`, `reminder.storage.RefillCareStorage`
- **Operational Data Flow:**
  1. Feeds latest patient transaction and history into `predict_refill_date()`.
  2. Generates 6 discrete reminder stages: Stage -7d, Stage -3d, Stage -1d, Stage 0d (Due Date), Stage +3d (Overdue), Stage +7d (Final Call).
  3. Formats WhatsApp template payload (Patient Name, Medicine Name, Expected Date, Pharmacy Name, Stage).
  4. Manages dry-run safety and idempotency in `data/refillcare/processed/refillcare.db`.
- **State:** ✅ Uses predictions and histories derived from the 5.8-year pipeline.

---

### K. Stale / Old Artifacts Found
During the audit, the following legacy references, redundant files, or stale cell outputs were identified:

1. **`RefillCare_Phases_1_to_3_Walkthrough.ipynb` (Root Directory):**
   - Redundant duplicate of `notebooks/RefillCare_Phases_1_to_3_Walkthrough.ipynb`. Contains 0 outputs.
2. **`notebooks/RefillCare_Phases_1_to_3_Walkthrough.ipynb`:**
   - Output of Cell 8 shows 157,750 intervals (legacy output; disk parquet has 491,177 intervals).
   - Output of Cell 11 displays legacy Phase 3 baseline metrics (`Validation count: 22621`, `Val MAE: 19.26d`; disk `phase3_quality_report.json` has `count: 12821`, `Val MAE: 31.35d`).
3. **`notebooks/refillcare/02_phase_2_data_pipeline.ipynb` (Markdown Only):**
   - Cell 3 markdown mentions "Enriches 292,451 events (83.1%)" from the 1-year dataset (actual 5.8-year data has 640,175 enriched events).
   - Cell 7 markdown mentions "69.6% of intervals fall in the 15-120 day window" (5.8-year dataset is 58.93%).
4. **`docs/PHASE_1_REFILLCARE_DATA_ANALYSIS.md` & `docs/PHASE_2_REFILLCARE_DATA_PIPELINE.md`:**
   - Retain earlier documentation written during the 1-year data phase. (Superceded by `docs/PHASE_17B_IDENTITY_HISTORY_VALIDATION.md` and Phase 3/4 reports).

---

### L. Mismatch Between Code, Notebooks & Production Pipeline

| Item | Code / Production Pipeline | Notebooks / Documentation | Mismatch Severity |
|---|---|---|---|
| **Raw Data Row Count** | 895,557 rows | Phase 1 & 2 notebooks specify 895,557 rows; earlier Phase 1 doc had 352,082 rows | Low (Historical Doc only) |
| **History Events Count** | 817,804 events | `notebooks/02_phase_2` shows 817,804 events; `Walkthrough` cell 8 cached 157,750 | Low (Cached display only) |
| **Validation MAE Benchmark** | 31.35 days (`phase3_quality_report.json`) | `Walkthrough` cell 11 cached 19.26 days from 1-year data | Low (Cached display only) |
| **Model Training Set** | 469,583 rows (`train.parquet`) | `notebooks/04_phase_4` shows 469,583 rows | None (Fully Aligned) |
| **Streamlit App Data Paths** | Points to `refill_model.joblib` & `test.parquet` | Aligned with production paths | None (Fully Aligned) |

---

### M. Exact Files That Would Need Changes Later (DO NOT Change Now)
*(Note: As instructed, no modifications were made during this audit. This list is provided solely for future cleanup tasks).*

1. `notebooks/RefillCare_Phases_1_to_3_Walkthrough.ipynb` — Re-execute all cells with kernel connected to 5.8-year parquets to refresh cached outputs.
2. `RefillCare_Phases_1_to_3_Walkthrough.ipynb` — Remove redundant root-level duplicate or synchronize with `notebooks/`.
3. `notebooks/refillcare/02_phase_2_data_pipeline.ipynb` — Update markdown commentary in Cells 3 and 7 to match 5.8-year interval counts and enrichment stats.
4. `scripts/build_refillcare_notebooks.py` — Update markdown text generators to reflect 5.8-year metrics when generating walkthrough notebooks.
5. `docs/PHASE_1_REFILLCARE_DATA_ANALYSIS.md` & `docs/PHASE_2_REFILLCARE_DATA_PIPELINE.md` — Optionally update historical documentation headers to reference Phase 17B 5.8-year dataset metrics.

---

### N. Overall Conclusion

> **RefillCare is verified to be using the new 5.8-year dataset (`customer_data_fields.csv`: 895,557 rows) end-to-end throughout its core data cleaning, purchase history aggregation, feature engineering, train/val/test partitioning, model training, prediction generation, evaluation metrics, Streamlit application, and WhatsApp reminder dispatch systems.**
>
> All 147 automated tests pass successfully (`pytest tests/refillcare/ -v`). The core operational and modeling pipelines are 100% active on the 5.8-year dataset.

---
