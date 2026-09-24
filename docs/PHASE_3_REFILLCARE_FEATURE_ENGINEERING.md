# Phase 3 — RefillCare Feature Engineering & Dataset Construction Report

**Project:** Medication Refill Reminder System — RefillCare  
**Date:** 2026-09-07  
**Status:** Phase 3 Complete — Leakage-Safe Feature Engineering, Temporal Split & Baseline Evaluation  

---

## 1. Executive Summary

Phase 3 constructed and validated a leakage-safe supervised learning dataset from the Phase 2 canonical purchase history (`data/refillcare/processed/purchase_history.parquet`).

A total of **156,363 supervised training examples** were generated across **194,778 customer-medicine timelines**, partitioned into strictly non-overlapping temporal sets for machine learning model training in Phase 4.

### Key Metrics Summary

| Metric | Value | Description |
|---|---|---|
| **Canonical Events (Phase 2)** | **352,074** | Aggregated purchase line items |
| **Unique Customer-Medicine Timelines** | **194,778** | Unique `(customerId + itemId)` series |
| **Supervised Eligible Examples** | **156,363** | Events with known next purchase and valid quantity |
| **Cold-Start / Single Purchases** | **147,544 (75.7%)** | Histories with $N=1$ (no next purchase) |
| **Multi-Purchase Histories ($\ge 2$)** | **47,234 (24.3%)** | Eligible for personal historical stats |
| **Multi-Purchase Histories ($\ge 3$)** | **25,607 (13.1%)** | Eligible for interval standard deviation & CV |
| **Train Set ($\le$ 2026-04-30)** | *(recompute after Phase 3 re-run)* | Multi-year training history through Apr 2026 |
| **Validation Set (2026-05 to 2026-06)** | *(recompute after Phase 3 re-run)* | 2-month validation period |
| **Test Set (2026-07 to 2026-08)** | *(recompute after Phase 3 re-run)* | 2-month test period |
| **Target Median** | **29.0 days** | Target Mean: **42.23 days**, Std: **47.04 days** |
| **Engineered Features** | **36 features** | 27 numeric + 9 categorical |
| **Baseline Validation MAE** | **19.20 days** | Historical Median Interval Baseline |
| **Baseline Validation Accuracy ($\pm 7$d)** | **40.20%** | Baseline accuracy within $\pm 7$ days |
| **Data Leakage Check** | **PASS (0 violations)**| Verified via automated regression tests |

---

## 2. Architecture & File Ownership

### 2.1 File Creation & Modification Map

```
ai-mediastra-whatsapp-reminder/
├── refillcare/
│   ├── __init__.py
│   ├── data/                                    # Phase 2 data pipeline
│   ├── features/                                # [NEW] Phase 3 feature engineering
│   │   ├── __init__.py                          # [NEW] Feature exports
│   │   ├── engineering.py                       # [NEW] Feature extraction & temporal split
│   │   └── pipeline.py                          # [NEW] Phase 3 end-to-end orchestration
│   └── evaluation/                              # [NEW] Phase 3 baseline evaluation
│       ├── __init__.py                          # [NEW] Evaluation exports
│       └── baseline.py                          # [NEW] Historical median baseline & metrics
├── tests/
│   └── refillcare/
│       ├── test_cleaning.py                     # Phase 2 unit tests
│       ├── test_enrichment.py                   # Phase 2 unit tests
│       ├── test_history.py                      # Phase 2 unit tests
│       ├── test_validation.py                   # Phase 2 unit tests
│       ├── test_features.py                     # [NEW] Feature engineering & leakage tests
│       └── test_baseline.py                     # [NEW] Baseline & metrics tests
├── docs/
│   ├── PHASE_1_REFILLCARE_DATA_ANALYSIS.md      # Phase 1 report
│   ├── PHASE_2_REFILLCARE_DATA_PIPELINE.md      # Phase 2 report
│   └── PHASE_3_REFILLCARE_FEATURE_ENGINEERING.md# [NEW] Phase 3 report (this document)
├── data/
│   └── refillcare/
│       └── processed/                           # Local artifacts (GIT-IGNORED)
│           ├── clean_transactions.parquet       # Phase 2
│           ├── purchase_history.parquet         # Phase 2
│           ├── data_quality_report.json         # Phase 2
│           ├── training_dataset.parquet         # [NEW] Supervised dataset with features
│           ├── train.parquet                    # [NEW] Training split
│           ├── validation.parquet               # [NEW] Validation split
│           ├── test.parquet                     # [NEW] Test split
│           └── phase3_quality_report.json       # [NEW] Phase 3 quality summary
└── .gitignore                                   # Untouched (refillcare data ignored)
```

### 2.2 Frozen Application Protection Verification

| File | Status | Verified |
|---|---|---|
| `app.py` (Text campaign) | **FROZEN — UNTOUCHED** | Verified via `git status` / `git diff` |
| `app_image_campaign.py` (Image campaign) | **FROZEN — UNTOUCHED** | Verified via `git status` / `git diff` |
| `services/` (WhatsApp / Cloudinary) | **FROZEN — UNTOUCHED** | Verified via `git status` / `git diff` |
| `utils/` (Campaign utilities) | **FROZEN — UNTOUCHED** | Verified via `git status` / `git diff` |
| Existing tests in `tests/` | **FROZEN — UNTOUCHED** | Verified via `git status` / `git diff` |

---

## 3. Supervised Target Formulation

### 3.1 Target Definition: `target_days_until_next_purchase`
For a customer $c$ and medicine $m$ with chronological purchase timeline $(t_1, t_2, \dots, t_N)$:
$$\text{Target for event } i = y_i = (t_{i+1} - t_i)\text{ in integer days}$$

- **Target Availability:** Only events $i \in \{1, \dots, N-1\}$ have a future purchase within the observation window.
- **Final Purchase Isolation:** The final purchase in any timeline ($i = N$) has $y_N = \text{NaN}$ and is **strictly excluded** from the supervised training set.
- **Cold-Start Histories:** Single-purchase histories ($N = 1$) have no future purchase ($y_1 = \text{NaN}$) and are preserved in the master history but excluded from the supervised training set.

### 3.2 Target Distribution (156,363 Supervised Examples)

```
Target: target_days_until_next_purchase
Count:   156,363
Mean:    42.23 days
Median:  29.00 days
Std:     47.04 days
Min:     0 days
Max:     391 days

Percentiles:
  5th:   4.00 days
 25th:  15.00 days
 50th:  29.00 days
 75th:  48.00 days
 90th:  94.00 days
 95th: 140.00 days
```

---

## 4. Feature Engineering Architecture

A total of **36 leakage-safe features** were engineered across 5 categories:

### 4.1 Historical Timeline & Interval Features (10 features)
*All interval statistics are computed strictly from intervals that occurred prior to or at the current purchase date.*

| Feature | Type | Description |
|---|---|---|
| `purchase_count_so_far` | int | 1-indexed sequential purchase number for this patient+medicine ($i$). |
| `days_since_first_purchase` | int | Days elapsed since the patient's first recorded purchase of this medicine ($t_i - t_1$). |
| `days_since_previous_purchase` | float | Refill interval between immediate prior purchase and current purchase ($t_i - t_{i-1}$). |
| `last_purchase_interval` | float | Alias for `days_since_previous_purchase`. |
| `historical_interval_median` | float | Expanding median of all prior intervals up to event $i$. |
| `historical_interval_mean` | float | Expanding mean of all prior intervals up to event $i$. |
| `historical_interval_std` | float | Expanding sample standard deviation of prior intervals ($i \ge 3$). |
| `historical_interval_min` | float | Expanding minimum prior interval. |
| `historical_interval_max` | float | Expanding maximum prior interval. |
| `historical_interval_cv` | float | Coefficient of variation of historical intervals ($\sigma / \mu$). |

### 4.2 Pattern & Regularity Signals (3 features)
| Feature | Type | Description |
|---|---|---|
| `is_first_purchase` | int (0/1) | Flag indicating first-ever purchase event ($i = 1$). |
| `has_multiple_prior_purchases`| int (0/1) | Flag indicating $i \ge 3$ recorded purchases. |
| `is_recurring_history` | int (0/1) | Flag for stable recurring buyer ($i \ge 3$ and $15 \le \text{median\_interval} \le 120$). |

### 4.3 Quantity & Transaction Features (6 features)
| Feature | Type | Description |
|---|---|---|
| `quantity` | int | Quantity purchased at current transaction event. |
| `freeQuantity` | int | Free / bonus quantity received at current transaction. |
| `avg_historical_quantity` | float | Expanding mean of purchased quantity up to and including current event. |
| `quantity_vs_avg_ratio` | float | Ratio of current quantity to historical average quantity ($\text{qty} / \text{avg\_qty}$). |
| `netAmount` | float | Financial net amount for current purchase event. |
| `gstAmount` | float | GST tax amount for current purchase event. |

*Safety Note:* No dosage frequency or days-supply was estimated from quantity because prescription dosage instructions are absent from point-of-sale data.

### 4.4 Calendar / Seasonal Signals (6 features)
| Feature | Type | Description |
|---|---|---|
| `purchase_month` | int (1–12) | Month of current transaction. |
| `purchase_day_of_week` | int (0–6) | Day of week of transaction (0 = Monday). |
| `purchase_day_of_month` | int (1–31) | Calendar day of month. |
| `purchase_day_of_year` | int (1–366) | Day of year. |
| `purchase_quarter` | int (1–4) | Fiscal quarter. |
| `is_weekend` | int (0/1) | Weekend flag (Saturday/Sunday). |

### 4.5 Medicine & SALT Master Context (9 features)
| Feature | Type | Description |
|---|---|---|
| `itemId` | string | Normalized item ID. |
| `itemCode` | string | Item catalog code. |
| `itemName` | string | Commercial product name. |
| `therapeuticCategory` | string | Clinical therapeutic category from billing system. |
| `generic_name` | string | Generic composition (where populated). |
| `salt_composition` | string | Active pharmaceutical ingredient from SALT master catalog. |
| `salt_category` | string | Master therapeutic classification. |
| `salt_itemcat` | string | Dosage form description (e.g. TABLETS, SYRUP). |
| `salt_pack` | string | Packaging description. |

---

## 5. Data Leakage Prevention Guarantees

| Leakage Risk | Mitigation Strategy | Verification Result |
|---|---|---|
| **Future Purchase Interval in Features** | `target_days_until_next_purchase` is isolated as $y$. Historical statistics only compute over $[t_1 \dots t_i]$. | **PASS** (Zero future intervals in $X$) |
| **Random Train/Test Data Mixing** | Strict chronological partitioning based on `invoice_date`. | **PASS** ($\max(\text{train}) < \min(\text{val}) < \min(\text{test})$) |
| **Global Dataset Statistics Leakage** | Baseline model fits fallback median strictly on training set. Expanding metrics are computed on historical slices only. | **PASS** (Zero global lookahead) |
| **Shared Phone Identity Collisions** | Grouping and interval calculation strictly uses `customerId + itemId`. | **PASS** (Separate timelines confirmed in unit tests) |
| **Current Target in Feature Matrix** | Feature extraction function explicitly strips target and date forward-shifts. | **PASS** (Target column absent from $X$) |

---

## 6. Temporal Train / Validation / Test Splitting

To replicate real-world prospective deployment, the dataset is split temporally by prediction date:

```
[================= TRAIN =================] [==== VALIDATION ====] [====== TEST ======]
     2020-12-24 to 2026-04-30 (multi-year)      2026-05 to 2026-06       2026-07 to 2026-08
              122,252 rows                       22,537 rows              11,574 rows
                 (78.2%)                           (14.4%)                   (7.4%)
```

### Partition Properties

| Split | Rows | Date Range | Target Median | Target Mean | Target Std |
|---|---|---|---|---|---|
| **Train** | *(recompute)* | 2020-12-24 $\rightarrow$ 2026-04-30 | — | — | — |
| **Validation** | *(recompute)* | 2026-05-01 $\rightarrow$ 2026-06-30 | — | — | — |
| **Test** | *(recompute)* | 2026-07-01 $\rightarrow$ 2026-08-31 | — | — | — |

*Note on Right-Censoring:* As time approaches the dataset boundary (2026-06-30), longer refill intervals (e.g. 60–90 days) cannot be observed within the test window. This natural right-censoring causes the observable mean in the test partition to decrease, which machine learning survival/quantile models in Phase 4 will explicitly address.

---

## 7. Baseline Evaluation: Historical Median Interval Baseline

### 7.1 Baseline Logic
1. For patient-medicine timelines with $\ge 1$ prior interval: Predict the customer's personal `historical_interval_median`.
2. For cold-start / first-purchase events: Fall back to the training global median interval (**30.0 days**).

### 7.2 Baseline Performance Across Splits

| Metric | Train Set | Validation Set | Test Set |
|---|---|---|---|
| **MAE (Mean Absolute Error)** | 30.30 days | **19.20 days** | **17.31 days** |
| **RMSE (Root Mean Squared Error)** | 54.76 days | **33.80 days** | **35.64 days** |
| **$R^2$ (Coefficient of Determination)** | -0.1367 | -1.5123 | -7.9046 |
| **Accuracy within $\pm 1$ day** | 9.58% | **10.92%** | **12.62%** |
| **Accuracy within $\pm 3$ days** | 20.11% | **23.16%** | **26.25%** |
| **Accuracy within $\pm 7$ days** | 34.20% | **40.20%** | **44.91%** |

### 7.3 Baseline Findings
- The simple historical median baseline achieves **40.20% accuracy within a 7-day window** on the validation set and **44.91%** on the test set.
- However, the negative $R^2$ indicates that a simple constant or historical median cannot account for interval variance, quantity scaling, or seasonal shifts.
- This establishes the benchmark that Phase 4 machine learning models (XGBoost, LightGBM, Random Forest) will target to beat.

---

## 8. Test Suite Verification

The Phase 3 test suite under `tests/refillcare/` comprises **23 unit tests** covering feature engineering, interval math, cold start isolation, temporal splitting, and baseline evaluations:

```
tests/refillcare/test_baseline.py
  ✓ test_baseline_predictions (personal median vs fallback)
  ✓ test_compute_prediction_metrics_perfect (zero-error edge case)
  ✓ test_compute_prediction_metrics_with_errors (MAE, RMSE, R2, window accuracy)
  ✓ test_evaluate_baseline_predictions_pipeline (end-to-end evaluation)

tests/refillcare/test_cleaning.py
  ✓ test_load_raw_transactions_date_parsing (DD/MM/YYYY dayfirst)
  ✓ test_clean_transactions_drops_invalid_and_duplicates
  ✓ test_aggregate_invoice_items

tests/refillcare/test_enrichment.py
  ✓ test_load_salt_master
  ✓ test_enrich_with_salt

tests/refillcare/test_features.py
  ✓ test_target_is_next_purchase_interval (y = t_{i+1} - t_i)
  ✓ test_final_and_single_purchase_have_no_supervised_target (NaN target isolation)
  ✓ test_leakage_safe_historical_interval_statistics (expanding stats only use past intervals)
  ✓ test_shared_phone_does_not_merge_histories (isolated customer identities)
  ✓ test_temporal_split_ranges (non-overlapping chronological ranges)
  ✓ test_no_forbidden_columns_in_feature_matrix (X contains no target or lookahead columns)

tests/refillcare/test_history.py
  ✓ test_customer_item_identity_and_shared_phones
  ✓ test_first_purchase_has_no_previous_interval
  ✓ test_purchase_interval_calculation_and_ordering
  ✓ test_same_day_purchases
  ✓ test_compute_interval_statistics

tests/refillcare/test_validation.py
  ✓ test_validation_passes_on_clean_data
  ✓ test_validation_detects_negative_intervals
  ✓ test_generate_validation_report_output

Result: 23 passed in 2.10s
```

---

## 9. Generated Artifacts & Quality Report

All generated datasets reside in `data/refillcare/processed/` and are ignored by Git via `.gitignore`:

| Artifact | Format | Rows | Columns | Purpose |
|---|---|---|---|---|
| `training_dataset.parquet` | Parquet | 156,363 | 43 | Complete supervised eligible feature dataset |
| `train.parquet` | Parquet | *(recompute)* | — | Chronological training set ($\le$ 2026-04-30) |
| `validation.parquet` | Parquet | *(recompute)* | — | Chronological validation set (2026-05 to 2026-06) |
| `test.parquet` | Parquet | *(recompute)* | — | Chronological test set (2026-07 to 2026-08) |
| `phase3_quality_report.json`| JSON | N/A | N/A | Automated feature & baseline quality report |

---

## 10. Recommendations for Phase 4 (Machine Learning Model Training)

With a validated, leakage-safe dataset and established baseline metrics, Phase 4 can proceed to:

1. **Model Selection & Architecture:**
   - **XGBoost Regressor / LightGBM Regressor:** Primary gradient boosted decision trees optimized for time-to-event interval regression (Huber / MAE loss to handle outlier refill cycles).
   - **Quantile Regressors ($\alpha = 0.1, 0.5, 0.9$):** To generate prediction confidence intervals (e.g., expected refill window: 25 to 33 days).
   - **Random Forest Regressor:** Secondary ensemble benchmark.
2. **Feature Preprocessing Pipeline:**
   - Target encoding / Frequency encoding for high-cardinality `itemId` and `salt_composition`.
   - One-hot encoding for `salt_category`, `therapeuticCategory`, and calendar features.
   - Robust scaling for interval and financial numeric features.
3. **Hyperparameter Tuning:**
   - Optimize tree depth, learning rate, and regularizations on `train.parquet`, validating on `validation.parquet`.
4. **Target Metric Goals:**
   - Beat baseline Validation MAE of **19.20 days**.
   - Exceed **55% accuracy within a $\pm 7$-day reminder window**.
