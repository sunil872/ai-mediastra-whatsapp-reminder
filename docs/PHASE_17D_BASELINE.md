# Phase 17D: RefillCare Baseline & Model Performance Benchmark Report

**Evaluation Date:** 2026-09-17  
**Dataset Version:** 5.8-Year Pharmacy Dataset (`data/refillcare/customer_data_fields.csv`)  
**Evaluation Scope:** Prospective out-of-time Test Holdout partition (`data/refillcare/processed/test.parquet`) and Validation partition (`data/refillcare/processed/validation.parquet`).  
**Mode:** Read-Only Empirical Audit (No code changes, no dataset alterations, no model retraining).

---

## 1. Executive Summary

This report establishes the verified performance baseline of the **serialized XGBoost regression pipeline** (`data/refillcare/processed/models/refill_model.joblib`) compared directly against the **Historical Median Baseline** across the full 5.8-year dataset.

```
========================================================================================================
AGGREGATE TEST SET BENCHMARK (Test Holdout: 2026-07-01 to 2026-08-31, N=7,556)
========================================================================================================
Evaluation Metric                 Historical Median Baseline      Current XGBoost Pipeline       Difference
--------------------------------------------------------------------------------------------------------
MAE (Mean Absolute Error)                 27.00 days                     43.71 days            +16.71 days
MedAE (Median Absolute Error)              7.00 days                     19.84 days            +12.84 days
RMSE (Root Mean Squared Error)           109.54 days                     70.31 days            -39.23 days
Accuracy within ±1 Day                    18.87%                          3.26%                -15.61%
Accuracy within ±3 Days                   31.82%                          9.85%                -21.97%
Accuracy within ±7 Days                   50.61%                         23.41%                -27.20%
Accuracy within ±14 Days                  70.33%                         41.97%                -28.36%
Observed Target Statistics           Mean: 18.80d, Median: 17.00d, Min: 0.0d, Max: 60.0d
Model Prediction Statistics          Mean: 41.11d, Median: 25.00d    Mean: 61.47d, Median: 38.84d
========================================================================================================
```

```
========================================================================================================
AGGREGATE VALIDATION SET BENCHMARK (Validation Window: 2026-05-01 to 2026-06-30, N=12,821)
========================================================================================================
Evaluation Metric                 Historical Median Baseline      Current XGBoost Pipeline       Difference
--------------------------------------------------------------------------------------------------------
MAE (Mean Absolute Error)                 31.35 days                     43.00 days            +11.65 days
MedAE (Median Absolute Error)              8.50 days                     19.65 days            +11.15 days
RMSE (Root Mean Squared Error)           115.85 days                     68.66 days            -47.19 days
Accuracy within ±1 Day                    14.66%                          3.04%                -11.62%
Accuracy within ±3 Days                   29.08%                          9.82%                -19.26%
Accuracy within ±7 Days                   46.38%                         23.03%                -23.35%
Accuracy within ±14 Days                  64.04%                         40.69%                -23.35%
========================================================================================================
```

---

## 2. Dataset & Partition Specifications

All evaluations were executed on the canonical temporal partitions generated from `data/refillcare/customer_data_fields.csv` (895,557 raw transaction rows):

| Partition | File Path | Event Rows | Date Coverage Range | Days Span | Purpose |
|---|---|---|---|---|---|
| **Train Set** | `data/refillcare/processed/train.parquet` | **469,583** | `2020-12-24` → `2026-04-30` | 1,954 days (~5.35 yrs) | Supervised model training |
| **Validation Set** | `data/refillcare/processed/validation.parquet` | **12,821** | `2026-05-01` → `2026-06-30` | 61 days (2 months) | Out-of-time hyperparameter tuning |
| **Test Set (Holdout)**| `data/refillcare/processed/test.parquet` | **7,556** | `2026-07-01` → `2026-08-31` | 62 days (2 months) | Unbiased prospective evaluation |

---

## 3. Results by Purchase-History Depth (Test Holdout: $N=7,556$)

The depth of a patient's historical purchase record (`purchase_count_so_far`) strongly governs refill regularity and prediction accuracy:

| History Depth Cohort | Test Rows ($N$) | Cohort Share (%) | True Mean / Median | Baseline MAE | XGBoost MAE | Baseline MedAE | XGBoost MedAE | Baseline $\pm 1$d (%) | XGBoost $\pm 1$d (%) | Baseline $\pm 3$d (%) | XGBoost $\pm 3$d (%) | Baseline $\pm 7$d (%) | XGBoost $\pm 7$d (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1 Purchase (Cold Start)** | **790** | 10.5% | 17.04d / 14.0d | **16.94d** | 159.96d | **17.0d** | 166.1d | **4.56%** | 0.00% | **9.49%** | 0.00% | **18.99%** | 0.00% |
| **2 Purchases (1 Interval)** | **517** | 6.8% | 18.49d / 16.0d | 127.12d | **93.96d** | **13.0d** | 69.3d | **13.54%** | 0.00% | **23.98%** | 0.00% | **38.49%** | 0.00% |
| **3–5 Purchases** | **1,066** | 14.1% | 18.86d / 17.0d | 63.97d | **58.07d** | **14.0d** | 42.5d | **10.23%** | 0.19% | **20.08%** | 0.56% | **34.15%** | 1.88% |
| **6–10 Purchases** | **1,059** | 14.0% | 20.21d / 18.0d | **20.15d** | 33.13d | **8.0d** | 23.1d | **14.26%** | 0.76% | **28.42%** | 3.49% | **49.10%** | 11.90% |
| **>10 Purchases (Deep Chronic)**| **4,124** | **54.6%** | 18.80d / 17.0d | **8.58d** | 14.14d | **5.0d** | 9.5d | **21.90%** | 5.67% | **40.98%** | 17.00% | **62.83%** | 39.35% |

### Key Depth Takeaways:
1. **Cold-Start Fragility (1st Purchase, $N=790$):**  
   XGBoost assigns 1st-time test patients to tree leaves predicting an average of **177.01 days** (resulting in an MAE of **159.96 days** and 0.0% $\pm 7$d accuracy). The Baseline fallback to the empirical dataset median (31.0 days) yields an MAE of **16.94 days**.
2. **Transition Regime (2 to 5 Purchases, $N=1,583$):**  
   For patients with only 1–4 prior intervals, multi-year historical gaps from the 5.8-year history distort the single-interval baseline (Baseline MAE: 127.12d for $P=2$), while XGBoost over-predicts due to tree depth splits (XGBoost MAE: 93.96d).
3. **Deep Chronic Regimes (>10 Purchases, $N=4,124$, 54.6% of volume):**  
   For established chronic patients, the Historical Median Baseline achieves **8.58 days MAE**, **5.00 days MedAE**, and **62.83% accuracy within $\pm 7$ days** (81.81% within $\pm 14$ days). XGBoost achieves **14.14 days MAE** and **39.35% within $\pm 7$ days**.

---

## 4. Results by Historical Interval Pattern (Test Holdout: $N=7,556$)

Segmenting the test population by their historical median refill interval (`historical_interval_median`) highlights where models succeed and fail:

| Historical Interval Pattern | Test Rows ($N$) | Cohort Share (%) | True Mean / Median | Baseline MAE | XGBoost MAE | Baseline MedAE | XGBoost MedAE | Baseline $\pm 1$d (%) | XGBoost $\pm 1$d (%) | Baseline $\pm 3$d (%) | XGBoost $\pm 3$d (%) | Baseline $\pm 7$d (%) | XGBoost $\pm 7$d (%) | Baseline $\pm 14$d (%) | XGBoost $\pm 14$d (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Cold Start (No Prior History)** | **790** | 10.5% | 17.04d / 14.0d | **16.94d** | 159.96d | **17.0d** | 166.1d | **4.56%** | 0.00% | **9.49%** | 0.00% | **18.99%** | 0.00% | **37.97%** | 0.00% |
| **Ultra-Short (< 15 Days)** | **1,892** | 25.0% | 12.23d / 11.0d | **5.16d** | 18.69d | **3.0d** | 9.6d | **31.45%** | 5.29% | **54.81%** | 17.02% | **78.81%** | 39.43% | **91.23%** | 62.42% |
| **Standard Chronic (15–30 Days)** | **2,789** | **36.9%** | 19.74d / 18.0d | **7.85d** | 19.12d | **6.0d** | 13.6d | **16.39%** | 4.02% | **33.88%** | 11.69% | **57.58%** | 28.07% | **83.29%** | 51.09% |
| **Extended Chronic (31–45 Days)** | **1,154** | 15.3% | 25.88d / 28.5d | **11.52d** | 27.30d | **8.0d** | 22.0d | **14.12%** | 2.51% | **27.12%** | 7.02% | **45.41%** | 18.72% | **65.77%** | 35.70% |
| **Bimonthly/Quarterly (46–90 Days)**| **468** | 6.2% | 23.86d / 24.0d | **37.12d** | 67.00d | **35.0d** | 59.6d | **0.64%** | 0.21% | **1.28%** | 0.43% | **2.99%** | 0.64% | **7.91%** | 2.35% |
| **Long / Irregular (> 90 Days)** | **463** | 6.1% | 20.22d / 18.0d | 277.16d | **113.06d** | 156.0d | 109.0d | **3.24%** | 0.43% | **6.05%** | 2.81% | **8.42%** | 4.54% | **10.58%** | 8.42% |

### Key Pattern Takeaways:
1. **Core Chronic Refills (15–45 Days, $N=3,943$, 52.2% of test volume):**  
   On standard 15–45 day chronic medication refills, the Baseline achieves **8.92 days MAE** and **54.02% accuracy within $\pm 7$ days** (78.16% within $\pm 14$ days). XGBoost predicts an average of 41.87 days, yielding **21.51 days MAE** and **25.34% within $\pm 7$ days**.
2. **Ultra-Short Recurring Events (< 15 Days, $N=1,892$, 25.0% of test volume):**  
   For high-frequency multi-pack or acute re-purchases, the Baseline achieves **5.16 days MAE**, **3.00 days MedAE**, and **78.81% accuracy within $\pm 7$ days** (91.23% within $\pm 14$ days).
3. **Long / Irregular History (> 90 Days, $N=463$, 6.1% of test volume):**  
   Patients with a history of long gaps (>90d) who suddenly repurchased rapidly within the 60-day test window cause large errors in both models (Baseline MAE: 277.16d, XGBoost MAE: 113.06d).

---

## 5. Results by Historical Interval Consistency (CV Cohorts, $P \ge 3$)

For patients with at least 3 historical purchases ($N=6,249$), interval consistency is measured by the historical coefficient of variation ($\text{CV} = \sigma / \mu$):

| Consistency Cohort (CV) | Test Rows ($N$) | Share of $P \ge 3$ (%) | Baseline MAE | XGBoost MAE | Baseline MedAE | XGBoost MedAE | Baseline $\pm 3$d (%) | XGBoost $\pm 3$d (%) | Baseline $\pm 7$d (%) | XGBoost $\pm 7$d (%) | Baseline $\pm 14$d (%) | XGBoost $\pm 14$d (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Highly Consistent ($\text{CV} \le 0.20$)** | **286** | 4.6% | **19.31d** | 25.74d | **3.5d** | 17.6d | **49.65%** | 11.89% | **66.08%** | 26.92% | **79.72%** | 42.66% |
| **Moderately Consistent ($0.20 < \text{CV} \le 0.50$)**| **1,585** | 25.4% | **10.22d** | 14.98d | **4.0d** | 8.4d | **44.16%** | 20.19% | **66.50%** | 44.29% | **82.33%** | 67.19% |
| **Highly Variable ($\text{CV} > 0.50$)** | **4,378** | 70.1% | **23.58d** | 28.37d | **7.0d** | 17.5d | **31.13%** | 8.91% | **50.98%** | 22.61% | **70.67%** | 42.94% |

---

## 6. Comprehensive Metric Comparison Summary

```
========================================================================================================================
COMPLETE TEST HOLDOUT BREAKDOWN TABLE (N=7,556)
========================================================================================================================
Cohort Segment                  Count   % Set  | Base MAE  XGB MAE | Base MedAE  XGB MedAE | Base ±3d  XGB ±3d | Base ±7d  XGB ±7d
------------------------------------------------------------------------------------------------------------------------
Overall Test Partition          7,556  100.0%  |   27.00     43.71 |     7.00      19.84 |   31.82%    9.85% |   50.61%   23.41%
------------------------------------------------------------------------------------------------------------------------
[History Depth]
- 1 Purchase (Cold Start)         790   10.5%  |   16.94    159.96 |    17.00     166.10 |    9.49%    0.00% |   18.99%    0.00%
- 2 Purchases                     517    6.8%  |  127.12     93.96 |    13.00      69.30 |   23.98%    0.00% |   38.49%    0.00%
- 3 to 5 Purchases              1,066   14.1%  |   63.97     58.07 |    14.00      42.49 |   20.08%    0.56% |   34.15%    1.88%
- 6 to 10 Purchases             1,059   14.0%  |   20.15     33.13 |     8.00      23.08 |   28.42%    3.49% |   49.10%   11.90%
- >10 Purchases                 4,124   54.6%  |    8.58     14.14 |     5.00       9.48 |   40.98%   17.00% |   62.83%   39.35%
------------------------------------------------------------------------------------------------------------------------
[Historical Interval Pattern]
- Cold Start (No History)         790   10.5%  |   16.94    159.96 |    17.00     166.10 |    9.49%    0.00% |   18.99%    0.00%
- Ultra-Short (<15d)            1,892   25.0%  |    5.16     18.69 |     3.00       9.57 |   54.81%   17.02% |   78.81%   39.43%
- Standard Chronic (15-30d)     2,789   36.9%  |    7.85     19.12 |     6.00      13.63 |   33.88%   11.69% |   57.58%   28.07%
- Extended Chronic (31-45d)     1,154   15.3%  |   11.52     27.30 |     8.00      22.03 |   27.12%    7.02% |   45.41%   18.72%
- Bimonthly/Quarterly (46-90d)    468    6.2%  |   37.12     67.00 |    35.00      59.60 |    1.28%    0.43% |    2.99%    0.64%
- Long/Irregular (>90d)           463    6.1%  |  277.16    113.06 |   156.00     108.99 |    6.05%    2.81% |    8.42%    4.54%
------------------------------------------------------------------------------------------------------------------------
[Consistency on P >= 3]
- Highly Consistent (CV <= 0.2)   286    3.8%  |   19.31     25.74 |     3.50      17.64 |   49.65%   11.89% |   66.08%   26.92%
- Moderately Consistent (0.2-0.5)1,585  21.0%  |   10.22     14.98 |     4.00       8.42 |   44.16%   20.19% |   66.50%   44.29%
- Variable (CV > 0.5)           4,378   57.9%  |   23.58     28.37 |     7.00      17.49 |   31.13%    8.91% |   50.98%   22.61%
========================================================================================================================
```

---
