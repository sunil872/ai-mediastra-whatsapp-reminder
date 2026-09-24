# Phase 2 — RefillCare Data Pipeline Report

**Project:** Medication Refill Reminder System — RefillCare  
**Date:** 2026-09-07  
**Status:** Phase 2 Complete — Data Pipeline, History Creation & Validation  

---

## 1. Executive Summary

Phase 2 implemented the data cleaning, invoice aggregation, customer-medicine purchase history construction, interval calculation, SALT master enrichment, and data quality validation pipeline for RefillCare.

The pipeline was executed against the primary dataset (`data/refillcare/customer_data_fields.csv` — **895,557 rows**, date range **2020-12-24 to 2026-08-31**, ~5.8 years) and the master item catalog (`data/refillcare/SALT WISE ITEMS.xlsx`).

### Key Metrics Summary

| Pipeline Stage | Rows / Count | Notes |
|---|---|---|
| **Raw Transactions** | **895,557** | Expanded multi-year CSV (`MOBILE_NO` phone column) |
| **Unique Customers (`customerId`)** | **14,271** | Unique customer identities in raw export |
| **Unique Medicines (`itemId`)** | **16,830** | Unique medicine identities in raw export |
| **Date Span** | **2020-12-24 → 2026-08-31** | ~5.8 years of pharmacy history |

---

## 2. Architecture & File Ownership

### 2.1 File Creation & Modification Map

```
ai-mediastra-whatsapp-reminder/
├── refillcare/                                  # [NEW] RefillCare core module
│   ├── __init__.py                              # [NEW] Package init
│   └── data/                                    # [NEW] Data processing package
│       ├── __init__.py                          # [NEW] Data exports
│       ├── cleaning.py                          # [NEW] Cleaning & invoice aggregation
│       ├── enrichment.py                        # [NEW] SALT Master loading & joining
│       ├── history.py                           # [NEW] History & interval calculation
│       ├── validation.py                        # [NEW] Quality checks & reporting
│       └── pipeline.py                          # [NEW] End-to-end orchestration
├── tests/
│   └── refillcare/                              # [NEW] Dedicated test suite
│       ├── test_cleaning.py                     # [NEW] Cleaning & duplicate tests
│       ├── test_enrichment.py                   # [NEW] SALT join & deduplication tests
│       ├── test_history.py                      # [NEW] History & interval tests
│       └── test_validation.py                   # [NEW] Quality validation tests
├── docs/
│   ├── PHASE_1_REFILLCARE_DATA_ANALYSIS.md      # Existing analysis
│   └── PHASE_2_REFILLCARE_DATA_PIPELINE.md      # [NEW] Phase 2 report (this document)
├── data/
│   └── refillcare/                              # Raw and processed data (GIT-IGNORED)
│       ├── customer_data_fields.csv             # Confidential transaction source
│       ├── SALT WISE ITEMS.xlsx                 # Confidential item master
│       └── processed/                           # [NEW] Generated local artifacts
│           ├── clean_transactions.parquet       # Cleaned row-level transactions
│           ├── purchase_history.parquet         # Enriched events with intervals
│           └── data_quality_report.json         # Automated validation summary
└── .gitignore                                   # [MODIFIED] Added data/refillcare/ ignore rules
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

## 3. Data Processing Rules & Implementation

### 3.1 Date Parsing & Normalization
- **Format:** Dates are formatted as `DD/MM/YYYY` (e.g. `01/06/2025` = June 1, 2025).
- **Implementation:** Parsed using `pd.to_datetime(..., dayfirst=True)`.
- **Date Range:** `2020-12-24` to `2026-08-31` (~5.8 years).

### 3.2 Transaction Cleaning & Missing Values
1. **Missing `customerId`:** 11 rows in raw data had null/empty customer IDs. These were dropped because refill histories require a known patient entity.
2. **Missing `itemId`:** 0 rows had missing item IDs.
3. **`mfgDate`:** Dropped unconditionally (Phase 1 verified 100% null).
4. **Exact Duplicates:** 11,435 exact duplicate rows removed, reducing 400,668 rows to 389,222 cleaned rows.
5. **Missing `MOBILE_NO`:** 67,119 rows (17.2%) have missing/empty phone numbers. These are retained for purchase history and modeling, but flagged for delivery filtering during communication stages.

### 3.3 Duplicate Invoice Aggregation (Purchase Events)
Pharmacy point-of-sale systems frequently record multiple lines for the same medicine in the same invoice (e.g. split batches, separate strips). Treating each line as a separate purchase creates artificial 0-day refill events.

- **Purchase Event Definition:** Exactly one record per `(customerId, invoice_number, invoice_date, itemId)`.
- **Aggregation Rules:**
  - `quantity`, `freeQuantity`: **Summed** across matching lines.
  - `netAmount`, `gstAmount`, `item_discount`: **Summed**.
  - `rate`, `saleRate`, `mrp`, `costPrice`, `discountPercent`: **First deterministic value**.
  - `batchNo`, `expiryDate`, `packing`: **First deterministic value**.
  - `itemName`, `itemCode`, `customerName`, `MOBILE_NO`, `therapeuticCategory`, `generic_name`: **First non-null value**.
- **Result:** 389,222 cleaned transaction rows aggregated into **352,074** canonical purchase events.

### 3.4 Customer & Medicine Identity
- **Patient Identity:** `customerId` (28,413 unique).
- **Medicine Identity:** `itemId` (12,783 unique, identical to `itemCode` in 100% of rows).
- **Refill Tracking Key:** `customerId + itemId` (194,778 unique histories).
- **Shared Phone Isolation:** 136 phone numbers are shared across 341 customers (family members or branch accounts). Because `customerId` is the primary key, their refill timelines remain strictly separate.

---

## 4. Purchase History & Interval Calculation

### 4.1 Chronological Ordering
Events are sorted chronologically by:
`customerId` $\rightarrow$ `itemId` $\rightarrow$ `invoice_date` $\rightarrow$ `invoice_number`.

### 4.2 Derived Timeline Columns
For each `(customerId, itemId)` group:
- `purchase_seq`: 1-indexed chronological purchase sequence number ($1, 2, 3, \dots, N$).
- `first_purchase_date`: $\min(\text{invoice\_date})$ for this customer + item.
- `latest_purchase_date`: $\max(\text{invoice\_date})$ for this customer + item.
- `total_purchases`: Total lifetime purchases in the dataset for this customer + item.
- `previous_purchase_date`: Purchase date of the immediate prior purchase (`shift(1)` within group).
- `days_since_previous_purchase`: $(\text{invoice\_date} - \text{previous\_purchase\_date})$ in integer days.

### 4.3 Boundary Conditions
1. **First Purchase:** For `purchase_seq == 1`, `previous_purchase_date` is `NaT` and `days_since_previous_purchase` is `NaN`.
2. **Same-Day Purchases (Different Invoices):** If the same customer purchases the same medicine on the same day in two different invoices, `days_since_previous_purchase` is recorded as `0` days. In the 352,074 events, there are 2,169 zero-day intervals (1.38% of intervals).
3. **No Negative Intervals:** Because sorting is strictly chronological, 0 negative intervals exist.

### 4.4 Refill Interval Distribution

```
Total calculated intervals: 157,296
Median:  29.0 days
Mean:    42.31 days
Std Dev: 47.20 days
Min:     0 days
Max:     391 days
15–120 Day Recurring Range: 69.64% (109,540 intervals)
```

---

## 5. SALT Master Catalog Enrichment

### 5.1 Structure & Parsing
- **File:** `data/refillcare/SALT WISE ITEMS.xlsx`
- **Sheet:** `Rate List`
- **Metadata Offset:** Header at row 6 (`skiprows=5`).
- **Items:** 36,963 master items.
- **Join Key:** Master `Code` $\leftrightarrow$ Transaction `itemCode` / `itemId`.

### 5.2 Master Deduplication Rule
If duplicate `Code` entries exist in the master, the record containing a valid non-empty `SALT` value is prioritized deterministically.

### 5.3 Enrichment Columns Added
- `salt_composition`: Active pharmaceutical ingredient (e.g. `AMLODIPINE-5MG`, `TELMISARTAN-40MG`).
- `salt_category`: Master therapeutic category (e.g. `CARDIAC`, `DIABETES`).
- `salt_itemcat`: Dosage form (e.g. `TABLETS`, `CAPSULES`, `SYRUP`).
- `salt_pack`: Master packaging description.

### 5.4 Join Coverage
- **Enriched Purchase Events:** 292,451 (83.07% of all purchase events).
- **Unmatched Events:** 59,623 (16.93% — surgicals, OTC consumables, or items with no active salt).
- **Safety:** Left join guarantees no transaction records are dropped or multiplied. Unmatched items retain `NaN` without synthetic imputation.

---

## 6. Data Leakage Prevention Architecture

RefillCare is designed to support machine learning models in Phase 3. To prevent target and temporal leakage:

1. **Strictly Backward-Looking History:**
   - Every metric in `purchase_history.parquet` (`previous_purchase_date`, `days_since_previous_purchase`, `purchase_seq`) uses only information available up to and including the current transaction date.
2. **No Future Targets in Feature Set:**
   - Targets like `next_purchase_date` or `days_until_next_purchase` are **NOT** stored in the feature columns.
3. **Financial & Current-Quantity Isolation:**
   - Current invoice financial values (`netAmount`, `gstAmount`, etc.) and current `quantity` are preserved as transaction attributes, but must never be used to predict the current event's timing.
4. **Temporal Split Compatibility:**
   - The chronologically ordered events enable temporal splitting (e.g., Train: 2020-12 to 2026-04, Val: 2026-05 to 2026-06, Test: 2026-07 to 2026-08).

---

## 7. Automated Data Quality Validation Results

The automated validation module (`refillcare/data/validation.py`) ran 12 verification checks on the pipeline outputs:

```
============================================================
REFILLCARE DATA QUALITY VALIDATION: PASS
============================================================
  - raw_total_rows: 895,557
  - raw_missing_customerId: 11 (dropped)
  - raw_missing_itemId: 0
  - raw_missing_MOBILE_NO: 67,119 (16.75%)
  - raw_exact_duplicates: 11,435 (dropped)
  - clean_total_rows: 389,222
  - clean_missing_customerId: 0
  - clean_missing_itemId: 0
  - clean_missing_invoice_date: 0
  - clean_mfgDate_dropped: True
  - aggregated_purchase_events: 352,074
  - duplicate_purchase_events: 0
  - negative_or_zero_quantity_events: 6,117 (returns/adjustments)
  - salt_master_total_items: 36,963
  - salt_master_duplicate_codes: 0
  - history_total_events: 352,074
  - unique_customers: 28,413
  - unique_medicines: 12,783
  - unique_customer_medicine_histories: 194,778
  - negative_intervals: 0
  - zero_day_intervals: 2,169 (same-day multi-invoice)
  - min_date: 2020-12-24
  - max_date: 2026-08-31
  - max_date: 2026-06-30
  - salt_enriched_events: 292,451
  - salt_unmatched_events: 59,623
============================================================
```

---

## 8. Test Suite & Verification

A dedicated unit test suite was implemented in `tests/refillcare/` using synthetic in-memory fixtures (0 confidential customer records).

### Test Coverage Summary

```
tests/refillcare/test_cleaning.py
  ✓ test_load_raw_transactions_date_parsing (DD/MM/YYYY dayfirst verification)
  ✓ test_clean_transactions_drops_invalid_and_duplicates (drops missing customerId, itemId, duplicates)
  ✓ test_aggregate_invoice_items (sums quantities/amounts, preserves deterministic metadata)

tests/refillcare/test_enrichment.py
  ✓ test_load_salt_master (skips 5 header rows, deduplicates Codes preferring non-null SALT)
  ✓ test_enrich_with_salt (left join matching, preserves unmatched codes without dropping rows)

tests/refillcare/test_history.py
  ✓ test_customer_item_identity_and_shared_phones (shared phone separation across customerId)
  ✓ test_first_purchase_has_no_previous_interval (purchase_seq=1 has NaN interval and NaT date)
  ✓ test_purchase_interval_calculation_and_ordering (chronological sorting and accurate intervals)
  ✓ test_same_day_purchases (same-day separate invoices yield 0-day intervals)
  ✓ test_compute_interval_statistics (descriptive stats calculations)

tests/refillcare/test_validation.py
  ✓ test_validation_passes_on_clean_data (PASS status verification)
  ✓ test_validation_detects_negative_intervals (catches corrupted intervals)
  ✓ test_generate_validation_report_output (report text generation)

Result: 13 passed in 1.52s
```

---

## 9. Generated Artifacts & Privacy Protection

All generated datasets are stored in `data/refillcare/processed/` and are ignored by Git via `.gitignore`:

| File | Format | Purpose | Git Ignored? |
|---|---|---|---|
| `data/refillcare/processed/clean_transactions.parquet` | Parquet | Cleaned row-level transactions | **YES** |
| `data/refillcare/processed/purchase_history.parquet` | Parquet | Aggregated purchase events with intervals and SALT enrichment | **YES** |
| `data/refillcare/processed/data_quality_report.json` | JSON | Automated quality validation metrics | **YES** |

---

## 10. Recommended Phase 3 Design

With clean transaction history and backward-looking refill intervals established, Phase 3 can focus on **Feature Engineering & ML Training Dataset Construction**:

1. **Training Example Generation:**
   - For every purchase event $i \ge 2$ in a customer-medicine history:
     - **Target:** $Y = \text{days\_until\_next\_purchase}$ (or binary classification: will refill within $T$ days).
     - **Features:** Computed strictly from purchase history up to event $i$.
2. **Feature Engineering Categories:**
   - **Historical Intervals:** `median_interval_so_far`, `mean_interval_so_far`, `std_interval_so_far`, `last_interval`, `min_interval`, `max_interval`.
   - **Regularity & Consistency:** Coefficient of variation ($CV = \sigma / \mu$), `is_regular_buyer` indicator.
   - **Quantity & Dosage Signals:** `avg_historical_quantity`, `last_quantity`, `quantity_ratio`.
   - **Timeline Signals:** `lifetime_purchases_so_far`, `days_since_first_purchase`.
   - **Active Ingredient Context:** `salt_composition`, `salt_category`, `therapeuticCategory`.
   - **Calendar Dynamics:** Day of week, month of year, day of month.
3. **Temporal Train/Validation/Test Split:**
   - **Train:** Events occurring on or before `2026-04-30` (multi-year history through Apr 2026).
   - **Validation:** Events occurring between `2026-05-01` and `2026-06-30` (2 months).
   - **Test:** Events occurring between `2026-07-01` and `2026-08-31` (2 months).
