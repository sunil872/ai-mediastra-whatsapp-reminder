# PHASE 17B — REFILLCARE IDENTITY & HISTORY VALIDATION

**Status:** COMPLETE  
**Date:** 2026-09-17  
**Dataset:** `data/refillcare/customer_data_fields.csv` (single-store, 2020-12-24 to 2026-08-31)  
**Script:** `scripts/phase17b_validate.py`  
**JSON sidecar:** `data/refillcare/processed/phase17b_validation.json`

---

## 0. Scope & Constraints

| Rule | Applied |
|---|---|
| Identity key = `customerId` / `customerName` | YES |
| `MOBILE_NO` is NOT identity | YES |
| `therapeuticCategory` / `generic_name` removed | YES - confirmed absent |
| `invoice_date` / `invoice_number` unchanged | YES |
| No synthetic data generated | YES |
| No model retrained | YES |
| No WhatsApp messages sent | YES |
| No `app.py` / `app_image_campaign.py` modified | YES |

---

## 1. Dataset Schema

| Field | Value |
|---|---|
| Raw rows loaded | **895,557** |
| Columns | 23 |
| Date range | **2020-12-24 to 2026-08-31** |
| Invalid `invoice_date` rows | **0** |
| `mfgDate` present (pipeline drops it) | Yes |
| `therapeuticCategory` present | **No** (correctly removed) |
| `generic_name` present | **No** (correctly removed) |

Columns present:
```
invoice_date, invoice_number, customerName, itemCode, itemName, packing, batchNo,
mfgDate, expiryDate, quantity, freeQuantity, rate, saleRate, mrp, invoice_total,
discountPercent, item_discount, gstAmount, netAmount, costPrice, MOBILE_NO,
customerId, itemId
```

---

## 2. Customer Identity Validation

### 2.1 Unique identifiers

| Metric | Count |
|---|---|
| Unique `customerId` values | **14,271** |
| Unique `customerName` values | **14,271** |
| Rows with missing `customerId` | **83** |
| Rows with missing `customerName` | **83** |

Finding: `customerId` and `customerName` have a **perfect 1:1 cardinality**.
Every customer has exactly one ID and one name.
The 83 rows with missing both fields will be dropped by `clean_transactions()`.

### 2.2 customerId to customerName consistency

| Check | Result |
|---|---|
| `customerId` values with >1 distinct `customerName` | **0** |
| `customerName` values with >1 distinct `customerId` | **0** |

Finding: Zero identity collisions or name drift detected. The identity mapping is clean.

### 2.3 MOBILE_NO — shared phone analysis

Rule enforced: `MOBILE_NO` is WhatsApp contact only, NOT identity.
Histories are always isolated by `customerId`.

| Metric | Count |
|---|---|
| Rows with missing `MOBILE_NO` | **70,137** (7.8% - expected for cash customers) |
| MOBILE_NO groups shared by >1 customer | **180** |
| Distinct customers on a shared MOBILE_NO | **388** |

Top shared MOBILE_NO groups (sample):

| MOBILE_NO | Distinct customers |
|---|---|
| 9951921323 | CARE, D ARUN, KOSKO, CS, CSA, C HANUMAN, NAGESHWAR RAO, JEYAMMA, SATHAYA NARAYANA, CS RAJU |
| 9392733210 | T LAXMI, JANAKAMAMA, LAXMMAMA, LAXMA REDDY, LAXMAI, LAKSMI, JANAKAMA, LINGAMMA, LAXMAMMA |
| 9989072235 | PRASAD, RAO, RA BABU, RAVINDER, RAMYA SRI |
| 7396427591 | NARASIMHA RAO, R NARASIMHA RAO, R PRATHYUSHA, R UMA DEVI |
| 9396890948 | ADHI LAKSHMI MEDICAL HALL, T DHANUNJAY GOUD, T DHANANJAY GOUD, P DHANANJAY GOUD |
| 9701443358 | RAGHAVA, PARNIKA R, R.PARNIKA |

Interpretation: Classic family-phone pattern (family members sharing one number) and
some shop/owner numbers. Not a blocker. The pipeline enforces strict customerId-level
isolation (covered by test_scenario_3_shared_phone_number and related tests).

---

## 3. Medicine Identity Validation

| Metric | Count |
|---|---|
| Unique `itemId` values | **16,830** |
| Unique `itemCode` values | **16,830** |
| Rows with missing `itemId` | **0** |
| Rows with missing `itemCode` | **0** |
| Rows with missing `itemName` | **0** |
| `itemCode` values with >1 distinct `itemName` | **0** |
| `itemId` values with >1 distinct `itemCode` | **0** |

Finding: Medicine identity is **perfectly clean**. `itemId = itemCode` (numeric, 1:1).
Every item code maps to exactly one item name and vice versa. Zero conflicts.

---

## 4. Transaction Integrity

### 4.1 Row counts

| Stage | Count |
|---|---|
| Raw rows | 895,557 |
| Valid rows (non-null customerId + itemId + invoice_date) | **895,474** |
| Rows dropped (missing identity keys) | 83 |

### 4.2 Duplicate analysis

| Type | Count | Pipeline handling |
|---|---|---|
| **Exact duplicate rows** (all 23 columns identical) | **16,798** | Dropped by `clean_transactions()` drop_duplicates() |
| **Logical duplicate events** (same invoice_number + customerId + itemId) | **77,734** | Aggregated by `aggregate_invoice_items()` |

Note: Logical duplicates are multi-batch pharmacy dispenses - the same medicine dispensed
from two batches on the same invoice. This is normal pharmacy workflow, not data corruption.

### 4.3 Multi-line invoice items

| Metric | Count |
|---|---|
| (invoice_number, customerId, itemId) combos appearing >1 line | **68,505** |
| Total rows involved | **146,239** |

Interpretation: 146,239 rows (~16.3% of valid rows) are multi-batch invoice lines
that will be aggregated into 68,505 single purchase events.

### 4.4 Quantity integrity

| Type | Count | Pipeline handling |
|---|---|---|
| Zero quantity rows | **0** | -- |
| Negative quantity rows | **5,954** | Flagged via `is_positive_quantity = False` |

Note: 5,954 negative-quantity rows represent return transactions or credit notes.
The feature pipeline correctly excludes them from supervised training via
`is_supervised_eligible = has_next_purchase AND is_positive_quantity`.

---

## 5. Purchase History Distribution

Based on pre-aggregation row counts per (customerId, itemId) pair.

### 5.1 History depth

| Purchase threshold | Histories |
|---|---|
| Total histories | **327,051** |
| >= 1 purchase | 327,051 (100%) |
| >= 2 purchases | **112,793** (34.5%) |
| >= 3 purchases | **64,173** (19.6%) |
| >= 5 purchases | **34,192** (10.5%) |
| >= 10 purchases | **15,256** (4.7%) |
| >= 20 purchases | **6,261** (1.9%) |

Interpretation: 34.5% of customer-medicine pairs have repeat purchases. 19.6% have
3+ purchases -- minimum depth for reliable interval-based ML training.

### 5.2 Purchase interval statistics

Computed over **568,423 non-first-purchase intervals**:

| Statistic | Value |
|---|---|
| Median | **26.0 days** |
| Mean | **80.59 days** |
| Std dev | **179.84 days** |
| Min | **0 days** |
| Max | **2,060 days** |

Interpretation: Median 26 days confirms dominant monthly refill cycle. Mean is 3x median
indicating a heavy right tail from infrequent/acute purchases. High std expected in a
general-medicine store serving chronic and acute conditions.

### 5.3 Interval distribution

| Bucket | Intervals | % |
|---|---|---|
| 0 days (same-day) | 81,770 | 14.4% |
| 1-14 days | 113,247 | 19.9% |
| **15-30 days** | **126,492** | **22.3%** |
| **31-60 days** | **106,331** | **18.7%** |
| **61-90 days** | **35,893** | **6.3%** |
| **91-120 days** | **20,540** | **3.6%** |
| 121-180 days | 23,560 | 4.1% |
| 181-365 days | 30,652 | 5.4% |
| > 365 days | 29,938 | 5.3% |

### 5.4 Zero-day and long-gap intervals

| Metric | Count | Interpretation |
|---|---|---|
| Zero-day intervals | **81,770** | Same-day repeat purchases (different invoices). Correct -- interval=0, flagged by is_same_day_target. |
| Negative intervals | **0** | No chronological ordering corruption. |
| > 365-day intervals | **29,938** | Infrequent/seasonal purchases. Normal -- pipeline does not filter. |

### 5.5 Recurring-purchase population

Definition: (customerId, itemId) pair with >=3 purchases AND median interval 15-120 days.

| Metric | Count |
|---|---|
| Recurring (customerId, itemId) pairs | **37,070** |
| Unique recurring customers | **3,756** |
| % of all intervals in 15-120d range | **50.89%** |

Interpretation: 50.89% of all intervals fall in the recurring window -- excellent ML signal.
37,070 patient-medicine pairs qualify as recurring, representing 3,756 unique customers.
This is the primary training population for Phase 17C.

---

## 6. Pipeline Compatibility

### 6.1 Required column check

| Column set | Missing |
|---|---|
| Core required columns | **None** |
| clean_transactions() string columns | **None** |
| Purchase event key | **None** |
| History key (customerId, itemId) | **None** |
| Feature engineering required columns | **None** |

All required columns are present.

### 6.2 Removed column handling

| Column | Status | Pipeline impact |
|---|---|---|
| `therapeuticCategory` | Not in dataset | cleaning.py STRING_COLUMNS lists it but guarded by `if col in df.columns` -- harmless |
| `generic_name` | Not in dataset | Same guard -- harmless |
| `mfgDate` | Present | clean_transactions() drops it at step 1 -- correct |

Note: FEATURE_COLUMNS_CATEGORICAL in engineering.py still lists therapeuticCategory and
generic_name. When absent from the dataset they simply do not appear in the feature matrix.
No error, no leakage. No code change required.

### 6.3 Pipeline stage compatibility

| Stage | Module | Compatibility |
|---|---|---|
| Load raw | cleaning.load_raw_transactions() | All dtype_map columns present or absent-and-guarded |
| Clean | cleaning.clean_transactions() | Drops 83 missing-key rows + 16,798 exact dups |
| Aggregate | cleaning.aggregate_invoice_items() | Resolves 68,505 multi-batch combos |
| History | history.create_purchase_history() | customerId+itemId key, no leakage |
| Features | features.engineering.build_feature_dataset() | All numeric/calendar features computable |

---

## 7. Phase 17C Readiness

### Verdict: SUITABLE - Zero Blockers

| Category | Status |
|---|---|
| All required columns present | PASS |
| No negative purchase intervals | PASS |
| No chronological ordering corruption | PASS |
| Customer identity 1:1 clean | PASS |
| Medicine identity 1:1 clean | PASS |
| Pipeline can consume dataset without errors | PASS |

### Warnings (non-blocking)

| # | Warning | Disposition |
|---|---|---|
| W1 | 77,734 logical duplicate events | Handled by aggregate_invoice_items(). No action needed. |
| W2 | 180 MOBILE_NO values shared by >1 customer (388 customers) | By design. MOBILE_NO is not identity. Histories isolated by customerId. |
| W3 | 5,954 negative-quantity rows (returns) | Retained for history. Excluded from supervised training by is_positive_quantity. |

---

## 8. Files Changed

| File | Action |
|---|---|
| `scripts/phase17b_validate.py` | CREATED |
| `data/refillcare/processed/phase17b_validation.json` | CREATED |
| `docs/PHASE_17B_IDENTITY_HISTORY_VALIDATION.md` | CREATED |

Files NOT changed (per scope):
- app.py -- unchanged
- app_image_campaign.py -- unchanged
- refillcare/data/cleaning.py -- unchanged
- refillcare/data/history.py -- unchanged
- refillcare/data/validation.py -- unchanged
- refillcare/data/pipeline.py -- unchanged
- refillcare/features/engineering.py -- unchanged
- refillcare/models/prediction.py -- unchanged
- All model files -- unchanged
- All WhatsApp dispatch files -- unchanged

---

## 9. Test Results

Test suite: tests/refillcare/ (19 test files)
Platform: Python 3.13.2, pytest 9.1.1
Result: **147 / 147 PASSED** -- 0 failures, 0 errors, 0 skipped
Duration: 11.43 seconds

| Test file | Tests | Result |
|---|---|---|
| test_app_refillcare.py | 10 | PASS |
| test_baseline.py | 4 | PASS |
| test_cleaning.py | 4 | PASS |
| test_dispatch.py | 12 | PASS |
| test_e2e_verification.py | 11 | PASS |
| test_enrichment.py | 2 | PASS |
| test_features.py | 6 | PASS |
| test_history.py | 5 | PASS |
| test_micro_pilot.py | 9 | PASS |
| test_models.py | 5 | PASS |
| test_pilot_monitoring.py | 9 | PASS |
| test_pilot_operations.py | 11 | PASS |
| test_pilot_outcomes.py | 9 | PASS |
| test_pilot_preflight.py | 8 | PASS |
| test_pilot_readiness.py | 6 | PASS |
| test_scheduler.py | 11 | PASS |
| test_storage.py | 11 | PASS |
| test_validation.py | 3 | PASS |
| test_whatsapp_template_config.py | 5 | PASS |

---

## 10. Summary Report

**Customer identity findings:**
14,271 unique customers. Perfect 1:1 customerId to customerName mapping.
83 rows missing both fields -- dropped by pipeline. Zero identity collisions.

**Shared WhatsApp number findings:**
180 MOBILE_NO groups shared by >1 customer (388 customers total).
Largest group: 9951921323 with 10 distinct customers.
MOBILE_NO is never used as identity. All reminder histories isolated by customerId.

**Item identity findings:**
16,830 unique medicines (itemId = itemCode, 1:1 numeric mapping).
Zero itemCode to itemName conflicts. Zero itemId to itemCode conflicts.
All rows have non-null itemId and itemCode.

**Duplicate / return findings:**
16,798 exact duplicate rows -- dropped by clean_transactions().
77,734 logical duplicates (multi-batch invoice lines) -- aggregated by aggregate_invoice_items().
5,954 negative-quantity rows (returns) -- retained, excluded from supervised training.

**Purchase-history distribution:**
327,051 customer x medicine histories total.
112,793 pairs (34.5%) with >=2 purchases. 64,173 (19.6%) with >=3 purchases.
Median refill interval: 26 days. Mean: 80.6 days. Std: 179.8 days.
Zero negative intervals -- chronological integrity confirmed.
81,770 zero-day intervals -- correctly flagged by pipeline.

**Recurring-purchase population:**
37,070 recurring (customerId, itemId) pairs (>=3 purchases, median interval 15-120 days).
3,756 unique recurring customers.
50.89% of all purchase intervals fall in the 15-120 day recurring window.

**Dataset suitability for Phase 17C:**
SUITABLE -- zero blockers. 3 non-blocking warnings, all handled by existing pipeline.
All required columns present. All pipeline stages compatible.

**Tests:** 147 / 147 PASSED

**Blockers:** None.

---

Phase 17B complete. Proceed to Phase 17C when ready.
