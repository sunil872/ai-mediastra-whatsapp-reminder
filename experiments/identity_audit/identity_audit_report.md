# Customer-Item Identity Impact Audit Report (READ-ONLY)

**Date of Audit:** 2026-09-23T19:14:58.467953  
**Status:** COMPLETE (READ-ONLY — Zero Production Mutations)  
**Target Repository:** `ai-mediastra-whatsapp-reminder`

---

## 1. Executive Summary & Identity Definitions

This audit compares the existing production customer-item entity against the proposed multi-factor composite identity:

| Dimension | Current Production Entity | Proposed Audit Entity |
| :--- | :--- | :--- |
| **Formula** | `customerId + itemId` | `normalized_store_id + normalized_phone + normalized_customer_name + normalized_item_id` |
| **Phone Used** | Delivery routing only (`MOBILE_NO`) | Primary disambiguator component |
| **Name Normalization** | Raw / Unmodified in key | Uppercase, whitespace-collapsed, punctuation-stripped |
| **Store Scope** | Implicit single branch (`MAIN`) | Explicit `store_id` (defaults to `MAIN`) |

> [!IMPORTANT]
> **Strict Read-Only Guarantee**:
> - Zero changes to `customerId`, `customerName`, `MOBILE_NO`, `itemId`, or historical transactions.
> - Zero production database migrations or ML model re-training.
> - All phone numbers in this report are strictly masked (`******XXXX`).

---

## 2. Dataset Scope & Data Quality

| Metric | Count | Details |
| :--- | :--- | :--- |
| **Total Raw Transactions** | **895,557** | Historical 5.8-year POS dataset |
| **Total Valid Transactions** | **889,520** | Cleaned (positive qty, valid dates) |
| **Date Range** | 2020-12-24 to 2026-08-31 | Longitudinal history |
| **Unique Customer IDs** | 14,271 | `customerId` |
| **Unique Customer Names** | 14,271 | `customerName` |
| **Unique Current Entities** | **326,282** | `customerId + itemId` |
| **Unique Proposed Entities** | **359,963** | `store + phone + name + item` |

### Phone Quality & Shared Phone Metrics
- **Valid WhatsApp Phone Format (`91XXXXXXXXXX`)**: 819,670 rows
- **Missing Phone Rows**: 69,687 rows
- **Invalid Phone Format Rows**: 163 rows
- **Distinct Normalized Phone Numbers**: 30,497
- **Phones Shared by Multiple Customer Names**: **179**
- **Phones Shared by Multiple Customer IDs**: **180**

---

## 3. Split Analysis (1 Current Entity ➔ Multiple Proposed Entities)

Cases where a single `customerId + itemId` splits into multiple proposed keys (e.g. customer changed phone number or name punctuation over time):

- **Affected Current Entities:** **20,897** (6.40% of total)
- **Affected Transactions:** 144,301 rows

### Representative Split Cases (Masked)

| Current Key | Customer ID | Item ID | Tx Count | Proposed Keys | Split Reasons | Sample Phones |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `A . VIJAYALAXMI_14957` | A . VIJAYALAXMI | 14957 | 4 | 2 | different/changing phone | MISSING_PHONE, ******9810 |
| `A . VIJAYALAXMI_2005` | A . VIJAYALAXMI | 2005 | 4 | 2 | different/changing phone | MISSING_PHONE, ******9810 |
| `A . VIJAYALAXMI_4060` | A . VIJAYALAXMI | 4060 | 6 | 2 | different/changing phone | MISSING_PHONE, ******9810 |
| `A . VIJAYALAXMI_441` | A . VIJAYALAXMI | 441 | 3 | 2 | different/changing phone | MISSING_PHONE, ******9810 |
| `A . VIJAYALAXMI_5118` | A . VIJAYALAXMI | 5118 | 6 | 2 | different/changing phone | MISSING_PHONE, ******9810 |

---

## 4. Merge / Collision Analysis (Multiple Current Entities ➔ 1 Proposed Entity)

Cases where multiple `customerId` values share the exact same phone, normalized name, store, and item:

- **Affected Proposed Entities:** **130**
- **Affected Current Entities Collapsed:** **266**
- **Affected Transactions:** 758 rows

### Representative Merge Cases (Masked)

| Proposed Key | Collapsed Entities | Customer IDs | Customer Name | Masked Phone | Item |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `MAIN_917416474176_P JAYA LATHA_7507` | 2 | P. JAYA LATHA, P JAYA LATHA | P JAYA LATHA | ******4176 | 7507 |
| `MAIN_918978652121_P RANJIT_1837` | 2 | P . RANJIT, P RANJIT | P RANJIT | ******2121 | 1837 |
| `MAIN_918978652121_P RANJIT_5830` | 2 | P . RANJIT, P RANJIT | P RANJIT | ******2121 | 5830 |
| `MAIN_918978652121_P RANJIT_8162` | 2 | P . RANJIT, P RANJIT | P RANJIT | ******2121 | 8162 |
| `MAIN_919912367599_M LAKSHMI_1974` | 2 | M.LAKSHMI, M LAKSHMI | M LAKSHMI | ******7599 | 1974 |

---

## 5. Shared Phone Analysis

| Scenario | Definition | Count (Pairs / Tuples) |
| :--- | :--- | :--- |
| **Scenario A** | Same Phone + Different Customer Names + Same Item | **756** |
| **Scenario B** | Same Phone + Same Customer Name + Same Item (Diff Customer IDs) | **10** |
| **Scenario C** | Same Phone + Different Customer IDs + Same Item | **764** |
| **Scenario D** | Same Phone + Same Customer Name + Different Items | **21,573** |

---

## 6. Path A & Path B Eligibility Impact

### Path A Impact (>= 6 Historical Purchases)
- **Current Path A Eligible Entities:** 27,369
- **Proposed Path A Eligible Entities:** 25,468
- **Entities Remaining Eligible:** 23,677
- **Newly Eligible Entities (via Merge):** 49
- **Entities Losing Eligibility (via Split):** 3,692
- **Parity / Unchanged Entities:** 322,541

### Path B Impact (Recency <= 180d, 3M >= 2 or 6M >= 3 Months)
- **Current Path B Eligible Entities:** 1,667
- **Proposed Path B Eligible Entities:** 1,563
- **Entities Remaining Eligible:** 1,548
- **Newly Eligible Entities:** 15
- **Entities Losing Eligibility:** 119
- **Parity / Unchanged Entities:** 326,148

---

## 7. Stability Tier Transitions & Prediction Inputs

### Stability Transitions
```
UNSTABLE -> UNSTABLE      : 326,282 entities
```

### Prediction Input Impact
- **Total Entities Evaluated:** 326,282
- **Entities with Purchase Count Shift:** 21,117 (6.47%)
- **Entities with Last Purchase Date Shift:** 18,227 (5.59%)
- **Entities with Median Interval Shift:** 20,431 (6.26%)
- **Entities with Stability Tier Shift:** 0 (0.00%)

---

## 8. Safety Classifications

Every evaluated entity is classified into one of 4 governance categories:

| Safety Tier | Count | Percentage | Description |
| :--- | :--- | :--- | :--- |
| **SAFE** | **271,834** | 83.31% | Stable store + phone + name + item (1:1 with current entity) |
| **REVIEW** | **19,017** | 5.83% | 1 current entity split into multiple keys (phone/name change) |
| **COLLISION_RISK** | **20** | 0.01% | Multiple current customer IDs collapsed into one key |
| **INSUFFICIENT_DATA**| **35,411** | 10.85% | Missing or invalid phone/name/item |

---

## 9. Recommendations & Next Steps

1. **Keep `customerId` as Canonical Traceability Key**: The proposed `customer_item_key` provides clean multi-branch deduplication, but `customerId` must always be retained for POS reconciliation.
2. **Review Split Cases**: 1:N splits should be surfaced to pharmacy staff rather than automatically fracturing patient history.
3. **Handle Missing Mobile Numbers**: Transactions lacking valid 10-digit Indian phone numbers should be routed to a "Manual Contact" queue rather than discarded.
