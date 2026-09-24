# Data Audit Report: Packing, Quantity, and Unit Computability

## Executive Summary
This read-only data audit inspects the canonical purchase history dataset (`data/refillcare/processed/purchase_history.parquet`, 817,804 total records across 5.8 years) to evaluate the completeness, parsability, consistency, and depth of the `packing` and `quantity` fields.

---

## 1. Availability of `packing` in Dataset
- **Status:** **Available and present** as a standard column across all transaction and history datasets.
- **Missing / Null Count:** Only **45 rows** out of 817,804 are null/blank (**0.01%**).
- **Non-null coverage:** **99.99%** (817,759 rows).

---

## 2. Usability & Format Breakdown

| Pack Format Category | Definition / Pattern | Rows | % of Dataset |
|---|---|:---:|:---:|
| **Solid Multi-pack (`SOLID_MULT`)** | Discrete solid units (e.g., `1X10`, `1X15`, `1X30`, `1X20`) | 587,912 | 71.89% |
| **Volume / Weight (`VOLUME_WEIGHT`)** | Liquid/ointment sizes (e.g., `200ML`, `10ML`, `400GM`, `15GM`) | 178,573 | 21.84% |
| **Unit Count (`UNIT_COUNT`)** | Discrete piece labels (e.g., `1PC`, `1 TAB`, `1 CAP`, `1 BOTTLE`) | 18,872 | 2.31% |
| **Plain Integer (`INTEGER`)** | Numeric count strings (e.g., `1`, `10`, `15`) | 6,657 | 0.81% |
| **Unparseable / Multi-dimensional** | Complex/nested formats (e.g., `10X4.4GM`, `1X5X2ML`, `200MDI`) | 25,745 | 3.15% |
| **Null / Missing** | Null or empty string | 45 | 0.01% |
| **Total** | | **817,804** | **100.00%** |

- **Total Usable Packing Rows:** **792,014 rows (96.85%)** can be programmatically parsed to numeric units/volumes.
- **Discrete Solid Formats (Tablets / Capsules / Pieces):** **613,441 rows (75.01%)**.

---

## 3. Examples of Distinct Packing Formats

| Format Type | Examples | Typical Medication Class |
|---|---|---|
| `1X10` | `1X10`, `1x10` | Antihypertensives, Statins, Antibiotics |
| `1X15` | `1X15`, `1x15` | Oral Antidiabetics (e.g., Metformin + Glimepiride) |
| `1X30` | `1X30`, `1x30` | Monthly chronic maintenance regimens |
| `1X20` | `1X20`, `1x20` | Specialized gastro & cardiac therapies |
| `1X14` / `1X7` | `1X14`, `1X7` | Weekly/fortnightly blister strips |
| `1X120` / `1X100` | `1X120`, `1X100` | Bulk institutional / large container bottles |
| `1PC` / `1` | `1PC`, `1 PIECE`, `1` | Inhalers, Devices, Kits, Single Injections |
| `200ML` / `100ML` | `200ML`, `100ML`, `60ML` | Syrups, Suspensions, Tonics |
| `10ML` / `15ML` | `10ML`, `15ML`, `5ML` | Eye / Ear drops |
| `400GM` / `100GM` | `400GM`, `100GM`, `75GM` | Protein powders, Topical creams |

---

## 4. Total Units Computability
Total units is defined as:
$$\text{Total Units} = \text{Quantity} \times \text{Units per Pack}$$

- **Criteria:** Parsable packing value AND valid positive numeric quantity (`quantity > 0`).
- **Rows where Total Units can be calculated:** **786,309 rows (96.15% of all transactions)**.
- **Rows where Discrete Solid Units (Tablets/Capsules) can be calculated:** **613,441 rows (75.01%)**.

---

## 5. Customer + Item History Depth (`customerId + itemId`)

| Cohort | Count | % of All Pairs |
|---|:---:|:---:|
| **Total unique `(customerId, itemId)` pairs in dataset** | 326,627 | 100.00% |
| **Pairs with $\ge 2$ total purchases** | 102,169 | 31.28% |
| **Pairs with $\ge 2$ usable parsed packing & positive quantity** | **96,646** | **29.59%** |
| **Pairs with $\ge 2$ discrete solid (tablets/capsules) purchases** | **72,752** | **22.27%** |

---

## 6. Examples of `1X15` + Quantity 4 (60 Tablets)
Across the dataset, there are **9,132 transactions** where a patient bought **quantity = 4** of a **1X15** pack:

| Customer | Medication | Packing | Quantity | Units/Pack | Total Units | Invoice Date |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `BALARAJU` | `GLYCOMET GP1 TAB` | `1X15` | 4 | 15 | **60 tabs** | 2026-04-28 |
| `BALARAJU` | `GLYCOMET GP1 TAB` | `1X15` | 4 | 15 | **60 tabs** | 2026-05-23 |
| `BALARAJU` | `GLYCOMET GP1 TAB` | `1X15` | 4 | 15 | **60 tabs** | 2026-06-26 |
| `9908043227` | `NEXPRO 40MG TAB` | `1X15` | 4 | 15 | **60 tabs** | 2026-02-19 |
| `9908043227` | `CLOPITAB 75MG TAB` | `1X15` | 4 | 15 | **60 tabs** | 2026-01-24 |
| `A ANU` | `AMLOKIND 5MG TAB` | `1X15` | 4 | 15 | **60 tabs** | 2023-11-25 |
| `A BAJI RAO` | `SHELCAL 500MG TAB` | `1X15` | 4 | 15 | **60 tabs** | 2023-09-15 |
| `A BALRAM MURTHY` | `PINOM 10MG TAB` | `1X15` | 4 | 15 | **60 tabs** | 2021-12-18 |

---

## 7. Ambiguous & Unparseable Packing Formats
A total of **25,745 rows (3.15%)** contain ambiguous, multi-dimensional, or special symbol formats:

1. **Multi-dimensional / Nested packs:**
   - `10X4.4GM` (2,291 rows)
   - `5X3ML` (1,845 rows)
   - `1X5X2ML` (1,844 rows)
   - `1X5X3ML` (467 rows)
2. **Metered Dose Inhalers & Special Devices:**
   - `200MDI` (1,309 rows)
   - `120MDI` (1,305 rows)
   - `1KIT` (885 rows)
   - `1PFS` (prefilled syringes, 271 rows)
3. **Punctuation / Typographical artifacts:**
   - `1X14\`` (774 rows)
   - `100\`S` (335 rows)
   - `1X10\`.` (247 rows)
   - `1X21\`` (192 rows)

*(Note: Most typographical artifacts like backticks or trailing dots can be cleaned with simple regex sanitization).*

---

## 8. Item-Level Packing Consistency (`itemId`)
- **Total distinct item IDs in dataset:** **16,830**
- **Items with exactly 1 consistent packing:** **16,820 items (99.94%)**
- **Items with >1 distinct packing representation:** **Only 10 items (0.06%)**
  - Most of these 10 are simple string formatting variants (e.g. `10` vs `1X10` for Item 33655) or manufacturer packaging redesigns (e.g. `1X15` vs `1X30` for Sobisis).

---

## Conclusion
1. **Data Completeness:** Packing data is present in **99.99%** of records.
2. **Discrete Parsability:** **75.01%** of transactions represent standard discrete oral solids (strips of tablets/capsules) and **96.15%** can compute total numeric units.
3. **Item Consistency:** Item-level packing is **99.94% consistent**, making pack size a highly reliable attribute for calculating total purchased units.
