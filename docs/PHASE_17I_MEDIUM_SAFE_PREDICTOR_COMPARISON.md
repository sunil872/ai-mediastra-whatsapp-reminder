# Phase 17I — MEDIUM-SAFE Predictor Comparison

**Date:** 2026-09-18  
**Status:** READ-ONLY offline evaluation — **no promotion / implementation**  
**Script:** `scripts/phase17i_medium_safe_predictor_comparison.py`  
**Raw JSON:** `data/refillcare/processed/phase17i_medium_safe_predictor_results.json`  
**Scope:** Path A only (`purchase_count >= 6`). Path B **not** implemented.  
**Feature flag:** `USE_PATH_A_CLASSIFIER_V17F` remains **OFF**.

---

## 1. Purpose

Answer the Phase 17H follow-up:

> On the **C6-like MEDIUM-SAFE** pocket, does **personal historical median** beat **production XGB**, and is the gap large enough to justify a future **SAFE = median** design?

Also check a **50/50 blend**, and evaluate any **existing quantile-XGB artifact** (none found — no new training).

---

## 2. Controls

| Control | Value |
|---|---|
| SAFE definition | 17F **MEDIUM** AND `pred ≤ 1.5 × hist_median` (**C6**) AND Strategy **E** (`pred ≤ 60` AND `pred ≤ 2 × hist`) |
| RISK residual | All other 17F MEDIUM rows |
| HIGH reference | 17F HIGH, personal median (also contrast XGB/blend) |
| Predictors | (1) personal median (2) production XGB (3) blend 50/50 |
| Quantile XGB artifact | **None** under `data/refillcare/processed/models/` (only `refill_model.joblib`) |
| Test | 2026-07-01 → 2026-08-30, `target>0`, N=7,488 |
| Validation | 2026-05-01 → 2026-06-30, `target>0`, N=12,767 |
| Retrain / routing / WhatsApp / Path B | **None** |

Identical rows compared across predictors within each cohort.

---

## 3. Cohort sizes

| Split | Path A | HIGH | MEDIUM | MEDIUM-SAFE (C6∧E) | MEDIUM-RISK |
|---|---:|---:|---:|---:|---:|
| **Test** | 5,155 | 669 | 1,540 | **1,091** | 449 |
| **Validation** | 8,658 | 1,277 | 2,896 | **1,962** | 934 |

SAFE coverage (test): **14.6%** of full test / **21.2%** of Path A.

---

## 4. MEDIUM-SAFE predictor comparison

### 4.1 Test holdout (n = 1,091)

| Predictor | MAE ↓ | MedAE ↓ | RMSE ↓ | ±1d ↑ | ±3d ↑ | ±7d ↑ | Cat % ↓ | Cat n | Mean bias | Total \|err\| ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Personal median** | **7.89** | **5.00** | **11.20** | **19.89%** | **38.59%** | **60.31%** | **1.74%** | **19** | **+4.20** | **8,612.5** |
| Blend 50/50 | 8.99 | 6.22 | 12.15 | 7.88% | 26.86% | 54.08% | 2.20% | 24 | +6.49 | 9,813.5 |
| Production XGB | 10.65 | 8.14 | 13.77 | 6.69% | 17.05% | 43.08% | 4.03% | 44 | +8.79 | 11,623.7 |

**Head-to-head (same SAFE rows):** median closer on **71.9%** of rows; ΔMAE **−2.76d**; Δ±7d **+17.2 pp** vs XGB.

### 4.2 Validation window (n = 1,962) — out-of-test confirmation

| Predictor | MAE ↓ | MedAE ↓ | RMSE ↓ | ±1d ↑ | ±3d ↑ | ±7d ↑ | Cat % ↓ | Mean bias | Total \|err\| ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Personal median** | **8.48** | **5.00** | **13.26** | **19.06%** | **39.45%** | **61.98%** | 3.92% | **−0.07** | **16,636.0** |
| Blend 50/50 | 9.32 | 6.13 | 13.53 | 8.56% | 27.32% | 54.49% | **3.67%** | +2.70 | 18,277.4 |
| Production XGB | 10.89 | 8.42 | 14.68 | 6.57% | 19.67% | 41.39% | 4.49% | +5.47 | 21,368.7 |

**Head-to-head:** median closer **67.3%**; ΔMAE **−2.41d**; Δ±7d **+20.6 pp**.

Median wins almost every metric on both windows. Blend is intermediate. Only nuance: on validation, blend edges catastrophic % (3.67% vs median 3.92%) — not enough to overturn the overall median lead.

---

## 5. HIGH reference (personal median is already the design)

### Test HIGH (n=669)

| Predictor | MAE | ±7d | Cat % |
|---|---:|---:|---:|
| Personal median | **7.26** | **69.2%** | **1.9%** |
| Blend | 8.80 | 59.5% | 3.4% |
| Production XGB | 11.60 | 46.6% | 5.8% |

SAFE median (±7d **60.3%**) approaches HIGH (±7d **69.2%**) much more closely than SAFE XGB (±7d **43%**).

---

## 6. MEDIUM-RISK residual

### Test RISK (n=449)

| Predictor | MAE | ±7d | Cat % | Mean bias |
|---|---:|---:|---:|---:|
| Personal median | 10.71 | 49.0% | 6.0% | +3.7 |
| Blend | 16.44 | 27.2% | 14.0% | +13.9 |
| Production XGB | **25.03** | **13.8%** | **26.3%** | **+24.2** |

RISK is where XGB fails badly. Median is still imperfect but far safer. Supports treating RISK as **no auto / review**, not “keep XGB.”

---

## 7. Pattern slices (MEDIUM-SAFE, test)

| Slice | N | Median MAE / ±7d | XGB MAE / ±7d | Blend MAE / ±7d |
|---|---:|---|---|---|
| Deeper P≥10 | 1,049 | **7.78 / 61.2%** | 10.38 / 44.4% | 8.79 / 55.3% |
| Shallow P 6–9 | 42 | **10.68 / 38.1%** | 17.54 / 9.5% | 14.03 / 23.8% |
| Pred inflation 1.25–1.5× | 394 | **7.31 / 62.4%** | 12.36 / 32.5% | 9.29 / 51.0% |
| Variable-like Drift 10–15 | 16 | **10.28 / 50.0%** | 13.21 / 31.3% | 11.72 / 43.8% |
| Collapse (actual < ½ hist) | 179 | 20.11 / **0%** | 23.57 / **0%** | 21.84 / **0%** |

### Canonical `30,60,45,30,90,30,60,30,90`

- Classifies as **MEDIUM** (Drift 15, NormMAD 0.33).  
- Stays **SAFE** if XGB ≤ 1.5× hist (and E); becomes **RISK** if XGB inflates.  
- On real variable-like SAFE rows, **median still beats XGB**.

### Collapse intervals

All predictors fail (±7d **0%**). These are structural surprises (next fill much earlier than cadence). Not fixed by switching to median — argue for RISK / monitoring, not for keeping XGB.

---

## 8. Best predictor by metric (MEDIUM-SAFE)

| Metric | Test winner | Validation winner |
|---|---|---|
| MAE | **Personal median** | **Personal median** |
| MedAE | **Personal median** | **Personal median** |
| RMSE | **Personal median** | **Personal median** |
| ±7d | **Personal median** | **Personal median** |
| Catastrophic % | **Personal median** | Blend (slight) |
| \|Mean bias\| | **Personal median** | **Personal median** |
| Total abs error | **Personal median** | **Personal median** |

---

## 9. Answers to the key questions

### Does MEDIUM-SAFE perform better with personal median than production XGB?

**Yes — consistently on test and validation.**

| Window | ΔMAE (median − XGB) | Δ±7d (pp) | Median closer |
|---|---:|---:|---:|
| Test | **−2.76d** | **+17.2** | 71.9% |
| Validation | **−2.41d** | **+20.6** | 67.3% |

### Is the improvement large enough to justify a future SAFE=median design?

**Yes (as an offline design recommendation — not promoted here).**  
Moving SAFE from XGB (~43% ±7d) to median (~60–62% ±7d) is a **large operational gap**, and SAFE median lands near HIGH quality. Validation reproduces the same ranking.

### Does blending help?

**Partially.** Blend sits between median and XGB on almost every metric. It never beats median on MAE/±7d in either window. Not preferred over pure median for SAFE.

### Is MEDIUM-SAFE suitable for automatic prediction?

**Conditionally yes — if the predictor is personal median (future design), not production XGB.**  
Caveats: shallow SAFE (P 6–9) and especially **collapse** rows remain weak; RISK must stay out of auto send.

### Quantile XGB?

**Not evaluated** — no saved artifact exists; this phase did **not** train one.

### Is another Path A investigation needed?

**Light follow-up only (optional before implementation):**  
define end-to-end Path A policy draft offline:

`HIGH → median` · `MEDIUM-SAFE (C6∧E) → median` · `MEDIUM-RISK / UNSTABLE → no auto`

plus collapse/shallow handling rules — still **no production enablement** until explicitly approved.

---

## 10. Artifact safety

| Check | Result |
|---|---|
| `refill_model.joblib` unchanged | **Yes** (`1789645154.9395924`) |
| Retrain | **No** |
| Production routing / scheduler / WhatsApp | **Unchanged** |
| Feature flag | **OFF** |
| Path B | **Not implemented** |
| Strategy promoted | **No** |
