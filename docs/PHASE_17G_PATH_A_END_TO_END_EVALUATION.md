# Phase 17G — Path A End-to-End Offline Evaluation

**Date:** 2026-09-18  
**Status:** Offline evaluation only — **no strategy promotion**, production routing/model/WhatsApp **unchanged**  
**Script:** `scripts/phase17g_path_a_end_to_end_evaluation.py`  
**Raw JSON:** `data/refillcare/processed/phase17g_path_a_evaluation_results.json`  
**Scope:** Path A only (`purchase_count >= 6`). Path B **not** implemented.  
**Feature flag:** `USE_PATH_A_CLASSIFIER_V17F` remains **OFF** (classifier called explicitly offline).

---

## 1. Purpose

Evaluate the complete Path A flow using the Phase 17F classifier:

| 17F class | Predictor |
|---|---|
| **HIGH** | Personal historical median |
| **MEDIUM** | Five safety strategies A–E (production XGBoost, read-only) |
| **UNSTABLE** | No prediction / rejected |

Compare against baseline `hybrid_routing_v17d` under the same MEDIUM safety sweeps. **Do not select or promote a strategy.**

---

## 2. Controls

| Control | Value |
|---|---|
| Test set | Same temporal holdout: `test.parquet`, `target > 0` |
| Universe | **7,488** rows |
| Dates | **2026-07-01 → 2026-08-30** |
| Path A rows | **5,155** (`purchase_count >= 6`) |
| Classifier | `path_a_classifier_v17f` (offline; flag **OFF**) |
| Baseline | `hybrid_routing_v17d` |
| MEDIUM model | Production `refill_model.joblib` (`reg:squarederror`), **read-only** |
| HIGH predictor | Personal historical median |
| Catastrophic | abs error **> 30 days** |
| WhatsApp | **0** |
| Retrain / routing change / Path B | **None** |

### MEDIUM strategies

| ID | Rule |
|---|---|
| **A** | Current XGBoost (no gate) |
| **B** | Reject when prediction > 60 days |
| **C** | Reject when prediction > 2 × historical median |
| **D** | Fallback to historical median when prediction > 2 × median (still accepted) |
| **E** | Combined B + C |

---

## 3. Cohort counts

| Class | Path A `v17f` | Baseline `v17d` | Δ |
|---|---:|---:|---:|
| HIGH | **669** | 1,819 | −1,150 |
| MEDIUM | **1,540** | 714 | +826 |
| UNSTABLE | **2,946** | 2,622 | +324 |

17F moves many contaminated “HIGH” rows out and recovers variable recurrers into MEDIUM (often previously UNSTABLE under Drift ≤ 10).

---

## 4. HIGH performance (17F)

HIGH is identical across A–E (personal median only):

| Metric | Value |
|---|---:|
| N | 669 |
| MAE | **7.26** |
| MedAE | 4.00 |
| RMSE | 11.74 |
| ±1d | 23.62% |
| ±3d | 46.19% |
| ±7d | **69.21%** |
| Total abs error | 4,854.5 |
| Catastrophic count | 13 (1.94%) |
| Catastrophic abs error | 573.0 (11.8% of HIGH abs error) |
| Mean bias | +2.65 |

Baseline v17d HIGH (n=1,819): MAE **7.39**, ±7d **66.74%** — slightly worse than the stricter 17F HIGH, as expected.

---

## 5. MEDIUM performance (17F) — strategies A–E

| Strategy | Accepted | MAE | MedAE | RMSE | ±1d | ±3d | ±7d | Total \|err\| | Cat n | Cat \|err\| | Cat share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A** Current XGB | 1,540 | 14.85 | 10.78 | 21.39 | 5.32% | 13.64% | 34.55% | 22,863.56 | 162 | 7,824.94 | 34.2% |
| **B** Reject >60 | 1,453 | 12.34 | 10.08 | 15.66 | 5.64% | 14.38% | 36.48% | 17,930.04 | 89 | 3,185.94 | 17.8% |
| **C** Reject >2× hist | 1,440 | 13.32 | 10.18 | 18.30 | 5.62% | 14.37% | 36.53% | 19,183.13 | 117 | 5,133.02 | 26.8% |
| **D** Fallback hist | 1,540 | 13.09 | 9.98 | 17.99 | 5.97% | 15.32% | 37.01% | 20,162.13 | 120 | 5,238.02 | 26.0% |
| **E** B+C | 1,387 | **11.88** | **9.64** | **15.10** | 5.84% | 14.92% | **37.85%** | **16,473.21** | **74** | **2,629.21** | **16.0%** |

---

## 6. Overall Path A performance (17F) — A–E

Accepted = all HIGH + MEDIUM rows kept by the safety strategy. UNSTABLE always rejected.

| Strategy | Accepted | Rejected | Cov Path A | Cov test | MAE | MedAE | RMSE | ±1d | ±3d | ±7d | Total \|err\| | Cat n | Cat \|err\| | Cat share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A** | 2,209 | 2,946 | 42.85% | 29.50% | 12.55 | 8.09 | 18.99 | 10.86% | 23.49% | 45.04% | 27,718.06 | 175 | 8,397.94 | 30.3% |
| **B** | 2,122 | 3,033 | 41.16% | 28.34% | 10.74 | 7.75 | 14.54 | 11.31% | 24.41% | 46.80% | 22,784.54 | 102 | 3,758.94 | 16.5% |
| **C** | 2,109 | 3,046 | 40.91% | 28.17% | 11.40 | 7.73 | 16.50 | 11.33% | 24.47% | 46.89% | 24,037.63 | 130 | 5,706.02 | 23.7% |
| **D** | 2,209 | 2,946 | 42.85% | 29.50% | 11.32 | 7.84 | 16.35 | 11.32% | 24.67% | 46.76% | 25,016.63 | 133 | 5,811.02 | 23.2% |
| **E** | 2,056 | 3,099 | 39.88% | 27.46% | **10.37** | **7.50** | **14.09** | **11.62%** | **25.10%** | **48.05%** | **21,327.71** | **87** | **3,202.21** | **15.0%** |

---

## 7. Comparison vs `hybrid_routing_v17d`

Same Path A universe and MEDIUM safety sweeps; only the **classifier** differs.

### Overall accepted metrics

| Setup | Accepted | Cov test | MAE | ±7d | Cat \|err\| |
|---|---:|---:|---:|---:|---:|
| v17d + A (current hybrid default) | **2,533** | **33.83%** | **10.39** | **55.11%** | 6,840.51 |
| v17d + E | 2,419 | 32.31% | 8.89 | 57.71% | 2,960.43 |
| 17F + A | 2,209 | 29.50% | 12.55 | 45.04% | 8,397.94 |
| 17F + E | 2,056 | 27.46% | 10.37 | 48.05% | 3,202.21 |

### MEDIUM-only

| Setup | MEDIUM accepted | MAE | ±7d | Cat \|err\| |
|---|---:|---:|---:|---:|
| v17d + A | 714 | 18.05 | 25.49% | (see 17G JSON MEDIUM block) |
| v17d + E | 600 | 13.46 | 30.33% | (see 17G JSON MEDIUM block) |
| 17F + A | 1,540 | 14.85 | 34.55% | 7,824.94 |
| 17F + E | 1,387 | 11.88 | 37.85% | 2,629.21 |

Baseline v17d+A MEDIUM matches Phase 17D (MAE 18.05 / ±7d 25.49%).

### How to read this (no promotion)

- **17F HIGH is cleaner** (±7d 69.2% on 669 rows vs 66.7% on 1,819).
- **17F MEDIUM is larger** (recovers Drift 10–15 variable recurrers) and **better than v17d MEDIUM** on MAE/±7d for the same safety strategy, but still far behind HIGH.
- **Overall Path A ±7d looks worse under 17F** at similar MAE bands because the accepted mix shifts toward more MEDIUM and fewer easy HIGH rows — classification purity ≠ automatic overall lift when MEDIUM still uses production squared-error XGB.
- Safety gates **B/D/E** still cut catastrophic mass sharply on both classifiers; **E** is strongest on error mass, **D** keeps full MEDIUM coverage.

**No strategy is selected or promoted in this phase.**

---

## 8. Variable recurring inspection

### Canonical pattern `30,60,45,30,90,30,60,30,90`

| System | Class | Notes |
|---|---|---|
| **17F Path A** | **MEDIUM** | NormMAD 0.33, Drift 15, max/med 2.0 — recovered |
| **v17d hybrid** | **UNSTABLE** | Rejected by Drift > 10 |

Under 17F this customer would enter the MEDIUM safety comparison (A–E), not be dropped.

### Real test examples (17F MEDIUM, Drift 10–15; often v17d UNSTABLE)

| Customer / item | Hist med | XGB | Actual | \|err\| median | \|err\| XGB | v17d → 17F |
|---|---:|---:|---:|---:|---:|---|
| A NARSIMHA RAO / 11171 | 18.5 | 29.8 | 5 | 13.5 | 24.8 | UNSTABLE → MEDIUM |
| AMAN / 965 | 45.0 | 55.2 | 24 | 21.0 | 31.2 | UNSTABLE → MEDIUM |
| ANIL KUMAR K / 1329 | 39.0 | 39.9 | 22 | 17.0 | 17.9 | UNSTABLE → MEDIUM |

These confirm 17F admits the intended variable-recurring band; production XGB on those rows remains noisy / late-biased — reinforcing why MEDIUM safety gates matter, without choosing one yet.

---

## 9. UNSTABLE

| Classifier | UNSTABLE n | Predictions |
|---|---:|---|
| 17F | 2,946 | **0** |
| v17d | 2,622 | **0** |

No automated refill date; not scored in accepted metrics.

---

## 10. Artifact safety

| Check | Result |
|---|---|
| `refill_model.joblib` mtime unchanged | **Yes** (`1789645154.9395924`) |
| Production routing replaced | **No** |
| Path A feature flag | **OFF** |
| WhatsApp messages | **0** |
| Retrain | **No** |
| Reminder stages modified | **No** |
| Path B implemented | **No** |
| Strategy promoted | **No** |

---

## 11. Stop point

Phase 17G stops after this offline evaluation.

**Not done:** selecting/promoting a MEDIUM safety strategy, enabling the Path A flag in production prediction, Path B, model overwrite, or WhatsApp changes.
