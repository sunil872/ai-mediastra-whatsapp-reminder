# Phase 17K — Final Path A Eligibility Architecture Audit

**Date:** 2026-09-18  
**Status:** READ-ONLY architecture audit — **no policy selected / no promotion**  
**Script:** `scripts/phase17k_path_a_eligibility_architecture_audit.py`  
**Raw JSON:** `data/refillcare/processed/phase17k_path_a_eligibility_results.json`  
**Scope:** Path A only (`purchase_count >= 6`). Path B **not** implemented.  
**Feature flag:** `USE_PATH_A_CLASSIFIER_V17F` remains **OFF**.

---

## 1. Question

17J MEDIUM-SAFE eligibility depends on **production XGB**, while the accepted predictor is the **personal historical median**.

Is that XGB-dependent gate:

1. logically coherent with a median predictor, and  
2. safely reproducible at real prediction time (no future leakage)?

This phase compares three architectures. **No approach is selected or promoted.**

---

## 2. Approaches compared

All accepted rows use **personal historical median**.  
Missing/invalid cadence stats demote to no-auto in every approach.  
No future actual is used for eligibility.

### A — Current 17J (XGB safety gate → median)

| Class | Rule |
|---|---|
| HIGH | 17F HIGH → median |
| MEDIUM-SAFE | 17F MEDIUM **AND** C6 (`XGB ≤ 1.5×hist`) **AND** Strategy E (`XGB ≤ 60` **AND** `XGB ≤ 2×hist`) → median |
| MEDIUM-RISK / UNSTABLE | no auto |

### B — Historical-only eligibility → median

| Class | Rule |
|---|---|
| HIGH | 17F HIGH → median |
| MEDIUM-SAFE | **all** 17F MEDIUM → median |
| UNSTABLE | no auto |

### C — History-stricter hybrid (prediction-time history only) → median

Thresholds taken from **Phase 17H C2 ∩ C3** (NormMAD ≤ 0.40 **AND** Drift ≤ 10).  
**Not re-optimized** on validation/test in this phase.

| Class | Rule |
|---|---|
| HIGH | 17F HIGH → median |
| MEDIUM-SAFE | 17F MEDIUM **AND** NormMAD ≤ 0.40 **AND** Drift ≤ 10 → median |
| MEDIUM-RISK / UNSTABLE | no auto |

---

## 3. Prediction-time availability & leakage

| Approach | Eligibility features | Available at predict time? | Future leakage? |
|---|---|---|---|
| **A** | 17F history + production XGB | **Yes** (if model is scored) | **No** |
| **B** | 17F history only | **Yes** | **No** |
| **C** | 17F + NormMAD + Drift | **Yes** | **No** |

### Architectural assessment of A (XGB gate + median predictor)

| Aspect | Finding |
|---|---|
| Reproducible offline vs online | **Yes**, given the same model artifact and feature vector |
| Future leakage | **None** — XGB and hist use only information available before the next purchase |
| Logical coherence | **Asymmetric** — eligibility depends on a model whose output is discarded |
| Fragility | Eligibility can change on **retrain** even when personal history is unchanged |
| Interpretation that makes A “safe” | Treat XGB as a **regime / inflation veto**, not as a competing predictor |

**Verdict on A’s safety:** operationally safe and reproducible at prediction time; architecturally coupled to a model that is not the SAFE output. Not rejected on leakage grounds; tension is design, not data leakage.

---

## 4. Controls

| Control | Value |
|---|---|
| Validation | 2026-05-01 → 2026-06-30, `target>0`, N=12,767; Path A n=8,658 |
| Test | 2026-07-01 → 2026-08-31, `target>0`, N=7,488; Path A n=5,155 |
| Predictor (all approaches) | Personal historical median |
| Model | Production `refill_model.joblib` read-only (used only in A) |
| Threshold tuning | **None** in this phase |
| Retrain / routing / WhatsApp / Path B / promotion | **None** |

---

## 5. Validation results

### Coverage & overall accepted metrics

| Approach | HIGH | MEDIUM-SAFE | RISK | UNSTABLE | Auto | Cov (univ) | MAE | MedAE | RMSE | ±1d | ±3d | ±7d | Cat n/% | Bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A** | 1,144 | 1,947 | 1,082 | 4,485 | 3,091 | **24.2%** | **8.00** | 4.0 | 12.67 | 22.0 | 43.0 | **64.8%** | 115 / **3.7%** | −0.09 |
| **B** | 1,144 | 2,803 | 226 | 4,485 | 3,947 | **30.9%** | 9.00 | 5.0 | 14.26 | 19.8 | 39.4 | 60.9% | 198 / 5.0% | −0.37 |
| **C** | 1,144 | 2,254 | 775 | 4,485 | 3,398 | 26.6% | 8.34 | 4.5 | 13.27 | 21.7 | 42.5 | 64.0% | 152 / 4.5% | −0.54 |

### HIGH / MEDIUM-SAFE separately (validation)

| Approach | Cohort | N | MAE | MedAE | ±7d | Cat % | Bias |
|---|---|---:|---:|---:|---:|---:|---:|
| A/B/C | HIGH | 1,144 | 7.31 | 3.5 | 69.1% | 3.5% | −0.18 |
| A | MEDIUM-SAFE | 1,947 | 8.41 | 5.0 | 62.2% | 3.9% | −0.04 |
| B | MEDIUM-SAFE | 2,803 | 9.69 | 6.0 | 57.5% | 5.6% | −0.44 |
| C | MEDIUM-SAFE | 2,254 | 8.87 | 5.0 | 61.5% | 5.0% | −0.73 |

---

## 6. Test results

### Coverage & overall accepted metrics

| Approach | HIGH | MEDIUM-SAFE | RISK | UNSTABLE | Auto | Cov (univ) | MAE | MedAE | RMSE | ±1d | ±3d | ±7d | Cat n/% | Bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A** | 619 | 1,084 | 506 | 2,946 | 1,703 | **22.7%** | 7.60 | 4.5 | 11.32 | 21.2 | 41.7 | 64.1% | 30 / **1.8%** | +3.57 |
| **B** | 619 | 1,504 | 86 | 2,946 | 2,123 | **28.4%** | 8.18 | 5.0 | 12.42 | 19.6 | 39.1 | 61.2% | 54 / 2.5% | +3.51 |
| **C** | 619 | 1,174 | 416 | 2,946 | 1,793 | 23.9% | **7.38** | **4.0** | 11.35 | 21.7 | 42.9 | **66.0%** | 36 / 2.0% | +2.76 |

### HIGH / MEDIUM-SAFE separately (test)

| Approach | Cohort | N | MAE | MedAE | ±7d | Cat % | Bias |
|---|---|---:|---:|---:|---:|---:|---:|
| A/B/C | HIGH | 619 | 7.08 | 4.0 | 70.4% | 1.8% | +2.51 |
| A | MEDIUM-SAFE | 1,084 | 7.90 | 5.0 | 60.4% | 1.8% | +4.18 |
| B | MEDIUM-SAFE | 1,504 | 8.63 | 6.0 | 57.5% | 2.9% | +3.92 |
| C | MEDIUM-SAFE | 1,174 | 7.53 | 5.0 | 63.6% | 2.1% | +2.89 |

### Overlap (test MEDIUM-SAFE)

| Relation | N |
|---|---:|
| A SAFE ⊆ B SAFE | 1,084 (all of A) |
| B SAFE not in A | 420 |
| C SAFE ∩ A SAFE | 878 |
| A SAFE not in C | 206 |

---

## 7. Canonical pattern `30,60,45,30,90,30,60,30,90`

| Property | Value |
|---|---|
| 17F class | **MEDIUM** |
| Hist median | 45 |
| NormMAD | 0.333 |
| Drift | **15.0** |
| Last / hist | 90 / 45 = **2.0** |

| Approach | Eligible for auto median? |
|---|---|
| **A** (XGB calm, e.g. 1.1× hist) | **Yes** (MEDIUM-SAFE) |
| **A** (XGB inflated ≥ ~1.5× with E fail, or >1.5×) | **No** |
| **B** | **Yes** (always, as 17F MEDIUM) |
| **C** (NormMAD≤0.40 ∧ Drift≤10) | **No** — Drift=15 fails C3 |

**Implication:** C’s history-stricter Drift gate **rejects the canonical variable recurrer** even when the next prediction would be calm. A last-interval ≤1.5×hist gate would also reject it (last=2×hist). **A preserves eligibility when XGB stays near the personal median**; that matches the 17J design intent for variable-but-recurring customers.

---

## 8. Comparative reading (no selection)

| Lens | Observation |
|---|---|
| Coverage | B > C > A |
| Accuracy (val) | A best MAE / ±7d / cat among the three |
| Accuracy (test) | C slightly better MAE / ±7d than A; A still lowest catastrophic % |
| Coherence with median predictor | B and C fully history-aligned; A couples to XGB |
| Canonical variable pattern | **A (when calm) and B keep it; C drops it** |
| Leakage | None of A/B/C leak future actual |

This audit **does not pick a winner**. Metrics are reported for architecture review only.

---

## 9. Conclusions (audit only)

1. **A’s XGB gate is prediction-time safe** (available, deterministic, no future leakage) but **architecturally asymmetric** with a median predictor and **retrain-coupled**.  
2. **B** is the cleanest history-only design; it buys coverage and loses accuracy / catastrophic control vs A.  
3. **C** (17H C2∩C3) is a justified history-only hybrid without new tuning; on test it looks competitive with A on MAE/±7d, but it **fails the canonical 30/60/90 pattern** because Drift=15.  
4. **No policy is selected or promoted** in Phase 17K.

---

## 10. Artifact safety

| Check | Result |
|---|---|
| Production model unchanged | **Yes** (`1789645154.9395924`) |
| Production routing / scheduler / WhatsApp | **Unchanged** |
| Feature flag | **OFF** |
| Path B | **Not implemented** |
| Retrain | **No** |
| Policy selected / promoted | **No** |
