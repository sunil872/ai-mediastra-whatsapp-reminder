# Phase 17J — Final Path A Policy Validation

**Date:** 2026-09-18  
**Status:** READ-ONLY offline validation — **no promotion / production changes**  
**Script:** `scripts/phase17j_final_path_a_policy_validation.py`  
**Raw JSON:** `data/refillcare/processed/phase17j_final_path_a_policy_results.json`  
**Scope:** Path A only (`purchase_count >= 6`). Path B **not** implemented.  
**Feature flag:** `USE_PATH_A_CLASSIFIER_V17F` remains **OFF**.

---

## 1. Proposed Path A policy (evaluated offline)

| Class | Action |
|---|---|
| **HIGH** | Personal historical median |
| **MEDIUM-SAFE** | Personal historical median |
| **MEDIUM-RISK** | No automatic prediction |
| **UNSTABLE** | No automatic prediction |

**MEDIUM-SAFE** = 17F MEDIUM **AND** C6 (`XGB ≤ 1.5 × hist`) **AND** Strategy E (`XGB ≤ 60` **AND** `XGB ≤ 2 × hist`).

**Prediction-time exclusions (no future leakage):**

- Insufficient Path A history (`P < 6`) — outside this frame  
- Invalid/missing cadence statistics → demote to no-auto (RISK/UNSTABLE)  
- **Oracle collapse** (`actual < 0.5 × hist`) is **not** used at prediction time  

---

## 2. Controls

| Control | Value |
|---|---|
| Validation | 2026-05-01 → 2026-06-30, `target>0`, N=12,767 |
| Test | 2026-07-01 → 2026-08-31, `target>0`, N=7,488 |
| Classifier | 17F Path A (offline call; flag OFF) |
| XGB | Production `refill_model.joblib` read-only (used only for SAFE gating, not SAFE prediction) |
| Retrain / routing / WhatsApp / Path B | **None** |

### Comparators

| Policy | Auto predictor |
|---|---|
| **17D hybrid** | HIGH=median; MEDIUM=production XGB (no E); UNSTABLE=reject |
| **17F + Strategy E** | HIGH=median; MEDIUM=XGB if E else reject; UNSTABLE=reject |
| **17J (proposed)** | HIGH=median; MEDIUM-SAFE=median; RISK/UNSTABLE=reject |

---

## 3. Cohort counts

### Validation (Path A n=8,658)

| Class | Count |
|---|---:|
| HIGH | 1,144 |
| MEDIUM-SAFE | 1,947 |
| MEDIUM-RISK | 1,082 |
| UNSTABLE | 4,485 |
| **Automatic** | **3,091** (35.7% Path A / **24.2%** universe) |

### Test (Path A n=5,155)

| Class | Count |
|---|---:|
| HIGH | 619 |
| MEDIUM-SAFE | 1,084 |
| MEDIUM-RISK | 506 |
| UNSTABLE | 2,946 |
| **Automatic** | **1,703** (33.0% Path A / **22.7%** universe) |

*(HIGH is slightly below raw 17F HIGH because missing/incomplete cadence diagnostics are demoted to no-auto.)*

---

## 4. Overall accepted metrics — 17J policy

### Validation (auto n=3,091)

| MAE | MedAE | RMSE | ±1d | ±3d | ±7d | Cat % | Cat n | Mean bias | Total \|err\| |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **8.00** | **4.00** | 12.67 | 22.00% | 43.00% | **64.77%** | **3.72%** | 115 | −0.09 | 24,730 |

### Test (auto n=1,703)

| MAE | MedAE | RMSE | ±1d | ±3d | ±7d | Cat % | Cat n | Mean bias | Total \|err\| |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **7.60** | **4.50** | 11.32 | 21.20% | 41.69% | **64.06%** | **1.76%** | 30 | +3.57 | 12,941 |

---

## 5. By cohort (17J)

### Test

| Cohort | N | Auto? | MAE | ±7d | Cat % |
|---|---:|---|---:|---:|---:|
| HIGH | 619 | Yes (median) | 7.08 | **70.4%** | 1.8% |
| MEDIUM-SAFE | 1,084 | Yes (median) | 7.90 | **60.4%** | 1.8% |
| MEDIUM-RISK | 506 | No | — | — | — |
| UNSTABLE | 2,946 | No | — | — | — |

### Validation

| Cohort | N | Auto? | MAE | ±7d | Cat % |
|---|---:|---|---:|---:|---:|
| HIGH | 1,144 | Yes | 7.31 | **69.1%** | 3.5% |
| MEDIUM-SAFE | 1,947 | Yes | 8.41 | **62.2%** | 3.9% |
| MEDIUM-RISK | 1,082 | No | — | — | — |
| UNSTABLE | 4,485 | No | — | — | — |

---

## 6. Comparison vs prior policies

### Test

| Policy | Accepted | Cov (universe) | MAE ↓ | ±7d ↑ | Cat % ↓ |
|---|---:|---:|---:|---:|---:|
| 17D hybrid | 2,533 | **33.8%** | 10.39 | 55.1% | 5.8% |
| 17F + Strategy E | 2,056 | 27.5% | 10.37 | 48.1% | 4.2% |
| **17J proposed** | 1,703 | 22.7% | **7.60** | **64.1%** | **1.8%** |

### Validation

| Policy | Accepted | Cov (universe) | MAE ↓ | ±7d ↑ | Cat % ↓ |
|---|---:|---:|---:|---:|---:|
| 17D hybrid | 4,703 | **36.8%** | 11.13 | 54.5% | 8.2% |
| 17F + Strategy E | 3,785 | 29.7% | 10.53 | 47.8% | 5.4% |
| **17J proposed** | 3,091 | 24.2% | **8.00** | **64.8%** | **3.7%** |

**Tradeoff:** 17J trades coverage for accuracy. On test, ~11 pp less universe coverage than 17D, but **+9 pp ±7d**, **−2.8d MAE**, and catastrophic rate cut by ~70% relative.

---

## 7. Collapse detection (critical)

**Definition (oracle / analysis only):** next interval `actual < 0.5 × hist_median`.

**Must not use future actual at prediction time.**

### Historical proxies tested on MEDIUM-SAFE

| Proxy | Test precision | Test recall | Val precision | Val recall |
|---|---:|---:|---:|---:|
| Last interval < 0.5× hist | 35.5% | 15.2% | 17.3% | 13.2% |
| Recent-3 < 0.5× hist | 44.4% | 6.7% | 18.9% | 4.0% |
| Either | 33.7% | 16.3% | 16.8% | 13.8% |

On test MEDIUM-SAFE, oracle collapse rate ≈ **16%** (178/1084). Proxies catch only a small fraction and with low precision.

### Verdict

**Collapse cannot be reliably detected at prediction time** with these historical proxies.  
Therefore the primary 17J policy **does not** exclude collapse via future actual, and **does not** adopt these weak proxies as hard gates.

---

## 8. Canonical variable pattern `30,60,45,30,90,30,60,30,90`

| Condition | 17J policy |
|---|---|
| 17F class | **MEDIUM** |
| Calm XGB (≤1.5× hist) + E | **MEDIUM-SAFE → personal median** |
| Inflated XGB (>1.5× hist) | **MEDIUM-RISK → no auto** |

**Requirement met:** pattern remains eligible for automatic median prediction when the model prediction is calm/near personal cadence.

---

## 9. Conclusions (no promotion)

1. **17J policy is validated offline** on both validation and test: auto median on HIGH + MEDIUM-SAFE delivers **~64% ±7d** with low catastrophic rate.  
2. It **beats 17D hybrid and 17F+E** on MAE / ±7d / catastrophic error, at lower coverage.  
3. **Collapse is not prediction-time detectable** with tested history-only proxies; do not leak future actual.  
4. Variable recurrers like `30/60/90` stay in the auto path when XGB is calm.  
5. **Not promoted / not implemented** in this phase.

---

## 10. Artifact safety

| Check | Result |
|---|---|
| Production model unchanged | **Yes** (`1789645154.9395924`) |
| Production routing / scheduler / WhatsApp | **Unchanged** |
| Feature flag | **OFF** |
| Path B | **Not implemented** |
| Retrain | **No** |
| Strategy promoted | **No** |
