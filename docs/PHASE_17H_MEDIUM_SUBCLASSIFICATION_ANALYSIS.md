# Phase 17H — MEDIUM Subclassification / Final Path A Investigation

**Date:** 2026-09-18  
**Status:** READ-ONLY offline investigation — **no strategy promotion / implementation**  
**Script:** `scripts/phase17h_medium_subclassification_analysis.py`  
**Raw JSON:** `data/refillcare/processed/phase17h_medium_subclassification_results.json`  
**Scope:** Path A only (`purchase_count >= 6`). Path B **not** implemented.  
**Feature flag:** `USE_PATH_A_CLASSIFIER_V17F` remains **OFF**.

---

## 1. Purpose

Ask whether the Phase 17F **MEDIUM** cohort (n = **1,540**) can be split into:

| Subclass | Intent |
|---|---|
| **MEDIUM-SAFE** | Recurring and sufficiently predictable to keep automated prediction |
| **MEDIUM-RISK** | Recurring in structure, but prediction unreliable → hold / no auto date |

Baseline MEDIUM predictor for scored comparisons: **strategy E** from 17G  
(`reject if pred > 60` **OR** `pred > 2 × hist median`).

Goal is **not** max accuracy by rejecting everyone — look for a **meaningful SAFE pocket with useful coverage**.

---

## 2. Controls

| Control | Value |
|---|---|
| Test | Same temporal holdout, `target > 0`, **N = 7,488** |
| Path A | **5,155** |
| MEDIUM (17F) | **1,540** |
| HIGH (17F) reference | **669**, personal median |
| MEDIUM E reference | **1,387** accepted, MAE **11.88**, ±7d **37.85%** |
| Model | Production XGB read-only |
| WhatsApp / routing / retrain / Path B | **Unchanged / none** |

Thresholds were **simple and coarse** (no dense grid search / no test-set overfitting loop).

---

## 3. MEDIUM cohort profile (drivers of large error)

### 3.1 Distributions (all MEDIUM, n=1,540)

| Feature | p05 | p25 | p50 | p75 | p95 | mean |
|---|---:|---:|---:|---:|---:|---:|
| Depth | 7 | 15 | 26 | 43 | 77 | 32.1 |
| Hist median | 15.5 | 19 | 26 | 31 | 42 | 26.5 |
| Hist mean | 18 | 23 | 31 | 41 | 74 | 36.5 |
| NormMAD | 0.07 | 0.15 | 0.25 | 0.34 | 0.47 | 0.25 |
| Drift | 0 | 1 | 3 | 7 | 15 | 5.4 |
| max / median | 1.8 | 2.6 | 3.4 | 5.9 | 23 | 6.5 |
| % in [15,120] | 58 | 71 | 82 | 91 | 100 | 80.5 |
| % within ±50% med | 54 | 67 | 75 | 83 | 91 | 74.2 |
| % extreme | 0 | 0 | 5.7 | 12 | 20 | 7.2 |
| Pred / hist | 0.94 | 1.12 | 1.27 | 1.54 | 2.08 | 1.38 |
| XGB pred | 19 | 26 | 33 | 40 | 62 | 35.8 |
| Actual | 5 | 15 | 22 | 30 | 40 | 22.5 |
| Bias (pred−actual) | −5.5 | 3.6 | 10.2 | 19.6 | 41 | 13.3 |

Key **error flags** on full MEDIUM (raw XGB):

| Pattern | Count | % of MEDIUM | Notes |
|---|---:|---:|---|
| Pred > 60d | 87 | 5.7% | Mean \|err\| **56.7d** |
| Pred > 2 × hist | 100 | 6.5% | Caught by strategy E |
| Depth < 10 | 164 | 10.7% | MAE **32.1** |
| Drift > 10 | 214 | 13.9% | MAE **20.1** (includes variable recurrers) |
| Late bias > 14d | 578 | 37.5% | Dominant failure mode |
| max/median > 2.5 | 1,185 | 77.0% | Common even in usable rows (MAD-robust histories) |

### 3.2 Slices on MEDIUM strategy-E accepted (n=1,387)

| Slice | N | MAE | ±7d | Catastrophic % |
|---|---:|---:|---:|---:|
| Depth 6–9 | 94 | **20.19** | **7.5%** | 17.0% |
| Depth ≥ 10 | 1,293 | 11.27 | 40.1% | 4.5% |
| NormMAD ≤ 0.40 | 1,236 | 11.38 | 39.6% | 5.0% |
| NormMAD > 0.40 | 151 | 15.97 | 23.2% | 8.0% |
| Drift ≤ 10 | 1,202 | 11.45 | 39.6% | 5.1% |
| Drift 10–15 | 125 | 13.49 | 32.8% | 5.6% |
| Pred ≤ 1.5 × hist | **1,091** | **10.65** | **43.1%** | **4.0%** |
| Pred 1.5–2 × hist | 296 | **16.38** | **18.6%** | **10.1%** |

**Largest actionable pattern:** when XGB stays near personal cadence (`pred ≤ 1.5 × hist`), errors drop materially; when XGB inflates to 1.5–2× hist (still inside E), catastrophic rate more than doubles and ±7d collapses.

---

## 4. Candidate MEDIUM-SAFE rules (simple)

| ID | SAFE rule | SAFE % of MEDIUM | E-accepted N | MAE | MedAE | RMSE | ±1d | ±3d | ±7d | Cat % | Cat \|err\| | vs E ΔMAE | Cov retained vs E |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **C6** | pred ≤ 1.5 × hist | **72.6%** | **1,091** | **10.65** | **8.14** | **13.77** | **6.69%** | **17.05%** | **43.1%** | **4.0%** | 1,526 | **−1.23** | **78.7%** |
| C1 | depth ≥ 10 | 89.4% | 1,293 | 11.27 | 9.10 | 14.41 | 6.26% | 15.78% | 40.1% | 4.5% | 2,061 | −0.61 | 93.2% |
| C2 | NormMAD ≤ 0.40 | 88.0% | 1,236 | 11.38 | 9.15 | 14.64 | 6.47% | 16.18% | 39.6% | 5.0% | 2,221 | −0.50 | 89.1% |
| C3 | Drift ≤ 10 | 86.1% | 1,202 | 11.45 | 9.16 | 14.64 | 6.16% | 15.89% | 39.6% | 5.1% | 2,135 | −0.43 | 86.7% |
| C4 | Drift ≤ 12 | 90.4% | 1,265 | 11.54 | 9.20 | 14.76 | 6.09% | 15.57% | 39.7% | 5.2% | 2,320 | −0.34 | 91.2% |
| C7 | within50 ≥ 60% | 91.0% | 1,276 | 11.57 | 9.31 | 14.80 | 6.11% | 15.52% | 39.0% | 5.2% | 2,355 | −0.31 | 92.0% |
| C8 | balanced combo | 14.1% | 217 | 11.13 | 9.04 | 14.33 | 7.83% | 17.97% | 42.9% | 4.2% | 319 | −0.75 | 15.7% |
| C9 | conservative combo | 4.6% | 71 | 9.88 | 7.35 | 12.86 | 9.86% | 26.76% | 49.3% | 2.8% | 67 | −2.00 | 5.1% |
| C10 | variable-friendly combo | 3.7% | 57 | 9.00 | 6.75 | 11.98 | 14.04% | 35.09% | 50.9% | 1.8% | 30 | −2.88 | 4.1% |
| C5 | max/med ≤ 2.5 | 23.1% | 313 | 13.81 | 12.49 | 17.13 | 5.75% | 13.10% | 33.6% | 8.3% | 913 | **+1.93** | 22.6% |

**C6 RISK residual** (422 rows failing pred≤1.5×): MAE **23.2**, ±7d **14.7%**, catastrophic **22.8%** — clearly the unreliable pocket.

### Reference anchors

| Cohort | N | MAE | ±7d | Cat % |
|---|---:|---:|---:|---:|
| **HIGH** (personal median) | 669 | **7.26** | **69.2%** | 1.9% |
| **MEDIUM E** (baseline) | 1,387 | 11.88 | 37.9% | 5.3% |
| Best meaningful SAFE (**C6** + E) | 1,091 | 10.65 | 43.1% | 4.0% |

Even the best **coverage-preserving** SAFE pocket remains **far from HIGH** under production XGB (±7d 43% vs 69%).

### Personal median on SAFE (important signal)

For several candidates, **hist median on SAFE beats XGB**:

| Candidate | SAFE hist-median MAE | SAFE hist ±7d | SAFE XGB+E MAE | SAFE XGB+E ±7d |
|---|---:|---:|---:|---:|
| C6 | 8.68 | **58.9%** | 10.65 | 43.1% |
| C1 | 8.17 | **59.2%** | 11.27 | 40.1% |
| C2 | 8.19 | **59.6%** | 11.38 | 39.6% |

This suggests MEDIUM-SAFE may be more of a **predictor-choice** problem (median vs XGB) than a pure reject/accept problem — deferred; not implemented here.

---

## 5. Canonical variable pattern `30,60,45,30,90,30,60,30,90`

| Field | Value |
|---|---|
| 17F class | **MEDIUM** |
| NormMAD | 0.33 |
| Drift | **15** |
| max/median | 2.0 |
| % in [15,120] | 100% |
| % within ±50% median | 77.8% |
| extremes | 0% |

| Rule | In SAFE if calm XGB (~1.1× hist)? | If wild XGB (~2.2× hist)? |
|---|---|---|
| C6 pred≤1.5× | **Yes** | **No** → RISK |
| C3/C4 Drift≤10/12 | **No** | No |
| C1/C2/C7 | Yes | Yes |
| C9 conservative | No | No |
| C10 variable-friendly | Yes (calm) | No (wild) |

**Implication:** Variable recurrers should **remain eligible MEDIUM**, but become **MEDIUM-RISK when the model prediction inflates** vs their personal median. Hard Drift≤10 SAFE rules **incorrectly exile** this pattern.

---

## 6. Examples

### Correctly accepted (C6 SAFE + E, \|err\| ≤ 7d)

`M YADHAGIRI REDDY / 5566` — intervals near ~30d, hist 31, XGB 35.0, actual 35, \|err\|≈0.  
NormMAD 0.13, Drift 6 — calm prediction near cadence.

### Incorrectly accepted catastrophic (C6 SAFE + E, \|err\| > 30d)

`K ANJAIAH / 1678` — hist 47, XGB 48.9 (~1.04× hist → passes C6), actual **4**, \|err\| 44.9.  
Cadence looks recurring; **next interval collapses**. Shows SAFE≠guarantee under production XGB.

---

## 7. Comparison summary

| Question | Evidence |
|---|---|
| Does a MEDIUM-SAFE pocket exist with useful coverage? | **Yes, weakly.** Best simple rule is **pred ≤ 1.5 × hist** (C6): keeps ~73% of MEDIUM / ~79% of E accepts, MAE 10.65 (−1.2d vs E), ±7d 43% (+5pp). |
| Does SAFE approach HIGH? | **No.** HIGH ±7d 69% vs SAFE~43% (XGB) or ~59% (hist median on SAFE). |
| Do ultra-tight combos help? | Yes on accuracy (C9/C10 ±7d ~50%), but only **4–5%** of MEDIUM — rejects almost everyone; not the goal. |
| Is Drift≤10 a good SAFE gate? | **No** for product intent — blocks canonical variable recurrers. |
| Is max/median≤2.5 useful alone? | **No** — worse MAE than baseline E. |

---

## 8. Clear conclusions (no promotion)

### Is MEDIUM-SAFE supported by the evidence?

**Partially yes.** There is a **real, coverage-preserving** subpopulation where production XGB is less catastrophic — primarily when **prediction stays close to personal historical median** (and secondarily deeper history / lower NormMAD).

It is **not** yet a HIGH-quality automated band under the current squared-error model.

### Which patterns should remain MEDIUM (SAFE-leaning)?

- Recurring cadence in [15,120], including **variable** series like `30/60/90`  
- Adequate depth (especially **P ≥ 10**)  
- Moderate NormMAD  
- **XGB prediction not inflated** vs hist (roughly ≤ 1.5×) and not >60d  
- Low extreme-gap share / decent concentration around median  

### Which patterns should become MEDIUM-RISK?

- XGB **pred > 1.5–2× hist** (even if still ≤2× / inside E)  
- XGB **pred > 60**  
- **Shallow** MEDIUM (P 6–9): MAE ~20 on E-accepted  
- Strong **late bias** / short actual vs long pred (collapse intervals)  
- High NormMAD tail with weak ±7d  

### Is another Path A investigation required?

**Yes.** Before any promotion, investigate offline (still no production change):

1. **MEDIUM-SAFE predictor choice:** personal median vs production XGB vs quantile XGB on the C6-like pocket (hist median already shows ~59% ±7d on SAFE).  
2. Whether SAFE should be defined as **“use median”** and RISK as **“no auto / review”**, rather than SAFE=filtered XGB.  
3. Confirm on a **validation window** (not only this test holdout) so pred/hist gates are not test-fit.

**Do not implement or enable any flag yet.**

---

## 9. Artifact safety

| Check | Result |
|---|---|
| Production model unchanged | **Yes** (mtime `1789645154.9395924`) |
| Production routing / scheduler / WhatsApp | **Unchanged** |
| Feature flag | **OFF** |
| Path B | **Not implemented** |
| Strategy promoted | **No** |
| Retrain | **No** |
