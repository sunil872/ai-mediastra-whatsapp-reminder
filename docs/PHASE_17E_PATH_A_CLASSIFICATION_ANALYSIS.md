# Phase 17E — Path A Classification Analysis (Read-Only)

**Date:** 2026-09-18  
**Status:** Offline analysis only — production routing, model, reminder stages, WhatsApp **unchanged**  
**Script:** `scripts/phase17e_path_a_classification_analysis.py`  
**Raw JSON:** `data/refillcare/processed/phase17e_path_a_classification_results.json`  
**Scope:** Path A only — established `customerId + itemId` with **≥6 purchases**. Path B / recent customers **out of scope**.

---

## 1. Purpose

Validate whether current Path A gates actually separate three **business behaviors**:

| Class | Business meaning | Intended action (conceptually) |
|---|---|---|
| **HIGH** | Stable / predictable recurring cadence | Trust personal historical median |
| **MEDIUM** | Genuinely recurring but variable cadence | Secondary / cautious prediction path |
| **UNSTABLE** | No reliable recurring cadence | Do not auto-predict |

**Do not assume** NormMAD ≤ 0.35 / Drift ≤ 7 (HIGH) or NormMAD ≤ 0.50 / Drift ≤ 10 (MEDIUM) are correct. This analysis checks whether they match real interval patterns.

---

## 2. Controls

| Control | Value |
|---|---|
| Dataset | RefillCare 2020–2026 (`purchase_history.parquet` + temporal test) |
| History range | **2020-12-24 → 2026-08-31** |
| Test holdout | `test.parquet`, `target > 0`, **N = 7,488** (2026-07-01 → 2026-08-30) |
| Path A events | `purchase_count_so_far ≥ 6` → **N = 5,155** (68.8% of test) |
| Established pairs (full history) | **24,791** pairs with ≥6 purchases and ≥5 intervals |
| Production model | **Not used / not modified** (classification + personal-median diagnostics only) |
| WhatsApp / reminder stages | **Unchanged; 0 messages** |
| Retrain | **None** |

Accuracy diagnostics below use **personal historical median vs actual next interval** as a behavior probe (not a production XGB score). That isolates whether the class is predictable from its own cadence.

---

## 3. Current Path A rules (`hybrid_routing_v17d`)

```
IF purchase_count < 6                         → UNSTABLE (not Path A)
ELSE IF hist_median ∉ [15, 120]               → UNSTABLE
ELSE IF NormMAD missing OR Drift missing      → UNSTABLE
ELSE IF NormMAD > 0.50 OR Drift > 10          → UNSTABLE
ELSE IF NormMAD ≤ 0.35 AND Drift ≤ 7          → HIGH
ELSE                                          → MEDIUM
```

Source: `refillcare/models/hybrid_strategy.py`  
(`REJECTED` in code ≡ **UNSTABLE** in this Path A business vocabulary.)

---

## 4. Canonical pattern check (critical)

| Pattern | Intervals | NormMAD | Drift | max/med | Current label | Business oracle |
|---|---|---:|---:|---:|---|---|
| Stable | `30,30,31,29,30,32` | 0.017 | 0.0 | 1.07 | **HIGH** ✓ | STABLE |
| Recurring variable | `30,60,45,30,90,30,60,30,90` | 0.333 | **15.0** | 2.00 | **UNSTABLE** ✗ | RECURRING_VARIABLE |
| Unstable / chaotic | `5,180,12,240,3,150` | 0.951 | 69.0 | 2.96 | **UNSTABLE** ✓ | UNSTABLE |

**Finding:** Current Drift ≤ 10 rejects the exact “recurring but variable” pattern the product wants as MEDIUM. NormMAD alone (0.33) would have allowed it into HIGH’s NormMAD band, but Drift alone forces UNSTABLE.

---

## 5. Cohort sizes

### 5.1 Test Path A events (N = 5,155)

| Current label | Count | % of Path A | % of test | Personal-median MAE | ±7d |
|---|---:|---:|---:|---:|---:|
| **HIGH** | 1,778 | 34.5% | 23.7% | 7.25 | 67.4% |
| **MEDIUM** | 731 | 14.2% | 9.8% | 9.87 | 49.3% |
| **UNSTABLE** | 2,646 | 51.3% | 35.3% | 13.64 | 58.3% |
| Accepted (H+M) | 2,509 | 48.7% | 33.5% | 8.01 | 62.1% |

### 5.2 Full-history established pairs (N = 24,791)

| Current label | Pairs |
|---|---:|
| HIGH | 6,262 |
| MEDIUM | 3,024 |
| UNSTABLE | 15,505 |

### 5.3 Independent business-oracle cohorts (test Path A)

Oracle labels are **not** NormMAD/Drift thresholds. They use interval shape:

- **STABLE:** ≥70% of intervals within ±30% of median, max/median ≤ 2.0, CV ≤ 0.45 (or ≥80% within ±50% with NormMAD ≤ 0.40)
- **RECURRING_VARIABLE:** hist median in [15,120], ≥60% of intervals in [15,120], extremes &lt;35%, not STABLE
- **UNSTABLE:** out-of-band median, many extremes, chaotic max/median, or high CV with poor concentration

| Oracle label | Count | % of Path A | Personal-median MAE | ±7d |
|---|---:|---:|---:|---:|
| STABLE | 635 | 12.3% | 7.46 | 68.7% |
| RECURRING_VARIABLE | 1,880 | 36.5% | 10.30 | 52.2% |
| UNSTABLE | 2,640 | 51.2% | 12.16 | 63.8% |

Oracle UNSTABLE ±7d looks “ok” at median because many chaotic series still have a median near occasional short refills; the **MAE / bias / extreme tails** remain worse. Classification quality is judged by **agreement with interval shape**, not ±7d alone.

---

## 6. Interval distributions by current label

### Hist median (days)

| Label | p05 | p25 | p50 | p75 | p95 | mean |
|---|---:|---:|---:|---:|---:|---:|
| HIGH | 15 | 18 | 28 | 31 | 37 | 25.9 |
| MEDIUM | 15 | 19 | 22.5 | 30 | 42 | 25.6 |
| UNSTABLE | 4 | 11 | 14 | 29 | 71 | 24.5 |

### NormMAD / Drift / outlier severity

| Label | NormMAD p50 | Drift p50 | CV p50 | max/median p50 | max/median p95 |
|---|---:|---:|---:|---:|---:|
| HIGH | 0.18 | 1.5 | 0.51 | **2.79** | **22.8** |
| MEDIUM | 0.40 | 6.0 | 0.72 | **3.91** | **28.1** |
| UNSTABLE | 0.33 | 4.0 | 0.82 | **4.49** | **65.0** |

**Finding:** HIGH is not “outlier-free.” Because MAD is robust, a series can keep NormMAD ≤ 0.35 while still containing gaps **20×** the median. Current HIGH therefore mixes true stables with contaminated histories.

---

## 7. Examples from real data

### Stable (correctly HIGH)

`AMRUTHAVANI / 12775` — intervals `31,30,31,31,28,31,30,31`  
NormMAD 0.00, Drift 1, max/med 1.0, actual=31, |error|=0.

### Recurring variable wrongly UNSTABLE (Drift gate)

`VYSHNAVI / 2057` — intervals `…28,40,30,23,31,32,30,16,20`  
NormMAD 0.33, Drift **10.5**, 0% extremes, 77% within ±50% of median.  
Oracle: RECURRING_VARIABLE. Current: **UNSTABLE**. Personal median error vs actual 27d = **3.5d**.

### Chaotic wrongly MEDIUM (NormMAD/Drift pass, extremes present)

`SHEEBA RANI / 8767` — intervals `5,249,98,217`  
NormMAD 0.48, Drift 0, but **75% extreme** gaps. Current: **MEDIUM**. Personal median error **78d**.

### Contaminated HIGH (robust MAD hides giant gaps)

`SRINIVAS / 1993` — hist median 15, NormMAD ~0.30, Drift 3, but max/median **59.6**, CV ~3.4.  
Current: **HIGH**. Oracle: UNSTABLE.

---

## 8. Does current NormMAD/Drift separate the three behaviors?

### Confusion: current Path A vs business oracle

| Current \ Oracle | STABLE | RECURRING_VARIABLE | UNSTABLE |
|---|---:|---:|---:|
| **HIGH** | 543 | 992 | **243** |
| **MEDIUM** | 44 | 409 | **278** |
| **UNSTABLE** | 48 | **479** | 2,119 |

Oracle agreement of current rules: **59.6%**.

### Failure modes

1. **HIGH is too broad**  
   - Only **30.5%** of current HIGH is oracle-STABLE (543/1778).  
   - **55.8%** is recurring-variable; **13.7%** is unstable.  
   - Cause: NormMAD/Drift ignore extreme gaps and max/median.

2. **MEDIUM is contaminated**  
   - **38.0%** of MEDIUM is oracle-UNSTABLE (278/731).  
   - Drivers among MEDIUM: NormMAD-driven 442, Drift-driven 158, both elevated 131.  
   - Shallow histories with one huge gap + a few short intervals often still pass Drift ≤ 10.

3. **True recurring-variable is under-accepted**  
   - **479** oracle RECURRING_VARIABLE rows are UNSTABLE under current rules (often Drift 10–15+ with otherwise refill-like intervals).  
   - Matches the canonical `30/60/90` failure.

4. **Threshold-only sweeps are insufficient**  
   Best NormMAD/Drift-only grid point reaches ~**71%** oracle agreement (e.g. core 0.20/5, broad 0.40/20).  
   Still far below a multi-feature tree (**~82%**). **Retuning 0.35/7 and 0.50/10 alone cannot fix Path A.**

---

## 9. Threshold sensitivity (NormMAD / Drift only)

Holding Path A depth ≥6 and cadence band [15,120], sweeping:

- Core NormMAD ∈ {0.20…0.45}, Drift ∈ {5,7,10}  
- Broad NormMAD ∈ {0.40…0.70}, Drift ∈ {7,10,15,20}

Top oracle-agreement candidates (personal-median probe):

| Core NM / Dr | Broad NM / Dr | Agree | Accepted cov | HIGH ±7d | False HIGH→Uns | False MED→Uns |
|---|---|---:|---:|---:|---:|---:|
| 0.20 / 5 | 0.40 / 20 | 71.4% | 47.4% | 72.8% | 79 | 292 |
| 0.20 / 7 | 0.40 / 20 | 71.3% | 47.4% | 72.1% | 82 | 289 |
| 0.20 / 10 | 0.40 / 20 | 71.3% | 47.4% | 70.9% | 83 | 288 |
| **Current 0.35 / 7** | **0.50 / 10** | **59.6%** | **48.7%** | **67.4%** | **243** | **278** |

**Interpretation:** Tighter HIGH helps precision; wider Drift helps recover variable recurrers; but without extreme-gap / max-median guards, MEDIUM still absorbs chaos.

---

## 10. Recommended Path A rules (proposal only — not implemented)

### Design principles

1. Keep Path A = **P ≥ 6** and refill cadence band **[15, 120]**.  
2. Add **instability guards** NormMAD/Drift cannot see.  
3. Make **HIGH stricter** (stable only).  
4. Make **MEDIUM wider on Drift (≤15)** *after* guards, so `30/60/90`-like series qualify.  
5. Everything else stays **UNSTABLE**.

### Final proposed decision tree

```
1. IF purchase_count < 6
      → UNSTABLE (not Path A; Path B later)

2. IF hist_median ∉ [15, 120]
      → UNSTABLE

3. IF NormMAD missing OR Drift missing
      → UNSTABLE

4. IF pct_intervals with (interval < 10 OR interval > 180) ≥ 35%
      → UNSTABLE

5. IF pct_intervals in [15, 120] < 55%
      → UNSTABLE

6. IF (max_interval / hist_median) ≥ 3.5
      AND pct_intervals within ±50% of median < 65%
      → UNSTABLE

7. HIGH (stable) IF either:
      (a) NormMAD ≤ 0.30 AND Drift ≤ 7 AND max/median ≤ 2.2
      OR
      (b) NormMAD ≤ 0.35 AND Drift ≤ 5 AND max/median ≤ 2.0

8. MEDIUM (recurring variable) IF
      NormMAD ≤ 0.55 AND Drift ≤ 15
      (and steps 1–6 passed)

9. ELSE → UNSTABLE
```

### Canonical patterns under proposed tree

| Pattern | Proposed |
|---|---|
| `30,30,31,29,30,32` | **HIGH** |
| `30,60,45,30,90,30,60,30,90` | **MEDIUM** (recovered) |
| `5,180,12,240,3,150` | **UNSTABLE** |

---

## 11. Expected coverage / accuracy impact (offline)

Personal-median probe on the same 5,155 Path A test events:

| System | HIGH n | MEDIUM n | UNSTABLE n | Accepted | Agree w/ oracle | Accepted MAE | Accepted ±7d |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Current** | 1,778 | 731 | 2,646 | 2,509 (48.7% Path A / 33.5% test) | 59.6% | 8.01 | 62.1% |
| **Proposed** | 673 | 1,538 | 2,944 | 2,211 (42.9% Path A / 29.5% test) | **81.8%** | 8.25 | 60.7% |

### Proposed vs oracle confusion

| Proposed \ Oracle | STABLE | RECURRING_VARIABLE | UNSTABLE |
|---|---:|---:|---:|
| HIGH | **484** | 126 | **63** |
| MEDIUM | 119 | **1,286** | **133** |
| UNSTABLE | 32 | 468 | **2,444** |

### What changes operationally (if later implemented)

| Effect | Direction |
|---|---|
| HIGH purity | **Much better** — false HIGH→unstable 243 → 63; HIGH becomes closer to true stables |
| MEDIUM meaning | **Much better** — grows with genuine variable recurrers; unstable inside MEDIUM 278 → 133 |
| Accepted coverage | **Slightly down** (−~4pp of test; −~6pp of Path A) |
| Accepted ±7d (personal median) | Roughly flat (~62% → ~61%) |
| Classification fidelity | **Large gain** (+22pp oracle agreement) |

Net: Path A becomes a **cleaner triage**, not a free accuracy boost. Accuracy gains for MEDIUM will still depend on later MEDIUM predictor / safety gates (Phase 17D evidence), not classification alone.

---

## 12. Conclusions

1. Current NormMAD ≤ 0.35 / Drift ≤ 7 and NormMAD ≤ 0.50 / Drift ≤ 10 **do not cleanly separate** HIGH / MEDIUM / UNSTABLE.  
2. HIGH over-accepts contaminated series (robust MAD).  
3. MEDIUM over-accepts chaotic shallow histories and **under-accepts** true variable recurrers (Drift wall at 10).  
4. NormMAD/Drift retunes alone top out ~71% agreement; **guards on extremes + max/median are required**.  
5. Recommended tree recovers the intended business taxonomy (~82% agreement) with a modest coverage trade.

**No production code, model, reminder logic, or WhatsApp behavior was changed in this phase.**

---

## 13. Artifact safety

| Check | Result |
|---|---|
| `refill_model.joblib` mtime unchanged | **Yes** (`1789645154.9395924`) |
| `phase4_model_report.json` mtime unchanged | **Yes** |
| `hybrid_strategy.py` / production routing modified | **No** |
| WhatsApp messages sent | **0** |
| Retrain | **No** |

---

## 14. What should change next (DO NOT IMPLEMENT YET)

1. **Implement Path A decision tree** from §10 in a non-production / feature-flagged classifier (keep current `hybrid_routing_v17d` until approved).  
2. **Add features** needed by guards: `pct_extreme_lt10_or_gt180`, `pct_in_refill_band_15_120`, `max_over_median` (or equivalent) into the feature pipeline / eligibility record.  
3. **Re-run hybrid evaluation** with proposed labels: HIGH → personal median; MEDIUM → safety-gated XGB (Phase 17D gates B/D/E); UNSTABLE → no auto date.  
4. **Add regression tests** for the three canonical interval sequences (stable / variable / chaotic).  
5. **Only after offline sign-off**, consider promoting Path A classifier; still no Path B, no WhatsApp enablement, no production model overwrite in that step unless separately authorized.
