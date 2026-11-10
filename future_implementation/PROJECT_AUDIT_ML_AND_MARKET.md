# RefillCare — Project Audit: Model Accuracy, Deep Learning, Go-Live Readiness and Market Comparison

Date: 6 Nov 2026 · Scope: the RefillCare refill-prediction and WhatsApp reminder system (single store today: NGOs Colony, `store-1`; target: 200+ stores, multi-tenant).
All accuracy numbers below come from the project's own database (`ml_evaluation_runs`, `ml_model_versions`) and the stored benchmark report. Market facts come from public sources listed at the end.

---

## 0. Short answers

| # | Question | Answer |
|---|---|---|
| 1 | Is 57–60% accuracy enough for live use? | **Yes, for the model.** "Within ±7 days" is a strict measure. Our engine beats every simple method a pharmacy could use instead (fixed interval / days-of-supply: 45.7%, personal median: 52.5%) and cuts late (stock-out) reminders from 26% to 19%. The multi-stage reminder sequence absorbs most of the remaining error. |
| 2 | Should we switch to neural networks / deep learning? | **No, not now.** On tabular sales data, gradient-boosted trees are still the state of the art; the limit today is data quality (shared phones, missing dose information), not model power. Revisit once 200+ stores are pooled. |
| 3 | Is the current implementation good to go for 200+ stores? | **Model: YES. Platform for 200+ stores: NOT YET.** The prediction engine, gate and reminder flow are production quality for a pilot. Multi-tenant hosting (tenant/store control plane, auth, PostgreSQL schemas, S3, async uploads, per-store retraining) is not built yet — see section 5. |
| 4 | What do competitors use? | **No Indian pharmacy publishes an ML refill-timing model.** Public evidence shows fixed-interval or subscription reminders (PharmEasy monthly, Go4whatsup 25 days after purchase, Marg ERP prescription reminders). Published ML in this area is US adherence-risk models (logistic regression, boosting) — not deep learning. |
| 4.3 | How do we compare? | On our own data, a fixed-interval method (the closest proxy for what competitors publicly describe) reaches **45.7%** within ±7 days with **33%** too-early reminders; our engine reaches **56.7%** with 24% too early. We cannot measure competitors' real accuracy because none publish it. |

---

## 1. What the system actually runs today

The prompt mentioned "XGBoost and median gradient descent". The precise picture:

| Component | What it is | Role |
|---|---|---|
| **Production model** `hgb-20260518-20261103071906` | scikit-learn **HistGradientBoostingRegressor** (the same family as XGBoost / LightGBM), loss = `absolute_error` → it predicts the **median** refill interval, robust to outliers | Predicts the next refill date for patients with a regular history (PATH_A, "ML" source) |
| Features | 44 point-in-time features: personal cadence (last / median / rolling intervals, cadence drift), quantity behaviour (stock-up, partial purchase, early top-up, post-lapse, anomaly), days-of-supply estimate, day-of-month habit, cross-item visit behaviour, salt-level norms, dose supply days | Same code for training and serving (no train/serve skew) |
| **Days-of-supply rule** | Pack size ÷ estimated daily use | Patients with short history (PATH_B, "DOS" source) |
| **Baseline** `baseline-median-v1` | Personal historical median interval, no ML | Fallback and the reference every new model must beat |
| **XGBoost** | `ChronicSpecializedXGBoost`, Sep 2026 benchmark | **Offline benchmark only, not in production.** It reached 50.9% within ±7 d vs the median baseline's 51.4% on that test, so it was not promoted |
| **Promotion gate** | Out-of-time evaluation, ≥ 2 pp gain with significant 95% CI, MAE / early / late / segment checks | No model reaches production without beating the baseline |
| Reminder lifecycle | 7 stages (−7, −3, −1, 0, +2, +5, +45 days), repurchase cancels pending stages, Med-Sync (one message for several medicines), pharmacist approval, dry-run default | Turns a predicted date into messages |

---

## 2. Current accuracy (real database)

### 2.1 Production model — out-of-time test, 18 Jun – 17 Jul 2026, 1,618 refills

| Method | Within ±3 d | **Within ±7 d** | MAE | Median error | Too early (spam risk) | Too late (stock-out risk) |
|---|---|---|---|---|---|---|
| Days-of-supply rule (≈ fixed interval) | 22.9% | 45.7% | 13.1 d | 8 d | 33.3% | 21.0% |
| Personal historical median | 32.6% | 52.5% | 14.6 d | 7 d | 21.1% | 26.3% |
| Baseline engine | 32.4% | 53.6% | 11.2 d | 7 d | 23.6% | 22.8% |
| **Production model (HGB)** | **35.7%** | **56.7%** | **10.4 d** | **6 d** | 24.4% | **19.0%** |

Earlier sandbox test (18 Jul – 16 Aug 2026, 1,740 refills): model 62.2%, with blend 62.8%, baseline 59.5%. The "58–60%" range quoted is between these two test months; accuracy moves by a few points from month to month.

### 2.2 By segment (production model vs baseline, same test)

| Segment | n | Model within ±7 d | Baseline | Model MAE |
|---|---|---|---|---|
| Regular patients (stability HIGH) | 308 | **72.1%** | 70.5% | 7.2 d |
| 12+ purchases | 1,009 | 62.0% | 58.4% | 8.7 d |
| 6–11 purchases | 297 | 51.2% | 45.1% | 12.3 d |
| 3–5 purchases (DOS path) | 312 | 44.6% | 40.7% | 14.1 d |
| Insulin | 44 | 52.3% | 52.3% | 10.5 d |
| Medium stability | 1,310 | 53.1% | 48.3% | 11.1 d |

**Coverage:** the engine makes a prediction for **53.9%** of refill events. The rest are acute / one-off / too irregular and are deliberately not reminded. This is a design choice that protects against spam, but it also means half of the store's purchases are out of scope.

---

## 3. Question 1 — Is ~57–60% good enough in real time?

**Yes, for go-live — with the right reading of the number.**

1. **The metric is strict.** "Within ±7 days" means the prediction landed inside a 15-day window around the actual purchase date. The median error is 6 days.
2. **Errors are not all harmful.**
   - *Too early (24%)*: the patient gets a reminder while still having medicine. Cost: mild annoyance, opt-out risk. The −7/−3/−1 stages are framed as "coming due", which softens this.
   - *Too late (19%)*: the patient could run out before the reminder. This is the costly error, and the model **cuts it from 26% (personal median) to 19%**.
   - If the patient buys first, the repurchase cancels all pending stages, so no wrong "you are due" message is sent.
3. **The reminder sequence widens the effective window.** Stages run from 7 days before to 5 days after the predicted date (plus a +45-day win-back), so a patient whose real date is within roughly a week either side is still reached at a sensible moment.
4. **There is a ceiling set by human behaviour.** Even the most regular patients reach 72%. People buy early before travel, late after a hospital stay, at another pharmacy, or under a family member's name. No model removes that noise.
5. **The comparison that matters** is against what a store would otherwise do: a fixed monthly reminder (≈ 45.7%) or nothing. The engine is clearly better on accuracy and on stock-out risk.

**What to measure once live (business KPIs, more important than ±7 d):**

| KPI | Definition | Suggested starting target |
|---|---|---|
| Reminder → refill conversion | Patients who refill at this store within 10 days after a reminder | Track vs a holdout group (no reminder) |
| Uplift | Conversion (reminded) − conversion (holdout) | > 0 with significance |
| Opt-out (STOP) rate | STOPs ÷ messaged patients per month | < 2% |
| Stock-out proxy | Refills > 7 days after predicted date | ≤ 19% (today's test) |
| Delivery failure rate | WhatsApp failed ÷ sent | < 5% |

---

## 4. Question 2 — Neural networks / deep learning?

**Recommendation: do not switch now. Keep gradient boosting; add targeted experiments.**

Reasons:
1. **Our data is tabular.** Each prediction is a row of engineered features. The NeurIPS 2022 benchmark (Grinsztajn et al., 45 datasets) found tree-based models (XGBoost, Random Forest) still beat deep learning on medium-sized tabular data, partly because neural networks struggle with irregular patterns and uninformative features.
2. **The bottleneck is data, not model power.** The biggest known error sources are: shared phone numbers (1,042 phone+medicine pairs billed under two names), no prescribed dose at billing, purchases at other pharmacies, and missing days of uploads. A bigger model learns the same noise.
3. **Operational cost.** Deep models need GPUs or longer CPU training, are harder to explain to pharmacists, and add version-pinning risk (we already saw a model become unloadable after a scikit-learn upgrade). With 200+ stores retrained monthly, simplicity matters.
4. **No evidence in this domain.** We found no published study showing LSTM / transformer models beating boosting for refill timing from pharmacy sales data. Published industry models (Walgreens, CVS-related research) use logistic regression and boosting.

**When deep learning could make sense later:**

| Option | When | Why |
|---|---|---|
| Sequence model (transformer / GRU over each patient's purchase history) | After pooling data from many stores (millions of events) | Learns patterns across stores; could help new and irregular patients |
| TabPFN (tabular foundation model, Nature 2025) | New stores with little history (< 10k rows) | Strong on small tabular data; slower to predict; test as a cold-start option |
| Survival / time-to-event model (e.g. gradient-boosted survival, XGBoost-AFT) | Next model iteration | Handles patients who have not refilled yet (censored data) properly — more relevant than "deep" |

Any of these must pass the same promotion gate before use.

---

## 5. Question 3 — How to improve, and is it ready for 200+ stores?

### 5.1 Verdict

- **Model and engine: YES — good to go** (pilot first, then roll out store by store).
- **Platform for 200+ stores: NOT YET.** It runs as a single-store app on SQLite with local file paths. The blockers are already itemised in [GO_LIVE_CHECKLIST.md](GO_LIVE_CHECKLIST.md), [PLAN_BACKEND_DEVOPS_TENANCY.md](PLAN_BACKEND_DEVOPS_TENANCY.md) and [PRODUCTION_READINESS_SCENARIO_1.md](PRODUCTION_READINESS_SCENARIO_1.md).

### 5.2 Must-fix before 200+ stores (platform)

| # | Item | Reference |
|---|---|---|
| P1 | Tenant / store control plane; every request scoped to one store | Checklist B1 |
| P2 | Mediastra service JWT verification (no API keys in the UI) | B2, decision D2 |
| P3 | PostgreSQL (RDS), one schema per store + control schema | D3 |
| P4 | Remove hard-coded local paths and legacy model files; models and uploads on S3 | B3, Scenario-1 M1–M4 |
| P5 | Asynchronous upload → worker job (large files must not block the API) | D1 |
| P6 | Per-store monthly retrain jobs with queueing and memory guards (200 models / month) | Scenario-1 M-items |
| P7 | One WhatsApp (WABA) account per tenant; STOP tenant-wide; per-tenant secrets | D4 |
| P8 | Monitoring and alerts: per-store accuracy drift, upload freshness, send failures | Checklist C |
| P9 | DPDP compliance: consent record, retention / deletion policy, phone masking in logs | Checklist A |
| P10 | Cold start: a new store needs an initial history import; until then baseline / DOS only | New |

### 5.3 Accuracy improvements (ranked by expected value ÷ effort)

| Rank | Improvement | Expected effect | Effort |
|---|---|---|---|
| 1 | **Shared-phone identity fix** (classify same-patient vs two-patients, link records, one message per phone per day) | Fixes broken histories of loyal chronic patients; removes wrong "due" messages | Medium |
| 2 | **Retrain with Oct 2026 data** and run monthly retrains | Keeps the model current | Low |
| 3 | **Asymmetric send timing**: send at an earlier quantile (e.g. P30) for high-risk drugs (insulin, cardiac) | Fewer stock-outs at the cost of slightly earlier messages | Low–Medium |
| 4 | **Capture dose / days-prescribed at billing** (store process) | The single strongest signal; would move short-history patients sharply | Store-side |
| 5 | **Outcome learning**: use reminder → refill outcomes and a holdout group to tune timing and stages | Optimises conversion, the real business goal | Medium |
| 6 | **Pooled multi-store model** with store-level features | Better cold start and rare medicines | Medium (after P3) |
| 7 | **Survival model** experiment | Correct handling of not-yet-refilled patients | Medium |
| 8 | **Coverage review**: safely bring part of the 46% unpredicted events into scope | More patients reminded | Medium |
| 9 | Brand / generic switch handling at salt level (verify current behaviour first) | Keeps history when a brand changes | Low–Medium |

---

## 6. Question 4 — What competitors use

### 6.1 Public evidence by company

| Company / product | What is publicly visible | Likely method | Evidence quality |
|---|---|---|---|
| **Apollo Pharmacy / Apollo 24\|7** | Medicine ordering, e-prescriptions, prescription upload, delivery. No public documentation of a refill-reminder feature or its logic found | Unknown | Low |
| **PharmEasy** | Subscription service for chronic medicines: "monthly reminder for refills" and scheduled delivery | Fixed schedule (subscription interval) | Medium (own marketing page) |
| **Tata 1mg** | Refill reminders mentioned in 2026 comparison articles. Published ML work covers warehouse auto-replenishment (ARIMA, LSTM), delivery ETA and chat classification — **not** patient refill timing | Reminders: unknown / likely rule-based | Low–Medium |
| **Netmeds** | Refill reminders and auto-subscription refill described in launch coverage and listicles | Subscription / fixed interval | Low |
| **MedPlus** | "Automatic refill" (recurring orders) mentioned in a low-authority roundup | Recurring order schedule | Low |
| **Truemeds** | Cheaper same-salt substitutes, monthly chronic medicines; no confirmed refill-reminder feature | — | Low |
| **Marg ERP** (pharmacy billing software; likely the "Merg" mentioned) | "Promise Order" SMS / WhatsApp when stock arrives; "Automatic Prescription Reminders on WhatsApp" | Rule-based reminders from billing data | Medium (vendor pages) |
| WhatsApp SaaS for chemists (Go4whatsup, RichAutomate, WappBlaster, Picky Assist) | Scheduled refill reminders; Go4whatsup sends **~25 days after purchase** | Fixed interval | Medium (vendor pages) |
| Medisafe (consumer app) | User enters pill count; reminder when "pills left" drops below a threshold | User-entered days of supply | Medium (own FAQ) |
| CVS Health (US) | ML personalisation engine (Databricks) to raise refill rates; research comparing logistic regression with boosting for adherence | ML (boosting / personalisation), adherence-focused | Medium |
| Walgreens (US) | Multinomial logistic regression to predict adherence and target pharmacist calls (40% calls suppressed, no adherence loss) | Logistic regression | Medium (conference poster, 2018) |
| Clover Health (US) | Patent: ML predicts which members will miss a refill and sends targeted reminders; picks channel from response history | ML (unspecified) | Patent |

### 6.2 What kind of models they use

- **Indian pharmacies and chemist tools:** every public description points to **fixed intervals, subscriptions or user-entered pill counts**. We found no public evidence of XGBoost, gradient boosting or deep learning being used for refill **timing** by Apollo, PharmEasy, 1mg, Netmeds, MedPlus, Truemeds or Marg.
- **US chains:** ML is used for **adherence risk** (who will miss a refill) and **personalisation** (who to contact, which channel), mainly **logistic regression and boosting**.
- **Deep learning:** used by 1mg for **warehouse demand forecasting** (LSTM), not for patient reminders. No public case of deep learning for refill timing in production.
- Caveat: large companies may run unpublished internal models. "Not public" is not proof that they do not exist.

### 6.3 How RefillCare compares

| Aspect | Typical competitor (public evidence) | RefillCare |
|---|---|---|
| Timing method | Fixed interval / subscription (e.g. 25–30 days) | Personal ML prediction per patient and medicine, with days-of-supply and baseline fallbacks |
| Accuracy on our data | Fixed-interval proxy: **45.7%** within ±7 d, 33% too early | **56.7%** within ±7 d, 24% too early, 19% too late |
| Quantity-aware (stock-up, partial pack, early top-up) | Not described | Yes (features) |
| Multiple medicines | Separate reminders or one subscription | Med-Sync: one message per patient |
| Safety | Not described | Pharmacist approval, dry run, idempotent sends, STOP handling, promotion gate, one-click rollback |
| Model governance | Not described | Registry, out-of-time evaluation, significance test, segment checks |
| Consumer ordering / payment / delivery link | Yes (apps) | **No** — reminder only; ordering happens at the store |
| Auto-refill subscription | Yes (PharmEasy, Netmeds, MedPlus) | **No** |
| Prescription upload / renewal | Yes (apps) | **No** |
| Multi-store scale | Yes (national chains) | **Not yet** (section 5.2) |

**Bottom line:** on prediction quality and safety, RefillCare is ahead of what competitors publicly describe. On the surrounding product (ordering, payment, subscriptions, app) and on multi-store scale, the national chains are ahead. Possible additions after go-live: a "reply 1 to reserve your medicine" WhatsApp flow and an opt-in auto-refill list for very regular patients (stability HIGH, 72% accurate).

---

## 7. Scenario coverage

| Scenario | Handled today | Gap / action |
|---|---|---|
| New patient (1–2 purchases) | Not predicted (no reliable cadence) | Optional: DOS reminder once dose data exists |
| Short history (3–5 purchases) | DOS path, 44.6% within ±7 d | Improves with dose capture (5.3 #4) |
| Regular chronic patient | ML path, up to 72% | — |
| Stock-up / partial pack / early top-up / post-lapse | Model features | — |
| Irregular / acute medicines | Excluded (not reminded) | Coverage review (5.3 #8) |
| Several medicines due together | Med-Sync, one message | — |
| Patient refills before reminder | Repurchase cancels pending stages | Depends on upload freshness (below) |
| **Sales upload delayed or missed** | Daily re-upload is safe (dedup by line key) | **Run sends only after the day's upload**, or patients who already bought get reminders. Add a freshness check before the send job |
| Bill edited after upload | New line added, old one kept | Upload only finished days; roll back and re-upload corrected days |
| **Shared phone, two names, same medicine** | Treated as two patients | Identity-link fix (5.3 #1) |
| Purchase at another pharmacy | Looks like a lapse | Cannot see; +45-day win-back handles it gracefully |
| Brand / generic switch | Salt-level features exist | Verify whether history carries over to the new item |
| STOP / opt-out | Honoured; tenant-wide in live (D4) | — |
| WhatsApp send failure | Logged | Add retry and alerting (P8) |
| Model gets worse after retrain | Gate blocks it; rollback available (skips models without a file) | Per-store drift alert (P8) |
| Library upgrade breaks a model | Library versions recorded with the model | Retrain all stores before deploying an upgrade (requirements.txt rule) |
| New store with no history | Baseline / nothing | Initial history import; pooled model later (P10, 5.3 #6) |
| Seasonality / festivals | `purchase_month` feature | Monitor month-to-month accuracy |
| Data protection | Phones masked in logs, local data git-ignored | DPDP consent and retention policy (P9) |

---

## 8. Final recommendation

1. **Go live with the current model** in a controlled pilot at the first stores, with pharmacist approval on and a no-reminder holdout group to measure real uplift.
2. **Do not invest in deep learning now.** Spend the effort on the shared-phone identity fix, monthly retraining, dose capture and asymmetric send timing.
3. **Build the multi-tenant platform (section 5.2) before onboarding 200+ stores.** Roll out in waves (e.g. 5 → 25 → 100 → all) with per-store accuracy and opt-out monitoring.
4. Re-run this comparison after 3 months of live data, measured on conversion and uplift, not only ±7-day accuracy.

---

## Sources

Internal: `ml_evaluation_runs` run `TEST-20261103071907-11f5`; `ml_model_versions`; `data/refillcare/processed/chronic_model_benchmark_report.json`; [implementation_plan.md](../implementation_plan.md).

External:
- [PharmEasy — subscription / refill reminders](https://pharmeasy.in//appointment)
- [Apollo 24|7 — App Store listing](https://apps.apple.com/in/app/apollo-247-health-medicine/id1496740273) · [Apollo Pharmacy FAQ](https://apollopharmacy.in/faq)
- [ML-driven auto-replenishment at Tata 1mg (ARIMA / LSTM, warehouse)](https://yash-singh-pathania.github.io/posts/ml-auto-replenishment-odin) · [1mg engineering — ETA estimation](https://medium.com/1mgofficial/estimating-the-optimum-estimated-time-of-arrival-b3ee71870dd6)
- [Netmeds app launch — refill reminders (Medical Dialogues)](https://medicaldialogues.in/netmeds-to-launch-e-pharmacy-mobile-app-for-android-and-ios-user) · [Instant medicine delivery apps (Shiprocket)](https://www.shiprocket.in/blog/instant-medicine-delivery-apps/amp/) · [Smart way to refill monthly medicines (Nagpur Today, 2026)](https://www.nagpurtoday.in/the-smart-way-to-refill-monthly-medicines-without-missing-doses/) · [Truemeds — About](https://www.truemeds.in/about-us) · [Top online healthcare apps (Kashmir Reader, 2024)](https://kashmirreader.com/2024/06/13/top-online-healthcare-apps-to-trust-for-your-medication-needs/)
- [Marg ERP — Promise Order](https://margcompusoft.com/m/?p=45672) · [Marg — medical store software](https://margcompusoft.com/retail/medical_store_software.html)
- [Go4whatsup — pharmacy](https://www.go4whatsup.com/industries/pharmacy/) · [RichAutomate — WhatsApp API for pharmacies](https://richautomate.in/blog/best-whatsapp-business-api-for-pharmacies-india-2026) · [WappBlaster](https://wappblaster.com/pl/partner/pharmacy-management-software-expiry-alerts) · [Picky Assist — refill reminders](https://pickyassist.com/blog/how-medication-refill-reminders-improve-patient-retention/)
- [Medisafe — refill reminder FAQ](https://www.medisafe.com/faq/how-do-i-set-a-refill-reminder-for-an-existing-med-2)
- [CIO Dive — CVS personalisation with AI/ML](https://www.ciodive.com/news/CVS-personalization-AI-ML/611270/) · [Emerj — AI at CVS Health](https://emerj.com/artificial-intelligence-at-cvs-health/) · [Franklin et al. 2016 — CVS Caremark adherence prediction](https://scholar.harvard.edu/sites/scholar.harvard.edu/files/alexiskrumme/files/franklin_et_al-2016-health_services_research.pdf) · [ISPOR 2018 — Walgreens predictive adherence model](https://www.ispor.org/heor-resources/presentations-database/presentation/ispor-2018/predictive-model-directed-interventions-improve-medication-adherence-program-efficiency)
- [Clover Health refill-prediction patent (Retail Banker International)](https://www.retailbankerinternational.com/data-insights/clover-health-investments-gets-grant-for-predicting-medication-refills-and-sending-targeted-reminders-to-users/)
- [Grinsztajn et al., NeurIPS 2022 — Why tree-based models still outperform deep learning on tabular data](https://arxiv.org/abs/2207.08815)
- [Hollmann et al., Nature 2025 — TabPFN](https://pubmed.ncbi.nlm.nih.gov/39780007/)
- [Gradient boosting for survival analysis (GBMCI)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3853154/) · [scikit-survival GradientBoostingSurvivalAnalysis](https://scikit-survival.readthedocs.io/en/v0.20.0/api/generated/sksurv.ensemble.GradientBoostingSurvivalAnalysis.html)
- [arXiv 2504.14469 — medication adherence prediction](https://arxiv.org/pdf/2504.14469)
