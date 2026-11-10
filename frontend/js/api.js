/**
 * RefillCare Enterprise REST API Client
 */

const API_BASE_URL = window.location.origin;

/**
 * Store / tenant: a `?store_id=` (and `?tenant_id=`) in the page URL wins and is remembered in
 * localStorage (`refillcare_store_id` / `refillcare_tenant_id`); otherwise the saved value is used.
 * Empty = the server's default store.
 */
function initStoreFromUrl() {
  try {
    const params = new URLSearchParams(window.location.search);
    const store = (params.get("store_id") || "").trim();
    const tenant = (params.get("tenant_id") || "").trim();
    if (store) localStorage.setItem("refillcare_store_id", store.toLowerCase());
    if (tenant) localStorage.setItem("refillcare_tenant_id", tenant);
  } catch (e) { /* storage blocked: fall back to the URL value below */ }
}
initStoreFromUrl();

export function currentStoreId() {
  try {
    const fromUrl = (new URLSearchParams(window.location.search).get("store_id") || "").trim().toLowerCase();
    return fromUrl || localStorage.getItem("refillcare_store_id") || "";
  } catch (e) {
    return "";
  }
}

export function currentTenantId() {
  try {
    const fromUrl = (new URLSearchParams(window.location.search).get("tenant_id") || "").trim();
    return fromUrl || localStorage.getItem("refillcare_tenant_id") || "";
  } catch (e) {
    return "";
  }
}

/**
 * Every /api call sends `X-API-Key` (localStorage `refillcare_api_key`), the store it works on
 * (`X-Store-Id`) and, when set, the tenant (`X-Tenant-Id`).
 * The key is not needed when the browser runs on the server machine and REFILLCARE_API_KEY is unset.
 */
export function authHeaders() {
  const h = {};
  try {
    const key = localStorage.getItem("refillcare_api_key");
    if (key) h["X-API-Key"] = key;
  } catch (e) { /* ignore */ }
  const store = currentStoreId();
  if (store) h["X-Store-Id"] = store;
  const tenant = currentTenantId();
  if (tenant) h["X-Tenant-Id"] = tenant;
  return h;
}

/** Append the store to a download URL (downloads cannot send headers). */
function withStore(url) {
  const store = currentStoreId();
  if (!store) return url;
  return `${url}${url.includes("?") ? "&" : "?"}store=${encodeURIComponent(store)}`;
}

async function apiFetch(url, options = {}) {
  const opts = { ...options, headers: { ...(options.headers || {}), ...authHeaders() } };
  const res = await fetch(url, opts);
  if (res.status === 401) {
    console.warn("RefillCare API: write action needs a valid X-API-Key (localStorage refillcare_api_key).");
  }
  return res;
}

export const ApiClient = {
  // 1. Analytics & Health
  async getHealth() {
    const res = await apiFetch(`${API_BASE_URL}/health`);
    return await res.json();
  },

  async getKpis() {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/analytics/kpi`);
    if (!res.ok) throw new Error("Failed to load operations KPIs");
    return await res.json();
  },

  async getTransactionTypes() {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/analytics/transaction-types`);
    if (!res.ok) throw new Error("Failed to load transaction channel analytics");
    return await res.json();
  },

  async getConsensusRegimenAnalytics() {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/analytics/consensus-regimen`);
    if (!res.ok) throw new Error("Failed to load consensus regimen analytics");
    return await res.json();
  },


  // 2. Sales Ingestion
  async previewSalesFile(file, columnMapping = null) {
    const formData = new FormData();
    formData.append("file", file);
    if (columnMapping) formData.append("column_mapping", JSON.stringify(columnMapping));
    const res = await apiFetch(`${API_BASE_URL}/api/v1/sales/preview`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Validation failed");
    }
    return await res.json();
  },

  async saveColumnMapping(headers, mapping) {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/sales/column-mapping`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ headers, mapping }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Could not save the column mapping");
    }
    return await res.json();
  },

  async ingestSalesFile(file, columnMapping = null, force = false) {
    const formData = new FormData();
    formData.append("file", file);
    if (columnMapping) formData.append("column_mapping", JSON.stringify(columnMapping));
    if (force) formData.append("force", "true");
    const res = await apiFetch(`${API_BASE_URL}/api/v1/sales/ingest`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json();
      const e = new Error(err.detail || "Ingestion failed");
      e.status = res.status;  // 409 = this exact file was already uploaded
      throw e;
    }
    return await res.json();
  },

  async getBatches() {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/sales/batches`);
    if (!res.ok) throw new Error("Failed to fetch import batches");
    return await res.json();
  },

  async rollbackBatch(batchId) {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/sales/batches/${batchId}/rollback`, {
      method: "POST",
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Rollback failed");
    }
    return await res.json();
  },

  // 3. Predictions & Accuracy
  async generatePredictions() {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/predictions/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Prediction generation failed");
    }
    return await res.json();
  },

  async getPredictionSnapshots(limit = 100) {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/predictions/snapshots?limit=${limit}`);
    if (!res.ok) throw new Error("Failed to fetch prediction snapshots");
    return await res.json();
  },

  async evaluateOutcomes() {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/predictions/evaluate`, {
      method: "POST",
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Evaluation failed");
    }
    return await res.json();
  },

  // 4. Reminders
  async getDailyReminders(targetDate, targetMonth = null, mobileStatus = "All", options = {}) {
    let url = `${API_BASE_URL}/api/v1/reminders/daily?mobile_status=${encodeURIComponent(mobileStatus)}`;
    if (options.startDate && options.endDate) {
      url += `&start_date=${encodeURIComponent(options.startDate)}&end_date=${encodeURIComponent(options.endDate)}`;
    } else if (targetMonth) {
      url += `&target_month=${encodeURIComponent(targetMonth)}`;
    } else if (targetDate) {
      url += `&target_date=${encodeURIComponent(targetDate)}`;
    }
    if (options.stage && options.stage !== "ALL") url += `&stage=${encodeURIComponent(options.stage)}`;
    if (options.refillFunction && options.refillFunction !== "ALL") url += `&refill_function=${encodeURIComponent(options.refillFunction)}`;
    if (options.path) url += `&path=${encodeURIComponent(options.path)}`;
    if (options.customerQuery) url += `&customer_query=${encodeURIComponent(options.customerQuery)}`;
    if (options.medicationQuery) url += `&medication_query=${encodeURIComponent(options.medicationQuery)}`;
    if (options.limit) url += `&limit=${encodeURIComponent(options.limit)}`;
    if (options.offset) url += `&offset=${encodeURIComponent(options.offset)}`;

    const res = await apiFetch(url);
    if (!res.ok) throw new Error("Failed to fetch reminders");
    return await res.json();
  },

  getExportCsvUrl(targetDate, targetMonth = null) {
    let url = `${API_BASE_URL}/api/v1/reminders/export-csv`;
    if (targetMonth) {
      url += `?target_month=${encodeURIComponent(targetMonth)}`;
    } else if (targetDate) {
      url += `?target_date=${encodeURIComponent(targetDate)}`;
    }
    return withStore(url);
  },

  getExportJsonUrl(targetDate, targetMonth = null) {
    let url = `${API_BASE_URL}/api/v1/reminders/export-json`;
    if (targetMonth) {
      url += `?target_month=${encodeURIComponent(targetMonth)}`;
    } else if (targetDate) {
      url += `?target_date=${encodeURIComponent(targetDate)}`;
    }
    return withStore(url);
  },

  async getReminderMonths() {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/reminders/months`);
    if (!res.ok) return [];
    return await res.json();
  },

  // 4.1. Clinical Review Queue (Enterprise Synchronized)
  async getClinicalReviewSummary(targetMonth = null) {
    let url = `${API_BASE_URL}/api/v1/review/summary`;
    if (targetMonth) url += `?target_month=${encodeURIComponent(targetMonth)}`;
    const res = await apiFetch(url);
    if (!res.ok) throw new Error("Failed to fetch clinical review summary");
    return await res.json();
  },

  async getClinicalReviewQueue(options = {}) {
    let url = `${API_BASE_URL}/api/v1/review/queue?`;
    const params = new URLSearchParams();
    if (options.targetMonth) params.append("target_month", options.targetMonth);
    if (options.category && options.category !== "All") params.append("category", options.category);
    if (options.missingMobileOnly) params.append("missing_mobile_only", "true");
    if (options.search) params.append("search", options.search);
    if (options.limit) params.append("limit", options.limit || 200);
    if (options.offset) params.append("offset", options.offset || 0);
    url += params.toString();

    const res = await apiFetch(url);
    if (!res.ok) throw new Error("Failed to fetch clinical review queue");
    return await res.json();
  },

  // 5. Models
  async getModels() {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/models`);
    if (!res.ok) throw new Error("Failed to fetch models");
    return await res.json();
  },

  async activateModel(versionId) {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/models/${versionId}/activate`, {
      method: "POST",
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Activation failed");
    }
    return await res.json();
  },

  // 6. Pharmacy Overview + Patients Needing Review (same data as the Streamlit app)
  async getOverview() {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/overview`);
    return await jsonOrThrow(res, "Failed to load the pharmacy overview");
  },

  async getReviewCandidates(day) {
    const q = new URLSearchParams({ start: day, end: day });
    const res = await apiFetch(`${API_BASE_URL}/api/v1/reminders/review-candidates?${q}`);
    return await jsonOrThrow(res, "Failed to load patients not getting reminders");
  },

  async getReviewOther(category, day) {
    const q = new URLSearchParams({ category, day });
    const res = await apiFetch(`${API_BASE_URL}/api/v1/reminders/review-other?${q}`);
    return await jsonOrThrow(res, "Failed to load the review list");
  },

  async pharmacistAdd(body) {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/reminders/pharmacist-add`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return await jsonOrThrow(res, "Could not add the patient to reminders");
  },

  async pharmacistAddBulk(keys, note = null) {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/reminders/pharmacist-add-bulk`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ customer_item_keys: keys, note }),
    });
    return await jsonOrThrow(res, "Could not add the selected patients to reminders");
  },

  async pharmacistRemove(key) {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/reminders/pharmacist-remove`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ customer_item_key: key }),
    });
    return await jsonOrThrow(res, "Could not remove the patient from reminders");
  },

  async getPharmacistAdded() {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/reminders/pharmacist-added`);
    return await jsonOrThrow(res, "Failed to load patients added by a pharmacist");
  },

  // 6.1 WhatsApp sending (live: only today's reminders, after the Yes/No confirmation)
  async getWhatsAppSendSummary(day) {
    const q = day ? `?target_date=${encodeURIComponent(day)}` : "";
    const res = await apiFetch(`${API_BASE_URL}/api/v1/whatsapp/send-summary${q}`);
    return await jsonOrThrow(res, "Failed to load the WhatsApp send summary");
  },

  async sendWhatsAppToday(overrideReason = null) {
    const body = overrideReason ? { override_reason: overrideReason } : {};
    const res = await apiFetch(`${API_BASE_URL}/api/v1/whatsapp/send-today`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return await jsonOrThrow(res, "Sending WhatsApp messages failed");
  },

  async searchCustomers(q, limit = 25) {
    const qs = new URLSearchParams({ q, limit: String(limit) });
    const res = await apiFetch(`${API_BASE_URL}/api/customers/search?${qs}`);
    return await jsonOrThrow(res, "Patient search failed");
  },

  async setWhatsAppMessages(allowed, customerId = null, mobile = null) {
    const qs = new URLSearchParams({ allowed: allowed ? "true" : "false" });
    if (customerId) qs.append("customer_id", customerId);
    if (mobile) qs.append("mobile_number", mobile);
    const res = await apiFetch(`${API_BASE_URL}/api/customers/whatsapp-messages?${qs}`, { method: "POST" });
    return await jsonOrThrow(res, allowed ? "Could not resume WhatsApp messages" : "Could not stop WhatsApp messages");
  },

  async getDeliveryLog(day, status = "ALL") {
    const qs = new URLSearchParams({ status });
    if (day) qs.append("day", day);
    const res = await apiFetch(`${API_BASE_URL}/api/v1/whatsapp/delivery-log?${qs}`);
    return await jsonOrThrow(res, "Failed to load the messages sent");
  },

  // 7. V1 Refill Decisions & Review Queue
  async getRefillDecisions(params = {}) {
    const query = new URLSearchParams(params).toString();
    const res = await apiFetch(`${API_BASE_URL}/api/refill-decisions${query ? `?${query}` : ""}`);
    if (!res.ok) throw new Error("Failed to fetch refill decisions");
    return await res.json();
  },

  async getTodayReminderQueue(targetDate, path = "", stability = "") {
    let url = `${API_BASE_URL}/api/reminders/today`;
    const params = new URLSearchParams();
    if (targetDate) params.append("target_date", targetDate);
    if (path) params.append("path", path);
    if (stability) params.append("stability", stability);
    const qs = params.toString();
    if (qs) url += `?${qs}`;

    const res = await apiFetch(url);
    if (!res.ok) throw new Error("Failed to fetch review queue");
    return await res.json();
  },

  async approveReminder(reminderId) {
    const res = await apiFetch(`${API_BASE_URL}/api/reminders/${reminderId}/approve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Approval failed");
    }
    return await res.json();
  },

  async rejectReminder(reminderId, reason = "Pharmacist rejected") {
    const res = await apiFetch(`${API_BASE_URL}/api/reminders/${reminderId}/reject`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reason }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Rejection failed");
    }
    return await res.json();
  },

  async getCustomerRefillHistory(customerId) {
    const res = await apiFetch(`${API_BASE_URL}/api/customers/${customerId}/refill-history`);
    if (!res.ok) throw new Error("Failed to fetch customer refill history");
    return await res.json();
  },

  // 7. Med-Sync & Quantile Uncertainty
  async getMedSyncBundles(syncWindowDays = 8, customerId = null, targetMonth = null, targetDate = null, lifecycleTier = null) {
    let url = `${API_BASE_URL}/api/v2/med-sync/bundles?sync_window_days=${syncWindowDays}`;
    if (customerId) url += `&customer_id=${encodeURIComponent(customerId)}`;
    if (targetDate) {
      url += `&target_date=${encodeURIComponent(targetDate)}`;
    } else if (targetMonth) {
      url += `&target_month=${encodeURIComponent(targetMonth)}`;
    }
    if (lifecycleTier && lifecycleTier !== "ALL") {
      url += `&lifecycle_tier=${encodeURIComponent(lifecycleTier)}`;
    }
    const res = await apiFetch(url);
    if (!res.ok) throw new Error("Failed to fetch Med-Sync bundles");
    return await res.json();
  },

  async getQuantileMetrics() {
    const res = await apiFetch(`${API_BASE_URL}/api/v2/models/quantiles`);
    if (!res.ok) throw new Error("Failed to fetch Quantile uncertainty metrics");
    return await res.json();
  },

  // 8. Pharmacist Supabase Medication Rules

  async getMedicationRules(ruleType = null, search = null) {
    let url = `${API_BASE_URL}/api/v1/medications/rules?`;
    if (ruleType && ruleType !== "All") url += `rule_type=${encodeURIComponent(ruleType)}&`;
    if (search) url += `search=${encodeURIComponent(search)}&`;
    const res = await apiFetch(url);
    if (!res.ok) throw new Error("Failed to fetch medication classification rules");
    return await res.json();
  },

  async addMedicationRule(ruleData) {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/medications/rules`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(ruleData),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to add medication classification rule");
    }
    return await res.json();
  },

  async deleteMedicationRule(ruleId) {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/medications/rules/${encodeURIComponent(ruleId)}`, {
      method: "DELETE",
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to delete medication rule");
    }
    return await res.json();
  },

  async classifyMedication(data) {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/medications/classify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Classification check failed");
    }
    return await res.json();
  },

  // 9. WhatsApp Gateway Methods
  async getWhatsAppCandidates({ targetDate = null, targetMonth = null, tier = "ALL", syncWindow = 0, mobileStatus = "All", search = null } = {}) {
    let url = `${API_BASE_URL}/api/v1/whatsapp/candidates?tier=${encodeURIComponent(tier)}&sync_window=${syncWindow}&mobile_status=${encodeURIComponent(mobileStatus)}`;
    if (targetDate) url += `&target_date=${encodeURIComponent(targetDate)}`;
    if (targetMonth) url += `&target_month=${encodeURIComponent(targetMonth)}`;
    if (search) url += `&search=${encodeURIComponent(search)}`;
    const res = await apiFetch(url);
    if (!res.ok) throw new Error("Failed to load WhatsApp candidate queue");
    return await res.json();
  },

  async previewWhatsApp(data) {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/whatsapp/preview`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to generate WhatsApp preview");
    }
    return await res.json();
  },

  // 10. Patient WhatsApp consent
  async getOptedOutCustomers() {
    const res = await apiFetch(`${API_BASE_URL}/api/customers/opt-outs`);
    if (!res.ok) throw new Error("Failed to load opted-out patients");
    return await res.json();
  },

  async setCustomerConsent(customerId, optIn) {
    const action = optIn ? "opt-in" : "opt-out";
    const res = await apiFetch(`${API_BASE_URL}/api/customers/${encodeURIComponent(customerId)}/${action}`, { method: "POST" });
    return await jsonOrThrow(res, `Failed to ${action} patient`);
  },

  // 11. ML lifecycle (registry, dynamic windows, evaluations, gate, retrain)
  async getMlPlan() {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/ml/plan`);
    return await jsonOrThrow(res, "Failed to load ML train/validation/test plan");
  },

  async getMlHealth() {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/ml/health`);
    return await jsonOrThrow(res, "Failed to load ML health");
  },

  async getMlEvaluations(limit = 10) {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/ml/evaluations?limit=${limit}`);
    return await jsonOrThrow(res, "Failed to load evaluation runs");
  },

  async getMlGate(versionId) {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/ml/models/${encodeURIComponent(versionId)}/gate`);
    return await jsonOrThrow(res, "Failed to check promotion gate");
  },

  async startMlRetrain(promoteBest = false) {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/ml/retrain`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ promote_best: promoteBest }),
    });
    return await jsonOrThrow(res, "Failed to start retrain cycle");
  },

  // 12. Prediction settings, rule A/B, rolling test, identity cut-over, column-mapping profiles
  async getMlSettings() {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/ml/settings`);
    return await jsonOrThrow(res, "Failed to load prediction settings");
  },

  async startMlJob(kind) {
    const path = kind === "rolling" ? "rolling" : "rules-ab";
    const res = await apiFetch(`${API_BASE_URL}/api/v1/ml/${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    return await jsonOrThrow(res, `Failed to start ${path}`);
  },

  async getMlReport(kind) {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/ml/reports/${kind}`);
    if (res.status === 404) return null;
    return await jsonOrThrow(res, `Failed to load ${kind} report`);
  },

  async getRekeyDryRun() {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/ml/identity/rekey`);
    return await jsonOrThrow(res, "Failed to run the re-key dry run");
  },

  async applyRekey() {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/ml/identity/rekey`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm: true }),
    });
    return await jsonOrThrow(res, "Failed to apply the re-key");
  },

  async getDataStatus() {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/sales/data-status`);
    return await jsonOrThrow(res, "Failed to load sales data status");
  },

  async retryBatchSync(batchId) {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/sales/batches/${encodeURIComponent(batchId)}/retry-sync`, { method: "POST" });
    return await jsonOrThrow(res, "Retry failed");
  },

  async rollbackModel() {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/ml/models/rollback`, { method: "POST" });
    return await jsonOrThrow(res, "Model rollback failed");
  },

  async getColumnMappings() {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/sales/column-mapping/profiles`);
    return await jsonOrThrow(res, "Failed to load saved column mappings");
  },

  async deleteColumnMapping(profileId) {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/sales/column-mapping/profiles/${encodeURIComponent(profileId)}`, { method: "DELETE" });
    return await jsonOrThrow(res, "Failed to delete the column mapping");
  },
};

async function jsonOrThrow(res, fallbackMessage) {
  if (res.ok) return await res.json();
  let detail = fallbackMessage;
  try {
    const err = await res.json();
    const d = err.detail;
    if (typeof d === "string") detail = d;
    else if (d && d.message) detail = `${d.message} ${(d.decision?.failed_checks || []).join(", ")}`.trim();
  } catch (e) { /* keep fallback */ }
  if (res.status === 401 || res.status === 403) detail += " (set the write API key in Settings)";
  const e = new Error(detail);
  e.status = res.status;  // e.g. 409 = sales data stale (send-today)
  throw e;
}



