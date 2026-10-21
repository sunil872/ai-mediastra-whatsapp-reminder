/**
 * RefillCare Enterprise REST API Client
 */

const API_BASE_URL = window.location.origin;

/**
 * v1 write protection: write actions send `X-API-Key` (value stored once in localStorage under
 * `refillcare_api_key`, e.g. `localStorage.setItem("refillcare_api_key", "<key>")`).
 * Not needed when the browser runs on the server machine and REFILLCARE_API_KEY is unset.
 */
export function authHeaders() {
  const h = {};
  try {
    const key = localStorage.getItem("refillcare_api_key");
    if (key) h["X-API-Key"] = key;
  } catch (e) { /* ignore */ }
  return h;
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
  async previewSalesFile(file) {
    const formData = new FormData();
    formData.append("file", file);
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

  async ingestSalesFile(file) {
    const formData = new FormData();
    formData.append("file", file);
    const res = await apiFetch(`${API_BASE_URL}/api/v1/sales/ingest`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Ingestion failed");
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
    return url;
  },

  getExportJsonUrl(targetDate, targetMonth = null) {
    let url = `${API_BASE_URL}/api/v1/reminders/export-json`;
    if (targetMonth) {
      url += `?target_month=${encodeURIComponent(targetMonth)}`;
    } else if (targetDate) {
      url += `?target_date=${encodeURIComponent(targetDate)}`;
    }
    return url;
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

  // 6. WhatsApp
  async dispatchWhatsApp(reminderDate, dryRun = true, batchSize = 50) {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/whatsapp/dispatch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        reminder_date: reminderDate,
        dry_run: dryRun,
        batch_size: batchSize,
      }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "WhatsApp dispatch failed");
    }
    return await res.json();
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

  async dispatchReminder(reminderId, isDryRun = true) {
    const res = await apiFetch(`${API_BASE_URL}/api/reminders/${reminderId}/dispatch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ is_dry_run: isDryRun }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Dispatch failed");
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

  async sendSingleWhatsApp(data) {
    const res = await apiFetch(`${API_BASE_URL}/api/v1/whatsapp/send-single`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to dispatch test message");
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
  throw new Error(detail);
}



