/**
 * RefillCare Enterprise REST API Client
 */

const API_BASE_URL = window.location.origin;

export const ApiClient = {
  // 1. Analytics & Health
  async getHealth() {
    const res = await fetch(`${API_BASE_URL}/health`);
    return await res.json();
  },

  async getKpis() {
    const res = await fetch(`${API_BASE_URL}/api/v1/analytics/kpi`);
    if (!res.ok) throw new Error("Failed to load operations KPIs");
    return await res.json();
  },

  // 2. Sales Ingestion
  async previewSalesFile(file) {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch(`${API_BASE_URL}/api/v1/sales/preview`, {
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
    const res = await fetch(`${API_BASE_URL}/api/v1/sales/ingest`, {
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
    const res = await fetch(`${API_BASE_URL}/api/v1/sales/batches`);
    if (!res.ok) throw new Error("Failed to fetch import batches");
    return await res.json();
  },

  async rollbackBatch(batchId) {
    const res = await fetch(`${API_BASE_URL}/api/v1/sales/batches/${batchId}/rollback`, {
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
    const res = await fetch(`${API_BASE_URL}/api/v1/predictions/generate`, {
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
    const res = await fetch(`${API_BASE_URL}/api/v1/predictions/snapshots?limit=${limit}`);
    if (!res.ok) throw new Error("Failed to fetch prediction snapshots");
    return await res.json();
  },

  async evaluateOutcomes() {
    const res = await fetch(`${API_BASE_URL}/api/v1/predictions/evaluate`, {
      method: "POST",
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Evaluation failed");
    }
    return await res.json();
  },

  // 4. Reminders
  async getDailyReminders(targetDate, mobileStatus = "All") {
    let url = `${API_BASE_URL}/api/v1/reminders/daily?mobile_status=${mobileStatus}`;
    if (targetDate) url += `&target_date=${targetDate}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error("Failed to fetch reminders");
    return await res.json();
  },

  getExportCsvUrl(targetDate) {
    let url = `${API_BASE_URL}/api/v1/reminders/export-csv`;
    if (targetDate) url += `?target_date=${targetDate}`;
    return url;
  },

  // 5. Models
  async getModels() {
    const res = await fetch(`${API_BASE_URL}/api/v1/models`);
    if (!res.ok) throw new Error("Failed to fetch models");
    return await res.json();
  },

  async activateModel(versionId) {
    const res = await fetch(`${API_BASE_URL}/api/v1/models/${versionId}/activate`, {
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
    const res = await fetch(`${API_BASE_URL}/api/v1/whatsapp/dispatch`, {
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
    const res = await fetch(`${API_BASE_URL}/api/refill-decisions${query ? `?${query}` : ""}`);
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

    const res = await fetch(url);
    if (!res.ok) throw new Error("Failed to fetch review queue");
    return await res.json();
  },

  async approveReminder(reminderId) {
    const res = await fetch(`${API_BASE_URL}/api/reminders/${reminderId}/approve`, {
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
    const res = await fetch(`${API_BASE_URL}/api/reminders/${reminderId}/reject`, {
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
    const res = await fetch(`${API_BASE_URL}/api/reminders/${reminderId}/dispatch`, {
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
    const res = await fetch(`${API_BASE_URL}/api/customers/${customerId}/refill-history`);
    if (!res.ok) throw new Error("Failed to fetch customer refill history");
    return await res.json();
  },
};

