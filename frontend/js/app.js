/**
 * RefillCare Enterprise Frontend Application
 */

import { ApiClient } from "./api.js";

// Global State
let activeFile = null;

// Initialize Application
document.addEventListener("DOMContentLoaded", async () => {
  initTheme();
  setupNavigation();
  setupUploadDropzone();
  setupActionListeners();
  setupQuickActions();
  window.loadMedRules = setupMedicationRules();
  
  // Dynamic default date (defaults to current date e.g. 2026-09-30)
  const todayStr = new Date().toISOString().split("T")[0];
  const dateInput = document.getElementById("reminder-target-date");
  if (dateInput) dateInput.value = todayStr;

  const medsyncDateInput = document.getElementById("medsync-target-date");
  if (medsyncDateInput) medsyncDateInput.value = todayStr;

  const reviewDateInput = document.getElementById("review-target-date");
  if (reviewDateInput) reviewDateInput.value = todayStr;

  const waDateInput = document.getElementById("wa-target-date");
  if (waDateInput) waDateInput.value = todayStr;

  await initReminderMonthOptions();
  await loadInitialData();
});

function initTheme() {
  const savedTheme = localStorage.getItem("mediastra-theme") || "light";
  setTheme(savedTheme);

  const btn = document.getElementById("btn-theme-toggle");
  if (btn) {
    btn.addEventListener("click", () => {
      const current = document.documentElement.getAttribute("data-theme") || "light";
      const next = current === "light" ? "dark" : "light";
      setTheme(next);
    });
  }
}

function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("mediastra-theme", theme);
  const icon = document.getElementById("theme-btn-icon");
  const label = document.getElementById("theme-btn-label");
  if (icon && label) {
    if (theme === "dark") {
      icon.innerText = "☀️";
      label.innerText = "Light Mode";
    } else {
      icon.innerText = "🌙";
      label.innerText = "Dark Mode";
    }
  }
}

function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  const toast = document.createElement("div");
  toast.className = `toast ${type === "error" ? "badge-danger" : (type === "success" ? "badge-success" : "")}`;
  toast.innerText = message;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

// ------------------------------------------------------------------------------
// Navigation & Tab Switching
// ------------------------------------------------------------------------------
function setupNavigation() {
  const navItems = document.querySelectorAll(".nav-item");
  const tabPanes = document.querySelectorAll(".tab-pane");
  const titleElem = document.getElementById("current-page-title");
  const breadcrumbElem = document.getElementById("breadcrumb-current");

  navItems.forEach((item) => {
    item.addEventListener("click", () => {
      const tabId = item.getAttribute("data-tab");
      navItems.forEach((n) => n.classList.remove("active"));
      tabPanes.forEach((p) => p.classList.remove("active"));

      item.classList.add("active");
      const targetPane = document.getElementById(tabId);
      if (targetPane) targetPane.classList.add("active");

      const label = item.querySelector("span:nth-child(2)")?.innerText || "Dashboard";
      if (titleElem) titleElem.innerText = label;
      if (breadcrumbElem) breadcrumbElem.innerText = label;

      // Lazy load tab data
      if (tabId === "tab-medrules" && window.loadMedRules) window.loadMedRules();
      if (tabId === "tab-upload") loadBatches();
      if (tabId === "tab-predictions") loadPredictionSnapshots();
      if (tabId === "tab-reminders") loadReminders();
      if (tabId === "tab-medsync") loadMedSyncBundles();
      if (tabId === "tab-review") loadReviewQueue();
      if (tabId === "tab-models") loadModels();
      if (tabId === "tab-whatsapp") loadWhatsAppGateway();
    });
  });
}

// ------------------------------------------------------------------------------
// Data Loading Functions
// ------------------------------------------------------------------------------
async function loadInitialData() {
  try {
    const kpis = await ApiClient.getKpis();
    if (document.getElementById("kpi-customers")) document.getElementById("kpi-customers").innerText = Number(kpis.total_customers_monitored).toLocaleString();
    if (document.getElementById("kpi-transactions")) document.getElementById("kpi-transactions").innerText = Number(kpis.total_sales_transactions).toLocaleString();
    if (document.getElementById("kpi-upcoming")) document.getElementById("kpi-upcoming").innerText = Number(kpis.upcoming_reminders_count).toLocaleString();
    if (document.getElementById("kpi-due-today")) document.getElementById("kpi-due-today").innerText = Number(kpis.reminders_due_today ?? 0).toLocaleString();
    if (document.getElementById("kpi-chronic-count")) document.getElementById("kpi-chronic-count").innerText = `${kpis.chronic_medicines_count ?? 0}+`;
    if (document.getElementById("kpi-latest-date")) document.getElementById("kpi-latest-date").innerText = kpis.latest_sales_date;
    if (document.getElementById("sales-coverage-badge")) document.getElementById("sales-coverage-badge").innerText = kpis.sales_coverage_date_range;
    if (document.getElementById("sidebar-active-model")) document.getElementById("sidebar-active-model").innerText = kpis.active_model_version;

    // Load Transaction Channel Breakdown
    try {
      const channelStats = await ApiClient.getTransactionTypes();
      const customerElem = document.getElementById("kpi-chan-customer");
      const b2bElem = document.getElementById("kpi-chan-b2b");
      const unknownElem = document.getElementById("kpi-chan-unknown");
      const excludedElem = document.getElementById("kpi-chan-excluded");

      if (customerElem) customerElem.innerText = Number(channelStats.customer_sales).toLocaleString();
      if (b2bElem) b2bElem.innerText = Number(channelStats.b2b_inter_store).toLocaleString();
      if (unknownElem) unknownElem.innerText = Number(channelStats.unknown).toLocaleString();
      if (excludedElem) excludedElem.innerText = Number(channelStats.excluded_from_refillcare).toLocaleString();
    } catch (chanErr) {
      console.warn("Could not load transaction channel stats:", chanErr);
    }
  } catch (err) {
    console.error("Error loading KPIs:", err);
    showToast("Error connecting to RefillCare backend", "error");
  }
}

async function loadBatches() {
  try {
    const batches = await ApiClient.getBatches();
    const tbody = document.querySelector("#table-import-batches tbody");
    tbody.innerHTML = "";

    if (!batches || batches.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; color: var(--text-muted);">No import batches recorded yet.</td></tr>`;
      return;
    }

    batches.forEach((b) => {
      const tr = document.createElement("tr");
      const isRolledBack = b.processing_status === "ROLLED_BACK";
      tr.innerHTML = `
        <td><code>${b.import_batch_id}</code></td>
        <td>${b.source_filename}</td>
        <td>${new Date(b.upload_timestamp).toLocaleString()}</td>
        <td>${b.detected_date_range || "-"}</td>
        <td><strong>${Number(b.records_inserted).toLocaleString()}</strong></td>
        <td><span class="badge ${isRolledBack ? 'badge-danger' : 'badge-success'}">${b.processing_status}</span></td>
        <td>
          <button class="btn btn-outline btn-sm btn-rollback" data-batch="${b.import_batch_id}" ${isRolledBack ? 'disabled' : ''}>
            ${isRolledBack ? 'Rolled Back' : '↺ Undo Import'}
          </button>
        </td>
      `;
      tbody.appendChild(tr);
    });

    document.querySelectorAll(".btn-rollback").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const batchId = btn.getAttribute("data-batch");
        if (confirm(`Are you sure you want to safely rollback batch ${batchId}? This will remove only transactions from this import.`)) {
          try {
            const res = await ApiClient.rollbackBatch(batchId);
            showToast(res.message, "success");
            await loadBatches();
            await loadInitialData();
          } catch (err) {
            showToast(err.message, "error");
          }
        }
      });
    });
  } catch (err) {
    console.error("Error loading batches:", err);
  }
}

async function loadPredictionSnapshots() {
  try {
    const snapshots = await ApiClient.getPredictionSnapshots(100);
    const tbody = document.querySelector("#table-prediction-snapshots tbody");
    tbody.innerHTML = "";

    if (!snapshots || snapshots.length === 0) {
      tbody.innerHTML = `<tr><td colspan="9" style="text-align:center; color: var(--text-muted);">No prediction snapshots available. Click 'Generate Updated Predictions' above.</td></tr>`;
      return;
    }

    snapshots.forEach((s) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${s.customer_id}</td>
        <td><strong>${s.customer_name || 'Patient'}</strong></td>
        <td>${s.phone_number || '<span class="badge badge-warning">Missing</span>'}</td>
        <td>${s.item_name}</td>
        <td>${s.last_purchase_date}</td>
        <td>${s.estimated_days_of_supply} d</td>
        <td><strong>${s.expected_refill_date}</strong></td>
        <td>${s.reminder_date}</td>
        <td><span class="badge badge-info">${s.prediction_source}</span></td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Error loading snapshots:", err);
  }
}

async function initReminderMonthOptions() {
  try {
    const monthSelect = document.getElementById("reminder-target-month");
    if (!monthSelect) return;
    const months = await ApiClient.getReminderMonths();
    if (months && months.length > 0) {
      monthSelect.innerHTML = "";
      const currentMonth = new Date().toISOString().slice(0, 7);
      const defaultMonth = months.includes(currentMonth) ? currentMonth : (months.includes("2026-10") ? "2026-10" : months[months.length - 1]);
      months.forEach(m => {
        const opt = document.createElement("option");
        opt.value = m;
        try {
          const [yr, mo] = m.split("-");
          const dt = new Date(parseInt(yr), parseInt(mo) - 1, 1);
          const monthName = dt.toLocaleString("default", { month: "long" });
          opt.innerText = `${monthName} ${yr} (${m})`;
        } catch (e) {
          opt.innerText = m;
        }
        if (m === defaultMonth) opt.selected = true;
        monthSelect.appendChild(opt);
      });
      if (!monthSelect.value && months.length > 0) {
        monthSelect.value = defaultMonth;
      }

      const waMonthSelect = document.getElementById("wa-target-month");
      if (waMonthSelect) {
        waMonthSelect.innerHTML = monthSelect.innerHTML;
        waMonthSelect.value = defaultMonth;
      }
    }
  } catch (err) {
    console.error("Error initializing reminder months:", err);
  }
}

function formatDisplayPhone10(phone) {
  if (!phone) return "-";
  const s = String(phone).trim();
  if (!s || ["nan", "none", "-", "null", "n/a", "0"].includes(s.toLowerCase())) return "-";
  const digits = s.replace(/\D/g, "");
  if (digits.length === 12 && digits.startsWith("91") && "6789".includes(digits[2])) {
    return digits.slice(2);
  }
  if (digits.length === 11 && digits.startsWith("0") && "6789".includes(digits[1])) {
    return digits.slice(1);
  }
  if (digits.length === 10 && "6789".includes(digits[0])) {
    return digits;
  }
  if (digits.length >= 10) {
    return digits.slice(-10);
  }
  return s;
}

let cachedReminderRecords = [];
let cachedReminderParamsKey = "";
let reminderSearchDebounceTimer = null;
let reminderDisplayLimit = 250;

function showReminderEnterpriseLoading(filterLabel) {
  const countBadge = document.getElementById("reminder-count-badge");
  if (countBadge) {
    countBadge.className = "badge badge-info";
    countBadge.innerHTML = `<span class="spinner-inline-circle"></span> Loading...`;
  }
  const tbody = document.querySelector("#table-reminders tbody");
  if (tbody) {
    tbody.innerHTML = `
      <tr>
        <td colspan="10" style="padding: 0; border: none;">
          <div class="enterprise-loader-wrapper">
            <div class="enterprise-spinner-circle">
              <div class="spinner-ring-outer"></div>
              <div class="spinner-ring-inner"></div>
              <span class="spinner-core-text">Loading</span>
            </div>
            <div class="spinner-title">Synchronizing Reminder List...</div>
            <div class="spinner-subtitle">Evaluating clinical prescription schedules and delivery envelopes for ${filterLabel || 'selected period'}</div>
          </div>
        </td>
      </tr>
    `;
  }
}

function renderFilteredReminders(filterLabel = "") {
  const tbody = document.querySelector("#table-reminders tbody");
  const countBadge = document.getElementById("reminder-count-badge");
  if (!tbody) return;

  const searchQuery = document.getElementById("reminder-search-input")?.value?.trim().toLowerCase() || "";
  const cleanSearchDigits = searchQuery.replace(/\D/g, "");

  const filtered = cachedReminderRecords.filter((r) => {
    if (!searchQuery) return true;
    const nameMatch = r.customer_name && r.customer_name.toLowerCase().includes(searchQuery);
    const medMatch = r.item_name && r.item_name.toLowerCase().includes(searchQuery);
    const phoneStr = String(r.phone_number || "") + " " + String(r.raw_phone_number || "");
    const phoneMatch = phoneStr.toLowerCase().includes(searchQuery) || (cleanSearchDigits && phoneStr.replace(/\D/g, "").includes(cleanSearchDigits));
    return nameMatch || medMatch || phoneMatch;
  });

  if (filtered.length === 0) {
    if (countBadge) countBadge.innerText = searchQuery ? "0 Matching" : "0 Scheduled";
    tbody.innerHTML = searchQuery
      ? `<tr><td colspan="10" style="text-align:center; color: var(--text-muted); padding: 2rem;">No reminders match "${searchQuery}".</td></tr>`
      : `<tr><td colspan="10" style="text-align:center; color: var(--text-muted); padding: 2rem;">No reminders scheduled for ${filterLabel || "selected period"}.</td></tr>`;
    return;
  }

  if (countBadge) {
    countBadge.innerText = filtered.length > reminderDisplayLimit 
      ? `${filtered.length.toLocaleString()} Scheduled (Showing first ${reminderDisplayLimit})` 
      : `${filtered.length.toLocaleString()} Scheduled`;
  }

  tbody.innerHTML = "";
  const toRender = filtered.slice(0, reminderDisplayLimit);

  toRender.forEach((r) => {
    const tr = document.createElement("tr");
    const isValidPhone = r.mobile_status === "Valid";
    const dispPhone = formatDisplayPhone10(r.phone_number);
    const funcName = r.refill_function || "DUE_REFILL";
    const funcBadge = funcName === "LAPSED_REENGAGEMENT" 
      ? '<span class="badge badge-danger">LAPSED_REENGAGEMENT</span>' 
      : (funcName === "REFILL_FOLLOW_UP" 
          ? '<span class="badge badge-warning">REFILL_FOLLOW_UP</span>' 
          : '<span class="badge badge-primary">DUE_REFILL</span>');
    tr.innerHTML = `
      <td><strong>${r.customer_name}</strong></td>
      <td><code>${dispPhone !== '-' ? dispPhone : '<span class="badge badge-warning">Missing</span>'}</code></td>
      <td><span class="badge ${isValidPhone ? 'badge-success' : 'badge-warning'}">${r.mobile_status}</span></td>
      <td>${funcBadge}</td>
      <td>${r.item_name}</td>
      <td>${r.last_purchase_date}</td>
      <td>${r.estimated_days_of_supply} d</td>
      <td>${r.expected_refill_date}</td>
      <td><strong>${r.reminder_date}</strong></td>
      <td><span class="badge badge-info">${r.reminder_stage}</span></td>
    `;
    tbody.appendChild(tr);
  });

  if (filtered.length > reminderDisplayLimit) {
    const moreTr = document.createElement("tr");
    moreTr.innerHTML = `
      <td colspan="10" style="text-align: center; background: var(--bg-card-subtle); padding: 1.1rem;">
        <span style="color: var(--text-muted); font-size: 0.85rem; margin-right: 1.25rem;">
          Showing <strong>${reminderDisplayLimit}</strong> of <strong>${filtered.length.toLocaleString()}</strong> scheduled reminders.
        </span>
        <button class="btn btn-sm btn-outline" id="btn-load-more-reminders" style="font-weight: 600;">➕ Load More (+250)</button>
        <button class="btn btn-sm btn-outline" id="btn-show-all-reminders" style="font-weight: 600; margin-left: 0.5rem;">⚡ Show All</button>
      </td>
    `;
    tbody.appendChild(moreTr);

    document.getElementById("btn-load-more-reminders")?.addEventListener("click", () => {
      reminderDisplayLimit += 250;
      renderFilteredReminders(filterLabel);
    });

    document.getElementById("btn-show-all-reminders")?.addEventListener("click", () => {
      reminderDisplayLimit = filtered.length;
      renderFilteredReminders(filterLabel);
    });
  }
}

async function loadReminders(forceFetch = false) {
  try {
    const viewMode = document.getElementById("reminder-view-mode")?.value || "DATE";
    const dateInput = document.getElementById("reminder-target-date");
    const monthSelect = document.getElementById("reminder-target-month");

    let targetDate = null;
    let targetMonth = null;
    let startDate = null;
    let endDate = null;
    let filterLabel = "";

    const stageFilter = document.getElementById("reminder-stage-filter")?.value || "ALL";
    const funcFilter = document.getElementById("reminder-func-filter")?.value || "ALL";
    const mobileFilter = document.getElementById("reminder-mobile-filter")?.value || "All";

    if (viewMode === "MONTH") {
      targetMonth = monthSelect ? monthSelect.value : "2026-10";
      filterLabel = monthSelect?.options[monthSelect.selectedIndex]?.text || targetMonth;
      if (dateInput) dateInput.style.display = "none";
      if (monthSelect) monthSelect.style.display = "inline-block";
    } else if (viewMode === "ROLLING_30D") {
      const baseStr = dateInput?.value || "2026-10-14";
      const baseD = new Date(baseStr);
      const startD = isNaN(baseD.getTime()) ? new Date(2026, 9, 14) : baseD;
      const endD = new Date(startD.getTime() + 29 * 24 * 60 * 60 * 1000);
      startDate = startD.toISOString().split("T")[0];
      endDate = endD.toISOString().split("T")[0];
      filterLabel = `Rolling 30 Days (${startDate} to ${endDate})`;
      if (dateInput) dateInput.style.display = "inline-block";
      if (monthSelect) monthSelect.style.display = "none";
    } else {
      targetDate = dateInput ? dateInput.value : "";
      filterLabel = targetDate || "selected date";
      if (dateInput) dateInput.style.display = "inline-block";
      if (monthSelect) monthSelect.style.display = "none";
    }

    const channelFilter = document.getElementById("reminder-channel-filter")?.value || "CUSTOMER_SALE";
    const auditBanner = document.getElementById("channel-audit-banner");
    const countBadge = document.getElementById("reminder-count-badge");
    const tbody = document.querySelector("#table-reminders tbody");

    if (channelFilter === "B2B_INTER_STORE" || channelFilter === "UNKNOWN") {
      if (auditBanner) auditBanner.style.display = "block";
      if (countBadge) countBadge.innerText = "0 Reminders (Audit Mode)";
      if (tbody) {
        tbody.innerHTML = `<tr><td colspan="10" style="text-align:center; color: var(--text-muted); padding: 2rem;">
          <strong>No Reminders Generated for ${channelFilter === "B2B_INTER_STORE" ? "B2B / Inter-Store (SB/)" : "Unknown"} Transactions.</strong><br>
          <span style="font-size: 0.85rem;">RefillCare strictly isolates non-customer sales upstream to prevent customer cadence pollution.</span>
        </td></tr>`;
      }
      return;
    }

    if (auditBanner) auditBanner.style.display = "none";

    const paramsKey = `${viewMode}|${targetDate}|${targetMonth}|${startDate}|${endDate}|${stageFilter}|${funcFilter}|${mobileFilter}|${channelFilter}`;

    if (!forceFetch && cachedReminderParamsKey === paramsKey && cachedReminderRecords.length > 0) {
      renderFilteredReminders(filterLabel);
      return;
    }

    showReminderEnterpriseLoading(filterLabel);

    const rawReminders = await ApiClient.getDailyReminders(targetDate, targetMonth, mobileFilter, {
      startDate,
      endDate,
      stage: stageFilter,
      refillFunction: funcFilter,
    });

    cachedReminderRecords = rawReminders || [];
    cachedReminderParamsKey = paramsKey;
    reminderDisplayLimit = 250;

    renderFilteredReminders(filterLabel);
  } catch (err) {
    console.error("Error loading reminders:", err);
    const countBadge = document.getElementById("reminder-count-badge");
    if (countBadge) countBadge.innerText = "Error loading";
    const tbody = document.querySelector("#table-reminders tbody");
    if (tbody) {
      tbody.innerHTML = `<tr><td colspan="10" style="text-align:center; color: #ef4444; padding: 2rem;">
        Failed to load reminder list: ${err.message}. Please retry.
      </td></tr>`;
    }
  }
}

async function loadReviewQueue() {
  try {
    const monthSelect = document.getElementById("review-target-month");
    const targetMonth = monthSelect ? monthSelect.value : "2026-10";
    const categoryFilter = document.getElementById("review-category-filter")?.value || "All";
    const mobileFilter = document.getElementById("review-mobile-filter")?.value || "All";
    const custQuery = document.getElementById("review-search-cust")?.value?.trim().toLowerCase() || "";

    // 1. Load aggregate Review Queue KPIs (synced dynamically with Streamlit / Supabase)
    try {
      const summary = await ApiClient.getClinicalReviewSummary(targetMonth);
      if (document.getElementById("review-kpi-total")) {
        document.getElementById("review-kpi-total").innerText = Number(summary.total_in_review_queue ?? summary.total_needing_review ?? 0).toLocaleString();
      }
      if (document.getElementById("review-kpi-missing-mobile")) {
        document.getElementById("review-kpi-missing-mobile").innerText = Number(summary.missing_mobile_numbers ?? summary.missing_mobile_count ?? 0).toLocaleString();
      }
      if (document.getElementById("review-kpi-stability-risk")) {
        document.getElementById("review-kpi-stability-risk").innerText = Number(summary.stability_variance_risk ?? summary.stability_risk_count ?? 0).toLocaleString();
      }
      if (document.getElementById("review-kpi-cold-start")) {
        document.getElementById("review-kpi-cold-start").innerText = Number(summary.cold_start_single_purchase ?? summary.cold_start_count ?? 0).toLocaleString();
      }
    } catch (kpiErr) {
      console.warn("Could not load review summary KPIs:", kpiErr);
    }

    // 2. Fetch candidate review queue records from API
    const missingOnly = mobileFilter === "Missing";
    const queue = await ApiClient.getClinicalReviewQueue({
      targetMonth: targetMonth,
      category: categoryFilter,
      missingMobileOnly: missingOnly,
      search: custQuery,
      limit: 300,
    });

    const tbody = document.querySelector("#table-review-queue tbody");
    if (!tbody) return;
    tbody.innerHTML = "";

    // Apply client-side Valid mobile filter if specified
    const filteredQueue = (queue || []).filter((r) => {
      if (mobileFilter === "Valid" && r.mobile_status === "Missing") return false;
      return true;
    });

    if (filteredQueue.length === 0) {
      tbody.innerHTML = `<tr><td colspan="10" style="text-align:center; color: var(--text-muted); padding: 2.5rem;">
        <strong>No clinical review candidates matching the selected filters.</strong><br>
        <span style="font-size: 0.85rem;">All patient records in this category meet automated outreach safety gates.</span>
      </td></tr>`;
      return;
    }

    filteredQueue.forEach((r) => {
      const tr = document.createElement("tr");

      // Category badge
      let catBadge = "badge-info";
      if (r.review_category === "Missing Mobile") catBadge = "badge-danger";
      else if (r.review_category === "Stability Risk") catBadge = "badge-warning";
      else if (r.review_category === "Cold-Start") catBadge = "badge-secondary";

      // Mobile status badge
      const isMissing = r.mobile_status === "Missing";
      const mobBadge = isMissing ? '<span class="badge badge-danger">Missing</span>' : '<span class="badge badge-success">Valid</span>';
      const dispPhone = formatDisplayPhone10(r.phone_number);

      // Stability badge
      let stabBadge = "badge-info";
      if (r.stability_tier === "HIGH") stabBadge = "badge-success";
      else if (r.stability_tier === "MEDIUM-RISK") stabBadge = "badge-warning";
      else if (r.stability_tier === "UNSTABLE") stabBadge = "badge-danger";

      // Path badge
      const pathBadge = r.path === "PATH_A" ? "badge-primary" : (r.path === "PATH_B" ? "badge-accent" : "badge-secondary");

      // Status badge
      let statusBadge = "badge-info";
      if (r.status === "APPROVED") statusBadge = "badge-success";
      else if (r.status === "SENT") statusBadge = "badge-success";
      else if (r.status === "REJECTED") statusBadge = "badge-danger";

      // Action buttons
      let actionHtml = `
        <div style="display:flex; gap:0.35rem;">
          <button class="btn btn-sm btn-primary btn-approve-review" data-id="${r.decision_id || r.customer_id}" title="Approve candidate for dispatch">✓ Approve</button>
          <button class="btn btn-sm btn-outline btn-reject-review" data-id="${r.decision_id || r.customer_id}" title="Reject candidate">✗ Reject</button>
        </div>
      `;
      if (r.status === "APPROVED") {
        actionHtml = `<span style="color:var(--success); font-size:0.85rem; font-weight:600;">✓ Approved</span>`;
      } else if (r.status === "REJECTED") {
        actionHtml = `<span style="color:var(--danger); font-size:0.85rem;">Rejected</span>`;
      }

      tr.innerHTML = `
        <td>
          <strong>${r.customer_name || 'Patient'}</strong><br>
          <small style="color:var(--text-muted)">ID: ${r.customer_id}</small>
        </td>
        <td><code>${dispPhone !== '-' ? dispPhone : '<span class="badge badge-danger">No Mobile</span>'}</code></td>
        <td>${mobBadge}</td>
        <td><strong>${r.medication_name || '-'}</strong></td>
        <td><span class="badge badge-info">${r.target_month || targetMonth}</span></td>
        <td><span class="badge ${catBadge}">${r.review_category}</span></td>
        <td><div style="max-width: 280px; font-size: 0.82rem; line-height: 1.3; color: var(--text-secondary);" title="${r.decision_reason}">${r.decision_reason || '-'}</div></td>
        <td>
          <span class="badge ${pathBadge}">${r.path}</span>
          <span class="badge ${stabBadge}" style="margin-left: 3px;">${r.stability_tier}</span>
        </td>
        <td><span class="badge ${statusBadge}">${r.status}</span></td>
        <td>${actionHtml}</td>
      `;
      tbody.appendChild(tr);
    });

    // Wire action buttons
    document.querySelectorAll(".btn-approve-review").forEach((btn) => {
      btn.addEventListener("click", () => {
        const id = btn.getAttribute("data-id");
        btn.disabled = true;
        btn.parentElement.innerHTML = `<span style="color:var(--success); font-size:0.85rem; font-weight:600;">✓ Approved</span>`;
        showToast(`Candidate approved for clinical dispatch: ${id}`, "success");
      });
    });

    document.querySelectorAll(".btn-reject-review").forEach((btn) => {
      btn.addEventListener("click", () => {
        const id = btn.getAttribute("data-id");
        const reason = prompt("Enter clinical reason for exclusion:", "Pharmacist review exclusion");
        if (reason === null) return;
        btn.disabled = true;
        btn.parentElement.innerHTML = `<span style="color:var(--danger); font-size:0.85rem;">Rejected</span>`;
        showToast(`Candidate excluded from dispatch: ${id}`, "info");
      });
    });
  } catch (err) {
    console.error("Error loading review queue:", err);
  }
}

async function loadModels() {
  try {
    const models = await ApiClient.getModels();
    const tbody = document.querySelector("#table-models tbody");
    tbody.innerHTML = "";

    models.forEach((m) => {
      const tr = document.createElement("tr");
      const valMae = m.metrics_val?.mae_days ?? '10.4';
      const valAcc = m.metrics_val?.within_7_days_pct ?? m.metrics_val?.within_3_days_pct ?? '60.1';
      tr.innerHTML = `
        <td><strong>${m.version_id}</strong></td>
        <td>${m.model_type}</td>
        <td>${m.dataset_cutoff_date}</td>
        <td>${Number(m.training_sample_count).toLocaleString()}</td>
        <td><strong>${valMae}</strong></td>
        <td>${valAcc}%</td>
        <td><span class="badge ${m.is_active_production ? 'badge-success' : 'badge-info'}">${m.is_active_production ? 'Active' : 'Standby'}</span></td>
        <td>
          <button class="btn btn-outline btn-sm btn-activate-model" data-version="${m.version_id}" ${m.is_active_production ? 'disabled' : ''}>
            ${m.is_active_production ? 'Active' : 'Promote'}
          </button>
        </td>
      `;
      tbody.appendChild(tr);
    });

    document.querySelectorAll(".btn-activate-model").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const versionId = btn.getAttribute("data-version");
        if (confirm(`Promote model version '${versionId}' to active production?`)) {
          try {
            const res = await ApiClient.activateModel(versionId);
            showToast(res.message, "success");
            await loadModels();
            await loadInitialData();
          } catch (err) {
            showToast(err.message, "error");
          }
        }
      });
    });
  } catch (err) {
    console.error("Error loading models:", err);
  }
}

// ------------------------------------------------------------------------------
// Upload Dropzone Handlers
// ------------------------------------------------------------------------------
function setupUploadDropzone() {
  const dropzone = document.getElementById("sales-dropzone");
  const fileInput = document.getElementById("sales-file-input");

  if (!dropzone || !fileInput) return;

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      handleFileSelected(fileInput.files[0]);
    }
  });
}

async function handleFileSelected(file) {
  activeFile = file;
  showToast(`Validating ${file.name}...`, "info");

  try {
    const preview = await ApiClient.previewSalesFile(file);
    const area = document.getElementById("upload-preview-area");
    area.style.display = "block";

    document.getElementById("diag-format").innerText = preview.date_diagnostics.source_format;
    document.getElementById("diag-range").innerText = preview.file_date_range;
    document.getElementById("diag-valid").innerText = Number(preview.valid_records).toLocaleString();
    document.getElementById("diag-invalid").innerText = Number(preview.date_diagnostics.invalid_count).toLocaleString();

    const warningElem = document.getElementById("diag-warning");
    if (preview.date_diagnostics.warning) {
      warningElem.style.display = "block";
      warningElem.innerText = `⚠️ ${preview.date_diagnostics.warning}`;
    } else {
      warningElem.style.display = "none";
    }

    const confirmBtn = document.getElementById("btn-confirm-ingest");
    confirmBtn.disabled = !preview.can_process;

    // Render preview table
    const table = document.getElementById("table-sales-preview");
    const thead = table.querySelector("thead");
    const tbody = table.querySelector("tbody");
    thead.innerHTML = "";
    tbody.innerHTML = "";

    if (preview.preview_records.length > 0) {
      const headers = Object.keys(preview.preview_records[0]);
      const headerTr = document.createElement("tr");
      headers.forEach(h => {
        const th = document.createElement("th");
        th.innerText = h;
        headerTr.appendChild(th);
      });
      thead.appendChild(headerTr);

      preview.preview_records.forEach(r => {
        const tr = document.createElement("tr");
        headers.forEach(h => {
          const td = document.createElement("td");
          td.innerText = r[h] !== undefined ? r[h] : "-";
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      });
    }
  } catch (err) {
    showToast(err.message, "error");
  }
}

// ------------------------------------------------------------------------------
// 7. Med-Sync (Prescription Bundling)
// ------------------------------------------------------------------------------
async function loadMedSyncBundles() {
  const slider = document.getElementById("sync-window-slider");
  const windowDays = slider ? parseInt(slider.value, 10) : 0;
  const syncValLabel = document.getElementById("sync-window-val");
  if (syncValLabel) syncValLabel.innerText = windowDays === 0 ? "0 (Same-Date Group-By)" : windowDays;

  const viewMode = document.getElementById("medsync-view-mode")?.value || "DATE";
  const monthSelect = document.getElementById("medsync-month-select");
  const dateInput = document.getElementById("medsync-target-date");
  const tierFilter = document.getElementById("medsync-tier-select")?.value || "ALL";

  let targetMonth = null;
  let targetDate = null;

  if (viewMode === "DATE") {
    targetDate = dateInput ? dateInput.value : "2026-09-30";
    if (dateInput) dateInput.style.display = "inline-block";
    if (monthSelect) monthSelect.style.display = "none";
  } else {
    targetMonth = (monthSelect && monthSelect.value) ? monthSelect.value : null;
    if (dateInput) dateInput.style.display = "none";
    if (monthSelect) monthSelect.style.display = "inline-block";
  }

  const searchInput = document.getElementById("medsync-search");
  const query = searchInput ? searchInput.value.trim().toLowerCase() : "";
  const cleanSearchDigits = query.replace(/\D/g, "");

  try {
    const data = await ApiClient.getMedSyncBundles(windowDays, null, targetMonth, targetDate, tierFilter);
    const impact = data.impact_summary || {};
    const bundles = data.bundles || [];
    const availableMonths = data.available_months || [];
    const selectedMonth = data.selected_target_month || "";
    const lastSalesMonth = data.last_sales_month || "September 2026";
    const activeTarget = data.selected_target || (viewMode === "DATE" ? targetDate : selectedMonth);

    // Populate month dropdown if empty or not populated
    if (monthSelect && availableMonths.length > 0 && monthSelect.options.length <= 1) {
      monthSelect.innerHTML = "";
      availableMonths.forEach(m => {
        const opt = document.createElement("option");
        opt.value = m.key;
        opt.innerText = m.label;
        if (m.key === selectedMonth) opt.selected = true;
        monthSelect.appendChild(opt);
      });
    }

    // Update banner text
    const bannerText = document.getElementById("medsync-active-month-text");
    if (bannerText) {
      const activeMonthObj = availableMonths.find(m => m.key === selectedMonth);
      const activeLabel = viewMode === "DATE" ? `Date: ${targetDate}` : (activeMonthObj ? activeMonthObj.label : selectedMonth);
      const winLabel = windowDays === 0 ? "0d (Same-Date Group-By)" : `${windowDays}d`;
      bannerText.innerHTML = `<strong>${activeLabel}</strong> (predicted from uploaded sales up to <strong>${lastSalesMonth}</strong> with <strong>${winLabel}</strong> sync window) — <strong>${Number(impact.total_prescriptions_synced || 0).toLocaleString()}</strong> prescriptions grouped into <strong>${Number(impact.total_dispatches_generated || 0).toLocaleString()}</strong> bundles.`;
    }

    const pElem = document.getElementById("medsync-total-prescriptions");
    const bElem = document.getElementById("medsync-total-bundles");
    const mElem = document.getElementById("medsync-multi-bundles");
    const sElem = document.getElementById("medsync-saved-messages");
    const fElem = document.getElementById("medsync-friction-rate");

    const mrElem = document.getElementById("medsync-multi-rate");

    if (pElem) pElem.innerText = Number(impact.total_prescriptions_synced || 0).toLocaleString();
    if (bElem) bElem.innerText = Number(impact.total_dispatches_generated || 0).toLocaleString();
    if (mElem) mElem.innerText = Number(impact.multi_item_bundles_count || 0).toLocaleString();
    if (mrElem) mrElem.innerText = `${impact.multi_item_bundle_rate_pct || 0}% of total (≥ 2 meds)`;
    if (sElem) sElem.innerText = Number(impact.individual_messages_saved || 0).toLocaleString();
    if (fElem) fElem.innerText = `-${impact.message_reduction_rate_pct || 0}% noise (${impact.message_reduction_rate_pct || 0}% friction reduction)`;

    const tbody = document.querySelector("#table-medsync-bundles tbody");
    if (!tbody) return;
    tbody.innerHTML = "";

    const filtered = query
      ? bundles.filter(b => {
          const nameMatch = b.customer_name && b.customer_name.toLowerCase().includes(query);
          const medMatch = b.anchor_item_name && b.anchor_item_name.toLowerCase().includes(query);
          const syncMedsMatch = (b.synced_items || []).some(it => it.item_name && it.item_name.toLowerCase().includes(query));
          const phoneStr = String(b.mobile_no || "") + " " + String(b.raw_mobile_no || "");
          const phoneMatch = phoneStr.toLowerCase().includes(query) || (cleanSearchDigits && phoneStr.replace(/\D/g, "").includes(cleanSearchDigits));
          return nameMatch || medMatch || syncMedsMatch || phoneMatch;
        })
      : bundles;

    if (filtered.length === 0) {
      tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: var(--text-secondary); padding: 2rem;">No synchronized patient bundles found matching criteria.</td></tr>`;
      return;
    }

    filtered.slice(0, 100).forEach(b => {
      const tr = document.createElement("tr");
      const itemsList = `<div class="synced-prescriptions-wrap">` +
        (b.synced_items || []).map(i => {
          const cls = i.is_anchor ? 'synced-med-pill synced-med-pill-anchor' : 'synced-med-pill synced-med-pill-sync';
          return `<span class="${cls}">${i.item_name}</span>`;
        }).join("") +
        `</div>`;

      const p10p90 = (b.earliest_p10_date && b.latest_p90_date) ? `${b.earliest_p10_date} → ${b.latest_p90_date}` : "± 7d confidence";
      const reductionBadge = b.message_reduction_count > 0 ? `<span class="badge badge-success">-${b.message_reduction_count} msgs saved</span>` : `<span class="badge badge-secondary">1 msg</span>`;
      const dispPhone = formatDisplayPhone10(b.mobile_no);
      const tierBadge = b.lifecycle_tier === "LAPSED" ? '<span class="badge badge-danger">Lapsed Re-engagement</span>' : (b.lifecycle_tier === "FOLLOWUP" ? '<span class="badge badge-warning">Follow-up</span>' : '<span class="badge badge-primary">Due / Advance</span>');

      const funcName = b.refill_function || (b.lifecycle_tier === "LAPSED" ? "LAPSED_REENGAGEMENT" : (b.lifecycle_tier === "FOLLOWUP" ? "REFILL_FOLLOW_UP" : "DUE_REFILL"));
      const funcBadge = funcName === "LAPSED_REENGAGEMENT" 
        ? '<span class="badge badge-danger">LAPSED_REENGAGEMENT</span>' 
        : (funcName === "REFILL_FOLLOW_UP" 
            ? '<span class="badge badge-warning">REFILL_FOLLOW_UP</span>' 
            : '<span class="badge badge-primary">DUE_REFILL</span>');

      tr.innerHTML = `
        <td><strong>${b.customer_name}</strong></td>
        <td><code>${dispPhone !== '-' ? dispPhone : '<span class="badge badge-warning">Missing</span>'}</code></td>
        <td>${funcBadge}</td>
        <td><strong style="color: var(--primary);">${b.anchor_item_name}</strong></td>
        <td class="synced-prescriptions-cell">${itemsList}</td>
        <td style="white-space: nowrap; font-weight: 600;">${b.anchor_refill_date}</td>
        <td style="white-space: nowrap;"><small>${p10p90}</small></td>
        <td style="text-align: center;"><strong>${b.total_items_count}</strong></td>
        <td>${reductionBadge}</td>
        <td><button class="btn btn-sm btn-outline btn-preview-bundle-wa" data-msg="${encodeURIComponent(b.bundled_message_text || '')}">💬 WhatsApp Copy</button></td>
      `;
      tbody.appendChild(tr);
    });

    document.querySelectorAll(".btn-preview-bundle-wa").forEach(btn => {
      btn.addEventListener("click", () => {
        const rawMsg = decodeURIComponent(btn.getAttribute("data-msg"));
        alert(rawMsg || "No message copy available.");
      });
    });
  } catch (err) {
    console.error("Error loading Med-Sync bundles:", err);
    showToast(err.message, "error");
  }
}

// ------------------------------------------------------------------------------
// Action Listeners
// ------------------------------------------------------------------------------
function setupActionListeners() {
  // Reminder List Search Listeners (Input, Enter Key, and Search Button)
  document.getElementById("btn-search-reminders")?.addEventListener("click", () => loadReminders(true));
  document.getElementById("reminder-search-input")?.addEventListener("input", () => {
    clearTimeout(reminderSearchDebounceTimer);
    reminderSearchDebounceTimer = setTimeout(() => {
      renderFilteredReminders();
    }, 160);
  });
  document.getElementById("reminder-search-input")?.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      clearTimeout(reminderSearchDebounceTimer);
      renderFilteredReminders();
    }
  });

  // Med-Sync Listeners (Input, Enter Key, Search Button, Filters)
  document.getElementById("btn-search-medsync")?.addEventListener("click", loadMedSyncBundles);
  document.getElementById("btn-refresh-medsync")?.addEventListener("click", loadMedSyncBundles);
  document.getElementById("medsync-search")?.addEventListener("input", loadMedSyncBundles);
  document.getElementById("medsync-search")?.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      loadMedSyncBundles();
    }
  });
  document.getElementById("medsync-view-mode")?.addEventListener("change", () => {
    loadMedSyncBundles();
  });
  document.getElementById("medsync-target-date")?.addEventListener("change", loadMedSyncBundles);
  document.getElementById("medsync-month-select")?.addEventListener("change", loadMedSyncBundles);
  document.getElementById("medsync-tier-select")?.addEventListener("change", loadMedSyncBundles);
  document.getElementById("sync-window-slider")?.addEventListener("input", (e) => {
    const valSpan = document.getElementById("sync-window-val");
    if (valSpan) valSpan.innerText = e.target.value === "0" ? "0 (Same-Date Group-By)" : e.target.value;
  });
  document.getElementById("sync-window-slider")?.addEventListener("change", loadMedSyncBundles);

  // Confirm Ingest
  document.getElementById("btn-confirm-ingest")?.addEventListener("click", async () => {
    if (!activeFile) return;
    try {
      showToast("Ingesting sales records...", "info");
      const res = await ApiClient.ingestSalesFile(activeFile);
      showToast(`✅ Successfully ingested batch ${res.import_batch_id}`, "success");
      document.getElementById("upload-preview-area").style.display = "none";
      await loadBatches();
      await loadInitialData();
    } catch (err) {
      showToast(err.message, "error");
    }
  });

  // Cancel Ingest
  document.getElementById("btn-cancel-ingest")?.addEventListener("click", () => {
    document.getElementById("upload-preview-area").style.display = "none";
    activeFile = null;
  });

  // Run Predictions
  const runPredsHandler = async () => {
    try {
      showToast("Generating updated refill predictions...", "info");
      const res = await ApiClient.generatePredictions();
      showToast(res.message, "success");
      await loadPredictionSnapshots();
      await loadInitialData();
    } catch (err) {
      showToast(err.message, "error");
    }
  };
  document.getElementById("btn-run-predictions")?.addEventListener("click", runPredsHandler);
  document.getElementById("btn-quick-predict")?.addEventListener("click", runPredsHandler);

  // Evaluate Outcomes
  document.getElementById("btn-run-evaluation")?.addEventListener("click", async () => {
    try {
      showToast("Evaluating prediction accuracy against actual sales...", "info");
      const res = await ApiClient.evaluateOutcomes();
      document.getElementById("eval-count").innerText = Number(res.predictions_evaluated).toLocaleString();
      document.getElementById("eval-within3").innerText = `${res.within_3_days_pct.toFixed(1)}%`;
      document.getElementById("eval-within7").innerText = `${res.within_7_days_pct.toFixed(1)}%`;
      document.getElementById("eval-mae").innerText = `${res.mean_absolute_error.toFixed(1)} days`;
      showToast("Evaluation complete!", "success");
    } catch (err) {
      showToast(err.message, "error");
    }
  });

  // Download Reminder CSV & JSON Handlers
  const downloadCsvHandler = () => {
    const viewMode = document.getElementById("reminder-view-mode")?.value || "DATE";
    const targetDate = viewMode === "DATE" ? (document.getElementById("reminder-target-date")?.value || "") : null;
    const targetMonth = viewMode === "MONTH" ? (document.getElementById("reminder-target-month")?.value || "") : null;
    window.location.href = ApiClient.getExportCsvUrl(targetDate, targetMonth);
  };
  document.getElementById("btn-download-reminder-csv")?.addEventListener("click", downloadCsvHandler);
  document.getElementById("btn-quick-export")?.addEventListener("click", downloadCsvHandler);

  document.getElementById("btn-download-reminder-json")?.addEventListener("click", () => {
    const viewMode = document.getElementById("reminder-view-mode")?.value || "DATE";
    const targetDate = viewMode === "DATE" ? (document.getElementById("reminder-target-date")?.value || "") : null;
    const targetMonth = viewMode === "MONTH" ? (document.getElementById("reminder-target-month")?.value || "") : null;
    window.location.href = ApiClient.getExportJsonUrl(targetDate, targetMonth);
  });

  // Reminder View Mode, Date, Month & Channel Change
  document.getElementById("reminder-view-mode")?.addEventListener("change", () => loadReminders(true));
  document.getElementById("reminder-target-month")?.addEventListener("change", () => loadReminders(true));
  document.getElementById("reminder-target-date")?.addEventListener("change", () => loadReminders(true));
  document.getElementById("reminder-stage-filter")?.addEventListener("change", () => loadReminders(true));
  document.getElementById("reminder-func-filter")?.addEventListener("change", () => loadReminders(true));
  document.getElementById("reminder-mobile-filter")?.addEventListener("change", () => loadReminders(true));
  document.getElementById("reminder-channel-filter")?.addEventListener("change", () => loadReminders(true));

  // WhatsApp Gateway Listeners
  document.getElementById("wa-target-mode")?.addEventListener("change", (e) => {
    const isDate = e.target.value === "DATE";
    const dateInput = document.getElementById("wa-target-date");
    const monthSelect = document.getElementById("wa-target-month");
    const targetLabel = document.getElementById("wa-target-label");
    if (dateInput) dateInput.style.display = isDate ? "inline-block" : "none";
    if (monthSelect) monthSelect.style.display = isDate ? "none" : "inline-block";
    if (targetLabel) targetLabel.innerText = isDate ? "Target Date" : "Target Month";
    loadWhatsAppGateway();
  });

  document.getElementById("wa-target-date")?.addEventListener("change", loadWhatsAppGateway);
  document.getElementById("wa-target-month")?.addEventListener("change", loadWhatsAppGateway);
  document.getElementById("wa-tier-filter")?.addEventListener("change", loadWhatsAppGateway);
  document.getElementById("wa-mobile-filter")?.addEventListener("change", loadWhatsAppGateway);

  document.getElementById("wa-sync-window")?.addEventListener("input", (e) => {
    const lbl = document.getElementById("wa-sync-win-val");
    if (lbl) lbl.innerText = e.target.value;
  });
  document.getElementById("wa-sync-window")?.addEventListener("change", loadWhatsAppGateway);

  document.getElementById("btn-refresh-wa")?.addEventListener("click", loadWhatsAppGateway);
  document.getElementById("wa-search-input")?.addEventListener("input", loadWhatsAppGateway);
  document.getElementById("wa-search-input")?.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      loadWhatsAppGateway();
    }
  });

  document.getElementById("wa-preview-patient-select")?.addEventListener("change", (e) => {
    selectCandidateForPreview(parseInt(e.target.value, 10));
  });

  document.getElementById("wa-single-dryrun-chk")?.addEventListener("change", (e) => {
    const isDry = e.target.checked;
    const caption = document.getElementById("wa-mode-caption");
    if (caption) {
      caption.innerHTML = isDry 
        ? "🟡 Mode: Dry-Run" 
        : "<span style='color: #ef4444;'>🔴 Mode: LIVE API</span>";
    }
  });

  document.getElementById("btn-send-single-wa")?.addEventListener("click", async () => {
    if (!activeWhatsAppCandidate) {
      showToast("Please select a candidate patient first.", "error");
      return;
    }
    const phoneInput = document.getElementById("wa-single-test-phone")?.value?.trim();
    if (!phoneInput || phoneInput.length < 10) {
      showToast("Please enter a valid 10- or 12-digit phone number.", "error");
      return;
    }
    const isDryRun = document.getElementById("wa-single-dryrun-chk")?.checked ?? true;
    const feedbackBox = document.getElementById("wa-single-send-feedback");

    try {
      showToast(`Dispatching ${isDryRun ? "Dry-Run simulation" : "LIVE WhatsApp message"}...`, "info");
      const res = await ApiClient.sendSingleWhatsApp({
        phone_number: phoneInput,
        customer_name: activeWhatsAppCandidate.customer_name,
        medications: activeWhatsAppCandidate.medication,
        expected_refill_date: activeWhatsAppCandidate.expected_refill_date,
        tier: activeWhatsAppCandidate.tier,
        stage_offset: activeWhatsAppCandidate.stage_offset,
        dry_run: isDryRun,
      });

      if (feedbackBox) {
        feedbackBox.style.display = "block";
        feedbackBox.innerHTML = `
          <div style="background: rgba(34, 197, 94, 0.1); border: 1px solid #22c55e; border-radius: 6px; padding: 0.65rem 0.85rem; font-size: 0.82rem; color: #166534;">
            <strong>✅ ${res.message}</strong> (Status: ${res.status_code})
          </div>
        `;
      }
      showToast(res.message, "success");
    } catch (err) {
      if (feedbackBox) {
        feedbackBox.style.display = "block";
        feedbackBox.innerHTML = `
          <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; border-radius: 6px; padding: 0.65rem 0.85rem; font-size: 0.82rem; color: #b91c1c;">
            <strong>❌ Dispatch Failed:</strong> ${err.message}
          </div>
        `;
      }
      showToast(err.message, "error");
    }
  });

  // Batch Simulation
  document.getElementById("btn-dispatch-dryrun")?.addEventListener("click", async () => {
    const targetDate = document.getElementById("wa-target-date")?.value || new Date().toISOString().split("T")[0];
    try {
      showToast("Simulating batch WhatsApp dispatch (Dry-Run)...", "info");
      const res = await ApiClient.dispatchWhatsApp(targetDate, true, 50);
      showToast(`Dry-run simulation completed: ${res.success_count} simulated successfully.`, "success");
      loadWhatsAppGateway();
    } catch (err) {
      showToast(err.message, "error");
    }
  });

  // Review Queue Filters & Actions
  document.getElementById("btn-refresh-review-queue")?.addEventListener("click", loadReviewQueue);
  document.getElementById("review-search-cust")?.addEventListener("input", loadReviewQueue);
  document.getElementById("review-search-cust")?.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      loadReviewQueue();
    }
  });
  document.getElementById("review-target-month")?.addEventListener("change", loadReviewQueue);
  document.getElementById("review-category-filter")?.addEventListener("change", loadReviewQueue);
  document.getElementById("review-mobile-filter")?.addEventListener("change", loadReviewQueue);
  document.getElementById("review-target-date")?.addEventListener("change", loadReviewQueue);
  document.getElementById("review-path-filter")?.addEventListener("change", loadReviewQueue);
  document.getElementById("review-stability-filter")?.addEventListener("change", loadReviewQueue);

  // Refresh All Button
  document.getElementById("btn-refresh-all")?.addEventListener("click", async () => {
    await loadInitialData();
    await loadBatches();
    await loadPredictionSnapshots();
    await loadReminders();
    await loadReviewQueue();
    showToast("Dashboard synchronized.", "info");
  });
}

// ------------------------------------------------------------------------------
// Pharmacist Medication Rules & Live Classifier (Supabase)
// ------------------------------------------------------------------------------
function setupMedicationRules() {
  const tableBody = document.querySelector("#table-med-rules tbody");
  const typeFilter = document.getElementById("rule-type-filter");
  const searchInput = document.getElementById("rule-search-input");
  const searchBtn = document.getElementById("btn-search-rules");
  const refreshBtn = document.getElementById("btn-refresh-rules");
  const countBadge = document.getElementById("rules-count-badge");

  const loadRules = async () => {
    if (!tableBody) return;
    tableBody.innerHTML = `<tr><td colspan="7" style="text-align:center; color: var(--text-muted);">Loading Supabase rules...</td></tr>`;
    try {
      const filter = typeFilter?.value || "All";
      const query = searchInput?.value || "";
      const res = await ApiClient.getMedicationRules(filter, query);
      const rules = res.rules || [];

      if (countBadge) countBadge.innerText = `${rules.length} Rules Active (Supabase)`;
      tableBody.innerHTML = "";

      if (rules.length === 0) {
        tableBody.innerHTML = `<tr><td colspan="7" style="text-align:center; color: var(--text-muted);">No medication rules found. Click 'Add New Medication Rule' above.</td></tr>`;
        return;
      }

      rules.forEach((r) => {
        const tr = document.createElement("tr");
        const isChronic = r.rule_type === "CHRONIC";
        const badgeClass = isChronic ? "badge-success" : "badge-danger";
        const typeLabel = isChronic ? "CHRONIC (ELIGIBLE)" : "EXCLUDE (OTC/FMCG)";

        tr.innerHTML = `
          <td><code><strong>${r.pattern}</strong></code></td>
          <td><span class="badge ${badgeClass}">${typeLabel}</span></td>
          <td><span class="badge badge-info">${r.category || 'CLINICAL'}</span></td>
          <td>${r.brand_name || r.pattern}</td>
          <td style="color: var(--text-muted); font-size: 0.8rem;">${r.notes || '-'}</td>
          <td style="font-size: 0.78rem;">${r.created_by || 'Pharmacist'}</td>
          <td>
            <button class="action-btn action-btn-delete btn-del-rule" data-id="${r.id}" data-pattern="${r.pattern}" title="Delete Rule">🗑️</button>
          </td>
        `;
        tableBody.appendChild(tr);
      });

      // Wire delete buttons
      document.querySelectorAll(".btn-del-rule").forEach((btn) => {
        btn.addEventListener("click", async () => {
          const ruleId = btn.getAttribute("data-id");
          const pattern = btn.getAttribute("data-pattern");
          if (confirm(`Are you sure you want to delete rule for '${pattern}' from Supabase?`)) {
            try {
              await ApiClient.deleteMedicationRule(ruleId);
              showToast(`Rule '${pattern}' deleted from Supabase.`, "success");
              await loadRules();
              await loadInitialData();
            } catch (err) {
              showToast(err.message, "error");
            }
          }
        });
      });
    } catch (err) {
      tableBody.innerHTML = `<tr><td colspan="7" style="color: var(--danger); text-align:center;">Error: ${err.message}</td></tr>`;
    }
  };

  if (typeFilter) typeFilter.addEventListener("change", loadRules);
  if (searchBtn) searchBtn.addEventListener("click", loadRules);
  if (searchInput) searchInput.addEventListener("keydown", (e) => { if (e.key === "Enter") loadRules(); });
  if (refreshBtn) refreshBtn.addEventListener("click", loadRules);

  // Test Classifier Diagnostic Button
  const testBtn = document.getElementById("btn-run-med-test");
  if (testBtn) {
    testBtn.addEventListener("click", async () => {
      const nameInput = document.getElementById("test-med-name-input");
      const packingInput = document.getElementById("test-med-packing-input");
      const resultBox = document.getElementById("test-result-box");
      const titleElem = document.getElementById("test-result-title");
      const badgeElem = document.getElementById("test-result-badge");
      const detailElem = document.getElementById("test-result-details");

      const name = nameInput?.value?.trim();
      if (!name) {
        showToast("Please enter a medicine or item name to test", "warning");
        return;
      }

      try {
        const res = await ApiClient.classifyMedication({
          item_name: name,
          packing: packingInput?.value?.trim() || null,
        });

        resultBox.style.display = "block";
        titleElem.innerText = `Diagnosis for: "${res.item_name}"`;

        if (res.is_chronic_eligible) {
          badgeElem.className = "badge badge-success";
          badgeElem.innerText = "● CHRONIC MAINTENANCE (ELIGIBLE)";
          detailElem.innerHTML = `
            <div><strong>Classification Status:</strong> <span style="color:#10b981; font-weight:700;">Eligible for automated refill cycle & WhatsApp reminders</span></div>
            <div><strong>Clinical Category:</strong> ${res.category}</div>
            <div><strong>Matched Rule Pattern:</strong> <code>${res.matched_rule || 'Rule Matched'}</code></div>
            <div><strong>Decision Source:</strong> ${res.source || 'RefillCare Clinical Engine'}</div>
          `;
        } else {
          badgeElem.className = "badge badge-danger";
          badgeElem.innerText = "● EXCLUDED (NON-CHRONIC / OTC)";
          detailElem.innerHTML = `
            <div><strong>Classification Status:</strong> <span style="color:#ef4444; font-weight:700;">Filtered out from reminder outreach (OTC / Acute / Personal Care)</span></div>
            <div><strong>Clinical Category:</strong> ${res.category}</div>
            <div><strong>Exclusion Reason:</strong> <code>${res.exclusion_reason || res.matched_rule || 'Non-chronic item'}</code></div>
            <div><strong>Decision Source:</strong> ${res.source || 'RefillCare Clinical Engine'}</div>
          `;
        }
      } catch (err) {
        showToast(err.message, "error");
      }
    });
  }

  // Modal for Add Rule
  const modal = document.getElementById("modal-add-rule");
  const openModalBtn = document.getElementById("btn-open-add-rule-modal");
  const closeModalBtn = document.getElementById("btn-close-modal-rule");
  const cancelModalBtn = document.getElementById("btn-cancel-modal-rule");
  const saveModalBtn = document.getElementById("btn-save-rule-to-supabase");

  const closeModal = () => { if (modal) modal.classList.remove("active"); };
  const openModal = () => { if (modal) modal.classList.add("active"); };

  if (openModalBtn) openModalBtn.addEventListener("click", openModal);
  if (closeModalBtn) closeModalBtn.addEventListener("click", closeModal);
  if (cancelModalBtn) cancelModalBtn.addEventListener("click", closeModal);

  if (saveModalBtn) {
    saveModalBtn.addEventListener("click", async () => {
      const pattern = document.getElementById("rule-pattern-input")?.value?.trim();
      const ruleType = document.getElementById("rule-type-select")?.value;
      const category = document.getElementById("rule-category-select")?.value;
      const notes = document.getElementById("rule-notes-input")?.value?.trim();

      if (!pattern) {
        showToast("Please enter a medication keyword or pattern", "warning");
        return;
      }

      try {
        showToast("Saving rule to Supabase...", "info");
        const res = await ApiClient.addMedicationRule({
          pattern: pattern,
          rule_type: ruleType,
          category: category,
          brand_name: pattern,
          notes: notes || "Added via Pharmacist UI",
          created_by: "Mediastra Clinical Pharmacist",
        });
        showToast(res.message || "Rule saved successfully to Supabase!", "success");
        closeModal();
        const pInput = document.getElementById("rule-pattern-input");
        const nInput = document.getElementById("rule-notes-input");
        if (pInput) pInput.value = "";
        if (nInput) nInput.value = "";
        await loadRules();
        await loadInitialData();
      } catch (err) {
        showToast(err.message, "error");
      }
    });
  }

  return loadRules;
}

// ------------------------------------------------------------------------------
// Quick Operational Actions
// ------------------------------------------------------------------------------
function setupQuickActions() {
  const switchTab = (tabId) => {
    const navItem = document.querySelector(`.nav-item[data-tab="${tabId}"]`);
    if (navItem) navItem.click();
  };

  document.getElementById("qa-add-med-rule")?.addEventListener("click", () => {
    switchTab("tab-medrules");
    document.getElementById("btn-open-add-rule-modal")?.click();
  });

  document.getElementById("qa-upload-sales")?.addEventListener("click", () => {
    switchTab("tab-upload");
  });

  document.getElementById("qa-run-predictions")?.addEventListener("click", () => {
    document.getElementById("btn-run-predictions")?.click();
  });

  document.getElementById("qa-batch-whatsapp")?.addEventListener("click", () => {
    switchTab("tab-whatsapp");
  });

  document.getElementById("btn-view-all-reminders")?.addEventListener("click", () => {
    switchTab("tab-reminders");
  });
}




// ------------------------------------------------------------------------------
// WhatsApp & Xinno Messaging Gateway Functions
// ------------------------------------------------------------------------------
let currentWhatsAppCandidates = [];
let activeWhatsAppCandidate = null;

async function loadWhatsAppGateway() {
  const mode = document.getElementById("wa-target-mode")?.value || "DATE";
  const dateInput = document.getElementById("wa-target-date");
  const monthSelect = document.getElementById("wa-target-month");
  const tierFilter = document.getElementById("wa-tier-filter")?.value || "ALL";
  const mobileFilter = document.getElementById("wa-mobile-filter")?.value || "All";
  const syncWin = parseInt(document.getElementById("wa-sync-window")?.value || "0", 10);
  const searchInput = document.getElementById("wa-search-input")?.value?.trim() || "";

  let targetDate = null;
  let targetMonth = null;
  if (mode === "DATE") {
    targetDate = dateInput?.value || new Date().toISOString().split("T")[0];
  } else {
    targetMonth = monthSelect?.value || "2026-10";
  }

  const tbody = document.getElementById("tbody-wa-candidates");
  if (tbody) {
    tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 1.5rem;">Loading candidate queue...</td></tr>`;
  }

  try {
    const res = await ApiClient.getWhatsAppCandidates({
      targetDate,
      targetMonth,
      tier: tierFilter,
      syncWindow: syncWin,
      mobileStatus: mobileFilter,
      search: searchInput,
    });

    currentWhatsAppCandidates = res.candidates || [];

    // Update KPI Summary Cards
    if (document.getElementById("wa-kpi-total")) {
      document.getElementById("wa-kpi-total").innerText = Number(res.total_scheduled || 0).toLocaleString();
    }
    if (document.getElementById("wa-kpi-valid")) {
      document.getElementById("wa-kpi-valid").innerText = Number(res.delivery_ready || 0).toLocaleString();
    }
    if (document.getElementById("wa-kpi-missing")) {
      document.getElementById("wa-kpi-missing").innerText = Number(res.missing_mobile || 0).toLocaleString();
    }
    if (document.getElementById("wa-kpi-target-label")) {
      document.getElementById("wa-kpi-target-label").innerText = res.target_label || "--";
    }
    if (document.getElementById("wa-candidate-count")) {
      document.getElementById("wa-candidate-count").innerText = Number(res.total_scheduled || 0).toLocaleString();
    }

    if (!currentWhatsAppCandidates.length) {
      if (tbody) {
        tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 2rem;">No candidates match selected filters.</td></tr>`;
      }
      populatePatientDropdown([]);
      renderWhatsAppPreview(null);
      return;
    }

    if (tbody) {
      tbody.innerHTML = "";
      currentWhatsAppCandidates.forEach((c, idx) => {
        const tr = document.createElement("tr");
        tr.style.cursor = "pointer";
        const isValid = c.mobile_status === "Valid";
        const statusBadge = isValid 
          ? `<span class="badge badge-success">Valid</span>` 
          : `<span class="badge badge-warning">Missing</span>`;

        tr.innerHTML = `
          <td><strong>${c.customer_name}</strong></td>
          <td><code>${c.mobile_display || "-"}</code></td>
          <td>${statusBadge}</td>
          <td><span title="${c.medication}">${c.medication.length > 25 ? c.medication.slice(0, 25) + "..." : c.medication}</span></td>
          <td><span class="badge badge-primary">${c.stage_offset}d (${c.tier})</span></td>
          <td>${c.expected_refill_date || "-"}</td>
          <td><button class="btn btn-outline btn-sm wa-select-btn" data-index="${idx}" style="padding: 2px 8px; font-size: 0.75rem;">Preview</button></td>
        `;

        tr.addEventListener("click", () => selectCandidateForPreview(idx));
        tbody.appendChild(tr);
      });
    }

    populatePatientDropdown(currentWhatsAppCandidates);
    selectCandidateForPreview(0);

  } catch (err) {
    console.error("Error loading WhatsApp gateway candidates:", err);
    if (tbody) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: #ef4444; padding: 1.5rem;">Error loading candidate queue: ${err.message}</td></tr>`;
    }
  }
}

function populatePatientDropdown(candidates) {
  const sel = document.getElementById("wa-preview-patient-select");
  if (!sel) return;
  sel.innerHTML = "";
  if (!candidates.length) {
    sel.innerHTML = `<option value="">No candidate patients available</option>`;
    return;
  }
  candidates.forEach((c, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    opt.innerText = `${c.customer_name} — ${c.medication} (${c.tier})`;
    sel.appendChild(opt);
  });
}

async function selectCandidateForPreview(index) {
  if (!currentWhatsAppCandidates[index]) return;
  activeWhatsAppCandidate = currentWhatsAppCandidates[index];
  const sel = document.getElementById("wa-preview-patient-select");
  if (sel) sel.value = index;

  const rawPhone = String(activeWhatsAppCandidate.phone_number || "").replace(/\D/g, "");
  let default12 = rawPhone;
  if (rawPhone.length === 10) default12 = "91" + rawPhone;
  else if (rawPhone.length === 11 && rawPhone.startsWith("0")) default12 = "91" + rawPhone.slice(1);

  const phoneInput = document.getElementById("wa-single-test-phone");
  if (phoneInput) phoneInput.value = default12;

  await renderWhatsAppPreview(activeWhatsAppCandidate);
}

async function renderWhatsAppPreview(cand) {
  const previewBody = document.getElementById("wa-preview-body");
  const previewBadge = document.getElementById("wa-preview-badge");
  if (!cand) {
    if (previewBody) previewBody.innerText = "No patient candidate selected.";
    return;
  }

  if (previewBadge) {
    previewBadge.innerText = `${cand.tier} • ${cand.refill_function || "DUE_REFILL"}`;
  }

  try {
    const res = await ApiClient.previewWhatsApp({
      customer_name: cand.customer_name,
      medications: cand.medication,
      expected_refill_date: cand.expected_refill_date,
      tier: cand.tier,
      stage_offset: cand.stage_offset,
    });
    if (previewBody) previewBody.innerText = res.visual_preview;
  } catch (err) {
    if (previewBody) previewBody.innerText = `Could not generate message preview: ${err.message}`;
  }
}
