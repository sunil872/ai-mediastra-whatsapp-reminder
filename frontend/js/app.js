/**
 * RefillCare Enterprise Frontend Application
 */

import { ApiClient } from "./api.js";


// ------------------------------------------------------------------------------
// Security: escape every data value interpolated into innerHTML (stored-XSS guard).
// Patient / medicine names come from uploaded POS files and must never be treated as HTML.
// ------------------------------------------------------------------------------
function esc(value) {
  if (value === null || value === undefined) return "";
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
window.refillcareEscape = esc;

function monthKey(d) {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}`;
}

function localDateStr(d) {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

// Global State
let activeFile = null;

// Initialize Application
document.addEventListener("DOMContentLoaded", async () => {
  initTheme();
  setupNavigation();
  setupUploadDropzone();
  setupActionListeners();
  setupModalHelper();
  setupOverviewActions();
  setupReviewTab();
  setupWhatsAppSend();
  setupConsentActions();
  setupDeliveryLog();
  setupApiKeyButton();

  // Dynamic default date (local calendar date)
  const todayStr = localDateStr(new Date());
  const dateInput = document.getElementById("reminder-target-date");
  if (dateInput) dateInput.value = todayStr;

  const medsyncDateInput = document.getElementById("medsync-target-date");
  if (medsyncDateInput) medsyncDateInput.value = todayStr;

  const reviewDateInput = document.getElementById("review-target-date");
  if (reviewDateInput) reviewDateInput.value = todayStr;

  const waDateInput = document.getElementById("wa-target-date");
  if (waDateInput) waDateInput.value = todayStr;

  const logDayInput = document.getElementById("wa-log-day");
  if (logDayInput) logDayInput.value = todayStr;

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
      icon.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>`;
      label.innerText = "Light Mode";
    } else {
      icon.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>`;
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
const TAB_LOADERS = {
  "tab-dashboard": () => loadInitialData(),
  "tab-upload": () => { loadBatches(); loadColumnMappings(); loadDataStatus(); },
  "tab-reminders": () => loadReminders(true),  // always fresh on tab switch (uploads / pharmacist adds change it)
  "tab-medsync": () => loadMedSyncBundles(),
  "tab-review": () => loadReviewTab(),
  "tab-whatsapp": () => { loadWhatsAppGateway(); loadDeliveryLog(); },
};

function switchTab(tabId) {
  const navItem = document.querySelector(`.nav-item[data-tab="${tabId}"]`);
  if (navItem) navItem.click();
}

function setupNavigation() {
  const navItems = document.querySelectorAll(".nav-item");
  const tabPanes = document.querySelectorAll(".tab-pane");
  const breadcrumbElem = document.getElementById("breadcrumb-current");

  navItems.forEach((item) => {
    item.addEventListener("click", () => {
      const tabId = item.getAttribute("data-tab");
      navItems.forEach((n) => n.classList.remove("active"));
      tabPanes.forEach((p) => p.classList.remove("active"));

      item.classList.add("active");
      const targetPane = document.getElementById(tabId);
      if (targetPane) targetPane.classList.add("active");

      const label = item.querySelector("strong")?.innerText?.trim() || "Pharmacy Overview";
      if (breadcrumbElem) breadcrumbElem.innerText = label;
      document.title = `${label} — Mediastra Refillcare`;

      // Lazy load tab data
      const loader = TAB_LOADERS[tabId];
      if (loader) loader();
    });
  });
}

// ------------------------------------------------------------------------------
// Pharmacy Overview (same numbers as the Streamlit overview)
// ------------------------------------------------------------------------------
function parseDMY(s) {
  // "DD-MM-YYYY" or "YYYY-MM-DD" -> local Date (or null)
  if (!s) return null;
  const str = String(s).trim().slice(0, 10);
  let m = str.match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (m) return new Date(+m[1], +m[2] - 1, +m[3]);
  m = str.match(/^(\d{2})-(\d{2})-(\d{4})$/);
  if (m) return new Date(+m[3], +m[2] - 1, +m[1]);
  return null;
}

function fmtDMY(s) {
  // any supported date string -> "dd-mm-yyyy"; unknown formats are returned unchanged
  const d = parseDMY(s);
  if (!d) return s ? String(s) : "-";
  return `${String(d.getDate()).padStart(2, "0")}-${String(d.getMonth() + 1).padStart(2, "0")}-${d.getFullYear()}`;
}

function fmtLongDate(isoDay) {
  // "2026-11-09" -> "09 Nov 2026"
  const d = parseDMY(isoDay);
  if (!d) return isoDay || "";
  const mon = d.toLocaleString("en-GB", { month: "short" });
  return `${String(d.getDate()).padStart(2, "0")} ${mon} ${d.getFullYear()}`;
}

function fmtNum(v) {
  return v === null || v === undefined || v === "" ? "–" : Number(v).toLocaleString();
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.innerText = value;
}

async function loadInitialData() {
  // Sidebar model label + latest sales date come from the analytics KPI endpoint.
  let kpis = null;
  try {
    kpis = await ApiClient.getKpis();
    if (document.getElementById("sidebar-active-model")) {
      const v = kpis.active_model_version || "none";
      const evidence = kpis.measured_backtest_within_7_days_pct != null
        ? ` · ${kpis.measured_backtest_within_7_days_pct}% ±7d (measured)`
        : " · not yet evaluated";
      document.getElementById("sidebar-active-model").innerText = `${v}${kpis.model_is_validated ? " (Validated)" : ""}${evidence}`;
    }
    setText("kpi-latest-sales-date", kpis.latest_sales_date || "–");
    setText("kpi-sales-coverage", kpis.sales_coverage_date_range ? `Sales on file: ${kpis.sales_coverage_date_range}` : "");

    const banner = document.getElementById("overview-stale-banner");
    if (banner) {
      const latest = parseDMY(kpis.latest_sales_date);
      const today = parseDMY(localDateStr(new Date()));
      const daysAgo = latest ? Math.round((today - latest) / 86400000) : null;
      if (daysAgo !== null && daysAgo > 2) {
        banner.innerHTML = `Latest sales uploaded are from <b>${esc(kpis.latest_sales_date)}</b> (${esc(daysAgo)} days ago). Upload the newer sales so reminders and review lists are up to date.`;
        banner.style.display = "block";
      } else {
        banner.style.display = "none";
        banner.innerHTML = "";
      }
    }
  } catch (err) {
    console.error("Error loading KPIs:", err);
    showToast("Error connecting to RefillCare backend", "error");
  }

  try {
    const ov = await ApiClient.getOverview();
    const monthName = (ov.month_label || "").split(" ")[0] || "";
    setText("ov-month-caption", ov.month_label || ov.month || "");
    setText("ov-title-messages-month", `Reminder Messages (${monthName})`);
    const activeLabel = ov.active_patients_label || "Active patients (last 12 months)";
    const activeCount = ov.active_patients !== undefined ? ov.active_patients : ov.customers_monitored;
    const allTimeCount = ov.all_time_customers !== undefined ? ov.all_time_customers : ov.customers_monitored;
    const subtext = ov.active_patients_subtext || `All-time: ${fmtNum(allTimeCount)}`;
    setText("kpi-customers-title", activeLabel);
    setText("kpi-customers", fmtNum(activeCount));
    setText("kpi-customers-subtext", subtext);
    setText("kpi-messages-month", fmtNum(ov.reminder_messages_month));
    setText("kpi-due-today", fmtNum(ov.reminders_due_today));
    setText("kpi-review-today", fmtNum(ov.patients_needing_review_today));
  } catch (err) {
    console.error("Error loading overview:", err);
  }
}

function setupOverviewActions() {
  const card = document.getElementById("kpi-card-review-today");
  if (!card) return;
  const go = () => switchTab("tab-review");
  card.addEventListener("click", go);
  card.addEventListener("keypress", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
}

async function loadBatches() {
  try {
    const batches = await ApiClient.getBatches();
    const tbody = document.querySelector("#table-import-batches tbody");
    tbody.innerHTML = "";

    if (!batches || batches.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; color: var(--text-muted);">No import batches recorded yet.</td></tr>`;
      return;
    }

    batches.forEach((b) => {
      const tr = document.createElement("tr");
      const isRolledBack = b.processing_status === "ROLLED_BACK";
      tr.innerHTML = `
        <td><code>${esc(b.import_batch_id)}</code></td>
        <td>${esc(b.source_filename)}</td>
        <td>${new Date(b.upload_timestamp).toLocaleString()}</td>
        <td>${esc(b.detected_date_range || "-")}</td>
        <td>${b.sync_status === "SYNC_FAILED"
          ? `<span class="badge badge-danger" title="${esc(b.sync_error || "")}">failed</span> <button class="btn btn-outline btn-sm btn-retry-sync" data-batch="${esc(b.import_batch_id)}">Retry</button>`
          : `<span class="badge ${b.sync_status === "SYNCED" ? "badge-success" : "badge-info"}">${esc(b.sync_status || "-")}</span>`}</td>
        <td><strong>${Number(b.records_inserted).toLocaleString()}</strong></td>
        <td><span class="badge ${isRolledBack ? 'badge-danger' : 'badge-success'}">${esc(b.processing_status)}</span></td>
        <td>
          <button class="btn btn-outline btn-sm btn-rollback" data-batch="${esc(b.import_batch_id)}" ${isRolledBack ? 'disabled' : ''}>
            ${isRolledBack ? 'Rolled Back' : '↺ Undo Import'}
          </button>
        </td>
      `;
      tbody.appendChild(tr);
    });

    document.querySelectorAll(".btn-retry-sync").forEach((btn) => {
      btn.addEventListener("click", async () => {
        try {
          const res = await ApiClient.retryBatchSync(btn.getAttribute("data-batch"));
          showToast(res.error ? `Retry failed: ${res.error}` : `Re-scored ${res.customer_items_rescored} medicines.`, res.error ? "error" : "success");
          await loadBatches();
          await loadDataStatus();
        } catch (err) {
          showToast(err.message, "error");
        }
      });
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

async function initReminderMonthOptions() {
  try {
    const monthSelect = document.getElementById("reminder-target-month");
    if (!monthSelect) return;
    const months = await ApiClient.getReminderMonths();
    if (months && months.length > 0) {
      monthSelect.innerHTML = "";
      const currentMonth = monthKey(new Date());
      const defaultMonth = months.includes(currentMonth) ? currentMonth : months[months.length - 1];
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
      ? `<tr><td colspan="10" style="text-align:center; color: var(--text-muted); padding: 2rem;">No reminders match "${esc(searchQuery)}".</td></tr>`
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
      <td><strong>${esc(r.customer_name)}</strong></td>
      <td><code>${dispPhone !== '-' ? dispPhone : '<span class="badge badge-warning">Missing</span>'}</code></td>
      <td><span class="badge ${isValidPhone ? 'badge-success' : 'badge-warning'}">${esc(r.mobile_status)}</span></td>
      <td>${funcBadge}</td>
      <td>${esc(r.item_name)}</td>
      <td>${esc(r.last_purchase_date)}</td>
      <td>${esc(r.estimated_days_of_supply)} d</td>
      <td>${esc(r.expected_refill_date)}</td>
      <td><strong>${esc(r.reminder_date)}</strong></td>
      <td><span class="badge badge-info">${esc(r.reminder_stage)}</span></td>
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
        <button class="btn btn-sm btn-outline" id="btn-load-more-reminders" style="font-weight: 600;">Load More (+250)</button>
        <button class="btn btn-sm btn-outline" id="btn-show-all-reminders" style="font-weight: 600; margin-left: 0.5rem;">Show All</button>
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
      targetMonth = monthSelect && monthSelect.value ? monthSelect.value : monthKey(new Date());
      filterLabel = monthSelect?.options[monthSelect.selectedIndex]?.text || targetMonth;
      if (dateInput) dateInput.style.display = "none";
      if (monthSelect) monthSelect.style.display = "inline-block";
    } else if (viewMode === "ROLLING_30D") {
      const baseStr = dateInput?.value || localDateStr(new Date());
      const startD = parseDMY(baseStr) || new Date();
      const endD = new Date(startD.getFullYear(), startD.getMonth(), startD.getDate() + 29);
      startDate = localDateStr(startD);
      endDate = localDateStr(endD);
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
        Failed to load reminder list: ${esc(err.message)}. Please retry.
      </td></tr>`;
    }
  }
}

async function loadStoreSummary() {
  // Same store summary as the overview / Streamlit Upload tab (GET /api/v1/overview)
  try {
    const ov = await ApiClient.getOverview();
    const dmy = (iso) => (iso ? iso.split("-").reverse().join("-") : "–");
    setText("up-date-range", ov.latest_sales_date ? `${dmy(ov.first_sales_date)} to ${dmy(ov.latest_sales_date)}` : "No sales yet");
    setText("up-lines-stored", fmtNum(ov.sales_lines_stored));
    // Same return / invalid-row breakdown as the Streamlit card (GET /api/v1/analytics/transaction-types)
    ApiClient.getTransactionTypes().then((t) => {
      if (t.return_lines == null) return;
      setText("up-lines-breakdown", `${fmtNum(t.return_lines)} return lines · ${fmtNum(t.invalid_rows)} invalid rows not stored`);
      const card = document.getElementById("up-lines-card");
      if (card) card.title = `Return lines are never stored as rows: ${fmtNum(t.returns_netted)} were subtracted from the sale they follow `
        + `(within ${t.return_window_days} days), ${fmtNum(t.returns_unmatched)} matched no sale. ${fmtNum(t.sales_fully_returned)} sales were `
        + `fully returned (${fmtNum(t.returned_corrections)} billing corrections within ${t.return_correction_days} days, `
        + `${fmtNum(t.returned_late)} later returns -> reminders paused, pharmacist review); they stay stored but do not count. `
        + `${fmtNum(t.invalid_rows)} rows had no date / customer / medicine / quantity.`;
    }).catch((err) => console.error("Error loading return breakdown:", err));
    const activeLabel = ov.active_patients_label || "Active patients (last 12 months)";
    const activeCount = ov.active_patients !== undefined ? ov.active_patients : ov.customers_monitored;
    const allTimeCount = ov.all_time_customers !== undefined ? ov.all_time_customers : ov.customers_monitored;
    const subtext = ov.active_patients_subtext || `All-time: ${fmtNum(allTimeCount)}`;
    setText("up-customers-title", activeLabel);
    setText("up-customers", fmtNum(activeCount));
    setText("up-customers-subtext", subtext);
  } catch (err) {
    console.error("Error loading store summary:", err);
  }
}

async function loadDataStatus() {
  loadStoreSummary();
  const box = document.getElementById("data-status-banner");
  if (!box) return;
  try {
    const st = await ApiClient.getDataStatus();
    const latest = st.latest_sales_date || "none yet";
    if (st.ok_to_send) {
      box.innerHTML = `<span class="badge badge-success">Sales data current</span> latest sale ${esc(latest)} (${esc(st.days_behind)} day(s) ago)`;
    } else {
      const why = st.batches_failed_to_sync ? `${esc(st.batches_failed_to_sync)} upload(s) failed to re-score - retry below` :
        `latest sale ${esc(latest)} is ${esc(st.days_behind ?? "-")} day(s) behind (max ${esc(st.max_allowed_lag_days)})`;
      box.innerHTML = `<span class="badge badge-danger">Live WhatsApp sending is blocked</span> ${why}. Upload the missing days first.`;
    }
  } catch (err) {
    box.textContent = "";
  }
}

async function loadColumnMappings() {
  const tbody = document.querySelector("#table-column-mappings tbody");
  if (!tbody) return;
  try {
    const rows = await ApiClient.getColumnMappings();
    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="5" style="color: var(--text-secondary);">No saved mappings yet - the first successful upload saves one.</td></tr>`;
      return;
    }
    tbody.innerHTML = rows.map(p => {
      const mapped = Object.entries(p.mapping || {}).filter(([, v]) => v).map(([f, c]) => `${f} ← ${c}`).join(", ");
      return `<tr><td style="font-size: 0.8rem;">${esc(mapped)}</td><td>${esc(p.source)} (${esc(p.confirmed_by)})</td><td>${esc(p.use_count)}</td><td>${esc(p.last_used_at || "-")}</td><td><button class="btn btn-outline" data-mapping-id="${esc(p.profile_id)}" style="padding: 0.2rem 0.6rem;">Delete</button></td></tr>`;
    }).join("");
    tbody.querySelectorAll("button[data-mapping-id]").forEach(btn => btn.addEventListener("click", async () => {
      if (!confirm("Delete this saved mapping? The next upload with this layout will be auto-detected again.")) return;
      try {
        await ApiClient.deleteColumnMapping(btn.dataset.mappingId);
        showToast("Mapping deleted.", "success");
        await loadColumnMappings();
      } catch (err) {
        showToast(err.message, "error");
      }
    }));
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="5">${esc(err.message)}</td></tr>`;
  }
}

// ------------------------------------------------------------------------------
// Write API key (sent as X-API-Key by api.js)
// ------------------------------------------------------------------------------
function setupApiKeyButton() {
  const btn = document.getElementById("btn-set-api-key");
  const label = document.getElementById("api-key-label");
  const refresh = () => {
    let has = false;
    try { has = !!localStorage.getItem("refillcare_api_key"); } catch (e) { /* storage blocked */ }
    if (label) label.innerText = has ? "Write key set" : "Set write key";
  };
  refresh();
  btn?.addEventListener("click", () => {
    const val = prompt("API key (REFILLCARE_API_KEY). Leave empty to clear:", "");
    if (val === null) return;
    let store = "";
    try { store = localStorage.getItem("refillcare_store_id") || ""; } catch (e) { /* ignore */ }
    const sid = prompt("Store id (X-Store-Id, e.g. ngos-colony). Leave empty for the server's default store:", store);
    try {
      if (val.trim()) localStorage.setItem("refillcare_api_key", val.trim());
      else localStorage.removeItem("refillcare_api_key");
      if (sid !== null) {
        if (sid.trim()) localStorage.setItem("refillcare_store_id", sid.trim().toLowerCase());
        else localStorage.removeItem("refillcare_store_id");
      }
    } catch (e) {
      showToast("Browser storage is blocked; the key cannot be saved.", "error");
    }
    refresh();
    loadInitialData();
  });
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

// Column mapping of the active file: {RefillCare field: store header or null}; null = server decides
let activeColumnMapping = null;
let activeMappingReport = null;

const MAPPING_SOURCE_LABEL = {
  saved: "saved for this store",
  user: "chosen by you",
  auto: "auto-matched",
  saved_hint: "auto-matched (known header)",
  suggestion: "suggested - please confirm",
  none: "not in file",
};

function renderColumnMapping(report, forceEdit = false) {
  activeMappingReport = report;
  const status = document.getElementById("mapping-status");
  const warnBox = document.getElementById("mapping-warnings");
  const wrap = document.getElementById("mapping-table-wrap");
  const actions = document.getElementById("mapping-actions");
  const ignored = document.getElementById("mapping-ignored");
  if (!report || !status) return;

  const needs = report.needs_confirmation;
  const fromSaved = (report.fields || []).some(f => f.source === "saved");
  status.className = `badge ${needs ? "badge-danger" : "badge-success"}`;
  status.textContent = needs
    ? `Needs confirmation: ${(report.missing_required || []).length} required field(s) not found`
    : (fromSaved ? "Saved mapping for this store applied" : "All required columns auto-matched");

  const warnings = report.warnings || [];
  warnBox.style.display = warnings.length ? "block" : "none";
  warnBox.textContent = warnings.join("  ");

  const editing = needs || forceEdit;
  wrap.style.display = editing ? "block" : "none";
  actions.style.display = editing ? "flex" : "none";
  ignored.textContent = (report.ignored_columns || []).length
    ? `Not stored (not needed for reminders): ${report.ignored_columns.join(", ")}`
    : "";

  const tbody = document.querySelector("#table-column-mapping tbody");
  tbody.innerHTML = "";
  (report.fields || []).forEach(f => {
    const tr = document.createElement("tr");
    const tdLabel = document.createElement("td");
    tdLabel.textContent = f.label;
    const tdRole = document.createElement("td");
    const role = document.createElement("span");
    role.className = `badge ${f.role === "required" ? "badge-danger" : (f.role === "expected" ? "badge-warning" : "badge-info")}`;
    role.textContent = f.role;
    tdRole.appendChild(role);
    const tdSel = document.createElement("td");
    const sel = document.createElement("select");
    sel.dataset.field = f.field;
    const none = document.createElement("option");
    none.value = "";
    none.textContent = "- not in file -";
    sel.appendChild(none);
    (report.headers || []).forEach(h => {
      const o = document.createElement("option");
      o.value = h;
      o.textContent = h;
      if (f.column === h) o.selected = true;
      sel.appendChild(o);
    });
    tdSel.appendChild(sel);
    const tdSamples = document.createElement("td");
    tdSamples.textContent = (f.samples || []).join(" | ") || "-";
    const tdMatch = document.createElement("td");
    tdMatch.textContent = f.column ? `${MAPPING_SOURCE_LABEL[f.source] || f.source} (${Math.round((f.confidence || 0) * 100)}%)` : MAPPING_SOURCE_LABEL.none;
    [tdLabel, tdRole, tdSel, tdSamples, tdMatch].forEach(td => tr.appendChild(td));
    tbody.appendChild(tr);
  });
}

function collectColumnMapping() {
  const mapping = {};
  document.querySelectorAll("#table-column-mapping select").forEach(sel => {
    mapping[sel.dataset.field] = sel.value || null;
  });
  return mapping;
}

async function handleFileSelected(file) {
  activeFile = file;
  activeColumnMapping = null;
  await runSalesPreview();
}

async function runSalesPreview() {
  const file = activeFile;
  if (!file) return;
  showToast(`Validating ${esc(file.name)}...`, "info");

  try {
    const preview = await ApiClient.previewSalesFile(file, activeColumnMapping);
    const area = document.getElementById("upload-preview-area");
    area.style.display = "block";
    renderColumnMapping(preview.column_mapping);

    document.getElementById("diag-format").innerText = preview.date_diagnostics.source_format;
    document.getElementById("diag-range").innerText = preview.file_date_range;
    document.getElementById("diag-valid").innerText = Number(preview.valid_records).toLocaleString();
    document.getElementById("diag-invalid").innerText = Number(preview.date_diagnostics.invalid_count).toLocaleString();

    const warningElem = document.getElementById("diag-warning");
    if (preview.date_diagnostics.warning) {
      warningElem.style.display = "block";
      warningElem.innerText = esc(preview.date_diagnostics.warning);
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
    targetDate = dateInput && dateInput.value ? dateInput.value : localDateStr(new Date());
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
    const lastSalesMonth = data.last_sales_month || "-";
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
      bannerText.innerHTML = `<strong>${activeLabel}</strong> (predicted from uploaded sales up to <strong>${esc(lastSalesMonth)}</strong> with <strong>${winLabel}</strong> sync window) — <strong>${Number(impact.total_prescriptions_synced || 0).toLocaleString()}</strong> prescriptions grouped into <strong>${Number(impact.total_dispatches_generated || 0).toLocaleString()}</strong> bundles.`;
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
          return `<span class="${cls}">${esc(i.item_name)}</span>`;
        }).join("") +
        `</div>`;

      const p10p90 = (b.earliest_p10_date && b.latest_p90_date) ? `${esc(b.earliest_p10_date)} → ${esc(b.latest_p90_date)}` : "± 7d confidence";
      const reductionBadge = b.message_reduction_count > 0 ? `<span class="badge badge-success">-${esc(b.message_reduction_count)} msgs saved</span>` : `<span class="badge badge-secondary">1 msg</span>`;
      const dispPhone = formatDisplayPhone10(b.mobile_no);
      const tierBadge = b.lifecycle_tier === "LAPSED" ? '<span class="badge badge-danger">Lapsed Re-engagement</span>' : (b.lifecycle_tier === "FOLLOWUP" ? '<span class="badge badge-warning">Follow-up</span>' : '<span class="badge badge-primary">Due / Advance</span>');

      const funcName = b.refill_function || (b.lifecycle_tier === "LAPSED" ? "LAPSED_REENGAGEMENT" : (b.lifecycle_tier === "FOLLOWUP" ? "REFILL_FOLLOW_UP" : "DUE_REFILL"));
      const funcBadge = funcName === "LAPSED_REENGAGEMENT" 
        ? '<span class="badge badge-danger">LAPSED_REENGAGEMENT</span>' 
        : (funcName === "REFILL_FOLLOW_UP" 
            ? '<span class="badge badge-warning">REFILL_FOLLOW_UP</span>' 
            : '<span class="badge badge-primary">DUE_REFILL</span>');

      tr.innerHTML = `
        <td><strong>${esc(b.customer_name)}</strong></td>
        <td><code>${dispPhone !== '-' ? dispPhone : '<span class="badge badge-warning">Missing</span>'}</code></td>
        <td>${funcBadge}</td>
        <td><strong style="color: var(--primary);">${esc(b.anchor_item_name)}</strong></td>
        <td class="synced-prescriptions-cell">${itemsList}</td>
        <td style="white-space: nowrap; font-weight: 600;">${esc(b.anchor_refill_date)}</td>
        <td style="white-space: nowrap;"><small>${p10p90}</small></td>
        <td style="text-align: center;"><strong>${esc(b.total_items_count)}</strong></td>
        <td>${reductionBadge}</td>
        <td><button class="btn btn-sm btn-outline btn-preview-bundle-wa" data-msg="${encodeURIComponent(b.bundled_message_text || '')}">WhatsApp Copy</button></td>
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
      let res;
      try {
        res = await ApiClient.ingestSalesFile(activeFile, activeColumnMapping);
      } catch (err) {
        if (err.status !== 409 || !confirm(`${err.message}\n\nUpload it again anyway? Only lines not stored yet are added.`)) throw err;
        res = await ApiClient.ingestSalesFile(activeFile, activeColumnMapping, true);
      }
      showToast(`Batch ${esc(res.import_batch_id)}: ${Number(res.records_inserted).toLocaleString()} new lines, ${Number(res.records_skipped).toLocaleString()} already stored.`, "success");
      const w = res.data_warnings || {};
      if (w.gap_before_file) showToast(`No sales uploaded for ${w.gap_before_file} - upload those days so reminders stay correct.`, "warning");
      await loadDataStatus();
      const fresh = res.ml_data_freshness;
      if (res.ml_retrain_queued) {
        showToast("New sales data: model retrain cycle queued in background.", "info");
      } else if (fresh && fresh.retrain_recommended) {
        showToast("New sales data moved evaluation windows: model retrain recommended.", "info");
      }
      document.getElementById("upload-preview-area").style.display = "none";
      await loadBatches();
      await loadColumnMappings();
      await loadInitialData();
    } catch (err) {
      showToast(err.message, "error");
    }
  });

  // Cancel Ingest
  document.getElementById("btn-cancel-ingest")?.addEventListener("click", () => {
    document.getElementById("upload-preview-area").style.display = "none";
    activeFile = null;
    activeColumnMapping = null;
  });

  // Column mapping: show the editor / save the confirmed mapping for this store and re-validate
  document.getElementById("btn-edit-mapping")?.addEventListener("click", () => {
    if (activeMappingReport) renderColumnMapping(activeMappingReport, true);
  });
  document.getElementById("btn-save-mapping")?.addEventListener("click", async () => {
    if (!activeFile || !activeMappingReport) return;
    const mapping = collectColumnMapping();
    try {
      await ApiClient.saveColumnMapping(activeMappingReport.headers, mapping);
      activeColumnMapping = mapping;
      showToast("Column mapping saved for this store.", "success");
      await runSalesPreview();
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

}

// ------------------------------------------------------------------------------
// Generic modal helper (replaces confirm() / prompt() for the pharmacist flows)
// ------------------------------------------------------------------------------
let modalOnClose = null;

function setupModalHelper() {
  const overlay = document.getElementById("app-modal");
  if (!overlay) return;
  overlay.addEventListener("click", (e) => { if (e.target === overlay) closeModal(); });
  document.getElementById("app-modal-close")?.addEventListener("click", () => closeModal());
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && overlay.classList.contains("active")) closeModal();
  });
}

/** Open the shared modal. bodyHtml / footerHtml must already be escaped. Returns the modal parts. */
function openModal(title, bodyHtml, footerHtml = "", { onClose = null, wide = false } = {}) {
  const overlay = document.getElementById("app-modal");
  const card = document.getElementById("app-modal-card");
  document.getElementById("app-modal-title").innerText = title;
  const body = document.getElementById("app-modal-body");
  const footer = document.getElementById("app-modal-footer");
  body.innerHTML = bodyHtml;
  footer.innerHTML = footerHtml;
  footer.style.display = footerHtml ? "" : "none";
  card.classList.toggle("modal-card-wide", !!wide);
  setModalLocked(false);
  modalOnClose = onClose;
  overlay.classList.add("active");
  return { overlay, body, footer };
}

function setModalLocked(locked) {
  const overlay = document.getElementById("app-modal");
  if (overlay) overlay.dataset.locked = locked ? "1" : "";
  const closeBtn = document.getElementById("app-modal-close");
  if (closeBtn) closeBtn.disabled = !!locked;
}

function closeModal(force = false) {
  const overlay = document.getElementById("app-modal");
  if (!overlay || !overlay.classList.contains("active")) return;
  if (overlay.dataset.locked && !force) return;  // e.g. while WhatsApp messages are being sent
  overlay.classList.remove("active");
  setModalLocked(false);
  const cb = modalOnClose;
  modalOnClose = null;
  if (cb && !force) cb();
}

/** Yes/No confirmation. onConfirm may be async; errors are shown inside the modal. */
function confirmModal({ title, messageHtml, confirmLabel = "Yes", confirmClass = "btn-primary", cancelLabel = "No, cancel", onConfirm }) {
  const { body, footer } = openModal(
    title,
    `${messageHtml}<div class="alert alert-danger modal-error" style="display:none;"></div>`,
    `<button class="btn btn-outline" data-modal-cancel>${esc(cancelLabel)}</button>
     <button class="btn ${esc(confirmClass)}" data-modal-confirm>${esc(confirmLabel)}</button>`,
  );
  const no = footer.querySelector("[data-modal-cancel]");
  const yes = footer.querySelector("[data-modal-confirm]");
  no.addEventListener("click", () => closeModal());
  yes.addEventListener("click", async () => {
    yes.disabled = true;
    no.disabled = true;
    setModalLocked(true);
    try {
      await onConfirm();
      setModalLocked(false);
      closeModal();
    } catch (err) {
      setModalLocked(false);
      yes.disabled = false;
      no.disabled = false;
      const box = body.querySelector(".modal-error");
      if (box) { box.style.display = "block"; box.innerText = err.message; }
    }
  });
}

function downloadCsv(filename, headers, rows) {
  const cell = (v) => {
    const s = v === null || v === undefined ? "" : String(v);
    return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const csv = [headers.map(cell).join(","), ...rows.map((r) => r.map(cell).join(","))].join("\r\n");
  const blob = new Blob(["﻿" + csv], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 0);
}

function mobileStatusBadge(status) {
  const s = status || "Missing";
  const cls = s === "Valid" ? "badge-success" : (s === "Invalid" ? "badge-danger" : "badge-warning");
  return `<span class="badge ${cls}">${esc(s)}</span>`;
}

// ------------------------------------------------------------------------------
// Patients Needing Review (same lists and actions as the Streamlit Review tab)
// ------------------------------------------------------------------------------
const REVIEW_CATEGORY_LABELS = {
  candidates: "Patients not getting reminders",
  missing_mobile: "Missing mobile number (in reminders)",
  unstable: "Unstable refill pattern (in reminders)",
  b2b: "B2B / Inter-Store excluded (audit)",
};

let reviewRows = [];
let reviewSelected = new Set();
let reviewLoadedCategory = "candidates";
let reviewLoadedDay = "";
let reviewSearchTimer = null;

function setupReviewTab() {
  const dateInput = document.getElementById("review-target-date");
  if (dateInput) dateInput.value = localDateStr(new Date());
  dateInput?.addEventListener("change", () => loadReviewList());
  document.getElementById("review-category")?.addEventListener("change", () => loadReviewList());
  document.getElementById("review-mobile-status")?.addEventListener("change", () => renderReview());
  const debounced = () => { clearTimeout(reviewSearchTimer); reviewSearchTimer = setTimeout(renderReview, 200); };
  document.getElementById("review-search")?.addEventListener("input", debounced);
  document.getElementById("review-medicine-search")?.addEventListener("input", debounced);
  document.getElementById("btn-review-export")?.addEventListener("click", exportReviewCsv);
}

function loadReviewTab() {
  loadReviewList();
  loadPharmacistAdded();
}

function reviewDay() {
  return document.getElementById("review-target-date")?.value || localDateStr(new Date());
}

async function loadReviewList() {
  const category = document.getElementById("review-category")?.value || "candidates";
  const day = reviewDay();
  const wrap = document.getElementById("review-table-wrap");
  if (wrap) wrap.innerHTML = `<div class="empty-note">Loading…</div>`;
  reviewSelected = new Set();
  reviewLoadedCategory = category;
  reviewLoadedDay = day;
  renderReviewAddPanel();
  try {
    const rows = category === "candidates"
      ? await ApiClient.getReviewCandidates(day)
      : await ApiClient.getReviewOther(category, day);
    reviewRows = Array.isArray(rows) ? rows : [];
    renderReview();
  } catch (err) {
    reviewRows = [];
    if (wrap) wrap.innerHTML = `<div class="alert alert-danger">${esc(err.message)}</div>`;
  }
}

function filteredReviewRows() {
  const q = (document.getElementById("review-search")?.value || "").trim().toLowerCase();
  const qDigits = q.replace(/\D/g, "");
  const med = (document.getElementById("review-medicine-search")?.value || "").trim().toLowerCase();
  const mob = document.getElementById("review-mobile-status")?.value || "All";
  return reviewRows.filter((r) => {
    if (q) {
      const nameHit = String(r.customer_name || "").toLowerCase().includes(q);
      const mobileHit = qDigits.length > 0 && String(r.mobile_number || "").replace(/\D/g, "").includes(qDigits);
      if (!nameHit && !mobileHit) return false;
    }
    if (med && !String(r.medicine || "").toLowerCase().includes(med)) return false;
    if (mob !== "All" && (r.mobile_status || "Missing") !== mob) return false;
    return true;
  });
}

function renderReview() {
  const wrap = document.getElementById("review-table-wrap");
  const heading = document.getElementById("review-heading");
  const caption = document.getElementById("review-caption");
  const kpis = document.getElementById("review-kpis");
  const exportBtn = document.getElementById("btn-review-export");
  if (!wrap) return;
  const rows = filteredReviewRows();
  const dayLabel = fmtLongDate(reviewLoadedDay);
  const isCandidates = reviewLoadedCategory === "candidates";

  if (kpis) {
    kpis.style.display = isCandidates ? "" : "none";
    if (isCandidates) {
      setText("review-kpi-not-in", fmtNum(rows.length));
      setText("review-kpi-chronic", fmtNum(rows.filter((r) => r.medicine_class === "Chronic").length));
      setText("review-kpi-unclassified", fmtNum(rows.filter((r) => r.medicine_class !== "Chronic").length));
      setText("review-kpi-no-mobile", fmtNum(rows.filter((r) => r.mobile_status !== "Valid").length));
    }
  }
  if (heading) heading.innerText = `${REVIEW_CATEGORY_LABELS[reviewLoadedCategory] || "Review"} (${rows.length.toLocaleString()}) · refill due on ${dayLabel}`;
  if (caption) caption.style.display = isCandidates && rows.length ? "" : "none";
  if (exportBtn) exportBtn.style.display = !isCandidates && rows.length ? "" : "none";

  // Drop selections that the filters now hide
  const visibleKeys = new Set(rows.map((r) => r.customer_item_key));
  reviewSelected = new Set([...reviewSelected].filter((k) => visibleKeys.has(k)));

  if (!rows.length) {
    wrap.innerHTML = isCandidates
      ? `<div class="empty-note">No patients left out of reminders (refill due on ${esc(dayLabel)}).</div>`
      : `<div class="empty-note">No records for this category (refill due on ${esc(dayLabel)}).</div>`;
    renderReviewAddPanel();
    return;
  }

  if (isCandidates) {
    const allTicked = rows.every((r) => reviewSelected.has(r.customer_item_key));
    wrap.innerHTML = `
      <div class="table-responsive" style="max-height: 520px;">
        <table class="data-table" id="table-review-candidates">
          <thead><tr>
            <th style="width:36px;"><input type="checkbox" id="review-select-all" ${allTicked ? "checked" : ""} title="Select all"></th>
            <th>Customer Name</th><th>Mobile Number</th><th>Mobile Status</th><th>Medication</th><th>Medicine Type</th>
            <th>Purchases</th><th>Last Purchase</th><th>Usual Gap (days)</th><th>Estimated Due</th><th>Why Not In Reminders</th>
          </tr></thead>
          <tbody>${rows.map((r) => {
            const ticked = reviewSelected.has(r.customer_item_key);
            const typeBadge = r.medicine_class === "Chronic" ? "badge-primary" : "badge-secondary";
            return `<tr class="${ticked ? "row-selected" : ""}">
              <td><input type="checkbox" class="review-row-chk" data-key="${esc(r.customer_item_key)}" ${ticked ? "checked" : ""}></td>
              <td><strong>${esc(r.customer_name || "-")}</strong></td>
              <td><code>${esc(formatDisplayPhone10(r.mobile_number))}</code></td>
              <td>${mobileStatusBadge(r.mobile_status)}</td>
              <td>${esc(r.medicine || "-")}</td>
              <td><span class="badge ${typeBadge}">${esc(r.medicine_class || "-")}</span></td>
              <td>${esc(r.purchase_count ?? "-")}</td>
              <td>${esc(fmtDMY(r.last_purchase_date))}</td>
              <td>${esc(r.usual_gap_days ?? "-")}</td>
              <td>${esc(fmtDMY(r.estimated_due_date))}</td>
              <td style="font-size:0.82rem; color: var(--text-secondary);">${esc(r.reason_not_in_reminders || "-")}</td>
            </tr>`;
          }).join("")}</tbody>
        </table>
      </div>`;
    wrap.querySelector("#review-select-all")?.addEventListener("change", (e) => {
      rows.forEach((r) => (e.target.checked ? reviewSelected.add(r.customer_item_key) : reviewSelected.delete(r.customer_item_key)));
      renderReview();
    });
    wrap.querySelectorAll(".review-row-chk").forEach((chk) => {
      chk.addEventListener("change", () => {
        const key = chk.getAttribute("data-key");
        if (chk.checked) reviewSelected.add(key); else reviewSelected.delete(key);
        chk.closest("tr")?.classList.toggle("row-selected", chk.checked);
        const all = wrap.querySelector("#review-select-all");
        if (all) all.checked = rows.every((r) => reviewSelected.has(r.customer_item_key));
        renderReviewAddPanel();
      });
    });
  } else {
    wrap.innerHTML = `
      <div class="table-responsive" style="max-height: 520px;">
        <table class="data-table">
          <thead><tr>
            <th>Customer Name</th><th>Mobile Number</th><th>Mobile Status</th><th>Medication</th>
            <th>Last Purchase Date</th><th>Expected Refill Date</th><th>Review Reason</th>
          </tr></thead>
          <tbody>${rows.map((r) => `<tr>
            <td><strong>${esc(r.customer_name || "-")}</strong></td>
            <td><code>${esc(formatDisplayPhone10(r.mobile_number))}</code></td>
            <td>${mobileStatusBadge(r.mobile_status)}</td>
            <td>${esc(r.medicine || "-")}</td>
            <td>${esc(fmtDMY(r.last_purchase_date))}</td>
            <td>${esc(fmtDMY(r.expected_refill_date))}</td>
            <td style="font-size:0.82rem; color: var(--text-secondary);">${esc(r.review_reason || "-")}</td>
          </tr>`).join("")}</tbody>
        </table>
      </div>`;
  }
  renderReviewAddPanel();
}

function exportReviewCsv() {
  const rows = filteredReviewRows();
  if (!rows.length) return;
  downloadCsv(
    `review_${reviewLoadedCategory}_${reviewLoadedDay}.csv`,
    ["Customer Name", "Mobile Number", "Mobile Status", "Medication", "Last Purchase Date", "Expected Refill Date", "Review Reason"],
    rows.map((r) => [r.customer_name, formatDisplayPhone10(r.mobile_number), r.mobile_status, r.medicine,
      fmtDMY(r.last_purchase_date), fmtDMY(r.expected_refill_date), r.review_reason]),
  );
}

function renderReviewAddPanel() {
  const panel = document.getElementById("review-add-panel");
  if (!panel) return;
  const selected = reviewLoadedCategory === "candidates"
    ? reviewRows.filter((r) => reviewSelected.has(r.customer_item_key))
    : [];
  if (!selected.length) {
    panel.style.display = "none";
    panel.innerHTML = "";
    return;
  }
  panel.style.display = "block";
  const today = localDateStr(new Date());

  if (selected.length === 1) {
    const r = selected[0];
    const gap = Math.min(180, Math.max(7, parseInt(r.usual_gap_days, 10) || 30));
    const est = parseDMY(r.estimated_due_date);
    const estIso = est ? localDateStr(est) : today;
    const due = estIso > today ? estIso : today;
    panel.innerHTML = `
      <h4 class="panel-title">➕ Add to reminders: ${esc(r.customer_name || "-")} — ${esc(r.medicine || "-")}</h4>
      <div class="panel-facts">
        <span>Mobile: <b>${esc(formatDisplayPhone10(r.mobile_number))}</b></span>
        <span>Status: ${mobileStatusBadge(r.mobile_status)}</span>
        <span>Type: <b>${esc(r.medicine_class || "-")}</b></span>
        <span>Purchases: <b>${esc(r.purchase_count ?? "-")}</b></span>
        <span>Last bought: <b>${esc(fmtDMY(r.last_purchase_date))}</b></span>
      </div>
      ${r.medicine_class !== "Chronic" ? `<div class="alert alert-info">This medicine is not in the medication rules yet. Add it only if it is a regular (chronic) medicine for this patient.</div>` : ""}
      ${r.mobile_status !== "Valid" ? `<div class="alert alert-warning">No valid mobile number: the reminder is scheduled but cannot be sent until the number is updated in billing.</div>` : ""}
      <div class="form-row">
        <label class="form-field"><span>Refill every (days)</span>
          <input type="number" id="review-add-interval" min="7" max="180" step="1" value="${esc(gap)}"></label>
        <label class="form-field"><span>Next refill due date</span>
          <input type="date" id="review-add-date" min="${esc(today)}" value="${esc(due)}"></label>
        <label class="form-field form-field-grow"><span>Note (optional)</span>
          <input type="text" id="review-add-note" maxlength="300"></label>
      </div>
      <button class="btn btn-primary" id="btn-review-add">➕ Add to reminders</button>`;
    panel.querySelector("#btn-review-add").addEventListener("click", async (e) => {
      const btn = e.currentTarget;
      const interval = parseInt(panel.querySelector("#review-add-interval").value, 10);
      const nextDate = panel.querySelector("#review-add-date").value;
      const note = panel.querySelector("#review-add-note").value.trim();
      if (!Number.isFinite(interval) || interval < 7 || interval > 180) { showToast("Refill every (days) must be between 7 and 180.", "error"); return; }
      if (!nextDate || nextDate < today) { showToast("Next refill due date must be today or later.", "error"); return; }
      btn.disabled = true;
      try {
        const res = await ApiClient.pharmacistAdd({
          customer_item_key: r.customer_item_key,
          interval_days: interval,
          next_refill_date: nextDate,
          note: note || null,
        });
        if (res.added) {
          showToast(`✅ ${r.customer_name} added to reminders for ${r.medicine} (next refill ${fmtDMY(res.next_refill_date || nextDate)}, every ${res.interval_days || interval} days).`, "success");
        } else {
          showToast("This patient is already getting reminders.", "info");
        }
        await afterReviewChange();
      } catch (err) {
        showToast(err.message, "error");
        btn.disabled = false;
      }
    });
    return;
  }

  const n = selected.length;
  const notClassified = selected.filter((r) => r.medicine_class !== "Chronic").length;
  const noMobile = selected.filter((r) => r.mobile_status !== "Valid").length;
  panel.innerHTML = `
    <h4 class="panel-title">➕ Add ${n} selected patients to reminders</h4>
    <p class="panel-caption">Each patient keeps their own usual refill gap and estimated due date (dates in the past move to today).</p>
    ${notClassified ? `<div class="alert alert-info">${notClassified} of these medicines are not in the medication rules yet. Add them only if they are regular (chronic) medicines for these patients.</div>` : ""}
    ${noMobile ? `<div class="alert alert-warning">${noMobile} of these patients have no valid mobile number: their reminders are scheduled but cannot be sent until the number is updated in billing.</div>` : ""}
    <div class="table-responsive" style="max-height: 260px; margin-bottom: 0.75rem;">
      <table class="data-table">
        <thead><tr><th>Customer Name</th><th>Mobile Number</th><th>Medication</th><th>Usual Gap (days)</th><th>Estimated Due</th></tr></thead>
        <tbody>${selected.map((r) => `<tr>
          <td>${esc(r.customer_name || "-")}</td><td><code>${esc(formatDisplayPhone10(r.mobile_number))}</code></td>
          <td>${esc(r.medicine || "-")}</td><td>${esc(r.usual_gap_days ?? "-")}</td><td>${esc(fmtDMY(r.estimated_due_date))}</td>
        </tr>`).join("")}</tbody>
      </table>
    </div>
    <div class="form-row">
      <label class="form-field form-field-grow"><span>Note for all (optional)</span>
        <input type="text" id="review-bulk-note" maxlength="300"></label>
    </div>
    <button class="btn btn-primary" id="btn-review-bulk-add">➕ Add ${n} patients to reminders</button>`;
  panel.querySelector("#btn-review-bulk-add").addEventListener("click", async (e) => {
    const btn = e.currentTarget;
    const note = panel.querySelector("#review-bulk-note").value.trim();
    btn.disabled = true;
    try {
      const res = await ApiClient.pharmacistAddBulk(selected.map((r) => r.customer_item_key), note || null);
      let msg = `✅ ${Number(res.added || 0).toLocaleString()} patients added to reminders.`;
      if (res.already_in_reminders) msg += ` ${Number(res.already_in_reminders).toLocaleString()} were already getting reminders.`;
      showToast(msg, "success");
      (res.failed || []).forEach((f) => {
        const row = reviewRows.find((r) => r.customer_item_key === f.customer_item_key);
        const who = row ? `${row.customer_name} — ${row.medicine}` : f.customer_item_key;
        showToast(`Could not add ${who}: ${f.reason}`, "error");
      });
      await afterReviewChange();
    } catch (err) {
      showToast(err.message, "error");
      btn.disabled = false;
    }
  });
}

async function afterReviewChange() {
  await Promise.all([loadReviewList(), loadPharmacistAdded()]);
  loadInitialData();
}

async function loadPharmacistAdded() {
  const wrap = document.getElementById("review-added-wrap");
  if (!wrap) return;
  try {
    const rows = await ApiClient.getPharmacistAdded();
    const list = Array.isArray(rows) ? rows : [];
    setText("review-added-count", list.length.toLocaleString());
    if (!list.length) {
      wrap.innerHTML = `<div class="empty-note">No patients have been added by a pharmacist yet.</div>`;
      return;
    }
    wrap.innerHTML = `
      <div class="table-responsive">
        <table class="data-table">
          <thead><tr><th>Patient</th><th>Medicine</th><th>Refill every (days)</th><th>Next refill</th><th>Added on</th><th>Note</th><th></th></tr></thead>
          <tbody>${list.map((r, i) => `<tr>
            <td><strong>${esc(r.customer_name || r.customer_id || "-")}</strong></td>
            <td>${esc(r.medicine || "-")}</td>
            <td>${esc(r.interval_days ?? "-")}</td>
            <td>${esc(fmtDMY(r.next_refill_date))}</td>
            <td>${esc(r.added_at ? String(r.added_at).slice(0, 10) : "-")}</td>
            <td style="font-size:0.82rem;">${esc(r.note || "")}</td>
            <td><button class="btn btn-outline btn-sm btn-pharmacist-remove" data-idx="${i}">Remove</button></td>
          </tr>`).join("")}</tbody>
        </table>
      </div>`;
    wrap.querySelectorAll(".btn-pharmacist-remove").forEach((btn) => {
      btn.addEventListener("click", () => {
        const r = list[parseInt(btn.getAttribute("data-idx"), 10)];
        if (!r) return;
        const who = r.customer_name || r.customer_id || "this patient";
        confirmModal({
          title: "Remove from reminders",
          messageHtml: `<p>Remove <b>${esc(who)}</b> — <b>${esc(r.medicine)}</b> from reminders? Their pending reminder messages are cancelled.</p>`,
          confirmLabel: "Yes, remove",
          confirmClass: "btn-danger",
          onConfirm: async () => {
            const res = await ApiClient.pharmacistRemove(r.customer_item_key);
            showToast(res.removed
              ? `${who} — ${r.medicine} removed from reminders (${Number(res.stages_cancelled || 0)} pending messages cancelled).`
              : "This patient was not in the pharmacist-added list.", res.removed ? "success" : "info");
            afterReviewChange();
          },
        });
      });
    });
  } catch (err) {
    wrap.innerHTML = `<div class="alert alert-danger">${esc(err.message)}</div>`;
  }
}

// ------------------------------------------------------------------------------
// WhatsApp Messages: candidate list + preview
// ------------------------------------------------------------------------------
let currentWhatsAppCandidates = [];
let waSearchTimer = null;

function waTarget() {
  const mode = document.getElementById("wa-target-mode")?.value || "DATE";
  if (mode === "DATE") {
    return { mode, targetDate: document.getElementById("wa-target-date")?.value || localDateStr(new Date()), targetMonth: null };
  }
  return { mode, targetDate: null, targetMonth: document.getElementById("wa-target-month")?.value || monthKey(new Date()) };
}

async function loadWhatsAppGateway() {
  const { targetDate, targetMonth } = waTarget();
  const tierFilter = document.getElementById("wa-tier-filter")?.value || "ALL";
  const mobileFilter = document.getElementById("wa-mobile-filter")?.value || "All";
  const syncWin = parseInt(document.getElementById("wa-sync-window")?.value || "0", 10);
  const searchInput = document.getElementById("wa-search-input")?.value?.trim() || "";

  loadWhatsAppSendSummary();

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
    setText("wa-kpi-total", fmtNum(res.total_scheduled || 0));
    setText("wa-kpi-valid", fmtNum(res.delivery_ready || 0));
    setText("wa-kpi-missing", fmtNum(res.missing_mobile || 0));
    setText("wa-kpi-target-label", res.target_label || "--");
    setText("wa-candidate-count", fmtNum(res.total_scheduled || 0));

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
        const med = String(c.medication || "");
        const stageDisp = c.stage_display || (c.tier === "FOLLOWUP" ? "FOLLOW" : (c.tier === "DUE" ? "DUE" : "LAPSED"));
        tr.innerHTML = `
          <td><strong>${esc(c.customer_name)}</strong></td>
          <td><code>${esc(c.mobile_display || "-")}</code></td>
          <td>${mobileStatusBadge(c.mobile_status)}</td>
          <td><span title="${esc(med)}">${esc(med.length > 25 ? med.slice(0, 25) + "..." : med)}</span></td>
          <td><span class="badge badge-primary">${esc(stageDisp)}</span></td>
          <td>${esc(c.expected_refill_date || "-")}</td>
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
      tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: #ef4444; padding: 1.5rem;">Error loading candidate queue: ${esc(err.message)}</td></tr>`;
    }
  }
}

function setupWhatsAppListeners() {
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
  ["wa-target-date", "wa-target-month", "wa-tier-filter", "wa-mobile-filter", "wa-sync-window"].forEach((id) => {
    document.getElementById(id)?.addEventListener("change", loadWhatsAppGateway);
  });
  document.getElementById("wa-sync-window")?.addEventListener("input", (e) => setText("wa-sync-win-val", e.target.value));
  document.getElementById("btn-refresh-wa")?.addEventListener("click", loadWhatsAppGateway);
  document.getElementById("wa-search-input")?.addEventListener("input", () => {
    clearTimeout(waSearchTimer);
    waSearchTimer = setTimeout(loadWhatsAppGateway, 300);
  });
  document.getElementById("wa-preview-patient-select")?.addEventListener("change", (e) => {
    selectCandidateForPreview(parseInt(e.target.value, 10));
  });
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
    opt.textContent = `${c.customer_name || "-"} — ${c.medication || "-"} (${c.tier || "-"})`;
    sel.appendChild(opt);
  });
}

async function selectCandidateForPreview(index) {
  const cand = currentWhatsAppCandidates[index];
  if (!cand) return;
  const sel = document.getElementById("wa-preview-patient-select");
  if (sel) sel.value = index;
  await renderWhatsAppPreview(cand);
}

/** Escape, then WhatsApp formatting: *bold* -> <b>, newlines -> <br>. */
function formatWhatsAppText(text) {
  return esc(text || "")
    .replace(/\*([^*\n]+)\*/g, "<b>$1</b>")
    .replace(/\r?\n/g, "<br>");
}

async function renderWhatsAppPreview(cand) {
  const previewBody = document.getElementById("wa-preview-body");
  const previewBadge = document.getElementById("wa-preview-badge");
  const previewFooter = document.getElementById("wa-preview-footer");
  if (previewFooter) previewFooter.innerText = "";
  if (!cand) {
    if (previewBody) previewBody.innerText = "No patient candidate selected.";
    if (previewBadge) previewBadge.innerText = "—";
    return;
  }
  if (previewBadge) previewBadge.innerText = `${cand.tier || "DUE"} • ${cand.refill_function || "DUE_REFILL"}`;
  try {
    const res = await ApiClient.previewWhatsApp({
      customer_name: cand.customer_name,
      medications: cand.medication,
      expected_refill_date: cand.expected_refill_date,
      tier: cand.tier,
      stage_offset: cand.stage_offset,
    });
    if (previewBody) previewBody.innerHTML = formatWhatsAppText(res.visual_preview);
    if (previewFooter && res.template_name) previewFooter.innerText = `Template: ${res.template_name}`;
  } catch (err) {
    if (previewBody) previewBody.innerText = `Could not generate message preview: ${err.message}`;
  }
}

// ------------------------------------------------------------------------------
// WhatsApp Messages: send today's reminders (Yes / No confirmation)
// ------------------------------------------------------------------------------
let waSendSummary = null;

function setupWhatsAppSend() {
  setupWhatsAppListeners();
  document.getElementById("btn-wa-send-today")?.addEventListener("click", () => openSendTodayModal());
}

function waSkipped(s) {
  return Number(s.no_valid_mobile || 0) + Number(s.opted_out || 0) + Number(s.already_sent || 0);
}

async function loadWhatsAppSendSummary() {
  const { mode, targetDate } = waTarget();
  const today = localDateStr(new Date());
  const day = mode === "DATE" ? targetDate : today;
  const btn = document.getElementById("btn-wa-send-today");
  const note = document.getElementById("wa-send-note");
  try {
    const s = await ApiClient.getWhatsAppSendSummary(day);
    waSendSummary = s;
    const toSend = Number(s.to_send || 0);
    const label = s.target_label || fmtLongDate(day);
    setText("wa-send-headline", `${toSend.toLocaleString()} messages ready for ${label}`);
    setText("wa-send-subline",
      `${fmtNum(s.medicines || 0)} medicines covered · ${waSkipped(s).toLocaleString()} skipped (no valid mobile, opted out or already sent) · ${fmtNum(s.holdout || 0)} in the no-reminder control group`);
    const isTodaySelected = mode === "DATE" && targetDate === today;
    const canSend = isTodaySelected && toSend > 0;
    if (btn) {
      btn.disabled = !canSend;
      btn.innerText = `📤  Send WhatsApp Messages (${toSend.toLocaleString()})`;
    }
    if (note) {
      if (!isTodaySelected) {
        note.innerText = "To send, choose Specific Date with today's date above. Messages are sent only for today's reminders.";
        note.style.display = "block";
      } else if (toSend === 0) {
        note.innerText = "No messages to send for today - every reminder is already sent, opted out or missing a mobile number.";
        note.style.display = "block";
      } else {
        note.style.display = "none";
      }
    }
  } catch (err) {
    waSendSummary = null;
    setText("wa-send-headline", "Send summary unavailable");
    setText("wa-send-subline", err.message);
    if (btn) btn.disabled = true;
    if (note) note.style.display = "none";
  }
}

function openSendTodayModal() {
  const s = waSendSummary;
  if (!s) return;
  const n = Number(s.to_send || 0);
  const nTxt = n.toLocaleString();
  const label = s.target_label || "today";
  const skipped = waSkipped(s) + Number(s.holdout || 0);
  const configured = s.whatsapp_configured !== false;
  const stale = !!s.sales_data_stale;

  const body = `
    <p>You are about to send <b>${esc(nTxt)}</b> WhatsApp messages to patients for <b>${esc(label)}</b>.</p>
    <div class="modal-stats">
      <div class="modal-stat"><div class="modal-stat-value">${esc(nTxt)}</div><div class="modal-stat-label">Messages to send</div></div>
      <div class="modal-stat"><div class="modal-stat-value">${esc(fmtNum(s.medicines || 0))}</div><div class="modal-stat-label">Medicines covered</div></div>
      <div class="modal-stat"><div class="modal-stat-value">${esc(skipped.toLocaleString())}</div><div class="modal-stat-label">Skipped</div></div>
    </div>
    <p class="panel-caption">Skipped: ${esc(fmtNum(s.no_valid_mobile || 0))} without a valid mobile number · ${esc(fmtNum(s.opted_out || 0))} opted out (STOP) · ${esc(fmtNum(s.already_sent || 0))} already sent · ${esc(fmtNum(s.holdout || 0))} in the control group (not messaged, so we can measure how many extra refills reminders bring).</p>
    ${!configured ? `<div class="alert alert-danger">WhatsApp sending is not set up (Xinno API key or WABA number missing). Please contact your administrator.</div>` : ""}
    ${configured && stale ? `
      <div class="alert alert-warning">${esc(s.stale_message || "The latest sales data is not up to date.")}</div>
      <label class="check-row"><input type="checkbox" id="wa-send-override-chk"> Send anyway - today's sales are already uploaded</label>
      <label class="form-field"><span>Reason (saved in the log)</span><input type="text" id="wa-send-override-reason" maxlength="300"></label>` : ""}
    <div class="alert alert-danger modal-error" style="display:none;"></div>
    <div class="modal-progress" style="display:none;"><span class="spinner"></span> Sending ${esc(nTxt)} WhatsApp messages… please keep this page open.</div>`;
  const footer = `
    <button class="btn btn-outline" id="wa-send-cancel">No, cancel</button>
    ${configured ? `<button class="btn btn-primary" id="wa-send-yes">Yes, send ${esc(nTxt)} messages</button>` : ""}`;
  const { body: bodyEl, footer: footerEl } = openModal("Send WhatsApp reminders", body, footer, { onClose: () => loadWhatsAppGateway() });

  const yes = footerEl.querySelector("#wa-send-yes");
  const cancel = footerEl.querySelector("#wa-send-cancel");
  cancel.addEventListener("click", () => closeModal());
  if (!yes) return;

  const chk = bodyEl.querySelector("#wa-send-override-chk");
  const reason = bodyEl.querySelector("#wa-send-override-reason");
  const refreshYes = () => {
    yes.disabled = n <= 0 || (!!chk && !(chk.checked && reason.value.trim()));
  };
  chk?.addEventListener("change", refreshYes);
  reason?.addEventListener("input", refreshYes);
  refreshYes();

  yes.addEventListener("click", async () => {
    const errBox = bodyEl.querySelector(".modal-error");
    const progress = bodyEl.querySelector(".modal-progress");
    errBox.style.display = "none";
    progress.style.display = "flex";
    yes.disabled = true;
    cancel.disabled = true;
    if (chk) { chk.disabled = true; reason.disabled = true; }
    setModalLocked(true);
    try {
      const res = await ApiClient.sendWhatsAppToday(chk ? reason.value.trim() : null);
      setModalLocked(false);
      showSendResult(res, bodyEl, footerEl);
    } catch (err) {
      setModalLocked(false);
      progress.style.display = "none";
      cancel.disabled = false;
      if (chk) { chk.disabled = false; reason.disabled = false; }
      if (err.status === 409 && !chk) {
        // Server says the sales data is stale: re-open with the "send anyway" fields.
        waSendSummary = { ...s, sales_data_stale: true, stale_message: err.message };
        closeModal(true);
        openSendTodayModal();
        return;
      }
      errBox.innerText = err.message;
      errBox.style.display = "block";
      refreshYes();
    }
  });
}

function showSendResult(res, bodyEl, footerEl) {
  const sent = Number(res.sent || 0);
  const failed = Number(res.failed || 0);
  const label = res.target_label || "today";
  const failedRows = res.failed_rows || [];
  bodyEl.innerHTML = failed === 0
    ? `<div class="alert alert-success">✅ ${esc(sent.toLocaleString())} WhatsApp messages sent for ${esc(label)}.</div>`
    : `<div class="alert alert-warning">${esc(sent.toLocaleString())} sent, ${esc(failed.toLocaleString())} failed for ${esc(label)}. Failed messages are listed below and can be sent again later.</div>
       ${failedRows.length ? `<div class="table-responsive" style="max-height: 300px;"><table class="data-table">
         <thead><tr><th>Customer</th><th>Mobile</th><th>Medicines</th><th>Reason</th></tr></thead>
         <tbody>${failedRows.map((f) => `<tr><td>${esc(f.customer)}</td><td><code>${esc(formatDisplayPhone10(f.mobile))}</code></td><td>${esc(f.medicines)}</td><td>${esc(f.reason)}</td></tr>`).join("")}</tbody>
       </table></div>` : ""}`;
  footerEl.innerHTML = `<button class="btn btn-primary" id="wa-send-done">Done</button>`;
  footerEl.style.display = "";
  footerEl.querySelector("#wa-send-done").addEventListener("click", () => {
    closeModal();  // onClose refreshes the send summary + candidates
    loadDeliveryLog();
    loadInitialData();
  });
}

// ------------------------------------------------------------------------------
// WhatsApp Messages: Stop / Resume messages (patient consent)
// ------------------------------------------------------------------------------
let consentSearchTimer = null;
let consentResults = [];

function setupConsentActions() {
  document.getElementById("consent-search-input")?.addEventListener("input", () => {
    clearTimeout(consentSearchTimer);
    consentSearchTimer = setTimeout(runConsentSearch, 300);
  });
}

async function runConsentSearch() {
  const q = (document.getElementById("consent-search-input")?.value || "").trim();
  const wrap = document.getElementById("consent-search-results");
  if (!wrap) return;
  if (q.length < 2) {
    consentResults = [];
    wrap.innerHTML = `<div class="empty-note">Type at least 2 letters of the name or digits of the mobile number.</div>`;
    return;
  }
  try {
    const rows = await ApiClient.searchCustomers(q);
    consentResults = Array.isArray(rows) ? rows : [];
    if (!consentResults.length) {
      wrap.innerHTML = `<div class="empty-note">No patient found for “${esc(q)}”.</div>`;
      return;
    }
    wrap.innerHTML = `
      <div class="table-responsive">
        <table class="data-table">
          <thead><tr><th>Patient</th><th>Mobile</th><th>WhatsApp messages</th><th></th></tr></thead>
          <tbody>${consentResults.map((r, i) => `<tr>
            <td><strong>${esc(r.customer_name || r.customer_id || "-")}</strong></td>
            <td><code>${esc(formatDisplayPhone10(r.mobile_number))}</code></td>
            <td>${r.messages_allowed ? `<span class="badge badge-success">Allowed</span>` : `<span class="badge badge-danger">Stopped</span>`}</td>
            <td><button class="btn btn-sm ${r.messages_allowed ? "btn-danger" : "btn-outline"} btn-consent-toggle" data-idx="${i}">${r.messages_allowed ? "Stop messages" : "Resume messages"}</button></td>
          </tr>`).join("")}</tbody>
        </table>
      </div>`;
    wrap.querySelectorAll(".btn-consent-toggle").forEach((btn) => {
      btn.addEventListener("click", () => {
        const r = consentResults[parseInt(btn.getAttribute("data-idx"), 10)];
        if (r) confirmConsentChange(r);
      });
    });
  } catch (err) {
    wrap.innerHTML = `<div class="alert alert-danger">${esc(err.message)}</div>`;
  }
}

function confirmConsentChange(r) {
  const allow = !r.messages_allowed;
  const name = r.customer_name || r.customer_id || "this patient";
  const mobile = formatDisplayPhone10(r.mobile_number);
  confirmModal({
    title: allow ? "Resume WhatsApp messages" : "Stop WhatsApp messages",
    messageHtml: allow
      ? `<p>Resume WhatsApp refill reminders for <b>${esc(name)}</b> (${esc(mobile)})?</p>`
      : `<p>Stop all WhatsApp refill reminders for <b>${esc(name)}</b> (${esc(mobile)})? They will not get any messages until resumed.</p>`,
    confirmLabel: allow ? "Yes, resume messages" : "Yes, stop messages",
    confirmClass: allow ? "btn-primary" : "btn-danger",
    onConfirm: async () => {
      const mob = r.mobile_number && mobile !== "-" ? r.mobile_number : null;
      await ApiClient.setWhatsAppMessages(allow, r.customer_id, mob);
      showToast(allow ? `WhatsApp messages resumed for ${name}.` : `WhatsApp messages stopped for ${name}.`, "success");
      runConsentSearch();
      loadWhatsAppSendSummary();
    },
  });
}

// ------------------------------------------------------------------------------
// WhatsApp Messages: Messages Sent (delivery log)
// ------------------------------------------------------------------------------
function setupDeliveryLog() {
  document.getElementById("wa-log-day")?.addEventListener("change", loadDeliveryLog);
  document.getElementById("wa-log-status")?.addEventListener("change", loadDeliveryLog);
}

async function loadDeliveryLog() {
  const wrap = document.getElementById("wa-log-table-wrap");
  if (!wrap) return;
  const dayInput = document.getElementById("wa-log-day");
  if (dayInput && !dayInput.value) dayInput.value = localDateStr(new Date());
  const day = dayInput?.value || localDateStr(new Date());
  const status = document.getElementById("wa-log-status")?.value || "ALL";
  try {
    const res = await ApiClient.getDeliveryLog(day, status);
    setText("wa-log-summary", `Sent: ${fmtNum(res.sent || 0)} · Failed: ${fmtNum(res.failed || 0)}`);
    const rows = res.rows || [];
    if (!rows.length) {
      wrap.innerHTML = `<div class="empty-note">No messages for ${esc(fmtLongDate(day))}.</div>`;
      return;
    }
    wrap.innerHTML = `
      <div class="table-responsive" style="max-height: 420px;">
        <table class="data-table">
          <thead><tr><th>Time (India)</th><th>Patient</th><th>Mobile</th><th>Status</th><th>Reason</th></tr></thead>
          <tbody>${rows.map((r) => {
            const st = String(r.status || "");
            const cls = /sent/i.test(st) ? "badge-success" : (/fail/i.test(st) ? "badge-danger" : "badge-secondary");
            return `<tr><td>${esc(r.time)}</td><td>${esc(r.patient)}</td><td><code>${esc(formatDisplayPhone10(r.mobile))}</code></td>
              <td><span class="badge ${cls}">${esc(st)}</span></td><td style="font-size:0.82rem;">${esc(r.reason || "")}</td></tr>`;
          }).join("")}</tbody>
        </table>
      </div>`;
  } catch (err) {
    setText("wa-log-summary", "");
    wrap.innerHTML = `<div class="alert alert-danger">${esc(err.message)}</div>`;
  }
}
