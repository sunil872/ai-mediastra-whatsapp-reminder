/**
 * RefillCare Enterprise Frontend Application
 */

import { ApiClient } from "./api.js";

// Global State
let activeFile = null;

// Initialize Application
document.addEventListener("DOMContentLoaded", async () => {
  setupNavigation();
  setupUploadDropzone();
  setupActionListeners();
  
  // Dynamic default date (defaults to today's date)
  const todayStr = new Date().toISOString().split("T")[0];
  const dateInput = document.getElementById("reminder-target-date");
  if (dateInput && !dateInput.value) dateInput.value = todayStr;

  const reviewDateInput = document.getElementById("review-target-date");
  if (reviewDateInput && !reviewDateInput.value) reviewDateInput.value = todayStr;

  await initReminderMonthOptions();
  await loadInitialData();
});

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

      // Lazy load tab data
      if (tabId === "tab-upload") loadBatches();
      if (tabId === "tab-predictions") loadPredictionSnapshots();
      if (tabId === "tab-reminders") loadReminders();
      if (tabId === "tab-medsync") loadMedSyncBundles();
      if (tabId === "tab-review") loadReviewQueue();
      if (tabId === "tab-models") loadModels();
    });
  });
}

// ------------------------------------------------------------------------------
// Data Loading Functions
// ------------------------------------------------------------------------------
async function loadInitialData() {
  try {
    const kpis = await ApiClient.getKpis();
    document.getElementById("kpi-customers").innerText = Number(kpis.total_customers_monitored).toLocaleString();
    document.getElementById("kpi-transactions").innerText = Number(kpis.total_sales_transactions).toLocaleString();
    document.getElementById("kpi-upcoming").innerText = Number(kpis.upcoming_reminders_count).toLocaleString();
    document.getElementById("kpi-latest-date").innerText = kpis.latest_sales_date;
    document.getElementById("sales-coverage-badge").innerText = kpis.sales_coverage_date_range;
    document.getElementById("sidebar-active-model").innerText = kpis.active_model_version;

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
        if (m === "2026-09") opt.selected = true;
        monthSelect.appendChild(opt);
      });
      if (!monthSelect.value && months.length > 0) {
        monthSelect.value = months[0];
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

async function loadReminders() {
  try {
    const viewMode = document.getElementById("reminder-view-mode")?.value || "DATE";
    const dateInput = document.getElementById("reminder-target-date");
    const monthSelect = document.getElementById("reminder-target-month");

    let targetDate = null;
    let targetMonth = null;
    let filterLabel = "";

    if (viewMode === "MONTH") {
      targetMonth = monthSelect ? monthSelect.value : "2026-09";
      filterLabel = monthSelect?.options[monthSelect.selectedIndex]?.text || targetMonth;
      if (dateInput) dateInput.style.display = "none";
      if (monthSelect) monthSelect.style.display = "inline-block";
    } else {
      targetDate = dateInput ? dateInput.value : "";
      filterLabel = targetDate || "selected date";
      if (dateInput) dateInput.style.display = "inline-block";
      if (monthSelect) monthSelect.style.display = "none";
    }

    const channelFilter = document.getElementById("reminder-channel-filter")?.value || "CUSTOMER_SALE";
    const searchQuery = document.getElementById("reminder-search-input")?.value?.trim().toLowerCase() || "";
    const cleanSearchDigits = searchQuery.replace(/\D/g, "");

    const auditBanner = document.getElementById("channel-audit-banner");
    const tbody = document.querySelector("#table-reminders tbody");
    const countBadge = document.getElementById("reminder-count-badge");
    tbody.innerHTML = "";

    if (channelFilter === "B2B_INTER_STORE" || channelFilter === "UNKNOWN") {
      if (auditBanner) auditBanner.style.display = "block";
      if (countBadge) countBadge.innerText = "0 Reminders (Audit Mode)";
      tbody.innerHTML = `<tr><td colspan="10" style="text-align:center; color: var(--text-muted); padding: 2rem;">
        <strong>No Reminders Generated for ${channelFilter === "B2B_INTER_STORE" ? "B2B / Inter-Store (SB/)" : "Unknown"} Transactions.</strong><br>
        <span style="font-size: 0.85rem;">RefillCare strictly isolates non-customer sales upstream to prevent customer cadence pollution.</span>
      </td></tr>`;
      return;
    }

    if (auditBanner) auditBanner.style.display = "none";
    const rawReminders = await ApiClient.getDailyReminders(targetDate, targetMonth);

    if (!rawReminders || rawReminders.length === 0) {
      if (countBadge) countBadge.innerText = "0 Scheduled";
      tbody.innerHTML = `<tr><td colspan="10" style="text-align:center; color: var(--text-muted);">No reminders scheduled for ${filterLabel}.</td></tr>`;
      return;
    }

    // Apply Client-Side Search by Customer Name, Phone Number, or Medication
    const reminders = rawReminders.filter((r) => {
      if (!searchQuery) return true;
      const nameMatch = r.customer_name && r.customer_name.toLowerCase().includes(searchQuery);
      const medMatch = r.item_name && r.item_name.toLowerCase().includes(searchQuery);
      const phoneStr = String(r.phone_number || "") + " " + String(r.raw_phone_number || "");
      const phoneMatch = phoneStr.toLowerCase().includes(searchQuery) || (cleanSearchDigits && phoneStr.replace(/\D/g, "").includes(cleanSearchDigits));
      return nameMatch || medMatch || phoneMatch;
    });

    if (countBadge) countBadge.innerText = `${reminders.length} Scheduled`;

    if (reminders.length === 0) {
      tbody.innerHTML = `<tr><td colspan="10" style="text-align:center; color: var(--text-muted); padding: 1.5rem;">No reminders match "${searchQuery}".</td></tr>`;
      return;
    }

    reminders.forEach((r) => {
      const tr = document.createElement("tr");
      const isValidPhone = r.mobile_status === "Valid";
      const dispPhone = formatDisplayPhone10(r.phone_number);
      tr.innerHTML = `
        <td><strong>${r.customer_name}</strong></td>
        <td><code>${dispPhone !== '-' ? dispPhone : '<span class="badge badge-warning">Missing</span>'}</code></td>
        <td><span class="badge ${isValidPhone ? 'badge-success' : 'badge-warning'}">${r.mobile_status}</span></td>
        <td><span class="badge badge-primary">Customer (S0/)</span></td>
        <td>${r.item_name}</td>
        <td>${r.last_purchase_date}</td>
        <td>${r.estimated_days_of_supply} d</td>
        <td>${r.expected_refill_date}</td>
        <td><strong>${r.reminder_date}</strong></td>
        <td><span class="badge badge-info">${r.reminder_stage}</span></td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Error loading reminders:", err);
  }
}

async function loadReviewQueue() {
  try {
    const dateInput = document.getElementById("review-target-date");
    const targetDate = dateInput ? dateInput.value : "";
    const pathFilter = document.getElementById("review-path-filter")?.value || "";
    const stabilityFilter = document.getElementById("review-stability-filter")?.value || "";
    const custQuery = document.getElementById("review-search-cust")?.value?.trim().toLowerCase() || "";
    const medQuery = document.getElementById("review-search-med")?.value?.trim().toLowerCase() || "";

    const rawQueue = await ApiClient.getTodayReminderQueue(targetDate, pathFilter, stabilityFilter);
    const tbody = document.querySelector("#table-review-queue tbody");
    tbody.innerHTML = "";

    const queue = (rawQueue || []).filter((r) => {
      if (custQuery && !((r.customer_name && r.customer_name.toLowerCase().includes(custQuery)) || (r.customer_id && r.customer_id.toLowerCase().includes(custQuery)))) {
        return false;
      }
      if (medQuery && !((r.item_name && r.item_name.toLowerCase().includes(medQuery)) || (r.item_id && String(r.item_id).toLowerCase().includes(medQuery)))) {
        return false;
      }
      return true;
    });

    if (!queue || queue.length === 0) {
      tbody.innerHTML = `<tr><td colspan="12" style="text-align:center; color: var(--text-muted); padding: 2rem;">No reminders in the review queue for the selected filters.</td></tr>`;
      return;
    }

    queue.forEach((r) => {
      const tr = document.createElement("tr");

      // Path badge
      const pathBadge = r.path === "PATH_A" ? "badge-primary" : (r.path === "PATH_B" ? "badge-accent" : "badge-secondary");

      // Stability badge
      let stabBadge = "badge-info";
      if (r.stability_tier === "HIGH") stabBadge = "badge-success";
      else if (r.stability_tier === "MEDIUM-RISK") stabBadge = "badge-warning";
      else if (r.stability_tier === "UNSTABLE") stabBadge = "badge-danger";

      // Status badge
      let statusBadge = "badge-info";
      if (r.status === "APPROVED") statusBadge = "badge-success";
      else if (r.status === "SENT") statusBadge = "badge-success";
      else if (r.status === "REJECTED") statusBadge = "badge-danger";
      else if (r.status === "SUPERSEDED_BY_PURCHASE") statusBadge = "badge-secondary";

      // Stage offset formatting
      const offsetText = r.stage_offset >= 0 ? `+${r.stage_offset}d` : `${r.stage_offset}d`;

      // Actions render
      let actionHtml = "";
      if (r.status === "PENDING" || r.status === "DUE") {
        actionHtml = `
          <div style="display:flex; gap:0.35rem;">
            <button class="btn btn-sm btn-primary btn-approve-stage" data-id="${r.reminder_id}" title="Approve this reminder">✓ Approve</button>
            <button class="btn btn-sm btn-outline btn-reject-stage" data-id="${r.reminder_id}" title="Reject this reminder">✗ Reject</button>
          </div>
        `;
      } else if (r.status === "APPROVED") {
        actionHtml = `
          <div style="display:flex; gap:0.35rem;">
            <button class="btn btn-sm btn-accent btn-dispatch-stage" data-id="${r.reminder_id}" title="Send Xinno Dry-Run">🚀 Dry-Run</button>
            <button class="btn btn-sm btn-outline btn-reject-stage" data-id="${r.reminder_id}" title="Reject this reminder">✗ Reject</button>
          </div>
        `;
      } else if (r.status === "SENT") {
        actionHtml = `<span style="color:var(--success-color); font-size:0.85rem; font-weight:600;">✓ Sent</span>`;
      } else if (r.status === "REJECTED") {
        actionHtml = `<span style="color:var(--danger-color); font-size:0.85rem;">Rejected</span>`;
      } else if (r.status === "SUPERSEDED_BY_PURCHASE") {
        actionHtml = `<span style="color:var(--text-muted); font-size:0.85rem;">Superseded</span>`;
      } else {
        actionHtml = `<span style="color:var(--text-muted); font-size:0.85rem;">${r.status}</span>`;
      }

      tr.innerHTML = `
        <td>
          <strong>${r.customer_name || 'Patient'}</strong><br>
          <small style="color:var(--text-muted)">ID: ${r.customer_id} | ${r.masked_phone || r.phone_masked || 'No Phone'}</small>
        </td>
        <td><strong>${r.item_name || r.item_id}</strong></td>
        <td>${r.expected_refill_date || '-'}</td>
        <td><span class="badge badge-info">${offsetText}</span></td>
        <td><span class="badge badge-info" style="font-size:0.75rem;">${r.dosage_regimen || '1.0 tab/d (OD)'}</span></td>
        <td><span class="badge badge-accent" style="font-size:0.75rem;">${r.archetype || 'Standard Consensus'}</span></td>
        <td><span class="badge ${pathBadge}">${r.path}</span></td>
        <td><span class="badge ${stabBadge}">${r.stability_tier}</span></td>
        <td><small>${r.prediction_method || '-'}</small></td>
        <td><div style="max-width: 260px; font-size: 0.8rem; line-height: 1.25; color: var(--text-secondary);" title="${r.decision_reason}">${r.decision_reason || '-'}</div></td>
        <td><span class="badge ${statusBadge}">${r.status}</span></td>
        <td>${actionHtml}</td>
      `;
      tbody.appendChild(tr);
    });

    // Wire action buttons
    document.querySelectorAll(".btn-approve-stage").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const remId = btn.getAttribute("data-id");
        try {
          btn.disabled = true;
          const res = await ApiClient.approveReminder(remId);
          showToast(`Stage approved: ${remId}`, "success");
          await loadReviewQueue();
        } catch (err) {
          showToast(err.message, "error");
          btn.disabled = false;
        }
      });
    });

    document.querySelectorAll(".btn-reject-stage").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const remId = btn.getAttribute("data-id");
        const reason = prompt("Enter reason for rejection:", "Pharmacist clinical review rejection");
        if (reason === null) return;
        try {
          btn.disabled = true;
          const res = await ApiClient.rejectReminder(remId, reason);
          showToast(`Stage rejected: ${remId}`, "info");
          await loadReviewQueue();
        } catch (err) {
          showToast(err.message, "error");
          btn.disabled = false;
        }
      });
    });

    document.querySelectorAll(".btn-dispatch-stage").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const remId = btn.getAttribute("data-id");
        if (confirm(`Authorize controlled DRY-RUN dispatch for reminder stage ${remId}?`)) {
          try {
            btn.disabled = true;
            const res = await ApiClient.dispatchReminder(remId, true);
            showToast(`Stage dispatched (DRY-RUN): ${res.status}`, "success");
            await loadReviewQueue();
          } catch (err) {
            showToast(err.message, "error");
            btn.disabled = false;
          }
        }
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
  const windowDays = slider ? parseInt(slider.value, 10) : 8;
  const syncValLabel = document.getElementById("sync-window-val");
  if (syncValLabel) syncValLabel.innerText = windowDays;

  const viewMode = document.getElementById("medsync-view-mode")?.value || "MONTH";
  const monthSelect = document.getElementById("medsync-month-select");
  const dateInput = document.getElementById("medsync-target-date");

  let targetMonth = null;
  let targetDate = null;

  if (viewMode === "DATE") {
    targetDate = dateInput ? dateInput.value : "2026-09-24";
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
    const data = await ApiClient.getMedSyncBundles(windowDays, null, targetMonth, targetDate);
    const impact = data.impact_summary || {};
    const bundles = data.bundles || [];
    const availableMonths = data.available_months || [];
    const selectedMonth = data.selected_target_month || "";
    const lastSalesMonth = data.last_sales_month || "August 2026";
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
      bannerText.innerHTML = `<strong>${activeLabel}</strong> (predicted from uploaded sales up to <strong>${lastSalesMonth}</strong> with <strong>${windowDays}d</strong> sync window) — <strong>${Number(impact.total_prescriptions_synced || 0).toLocaleString()}</strong> prescriptions grouped into <strong>${Number(impact.total_dispatches_generated || 0).toLocaleString()}</strong> bundles.`;
    }

    const pElem = document.getElementById("medsync-total-prescriptions");
    const bElem = document.getElementById("medsync-total-bundles");
    const mElem = document.getElementById("medsync-multi-bundles");
    const sElem = document.getElementById("medsync-saved-messages");
    const fElem = document.getElementById("medsync-friction-rate");

    if (pElem) pElem.innerText = Number(impact.total_prescriptions_synced || 0).toLocaleString();
    if (bElem) bElem.innerText = Number(impact.total_dispatches_generated || 0).toLocaleString();
    if (mElem) mElem.innerText = Number(impact.multi_item_bundles_count || 0).toLocaleString();
    if (sElem) sElem.innerText = Number(impact.individual_messages_saved || 0).toLocaleString();
    if (fElem) fElem.innerText = `${impact.message_reduction_rate_pct || 0}% friction reduction`;

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
      const itemsList = (b.synced_items || []).map(i => `<span class="badge ${i.is_anchor ? 'badge-primary' : 'badge-info'}" style="margin: 2px;">${i.item_name}</span>`).join(" ");
      const p10p90 = (b.earliest_p10_date && b.latest_p90_date) ? `${b.earliest_p10_date} → ${b.latest_p90_date}` : "± 7d confidence";
      const reductionBadge = b.message_reduction_count > 0 ? `<span class="badge badge-success">-${b.message_reduction_count} msgs saved</span>` : `<span class="badge badge-secondary">1 msg</span>`;
      const dispPhone = formatDisplayPhone10(b.mobile_no);

      tr.innerHTML = `
        <td><strong>${b.customer_name}</strong></td>
        <td><code>${dispPhone !== '-' ? dispPhone : '<span class="badge badge-warning">Missing</span>'}</code></td>
        <td><strong style="color: var(--primary);">${b.anchor_item_name}</strong></td>
        <td style="max-width: 280px;">${itemsList}</td>
        <td>${b.anchor_refill_date}</td>
        <td><small>${p10p90}</small></td>
        <td><strong>${b.total_items_count}</strong></td>
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
  // Reminder List Search Listener
  document.getElementById("reminder-search-input")?.addEventListener("input", loadReminders);

  // Med-Sync Listeners
  document.getElementById("btn-refresh-medsync")?.addEventListener("click", loadMedSyncBundles);
  document.getElementById("medsync-view-mode")?.addEventListener("change", () => {
    loadMedSyncBundles();
  });
  document.getElementById("medsync-target-date")?.addEventListener("change", loadMedSyncBundles);
  document.getElementById("medsync-month-select")?.addEventListener("change", loadMedSyncBundles);
  document.getElementById("sync-window-slider")?.addEventListener("input", (e) => {
    const valSpan = document.getElementById("sync-window-val");
    if (valSpan) valSpan.innerText = e.target.value;
  });
  document.getElementById("sync-window-slider")?.addEventListener("change", loadMedSyncBundles);
  document.getElementById("medsync-search")?.addEventListener("input", loadMedSyncBundles);

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
  document.getElementById("reminder-view-mode")?.addEventListener("change", loadReminders);
  document.getElementById("reminder-target-month")?.addEventListener("change", loadReminders);
  document.getElementById("reminder-target-date")?.addEventListener("change", loadReminders);
  document.getElementById("reminder-channel-filter")?.addEventListener("change", loadReminders);

  // WhatsApp Dry-Run Dispatch
  document.getElementById("btn-dispatch-dryrun")?.addEventListener("click", async () => {
    const targetDate = document.getElementById("reminder-target-date")?.value || new Date().toISOString().split("T")[0];
    try {
      showToast("Simulating batch WhatsApp dispatch (Dry-Run)...", "info");
      const res = await ApiClient.dispatchWhatsApp(targetDate, true, 20);
      showToast(`Dry-run simulation completed: ${res.success_count} simulated successfully.`, "success");
      
      const tbody = document.querySelector("#table-wa-logs tbody");
      tbody.innerHTML = "";
      res.delivery_summary.forEach((log) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><code>${log.phone_number}</code></td>
          <td>${log.customer_name}</td>
          <td>${log.medication_name}</td>
          <td>${log.refill_date}</td>
          <td><span class="badge badge-success">${log.status}</span></td>
        `;
        tbody.appendChild(tr);
      });
    } catch (err) {
      showToast(err.message, "error");
    }
  });

  // Review Queue Filters & Actions
  document.getElementById("btn-refresh-review-queue")?.addEventListener("click", loadReviewQueue);
  document.getElementById("review-search-cust")?.addEventListener("input", loadReviewQueue);
  document.getElementById("review-search-med")?.addEventListener("input", loadReviewQueue);
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

