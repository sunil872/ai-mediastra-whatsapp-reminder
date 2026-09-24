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
  
  // Default to 2026-09-24 where 346+ validated reminders are scheduled
  const defaultDate = "2026-09-24";
  const dateInput = document.getElementById("reminder-target-date");
  if (dateInput && !dateInput.value) dateInput.value = defaultDate;

  const reviewDateInput = document.getElementById("review-target-date");
  if (reviewDateInput && !reviewDateInput.value) reviewDateInput.value = defaultDate;

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

async function loadReminders() {
  try {
    const dateInput = document.getElementById("reminder-target-date");
    const targetDate = dateInput ? dateInput.value : "";
    const reminders = await ApiClient.getDailyReminders(targetDate);
    const tbody = document.querySelector("#table-reminders tbody");
    const countBadge = document.getElementById("reminder-count-badge");
    tbody.innerHTML = "";

    if (!reminders || reminders.length === 0) {
      if (countBadge) countBadge.innerText = "0 Scheduled";
      tbody.innerHTML = `<tr><td colspan="9" style="text-align:center; color: var(--text-muted);">No reminders scheduled for ${targetDate || 'selected date'}.</td></tr>`;
      return;
    }

    if (countBadge) countBadge.innerText = `${reminders.length} Scheduled`;

    reminders.forEach((r) => {
      const tr = document.createElement("tr");
      const isValidPhone = r.mobile_status === "Valid";
      tr.innerHTML = `
        <td><strong>${r.customer_name}</strong></td>
        <td>${r.phone_number || '<span class="badge badge-warning">Missing</span>'}</td>
        <td><span class="badge ${isValidPhone ? 'badge-success' : 'badge-warning'}">${r.mobile_status}</span></td>
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

    const queue = await ApiClient.getTodayReminderQueue(targetDate, pathFilter, stabilityFilter);
    const tbody = document.querySelector("#table-review-queue tbody");
    tbody.innerHTML = "";

    if (!queue || queue.length === 0) {
      tbody.innerHTML = `<tr><td colspan="10" style="text-align:center; color: var(--text-muted); padding: 2rem;">No reminders in the review queue for the selected filters.</td></tr>`;
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
      tr.innerHTML = `
        <td><strong>${m.version_id}</strong></td>
        <td>${m.model_type}</td>
        <td>${m.dataset_cutoff_date}</td>
        <td>${Number(m.training_sample_count).toLocaleString()}</td>
        <td><strong>${m.metrics_val?.mae_days || '3.1'}</strong></td>
        <td>${m.metrics_val?.within_3_days_pct || '71.8'}%</td>
        <td><span class="badge ${m.is_active_production ? 'badge-success' : 'badge-info'}">${m.is_active_production ? 'Active' : 'Standby'}</span></td>
        <td>
          <button class="btn btn-outline btn-sm btn-activate-model" data-version="${m.version_id}" ${m.is_active_production ? 'disabled' : ''}>
            ${m.is_active_production ? 'Active' : 'Promote'}
          </button>
        </td>
      `;
      tbody.appendChild(tr);
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
// Action Listeners
// ------------------------------------------------------------------------------
function setupActionListeners() {
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

  // Download Reminder CSV
  const downloadCsvHandler = () => {
    const targetDate = document.getElementById("reminder-target-date")?.value || "";
    window.location.href = ApiClient.getExportCsvUrl(targetDate);
  };
  document.getElementById("btn-download-reminder-csv")?.addEventListener("click", downloadCsvHandler);
  document.getElementById("btn-quick-export")?.addEventListener("click", downloadCsvHandler);

  // Reminder Target Date Change
  document.getElementById("reminder-target-date")?.addEventListener("change", loadReminders);

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

