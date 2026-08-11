"use strict";

/**
 * Agentic AI Test Automation — dashboard frontend.
 *
 * Plain vanilla JS. This file only ever talks to the existing FastAPI
 * backend via fetch() — it contains no test-generation, execution, or
 * analysis logic of its own. Every button click maps to exactly one
 * backend endpoint; this file just renders whatever comes back.
 */

// Single place every API call goes through. Since the dashboard is served
// by FastAPI itself (see app.py), this also works unmodified — same host
// and port. Change this if you serve the frontend separately (and set
// CORS_ORIGINS on the backend to match).
const API_BASE_URL = "http://127.0.0.1:8000";

// --- In-memory state for the current session (nothing here is persisted
// client-side; the database is the source of truth) ---
const state = {
  requirementId: null,
  testCases: {}, // test_case_id -> full TestCase (from POST /test-cases)
  generatedTests: {}, // test_case_id -> {generated_file, validation}
  lastExecution: null,
  lastFailureAnalysis: null,
  lastHealingResult: null,
};

let historyExecutions = [];
let historyFilter = "all";

// --- Small utilities ---

function escapeHtml(value) {
  if (value === null || value === undefined) return "";
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function truncate(text, maxLength) {
  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength - 1)}…`;
}

function detailItem(label, valueHtml, full = false) {
  return `<div class="detail-item${full ? " detail-item--full" : ""}">
    <div class="detail-item__label">${escapeHtml(label)}</div>
    <div class="detail-item__value">${valueHtml}</div>
  </div>`;
}

function listHtml(items) {
  if (!items || items.length === 0) return "<em>None</em>";
  return `<ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
}

function priorityBadge(priority) {
  return `<span class="badge badge--neutral">${escapeHtml(priority)}</span>`;
}

function statusBadgeHtml(status, healed) {
  const classByStatus = {
    passed: "badge--passed",
    failed: "badge--failed",
    error: "badge--error",
    skipped: "badge--skipped",
  };
  if (healed) {
    return '<span class="badge badge--healed">HEALED</span>';
  }
  const cls = classByStatus[status] || "badge--neutral";
  return `<span class="badge ${cls}">${escapeHtml(String(status).toUpperCase())}</span>`;
}

function healingStatusBadge(status) {
  const classByStatus = { healed: "badge--healed", failed: "badge--failed", skipped: "badge--skipped" };
  const cls = classByStatus[status] || "badge--neutral";
  return `<span class="badge ${cls}">${escapeHtml(String(status).toUpperCase())}</span>`;
}

function setButtonLoading(button, loadingText) {
  if (!button) return;
  button.dataset.originalText = button.textContent;
  button.textContent = loadingText;
  button.disabled = true;
}

function clearButtonLoading(button) {
  if (!button) return;
  if (button.dataset.originalText !== undefined) {
    button.textContent = button.dataset.originalText;
    delete button.dataset.originalText;
  }
  button.disabled = false;
}

let toastTimeoutId = null;

function showToast(message, type = "error") {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.className = `toast toast--${type}`;
  toast.hidden = false;
  if (toastTimeoutId) clearTimeout(toastTimeoutId);
  toastTimeoutId = setTimeout(() => {
    toast.hidden = true;
  }, 6000);
}

function showSection(id) {
  const section = document.getElementById(id);
  section.hidden = false;
  section.scrollIntoView({ behavior: "smooth", block: "start" });
}

function openModal(id) {
  document.getElementById(id).hidden = false;
}

function closeModal(id) {
  document.getElementById(id).hidden = true;
}

/**
 * Every network call goes through here. Network failures and API error
 * responses are both normalized into a single, user-friendly Error — the
 * backend's own sanitized `detail` message when available, otherwise a
 * generic message. Nothing here ever surfaces a raw stack trace.
 */
async function apiRequest(method, path, body) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method,
      headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch (_networkError) {
    throw new Error("Unable to reach the server. Please check that the FastAPI server is running.");
  }

  let data = null;
  try {
    data = await response.json();
  } catch (_parseError) {
    data = null;
  }

  if (!response.ok) {
    const detail = data ? data.detail : null;
    let message;
    if (typeof detail === "string") {
      message = detail;
    } else if (Array.isArray(detail) && detail.length > 0 && detail[0].msg) {
      message = detail.map((item) => item.msg).join(" ");
    } else {
      message = `Request failed (HTTP ${response.status}).`;
    }
    throw new Error(message);
  }

  return data;
}

// --- Connection status + stats ---

async function checkHealth() {
  const dot = document.getElementById("connection-dot");
  const text = document.getElementById("connection-text");
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) throw new Error("not ok");
    dot.classList.add("is-online");
    dot.classList.remove("is-offline");
    text.textContent = "Backend connected";
  } catch (_error) {
    dot.classList.add("is-offline");
    dot.classList.remove("is-online");
    text.textContent = "Backend unreachable";
  }
}

async function loadStats() {
  try {
    const stats = await apiRequest("GET", "/api/v1/stats");
    document.getElementById("stat-total").textContent = stats.total_test_cases;
    document.getElementById("stat-passed").textContent = stats.passed;
    document.getElementById("stat-failed").textContent = stats.failed;
    document.getElementById("stat-healed").textContent = stats.healed;
    document.getElementById("stat-success-rate").textContent = `${stats.success_rate}%`;
  } catch (error) {
    showToast(error.message);
  }
}

// --- 1. Requirement -> 2. Requirement Analysis ---

function renderAnalysis(analysis) {
  document.getElementById("analysis-content").innerHTML = [
    detailItem("Test Name", escapeHtml(analysis.test_name)),
    detailItem("Priority", priorityBadge(analysis.priority)),
    detailItem("Description", escapeHtml(analysis.description), true),
    detailItem("Preconditions", listHtml(analysis.preconditions), true),
    detailItem("Steps", listHtml(analysis.steps), true),
    detailItem("Expected Result", escapeHtml(analysis.expected_result), true),
  ].join("");
}

document.getElementById("requirement-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const textarea = document.getElementById("requirement-textarea");
  const button = document.getElementById("analyze-requirement-btn");
  const errorEl = document.getElementById("requirement-error");
  errorEl.hidden = true;

  setButtonLoading(button, "Analyzing...");
  try {
    const result = await apiRequest("POST", "/api/v1/requirements", { requirement: textarea.value });
    state.requirementId = result.requirement_id;
    renderAnalysis(result.analysis);
    showSection("analysis-section");
  } catch (error) {
    errorEl.textContent = error.message;
    errorEl.hidden = false;
  } finally {
    clearButtonLoading(button);
  }
});

// --- 3. Test Cases ---

function renderTestCasesTable(testCases) {
  const tbody = document.getElementById("test-cases-tbody");
  if (testCases.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" class="empty-row">No test cases yet.</td></tr>';
    return;
  }
  tbody.innerHTML = testCases
    .map(
      (tc) => `
    <tr>
      <td>${escapeHtml(tc.test_case_id)}</td>
      <td>${escapeHtml(tc.title)}</td>
      <td>${escapeHtml(tc.type)}</td>
      <td>${priorityBadge(tc.priority)}</td>
      <td>${escapeHtml(tc.expected_result)}</td>
      <td class="row-actions">
        <button class="btn btn--ghost btn--small" type="button" data-action="view-details" data-test-case-id="${escapeHtml(tc.test_case_id)}">Details</button>
        <button class="btn btn--secondary btn--small" type="button" data-action="generate-test" data-test-case-id="${escapeHtml(tc.test_case_id)}">Generate Playwright Test</button>
      </td>
    </tr>`
    )
    .join("");
}

document.getElementById("generate-test-cases-btn").addEventListener("click", async () => {
  const button = document.getElementById("generate-test-cases-btn");
  if (state.requirementId === null) {
    showToast("Analyze a requirement first.");
    return;
  }
  setButtonLoading(button, "Generating...");
  try {
    const result = await apiRequest("POST", "/api/v1/test-cases", { requirement_id: state.requirementId });
    state.testCases = {};
    result.test_cases.forEach((tc) => {
      state.testCases[tc.test_case_id] = tc;
    });
    renderTestCasesTable(result.test_cases);
    showSection("test-cases-section");
    loadStats();
  } catch (error) {
    showToast(error.message);
  } finally {
    clearButtonLoading(button);
  }
});

document.getElementById("test-cases-tbody").addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  const testCaseId = button.dataset.testCaseId;

  if (button.dataset.action === "view-details") {
    showTestCaseDetails(state.testCases[testCaseId]);
  } else if (button.dataset.action === "generate-test") {
    await handleGeneratePlaywrightTest(testCaseId, button);
  }
});

function showTestCaseDetails(testCase) {
  if (!testCase) {
    showToast("Test case details are not available.");
    return;
  }
  document.getElementById("test-case-modal-content").innerHTML = [
    detailItem("Test Case ID", escapeHtml(testCase.test_case_id)),
    detailItem("Priority", priorityBadge(testCase.priority)),
    detailItem("Type", escapeHtml(testCase.type)),
    detailItem("Description", escapeHtml(testCase.description), true),
    detailItem("Preconditions", listHtml(testCase.preconditions), true),
    detailItem("Steps", listHtml(testCase.steps), true),
    detailItem(
      "Test Data",
      `<pre class="error-pre">${escapeHtml(JSON.stringify(testCase.test_data, null, 2))}</pre>`,
      true
    ),
    detailItem("Expected Result", escapeHtml(testCase.expected_result), true),
  ].join("");
  openModal("test-case-modal");
}

// --- 4. Generated Tests ---

function renderGeneratedTestsTable() {
  const tbody = document.getElementById("generated-tests-tbody");
  const entries = Object.entries(state.generatedTests);
  if (entries.length === 0) {
    tbody.innerHTML = '<tr><td colspan="4" class="empty-row">No generated tests yet.</td></tr>';
    return;
  }
  tbody.innerHTML = entries
    .map(
      ([testCaseId, data]) => `
    <tr>
      <td>${escapeHtml(testCaseId)}</td>
      <td><code>${escapeHtml(data.generated_file)}</code></td>
      <td>${
        data.validation.valid
          ? '<span class="badge badge--passed">VALID</span>'
          : '<span class="badge badge--failed">INVALID</span>'
      }</td>
      <td class="row-actions">
        <button class="btn btn--primary btn--small" type="button" data-action="run-test" data-test-case-id="${escapeHtml(testCaseId)}">Run Test</button>
      </td>
    </tr>`
    )
    .join("");
}

async function handleGeneratePlaywrightTest(testCaseId, button) {
  setButtonLoading(button, "Generating...");
  try {
    const result = await apiRequest("POST", "/api/v1/tests/generate", { test_case_id: testCaseId });
    state.generatedTests[testCaseId] = result;
    renderGeneratedTestsTable();
    showSection("generated-tests-section");
  } catch (error) {
    showToast(error.message);
  } finally {
    clearButtonLoading(button);
  }
}

document.getElementById("generated-tests-tbody").addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-action='run-test']");
  if (!button) return;
  await handleRunTest(button.dataset.testCaseId, button);
});

// --- 5. Execution Results ---

function renderExecutionResult(result) {
  const parts = [
    detailItem("Test Case", escapeHtml(result.test_case_id)),
    detailItem("Execution ID", result.execution_id !== null ? escapeHtml(String(result.execution_id)) : "–"),
    detailItem("Status", statusBadgeHtml(result.status, result.healed)),
    detailItem("Duration", `${result.duration.toFixed(2)}s`),
    detailItem("Healed", result.healed ? "Yes" : "No"),
  ];
  if (result.screenshot) {
    parts.push(detailItem("Screenshot", `<code>${escapeHtml(result.screenshot)}</code>`, true));
  }
  if (result.error) {
    parts.push(detailItem("Error", `<pre class="error-pre">${escapeHtml(result.error)}</pre>`, true));
  }
  document.getElementById("execution-content").innerHTML = parts.join("");

  const analyzeBtn = document.getElementById("analyze-failure-btn");
  analyzeBtn.hidden = !(result.status === "failed" || result.status === "error");
}

async function handleRunTest(testCaseId, button) {
  setButtonLoading(button, "Running test...");
  try {
    const result = await apiRequest("POST", "/api/v1/tests/run", { test_case_id: testCaseId });
    state.lastExecution = result;
    state.lastFailureAnalysis = null;
    state.lastHealingResult = null;
    renderExecutionResult(result);
    showSection("execution-section");
    document.getElementById("failure-analysis-section").hidden = true;
    document.getElementById("healing-section").hidden = true;
    loadStats();
  } catch (error) {
    showToast(error.message);
  } finally {
    clearButtonLoading(button);
  }
}

// --- 6. AI Failure Analysis ---

function renderFailureAnalysis(analysis) {
  document.getElementById("failure-analysis-content").innerHTML = [
    detailItem("Failure Type", `<span class="badge badge--failed">${escapeHtml(analysis.failure_type)}</span>`),
    detailItem("Confidence", `${Math.round(analysis.confidence * 100)}%`),
    detailItem("Test Issue", analysis.is_likely_test_issue ? "Yes" : "No"),
    detailItem("Environment Issue", analysis.is_likely_environment_issue ? "Yes" : "No"),
    detailItem("Summary", escapeHtml(analysis.summary), true),
    detailItem("Root Cause", escapeHtml(analysis.root_cause), true),
    detailItem("Suggested Fix", escapeHtml(analysis.suggested_fix), true),
  ].join("");
}

document.getElementById("analyze-failure-btn").addEventListener("click", async () => {
  const button = document.getElementById("analyze-failure-btn");
  if (!state.lastExecution || state.lastExecution.execution_id === null) {
    showToast("No execution id available to analyze.");
    return;
  }
  setButtonLoading(button, "Analyzing failure...");
  try {
    const analysis = await apiRequest("POST", "/api/v1/tests/analyze-failure", {
      execution_id: state.lastExecution.execution_id,
    });
    state.lastFailureAnalysis = analysis;
    renderFailureAnalysis(analysis);
    showSection("failure-analysis-section");
  } catch (error) {
    showToast(error.message);
  } finally {
    clearButtonLoading(button);
  }
});

document.getElementById("view-full-analysis-btn").addEventListener("click", () => {
  if (!state.lastFailureAnalysis) {
    showToast("No failure analysis available yet.");
    return;
  }
  const analysis = state.lastFailureAnalysis;
  document.getElementById("failure-analysis-modal-content").innerHTML = [
    detailItem("Execution ID", escapeHtml(String(analysis.execution_id))),
    detailItem("Failure Type", escapeHtml(analysis.failure_type)),
    detailItem("Confidence", `${Math.round(analysis.confidence * 100)}%`),
    detailItem("Likely Test Issue", analysis.is_likely_test_issue ? "Yes" : "No"),
    detailItem("Likely Environment Issue", analysis.is_likely_environment_issue ? "Yes" : "No"),
    detailItem("Summary", escapeHtml(analysis.summary), true),
    detailItem("Root Cause", escapeHtml(analysis.root_cause), true),
    detailItem("Suggested Fix", escapeHtml(analysis.suggested_fix), true),
  ].join("");
  openModal("failure-analysis-modal");
});

// --- 7. Self-Healing ---

function renderHealingResult(result) {
  document.getElementById("healing-content").innerHTML = [
    detailItem("Result", healingStatusBadge(result.status)),
    detailItem("Original Selector", `<code>${escapeHtml(result.original_selector)}</code>`),
    detailItem(
      "Selected Selector",
      result.selected_selector ? `<code>${escapeHtml(result.selected_selector)}</code>` : "–"
    ),
    detailItem(
      "Retry Succeeded",
      result.retry_succeeded === null ? "–" : result.retry_succeeded ? "Yes" : "No"
    ),
    detailItem("Confidence", result.confidence !== null ? `${Math.round(result.confidence * 100)}%` : "–"),
    detailItem("Candidate Selectors", listHtml(result.candidate_selectors), true),
    detailItem("Reason", escapeHtml(result.reason), true),
  ].join("");
}

document.getElementById("heal-selector-btn").addEventListener("click", async () => {
  const button = document.getElementById("heal-selector-btn");
  if (!state.lastExecution || state.lastExecution.execution_id === null) {
    showToast("No execution id available to heal.");
    return;
  }
  setButtonLoading(button, "Healing selector...");
  try {
    const result = await apiRequest("POST", "/api/v1/tests/heal", {
      execution_id: state.lastExecution.execution_id,
    });
    state.lastHealingResult = result;
    renderHealingResult(result);
    showSection("healing-section");
  } catch (error) {
    showToast(error.message);
  } finally {
    clearButtonLoading(button);
  }
});

// --- 8. Execution History ---

async function loadHistoryRequirements() {
  try {
    const requirements = await apiRequest("GET", "/api/v1/requirements");
    const select = document.getElementById("history-requirement-select");
    select.innerHTML =
      '<option value="">Select a requirement…</option>' +
      requirements
        .map((r) => `<option value="${r.id}">#${r.id} — ${escapeHtml(truncate(r.requirement_text, 60))}</option>`)
        .join("");
  } catch (error) {
    showToast(error.message);
  }
}

function clearHistoryTable() {
  historyExecutions = [];
  renderHistoryTable();
}

function renderHistoryTable() {
  const tbody = document.getElementById("history-tbody");
  let rows = historyExecutions;
  if (historyFilter === "passed") {
    rows = rows.filter((e) => e.status === "passed" && !e.healed);
  } else if (historyFilter === "failed") {
    rows = rows.filter((e) => e.status === "failed" || e.status === "error");
  } else if (historyFilter === "healed") {
    rows = rows.filter((e) => e.healed);
  }

  if (rows.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty-row">No executions to show.</td></tr>';
    return;
  }

  tbody.innerHTML = rows
    .map(
      (e) => `
    <tr>
      <td>${escapeHtml(e.test_case_id)}</td>
      <td>${statusBadgeHtml(e.status, e.healed)}</td>
      <td>${e.duration.toFixed(2)}s</td>
      <td>${e.healed ? "Yes" : "No"}</td>
      <td>${escapeHtml(e.created_at)}</td>
    </tr>`
    )
    .join("");
}

document.getElementById("history-requirement-select").addEventListener("change", async (event) => {
  const requirementId = event.target.value;
  const testCaseSelect = document.getElementById("history-test-case-select");
  testCaseSelect.innerHTML = '<option value="">Select a test case…</option>';
  testCaseSelect.disabled = true;
  clearHistoryTable();

  if (!requirementId) return;

  try {
    const testCases = await apiRequest("GET", `/api/v1/requirements/${requirementId}/test-cases`);
    testCaseSelect.innerHTML =
      '<option value="">Select a test case…</option>' +
      testCases
        .map((tc) => `<option value="${escapeHtml(tc.test_case_id)}">${escapeHtml(tc.test_case_id)} — ${escapeHtml(tc.title)}</option>`)
        .join("");
    testCaseSelect.disabled = testCases.length === 0;
  } catch (error) {
    showToast(error.message);
  }
});

document.getElementById("history-test-case-select").addEventListener("change", async (event) => {
  const testCaseId = event.target.value;
  if (!testCaseId) {
    clearHistoryTable();
    return;
  }
  try {
    historyExecutions = await apiRequest("GET", `/api/v1/test-cases/${encodeURIComponent(testCaseId)}/executions`);
    renderHistoryTable();
  } catch (error) {
    showToast(error.message);
  }
});

document.querySelectorAll(".filter-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".filter-btn").forEach((b) => b.classList.remove("is-active"));
    btn.classList.add("is-active");
    historyFilter = btn.dataset.filter;
    renderHistoryTable();
  });
});

// --- Modals: shared close behavior ---

document.querySelectorAll("[data-close-modal]").forEach((btn) => {
  btn.addEventListener("click", () => closeModal(btn.dataset.closeModal));
});

document.querySelectorAll(".modal-overlay").forEach((overlay) => {
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) overlay.hidden = true;
  });
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    document.querySelectorAll(".modal-overlay").forEach((overlay) => {
      overlay.hidden = true;
    });
  }
});

// --- Refresh + initial load ---

document.getElementById("refresh-stats-btn").addEventListener("click", () => {
  checkHealth();
  loadStats();
});

checkHealth();
loadStats();
loadHistoryRequirements();
