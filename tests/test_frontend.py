"""Frontend/API integration tests: the dashboard, driven by a real browser
(Playwright) against a REAL FastAPI server running in-process.

The server runs via `uvicorn.Server` in a background thread, on the SAME
`app` object the rest of the test suite imports — dependency overrides set
via `app.dependency_overrides` before a browser action triggers the
corresponding request apply exactly as in tests/test_api.py. Every
scenario here is deterministic and never contacts a real LLM or an
external website. The database is isolated by the autouse
`_isolated_database` fixture in conftest.py; the generated-test workspace
is isolated explicitly here, the same way tests/test_api.py does it.
"""

import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
import uvicorn
from playwright.sync_api import expect

from agents.failure_analysis_agent import FailureAnalysisAgent
from agents.requirement_agent import RequirementAgent
from agents.test_generation_agent import TestCaseGeneratorAgent
from api.dependencies import (
    get_failure_analysis_agent,
    get_healing_executor,
    get_page_context,
    get_playwright_code_generator,
    get_requirement_agent,
    get_test_case_generator_agent,
)
from app import app
from core.llm_client import LLMClient
from core.models import (
    ExecutionStatus,
    HealingResult,
    HealingStatus,
    RequirementAnalysis,
    TestCase,
    TestExecutionResult,
)
from core.repositories import ExecutionRepository, RequirementRepository, TestCaseRepository
from generators.playwright_generator import PlaywrightCodeGenerator


class FakeLLMClient(LLMClient):
    """A stand-in LLMClient that returns a canned response instead of calling a provider."""

    def __init__(self, response: str) -> None:
        self._response = response

    def generate(self, prompt: str) -> str:
        return self._response

# Must match frontend/app.js's hardcoded API_BASE_URL — the dashboard
# fetches that absolute URL regardless of what page/port served it.
FRONTEND_API_PORT = 8000


def make_analysis(**overrides: object) -> RequirementAnalysis:
    payload = {
        "test_name": "Valid Login Test",
        "description": "Verify that a user can successfully login.",
        "preconditions": ["User has valid credentials"],
        "steps": ["Open login page", "Enter username", "Enter password", "Click login"],
        "expected_result": "Dashboard is displayed",
        "priority": "high",
    }
    payload.update(overrides)
    return RequirementAnalysis(**payload)


def make_test_case(**overrides: object) -> TestCase:
    payload = {
        "test_case_id": "TC_LOGIN_001",
        "title": "Valid Login",
        "description": "Verify that a valid user can log in.",
        "steps": ["Open login page", "Enter valid username", "Enter valid password", "Click Login"],
        "test_data": {"username": "valid_username", "password": "valid_password"},
        "expected_result": "Dashboard is displayed",
        "priority": "high",
        "type": "positive",
    }
    payload.update(overrides)
    return TestCase(**payload)


class FakeHealingExecutor:
    def __init__(self, result: HealingResult) -> None:
        self._result = result

    def attempt_heal(self, **kwargs) -> HealingResult:
        return self._result


@contextmanager
def _fake_page_context(url: str | None):
    yield object()


@pytest.fixture(autouse=True)
def _clear_dependency_overrides():
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def live_server(tmp_path: Path, monkeypatch):
    """Run the real FastAPI app in a background thread and yield its base URL.

    Isolates the generated-test workspace the same way tests/test_api.py
    does (the database is already isolated by the autouse
    `_isolated_database` fixture).

    Binds to the SAME fixed port the dashboard's `API_BASE_URL` expects
    (127.0.0.1:8000) — the frontend calls that absolute URL regardless of
    what page served it, so the test server must actually be there. A
    short retry loop absorbs the brief moment after the previous test's
    server closes its socket before the OS fully releases the port.
    """
    workspace = tmp_path / "generated"
    workspace.mkdir()

    class _StubSettings:
        allowed_workspace_path = workspace
        allowed_workspace_dir = "tests/generated"
        reports_path = tmp_path / "reports"
        screenshots_path = tmp_path / "screenshots"

    stub = _StubSettings()
    monkeypatch.setattr("generators.playwright_generator.get_settings", lambda: stub)
    monkeypatch.setattr("executor.test_runner.get_settings", lambda: stub)
    monkeypatch.setattr("api.routes.get_settings", lambda: stub)

    server = None
    thread = None
    last_error = None
    for _attempt in range(20):
        config = uvicorn.Config(app, host="127.0.0.1", port=FRONTEND_API_PORT, log_level="warning")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        deadline = time.time() + 5
        started_ok = False
        while time.time() < deadline:
            if server.started:
                started_ok = True
                break
            if not thread.is_alive():
                break  # startup failed (e.g. port still in use) — retry
            time.sleep(0.05)

        if started_ok:
            last_error = None
            break

        last_error = RuntimeError("Server thread exited before startup completed (port likely still in use).")
        thread.join(timeout=2)
        time.sleep(0.25)

    if last_error is not None:
        raise last_error

    base_url = f"http://127.0.0.1:{FRONTEND_API_PORT}"
    httpx.get(f"{base_url}/health", timeout=2.0)  # confirm it actually answers

    yield base_url

    server.should_exit = True
    thread.join(timeout=5)


# --- A. Dashboard loads ---


def test_dashboard_loads_and_shows_header(live_server, page):
    page.goto(f"{live_server}/")
    expect(page.locator("h1")).to_have_text("Agentic AI Test Automation")
    expect(page.locator(".subtitle")).to_contain_text("self-healing")


# --- B. Health endpoint works (reflected in the connection indicator) ---


def test_dashboard_shows_backend_connected(live_server, page):
    page.goto(f"{live_server}/")
    expect(page.locator("#connection-text")).to_have_text("Backend connected", timeout=5000)
    expect(page.locator("#connection-dot")).to_have_class(re.compile("is-online"))


# --- C-G. Full workflow: requirement -> test cases -> generate -> run -> analyze -> heal ---


def test_full_dashboard_workflow(live_server, page):
    # Steps C/D use the REAL RequirementAgent/TestCaseGeneratorAgent/
    # PlaywrightCodeGenerator wired to a fake LLMClient (not hand-rolled
    # fake agents) — they genuinely persist to the (isolated) database, so
    # every later id lookup (test-cases, executions) resolves for real,
    # exactly as it would with a real LLM. Only the LLM call itself is faked.
    requirement_json = """{
        "test_name": "Valid Login Test",
        "description": "Verify that a user can successfully login.",
        "preconditions": ["User has valid credentials"],
        "steps": ["Open login page", "Enter username", "Enter password", "Click login"],
        "expected_result": "Dashboard is displayed",
        "priority": "high"
    }"""
    app.dependency_overrides[get_requirement_agent] = lambda: RequirementAgent(
        llm_client=FakeLLMClient(requirement_json)
    )

    page.goto(f"{live_server}/")
    page.fill(
        "#requirement-textarea",
        "User should be able to login with valid credentials and see the dashboard.",
    )
    page.click("#analyze-requirement-btn")

    analysis_section = page.locator("#analysis-section")
    expect(analysis_section).to_be_visible(timeout=5000)
    expect(analysis_section).to_contain_text("Valid Login Test")
    expect(analysis_section).to_contain_text("Dashboard is displayed")

    # --- D. Test case generation (real agent, fake LLM) ---
    test_case_json = """[{
        "test_case_id": "TC_LOGIN_001",
        "title": "Valid Login",
        "description": "Verify that a valid user can log in.",
        "preconditions": ["User has valid credentials"],
        "steps": ["Open login page", "Enter valid username", "Enter valid password", "Click Login"],
        "test_data": {"username": "valid_username", "password": "valid_password"},
        "expected_result": "Dashboard is displayed",
        "priority": "high",
        "type": "positive"
    }]"""
    app.dependency_overrides[get_test_case_generator_agent] = lambda: TestCaseGeneratorAgent(
        llm_client=FakeLLMClient(test_case_json)
    )

    page.click("#generate-test-cases-btn")
    test_cases_section = page.locator("#test-cases-section")
    expect(test_cases_section).to_be_visible(timeout=5000)
    expect(page.locator("#test-cases-tbody")).to_contain_text("TC_LOGIN_001")
    expect(page.locator("#test-cases-tbody")).to_contain_text("Valid Login")

    # Test case details modal (item 12)
    page.click("button[data-action='view-details'][data-test-case-id='TC_LOGIN_001']")
    details_modal = page.locator("#test-case-modal")
    expect(details_modal).to_be_visible()
    expect(details_modal).to_contain_text("Open login page")
    page.click("#test-case-modal button[data-close-modal]")
    expect(details_modal).to_be_hidden()

    # Generate Playwright test (real generator + AST validator, fake LLM).
    # The code deliberately raises with a "locator(...)" message when run,
    # so the real execution genuinely fails and stores extractable selector
    # text for the later Heal step — no Playwright/network dependency needed.
    failing_code = 'def test_valid_login():\n    raise TimeoutError(\'waiting for locator("#username")\')\n'
    app.dependency_overrides[get_playwright_code_generator] = lambda: PlaywrightCodeGenerator(
        llm_client=FakeLLMClient(failing_code)
    )
    page.click("button[data-action='generate-test'][data-test-case-id='TC_LOGIN_001']")
    generated_section = page.locator("#generated-tests-section")
    expect(generated_section).to_be_visible(timeout=5000)
    expect(page.locator("#generated-tests-tbody")).to_contain_text("tests/generated/test_tc_login_001.py")
    expect(page.locator("#generated-tests-tbody")).to_contain_text("VALID")

    # --- E. Execution results (REAL TestRunner — a real subprocess pytest run) ---
    page.click("button[data-action='run-test']")
    execution_section = page.locator("#execution-section")
    expect(execution_section).to_be_visible(timeout=15000)
    expect(execution_section).to_contain_text("FAILED")
    expect(page.locator("#analyze-failure-btn")).to_be_visible()

    # --- F. Failure analysis (real agent, fake LLM) ---
    # Using the REAL agent (not a hand-rolled fake) matters here: it's what
    # actually persists the FailureAnalysisRecord that /tests/heal requires
    # to exist before it will attempt healing — a fake that only returns a
    # canned analysis without saving it would make the next step 400.
    failure_analysis_json = """{
        "failure_type": "selector_not_found",
        "summary": "Login button was not found.",
        "root_cause": "The selector no longer matches the DOM.",
        "suggested_fix": "Use a stable selector.",
        "confidence": 0.92,
        "is_likely_environment_issue": false,
        "is_likely_test_issue": true
    }"""
    app.dependency_overrides[get_failure_analysis_agent] = lambda: FailureAnalysisAgent(
        llm_client=FakeLLMClient(failure_analysis_json)
    )

    page.click("#analyze-failure-btn")
    failure_section = page.locator("#failure-analysis-section")
    expect(failure_section).to_be_visible(timeout=5000)
    expect(failure_section).to_contain_text("selector_not_found")
    expect(failure_section).to_contain_text("Use a stable selector.")

    # Full failure analysis modal (item 13) — execution_id is DB-assigned
    # (real persistence now, not a hardcoded fake), so assert on content
    # that doesn't depend on knowing that exact number ahead of time.
    page.click("#view-full-analysis-btn")
    failure_modal = page.locator("#failure-analysis-modal")
    expect(failure_modal).to_be_visible()
    expect(failure_modal).to_contain_text("Execution ID")
    expect(failure_modal).to_contain_text("Root Cause")
    expect(failure_modal).to_contain_text("Use a stable selector.")
    page.keyboard.press("Escape")
    expect(failure_modal).to_be_hidden()

    # --- G. Healing (mocked executor + page context) ---
    healing_result = HealingResult(
        test_id="test_tc_login_001",
        original_selector="#username",
        candidate_selectors=["[data-testid='username']"],
        selected_selector="[data-testid='username']",
        status=HealingStatus.HEALED,
        retry_succeeded=True,
        confidence=0.8,
        reason="A stable attribute selector was found.",
        timestamp="2026-01-01T00:00:00+00:00",
    )
    app.dependency_overrides[get_healing_executor] = lambda: FakeHealingExecutor(healing_result)
    app.dependency_overrides[get_page_context] = lambda: _fake_page_context

    page.click("#heal-selector-btn")
    healing_section = page.locator("#healing-section")
    expect(healing_section).to_be_visible(timeout=5000)
    expect(healing_section).to_contain_text("HEALED")
    expect(healing_section).to_contain_text("#username")
    expect(healing_section).to_contain_text("[data-testid='username']")


# --- Execution history + filters (real repository data, no mocks needed) ---


def test_history_section_loads_and_filters(live_server, page):
    requirement_id = RequirementRepository().save("Some requirement", make_analysis())
    TestCaseRepository().save(make_test_case(), requirement_id=requirement_id)
    ExecutionRepository().save(
        TestExecutionResult(test_id="test_tc_login_001", status=ExecutionStatus.PASSED, duration=1.1)
    )
    ExecutionRepository().save(
        TestExecutionResult(test_id="test_tc_login_001", status=ExecutionStatus.FAILED, duration=2.2, error="boom")
    )

    page.goto(f"{live_server}/")
    page.select_option("#history-requirement-select", str(requirement_id))
    # <option> elements are never reported "visible" by Playwright even when
    # populated, since they only render inside an open dropdown — wait for
    # them to exist in the DOM instead.
    page.wait_for_selector("#history-test-case-select option[value='TC_LOGIN_001']", state="attached")
    page.select_option("#history-test-case-select", "TC_LOGIN_001")

    history_tbody = page.locator("#history-tbody")
    expect(history_tbody).to_contain_text("PASSED", timeout=5000)
    expect(history_tbody).to_contain_text("FAILED")

    page.click("button.filter-btn[data-filter='failed']")
    expect(history_tbody).not_to_contain_text("PASSED")
    expect(history_tbody).to_contain_text("FAILED")

    page.click("button.filter-btn[data-filter='all']")
    expect(history_tbody).to_contain_text("PASSED")
    expect(history_tbody).to_contain_text("FAILED")


# --- Statistics reflect real backend data ---


def test_stats_cards_reflect_backend_data(live_server, page):
    TestCaseRepository().save(make_test_case())
    ExecutionRepository().save(
        TestExecutionResult(test_id="test_tc_login_001", status=ExecutionStatus.PASSED, duration=1.0)
    )

    page.goto(f"{live_server}/")
    expect(page.locator("#stat-total")).to_have_text("1", timeout=5000)
    expect(page.locator("#stat-passed")).to_have_text("1")
    # The backend rounds to 1 decimal (100.0), but JS's default Number->String
    # conversion drops a trailing ".0" for whole numbers — "100%" is correct.
    expect(page.locator("#stat-success-rate")).to_have_text("100%")


# --- Error handling: a backend failure shows a friendly message, never a stack trace ---


def test_requirement_error_shows_friendly_message_not_a_stack_trace(live_server, page):
    class FailingAgent:
        last_requirement_id = None

        def analyze(self, requirement):
            from agents.requirement_agent import RequirementAgentError

            raise RequirementAgentError("simulated LLM failure with internal details")

    app.dependency_overrides[get_requirement_agent] = lambda: FailingAgent()

    page.goto(f"{live_server}/")
    page.fill("#requirement-textarea", "Some requirement text.")
    page.click("#analyze-requirement-btn")

    error_el = page.locator("#requirement-error")
    expect(error_el).to_be_visible(timeout=5000)
    expect(error_el).to_have_text("Failed to analyze the requirement.")
    expect(error_el).not_to_contain_text("Traceback")
    expect(error_el).not_to_contain_text("simulated LLM failure")
