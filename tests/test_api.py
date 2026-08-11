"""Unit and integration tests for the FastAPI layer.

Every agent/executor dependency is overridden via
`app.dependency_overrides`, so these tests never contact a real LLM
provider or depend on an external website. The one full-workflow
integration test uses REAL agents/generator/runner wired to a fake
LLMClient (not fully-fake agent objects), so it genuinely exercises JSON
extraction, Pydantic validation, code validation, and persistence — while
staying deterministic and offline. The database is isolated by the
autouse `_isolated_database` fixture in conftest.py; the generated-test
workspace is isolated explicitly here (see `isolated_workspace`), since
that's a filesystem path, not a database path.
"""

from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agents.requirement_agent import RequirementAgent
from agents.test_generation_agent import TestCaseGeneratorAgent
from api.dependencies import (
    get_failure_analysis_agent,
    get_healing_executor,
    get_page_context,
    get_playwright_code_generator,
    get_requirement_agent,
    get_test_case_generator_agent,
    get_test_runner,
)
from app import app
from core.config import PROJECT_ROOT
from core.llm_client import LLMClient
from core.models import (
    ExecutionStatus,
    FailureAnalysis,
    HealingResult,
    HealingStatus,
    RequirementAnalysis,
    TestCase,
    TestExecutionResult,
)
from core.repositories import (
    ExecutionRepository,
    FailureAnalysisRepository,
    HealingRepository,
    RequirementRepository,
    TestCaseRepository,
)
from generators.playwright_generator import PlaywrightCodeGenerator


class FakeLLMClient(LLMClient):
    """A stand-in LLMClient that returns a canned response instead of calling a provider."""

    def __init__(self, response: str) -> None:
        self._response = response

    def generate(self, prompt: str) -> str:
        return self._response


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


def make_failure_analysis(**overrides: object) -> FailureAnalysis:
    payload = {
        "failure_type": "selector_not_found",
        "summary": "Login button was not found.",
        "root_cause": "The selector no longer matches the DOM.",
        "suggested_fix": "Use a stable selector.",
        "confidence": 0.92,
        "is_likely_environment_issue": False,
        "is_likely_test_issue": True,
    }
    payload.update(overrides)
    return FailureAnalysis(**payload)


@contextmanager
def _fake_page_context(url: str | None):
    yield object()  # never dereferenced when the healing executor itself is faked


@pytest.fixture(autouse=True)
def _clear_dependency_overrides():
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def isolated_workspace(tmp_path: Path, monkeypatch) -> Path:
    """Redirect the generated-test workspace (and reports/screenshots) to a tmp dir.

    This is a *filesystem* path, separate from the database isolation the
    autouse `_isolated_database` fixture already provides — both are
    needed for the full-workflow test, which really saves and runs a file.
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
    return workspace


# --- A. GET /health ---


def test_health_returns_ok(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- B. Invalid requirement ---


def test_create_requirement_rejects_blank_requirement(client: TestClient):
    response = client.post("/api/v1/requirements", json={"requirement": ""})
    assert response.status_code == 422  # request-shape validation (min_length=1)


def test_create_requirement_rejects_whitespace_only_requirement(client: TestClient):
    response = client.post("/api/v1/requirements", json={"requirement": "   "})
    assert response.status_code == 400  # passes shape validation, fails domain validation


# --- C. Requirement endpoint with mocked Requirement Agent ---


class FakeRequirementAgent:
    def __init__(self, analysis: RequirementAnalysis, requirement_id: int | None) -> None:
        self._analysis = analysis
        self.last_requirement_id = requirement_id

    def analyze(self, requirement) -> RequirementAnalysis:
        return self._analysis


def test_create_requirement_returns_201_with_analysis(client: TestClient):
    app.dependency_overrides[get_requirement_agent] = lambda: FakeRequirementAgent(
        analysis=make_analysis(), requirement_id=1
    )

    response = client.post(
        "/api/v1/requirements",
        json={"requirement": "User should be able to login with valid credentials."},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["requirement_id"] == 1
    assert body["analysis"]["test_name"] == "Valid Login Test"


# --- D. Test-case generation endpoint with mocked generator ---


class FakeTestCaseGeneratorAgent:
    def __init__(self, test_cases: list[TestCase]) -> None:
        self._test_cases = test_cases

    def generate(self, requirement_analysis, requirement_id=None) -> list[TestCase]:
        return self._test_cases


def test_generate_test_cases_returns_201(client: TestClient):
    requirement_id = RequirementRepository().save("User should be able to login.", make_analysis())
    app.dependency_overrides[get_test_case_generator_agent] = lambda: FakeTestCaseGeneratorAgent(
        [make_test_case()]
    )

    response = client.post("/api/v1/test-cases", json={"requirement_id": requirement_id})

    assert response.status_code == 201
    body = response.json()
    assert body["requirement_id"] == requirement_id
    assert body["test_cases"][0]["test_case_id"] == "TC_LOGIN_001"


# --- E. Test generation endpoint ---


class FakePlaywrightCodeGenerator:
    def __init__(self, code: str, saved_path: Path) -> None:
        self._code = code
        self._saved_path = saved_path

    def generate_code(self, test_case: TestCase) -> str:
        return self._code

    def save_code(self, test_case: TestCase, code: str) -> Path:
        return self._saved_path


def test_generate_playwright_test_returns_201(client: TestClient):
    TestCaseRepository().save(make_test_case())
    saved_path = PROJECT_ROOT / "tests" / "generated" / "test_tc_login_001.py"
    app.dependency_overrides[get_playwright_code_generator] = lambda: FakePlaywrightCodeGenerator(
        code="def test_valid_login():\n    assert True\n", saved_path=saved_path
    )

    response = client.post("/api/v1/tests/generate", json={"test_case_id": "TC_LOGIN_001"})

    assert response.status_code == 201
    body = response.json()
    assert body["test_case_id"] == "TC_LOGIN_001"
    assert body["generated_file"] == "tests/generated/test_tc_login_001.py"
    assert body["validation"]["valid"] is True


# --- F. Run test endpoint with mocked TestRunner ---


class FakeTestRunner:
    def __init__(self, result: TestExecutionResult, execution_id: int | None) -> None:
        self._result = result
        self.last_execution_id = execution_id

    def run_test(self, test_file: Path) -> TestExecutionResult:
        return self._result


def test_run_test_returns_execution_result(client: TestClient):
    TestCaseRepository().save(make_test_case())
    fake_result = TestExecutionResult(test_id="test_tc_login_001", status=ExecutionStatus.PASSED, duration=1.2)
    app.dependency_overrides[get_test_runner] = lambda: FakeTestRunner(fake_result, execution_id=42)

    response = client.post("/api/v1/tests/run", json={"test_case_id": "TC_LOGIN_001"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "passed"
    assert body["execution_id"] == 42
    assert body["healed"] is False


# --- G. Missing requirement -> 404 ---


def test_generate_test_cases_with_missing_requirement_returns_404(client: TestClient):
    response = client.post("/api/v1/test-cases", json={"requirement_id": 9999})
    assert response.status_code == 404


# --- H. Missing test case -> 404 ---


def test_generate_playwright_test_with_missing_test_case_returns_404(client: TestClient):
    response = client.post("/api/v1/tests/generate", json={"test_case_id": "TC_DOES_NOT_EXIST"})
    assert response.status_code == 404


def test_run_test_with_missing_test_case_returns_404(client: TestClient):
    response = client.post("/api/v1/tests/run", json={"test_case_id": "TC_DOES_NOT_EXIST"})
    assert response.status_code == 404


# --- I. Failure analysis endpoint with mocked FailureAnalysisAgent ---


class FakeFailureAnalysisAgent:
    def __init__(self, analysis: FailureAnalysis) -> None:
        self._analysis = analysis

    def analyze(self, analysis_input, execution_id=None) -> FailureAnalysis:
        return self._analysis


def test_analyze_failure_returns_201(client: TestClient):
    execution_id = ExecutionRepository().save(
        TestExecutionResult(test_id="test_tc_login_001", status=ExecutionStatus.FAILED, duration=5.0, error="boom")
    )
    app.dependency_overrides[get_failure_analysis_agent] = lambda: FakeFailureAnalysisAgent(
        make_failure_analysis()
    )

    response = client.post("/api/v1/tests/analyze-failure", json={"execution_id": execution_id})

    assert response.status_code == 201
    assert response.json()["failure_type"] == "selector_not_found"


def test_analyze_failure_on_passed_execution_returns_400(client: TestClient):
    execution_id = ExecutionRepository().save(
        TestExecutionResult(test_id="t", status=ExecutionStatus.PASSED, duration=1.0)
    )

    response = client.post("/api/v1/tests/analyze-failure", json={"execution_id": execution_id})

    assert response.status_code == 400


def test_analyze_failure_on_missing_execution_returns_404(client: TestClient):
    response = client.post("/api/v1/tests/analyze-failure", json={"execution_id": 9999})
    assert response.status_code == 404


# --- J. Healing endpoint with mocked healing executor ---


class FakeHealingExecutor:
    def __init__(self, result: HealingResult) -> None:
        self._result = result
        self.calls: list[dict] = []

    def attempt_heal(self, **kwargs) -> HealingResult:
        self.calls.append(kwargs)
        return self._result


def test_heal_selector_returns_201(client: TestClient):
    execution_id = ExecutionRepository().save(
        TestExecutionResult(
            test_id="test_tc_login_001",
            status=ExecutionStatus.FAILED,
            duration=5.0,
            error='TimeoutError: waiting for locator("#username")',
        )
    )
    FailureAnalysisRepository().save(execution_id, make_failure_analysis())

    fake_result = HealingResult(
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
    app.dependency_overrides[get_healing_executor] = lambda: FakeHealingExecutor(fake_result)
    app.dependency_overrides[get_page_context] = lambda: _fake_page_context

    response = client.post("/api/v1/tests/heal", json={"execution_id": execution_id})

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "healed"
    assert body["selected_selector"] == "[data-testid='username']"


def test_heal_selector_without_failure_analysis_returns_400(client: TestClient):
    execution_id = ExecutionRepository().save(
        TestExecutionResult(
            test_id="t", status=ExecutionStatus.FAILED, duration=1.0, error='waiting for locator("#x")'
        )
    )

    response = client.post("/api/v1/tests/heal", json={"execution_id": execution_id})

    assert response.status_code == 400


def test_heal_selector_when_selector_cannot_be_extracted_returns_400(client: TestClient):
    execution_id = ExecutionRepository().save(
        TestExecutionResult(
            test_id="t", status=ExecutionStatus.FAILED, duration=1.0, error="a generic failure, no locator info"
        )
    )
    FailureAnalysisRepository().save(execution_id, make_failure_analysis())

    response = client.post("/api/v1/tests/heal", json={"execution_id": execution_id})

    assert response.status_code == 400


def test_heal_selector_on_missing_execution_returns_404(client: TestClient):
    response = client.post("/api/v1/tests/heal", json={"execution_id": 9999})
    assert response.status_code == 404


# --- K. History endpoints ---


def test_list_requirements(client: TestClient):
    repo = RequirementRepository()
    repo.save("Requirement one")
    repo.save("Requirement two")

    response = client.get("/api/v1/requirements")

    assert response.status_code == 200
    body = response.json()
    assert [r["requirement_text"] for r in body] == ["Requirement one", "Requirement two"]


def test_get_test_cases_for_requirement(client: TestClient):
    requirement_id = RequirementRepository().save("Some requirement", make_analysis())
    TestCaseRepository().save(make_test_case(), requirement_id=requirement_id)

    response = client.get(f"/api/v1/requirements/{requirement_id}/test-cases")

    assert response.status_code == 200
    assert response.json()[0]["test_case_id"] == "TC_LOGIN_001"


def test_get_test_cases_for_missing_requirement_returns_404(client: TestClient):
    response = client.get("/api/v1/requirements/9999/test-cases")
    assert response.status_code == 404


def test_get_execution_history_for_test_case(client: TestClient):
    TestCaseRepository().save(make_test_case())
    ExecutionRepository().save(
        TestExecutionResult(test_id="test_tc_login_001", status=ExecutionStatus.PASSED, duration=1.0)
    )

    response = client.get("/api/v1/test-cases/TC_LOGIN_001/executions")

    assert response.status_code == 200
    assert len(response.json()) == 1


def test_get_execution_by_id(client: TestClient):
    execution_id = ExecutionRepository().save(
        TestExecutionResult(test_id="t", status=ExecutionStatus.PASSED, duration=1.0)
    )

    response = client.get(f"/api/v1/executions/{execution_id}")

    assert response.status_code == 200
    assert response.json()["id"] == execution_id


def test_get_execution_by_id_missing_returns_404(client: TestClient):
    response = client.get("/api/v1/executions/9999")
    assert response.status_code == 404


def test_get_failure_analysis_history(client: TestClient):
    execution_id = ExecutionRepository().save(
        TestExecutionResult(test_id="t", status=ExecutionStatus.FAILED, duration=1.0)
    )
    FailureAnalysisRepository().save(execution_id, make_failure_analysis())

    response = client.get(f"/api/v1/executions/{execution_id}/failure-analysis")

    assert response.status_code == 200
    assert len(response.json()) == 1


def test_get_healing_history_for_execution(client: TestClient):
    execution_id = ExecutionRepository().save(
        TestExecutionResult(test_id="t", status=ExecutionStatus.FAILED, duration=1.0)
    )
    HealingRepository().save(
        execution_id,
        HealingResult(
            test_id="t",
            original_selector="#x",
            status=HealingStatus.FAILED,
            reason="no valid candidate",
            timestamp="2026-01-01T00:00:00+00:00",
        ),
    )

    response = client.get(f"/api/v1/executions/{execution_id}/healing")

    assert response.status_code == 200
    assert response.json()[0]["status"] == "failed"


# --- Integration test: requirement -> test cases -> Playwright test -> run ---


def test_full_workflow_requirement_to_execution(client: TestClient, isolated_workspace: Path):
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

    create_response = client.post(
        "/api/v1/requirements",
        json={"requirement": "User should be able to login with valid credentials and see the dashboard."},
    )
    assert create_response.status_code == 201
    requirement_id = create_response.json()["requirement_id"]

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

    generate_tc_response = client.post("/api/v1/test-cases", json={"requirement_id": requirement_id})
    assert generate_tc_response.status_code == 201
    test_case_id = generate_tc_response.json()["test_cases"][0]["test_case_id"]
    assert test_case_id == "TC_LOGIN_001"

    # A trivial, dependency-free test body (no network, no Playwright import) —
    # keeps this deterministic while still exercising the real generator,
    # AST validator, and subprocess-based runner end to end.
    simple_code = "def test_valid_login():\n    assert True\n"
    app.dependency_overrides[get_playwright_code_generator] = lambda: PlaywrightCodeGenerator(
        llm_client=FakeLLMClient(simple_code)
    )

    generate_test_response = client.post("/api/v1/tests/generate", json={"test_case_id": test_case_id})
    assert generate_test_response.status_code == 201
    assert generate_test_response.json()["validation"]["valid"] is True

    run_response = client.post("/api/v1/tests/run", json={"test_case_id": test_case_id})
    assert run_response.status_code == 200
    body = run_response.json()
    assert body["status"] == "passed"
    assert body["execution_id"] is not None
