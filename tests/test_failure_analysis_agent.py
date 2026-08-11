"""Unit tests for the Failure Analysis Agent.

The agent is driven through a fake LLMClient, so these tests never contact
a real LLM provider and pass with or without a configured LLM_API_KEY.
"""

import pytest

from agents.failure_analysis_agent import FailureAnalysisAgent, FailureAnalysisError
from core.llm_client import LLMClient
from core.models import ExecutionStatus, FailureAnalysisInput, FailureType

SELECTOR_FAILURE_JSON = """{
    "failure_type": "selector_not_found",
    "summary": "Login button was not found.",
    "root_cause": "The selector no longer matches the DOM.",
    "suggested_fix": "Use a stable selector.",
    "confidence": 0.92,
    "is_likely_environment_issue": false,
    "is_likely_test_issue": true
}"""

TIMEOUT_FAILURE_JSON = """{
    "failure_type": "timeout",
    "summary": "Waiting for the dashboard element timed out.",
    "root_cause": "The page took longer than the configured timeout to load.",
    "suggested_fix": "Increase the timeout or wait for a more specific ready signal.",
    "confidence": 0.7,
    "is_likely_environment_issue": true,
    "is_likely_test_issue": false
}"""

ASSERTION_FAILURE_JSON = """{
    "failure_type": "assertion_failure",
    "summary": "Expected the dashboard to be visible, but it was not.",
    "root_cause": "The assertion condition was not met after the login attempt.",
    "suggested_fix": "Verify the login step actually succeeded before asserting.",
    "confidence": 0.8,
    "is_likely_environment_issue": false,
    "is_likely_test_issue": true
}"""


class FakeLLMClient(LLMClient):
    """A stand-in LLMClient that returns a canned response instead of calling a provider."""

    def __init__(self, response: str | None = None, error: Exception | None = None) -> None:
        self._response = response
        self._error = error
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


def make_agent(response: str | None = None, error: Exception | None = None) -> tuple:
    fake_client = FakeLLMClient(response=response, error=error)
    agent = FailureAnalysisAgent(llm_client=fake_client)
    return agent, fake_client


def make_failed_input(**overrides: object) -> FailureAnalysisInput:
    payload = {
        "test_id": "test_tc_login_001",
        "status": ExecutionStatus.FAILED,
        "duration": 12.5,
        "stdout": "1 failed in 12.50s",
        "stderr": "",
        "error": "TimeoutError: Page.fill: Timeout 30000ms exceeded waiting for locator(\"#username\")",
    }
    payload.update(overrides)
    return FailureAnalysisInput(**payload)


# --- A. Selector failure ---


def test_analyze_returns_valid_failure_analysis_for_selector_failure():
    agent, fake_client = make_agent(response=SELECTOR_FAILURE_JSON)

    analysis = agent.analyze(make_failed_input())

    assert analysis.failure_type == FailureType.SELECTOR_NOT_FOUND
    assert analysis.confidence == 0.92
    assert analysis.is_likely_test_issue is True
    assert "test_tc_login_001" in fake_client.last_prompt


# --- B. Timeout failure ---


def test_analyze_returns_timeout_failure_type():
    agent, _ = make_agent(response=TIMEOUT_FAILURE_JSON)

    analysis = agent.analyze(make_failed_input())

    assert analysis.failure_type == FailureType.TIMEOUT


# --- C. Assertion failure ---


def test_analyze_returns_assertion_failure_type():
    agent, _ = make_agent(response=ASSERTION_FAILURE_JSON)

    analysis = agent.analyze(make_failed_input())

    assert analysis.failure_type == FailureType.ASSERTION_FAILURE


# --- D. Successful execution result -> no analysis attempted ---


def test_analyze_raises_for_passed_execution_result():
    agent, fake_client = make_agent(response=SELECTOR_FAILURE_JSON)
    passed_input = make_failed_input(status=ExecutionStatus.PASSED, error=None)

    with pytest.raises(FailureAnalysisError):
        agent.analyze(passed_input)

    assert fake_client.last_prompt is None  # the LLM must never even be called


def test_analyze_raises_for_skipped_execution_result():
    agent, fake_client = make_agent(response=SELECTOR_FAILURE_JSON)
    skipped_input = make_failed_input(status=ExecutionStatus.SKIPPED, error=None)

    with pytest.raises(FailureAnalysisError):
        agent.analyze(skipped_input)

    assert fake_client.last_prompt is None


# --- E. Invalid JSON -> graceful error ---


def test_analyze_raises_on_malformed_json():
    agent, _ = make_agent(response="{failure_type: 'missing quotes'}")

    with pytest.raises(FailureAnalysisError):
        agent.analyze(make_failed_input())


def test_analyze_raises_when_response_has_no_json():
    agent, _ = make_agent(response="I think the selector was wrong, but I'm not sure.")

    with pytest.raises(FailureAnalysisError):
        agent.analyze(make_failed_input())


# --- F. Missing fields -> Pydantic validation error (wrapped) ---


def test_analyze_raises_when_required_field_is_missing():
    incomplete = '{"failure_type": "timeout", "summary": "Something timed out."}'
    agent, _ = make_agent(response=incomplete)

    with pytest.raises(FailureAnalysisError):
        agent.analyze(make_failed_input())


# --- G. Invalid confidence -> validation failure ---


def test_analyze_raises_when_confidence_out_of_range():
    bad_confidence = SELECTOR_FAILURE_JSON.replace('"confidence": 0.92', '"confidence": 1.5')
    agent, _ = make_agent(response=bad_confidence)

    with pytest.raises(FailureAnalysisError):
        agent.analyze(make_failed_input())


# --- H. Secret redaction ---


def test_analyze_redacts_secrets_before_sending_to_llm():
    agent, fake_client = make_agent(response=SELECTOR_FAILURE_JSON)
    leaky_input = make_failed_input(stdout="password=SuperSecret123\nlogin attempted")

    agent.analyze(leaky_input)

    assert "password=[REDACTED]" in fake_client.last_prompt
    assert "SuperSecret123" not in fake_client.last_prompt


def test_analyze_redacts_secrets_in_dom_snippet_and_traceback():
    agent, fake_client = make_agent(response=SELECTOR_FAILURE_JSON)
    leaky_input = make_failed_input(
        dom_snippet='<input name="api_key" value="sk-abcdEFGH12345678ijklMNOP" />',
        traceback="Authorization: Bearer abc123.def456.ghi789",
    )

    agent.analyze(leaky_input)

    assert "sk-abcdEFGH12345678ijklMNOP" not in fake_client.last_prompt
    assert "abc123.def456.ghi789" not in fake_client.last_prompt
    assert "[REDACTED]" in fake_client.last_prompt


# --- I. Huge HTML input -> truncated before sending to the LLM ---


def test_analyze_truncates_huge_dom_snippet_before_sending_to_llm():
    agent, fake_client = make_agent(response=SELECTOR_FAILURE_JSON)
    huge_dom = "<div>" + ("x" * 50_000) + "</div>"
    huge_input = make_failed_input(dom_snippet=huge_dom)

    agent.analyze(huge_input)

    assert huge_dom not in fake_client.last_prompt
    assert "...[truncated" in fake_client.last_prompt
    assert len(fake_client.last_prompt) < len(huge_dom)


# --- Input validation / wiring ---


def test_analyze_rejects_non_failure_analysis_input():
    agent, fake_client = make_agent(response=SELECTOR_FAILURE_JSON)

    with pytest.raises(FailureAnalysisError):
        agent.analyze(None)

    assert fake_client.last_prompt is None


def test_from_execution_result_builds_valid_input():
    from core.models import TestExecutionResult

    execution_result = TestExecutionResult(
        test_id="test_tc_login_001",
        status=ExecutionStatus.FAILED,
        duration=5.0,
        error="TimeoutError",
    )

    analysis_input = FailureAnalysisInput.from_execution_result(
        execution_result, failed_selector="#username", url="https://example.com/login"
    )

    assert analysis_input.test_id == "test_tc_login_001"
    assert analysis_input.status == ExecutionStatus.FAILED
    assert analysis_input.failed_selector == "#username"
