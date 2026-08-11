"""Unit and integration tests for controlled selector self-healing.

Unit tests use a fake LLMClient and a fake, duck-typed Page (just enough to
satisfy `.locator(selector).count()`), so they never contact a real LLM
provider or launch a browser, and pass with or without a configured
LLM_API_KEY. One true integration test (at the bottom) proves the full
pipeline against a REAL Playwright page loaded from an inline HTML string
via `page.set_content(...)` — never an external website.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from agents.healing_agent import SelectorHealingAgent, is_selector_related
from agents.selector_validator import select_best_candidate, validate_selector
from core.llm_client import LLMClient, LLMRequestError
from core.models import (
    ExecutionStatus,
    FailureAnalysis,
    FailureType,
    HealingResult,
    HealingStatus,
    HealingSuggestion,
    SelectorValidation,
    TestExecutionResult,
)
from executor.healing_executor import (
    MAX_HEALING_ATTEMPTS,
    HealingExecutor,
    apply_healing_to_execution_result,
)

VALID_HEALING_JSON = """{
    "original_selector": "#login-button",
    "candidate_selectors": ["button.login", "[data-testid='login']", "button:has-text('Login')"],
    "reason": "Original selector was not found.",
    "confidence": 0.91
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


class _FakeLocator:
    def __init__(self, count: int) -> None:
        self._count = count

    def count(self) -> int:
        return self._count


class _FakePage:
    """Just enough of a Playwright Page to drive selector_validator without a real browser."""

    def __init__(self, match_counts: dict[str, int]) -> None:
        self._match_counts = match_counts

    def locator(self, selector: str) -> _FakeLocator:
        return _FakeLocator(self._match_counts.get(selector, 0))


class _StubSettings:
    """Minimal stand-in for core.config.Settings, pointing reports at a tmp dir."""

    def __init__(self, base: Path) -> None:
        self.reports_path = base


def make_failure(failure_type: FailureType = FailureType.SELECTOR_NOT_FOUND, **overrides) -> FailureAnalysis:
    payload = {
        "failure_type": failure_type,
        "summary": "The login button could not be located.",
        "root_cause": "The selector no longer matches the DOM.",
        "suggested_fix": "Use a more stable selector.",
        "confidence": 0.85,
        "is_likely_environment_issue": False,
        "is_likely_test_issue": True,
    }
    payload.update(overrides)
    return FailureAnalysis(**payload)


def make_executor(
    tmp_path: Path, monkeypatch, *, llm_response: str | None = None, llm_error: Exception | None = None
) -> HealingExecutor:
    monkeypatch.setattr("executor.healing_executor.get_settings", lambda: _StubSettings(tmp_path))
    fake_llm = FakeLLMClient(response=llm_response, error=llm_error)
    agent = SelectorHealingAgent(llm_client=fake_llm)
    return HealingExecutor(healing_agent=agent)


# --- A. Selector-related failure -> healing is attempted ---


def test_is_selector_related_true_for_selector_not_found():
    assert is_selector_related(make_failure(FailureType.SELECTOR_NOT_FOUND)) is True


def test_is_selector_related_true_for_element_not_interactable():
    assert is_selector_related(make_failure(FailureType.ELEMENT_NOT_INTERACTABLE)) is True


def test_healing_executor_attempts_healing_for_selector_related_failure(tmp_path, monkeypatch):
    executor = make_executor(tmp_path, monkeypatch, llm_response=VALID_HEALING_JSON)
    page = _FakePage({"button.login": 1, "[data-testid='login']": 1, "button:has-text('Login')": 1})

    result = executor.attempt_heal(
        test_id="test_tc_login_001",
        page=page,
        original_selector="#login-button",
        failed_action="click",
        retry_action=lambda p, s: None,
        failure=make_failure(FailureType.SELECTOR_NOT_FOUND),
    )

    assert result.status != HealingStatus.SKIPPED


# --- B. Non-selector failure -> healing is skipped ---


@pytest.mark.parametrize(
    "failure_type",
    [
        FailureType.AUTHENTICATION_ERROR,
        FailureType.ENVIRONMENT_ERROR,
        FailureType.NAVIGATION_ERROR,
        FailureType.TEST_DATA_ERROR,
        FailureType.ASSERTION_FAILURE,
        FailureType.TIMEOUT,
        FailureType.UNKNOWN,
    ],
)
def test_is_selector_related_false_for_non_selector_failures(failure_type):
    assert is_selector_related(make_failure(failure_type)) is False


def test_healing_executor_skips_non_selector_failure(tmp_path, monkeypatch):
    executor = make_executor(tmp_path, monkeypatch, llm_response=VALID_HEALING_JSON)
    retry_calls = []

    result = executor.attempt_heal(
        test_id="test_tc_login_001",
        page=_FakePage({}),
        original_selector="#login-button",
        failed_action="click",
        retry_action=lambda p, s: retry_calls.append(s),
        failure=make_failure(FailureType.AUTHENTICATION_ERROR),
    )

    assert result.status == HealingStatus.SKIPPED
    assert retry_calls == []  # the LLM must never even be consulted for this


# --- C. Valid candidate selected ---


def test_validate_selector_accepts_unique_match():
    page = _FakePage({"button.login": 1})
    result = validate_selector(page, "button.login")
    assert result.valid is True
    assert result.match_count == 1


# --- D. Zero-match candidate rejected ---


def test_validate_selector_rejects_zero_match_candidate():
    page = _FakePage({"button.login": 1})
    result = validate_selector(page, "#does-not-exist")
    assert result.valid is False
    assert result.match_count == 0


# --- E. Ambiguous candidate (matches many elements) rejected ---


def test_validate_selector_rejects_ambiguous_candidate():
    page = _FakePage({".item": 5})
    result = validate_selector(page, ".item")
    assert result.valid is False
    assert result.match_count == 5
    assert "ambiguous" in result.reason.lower()


# --- F. Dangerous selector rejected ---


def test_healing_suggestion_drops_dangerous_candidate_but_keeps_safe_ones():
    suggestion = HealingSuggestion(
        original_selector="#login-button",
        candidate_selectors=["javascript:alert(1)", "button.login"],
        reason="test",
        confidence=0.5,
    )
    assert "javascript:alert(1)" not in suggestion.candidate_selectors
    assert suggestion.candidate_selectors == ["button.login"]


def test_healing_suggestion_rejects_when_every_candidate_is_dangerous():
    with pytest.raises(ValidationError):
        HealingSuggestion(
            original_selector="#login-button",
            candidate_selectors=["javascript:alert(1)", "<script>alert(1)</script>"],
            reason="test",
            confidence=0.5,
        )


# --- G. Multiple candidates -> best selected per stability ranking ---


def test_select_best_candidate_prefers_data_testid_over_css_text_and_xpath():
    validations = [
        SelectorValidation(selector="button.login", valid=True, match_count=1),
        SelectorValidation(selector="[data-testid='login']", valid=True, match_count=1),
        SelectorValidation(selector="text=Login", valid=True, match_count=1),
        SelectorValidation(selector="//button[1]", valid=True, match_count=1),
    ]
    best = select_best_candidate(validations)
    assert best.selector == "[data-testid='login']"


def test_select_best_candidate_ignores_invalid_candidates():
    validations = [
        SelectorValidation(selector="[data-testid='login']", valid=False, match_count=0),
        SelectorValidation(selector="button.login", valid=True, match_count=1),
    ]
    best = select_best_candidate(validations)
    assert best.selector == "button.login"


def test_select_best_candidate_returns_none_when_nothing_valid():
    validations = [SelectorValidation(selector="button.login", valid=False, match_count=0)]
    assert select_best_candidate(validations) is None


# --- H. Maximum retry count: exactly one healing attempt ---


def test_healing_executor_retries_exactly_once(tmp_path, monkeypatch):
    assert MAX_HEALING_ATTEMPTS == 1

    executor = make_executor(tmp_path, monkeypatch, llm_response=VALID_HEALING_JSON)
    page = _FakePage({"button.login": 1, "[data-testid='login']": 1, "button:has-text('Login')": 1})
    retry_calls = []

    result = executor.attempt_heal(
        test_id="test_tc_login_001",
        page=page,
        original_selector="#login-button",
        failed_action="click",
        retry_action=lambda p, s: retry_calls.append(s),
        failure=make_failure(FailureType.SELECTOR_NOT_FOUND),
    )

    assert len(retry_calls) == 1
    assert result.status == HealingStatus.HEALED
    assert result.selected_selector == "[data-testid='login']"  # best-ranked of the three valid ones


# --- I. No valid candidates -> healing fails gracefully, original failure preserved ---


def test_healing_executor_fails_gracefully_when_no_candidate_validates(tmp_path, monkeypatch):
    executor = make_executor(tmp_path, monkeypatch, llm_response=VALID_HEALING_JSON)
    page = _FakePage({})  # nothing matches any candidate
    retry_calls = []

    result = executor.attempt_heal(
        test_id="test_tc_login_001",
        page=page,
        original_selector="#login-button",
        failed_action="click",
        retry_action=lambda p, s: retry_calls.append(s),
        failure=make_failure(FailureType.SELECTOR_NOT_FOUND),
    )

    assert result.status == HealingStatus.FAILED
    assert result.selected_selector is None
    assert retry_calls == []


def test_healing_executor_fails_gracefully_when_llm_call_fails(tmp_path, monkeypatch):
    executor = make_executor(tmp_path, monkeypatch, llm_error=LLMRequestError("provider is down"))

    result = executor.attempt_heal(
        test_id="test_tc_login_001",
        page=_FakePage({}),
        original_selector="#login-button",
        failed_action="click",
        retry_action=lambda p, s: None,
        failure=make_failure(FailureType.SELECTOR_NOT_FOUND),
    )

    assert result.status == HealingStatus.FAILED


def test_apply_healing_to_execution_result_preserves_original_on_failure():
    original = TestExecutionResult(test_id="t1", status=ExecutionStatus.FAILED, duration=5.0, error="boom")
    healing = HealingResult(
        test_id="t1",
        original_selector="#login-button",
        status=HealingStatus.FAILED,
        reason="no valid candidate",
        timestamp="2026-01-01T00:00:00+00:00",
    )

    updated = apply_healing_to_execution_result(original, healing)

    assert updated == original
    assert updated.healed is False


def test_apply_healing_to_execution_result_reports_passed_after_healing():
    original = TestExecutionResult(test_id="t1", status=ExecutionStatus.FAILED, duration=5.0, error="boom")
    healing = HealingResult(
        test_id="t1",
        original_selector="#login-button",
        selected_selector="button.login",
        status=HealingStatus.HEALED,
        retry_succeeded=True,
        reason="healed",
        timestamp="2026-01-01T00:00:00+00:00",
    )

    updated = apply_healing_to_execution_result(original, healing)

    assert updated.status == ExecutionStatus.PASSED
    assert updated.healed is True
    assert updated.original_selector == "#login-button"
    assert updated.healed_selector == "button.login"
    assert original.healed is False  # original object must not be mutated


# --- Healing records are persisted under reports/healing/ ---


def test_healing_executor_persists_healing_record_as_json(tmp_path, monkeypatch):
    executor = make_executor(tmp_path, monkeypatch, llm_response=VALID_HEALING_JSON)
    page = _FakePage({"button.login": 1, "[data-testid='login']": 1, "button:has-text('Login')": 1})

    executor.attempt_heal(
        test_id="test_tc_login_001",
        page=page,
        original_selector="#login-button",
        failed_action="click",
        retry_action=lambda p, s: None,
        failure=make_failure(FailureType.SELECTOR_NOT_FOUND),
    )

    records = list((tmp_path / "healing").glob("test_tc_login_001_*.json"))
    assert len(records) == 1


# --- Integration test: real browser, local HTML only, no external website ---


def test_integration_broken_selector_is_healed_on_real_local_page(page, tmp_path, monkeypatch):
    """End-to-end proof against a REAL Playwright page.

    #login-button does not exist on this page; button.login does. The
    healing pipeline must propose candidates, reject #login-button (zero
    matches) via real browser validation, select button.login, retry the
    click for real, and the click's real effect must be observable.
    """
    monkeypatch.setattr("executor.healing_executor.get_settings", lambda: _StubSettings(tmp_path))

    page.set_content(
        """
        <html>
          <body>
            <button class="login" onclick="document.getElementById('result').textContent='clicked'">
              Login
            </button>
            <div id="result"></div>
          </body>
        </html>
        """
    )

    healing_json = """{
        "original_selector": "#login-button",
        "candidate_selectors": ["#login-button", "button.login"],
        "reason": "The id-based selector is absent; a class-based selector matches the visible login button.",
        "confidence": 0.8
    }"""
    executor = HealingExecutor(healing_agent=SelectorHealingAgent(llm_client=FakeLLMClient(response=healing_json)))

    result = executor.attempt_heal(
        test_id="test_integration_login",
        page=page,
        original_selector="#login-button",
        failed_action="click",
        retry_action=lambda p, selector: p.click(selector),
        failure=make_failure(FailureType.SELECTOR_NOT_FOUND),
    )

    assert result.status == HealingStatus.HEALED
    assert result.selected_selector == "button.login"
    assert result.retry_succeeded is True
    assert page.locator("#result").inner_text() == "clicked"
