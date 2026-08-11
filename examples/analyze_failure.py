"""Example: TestExecutionResult -> FailureAnalysisAgent -> FailureAnalysis.

Run from the project root:

    .venv\\Scripts\\python.exe -m examples.analyze_failure

If a real LLM_API_KEY is configured in .env, this attempts a real LLM call.
If not (or if that call fails), it falls back to a small canned response so
the example still runs with no setup required.

This is analysis only — it does not modify, rerun, or "fix" anything.
"""

from agents.failure_analysis_agent import FailureAnalysisAgent, FailureAnalysisError
from core.config import get_settings
from core.llm_client import LLMClient
from core.models import ExecutionStatus, FailureAnalysisInput, TestExecutionResult

_DEMO_RESPONSE = """{
  "failure_type": "selector_not_found",
  "summary": "The username field could not be located on the login page.",
  "root_cause": "TimeoutError on locator(\\"#username\\"); page loaded fine, so the selector no longer matches DOM.",
  "suggested_fix": "Use a more stable selector (e.g. a data-test attribute) for the username field.",
  "confidence": 0.65,
  "is_likely_environment_issue": false,
  "is_likely_test_issue": true
}"""


class _DemoLLMClient(LLMClient):
    """Canned LLMClient used when no real LLM_API_KEY is configured, or the real call fails."""

    def generate(self, prompt: str) -> str:
        return _DEMO_RESPONSE


def build_sample_execution_result() -> TestExecutionResult:
    """The kind of failed TestExecutionResult the Execution Engine (Step 7) would produce."""
    return TestExecutionResult(
        test_id="test_tc_login_001",
        status=ExecutionStatus.FAILED,
        duration=39.66,
        exit_code=1,
        stdout=(
            "FAILED tests/generated/test_tc_login_001.py::test_valid_login[chromium]\n"
            "1 failed in 38.71s"
        ),
        stderr="",
        error=(
            'TimeoutError: Page.fill: Timeout 30000ms exceeded.\n'
            'Call log:\nwaiting for locator("#username")'
        ),
        screenshot=(
            "screenshots/tests-generated-test-tc-login-001-py-test-valid-login-chromium/"
            "test-failed-1.png"
        ),
    )


def _analyze_with_fallback(analysis_input: FailureAnalysisInput) -> tuple:
    """Use the real LLM if a key is configured; fall back to a demo response on any failure."""
    settings = get_settings()
    has_configured_key = bool(settings.llm_api_key.get_secret_value().strip())

    if has_configured_key:
        print("LLM_API_KEY is configured — attempting a real LLM call.")
        agent = FailureAnalysisAgent()
        try:
            return agent, agent.analyze(analysis_input)
        except FailureAnalysisError as exc:
            print(f"Real LLM call failed ({exc})\nFalling back to a canned demo response.\n")

    print("Running in offline demo mode with a canned response.")
    agent = FailureAnalysisAgent(llm_client=_DemoLLMClient())
    return agent, agent.analyze(analysis_input)


def main() -> None:
    execution_result = build_sample_execution_result()
    print("TestExecutionResult input:")
    print(execution_result.model_dump_json(indent=2))

    analysis_input = FailureAnalysisInput.from_execution_result(
        execution_result,
        failed_selector="#username",
        url="https://example.com/login",
        browser="chromium",
    )

    _, analysis = _analyze_with_fallback(analysis_input)

    print("\nFailureAnalysis output:")
    print(analysis.model_dump_json(indent=2))
    print(
        "\n(This is advisory analysis only — nothing was modified, rerun, "
        "or automatically fixed.)"
    )


if __name__ == "__main__":
    main()
