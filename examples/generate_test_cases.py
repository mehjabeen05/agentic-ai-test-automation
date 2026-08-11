"""Example: RequirementAnalysis -> TestCaseGeneratorAgent -> List[TestCase].

Run from the project root:

    .venv\\Scripts\\python.exe -m examples.generate_test_cases

If a real LLM_API_KEY is configured in .env, this makes a real LLM call. If
not, it falls back to a small canned response (clearly labelled) so the
example still runs with no setup required.
"""

import json

from agents.test_generation_agent import TestCaseGenerationError, TestCaseGeneratorAgent
from core.config import get_settings
from core.llm_client import LLMClient
from core.models import RequirementAnalysis, TestCase

_DEMO_RESPONSE = """[
  {
    "test_case_id": "TC_LOGIN_001",
    "title": "Valid Login",
    "description": "Verify that a valid user can log in.",
    "preconditions": ["User has valid credentials"],
    "steps": ["Open login page", "Enter valid username", "Enter valid password", "Click Login"],
    "test_data": {"username": "valid_username", "password": "valid_password"},
    "expected_result": "Dashboard is displayed",
    "priority": "high",
    "type": "positive"
  },
  {
    "test_case_id": "TC_LOGIN_002",
    "title": "Invalid Password",
    "description": "Verify that login fails when the password is incorrect.",
    "preconditions": ["User has a valid username"],
    "steps": ["Open login page", "Enter valid username", "Enter invalid password", "Click Login"],
    "test_data": {"username": "valid_username", "password": "wrong_password"},
    "expected_result": "An error message is displayed and the dashboard is not shown.",
    "priority": "high",
    "type": "negative"
  },
  {
    "test_case_id": "TC_LOGIN_003",
    "title": "Empty Username",
    "description": "Verify that login fails when the username field is left empty.",
    "preconditions": [],
    "steps": ["Open login page", "Leave username empty", "Enter valid password", "Click Login"],
    "test_data": {"username": "", "password": "valid_password"},
    "expected_result": "A validation error is displayed and the dashboard is not shown.",
    "priority": "medium",
    "type": "validation"
  },
  {
    "test_case_id": "TC_LOGIN_004",
    "title": "Maximum Username Length",
    "description": "Verify login behavior when the username is at its maximum allowed length.",
    "preconditions": ["A username at the maximum allowed length exists"],
    "steps": [
      "Open login page",
      "Enter a username at the maximum allowed length",
      "Enter valid password",
      "Click Login"
    ],
    "test_data": {"username": "a_very_long_username_at_the_limit", "password": "valid_password"},
    "expected_result": "The user logs in successfully and the dashboard is displayed.",
    "priority": "low",
    "type": "boundary"
  },
  {
    "test_case_id": "TC_LOGIN_005",
    "title": "Special Characters In Username",
    "description": "Verify that the login form safely handles special characters in the username field.",
    "preconditions": [],
    "steps": [
      "Open login page",
      "Enter a username containing special characters",
      "Enter valid password",
      "Click Login"
    ],
    "test_data": {"username": "user'\\"<script>", "password": "valid_password"},
    "expected_result": "The login form safely handles the input with no error and no unexpected behavior.",
    "priority": "medium",
    "type": "security"
  }
]"""


class _DemoLLMClient(LLMClient):
    """Canned LLMClient used only when no real LLM_API_KEY is configured."""

    def generate(self, prompt: str) -> str:
        return _DEMO_RESPONSE


def build_sample_requirement_analysis() -> RequirementAnalysis:
    """The kind of RequirementAnalysis the Requirement Agent (Step 4) would produce."""
    return RequirementAnalysis(
        test_name="Valid Login Test",
        description="Verify that a user can successfully login.",
        preconditions=["User has valid credentials"],
        steps=["Open login page", "Enter username", "Enter password", "Click login"],
        expected_result="Dashboard is displayed",
        priority="high",
    )


def _generate_with_fallback(requirement_analysis: RequirementAnalysis) -> list[TestCase]:
    """Use the real LLM if a key is configured; fall back to a demo response on any failure.

    This keeps the example runnable with zero setup, while still exercising
    the real LLMClient when a working key is present.
    """
    settings = get_settings()
    has_configured_key = bool(settings.llm_api_key.get_secret_value().strip())

    if has_configured_key:
        print("LLM_API_KEY is configured — attempting a real LLM call.")
        try:
            return TestCaseGeneratorAgent().generate(requirement_analysis)
        except TestCaseGenerationError as exc:
            print(f"Real LLM call failed ({exc})\nFalling back to a canned demo response.\n")

    print("Running in offline demo mode with a canned response.")
    return TestCaseGeneratorAgent(llm_client=_DemoLLMClient()).generate(requirement_analysis)


def main() -> None:
    requirement_analysis = build_sample_requirement_analysis()
    print("RequirementAnalysis input:")
    print(requirement_analysis.model_dump_json(indent=2))

    test_cases = _generate_with_fallback(requirement_analysis)

    print(f"\nGenerated {len(test_cases)} test case(s):\n")
    for test_case in test_cases:
        print(json.dumps(test_case.model_dump(), indent=2))
        print()


if __name__ == "__main__":
    main()
