"""Unit tests for the Test Case Generator Agent.

The agent is driven through a fake LLMClient, so these tests never contact a
real LLM provider and pass with or without a configured LLM_API_KEY.
"""

import pytest
from pydantic import ValidationError

from agents.test_generation_agent import TestCaseGenerationError, TestCaseGeneratorAgent
from core.llm_client import LLMClient, LLMResponseError
from core.models import RequirementAnalysis, TestCase, TestCaseType

VALID_JSON_ARRAY = """[
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
  }
]"""

DUPLICATE_JSON_ARRAY = """[
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
    "test_case_id": "TC_LOGIN_001B",
    "title": "  valid login  ",
    "description": "A near-duplicate of the first test case, differing only in casing/whitespace.",
    "preconditions": ["User has valid credentials"],
    "steps": ["Open login page", "Enter valid username", "Enter valid password", "Click Login"],
    "test_data": {"username": "valid_username", "password": "valid_password"},
    "expected_result": "Dashboard is displayed",
    "priority": "high",
    "type": "positive"
  }
]"""


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
    agent = TestCaseGeneratorAgent(llm_client=fake_client)
    return agent, fake_client


def make_valid_analysis() -> RequirementAnalysis:
    return RequirementAnalysis(
        test_name="Valid Login Test",
        description="Verify that a user can successfully login.",
        preconditions=["User has valid credentials"],
        steps=["Open login page", "Enter username", "Enter password", "Click login"],
        expected_result="Dashboard is displayed",
        priority="high",
    )


# --- A. Valid RequirementAnalysis -> multiple valid TestCase objects ---


def test_generate_returns_multiple_valid_test_cases():
    agent, fake_client = make_agent(response=VALID_JSON_ARRAY)

    test_cases = agent.generate(make_valid_analysis())

    assert len(test_cases) == 3
    assert all(isinstance(tc, TestCase) for tc in test_cases)
    assert {tc.type for tc in test_cases} == {
        TestCaseType.POSITIVE,
        TestCaseType.NEGATIVE,
        TestCaseType.VALIDATION,
    }
    assert "Valid Login Test" in fake_client.last_prompt


# --- B. Invalid JSON -> graceful error ---


def test_generate_raises_on_malformed_json():
    agent, _ = make_agent(response="[{test_case_id: 'missing quotes'}]")

    with pytest.raises(TestCaseGenerationError):
        agent.generate(make_valid_analysis())


def test_generate_raises_when_response_has_no_json_array():
    agent, _ = make_agent(response="Sure! Here are some test ideas for you.")

    with pytest.raises(TestCaseGenerationError):
        agent.generate(make_valid_analysis())


def test_generate_raises_when_response_contains_executable_code_instead_of_json():
    code_response = "import os\nos.system('rm -rf /')"
    agent, _ = make_agent(response=code_response)

    with pytest.raises(TestCaseGenerationError):
        agent.generate(make_valid_analysis())


# --- C. Missing required fields -> (Pydantic) schema validation failure, wrapped ---


def test_generate_raises_when_required_field_is_missing():
    incomplete = '[{"test_case_id": "TC_1", "title": "Missing fields"}]'
    agent, _ = make_agent(response=incomplete)

    with pytest.raises(TestCaseGenerationError):
        agent.generate(make_valid_analysis())


# --- D. Duplicate test cases -> removed safely ---


def test_generate_deduplicates_test_cases_with_same_title():
    agent, _ = make_agent(response=DUPLICATE_JSON_ARRAY)

    test_cases = agent.generate(make_valid_analysis())

    assert len(test_cases) == 1
    assert test_cases[0].test_case_id == "TC_LOGIN_001"


# --- E. Empty requirement / empty RequirementAnalysis -> validation error ---


def test_generate_rejects_non_requirement_analysis_input():
    agent, _ = make_agent(response=VALID_JSON_ARRAY)

    with pytest.raises(TestCaseGenerationError):
        agent.generate(None)


def test_requirement_analysis_with_empty_fields_is_rejected_before_reaching_agent():
    with pytest.raises(ValidationError):
        RequirementAnalysis(
            test_name="",
            description="",
            preconditions=[],
            steps=[],
            expected_result="",
            priority="high",
        )


# --- F. Invalid test type -> validation failure ---


def test_generate_raises_when_test_case_type_is_invalid():
    bad_json = VALID_JSON_ARRAY.replace('"type": "positive"', '"type": "random"')
    agent, _ = make_agent(response=bad_json)

    with pytest.raises(TestCaseGenerationError):
        agent.generate(make_valid_analysis())


# --- G. Invalid priority -> validation failure ---


def test_generate_raises_when_priority_is_invalid():
    bad_json = VALID_JSON_ARRAY.replace('"priority": "high"', '"priority": "urgent"', 1)
    agent, _ = make_agent(response=bad_json)

    with pytest.raises(TestCaseGenerationError):
        agent.generate(make_valid_analysis())


# --- Extra: empty LLM response is also handled gracefully ---


def test_generate_wraps_llm_client_errors():
    agent, _ = make_agent(error=LLMResponseError("The LLM response was empty."))

    with pytest.raises(TestCaseGenerationError):
        agent.generate(make_valid_analysis())
