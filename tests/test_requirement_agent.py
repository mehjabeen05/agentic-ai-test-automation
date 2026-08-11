"""Unit tests for the Requirement Agent.

The agent is driven through a fake LLMClient, so these tests never contact a
real LLM provider and pass with or without a configured LLM_API_KEY. This
also exercises the LLMClient abstraction exactly the way a real provider
swap would: only the injected client changes, not the agent.
"""

import pytest

from agents.requirement_agent import RequirementAgent, RequirementAgentError
from core.llm_client import LLMClient, LLMRequestError
from core.models import TestRequirement

VALID_JSON = """{
    "test_name": "Valid Login Test",
    "description": "Verify that a user can successfully login.",
    "preconditions": ["User has valid credentials"],
    "steps": ["Open login page", "Enter username", "Enter password", "Click login"],
    "expected_result": "Dashboard is displayed",
    "priority": "high"
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
    agent = RequirementAgent(llm_client=fake_client)
    return agent, fake_client


def test_analyze_returns_valid_requirement_analysis():
    agent, fake_client = make_agent(response=VALID_JSON)
    requirement = TestRequirement(text="User should be able to login with valid credentials.")

    analysis = agent.analyze(requirement)

    assert analysis.test_name == "Valid Login Test"
    assert analysis.expected_result == "Dashboard is displayed"
    assert analysis.priority.value == "high"
    assert requirement.text in fake_client.last_prompt


def test_analyze_strips_prose_wrapped_around_json():
    wrapped = f"Here is your JSON:\n{VALID_JSON}\nLet me know if you need anything else."
    agent, _ = make_agent(response=wrapped)
    requirement = TestRequirement(text="User should be able to login.")

    analysis = agent.analyze(requirement)

    assert analysis.test_name == "Valid Login Test"


def test_analyze_strips_markdown_code_fence():
    fenced = f"```json\n{VALID_JSON}\n```"
    agent, _ = make_agent(response=fenced)
    requirement = TestRequirement(text="User should be able to login.")

    analysis = agent.analyze(requirement)

    assert analysis.test_name == "Valid Login Test"


def test_analyze_raises_when_response_has_no_json():
    agent, _ = make_agent(response="Sure, I can help with that! What should the test do?")
    requirement = TestRequirement(text="User should be able to login.")

    with pytest.raises(RequirementAgentError):
        agent.analyze(requirement)


def test_analyze_raises_when_response_is_malformed_json():
    agent, _ = make_agent(response="{test_name: 'missing quotes'}")
    requirement = TestRequirement(text="User should be able to login.")

    with pytest.raises(RequirementAgentError):
        agent.analyze(requirement)


def test_analyze_raises_when_response_fails_schema_validation():
    incomplete = '{"test_name": "Login Test"}'
    agent, _ = make_agent(response=incomplete)
    requirement = TestRequirement(text="User should be able to login.")

    with pytest.raises(RequirementAgentError):
        agent.analyze(requirement)


def test_analyze_raises_when_response_contains_executable_code_instead_of_json():
    code_response = "import os\nos.system('rm -rf /')"
    agent, _ = make_agent(response=code_response)
    requirement = TestRequirement(text="User should be able to login.")

    with pytest.raises(RequirementAgentError):
        agent.analyze(requirement)


def test_analyze_wraps_llm_client_errors():
    agent, _ = make_agent(error=LLMRequestError("provider is down"))
    requirement = TestRequirement(text="User should be able to login.")

    with pytest.raises(RequirementAgentError):
        agent.analyze(requirement)
