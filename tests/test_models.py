"""Unit tests for the shared Pydantic models used by the Requirement Agent."""

import pytest
from pydantic import ValidationError

from core.models import Priority, RequirementAnalysis, TestRequirement, TestStep


def make_valid_payload(**overrides: object) -> dict:
    payload = {
        "test_name": "Valid Login Test",
        "description": "Verify that a user can successfully login.",
        "preconditions": ["User has valid credentials"],
        "steps": ["Open login page", "Enter username", "Enter password", "Click login"],
        "expected_result": "Dashboard is displayed",
        "priority": "high",
    }
    payload.update(overrides)
    return payload


def test_requirement_analysis_accepts_valid_payload():
    analysis = RequirementAnalysis.model_validate(make_valid_payload())

    assert analysis.test_name == "Valid Login Test"
    assert analysis.priority == Priority.HIGH
    assert analysis.steps == [
        "Open login page",
        "Enter username",
        "Enter password",
        "Click login",
    ]


def test_requirement_analysis_defaults_empty_preconditions():
    analysis = RequirementAnalysis.model_validate(make_valid_payload(preconditions=[]))
    assert analysis.preconditions == []


def test_requirement_analysis_normalizes_priority_case():
    analysis = RequirementAnalysis.model_validate(make_valid_payload(priority="HIGH"))
    assert analysis.priority == Priority.HIGH


def test_requirement_analysis_rejects_invalid_priority():
    with pytest.raises(ValidationError):
        RequirementAnalysis.model_validate(make_valid_payload(priority="urgent"))


def test_requirement_analysis_rejects_empty_steps():
    with pytest.raises(ValidationError):
        RequirementAnalysis.model_validate(make_valid_payload(steps=[]))


def test_requirement_analysis_rejects_blank_step():
    with pytest.raises(ValidationError):
        RequirementAnalysis.model_validate(make_valid_payload(steps=["Open login page", "   "]))


def test_requirement_analysis_rejects_missing_required_field():
    payload = make_valid_payload()
    del payload["expected_result"]
    with pytest.raises(ValidationError):
        RequirementAnalysis.model_validate(payload)


def test_requirement_analysis_ignores_unexpected_extra_fields():
    analysis = RequirementAnalysis.model_validate(make_valid_payload(extra_field="ignored"))
    assert not hasattr(analysis, "extra_field")


def test_test_requirement_rejects_blank_text():
    with pytest.raises(ValidationError):
        TestRequirement(text="   ")


def test_test_requirement_strips_whitespace():
    requirement = TestRequirement(text="  Login works  ")
    assert requirement.text == "Login works"


def test_test_step_rejects_blank_description():
    with pytest.raises(ValidationError):
        TestStep(description="")
