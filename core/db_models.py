"""Pydantic models for rows read back from the persistence layer.

Deliberately separate from core/models.py's domain models (TestCase,
TestExecutionResult, FailureAnalysis, HealingResult, ...): those represent
LLM-facing trust boundaries and API-shaped data, while these represent
what a database ROW actually looks like — including surrogate keys,
foreign keys, and stored timestamps the domain models don't carry.
Reusing the domain models one-for-one as database rows would leak
DB-specific concerns (autoincrement ids, FK columns) into the LLM/agent
layer, which is exactly the separation this project maintains elsewhere.
"""

from pydantic import BaseModel

from core.models import (
    ExecutionStatus,
    FailureType,
    HealingStatus,
    Priority,
    RequirementAnalysis,
    TestCaseType,
)


class RequirementRecord(BaseModel):
    """A row from the `requirements` table."""

    id: int
    requirement_text: str
    analysis: RequirementAnalysis | None = None
    created_at: str


class TestCaseRecord(BaseModel):
    """A row from the `test_cases` table."""

    id: int
    requirement_id: int | None
    test_case_id: str
    title: str
    description: str
    priority: Priority
    type: TestCaseType
    test_data: dict[str, str]
    preconditions: list[str] = []
    steps: list[str] = []
    expected_result: str = ""
    created_at: str


class ExecutionRecord(BaseModel):
    """A row from the `test_executions` table."""

    id: int
    test_case_id: str
    status: ExecutionStatus
    duration: float
    error: str | None
    stdout: str | None
    stderr: str | None
    screenshot: str | None
    healed: bool
    created_at: str


class FailureAnalysisRecord(BaseModel):
    """A row from the `failure_analyses` table."""

    id: int
    execution_id: int
    failure_type: FailureType
    summary: str
    root_cause: str
    suggested_fix: str
    confidence: float
    is_likely_environment_issue: bool
    is_likely_test_issue: bool
    created_at: str


class HealingRecord(BaseModel):
    """A row from the `healing_attempts` table."""

    id: int
    execution_id: int
    status: HealingStatus
    original_selector: str
    candidate_selectors: list[str]
    selected_selector: str | None
    validation_result: list[dict]
    retry_succeeded: bool | None
    confidence: float | None
    reason: str
    created_at: str
