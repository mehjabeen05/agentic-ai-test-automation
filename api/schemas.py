"""Pydantic request/response schemas for the FastAPI layer.

Kept separate from core/models.py's domain models and core/db_models.py's
persistence records: these describe the shape of HTTP request/response
bodies specifically. Some schemas simply mirror a domain model's public
fields where that's the natural, non-leaky shape (e.g. `RequirementAnalysis`
in `RequirementResponse`); history/list endpoints use dedicated summary
schemas instead of the raw DB record models, so the API contract doesn't
change automatically just because the database schema does.
"""

from pydantic import BaseModel, Field

from core.models import (
    ExecutionStatus,
    FailureType,
    HealingStatus,
    Priority,
    RequirementAnalysis,
    TestCase,
    TestCaseType,
)


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = "ok"


class StatsResponse(BaseModel):
    """Response for GET /api/v1/stats — dashboard summary counters.

    Counts are computed across ALL recorded executions (not just the
    latest one per test case), via the existing repositories — nothing
    here is hardcoded or duplicated from any agent's logic. `passed`
    includes executions that ultimately passed after healing (their
    status is `passed` either way); `healed` is a separate, overlapping
    count of how many of those passed executions got there via healing.
    """

    total_test_cases: int
    passed: int
    failed: int
    healed: int
    success_rate: float = Field(..., description="Percentage (0-100), rounded to 1 decimal place.")


# --- Requirements ---


class RequirementRequest(BaseModel):
    """Request body for POST /api/v1/requirements."""

    requirement: str = Field(
        ..., min_length=1, description="The requirement, in plain English."
    )


class RequirementResponse(BaseModel):
    """Response for POST /api/v1/requirements."""

    requirement_id: int
    analysis: RequirementAnalysis


class RequirementSummary(BaseModel):
    """One row of GET /api/v1/requirements."""

    id: int
    requirement_text: str
    created_at: str


# --- Test case generation ---


class TestCaseGenerationRequest(BaseModel):
    """Request body for POST /api/v1/test-cases."""

    requirement_id: int


class TestCaseGenerationResponse(BaseModel):
    """Response for POST /api/v1/test-cases."""

    requirement_id: int
    test_cases: list[TestCase]


class TestCaseSummary(BaseModel):
    """One row of GET /api/v1/requirements/{requirement_id}/test-cases."""

    id: int
    requirement_id: int | None
    test_case_id: str
    title: str
    description: str
    priority: Priority
    type: TestCaseType
    created_at: str


# --- Playwright code generation ---


class TestGenerationRequest(BaseModel):
    """Request body for POST /api/v1/tests/generate."""

    test_case_id: str = Field(..., min_length=1)


class ValidationSummary(BaseModel):
    """Safety-validation outcome embedded in TestGenerationResponse."""

    valid: bool
    issues: list[str] = Field(default_factory=list)


class TestGenerationResponse(BaseModel):
    """Response for POST /api/v1/tests/generate."""

    test_case_id: str
    generated_file: str
    validation: ValidationSummary


# --- Test execution ---


class TestExecutionRequest(BaseModel):
    """Request body for POST /api/v1/tests/run."""

    test_case_id: str = Field(..., min_length=1)


class TestExecutionResponse(BaseModel):
    """Response for POST /api/v1/tests/run."""

    execution_id: int | None
    test_case_id: str
    status: ExecutionStatus
    duration: float
    error: str | None = None
    screenshot: str | None = None
    healed: bool = False


class ExecutionSummary(BaseModel):
    """One row of GET /api/v1/test-cases/{test_case_id}/executions or GET /api/v1/executions/{id}."""

    id: int
    test_case_id: str
    status: ExecutionStatus
    duration: float
    error: str | None
    screenshot: str | None
    healed: bool
    created_at: str


# --- Failure analysis ---


class FailureAnalysisRequest(BaseModel):
    """Request body for POST /api/v1/tests/analyze-failure."""

    execution_id: int


class FailureAnalysisResponse(BaseModel):
    """Response for POST /api/v1/tests/analyze-failure and GET .../failure-analysis."""

    execution_id: int
    failure_type: FailureType
    summary: str
    root_cause: str
    suggested_fix: str
    confidence: float
    is_likely_environment_issue: bool
    is_likely_test_issue: bool


# --- Healing ---


class HealingRequest(BaseModel):
    """Request body for POST /api/v1/tests/heal."""

    execution_id: int
    url: str | None = Field(
        default=None,
        description=(
            "Optional live target URL to validate candidate selectors against. "
            "Without it, healing runs with no live page loaded and will "
            "typically report that no candidate could be validated — this "
            "is a known limitation, not a silent failure (see README)."
        ),
    )


class HealingResponse(BaseModel):
    """Response for POST /api/v1/tests/heal and GET .../healing."""

    execution_id: int
    status: HealingStatus
    original_selector: str
    candidate_selectors: list[str]
    selected_selector: str | None
    retry_succeeded: bool | None
    confidence: float | None
    reason: str
