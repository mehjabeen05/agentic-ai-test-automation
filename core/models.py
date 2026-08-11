"""Shared Pydantic models for the framework.

Populated incrementally as later steps introduce structured data
(test cases, execution results, failure analysis, healing attempts).
"""

from enum import Enum

from pydantic import BaseModel, Field, field_validator

from core.logger import get_logger

logger = get_logger(__name__)


class Priority(str, Enum):
    """Test case priority."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TestRequirement(BaseModel):
    """A raw natural-language software testing requirement submitted by a user."""

    text: str = Field(..., min_length=1, description="The requirement, in plain English.")

    @field_validator("text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("requirement text must not be blank")
        return stripped


class TestStep(BaseModel):
    """A single, atomic action within an ordered test case.

    Used to validate each step individually before it is stored as a plain
    string on `RequirementAnalysis.steps`.
    """

    description: str = Field(..., min_length=1)

    @field_validator("description")
    @classmethod
    def not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("step description must not be blank")
        return stripped


class RequirementAnalysis(BaseModel):
    """Structured, validated test-case specification derived from a natural-language requirement.

    This is the trust boundary for LLM output: raw JSON from the LLM is only
    ever treated as data once it has passed validation into this model.
    """

    test_name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    preconditions: list[str] = Field(default_factory=list)
    steps: list[str] = Field(..., min_length=1)
    expected_result: str = Field(..., min_length=1)
    priority: Priority = Priority.MEDIUM

    @field_validator("test_name", "description", "expected_result")
    @classmethod
    def not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped

    @field_validator("preconditions")
    @classmethod
    def preconditions_not_blank(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item and item.strip()]

    @field_validator("steps", mode="before")
    @classmethod
    def validate_steps(cls, value: object) -> list[str]:
        if not isinstance(value, list) or not value:
            raise ValueError("steps must be a non-empty list")
        # Route every raw step through TestStep so each one is individually
        # validated (non-blank, stripped) rather than accepted as-is.
        return [TestStep(description=item).description for item in value]

    @field_validator("priority", mode="before")
    @classmethod
    def normalize_priority(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value


class TestCaseType(str, Enum):
    """Category of a generated test case."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    BOUNDARY = "boundary"
    VALIDATION = "validation"
    SECURITY = "security"


class TestCase(BaseModel):
    """A single, structured, validated test case generated from a RequirementAnalysis.

    Like RequirementAnalysis, this is a trust boundary: raw LLM JSON is only
    treated as data once it has passed validation into this model.
    """

    test_case_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    preconditions: list[str] = Field(default_factory=list)
    steps: list[str] = Field(..., min_length=1)
    test_data: dict[str, str] = Field(default_factory=dict)
    expected_result: str = Field(..., min_length=1)
    priority: Priority = Priority.MEDIUM
    type: TestCaseType

    @field_validator("test_case_id", "title", "description", "expected_result")
    @classmethod
    def not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped

    @field_validator("preconditions")
    @classmethod
    def preconditions_not_blank(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item and item.strip()]

    @field_validator("steps", mode="before")
    @classmethod
    def validate_steps(cls, value: object) -> list[str]:
        if not isinstance(value, list) or not value:
            raise ValueError("steps must be a non-empty list")
        # Route every raw step through TestStep so each one is individually
        # validated (non-blank, stripped) rather than accepted as-is.
        return [TestStep(description=item).description for item in value]

    @field_validator("priority", mode="before")
    @classmethod
    def normalize_priority(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value


class ExecutionStatus(str, Enum):
    """Outcome of running a generated test through the Test Execution Engine."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


class TestExecutionResult(BaseModel):
    """Structured result of executing one generated Playwright test file."""

    test_id: str = Field(..., min_length=1)
    status: ExecutionStatus
    duration: float = Field(..., ge=0.0)
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    screenshot: str | None = None

    # Populated only when Step 9's healing pipeline actually retried and
    # succeeded. All default to "no healing happened" for backward
    # compatibility with every TestExecutionResult built by earlier steps.
    healed: bool = False
    original_selector: str | None = None
    healed_selector: str | None = None

    @field_validator("test_id")
    @classmethod
    def not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("test_id must not be blank")
        return stripped


class FailureType(str, Enum):
    """Category of an automated test failure, as classified by the Failure Analysis Agent."""

    SELECTOR_NOT_FOUND = "selector_not_found"
    TIMEOUT = "timeout"
    ASSERTION_FAILURE = "assertion_failure"
    NAVIGATION_ERROR = "navigation_error"
    ELEMENT_NOT_INTERACTABLE = "element_not_interactable"
    TEST_DATA_ERROR = "test_data_error"
    AUTHENTICATION_ERROR = "authentication_error"
    ENVIRONMENT_ERROR = "environment_error"
    UNKNOWN = "unknown"


class FailureAnalysisInput(BaseModel):
    """Structured input to the Failure Analysis Agent.

    Built from a TestExecutionResult plus optional extra context. All
    string fields are treated as untrusted: the agent redacts and truncates
    them before they are ever sent to an LLM.
    """

    test_id: str = Field(..., min_length=1)
    status: ExecutionStatus
    duration: float = Field(..., ge=0.0)
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    screenshot_path: str | None = None

    # Optional additional context, none of it required.
    test_source: str | None = None
    dom_snippet: str | None = None
    browser: str | None = None
    url: str | None = None
    failed_selector: str | None = None
    traceback: str | None = None

    @field_validator("test_id")
    @classmethod
    def not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("test_id must not be blank")
        return stripped

    @classmethod
    def from_execution_result(
        cls,
        result: TestExecutionResult,
        *,
        test_source: str | None = None,
        dom_snippet: str | None = None,
        browser: str | None = None,
        url: str | None = None,
        failed_selector: str | None = None,
        traceback: str | None = None,
    ) -> "FailureAnalysisInput":
        """Build analysis input from a TestExecutionResult, with optional extra context."""
        return cls(
            test_id=result.test_id,
            status=result.status,
            duration=result.duration,
            stdout=result.stdout,
            stderr=result.stderr,
            error=result.error,
            screenshot_path=result.screenshot,
            test_source=test_source,
            dom_snippet=dom_snippet,
            browser=browser,
            url=url,
            failed_selector=failed_selector,
            traceback=traceback,
        )


class FailureAnalysis(BaseModel):
    """Structured, advisory explanation of a test failure.

    This is the trust boundary for the Failure Analysis Agent's LLM output:
    raw JSON is only ever treated as data once it has passed validation
    into this model. It is advisory — nothing in the framework acts on it
    automatically as of this step.
    """

    failure_type: FailureType
    summary: str = Field(..., min_length=1)
    root_cause: str = Field(..., min_length=1)
    suggested_fix: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_likely_environment_issue: bool
    is_likely_test_issue: bool

    @field_validator("summary", "root_cause", "suggested_fix")
    @classmethod
    def not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped

    @field_validator("failure_type", mode="before")
    @classmethod
    def normalize_failure_type(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value


# Patterns that disqualify a candidate "selector" outright: these indicate an
# attempt to smuggle executable code or markup into what must be a plain
# selector string. Checked case-insensitively.
_DANGEROUS_SELECTOR_MARKERS = ("javascript:", "<script", "eval(", "exec(", "```", "\n", "\r")
_MAX_SELECTOR_LENGTH = 300


class HealingSuggestion(BaseModel):
    """Raw, validated output of the Selector Healing Agent.

    This model is a hard trust boundary: it can only ever hold selector
    strings, never code. `candidate_selectors` is sanitized before normal
    field validation runs — any candidate containing a disallowed pattern
    (a script tag, a `javascript:` URI, `eval(`/`exec(`, a code fence, or an
    embedded newline) is dropped, not merely flagged. If nothing usable
    remains, the whole suggestion is rejected.
    """

    original_selector: str = Field(..., min_length=1)
    candidate_selectors: list[str] = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("original_selector", "reason")
    @classmethod
    def not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped

    @field_validator("candidate_selectors", mode="before")
    @classmethod
    def sanitize_candidates(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            raise ValueError("candidate_selectors must be a list of strings")

        cleaned: list[str] = []
        for raw in value:
            if not isinstance(raw, str):
                continue
            candidate = raw.strip()
            if not candidate:
                continue
            if len(candidate) > _MAX_SELECTOR_LENGTH:
                logger.warning(
                    "Dropping implausibly long candidate selector (%d chars)", len(candidate)
                )
                continue
            lowered = candidate.lower()
            if any(marker in lowered for marker in _DANGEROUS_SELECTOR_MARKERS):
                logger.warning("Dropping unsafe candidate selector: %r", candidate)
                continue
            cleaned.append(candidate)

        if not cleaned:
            raise ValueError("No safe, usable candidate selectors were provided.")
        return cleaned


class SelectorValidation(BaseModel):
    """Result of checking one candidate selector against the live browser DOM.

    A candidate is never considered usable on the strength of the LLM's
    suggestion alone — only this model, produced by actually querying a
    real Playwright Page, can mark one `valid`.
    """

    selector: str = Field(..., min_length=1)
    valid: bool
    match_count: int = Field(..., ge=0)
    reason: str | None = None


class HealingStatus(str, Enum):
    """Outcome of one controlled selector-healing attempt."""

    HEALED = "healed"
    FAILED = "failed"
    SKIPPED = "skipped"


class HealingResult(BaseModel):
    """A complete, structured record of one selector-healing attempt.

    Persisted as JSON under reports/healing/ regardless of outcome, so
    every attempt — successful, failed, or skipped — is auditable.
    """

    test_id: str = Field(..., min_length=1)
    original_selector: str = Field(..., min_length=1)
    candidate_selectors: list[str] = Field(default_factory=list)
    selected_selector: str | None = None
    validations: list[SelectorValidation] = Field(default_factory=list)
    status: HealingStatus
    retry_succeeded: bool | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str = Field(..., min_length=1)
    timestamp: str = Field(..., min_length=1)

    @field_validator("test_id", "original_selector", "reason", "timestamp")
    @classmethod
    def not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped
