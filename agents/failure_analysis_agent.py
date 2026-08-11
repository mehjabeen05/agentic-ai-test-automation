"""Failure Analysis Agent: turns a failed TestExecutionResult (plus optional
context) into a structured, advisory explanation of what likely went wrong.

Architecture (per the project's design):

    TestExecutionResult -> FailureAnalysisInput -> FailureAnalysisAgent
        -> LLMClient -> LLM -> raw text -> JSON extraction
        -> Pydantic validation -> FailureAnalysis

Like the other agents, this one depends only on the `LLMClient` interface —
never an LLM provider SDK directly — and never executes anything the LLM
returns. All text fields are redacted for obvious secrets and truncated
before being sent to the LLM. This step is analysis only: it never modifies
a test, reruns a test, or attempts any repair — that is Step 9. Its output
is advisory, not an instruction the framework acts on automatically.
"""

import json
import re

from pydantic import ValidationError

from core.config import PROJECT_ROOT
from core.llm_client import LLMClient, LLMClientError, get_llm_client
from core.logger import get_logger
from core.models import ExecutionStatus, FailureAnalysis, FailureAnalysisInput
from core.redaction import redact_secrets
from core.repositories import FailureAnalysisRepository

logger = get_logger(__name__)

PROMPT_TEMPLATE_PATH = PROJECT_ROOT / "prompts" / "failure_analysis_prompt.txt"
_CONTEXT_PLACEHOLDER = "{{FAILURE_CONTEXT_JSON}}"

_CODE_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

# Caps how much of any single verbose field (stdout, DOM, traceback, ...) is
# sent to the LLM. Applied per-field, after redaction.
_MAX_FIELD_CHARS = 4000

# Statuses that mean "nothing failed" — there is no failure to explain.
_NO_FAILURE_STATUSES = frozenset({ExecutionStatus.PASSED, ExecutionStatus.SKIPPED})


class FailureAnalysisError(Exception):
    """Raised when a failure cannot be analyzed, or the analysis fails validation."""


def _load_prompt_template() -> str:
    try:
        return PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise FailureAnalysisError(
            f"Could not read prompt template at {PROMPT_TEMPLATE_PATH}"
        ) from exc


def _extract_json_object(raw_text: str) -> str:
    """Best-effort extraction of a single JSON object from raw LLM text.

    LLMs sometimes wrap JSON in prose or a markdown code fence even when
    told not to. This locates the JSON object for parsing; it never
    evaluates or executes any of the text.
    """
    text = raw_text.strip()

    fence_match = _CODE_FENCE_PATTERN.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise FailureAnalysisError("The LLM response did not contain a JSON object.")

    return text[start : end + 1]


def _truncate(text: str | None, max_chars: int = _MAX_FIELD_CHARS) -> str | None:
    """Cap `text` to `max_chars`, keeping the start and end (both are often useful)."""
    if text is None or len(text) <= max_chars:
        return text
    half = max_chars // 2
    omitted = len(text) - max_chars
    return f"{text[:half]}\n...[truncated {omitted} chars]...\n{text[-half:]}"


def _redact_and_truncate(text: str | None) -> str | None:
    return _truncate(redact_secrets(text))


def _build_context(analysis_input: FailureAnalysisInput) -> dict:
    """Redact secrets and truncate verbose fields before building the prompt.

    This is the only place raw execution/test data is transformed before
    being sent to the LLM. Screenshot content itself is never sent — only
    its path and whether one exists — since the LLMClient abstraction has
    no image-input support yet.
    """
    return {
        "test_id": analysis_input.test_id,
        "status": analysis_input.status.value,
        "duration": analysis_input.duration,
        "stdout": _redact_and_truncate(analysis_input.stdout),
        "stderr": _redact_and_truncate(analysis_input.stderr),
        "error": _redact_and_truncate(analysis_input.error),
        "screenshot_available": bool(analysis_input.screenshot_path),
        "screenshot_path": analysis_input.screenshot_path,
        "test_source": _redact_and_truncate(analysis_input.test_source),
        "dom_snippet": _redact_and_truncate(analysis_input.dom_snippet),
        "browser": analysis_input.browser,
        "url": analysis_input.url,
        "failed_selector": analysis_input.failed_selector,
        "traceback": _redact_and_truncate(analysis_input.traceback),
    }


class FailureAnalysisAgent:
    """Analyzes a failed test execution and produces a structured, advisory FailureAnalysis."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        failure_analysis_repository: FailureAnalysisRepository | None = None,
    ) -> None:
        self._llm_client = llm_client or get_llm_client()
        self._prompt_template = _load_prompt_template()
        self._failure_analysis_repository = failure_analysis_repository or FailureAnalysisRepository()

    def analyze(
        self, analysis_input: FailureAnalysisInput, execution_id: int | None = None
    ) -> FailureAnalysis:
        """Analyze a failed test execution and return a structured FailureAnalysis.

        `execution_id` (e.g. from `TestRunner.last_execution_id`) links the
        persisted analysis back to its execution row; it's optional — when
        omitted, the analysis is still returned but not persisted, since
        `failure_analyses.execution_id` is required.

        Raises:
            FailureAnalysisError: `analysis_input` is invalid, describes a
                passed/skipped execution (nothing to analyze), the LLM call
                failed, or its response could not be parsed or validated.
        """
        if not isinstance(analysis_input, FailureAnalysisInput):
            raise FailureAnalysisError(
                "analysis_input must be a validated FailureAnalysisInput instance."
            )

        if analysis_input.status in _NO_FAILURE_STATUSES:
            raise FailureAnalysisError(
                f"Execution result status is '{analysis_input.status.value}' — "
                "there is no failure to analyze."
            )

        context = _build_context(analysis_input)
        context_json = json.dumps(context, indent=2)
        prompt = self._prompt_template.replace(_CONTEXT_PLACEHOLDER, context_json)

        logger.info(
            "Analyzing failure for '%s' (status=%s)",
            analysis_input.test_id,
            analysis_input.status.value,
        )

        try:
            raw_response = self._llm_client.generate(prompt)
        except LLMClientError as exc:
            logger.error("Failure Analysis Agent: LLM call failed: %s", exc)
            raise FailureAnalysisError(f"Failed to analyze failure: {exc}") from exc

        try:
            json_text = _extract_json_object(raw_response)
            data = json.loads(json_text)
        except FailureAnalysisError:
            logger.error("Failure Analysis Agent: no JSON object found in LLM response")
            raise
        except json.JSONDecodeError as exc:
            logger.error("Failure Analysis Agent: LLM response was not valid JSON")
            raise FailureAnalysisError("The LLM response could not be parsed as JSON.") from exc

        if not isinstance(data, dict):
            logger.error("Failure Analysis Agent: parsed JSON was not an object")
            raise FailureAnalysisError("The LLM response JSON must be an object.")

        try:
            analysis = FailureAnalysis.model_validate(data)
        except ValidationError as exc:
            logger.error("Failure Analysis Agent: LLM response failed schema validation")
            raise FailureAnalysisError(
                f"The LLM response did not match the expected schema: {exc}"
            ) from exc

        logger.info(
            "Failure analyzed for '%s': type=%s confidence=%.2f",
            analysis_input.test_id,
            analysis.failure_type.value,
            analysis.confidence,
        )

        if execution_id is not None:
            try:
                self._failure_analysis_repository.save(execution_id, analysis)
            except Exception as exc:  # noqa: BLE001 - persistence must never break analysis
                logger.error(
                    "Could not persist failure analysis for execution %d: %s", execution_id, exc
                )

        return analysis
