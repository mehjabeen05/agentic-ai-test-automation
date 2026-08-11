"""Test Case Generator Agent: turns a validated RequirementAnalysis into a
comprehensive, deduplicated set of structured test cases.

Architecture (per the project's design):

    RequirementAnalysis -> TestCaseGeneratorAgent -> LLMClient -> LLM
        -> raw text -> JSON extraction -> Pydantic validation -> List[TestCase]

Like the Requirement Agent, this agent depends only on the `LLMClient`
interface from core/llm_client.py — never on an LLM provider SDK directly —
and it never executes anything the LLM returns. This step does not generate
or execute Playwright code; it only produces validated TestCase objects for
Step 6 to consume.
"""

import json
import re

from pydantic import ValidationError

from core.config import PROJECT_ROOT
from core.llm_client import LLMClient, LLMClientError, get_llm_client
from core.logger import get_logger
from core.models import RequirementAnalysis, TestCase
from core.repositories import TestCaseRepository

logger = get_logger(__name__)

PROMPT_TEMPLATE_PATH = PROJECT_ROOT / "prompts" / "test_generation_prompt.txt"
_REQUIREMENT_ANALYSIS_PLACEHOLDER = "{{REQUIREMENT_ANALYSIS_JSON}}"

_CODE_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TITLE_NORMALIZE_PATTERN = re.compile(r"[^a-z0-9]+")


class TestCaseGenerationError(Exception):
    """Raised when test cases cannot be generated or validated from a RequirementAnalysis."""


def _load_prompt_template() -> str:
    try:
        return PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise TestCaseGenerationError(
            f"Could not read prompt template at {PROMPT_TEMPLATE_PATH}"
        ) from exc


def _extract_json_array(raw_text: str) -> str:
    """Best-effort extraction of a single JSON array from raw LLM text.

    LLMs sometimes wrap JSON in prose or a markdown code fence even when told
    not to. This locates the array for parsing; it never evaluates or
    executes any of the text.
    """
    text = raw_text.strip()

    fence_match = _CODE_FENCE_PATTERN.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise TestCaseGenerationError(
            "The LLM response did not contain a JSON array of test cases."
        )

    return text[start : end + 1]


def _normalize_title(title: str) -> str:
    """Normalize a title for duplicate detection (case/whitespace/punctuation-insensitive)."""
    collapsed = _TITLE_NORMALIZE_PATTERN.sub(" ", title.lower()).strip()
    return re.sub(r"\s+", " ", collapsed)


def _deduplicate_test_cases(test_cases: list[TestCase]) -> list[TestCase]:
    """Drop test cases whose title or ID duplicates one already seen, keeping the first."""
    seen_titles: set[str] = set()
    seen_ids: set[str] = set()
    deduplicated: list[TestCase] = []

    for test_case in test_cases:
        normalized_title = _normalize_title(test_case.title)
        if normalized_title in seen_titles or test_case.test_case_id in seen_ids:
            logger.warning(
                "Dropping duplicate test case: id=%s title=%r",
                test_case.test_case_id,
                test_case.title,
            )
            continue
        seen_titles.add(normalized_title)
        seen_ids.add(test_case.test_case_id)
        deduplicated.append(test_case)

    return deduplicated


class TestCaseGeneratorAgent:
    """Generates a comprehensive, validated, deduplicated set of test cases
    from a RequirementAnalysis."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        test_case_repository: TestCaseRepository | None = None,
    ) -> None:
        self._llm_client = llm_client or get_llm_client()
        self._prompt_template = _load_prompt_template()
        self._test_case_repository = test_case_repository or TestCaseRepository()

    def generate(
        self, requirement_analysis: RequirementAnalysis, requirement_id: int | None = None
    ) -> list[TestCase]:
        """Turn a validated RequirementAnalysis into a deduplicated list of TestCase objects.

        `requirement_id` (e.g. from `RequirementAgent.last_requirement_id`)
        links each persisted test case back to its requirement; it's
        optional, since a caller may not have or want that link.

        Raises:
            TestCaseGenerationError: `requirement_analysis` is missing, the
                LLM call failed, its response could not be parsed as JSON,
                contained no valid test cases, or a test case failed schema
                validation.
        """
        if not isinstance(requirement_analysis, RequirementAnalysis):
            raise TestCaseGenerationError(
                "requirement_analysis must be a validated RequirementAnalysis instance."
            )

        analysis_json = requirement_analysis.model_dump_json(indent=2)
        prompt = self._prompt_template.replace(_REQUIREMENT_ANALYSIS_PLACEHOLDER, analysis_json)

        logger.info(
            "Generating test cases for requirement: '%s'", requirement_analysis.test_name
        )

        try:
            raw_response = self._llm_client.generate(prompt)
        except LLMClientError as exc:
            logger.error("Test Case Generator: LLM call failed: %s", exc)
            raise TestCaseGenerationError(f"Failed to generate test cases: {exc}") from exc

        try:
            json_text = _extract_json_array(raw_response)
            data = json.loads(json_text)
        except TestCaseGenerationError:
            logger.error("Test Case Generator: no JSON array found in LLM response")
            raise
        except json.JSONDecodeError as exc:
            logger.error("Test Case Generator: LLM response was not valid JSON")
            raise TestCaseGenerationError(
                "The LLM response could not be parsed as JSON."
            ) from exc

        if not isinstance(data, list) or not data:
            logger.error("Test Case Generator: parsed JSON was not a non-empty array")
            raise TestCaseGenerationError(
                "The LLM response must be a non-empty JSON array of test cases."
            )

        test_cases: list[TestCase] = []
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                logger.error("Test Case Generator: item %d is not a JSON object", index)
                raise TestCaseGenerationError(f"Test case at index {index} is not a JSON object.")
            try:
                test_cases.append(TestCase.model_validate(item))
            except ValidationError as exc:
                logger.error("Test Case Generator: test case at index %d failed validation", index)
                raise TestCaseGenerationError(
                    f"Test case at index {index} did not match the expected schema: {exc}"
                ) from exc

        deduplicated = _deduplicate_test_cases(test_cases)
        if not deduplicated:
            raise TestCaseGenerationError("No valid, unique test cases were generated.")

        logger.info(
            "Generated %d test case(s) (%d duplicate(s) removed) for '%s'",
            len(deduplicated),
            len(test_cases) - len(deduplicated),
            requirement_analysis.test_name,
        )

        for test_case in deduplicated:
            try:
                self._test_case_repository.save(test_case, requirement_id=requirement_id)
            except Exception as exc:  # noqa: BLE001 - persistence must never break generation
                logger.error("Could not persist test case '%s': %s", test_case.test_case_id, exc)

        return deduplicated
