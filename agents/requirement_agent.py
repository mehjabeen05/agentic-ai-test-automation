"""Requirement Agent: turns a natural-language requirement into a validated,
structured test-case specification.

Architecture (per the project's design):

    TestRequirement -> RequirementAgent -> LLMClient -> LLM
        -> raw text -> JSON extraction -> Pydantic validation -> RequirementAnalysis

The agent never talks to an LLM provider SDK directly — only to the
`LLMClient` interface from core/llm_client.py — and it never executes
anything the LLM returns. Raw LLM text is treated as untrusted until it has
passed Pydantic validation into a `RequirementAnalysis`.
"""

import json
import re

from pydantic import ValidationError

from core.config import PROJECT_ROOT
from core.llm_client import LLMClient, LLMClientError, get_llm_client
from core.logger import get_logger
from core.models import RequirementAnalysis, TestRequirement
from core.repositories import RequirementRepository

logger = get_logger(__name__)

PROMPT_TEMPLATE_PATH = PROJECT_ROOT / "prompts" / "requirement_prompt.txt"
_REQUIREMENT_PLACEHOLDER = "{{REQUIREMENT}}"

_CODE_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class RequirementAgentError(Exception):
    """Raised when a requirement cannot be turned into a valid RequirementAnalysis."""


def _load_prompt_template() -> str:
    try:
        return PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise RequirementAgentError(
            f"Could not read prompt template at {PROMPT_TEMPLATE_PATH}"
        ) from exc


def _extract_json_object(raw_text: str) -> str:
    """Best-effort extraction of a single JSON object from raw LLM text.

    LLMs sometimes wrap JSON in prose ("Here is your JSON:\\n{...}") or a
    markdown code fence even when told not to. This locates the JSON object
    for parsing; it never evaluates or executes any of the text.
    """
    text = raw_text.strip()

    fence_match = _CODE_FENCE_PATTERN.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise RequirementAgentError("The LLM response did not contain a JSON object.")

    return text[start : end + 1]


class RequirementAgent:
    """Converts natural-language requirements into structured, validated test specs."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        requirement_repository: RequirementRepository | None = None,
    ) -> None:
        self._llm_client = llm_client or get_llm_client()
        self._prompt_template = _load_prompt_template()
        self._requirement_repository = requirement_repository or RequirementRepository()
        # Set after a successful analyze() call, so a caller can chain this
        # id into TestCaseGeneratorAgent.generate(requirement_id=...). None
        # if nothing has been persisted yet (including on a DB failure).
        self.last_requirement_id: int | None = None

    def analyze(self, requirement: TestRequirement) -> RequirementAnalysis:
        """Turn a natural-language requirement into a validated RequirementAnalysis.

        Raises:
            RequirementAgentError: the LLM call failed, or its response could
                not be parsed as JSON or did not match the expected schema.
        """
        self.last_requirement_id = None
        prompt = self._prompt_template.replace(_REQUIREMENT_PLACEHOLDER, requirement.text)

        logger.info("Analyzing requirement (length=%d chars)", len(requirement.text))

        try:
            raw_response = self._llm_client.generate(prompt)
        except LLMClientError as exc:
            logger.error("Requirement Agent: LLM call failed: %s", exc)
            raise RequirementAgentError(f"Failed to analyze requirement: {exc}") from exc

        try:
            json_text = _extract_json_object(raw_response)
            data = json.loads(json_text)
        except RequirementAgentError:
            logger.error("Requirement Agent: no JSON object found in LLM response")
            raise
        except json.JSONDecodeError as exc:
            logger.error("Requirement Agent: LLM response was not valid JSON")
            raise RequirementAgentError("The LLM response could not be parsed as JSON.") from exc

        if not isinstance(data, dict):
            logger.error("Requirement Agent: parsed JSON was not an object")
            raise RequirementAgentError("The LLM response JSON must be an object.")

        try:
            analysis = RequirementAnalysis.model_validate(data)
        except ValidationError as exc:
            logger.error("Requirement Agent: LLM response failed schema validation")
            raise RequirementAgentError(
                f"The LLM response did not match the expected schema: {exc}"
            ) from exc

        logger.info(
            "Requirement analyzed successfully: '%s' (priority=%s, %d step(s))",
            analysis.test_name,
            analysis.priority.value,
            len(analysis.steps),
        )

        try:
            self.last_requirement_id = self._requirement_repository.save(requirement.text, analysis)
        except Exception as exc:  # noqa: BLE001 - persistence must never break the agent
            logger.error("Could not persist requirement: %s", exc)

        return analysis
