"""Selector Healing Agent: proposes CANDIDATE SELECTORS ONLY for a broken
Playwright locator — never Python, never JavaScript, never arbitrary code.

Architecture (per the project's design):

    FailureAnalysis (selector-related) -> SelectorHealingAgent -> LLMClient
        -> LLM -> raw text -> JSON extraction -> Pydantic validation
        -> HealingSuggestion (candidate selectors only)

This agent only proposes; it never decides a selector is "healed." A
candidate is only usable once agents/selector_validator.py confirms it
resolves to exactly one element in the real, live browser DOM. Like every
other agent in this project, it depends only on the `LLMClient` interface —
never an LLM provider SDK directly — and never executes anything the LLM
returns.
"""

import json
import re

from pydantic import ValidationError

from core.config import PROJECT_ROOT
from core.llm_client import LLMClient, LLMClientError, get_llm_client
from core.logger import get_logger
from core.models import FailureAnalysis, FailureType, HealingSuggestion
from core.redaction import redact_secrets

logger = get_logger(__name__)

PROMPT_TEMPLATE_PATH = PROJECT_ROOT / "prompts" / "healing_prompt.txt"
_CONTEXT_PLACEHOLDER = "{{HEALING_CONTEXT_JSON}}"

_CODE_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_MAX_DOM_CHARS = 4000

# Only these FailureAnalysis categories are plausibly a broken-selector
# problem. Healing must NEVER be attempted for anything outside this set —
# a selector change cannot fix an auth, environment, navigation, test-data,
# or unrelated assertion failure.
SELECTOR_RELATED_FAILURE_TYPES = frozenset(
    {FailureType.SELECTOR_NOT_FOUND, FailureType.ELEMENT_NOT_INTERACTABLE}
)


class HealingAgentError(Exception):
    """Raised when a healing suggestion cannot be produced or fails validation."""


def is_selector_related(failure: FailureAnalysis) -> bool:
    """Whether a FailureAnalysis is plausibly a broken-selector problem.

    This is the single gate the whole healing pipeline is built around:
    callers (e.g. HealingExecutor) must check this before ever invoking
    the healing agent.
    """
    return failure.failure_type in SELECTOR_RELATED_FAILURE_TYPES


def _load_prompt_template() -> str:
    try:
        return PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise HealingAgentError(f"Could not read prompt template at {PROMPT_TEMPLATE_PATH}") from exc


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
        raise HealingAgentError("The LLM response did not contain a JSON object.")

    return text[start : end + 1]


def _truncate(text: str | None, max_chars: int = _MAX_DOM_CHARS) -> str | None:
    if text is None or len(text) <= max_chars:
        return text
    half = max_chars // 2
    omitted = len(text) - max_chars
    return f"{text[:half]}\n...[truncated {omitted} chars]...\n{text[-half:]}"


class SelectorHealingAgent:
    """Proposes candidate replacement selectors for a broken Playwright locator."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client or get_llm_client()
        self._prompt_template = _load_prompt_template()

    def suggest(
        self,
        *,
        original_selector: str,
        failed_action: str,
        failure: FailureAnalysis,
        dom_snippet: str | None = None,
    ) -> HealingSuggestion:
        """Propose candidate selectors for a broken locator.

        This method does not itself re-check `is_selector_related(failure)`
        — the caller (HealingExecutor) is responsible for that gate. It
        only ever returns selector strings, validated by `HealingSuggestion`
        (which drops anything resembling code or markup).

        Raises:
            HealingAgentError: inputs are invalid, the LLM call failed, or
                its response could not be parsed or validated.
        """
        if not original_selector or not original_selector.strip():
            raise HealingAgentError("original_selector must not be blank")
        if not isinstance(failure, FailureAnalysis):
            raise HealingAgentError("failure must be a validated FailureAnalysis instance.")

        context = {
            "original_selector": original_selector.strip(),
            "failed_action": (failed_action or "unknown").strip(),
            "failure_type": failure.failure_type.value,
            "failure_summary": failure.summary,
            "failure_root_cause": failure.root_cause,
            "dom_snippet": _truncate(redact_secrets(dom_snippet)),
        }
        context_json = json.dumps(context, indent=2)
        prompt = self._prompt_template.replace(_CONTEXT_PLACEHOLDER, context_json)

        logger.info("Requesting healing candidates for selector '%s'", original_selector)

        try:
            raw_response = self._llm_client.generate(prompt)
        except LLMClientError as exc:
            logger.error("Healing Agent: LLM call failed: %s", exc)
            raise HealingAgentError(f"Failed to generate healing candidates: {exc}") from exc

        try:
            json_text = _extract_json_object(raw_response)
            data = json.loads(json_text)
        except HealingAgentError:
            logger.error("Healing Agent: no JSON object found in LLM response")
            raise
        except json.JSONDecodeError as exc:
            logger.error("Healing Agent: LLM response was not valid JSON")
            raise HealingAgentError("The LLM response could not be parsed as JSON.") from exc

        if not isinstance(data, dict):
            logger.error("Healing Agent: parsed JSON was not an object")
            raise HealingAgentError("The LLM response JSON must be an object.")

        try:
            suggestion = HealingSuggestion.model_validate(data)
        except ValidationError as exc:
            logger.error("Healing Agent: LLM response failed schema validation")
            raise HealingAgentError(
                f"The LLM response did not match the expected schema: {exc}"
            ) from exc

        logger.info(
            "Healing Agent proposed %d candidate(s) for '%s' (confidence=%.2f)",
            len(suggestion.candidate_selectors),
            original_selector,
            suggestion.confidence,
        )
        return suggestion
