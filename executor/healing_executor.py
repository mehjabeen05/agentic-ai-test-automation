"""Controlled, runtime-only selector self-healing.

Architecture (per the project's design):

    Failed action (selector, action name) + FailureAnalysis
        -> HealingExecutor -> SelectorHealingAgent (candidates)
        -> agents.selector_validator (browser DOM validation, per candidate)
        -> best valid candidate -> retry the ONE failed action -> HealingResult

Hard safety rules enforced here:
    - Healing is only ATTEMPTED when `agents.healing_agent.is_selector_related`
      says the failure category is plausibly a broken selector — never for
      authentication, environment, navigation, test-data, or unrelated
      assertion failures.
    - The LLM only ever supplies selector STRINGS (enforced by the
      `HealingSuggestion` model). The actual browser action performed on
      retry — `retry_action` — is a plain Python callable supplied by the
      *caller*, never derived from LLM output, so no LLM-generated Python
      or JavaScript is ever executed. A selector string is data resolved by
      Playwright's own selector engine, not executable code.
    - At most ONE healing attempt per failure (MAX_HEALING_ATTEMPTS = 1):
      exactly one candidate is chosen and exactly one retry is performed —
      never a loop over multiple candidates.
    - Nothing here writes to the original generated test file, or to any
      file outside reports/healing/. The original source is never rewritten.
"""

import datetime as dt
from typing import Callable

from agents.healing_agent import HealingAgentError, SelectorHealingAgent, is_selector_related
from agents.selector_validator import select_best_candidate, validate_selector
from core.config import get_settings
from core.logger import get_logger
from core.models import (
    ExecutionStatus,
    FailureAnalysis,
    HealingResult,
    HealingStatus,
    SelectorValidation,
    TestExecutionResult,
)
from core.repositories import HealingRepository

logger = get_logger(__name__)

MAX_HEALING_ATTEMPTS = 1


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _file_safe_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%f")


class HealingExecutor:
    """Attempts one controlled, runtime-only selector-healing retry for a failed action."""

    def __init__(
        self,
        healing_agent: SelectorHealingAgent | None = None,
        healing_repository: HealingRepository | None = None,
    ) -> None:
        self._healing_agent = healing_agent or SelectorHealingAgent()
        self._reports_dir = get_settings().reports_path / "healing"
        self._healing_repository = healing_repository or HealingRepository()

    def attempt_heal(
        self,
        *,
        test_id: str,
        page: object,
        original_selector: str,
        failed_action: str,
        retry_action: Callable[[object, str], None],
        failure: FailureAnalysis,
        dom_snippet: str | None = None,
        execution_id: int | None = None,
    ) -> HealingResult:
        """Attempt exactly one controlled healing retry.

        `retry_action(page, candidate_selector)` must perform ONLY the same
        browser action that originally failed (e.g.
        `lambda page, selector: page.click(selector)`). It is supplied by
        the caller — never derived from LLM output.

        `execution_id` (e.g. from `TestRunner.last_execution_id`) links the
        persisted healing record back to its execution row; it's optional
        — when omitted, the record is still returned but not persisted to
        the database (only to reports/healing/), since
        `healing_attempts.execution_id` is required.

        Always returns a HealingResult; only a genuinely invalid call (a
        bad `failure` argument) raises.
        """
        if not isinstance(failure, FailureAnalysis):
            raise ValueError("failure must be a validated FailureAnalysis instance.")

        if not is_selector_related(failure):
            logger.info(
                "Healing skipped for '%s': failure_type=%s is not selector-related",
                test_id,
                failure.failure_type.value,
            )
            return self._finalize(
                execution_id,
                HealingResult(
                    test_id=test_id,
                    original_selector=original_selector,
                    status=HealingStatus.SKIPPED,
                    reason=f"Failure type '{failure.failure_type.value}' is not selector-related.",
                    timestamp=_now_iso(),
                )
            )

        try:
            suggestion = self._healing_agent.suggest(
                original_selector=original_selector,
                failed_action=failed_action,
                failure=failure,
                dom_snippet=dom_snippet,
            )
        except HealingAgentError as exc:
            logger.error("Healing failed for '%s': could not get candidates: %s", test_id, exc)
            return self._finalize(
                execution_id,
                HealingResult(
                    test_id=test_id,
                    original_selector=original_selector,
                    status=HealingStatus.FAILED,
                    reason=f"Could not generate healing candidates: {exc}",
                    timestamp=_now_iso(),
                )
            )

        validations: list[SelectorValidation] = [
            validate_selector(page, candidate) for candidate in suggestion.candidate_selectors
        ]
        best = select_best_candidate(validations)

        if best is None:
            logger.error("Healing failed for '%s': no candidate resolved to one element", test_id)
            return self._finalize(
                execution_id,
                HealingResult(
                    test_id=test_id,
                    original_selector=original_selector,
                    candidate_selectors=suggestion.candidate_selectors,
                    validations=validations,
                    status=HealingStatus.FAILED,
                    confidence=suggestion.confidence,
                    reason="No candidate selector resolved to exactly one element.",
                    timestamp=_now_iso(),
                )
            )

        # Exactly one attempt: one chosen candidate, one retry. No loop.
        logger.info("Retrying failed action for '%s' with candidate '%s'", test_id, best.selector)
        try:
            retry_action(page, best.selector)
            retry_succeeded = True
        except Exception as exc:  # noqa: BLE001 - the retry itself may legitimately fail
            logger.error(
                "Retry with healed selector '%s' failed for '%s': %s", best.selector, test_id, exc
            )
            retry_succeeded = False

        return self._finalize(
            execution_id,
            HealingResult(
                test_id=test_id,
                original_selector=original_selector,
                candidate_selectors=suggestion.candidate_selectors,
                selected_selector=best.selector,
                validations=validations,
                status=HealingStatus.HEALED if retry_succeeded else HealingStatus.FAILED,
                retry_succeeded=retry_succeeded,
                confidence=suggestion.confidence,
                reason=suggestion.reason,
                timestamp=_now_iso(),
            ),
        )

    def _finalize(self, execution_id: int | None, result: HealingResult) -> HealingResult:
        """Persist `result` to both reports/healing/ and the database, then return it.

        Both are best-effort: a failure in either is logged and never
        raised, and never changes the `result` the caller already has.
        """
        self._persist_report(result)

        if execution_id is not None:
            try:
                self._healing_repository.save(execution_id, result)
            except Exception as exc:  # noqa: BLE001 - persistence must never break healing
                logger.error(
                    "Could not persist healing record to database for '%s': %s",
                    result.test_id,
                    exc,
                )

        return result

    def _persist_report(self, result: HealingResult) -> None:
        """Save the healing record as JSON under reports/healing/, regardless of outcome."""
        try:
            self._reports_dir.mkdir(parents=True, exist_ok=True)
            path = self._reports_dir / f"{result.test_id}_{_file_safe_stamp()}.json"
            path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        except OSError as exc:
            logger.error("Could not persist healing record for '%s': %s", result.test_id, exc)


def apply_healing_to_execution_result(
    original_result: TestExecutionResult, healing_result: HealingResult
) -> TestExecutionResult:
    """Combine an execution result with a healing outcome for reporting.

    Returns a NEW TestExecutionResult — never mutates `original_result`.
    Only reports "passed after healing" when the retry itself actually
    succeeded; otherwise the original result is returned unchanged, so the
    original failure is preserved rather than papered over.
    """
    if healing_result.status != HealingStatus.HEALED or not healing_result.retry_succeeded:
        return original_result

    return original_result.model_copy(
        update={
            "status": ExecutionStatus.PASSED,
            "healed": True,
            "original_selector": healing_result.original_selector,
            "healed_selector": healing_result.selected_selector,
        }
    )
