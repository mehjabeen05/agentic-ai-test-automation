"""Browser-backed selector validation and candidate ranking.

A candidate selector proposed by the Healing Agent is never trusted just
because the LLM suggested it — it is only considered usable once this
module confirms it resolves to exactly one element in the real, live
Playwright DOM. Validation only ever calls `Page.locator(...).count()`;
Playwright resolves the selector string through its own selector engine,
never via `eval`/`exec` or any code execution.
"""

from typing import Protocol

from core.logger import get_logger
from core.models import SelectorValidation

logger = get_logger(__name__)

# More than one match is treated as ambiguous — healing must never
# auto-select a selector that could hit the wrong element.
_UNIQUE_MATCH_COUNT = 1


class _LocatorLike(Protocol):
    def count(self) -> int: ...


class _PageLike(Protocol):
    """The only Page capability this module needs — easy to satisfy with a fake in tests."""

    def locator(self, selector: str) -> _LocatorLike: ...


def validate_selector(page: _PageLike, selector: str) -> SelectorValidation:
    """Check whether `selector` resolves to exactly one element on `page`.

    Never raises for a bad/unsupported selector string — that is itself a
    validation outcome (`valid=False`), not an error.
    """
    try:
        match_count = page.locator(selector).count()
    except Exception as exc:  # noqa: BLE001 - an unparseable selector must not crash healing
        logger.warning("Selector '%s' could not be evaluated: %s", selector, exc)
        return SelectorValidation(
            selector=selector,
            valid=False,
            match_count=0,
            reason=f"Selector could not be evaluated: {exc}",
        )

    if match_count == 0:
        logger.info("Selector '%s' matched zero elements", selector)
        return SelectorValidation(
            selector=selector, valid=False, match_count=0, reason="No matching elements."
        )

    if match_count > _UNIQUE_MATCH_COUNT:
        logger.info("Selector '%s' matched %d elements (ambiguous)", selector, match_count)
        return SelectorValidation(
            selector=selector,
            valid=False,
            match_count=match_count,
            reason=f"Ambiguous: matched {match_count} elements.",
        )

    logger.info("Selector '%s' matched exactly one element", selector)
    return SelectorValidation(
        selector=selector, valid=True, match_count=1, reason="Matched exactly one element."
    )


def _selector_stability_rank(selector: str) -> int:
    """Lower is better. Mirrors the project's documented ranking order:

    1. (handled by validate_selector: only unique matches are ever valid)
    2. Stable attributes (data-testid / data-test)
    3. Accessible role/name
    4. Simple CSS
    5. Text selectors
    6. XPath (lowest priority)
    """
    lowered = selector.lower()
    if "data-testid" in lowered or "data-test" in lowered:
        return 1
    if lowered.startswith("role=") or "[role=" in lowered:
        return 2
    if lowered.startswith(("//", "xpath=")):
        return 5
    if lowered.startswith("text=") or ":has-text(" in lowered:
        return 4
    return 3  # plain CSS


def select_best_candidate(validations: list[SelectorValidation]) -> SelectorValidation | None:
    """Pick the best candidate among already-validated results.

    Only ever considers `valid=True` entries (i.e. already confirmed to
    match exactly one element) — this function never overrides that
    browser-verified fact. Among those, it ranks by selector stability.
    Returns None if no candidate validated successfully.
    """
    valid_candidates = [validation for validation in validations if validation.valid]
    if not valid_candidates:
        return None
    return min(valid_candidates, key=lambda validation: _selector_stability_rank(validation.selector))
