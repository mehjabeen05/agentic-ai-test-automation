"""FastAPI dependency providers for the API layer.

Every provider here constructs a real component (agent, executor,
repository) with its default configuration — the same components used by
the CLI examples in earlier steps. Tests override these via
`app.dependency_overrides` to inject fakes; the API layer itself never
special-cases "test mode". This is also the one place FastAPI-specific
code is allowed to appear near agent construction — the agents themselves
(agents/, executor/, generators/) have no FastAPI dependency at all.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Callable

from agents.failure_analysis_agent import FailureAnalysisAgent
from agents.requirement_agent import RequirementAgent
from agents.test_generation_agent import TestCaseGeneratorAgent
from core.repositories import (
    ExecutionRepository,
    FailureAnalysisRepository,
    HealingRepository,
    RequirementRepository,
    TestCaseRepository,
)
from executor.healing_executor import HealingExecutor
from executor.test_runner import TestRunner
from generators.playwright_generator import PlaywrightCodeGenerator


def get_requirement_agent() -> RequirementAgent:
    return RequirementAgent()


def get_test_case_generator_agent() -> TestCaseGeneratorAgent:
    return TestCaseGeneratorAgent()


def get_playwright_code_generator() -> PlaywrightCodeGenerator:
    return PlaywrightCodeGenerator()


def get_test_runner() -> TestRunner:
    return TestRunner()


def get_failure_analysis_agent() -> FailureAnalysisAgent:
    return FailureAnalysisAgent()


def get_healing_executor() -> HealingExecutor:
    return HealingExecutor()


def get_requirement_repository() -> RequirementRepository:
    return RequirementRepository()


def get_test_case_repository() -> TestCaseRepository:
    return TestCaseRepository()


def get_execution_repository() -> ExecutionRepository:
    return ExecutionRepository()


def get_failure_analysis_repository() -> FailureAnalysisRepository:
    return FailureAnalysisRepository()


def get_healing_repository() -> HealingRepository:
    return HealingRepository()


def get_page_context() -> Callable[[str | None], "contextmanager"]:
    """A factory for a short-lived Playwright page, used only by /tests/heal.

    Returns a context-manager function: `with page_context(url) as page: ...`.
    A fresh, headless Chromium browser is launched per call and closed
    afterward — this is intentionally simple (no pooling) for this step's
    scope. Overridden in tests with a fake page so healing-endpoint tests
    never need a real browser.
    """

    @contextmanager
    def _page_context(url: str | None) -> Iterator[object]:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                if url:
                    page.goto(url, wait_until="domcontentloaded")
                yield page
            finally:
                browser.close()

    return _page_context
