"""Shared Playwright + PyTest fixtures for the tests/ suite.

This is the non-AI baseline layer: a fixed demo site, a reusable
"already on the login page" fixture, and pass/fail logging.
"""

import pytest
from playwright.sync_api import Page

from core.logger import get_logger

logger = get_logger(__name__)

BASE_URL = "https://the-internet.herokuapp.com"
DEFAULT_TIMEOUT_MS = 30_000


@pytest.fixture(autouse=True)
def _isolated_database(tmp_path, monkeypatch):
    """Redirect every default-path repository to a per-test temp database.

    Every repository (and every agent/executor that constructs one with no
    explicit override) falls back to the real configured database when no
    `db_path` is given. Without this fixture, simply running the test
    suite would write real rows into data/test_automation.db. This is
    autouse so no individual test file needs to opt in — it only affects
    the *default* fallback path; any test that passes an explicit db_path
    (e.g. tests/test_database.py) is unaffected.
    """

    class _StubSettings:
        database_path = tmp_path / "test_automation.db"

    monkeypatch.setattr("core.database.get_settings", lambda: _StubSettings())


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args: dict) -> dict:
    """Extend pytest-playwright's default browser context with a fixed viewport and base URL."""
    return {
        **browser_context_args,
        "viewport": {"width": 1280, "height": 720},
        "base_url": BASE_URL,
    }


@pytest.fixture
def login_page(page: Page) -> Page:
    """A Page already navigated to the demo site's login page, ready for reuse across tests."""
    page.set_default_timeout(DEFAULT_TIMEOUT_MS)
    logger.info("Navigating to login page: %s/login", BASE_URL)
    # "domcontentloaded" is enough to interact with form fields and is far less
    # flaky than "load", which waits on every image/font/analytics request.
    page.goto("/login", wait_until="domcontentloaded")
    return page


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Log the pass/fail outcome of every test after it runs."""
    outcome = yield
    report = outcome.get_result()
    if report.when == "call":
        if report.passed:
            logger.info("TEST PASSED: %s", item.nodeid)
        elif report.failed:
            logger.error("TEST FAILED: %s", item.nodeid)
