"""Baseline (non-AI) Playwright + PyTest test against a public demo site.

Site: https://the-internet.herokuapp.com/login
This is a small demo app maintained specifically for automation practice,
so it is safe to script against (no anti-bot blocking, no real user data).
"""

import re

from playwright.sync_api import Page, expect

from core.logger import get_logger

logger = get_logger(__name__)

DEMO_USERNAME = "tomsmith"
DEMO_PASSWORD = "SuperSecretPassword!"


def test_valid_login(login_page: Page) -> None:
    """A user with valid credentials can log in and reach the secure area."""
    logger.info("Filling in demo login credentials")
    login_page.fill("#username", DEMO_USERNAME)
    login_page.fill("#password", DEMO_PASSWORD)
    login_page.click("button[type='submit']")

    flash_message = login_page.locator("#flash")
    expect(flash_message).to_be_visible(timeout=5_000)
    expect(flash_message).to_contain_text("You logged into a secure area!")
    expect(login_page).to_have_url(re.compile(r".*/secure$"))

    logger.info("Login succeeded, secure area reached")
