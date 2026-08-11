import os

from playwright.sync_api import Page, expect

BASE_URL = os.getenv("TEST_BASE_URL", "https://example.com")
TEST_USERNAME = os.getenv("TEST_USERNAME", "placeholder_username")
TEST_PASSWORD = os.getenv("TEST_PASSWORD", "placeholder_password")


def test_valid_login(page: Page):
    page.goto(f"{BASE_URL}/login")
    page.fill("#username", TEST_USERNAME)
    page.fill("#password", TEST_PASSWORD)
    page.click("#login-button")
    expect(page.locator("#dashboard")).to_be_visible()