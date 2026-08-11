"""Example: TestCase -> PlaywrightCodeGenerator -> validated Python code -> tests/generated/.

Run from the project root:

    .venv\\Scripts\\python.exe -m examples.generate_playwright_test

This does NOT execute the generated test — it only generates, AST-validates,
and saves it. Running generated tests is a later step.

If a real LLM_API_KEY is configured in .env, this attempts a real LLM call.
If not (or if that call fails), it falls back to a small canned response so
the example still runs with no setup required.
"""

from core.config import get_settings
from core.llm_client import LLMClient
from core.models import TestCase
from generators.playwright_generator import PlaywrightCodeGenerator, PlaywrightGenerationError

_DEMO_CODE = '''import os

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
'''


class _DemoLLMClient(LLMClient):
    """Canned LLMClient used when no real LLM_API_KEY is configured, or the real call fails."""

    def generate(self, prompt: str) -> str:
        return _DEMO_CODE


def build_sample_test_case() -> TestCase:
    """The kind of TestCase the Test Case Generator Agent (Step 5) would produce."""
    return TestCase(
        test_case_id="TC_LOGIN_001",
        title="Valid Login",
        description="Verify that a valid user can log in.",
        preconditions=["User has valid credentials"],
        steps=["Open login page", "Enter valid username", "Enter valid password", "Click Login"],
        test_data={"username": "valid_username", "password": "valid_password"},
        expected_result="Dashboard is displayed",
        priority="high",
        type="positive",
    )


def _generate_code_with_fallback(test_case: TestCase) -> tuple[PlaywrightCodeGenerator, str]:
    """Use the real LLM if a key is configured; fall back to a demo response on any failure."""
    settings = get_settings()
    has_configured_key = bool(settings.llm_api_key.get_secret_value().strip())

    if has_configured_key:
        print("LLM_API_KEY is configured — attempting a real LLM call.")
        generator = PlaywrightCodeGenerator()
        try:
            return generator, generator.generate_code(test_case)
        except PlaywrightGenerationError as exc:
            print(f"Real LLM call failed ({exc})\nFalling back to a canned demo response.\n")

    print("Running in offline demo mode with a canned response.")
    generator = PlaywrightCodeGenerator(llm_client=_DemoLLMClient())
    return generator, generator.generate_code(test_case)


def main() -> None:
    test_case = build_sample_test_case()
    print("TestCase input:")
    print(test_case.model_dump_json(indent=2))

    generator, code = _generate_code_with_fallback(test_case)
    print("\nGenerated and AST-validated Playwright code:\n")
    print(code)

    saved_path = generator.save_code(test_case, code)
    print(f"\nSaved to: {saved_path}")
    print("(This example does not execute the generated test — that's a later step.)")


if __name__ == "__main__":
    main()
