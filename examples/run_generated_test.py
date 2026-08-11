"""Example: identify a generated test, validate it, execute it via TestRunner,
and print the structured TestExecutionResult.

Run from the project root:

    .venv\\Scripts\\python.exe -m examples.run_generated_test

This targets ONE specific, known generated test file by name — it never
accepts an arbitrary caller/user-supplied path, per the execution engine's
safety rules (Step 7). Code generation itself is Step 6's concern; this
example uses a small known-good Playwright test file so it can focus on
execution.
"""

from pathlib import Path

from core.config import get_settings
from core.models import TestCase
from executor.test_runner import TestRunner
from generators.playwright_generator import PlaywrightCodeGenerator

# A single, known, hardcoded target filename — never a path supplied by a caller/user.
_TARGET_FILENAME = "test_tc_login_001.py"

_KNOWN_GOOD_CODE = '''import os

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


def _ensure_generated_test_exists() -> Path:
    """Return the path to the target generated test, writing it if it doesn't exist yet.

    This bypasses the LLM entirely (Step 6 already demonstrates that flow);
    the file still goes through PlaywrightCodeGenerator.save_code(), so it
    lands at the same deterministic, sandboxed path a real generation run
    would use.
    """
    workspace = get_settings().allowed_workspace_path
    target_path = workspace / _TARGET_FILENAME

    if target_path.exists():
        print(f"Using existing generated test: {target_path}")
        return target_path

    print(f"'{_TARGET_FILENAME}' not found yet — writing a known-good test file first.")
    test_case = TestCase(
        test_case_id="TC_LOGIN_001",
        title="Valid Login",
        description="Verify that a valid user can log in.",
        steps=["Open login page", "Enter valid username", "Enter valid password", "Click Login"],
        test_data={"username": "valid_username", "password": "valid_password"},
        expected_result="Dashboard is displayed",
        priority="high",
        type="positive",
    )
    generator = PlaywrightCodeGenerator()
    saved_path = generator.save_code(test_case, _KNOWN_GOOD_CODE)
    print(f"Saved: {saved_path}")
    return saved_path


def main() -> None:
    test_file = _ensure_generated_test_exists()

    print(f"\nExecuting '{test_file.name}' through TestRunner...")
    print("(TestRunner independently re-validates the file's safety before running it,")
    print(" regardless of how or when it was generated.)\n")

    runner = TestRunner()
    result = runner.run_test(test_file)

    print("TestExecutionResult:")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
