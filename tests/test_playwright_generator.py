"""Unit tests for the Playwright Code Generator and its AST-based code validator.

The generator is driven through a fake LLMClient, so these tests never
contact a real LLM provider and pass with or without a configured
LLM_API_KEY. Generated code is never executed here — only parsed via `ast`
(inside generators/code_validator.py) and, in the save tests, written to a
pytest tmp_path (never the real tests/generated/ directory).
"""

from pathlib import Path

import pytest

from core.llm_client import LLMClient, LLMRequestError
from core.models import TestCase
from generators.code_validator import CodeValidationError, validate_generated_code
from generators.playwright_generator import PlaywrightCodeGenerator, PlaywrightGenerationError

VALID_CODE = '''import os

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

INVALID_SYNTAX_CODE = "def test_broken(page: Page:\n    page.goto('https://example.com')"

SUBPROCESS_CODE = '''
import subprocess
from playwright.sync_api import Page

def test_something(page: Page):
    subprocess.run(["ls"])
'''

EVAL_CODE = '''
from playwright.sync_api import Page

def test_something(page: Page):
    eval("1 + 1")
'''

EXEC_CODE = '''
from playwright.sync_api import Page

def test_something(page: Page):
    exec("print(1)")
'''

OS_SYSTEM_CODE = '''
import os
from playwright.sync_api import Page

def test_something(page: Page):
    os.system("rm -rf /")
'''

MISSING_TEST_FUNCTION_CODE = '''
from playwright.sync_api import Page

def helper(page: Page):
    page.goto("https://example.com")
'''

HARDCODED_PASSWORD_VARIABLE_CODE = '''
from playwright.sync_api import Page

PASSWORD = "SuperSecret123!"

def test_login(page: Page):
    page.fill("#password", PASSWORD)
'''

HARDCODED_PASSWORD_INLINE_CODE = '''
from playwright.sync_api import Page

def test_login(page: Page):
    page.fill("#password", "SuperSecret123!")
'''


class FakeLLMClient(LLMClient):
    """A stand-in LLMClient that returns a canned response instead of calling a provider."""

    def __init__(self, response: str | None = None, error: Exception | None = None) -> None:
        self._response = response
        self._error = error
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


class _StubSettings:
    """Minimal stand-in for core.config.Settings, pointing the workspace at a tmp dir."""

    def __init__(self, workspace: Path) -> None:
        self.allowed_workspace_path = workspace


def make_test_case(**overrides: object) -> TestCase:
    payload = {
        "test_case_id": "TC_LOGIN_001",
        "title": "Valid Login",
        "description": "Verify that a valid user can log in.",
        "preconditions": ["User has valid credentials"],
        "steps": ["Open login page", "Enter valid username", "Enter valid password", "Click Login"],
        "test_data": {"username": "valid_username", "password": "valid_password"},
        "expected_result": "Dashboard is displayed",
        "priority": "high",
        "type": "positive",
    }
    payload.update(overrides)
    return TestCase.model_validate(payload)


# --- Code validator: direct unit tests (A validated below via the full generator flow) ---


def test_validate_accepts_valid_playwright_code():
    result = validate_generated_code(VALID_CODE)
    assert result.is_valid
    assert result.issues == []


def test_validate_rejects_invalid_syntax():
    result = validate_generated_code(INVALID_SYNTAX_CODE)
    assert not result.is_valid
    assert any(issue.code == "syntax_error" for issue in result.issues)


def test_validate_rejects_subprocess():
    result = validate_generated_code(SUBPROCESS_CODE)
    assert not result.is_valid
    assert any(issue.code == "forbidden_import" for issue in result.issues)


def test_validate_rejects_eval():
    result = validate_generated_code(EVAL_CODE)
    assert not result.is_valid
    assert any(issue.code == "forbidden_call" for issue in result.issues)


def test_validate_rejects_exec():
    result = validate_generated_code(EXEC_CODE)
    assert not result.is_valid
    assert any(issue.code == "forbidden_call" for issue in result.issues)


def test_validate_rejects_os_system():
    result = validate_generated_code(OS_SYSTEM_CODE)
    assert not result.is_valid
    assert any(issue.code == "forbidden_call" for issue in result.issues)


def test_validate_rejects_missing_test_function():
    result = validate_generated_code(MISSING_TEST_FUNCTION_CODE)
    assert not result.is_valid
    assert any(issue.code == "missing_test_function" for issue in result.issues)


def test_validate_rejects_hardcoded_password_in_variable():
    result = validate_generated_code(HARDCODED_PASSWORD_VARIABLE_CODE)
    assert not result.is_valid
    assert any(issue.code == "hardcoded_credential" for issue in result.issues)


def test_validate_rejects_hardcoded_password_inline():
    result = validate_generated_code(HARDCODED_PASSWORD_INLINE_CODE)
    assert not result.is_valid
    assert any(issue.code == "hardcoded_credential" for issue in result.issues)


# --- A. Valid TestCase -> generator flow accepts the code ---


def test_generate_code_returns_valid_code_for_valid_test_case():
    fake_client = FakeLLMClient(response=VALID_CODE)
    generator = PlaywrightCodeGenerator(llm_client=fake_client)

    code = generator.generate_code(make_test_case())

    assert "def test_" in code
    assert "TC_LOGIN_001" in fake_client.last_prompt


def test_generate_code_strips_markdown_fence():
    fenced = f"```python\n{VALID_CODE}\n```"
    generator = PlaywrightCodeGenerator(llm_client=FakeLLMClient(response=fenced))

    code = generator.generate_code(make_test_case())

    assert "def test_" in code
    assert "```" not in code


def test_generate_code_raises_code_validation_error_for_dangerous_code():
    generator = PlaywrightCodeGenerator(llm_client=FakeLLMClient(response=OS_SYSTEM_CODE))

    with pytest.raises(CodeValidationError):
        generator.generate_code(make_test_case())


def test_generate_code_wraps_llm_client_errors():
    generator = PlaywrightCodeGenerator(llm_client=FakeLLMClient(error=LLMRequestError("provider is down")))

    with pytest.raises(PlaywrightGenerationError):
        generator.generate_code(make_test_case())


def test_generate_code_rejects_non_test_case_input():
    generator = PlaywrightCodeGenerator(llm_client=FakeLLMClient(response=VALID_CODE))

    with pytest.raises(PlaywrightGenerationError):
        generator.generate_code(None)


def test_generate_code_raises_when_response_has_no_code():
    generator = PlaywrightCodeGenerator(llm_client=FakeLLMClient(response="   "))

    with pytest.raises(PlaywrightGenerationError):
        generator.generate_code(make_test_case())


# --- File saving: deterministic filename, no blind overwrite, sandboxed workspace ---


def test_save_code_writes_deterministic_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "generators.playwright_generator.get_settings", lambda: _StubSettings(tmp_path)
    )
    generator = PlaywrightCodeGenerator(llm_client=FakeLLMClient(response=VALID_CODE))

    saved_path = generator.save_code(make_test_case(), VALID_CODE)

    assert saved_path == tmp_path / "test_tc_login_001.py"
    assert saved_path.read_text(encoding="utf-8") == VALID_CODE


def test_save_code_versions_instead_of_overwriting(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "generators.playwright_generator.get_settings", lambda: _StubSettings(tmp_path)
    )
    generator = PlaywrightCodeGenerator(llm_client=FakeLLMClient(response=VALID_CODE))
    test_case = make_test_case()

    first_path = generator.save_code(test_case, VALID_CODE)
    first_path.write_text("ORIGINAL CONTENT - MUST NOT BE OVERWRITTEN", encoding="utf-8")

    second_path = generator.save_code(test_case, VALID_CODE)

    assert second_path != first_path
    assert second_path.name == "test_tc_login_001_v2.py"
    assert first_path.read_text(encoding="utf-8") == "ORIGINAL CONTENT - MUST NOT BE OVERWRITTEN"


def test_generate_and_save_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "generators.playwright_generator.get_settings", lambda: _StubSettings(tmp_path)
    )
    generator = PlaywrightCodeGenerator(llm_client=FakeLLMClient(response=VALID_CODE))

    saved_path = generator.generate_and_save(make_test_case())

    assert saved_path.exists()
    assert "def test_" in saved_path.read_text(encoding="utf-8")
