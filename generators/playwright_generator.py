"""Playwright Code Generator: turns a validated TestCase into safe, executable
Playwright Python test source, saved under tests/generated/.

Architecture (per the project's design):

    TestCase -> PlaywrightCodeGenerator -> LLMClient -> LLM
        -> raw text -> code extraction -> AST safety validation -> saved .py file

Like the other agents, this generator depends only on the `LLMClient`
interface — never on an LLM provider SDK directly. Generated code is never
executed, evaluated, or imported here; it is only parsed via the `ast`
module (see generators/code_validator.py) before being written to disk.
Execution is a later step.
"""

import re
from pathlib import Path

from core.config import PROJECT_ROOT, get_settings
from core.llm_client import LLMClient, LLMClientError, get_llm_client
from core.logger import get_logger
from core.models import TestCase
from generators.code_validator import CodeValidationError, validate_generated_code

logger = get_logger(__name__)

PROMPT_TEMPLATE_PATH = PROJECT_ROOT / "prompts" / "playwright_generation_prompt.txt"
_TEST_CASE_PLACEHOLDER = "{{TEST_CASE_JSON}}"

_CODE_FENCE_PATTERN = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_SAFE_FILENAME_PATTERN = re.compile(r"[^a-z0-9]+")


class PlaywrightGenerationError(Exception):
    """Raised when Playwright code cannot be generated or saved for a TestCase."""


def _load_prompt_template() -> str:
    try:
        return PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise PlaywrightGenerationError(
            f"Could not read prompt template at {PROMPT_TEMPLATE_PATH}"
        ) from exc


def _extract_python_code(raw_text: str) -> str:
    """Strip a markdown code fence from raw LLM text if present, else return it as-is.

    Unlike JSON extraction, Python source has no bracket delimiter to search
    for, so this only handles the fence case. Anything left over is handed
    to `ast.parse` by the validator, which is the real safety net.
    """
    text = raw_text.strip()
    fence_match = _CODE_FENCE_PATTERN.search(text)
    if fence_match:
        return fence_match.group(1).strip()
    return text


def build_generated_filename(test_case_id: str) -> str:
    """Deterministic, sanitized filename for a test case, e.g. TC_LOGIN_001 -> test_tc_login_001.py.

    Public so callers (e.g. the API layer's `/tests/run` endpoint) can
    resolve a `test_case_id` to the same filename this generator itself
    uses, without duplicating the sanitization rule.
    """
    normalized = _SAFE_FILENAME_PATTERN.sub("_", test_case_id.strip().lower()).strip("_")
    if not normalized:
        raise PlaywrightGenerationError("test_case_id produced an empty filename after sanitization.")
    return f"test_{normalized}.py"


def _resolve_versioned_path(directory: Path, filename: str) -> Path:
    """Return a free path for `filename` in `directory`, versioning on collision.

    Existing generated tests are never overwritten or deleted; a new
    `_v2`, `_v3`, ... suffix is used instead.
    """
    candidate = directory / filename
    if not candidate.exists():
        return candidate

    stem, suffix = candidate.stem, candidate.suffix
    version = 2
    while True:
        versioned = directory / f"{stem}_v{version}{suffix}"
        if not versioned.exists():
            logger.warning(
                "Generated test file %s already exists; saving as %s instead.",
                candidate.name,
                versioned.name,
            )
            return versioned
        version += 1


class PlaywrightCodeGenerator:
    """Generates and safety-validates Playwright test code from a TestCase, and saves it."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client or get_llm_client()
        self._prompt_template = _load_prompt_template()

    def generate_code(self, test_case: TestCase) -> str:
        """Generate and safety-validate Playwright Python source for a TestCase.

        The generated code is parsed and inspected via the `ast` module —
        never executed — before being returned.

        Raises:
            PlaywrightGenerationError: `test_case` is invalid, the LLM call
                failed, or its response contained no code.
            CodeValidationError: the generated code failed the AST-based
                safety checks in generators/code_validator.py.
        """
        if not isinstance(test_case, TestCase):
            raise PlaywrightGenerationError("test_case must be a validated TestCase instance.")

        test_case_json = test_case.model_dump_json(indent=2)
        prompt = self._prompt_template.replace(_TEST_CASE_PLACEHOLDER, test_case_json)

        logger.info("Generating Playwright code for test case '%s'", test_case.test_case_id)

        try:
            raw_response = self._llm_client.generate(prompt)
        except LLMClientError as exc:
            logger.error("Playwright Generator: LLM call failed: %s", exc)
            raise PlaywrightGenerationError(f"Failed to generate Playwright code: {exc}") from exc

        code = _extract_python_code(raw_response)
        if not code.strip():
            raise PlaywrightGenerationError("The LLM response did not contain any code.")

        result = validate_generated_code(code)
        if not result.is_valid:
            reasons = "; ".join(issue.message for issue in result.issues)
            logger.error(
                "Playwright Generator: code for '%s' failed validation: %s",
                test_case.test_case_id,
                reasons,
            )
            raise CodeValidationError(result)

        logger.info(
            "Generated and validated Playwright code for '%s' (%d chars)",
            test_case.test_case_id,
            len(code),
        )
        return code

    def save_code(self, test_case: TestCase, code: str) -> Path:
        """Write already-validated code under the allowed workspace directory.

        Never overwrites an existing generated test; collisions are versioned
        (`_v2`, `_v3`, ...) instead.
        """
        workspace = get_settings().allowed_workspace_path
        workspace.mkdir(parents=True, exist_ok=True)

        filename = build_generated_filename(test_case.test_case_id)
        target_path = _resolve_versioned_path(workspace, filename)
        resolved = target_path.resolve()

        # Defensive guard: build_generated_filename only ever produces [a-z0-9_].py,
        # so this should be unreachable, but never write outside the sandbox.
        if resolved.parent != workspace:
            raise PlaywrightGenerationError(
                "Refusing to write generated test outside the allowed workspace."
            )

        resolved.write_text(code, encoding="utf-8")
        logger.info("Saved generated Playwright test to %s", resolved)
        return resolved

    def generate_and_save(self, test_case: TestCase) -> Path:
        """Generate, validate, and save Playwright code for a TestCase in one call."""
        code = self.generate_code(test_case)
        return self.save_code(test_case, code)
