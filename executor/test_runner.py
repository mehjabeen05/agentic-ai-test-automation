"""Test Execution Engine: safely runs validated, AI-generated Playwright
tests through pytest, in a subprocess, with no shell involved.

Architecture (per the project's design):

    Generated test file -> TestRunner -> path safety checks -> AST validation
        -> pytest subprocess (argument list, shell=False) -> TestExecutionResult

Security model — a file is only ever handed to pytest if it:
    1. Resolves to a direct child of the approved generated-test directory
       (tests/generated/); path traversal and absolute paths elsewhere are
       rejected by comparing the *resolved* parent directory, not by string
       matching.
    2. Has a .py extension.
    3. Exists and is a regular file.
    4. Passes the Step 6 AST safety validator (generators/code_validator.py).

TestRunner never builds a shell command string, never sets shell=True, and
never calls an LLM — it is intentionally independent of core/llm_client.py
and every agent/generator module except the code validator.
"""

import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from core.config import PROJECT_ROOT, get_settings
from core.logger import get_logger
from core.models import ExecutionStatus, TestExecutionResult
from core.repositories import ExecutionRepository
from generators.code_validator import validate_generated_code

logger = get_logger(__name__)

_EXECUTION_TIMEOUT_SECONDS = 120
_SCREENSHOT_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class TestRunnerError(Exception):
    """Raised internally for a rejected file; always caught and turned into
    an ERROR TestExecutionResult rather than propagating to the caller."""


class TestRunner:
    """Executes a single, pre-validated generated test file via pytest in a subprocess."""

    def __init__(self, execution_repository: ExecutionRepository | None = None) -> None:
        settings = get_settings()
        self._workspace = settings.allowed_workspace_path
        self._reports_dir = settings.reports_path / "execution"
        self._screenshots_dir = settings.screenshots_path
        self._execution_repository = execution_repository or ExecutionRepository()
        # Set after every run_test() call (including a rejected one), so a
        # caller can chain this id into FailureAnalysisAgent.analyze() and
        # HealingExecutor.attempt_heal(). None only if persistence itself failed.
        self.last_execution_id: int | None = None

    def run_test(self, test_file: Path) -> TestExecutionResult:
        """Validate and safely execute one generated test file.

        This never raises for an untrusted/invalid input — a rejected file
        (wrong location, wrong extension, missing, or unsafe code) always
        comes back as a TestExecutionResult with status=ERROR and a clear
        `error` message, never a bare exception.
        """
        raw_test_id = Path(test_file).name
        logger.info("Execution requested for '%s'", raw_test_id)

        try:
            resolved_path = self._resolve_and_check(test_file)
        except TestRunnerError as exc:
            logger.error("Execution rejected for '%s': %s", raw_test_id, exc)
            result = TestExecutionResult(
                test_id=Path(test_file).stem or "unknown",
                status=ExecutionStatus.ERROR,
                duration=0.0,
                error=str(exc),
            )
            self._record_result(result)
            return result

        test_id = resolved_path.stem

        try:
            source = resolved_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.error("Could not read '%s': %s", resolved_path, exc)
            result = TestExecutionResult(
                test_id=test_id,
                status=ExecutionStatus.ERROR,
                duration=0.0,
                error=f"Could not read test file: {exc}",
            )
            self._record_result(result)
            return result

        validation = validate_generated_code(source)
        if not validation.is_valid:
            reasons = "; ".join(issue.message for issue in validation.issues)
            logger.error("Execution rejected for '%s': failed validation: %s", test_id, reasons)
            result = TestExecutionResult(
                test_id=test_id,
                status=ExecutionStatus.ERROR,
                duration=0.0,
                error=f"Generated code failed safety validation: {reasons}",
            )
            self._record_result(result)
            return result

        logger.info("Validation passed for '%s'; proceeding to execution", test_id)
        result = self._execute(resolved_path, test_id)
        self._record_result(result)
        return result

    def _resolve_and_check(self, test_file: Path) -> Path:
        """Resolve `test_file` and enforce every execution safety rule.

        Raises TestRunnerError with a clear reason if any rule fails. The
        containment check happens before any filesystem existence check, so
        a path outside the sandbox is rejected without probing whether it
        exists.
        """
        try:
            resolved = Path(test_file).resolve()
        except OSError as exc:
            raise TestRunnerError(f"Could not resolve path '{test_file}': {exc}") from exc

        if resolved.parent != self._workspace:
            raise TestRunnerError(
                f"'{test_file}' is not inside the approved generated-test "
                f"directory ({self._workspace})."
            )
        if resolved.suffix != ".py":
            raise TestRunnerError(f"'{test_file}' is not a .py file.")
        if not resolved.exists():
            raise TestRunnerError(f"Test file does not exist: {resolved}")
        if not resolved.is_file():
            raise TestRunnerError(f"Not a regular file: {resolved}")

        return resolved

    def _execute(self, resolved_path: Path, test_id: str) -> TestExecutionResult:
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        junit_path = self._reports_dir / f"{test_id}.xml"

        # Fixed, hardcoded argument list — never a shell string, never
        # user-supplied. shell=False (the default) is used explicitly below.
        command = [sys.executable, "-m", "pytest", str(resolved_path), f"--junitxml={junit_path}"]

        logger.info("Executing '%s' via pytest subprocess (shell=False)", test_id)
        started_at = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                shell=False,
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                timeout=_EXECUTION_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - started_at
            logger.error("Test '%s' timed out after %.2fs", test_id, duration)
            return TestExecutionResult(
                test_id=test_id,
                status=ExecutionStatus.ERROR,
                duration=duration,
                stdout=(exc.stdout or ""),
                stderr=(exc.stderr or ""),
                error=f"Execution timed out after {_EXECUTION_TIMEOUT_SECONDS}s.",
            )
        except OSError as exc:
            duration = time.monotonic() - started_at
            logger.error("Failed to launch pytest subprocess for '%s': %s", test_id, exc)
            return TestExecutionResult(
                test_id=test_id,
                status=ExecutionStatus.ERROR,
                duration=duration,
                error=f"Failed to launch test execution: {exc}",
            )

        duration = time.monotonic() - started_at
        status = self._status_from_junit(junit_path) or self._status_from_exit_code(
            completed.returncode
        )
        screenshot = self._find_screenshot(test_id)

        logger.info(
            "Test '%s' finished: status=%s duration=%.2fs exit_code=%d",
            test_id,
            status.value,
            duration,
            completed.returncode,
        )

        return TestExecutionResult(
            test_id=test_id,
            status=status,
            duration=duration,
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            error=None if status == ExecutionStatus.PASSED else _summarize_failure(completed.stdout),
            screenshot=str(screenshot) if screenshot else None,
        )

    @staticmethod
    def _status_from_exit_code(exit_code: int) -> ExecutionStatus:
        """Fallback used only if the JUnit report is missing or unparseable.

        pytest exit codes: 0 = all collected tests passed (or were
        skipped), 1 = at least one test failed. Anything else (usage error,
        internal error, no tests collected) is treated as ERROR.
        """
        if exit_code == 0:
            return ExecutionStatus.PASSED
        if exit_code == 1:
            return ExecutionStatus.FAILED
        return ExecutionStatus.ERROR

    @staticmethod
    def _status_from_junit(junit_path: Path) -> ExecutionStatus | None:
        """Determine status from pytest's own JUnit XML report, when available.

        More precise than the exit code alone (it distinguishes `skipped`),
        and is just XML parsing — no code from the report is ever executed.
        """
        if not junit_path.exists():
            return None
        try:
            tree = ET.parse(junit_path)
        except ET.ParseError:
            return None

        severity = {
            ExecutionStatus.ERROR: 3,
            ExecutionStatus.FAILED: 2,
            ExecutionStatus.SKIPPED: 1,
            ExecutionStatus.PASSED: 0,
        }
        statuses: list[ExecutionStatus] = []
        for testcase in tree.getroot().iter("testcase"):
            if testcase.find("error") is not None:
                statuses.append(ExecutionStatus.ERROR)
            elif testcase.find("failure") is not None:
                statuses.append(ExecutionStatus.FAILED)
            elif testcase.find("skipped") is not None:
                statuses.append(ExecutionStatus.SKIPPED)
            else:
                statuses.append(ExecutionStatus.PASSED)

        if not statuses:
            return None
        return max(statuses, key=lambda status: severity[status])

    def _find_screenshot(self, test_id: str) -> Path | None:
        """Best-effort lookup of a failure screenshot pytest-playwright may have saved.

        pytest-playwright sanitizes the node ID into a directory name,
        replacing non-alphanumeric characters (including underscores) with
        hyphens, so the search matches either separator.
        """
        if not self._screenshots_dir.exists():
            return None

        parts = [re.escape(part) for part in test_id.split("_") if part]
        if not parts:
            return None
        pattern = re.compile("[-_]".join(parts), re.IGNORECASE)

        candidates = [
            path
            for path in self._screenshots_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in _SCREENSHOT_EXTENSIONS
            and pattern.search(path.as_posix())
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def _record_result(self, result: TestExecutionResult) -> None:
        """Persist `result` to both the JSON report file and the database.

        Both are best-effort: a failure in either is logged and never
        raised, and never changes the `result` the caller already has —
        `run_test()` always returns the real execution outcome regardless
        of whether it could be recorded.
        """
        self._persist_metadata(result)

        self.last_execution_id = None
        try:
            self.last_execution_id = self._execution_repository.save(result)
        except Exception as exc:  # noqa: BLE001 - persistence must never break execution
            logger.error("Could not persist execution to database for '%s': %s", result.test_id, exc)

    def _persist_metadata(self, result: TestExecutionResult) -> None:
        """Save the structured result as JSON under reports/execution/ for later steps."""
        try:
            self._reports_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = self._reports_dir / f"{result.test_id}.json"
            metadata_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        except OSError as exc:
            # Metadata persistence is best-effort; never let it mask the real result.
            logger.error("Could not persist execution metadata for '%s': %s", result.test_id, exc)


def _summarize_failure(stdout: str) -> str | None:
    """A short, human-readable excerpt of pytest's own output for a non-passed result."""
    tail = [line for line in stdout.strip().splitlines() if line.strip()][-8:]
    return "\n".join(tail) if tail else None
