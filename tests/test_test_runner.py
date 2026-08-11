"""Unit tests for the Test Execution Engine (TestRunner).

Deterministic by design: the "successful"/"failing" scenarios use tiny,
local, non-Playwright test files (no external website, no browser), so
these tests are fast and network-independent. All file operations happen
under pytest's tmp_path — never the real tests/generated/, reports/, or
screenshots/ directories.
"""

from pathlib import Path

import pytest

from core.models import ExecutionStatus
from executor.test_runner import TestRunner

GOOD_SIMPLE_CODE = "def test_ok():\n    assert True\n"
FAILING_SIMPLE_CODE = "def test_not_ok():\n    assert False, 'deliberately failing'\n"
SYNTAX_ERROR_CODE = "def test_broken(:\n    assert True\n"
DANGEROUS_CODE = "import os\n\n\ndef test_dangerous():\n    os.system('rm -rf /')\n"


class _StubSettings:
    """Minimal stand-in for core.config.Settings, pointing every path at a tmp dir."""

    def __init__(self, base: Path) -> None:
        self.allowed_workspace_path = base / "generated"
        self.reports_path = base / "reports"
        self.screenshots_path = base / "screenshots"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    directory = tmp_path / "generated"
    directory.mkdir()
    return directory


@pytest.fixture
def runner(tmp_path: Path, monkeypatch, workspace: Path) -> TestRunner:
    monkeypatch.setattr("executor.test_runner.get_settings", lambda: _StubSettings(tmp_path))
    return TestRunner()


def _write(workspace: Path, name: str, content: str) -> Path:
    path = workspace / name
    path.write_text(content, encoding="utf-8")
    return path


# --- A / H. Valid, approved test -> executes and reports passed ---


def test_run_test_executes_a_valid_test_and_reports_passed(runner: TestRunner, workspace: Path):
    test_file = _write(workspace, "test_ok.py", GOOD_SIMPLE_CODE)

    result = runner.run_test(test_file)

    assert result.status == ExecutionStatus.PASSED
    assert result.test_id == "test_ok"
    assert result.exit_code == 0
    assert result.error is None
    assert result.duration >= 0.0


# --- I. Failing test -> reports failed ---


def test_run_test_reports_failed_for_a_failing_test(runner: TestRunner, workspace: Path):
    test_file = _write(workspace, "test_not_ok.py", FAILING_SIMPLE_CODE)

    result = runner.run_test(test_file)

    assert result.status == ExecutionStatus.FAILED
    assert result.exit_code == 1
    assert result.error is not None
    assert "assert" in result.stdout.lower()


# --- B. Nonexistent file -> structured error ---


def test_run_test_rejects_nonexistent_file(runner: TestRunner, workspace: Path):
    missing = workspace / "test_missing.py"

    result = runner.run_test(missing)

    assert result.status == ExecutionStatus.ERROR
    assert "does not exist" in result.error.lower()


# --- C. File outside tests/generated -> rejected ---


def test_run_test_rejects_relative_traversal_outside_workspace(
    runner: TestRunner, workspace: Path, tmp_path: Path
):
    outside_file = _write(tmp_path, "test_outside.py", GOOD_SIMPLE_CODE)
    traversal_path = workspace / ".." / "test_outside.py"

    result = runner.run_test(traversal_path)

    assert result.status == ExecutionStatus.ERROR
    assert "not inside the approved" in result.error.lower()
    assert outside_file.exists()  # untouched


def test_run_test_rejects_file_in_an_unrelated_directory(runner: TestRunner, tmp_path: Path):
    unrelated_dir = tmp_path / "unrelated"
    unrelated_dir.mkdir()
    outside_file = _write(unrelated_dir, "test_outside.py", GOOD_SIMPLE_CODE)

    result = runner.run_test(outside_file)

    assert result.status == ExecutionStatus.ERROR
    assert "not inside the approved" in result.error.lower()


# --- D. Absolute path outside the project -> rejected ---


def test_run_test_rejects_absolute_path_outside_project(runner: TestRunner, tmp_path: Path):
    # An absolute path that has nothing to do with the stubbed workspace or the project.
    far_away = tmp_path.parent / f"far-away-{tmp_path.name}"
    far_away.mkdir(exist_ok=True)
    outside_file = _write(far_away, "test_far_away.py", GOOD_SIMPLE_CODE)

    result = runner.run_test(outside_file.resolve())

    assert result.status == ExecutionStatus.ERROR
    assert "not inside the approved" in result.error.lower()


# --- E. Non-Python file -> rejected ---


def test_run_test_rejects_non_python_file(runner: TestRunner, workspace: Path):
    text_file = _write(workspace, "notes.txt", "this is not python")

    result = runner.run_test(text_file)

    assert result.status == ExecutionStatus.ERROR
    assert ".py" in result.error


# --- F. Invalid generated code -> rejected before execution ---


def test_run_test_rejects_invalid_syntax_before_execution(
    runner: TestRunner, workspace: Path, monkeypatch
):
    test_file = _write(workspace, "test_broken.py", SYNTAX_ERROR_CODE)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("subprocess.run must not be called for invalid code")

    monkeypatch.setattr("executor.test_runner.subprocess.run", _fail_if_called)

    result = runner.run_test(test_file)

    assert result.status == ExecutionStatus.ERROR
    assert "validation" in result.error.lower()


# --- G. Dangerous generated code -> rejected before execution ---


def test_run_test_rejects_dangerous_code_before_execution(
    runner: TestRunner, workspace: Path, monkeypatch
):
    test_file = _write(workspace, "test_dangerous.py", DANGEROUS_CODE)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("subprocess.run must not be called for dangerous code")

    monkeypatch.setattr("executor.test_runner.subprocess.run", _fail_if_called)

    result = runner.run_test(test_file)

    assert result.status == ExecutionStatus.ERROR
    assert "validation" in result.error.lower()


# --- Metadata persistence (reports/execution/) ---


def test_run_test_persists_execution_metadata_as_json(runner: TestRunner, workspace: Path, tmp_path: Path):
    test_file = _write(workspace, "test_ok.py", GOOD_SIMPLE_CODE)

    result = runner.run_test(test_file)

    metadata_path = tmp_path / "reports" / "execution" / "test_ok.json"
    assert metadata_path.exists()
    assert result.test_id in metadata_path.read_text(encoding="utf-8")


# --- Never invokes a shell ---


def test_run_test_never_uses_shell_true(runner: TestRunner, workspace: Path, monkeypatch):
    test_file = _write(workspace, "test_ok.py", GOOD_SIMPLE_CODE)
    captured_kwargs = {}
    original_run = __import__("subprocess").run

    def _capturing_run(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return original_run(*args, **kwargs)

    monkeypatch.setattr("executor.test_runner.subprocess.run", _capturing_run)

    runner.run_test(test_file)

    assert captured_kwargs.get("shell") is False
