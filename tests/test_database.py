"""Unit tests for the SQLite persistence layer (core/database.py, core/repositories.py).

Every test here uses an explicit temporary database path under pytest's
tmp_path — never the real data/test_automation.db. (The autouse
`_isolated_database` fixture in tests/conftest.py protects every test in
the suite from touching the real database by default; the explicit paths
here are additional, deliberate clarity for a module that is *about* the
database.)
"""

import sqlite3
from pathlib import Path

import pytest

from core.database import connect, from_json, initialize_database, to_json
from core.models import (
    ExecutionStatus,
    FailureAnalysis,
    FailureType,
    HealingResult,
    HealingStatus,
    SelectorValidation,
    TestCase,
    TestExecutionResult,
)
from core.repositories import (
    ExecutionRepository,
    FailureAnalysisRepository,
    HealingRepository,
    RequirementRepository,
    TestCaseRepository,
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_automation.db"


def make_test_case(**overrides: object) -> TestCase:
    payload = {
        "test_case_id": "TC_LOGIN_001",
        "title": "Valid Login",
        "description": "Verify that a valid user can log in.",
        "steps": ["Open login page", "Enter valid username", "Enter valid password", "Click Login"],
        "test_data": {"username": "valid_username", "password": "valid_password"},
        "expected_result": "Dashboard is displayed",
        "priority": "high",
        "type": "positive",
    }
    payload.update(overrides)
    return TestCase(**payload)


def make_execution_result(**overrides: object) -> TestExecutionResult:
    payload = {
        "test_id": "test_tc_login_001",
        "status": ExecutionStatus.FAILED,
        "duration": 12.5,
        "error": 'TimeoutError: waiting for locator("#username")',
    }
    payload.update(overrides)
    return TestExecutionResult(**payload)


def make_failure_analysis(**overrides: object) -> FailureAnalysis:
    payload = {
        "failure_type": FailureType.SELECTOR_NOT_FOUND,
        "summary": "Login button was not found.",
        "root_cause": "The selector no longer matches the DOM.",
        "suggested_fix": "Use a stable selector.",
        "confidence": 0.92,
        "is_likely_environment_issue": False,
        "is_likely_test_issue": True,
    }
    payload.update(overrides)
    return FailureAnalysis(**payload)


def make_healing_result(**overrides: object) -> HealingResult:
    payload = {
        "test_id": "test_tc_login_001",
        "original_selector": "#login-button",
        "candidate_selectors": ["button.login", "[data-testid='login']"],
        "selected_selector": "button.login",
        "status": HealingStatus.HEALED,
        "retry_succeeded": True,
        "confidence": 0.91,
        "reason": "Original selector was not found.",
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
    payload.update(overrides)
    return HealingResult(**payload)


# --- A. Database initializes correctly ---


def test_database_initializes_correctly(db_path):
    returned_path = initialize_database(db_path)
    assert returned_path == db_path
    assert db_path.exists()


def test_database_initialization_creates_parent_directory(tmp_path):
    nested_path = tmp_path / "nested" / "dir" / "test_automation.db"
    initialize_database(nested_path)
    assert nested_path.exists()


# --- B. Tables exist ---


def test_all_tables_exist(db_path):
    initialize_database(db_path)
    with connect(db_path) as connection:
        rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    table_names = {row["name"] for row in rows}
    assert {
        "requirements",
        "test_cases",
        "test_executions",
        "failure_analyses",
        "healing_attempts",
    }.issubset(table_names)


def test_foreign_keys_are_enabled_on_every_connection(db_path):
    with connect(db_path) as connection:
        result = connection.execute("PRAGMA foreign_keys").fetchone()
    assert result[0] == 1


# --- C. Insert requirement / D. Retrieve requirement ---


def test_insert_and_retrieve_requirement(db_path):
    repo = RequirementRepository(db_path)
    requirement_id = repo.save("User should be able to login with valid credentials.")

    record = repo.get_by_id(requirement_id)

    assert record is not None
    assert record.id == requirement_id
    assert record.requirement_text == "User should be able to login with valid credentials."
    assert record.created_at


def test_get_all_requirements_returns_every_saved_requirement(db_path):
    repo = RequirementRepository(db_path)
    repo.save("Requirement one")
    repo.save("Requirement two")

    records = repo.get_all()

    assert [record.requirement_text for record in records] == ["Requirement one", "Requirement two"]


# --- E. Insert test case / F. Retrieve test cases by requirement ---


def test_insert_test_case_and_retrieve_by_requirement(db_path):
    requirement_id = RequirementRepository(db_path).save("User should be able to login.")
    tc_repo = TestCaseRepository(db_path)
    tc_repo.save(make_test_case(), requirement_id=requirement_id)
    tc_repo.save(
        make_test_case(test_case_id="TC_LOGIN_002", title="Invalid Password", type="negative"),
        requirement_id=requirement_id,
    )

    records = tc_repo.get_by_requirement(requirement_id)

    assert len(records) == 2
    assert {record.test_case_id for record in records} == {"TC_LOGIN_001", "TC_LOGIN_002"}
    assert records[0].test_data == {"username": "valid_username", "password": "valid_password"}


def test_get_test_case_by_test_case_id(db_path):
    TestCaseRepository(db_path).save(make_test_case())

    record = TestCaseRepository(db_path).get_by_test_case_id("TC_LOGIN_001")

    assert record is not None
    assert record.title == "Valid Login"


def test_save_test_case_upserts_instead_of_duplicating(db_path):
    tc_repo = TestCaseRepository(db_path)
    tc_repo.save(make_test_case(title="Original Title"))
    tc_repo.save(make_test_case(title="Updated Title"))

    record = tc_repo.get_by_test_case_id("TC_LOGIN_001")

    assert record.title == "Updated Title"
    assert len(tc_repo.get_by_requirement(1)) == 0  # no requirement_id was ever given


# --- G. Insert execution / H. Retrieve execution history ---


def test_insert_execution_and_retrieve_history(db_path):
    repo = ExecutionRepository(db_path)
    repo.save(make_execution_result(status=ExecutionStatus.FAILED))
    repo.save(make_execution_result(status=ExecutionStatus.PASSED))

    history = repo.get_history("test_tc_login_001")

    assert len(history) == 2
    assert [record.status for record in history] == [ExecutionStatus.FAILED, ExecutionStatus.PASSED]


def test_get_latest_execution_returns_the_most_recent_one(db_path):
    repo = ExecutionRepository(db_path)
    repo.save(make_execution_result(status=ExecutionStatus.FAILED))
    repo.save(make_execution_result(status=ExecutionStatus.PASSED))

    latest = repo.get_latest("test_tc_login_001")

    assert latest.status == ExecutionStatus.PASSED


def test_get_failed_executions_includes_failed_and_error_statuses(db_path):
    repo = ExecutionRepository(db_path)
    repo.save(make_execution_result(status=ExecutionStatus.PASSED))
    repo.save(make_execution_result(test_id="test_tc_login_002", status=ExecutionStatus.FAILED))
    repo.save(make_execution_result(test_id="test_tc_login_003", status=ExecutionStatus.ERROR))

    failed = repo.get_failed()

    assert {record.test_case_id for record in failed} == {"test_tc_login_002", "test_tc_login_003"}


def test_get_healed_executions(db_path):
    repo = ExecutionRepository(db_path)
    repo.save(make_execution_result(healed=False))
    repo.save(
        make_execution_result(
            test_id="test_tc_login_002",
            status=ExecutionStatus.PASSED,
            healed=True,
            original_selector="#login-button",
            healed_selector="button.login",
        )
    )

    healed = repo.get_healed()

    assert len(healed) == 1
    assert healed[0].test_case_id == "test_tc_login_002"
    assert healed[0].healed is True


# --- I. Insert failure analysis / J. Retrieve failure analysis ---


def test_insert_and_retrieve_failure_analysis(db_path):
    execution_id = ExecutionRepository(db_path).save(make_execution_result())
    fa_repo = FailureAnalysisRepository(db_path)
    fa_repo.save(execution_id, make_failure_analysis())

    records = fa_repo.get_for_execution(execution_id)

    assert len(records) == 1
    assert records[0].failure_type == FailureType.SELECTOR_NOT_FOUND
    assert records[0].confidence == 0.92


def test_get_all_failure_analyses(db_path):
    execution_id = ExecutionRepository(db_path).save(make_execution_result())
    FailureAnalysisRepository(db_path).save(execution_id, make_failure_analysis())

    records = FailureAnalysisRepository(db_path).get_all()

    assert len(records) == 1


# --- K. Insert healing attempt / L. Retrieve healing history ---


def test_insert_and_retrieve_healing_attempt(db_path):
    execution_id = ExecutionRepository(db_path).save(make_execution_result())
    heal_repo = HealingRepository(db_path)
    heal_repo.save(execution_id, make_healing_result())

    records = heal_repo.get_for_execution(execution_id)

    assert len(records) == 1
    assert records[0].original_selector == "#login-button"
    assert records[0].selected_selector == "button.login"
    assert records[0].candidate_selectors == ["button.login", "[data-testid='login']"]
    assert records[0].retry_succeeded is True


def test_get_healing_history(db_path):
    execution_id = ExecutionRepository(db_path).save(make_execution_result())
    HealingRepository(db_path).save(execution_id, make_healing_result())

    history = HealingRepository(db_path).get_history()

    assert len(history) == 1


# --- M. Foreign key relationships work ---


def test_deleting_requirement_cascades_to_test_cases(db_path):
    requirement_id = RequirementRepository(db_path).save("Some requirement")
    tc_repo = TestCaseRepository(db_path)
    tc_repo.save(make_test_case(), requirement_id=requirement_id)

    with connect(db_path) as connection:
        connection.execute("DELETE FROM requirements WHERE id = ?", (requirement_id,))
        connection.commit()

    assert tc_repo.get_by_requirement(requirement_id) == []


def test_deleting_execution_cascades_to_failure_analyses_and_healing_attempts(db_path):
    execution_id = ExecutionRepository(db_path).save(make_execution_result())
    fa_repo = FailureAnalysisRepository(db_path)
    fa_repo.save(execution_id, make_failure_analysis())
    heal_repo = HealingRepository(db_path)
    heal_repo.save(execution_id, make_healing_result())

    with connect(db_path) as connection:
        connection.execute("DELETE FROM test_executions WHERE id = ?", (execution_id,))
        connection.commit()

    assert fa_repo.get_for_execution(execution_id) == []
    assert heal_repo.get_for_execution(execution_id) == []


def test_inserting_test_case_with_nonexistent_requirement_id_is_rejected(db_path):
    tc_repo = TestCaseRepository(db_path)
    with pytest.raises(sqlite3.IntegrityError):
        tc_repo.save(make_test_case(), requirement_id=9999)


# --- N. JSON serialization/deserialization works ---


def test_to_json_and_from_json_round_trip():
    value = {"username": "u", "flags": [1, 2, 3]}
    serialized = to_json(value)
    assert isinstance(serialized, str)
    assert from_json(serialized) == value


def test_from_json_returns_default_for_none():
    assert from_json(None, default=[]) == []


def test_from_json_returns_default_for_malformed_text():
    assert from_json("not valid json{{{", default={}) == {}


def test_healing_record_round_trips_nested_validation_results(db_path):
    execution_id = ExecutionRepository(db_path).save(make_execution_result())
    heal_repo = HealingRepository(db_path)
    healing_result = make_healing_result(
        validations=[
            SelectorValidation(selector="button.login", valid=True, match_count=1, reason="Matched one element."),
            SelectorValidation(selector="#login-button", valid=False, match_count=0, reason="No matches."),
        ]
    )
    heal_repo.save(execution_id, healing_result)

    record = heal_repo.get_for_execution(execution_id)[0]

    assert len(record.validation_result) == 2
    assert record.validation_result[0]["selector"] == "button.login"
    assert record.validation_result[0]["valid"] is True
    assert record.validation_result[1]["match_count"] == 0


# --- O. Database failure does not crash TestRunner ---


def test_database_failure_does_not_crash_test_runner(tmp_path, monkeypatch):
    from executor.test_runner import TestRunner

    class _ExplodingExecutionRepository:
        def save(self, execution_result: TestExecutionResult) -> int:
            raise sqlite3.OperationalError("simulated database failure")

    class _StubSettings:
        def __init__(self, base: Path) -> None:
            self.allowed_workspace_path = base / "generated"
            self.reports_path = base / "reports"
            self.screenshots_path = base / "screenshots"

    workspace = tmp_path / "generated"
    workspace.mkdir()
    test_file = workspace / "test_ok.py"
    test_file.write_text("def test_ok():\n    assert True\n", encoding="utf-8")

    monkeypatch.setattr("executor.test_runner.get_settings", lambda: _StubSettings(tmp_path))

    runner = TestRunner(execution_repository=_ExplodingExecutionRepository())
    result = runner.run_test(test_file)

    assert result.status == ExecutionStatus.PASSED  # the real result survives the DB failure
    assert runner.last_execution_id is None  # persistence failed, and that's reflected honestly
