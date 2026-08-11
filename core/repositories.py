"""Repository layer: all SQL for this project lives here, and only here.

Each repository owns one entity's persistence. Agents, the execution
engine, and the healing executor call these methods — they never write SQL
themselves (see core/database.py's module docstring for the dependency
direction this maintains).

A note on `test_executions.test_case_id`: it is a plain, indexed TEXT
column, not a hard SQL foreign key to `test_cases.test_case_id`. TestRunner
(Step 7) derives its execution's id from the *generated filename* (e.g.
"test_tc_login_001"), entirely independently of whether any `test_cases`
row exists for it and regardless of that row's original `test_case_id`
casing (e.g. "TC_LOGIN_001") — by design, TestRunner has no dependency on
the TestCase model or the LLM layer. Enforcing a strict foreign key here
would reject routine execution inserts whenever that string doesn't happen
to match a `test_cases` row exactly, which would silently defeat the whole
point of "persistence failures must never crash execution." The two IDs
are related by a deterministic, known transformation, but are correlated
by convention rather than by database constraint.
"""

import sqlite3
from pathlib import Path

from core.database import connect, from_json, to_json, utc_now_iso
from core.db_models import (
    ExecutionRecord,
    FailureAnalysisRecord,
    HealingRecord,
    RequirementRecord,
    TestCaseRecord,
)
from core.logger import get_logger
from core.models import (
    ExecutionStatus,
    FailureAnalysis,
    HealingResult,
    RequirementAnalysis,
    TestCase,
    TestExecutionResult,
)

logger = get_logger(__name__)


def _row_to_requirement_record(row: sqlite3.Row) -> RequirementRecord:
    stored_analysis = from_json(row["analysis"], default=None)
    return RequirementRecord(
        id=row["id"],
        requirement_text=row["requirement_text"],
        analysis=RequirementAnalysis.model_validate(stored_analysis) if stored_analysis else None,
        created_at=row["created_at"],
    )


def _row_to_test_case_record(row: sqlite3.Row) -> TestCaseRecord:
    return TestCaseRecord(
        id=row["id"],
        requirement_id=row["requirement_id"],
        test_case_id=row["test_case_id"],
        title=row["title"],
        description=row["description"],
        priority=row["priority"],
        type=row["type"],
        test_data=from_json(row["test_data"], default={}),
        preconditions=from_json(row["preconditions"], default=[]),
        steps=from_json(row["steps"], default=[]),
        expected_result=row["expected_result"] or "",
        created_at=row["created_at"],
    )


def _row_to_execution_record(row: sqlite3.Row) -> ExecutionRecord:
    return ExecutionRecord(
        id=row["id"],
        test_case_id=row["test_case_id"],
        status=row["status"],
        duration=row["duration"],
        error=row["error"],
        stdout=row["stdout"],
        stderr=row["stderr"],
        screenshot=row["screenshot"],
        healed=bool(row["healed"]),
        created_at=row["created_at"],
    )


def _row_to_failure_analysis_record(row: sqlite3.Row) -> FailureAnalysisRecord:
    return FailureAnalysisRecord(
        id=row["id"],
        execution_id=row["execution_id"],
        failure_type=row["failure_type"],
        summary=row["summary"],
        root_cause=row["root_cause"],
        suggested_fix=row["suggested_fix"],
        confidence=row["confidence"],
        is_likely_environment_issue=bool(row["is_likely_environment_issue"]),
        is_likely_test_issue=bool(row["is_likely_test_issue"]),
        created_at=row["created_at"],
    )


def _row_to_healing_record(row: sqlite3.Row) -> HealingRecord:
    return HealingRecord(
        id=row["id"],
        execution_id=row["execution_id"],
        status=row["status"],
        original_selector=row["original_selector"],
        candidate_selectors=from_json(row["candidate_selectors"], default=[]),
        selected_selector=row["selected_selector"],
        validation_result=from_json(row["validation_result"], default=[]),
        retry_succeeded=None if row["retry_succeeded"] is None else bool(row["retry_succeeded"]),
        confidence=row["confidence"],
        reason=row["reason"],
        created_at=row["created_at"],
    )


class RequirementRepository:
    """Persistence for raw natural-language requirements."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path

    def save(self, requirement_text: str, analysis: RequirementAnalysis | None = None) -> int:
        """Persist a requirement's raw text (and its structured analysis, if given).

        `analysis` is optional so this remains a simple, standalone call
        when only the raw text matters. When provided, it's stored as JSON
        so `get_analysis()` can hand back a real `RequirementAnalysis`
        later, without ever re-calling the LLM.
        """
        with connect(self._db_path) as connection:
            cursor = connection.execute(
                "INSERT INTO requirements (requirement_text, analysis, created_at) VALUES (?, ?, ?)",
                (
                    requirement_text,
                    to_json(analysis.model_dump(mode="json")) if analysis else None,
                    utc_now_iso(),
                ),
            )
            connection.commit()
            return cursor.lastrowid

    def get_all(self) -> list[RequirementRecord]:
        """All requirements, oldest first."""
        with connect(self._db_path) as connection:
            rows = connection.execute(
                "SELECT id, requirement_text, analysis, created_at FROM requirements ORDER BY id"
            ).fetchall()
        return [_row_to_requirement_record(row) for row in rows]

    def get_by_id(self, requirement_id: int) -> RequirementRecord | None:
        with connect(self._db_path) as connection:
            row = connection.execute(
                "SELECT id, requirement_text, analysis, created_at FROM requirements WHERE id = ?",
                (requirement_id,),
            ).fetchone()
        return _row_to_requirement_record(row) if row is not None else None

    def get_analysis(self, requirement_id: int) -> RequirementAnalysis | None:
        """The stored structured analysis for a requirement, if one was saved."""
        record = self.get_by_id(requirement_id)
        return record.analysis if record is not None else None


class TestCaseRepository:
    """Persistence for structured test cases."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path

    def save(self, test_case: TestCase, requirement_id: int | None = None) -> int:
        """Insert or update (matched by `test_case_id`) a TestCase, returning its row id."""
        with connect(self._db_path) as connection:
            connection.execute(
                """
                INSERT INTO test_cases
                    (requirement_id, test_case_id, title, description, priority, type,
                     test_data, preconditions, steps, expected_result, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(test_case_id) DO UPDATE SET
                    requirement_id = excluded.requirement_id,
                    title = excluded.title,
                    description = excluded.description,
                    priority = excluded.priority,
                    type = excluded.type,
                    test_data = excluded.test_data,
                    preconditions = excluded.preconditions,
                    steps = excluded.steps,
                    expected_result = excluded.expected_result
                """,
                (
                    requirement_id,
                    test_case.test_case_id,
                    test_case.title,
                    test_case.description,
                    test_case.priority.value,
                    test_case.type.value,
                    to_json(test_case.test_data),
                    to_json(test_case.preconditions),
                    to_json(test_case.steps),
                    test_case.expected_result,
                    utc_now_iso(),
                ),
            )
            connection.commit()
            row = connection.execute(
                "SELECT id FROM test_cases WHERE test_case_id = ?", (test_case.test_case_id,)
            ).fetchone()
            return row["id"]

    def get_by_requirement(self, requirement_id: int) -> list[TestCaseRecord]:
        with connect(self._db_path) as connection:
            rows = connection.execute(
                "SELECT * FROM test_cases WHERE requirement_id = ? ORDER BY id", (requirement_id,)
            ).fetchall()
        return [_row_to_test_case_record(row) for row in rows]

    def get_by_test_case_id(self, test_case_id: str) -> TestCaseRecord | None:
        with connect(self._db_path) as connection:
            row = connection.execute(
                "SELECT * FROM test_cases WHERE test_case_id = ?", (test_case_id,)
            ).fetchone()
        return _row_to_test_case_record(row) if row is not None else None

    def get_all(self) -> list[TestCaseRecord]:
        """Every stored test case, oldest first. Used by the dashboard's stats endpoint."""
        with connect(self._db_path) as connection:
            rows = connection.execute("SELECT * FROM test_cases ORDER BY id").fetchall()
        return [_row_to_test_case_record(row) for row in rows]

    def get_full_test_case(self, test_case_id: str) -> TestCase | None:
        """Reconstruct the full domain `TestCase` for `test_case_id`, if one is stored.

        Unlike `get_by_test_case_id` (which returns the lighter,
        DB-row-shaped `TestCaseRecord`), this returns the same `TestCase`
        model the agents/generators use — e.g. so the Playwright Code
        Generator can be handed a real TestCase without the API layer
        having to reconstruct or duplicate that shape itself.
        """
        record = self.get_by_test_case_id(test_case_id)
        if record is None:
            return None
        return TestCase(
            test_case_id=record.test_case_id,
            title=record.title,
            description=record.description,
            preconditions=record.preconditions,
            steps=record.steps,
            test_data=record.test_data,
            expected_result=record.expected_result,
            priority=record.priority,
            type=record.type,
        )


class ExecutionRepository:
    """Persistence for test execution results."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path

    def save(self, execution_result: TestExecutionResult) -> int:
        """Persist one TestExecutionResult and return its new row id."""
        with connect(self._db_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO test_executions
                    (test_case_id, status, duration, error, stdout, stderr, screenshot, healed, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    execution_result.test_id,
                    execution_result.status.value,
                    execution_result.duration,
                    execution_result.error,
                    execution_result.stdout,
                    execution_result.stderr,
                    execution_result.screenshot,
                    int(execution_result.healed),
                    utc_now_iso(),
                ),
            )
            connection.commit()
            return cursor.lastrowid

    def get_by_id(self, execution_id: int) -> ExecutionRecord | None:
        with connect(self._db_path) as connection:
            row = connection.execute(
                "SELECT * FROM test_executions WHERE id = ?", (execution_id,)
            ).fetchone()
        return _row_to_execution_record(row) if row is not None else None

    def get_all(self) -> list[ExecutionRecord]:
        """Every recorded execution, oldest first. Used by the dashboard's stats endpoint."""
        with connect(self._db_path) as connection:
            rows = connection.execute("SELECT * FROM test_executions ORDER BY id").fetchall()
        return [_row_to_execution_record(row) for row in rows]

    def get_history(self, test_case_id: str) -> list[ExecutionRecord]:
        """All executions recorded for `test_case_id`, oldest first."""
        with connect(self._db_path) as connection:
            rows = connection.execute(
                "SELECT * FROM test_executions WHERE test_case_id = ? ORDER BY id", (test_case_id,)
            ).fetchall()
        return [_row_to_execution_record(row) for row in rows]

    def get_latest(self, test_case_id: str) -> ExecutionRecord | None:
        with connect(self._db_path) as connection:
            row = connection.execute(
                "SELECT * FROM test_executions WHERE test_case_id = ? ORDER BY id DESC LIMIT 1",
                (test_case_id,),
            ).fetchone()
        return _row_to_execution_record(row) if row is not None else None

    def get_failed(self) -> list[ExecutionRecord]:
        """Every execution whose status is `failed` or `error`, most recent first."""
        with connect(self._db_path) as connection:
            rows = connection.execute(
                "SELECT * FROM test_executions WHERE status IN (?, ?) ORDER BY id DESC",
                (ExecutionStatus.FAILED.value, ExecutionStatus.ERROR.value),
            ).fetchall()
        return [_row_to_execution_record(row) for row in rows]

    def get_healed(self) -> list[ExecutionRecord]:
        """Every execution that passed after selector healing, most recent first."""
        with connect(self._db_path) as connection:
            rows = connection.execute(
                "SELECT * FROM test_executions WHERE healed = 1 ORDER BY id DESC"
            ).fetchall()
        return [_row_to_execution_record(row) for row in rows]


class FailureAnalysisRepository:
    """Persistence for failure analyses, each linked to the execution it explains."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path

    def save(self, execution_id: int, analysis: FailureAnalysis) -> int:
        with connect(self._db_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO failure_analyses
                    (execution_id, failure_type, summary, root_cause, suggested_fix,
                     confidence, is_likely_environment_issue, is_likely_test_issue, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    execution_id,
                    analysis.failure_type.value,
                    analysis.summary,
                    analysis.root_cause,
                    analysis.suggested_fix,
                    analysis.confidence,
                    int(analysis.is_likely_environment_issue),
                    int(analysis.is_likely_test_issue),
                    utc_now_iso(),
                ),
            )
            connection.commit()
            return cursor.lastrowid

    def get_for_execution(self, execution_id: int) -> list[FailureAnalysisRecord]:
        with connect(self._db_path) as connection:
            rows = connection.execute(
                "SELECT * FROM failure_analyses WHERE execution_id = ? ORDER BY id", (execution_id,)
            ).fetchall()
        return [_row_to_failure_analysis_record(row) for row in rows]

    def get_all(self) -> list[FailureAnalysisRecord]:
        with connect(self._db_path) as connection:
            rows = connection.execute("SELECT * FROM failure_analyses ORDER BY id DESC").fetchall()
        return [_row_to_failure_analysis_record(row) for row in rows]


class HealingRepository:
    """Persistence for selector-healing attempts, each linked to the execution that triggered them."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path

    def save(self, execution_id: int, healing_result: HealingResult) -> int:
        with connect(self._db_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO healing_attempts
                    (execution_id, status, original_selector, candidate_selectors, selected_selector,
                     validation_result, retry_succeeded, confidence, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    execution_id,
                    healing_result.status.value,
                    healing_result.original_selector,
                    to_json(healing_result.candidate_selectors),
                    healing_result.selected_selector,
                    to_json([validation.model_dump() for validation in healing_result.validations]),
                    None if healing_result.retry_succeeded is None else int(healing_result.retry_succeeded),
                    healing_result.confidence,
                    healing_result.reason,
                    utc_now_iso(),
                ),
            )
            connection.commit()
            return cursor.lastrowid

    def get_for_execution(self, execution_id: int) -> list[HealingRecord]:
        with connect(self._db_path) as connection:
            rows = connection.execute(
                "SELECT * FROM healing_attempts WHERE execution_id = ? ORDER BY id", (execution_id,)
            ).fetchall()
        return [_row_to_healing_record(row) for row in rows]

    def get_history(self) -> list[HealingRecord]:
        with connect(self._db_path) as connection:
            rows = connection.execute("SELECT * FROM healing_attempts ORDER BY id DESC").fetchall()
        return [_row_to_healing_record(row) for row in rows]
