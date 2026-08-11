"""SQLite persistence layer: connection management and schema initialization.

Intentionally independent of the LLM client and every agent/executor
module — this file only ever deals with plain Python values and
JSON-serialized text. Nothing here imports core.llm_client or anything
under agents/, generators/, or executor/; the dependency direction is
one-way (agents/executor may depend on core.database and
core.repositories, never the reverse).

SQLite does not enforce foreign keys by default — `PRAGMA foreign_keys = ON`
is set on every connection this module opens.
"""

import datetime as dt
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from core.config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)

# CREATE TABLE IF NOT EXISTS is idempotent and cheap, so this runs on every
# connection rather than requiring a separate one-time setup step — the
# schema is always guaranteed to exist by the time a caller gets a
# connection back.
_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS requirements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        requirement_text TEXT NOT NULL,
        analysis TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS test_cases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        requirement_id INTEGER REFERENCES requirements(id) ON DELETE CASCADE,
        test_case_id TEXT NOT NULL UNIQUE,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        priority TEXT NOT NULL,
        type TEXT NOT NULL,
        test_data TEXT NOT NULL DEFAULT '{}',
        preconditions TEXT NOT NULL DEFAULT '[]',
        steps TEXT NOT NULL DEFAULT '[]',
        expected_result TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_test_cases_requirement_id
        ON test_cases(requirement_id)
    """,
    # test_case_id here is a logical reference to test_cases.test_case_id,
    # not a hard SQL foreign key — see core/repositories.py's module
    # docstring for why (TestRunner derives this id from a generated
    # filename, independently of whether any test_cases row exists).
    """
    CREATE TABLE IF NOT EXISTS test_executions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        test_case_id TEXT NOT NULL,
        status TEXT NOT NULL,
        duration REAL NOT NULL,
        error TEXT,
        stdout TEXT,
        stderr TEXT,
        screenshot TEXT,
        healed INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_test_executions_test_case_id
        ON test_executions(test_case_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS failure_analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        execution_id INTEGER NOT NULL REFERENCES test_executions(id) ON DELETE CASCADE,
        failure_type TEXT NOT NULL,
        summary TEXT NOT NULL,
        root_cause TEXT NOT NULL,
        suggested_fix TEXT NOT NULL,
        confidence REAL NOT NULL,
        is_likely_environment_issue INTEGER NOT NULL,
        is_likely_test_issue INTEGER NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_failure_analyses_execution_id
        ON failure_analyses(execution_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS healing_attempts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        execution_id INTEGER NOT NULL REFERENCES test_executions(id) ON DELETE CASCADE,
        status TEXT NOT NULL DEFAULT 'failed',
        original_selector TEXT NOT NULL,
        candidate_selectors TEXT NOT NULL DEFAULT '[]',
        selected_selector TEXT,
        validation_result TEXT NOT NULL DEFAULT '[]',
        retry_succeeded INTEGER,
        confidence REAL,
        reason TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_healing_attempts_execution_id
        ON healing_attempts(execution_id)
    """,
)

# Lightweight, idempotent migrations for a database created before a column
# existed. `_SCHEMA_STATEMENTS` above only creates *new* tables — it can't
# retroactively add a column to a table that already exists, so any
# database created before this column was introduced needs an explicit
# `ALTER TABLE`. Each one is a no-op (via the "duplicate column" check
# below) once applied — there's no separate migration-tracking table for
# a project this size.
_MIGRATIONS: tuple[str, ...] = (
    "ALTER TABLE requirements ADD COLUMN analysis TEXT",
    "ALTER TABLE test_cases ADD COLUMN preconditions TEXT NOT NULL DEFAULT '[]'",
    "ALTER TABLE test_cases ADD COLUMN steps TEXT NOT NULL DEFAULT '[]'",
    "ALTER TABLE test_cases ADD COLUMN expected_result TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE healing_attempts ADD COLUMN status TEXT NOT NULL DEFAULT 'failed'",
)


def _apply_migrations(connection: sqlite3.Connection) -> None:
    for statement in _MIGRATIONS:
        try:
            connection.execute(statement)
        except sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).lower():
                raise


def get_database_path() -> Path:
    """The configured, absolute path to the SQLite database file."""
    return get_settings().database_path


@contextmanager
def connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection with foreign keys enabled and the schema ensured.

    Always used as a context manager, so the connection is reliably closed.
    Callers must call `.commit()` themselves after writes — commits are
    kept explicit rather than implicit, so a caller doing multiple related
    writes controls their own transaction boundary.
    """
    path = db_path or get_database_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    for statement in _SCHEMA_STATEMENTS:
        connection.execute(statement)
    _apply_migrations(connection)
    connection.commit()

    try:
        yield connection
    finally:
        connection.close()


def initialize_database(db_path: Path | None = None) -> Path:
    """Explicitly create the database file and all tables if they don't exist.

    Also happens automatically on every `connect()` call — this is a
    convenience for an explicit startup step or a test assertion.
    """
    path = db_path or get_database_path()
    with connect(path):
        pass
    logger.info("Database ready at %s", path)
    return path


def to_json(value: Any) -> str:
    """Serialize a Python value to a JSON string for storage in a TEXT column."""
    return json.dumps(value)


def from_json(text: str | None, default: Any = None) -> Any:
    """Deserialize a JSON string from a TEXT column back to a Python value.

    Returns `default` for `None` input or text that isn't valid JSON,
    rather than raising — a malformed stored value should degrade
    gracefully, not break a read.
    """
    if text is None:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Could not parse stored JSON value; returning default")
        return default


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string, for consistent `created_at` values."""
    return dt.datetime.now(dt.timezone.utc).isoformat()
