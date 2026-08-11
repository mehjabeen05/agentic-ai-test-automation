"""AST-based safety validation for LLM-generated Playwright Python source.

LLM-generated code is untrusted. This module NEVER executes, evaluates, or
imports the generated code — it only parses it with Python's `ast` module
and inspects the resulting syntax tree.
"""

import ast
import re
from dataclasses import dataclass, field

from core.logger import get_logger

logger = get_logger(__name__)

# Whole modules that are never allowed in generated test code: they enable
# arbitrary command execution, raw sockets, or network calls that bypass
# Playwright entirely. "os" itself is allowed (needed for os.getenv), but
# specific dangerous functions on it are still blocked below.
_FORBIDDEN_MODULE_PREFIXES = (
    "subprocess",
    "shutil",
    "socket",
    "ctypes",
    "pickle",
    "multiprocessing",
    "threading",
    "sys",
    "importlib",
    "ftplib",
    "smtplib",
    "telnetlib",
    "requests",
    "urllib",
    "urllib2",
    "urllib3",
    "httpx",
    "http.client",
)

# Matched against the LAST segment of a call's dotted name, so both
# `os.system(...)` and `from os import system; system(...)` are caught.
_EXECUTION_DANGEROUS_NAMES = {"eval", "exec", "compile", "__import__", "system", "popen"}
_FILE_DELETION_NAMES = {"remove", "unlink", "rmdir", "removedirs", "rmtree"}

_CREDENTIAL_NAME_PATTERN = re.compile(r"(password|passwd|pwd|secret|api[_-]?key|token)", re.IGNORECASE)


@dataclass
class ValidationIssue:
    """A single reason generated code was rejected."""

    code: str
    message: str


@dataclass
class ValidationResult:
    """The outcome of validating a block of generated code."""

    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)


class CodeValidationError(Exception):
    """Raised when generated code fails AST-based safety validation."""

    def __init__(self, result: ValidationResult) -> None:
        self.result = result
        reasons = "; ".join(issue.message for issue in result.issues)
        super().__init__(f"Generated code failed validation: {reasons}")


def _is_forbidden_module(dotted_name: str) -> bool:
    return any(
        dotted_name == prefix or dotted_name.startswith(prefix + ".")
        for prefix in _FORBIDDEN_MODULE_PREFIXES
    )


def _call_full_name(node: ast.Call) -> str | None:
    """Best-effort dotted name of a call target, e.g. 'os.system' or 'eval'."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts: list[str] = []
        current = func
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            return ".".join(reversed(parts))
    return None


def _check_forbidden_imports(tree: ast.AST) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden_module(alias.name):
                    issues.append(
                        ValidationIssue(
                            "forbidden_import",
                            f"Importing '{alias.name}' is not allowed in generated test code.",
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _is_forbidden_module(module):
                issues.append(
                    ValidationIssue(
                        "forbidden_import",
                        f"Importing from '{module}' is not allowed in generated test code.",
                    )
                )
    return issues


def _check_dangerous_calls(tree: ast.AST) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        full_name = _call_full_name(node)
        if full_name is None:
            continue
        last_segment = full_name.rsplit(".", maxsplit=1)[-1]
        if last_segment in _EXECUTION_DANGEROUS_NAMES:
            issues.append(
                ValidationIssue(
                    "forbidden_call",
                    f"Call to '{full_name}(...)' is not allowed in generated test code.",
                )
            )
        elif last_segment in _FILE_DELETION_NAMES:
            issues.append(
                ValidationIssue(
                    "file_deletion",
                    f"Call to '{full_name}(...)' can delete files and is not allowed in generated test code.",
                )
            )
    return issues


def _check_hardcoded_credentials(tree: ast.AST) -> list[ValidationIssue]:
    """Best-effort detection of obvious hardcoded credentials.

    Flags two patterns: a credential-like variable assigned a literal
    string, and a credential-like selector passed a literal string via
    `.fill(...)`/`.type(...)`. This is a heuristic, not a guarantee.
    """
    issues: list[ValidationIssue] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if not (isinstance(target, ast.Name) and _CREDENTIAL_NAME_PATTERN.search(target.id)):
                    continue
                if (
                    isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                    and node.value.value.strip()
                ):
                    issues.append(
                        ValidationIssue(
                            "hardcoded_credential",
                            f"Variable '{target.id}' appears to hold a hardcoded credential; "
                            "use os.getenv(...) instead.",
                        )
                    )
        elif isinstance(node, ast.Call):
            full_name = _call_full_name(node)
            if not full_name or full_name.rsplit(".", maxsplit=1)[-1] not in {"fill", "type"}:
                continue
            if len(node.args) < 2:
                continue
            selector_arg, value_arg = node.args[0], node.args[1]
            if (
                isinstance(selector_arg, ast.Constant)
                and isinstance(selector_arg.value, str)
                and _CREDENTIAL_NAME_PATTERN.search(selector_arg.value)
                and isinstance(value_arg, ast.Constant)
                and isinstance(value_arg.value, str)
                and value_arg.value.strip()
            ):
                issues.append(
                    ValidationIssue(
                        "hardcoded_credential",
                        f"Call to '{full_name}' passes a hardcoded value into a "
                        f"credential-like field ('{selector_arg.value}').",
                    )
                )
    return issues


def _has_test_function(tree: ast.Module) -> bool:
    """Whether the module defines at least one top-level pytest-style test function.

    A pytest test function is valid with any number of parameters (zero if
    it needs no fixtures), so only the `test_*` naming convention is checked.
    """
    return any(
        isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        for node in ast.iter_child_nodes(tree)
    )


def validate_generated_code(code: str) -> ValidationResult:
    """Statically validate generated Playwright Python source.

    The code is parsed and inspected as a syntax tree; it is never executed,
    evaluated, or imported.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        logger.error("Generated code failed to parse: %s", exc)
        return ValidationResult(
            is_valid=False,
            issues=[ValidationIssue("syntax_error", f"Code is not valid Python: {exc}")],
        )

    issues: list[ValidationIssue] = []
    issues.extend(_check_forbidden_imports(tree))
    issues.extend(_check_dangerous_calls(tree))
    issues.extend(_check_hardcoded_credentials(tree))

    if not _has_test_function(tree):
        issues.append(
            ValidationIssue(
                "missing_test_function",
                "No pytest-compatible test function (a top-level function named test_*) was found.",
            )
        )

    if issues:
        logger.error(
            "Generated code failed validation (%d issue(s)): %s",
            len(issues),
            "; ".join(issue.code for issue in issues),
        )
    return ValidationResult(is_valid=not issues, issues=issues)
