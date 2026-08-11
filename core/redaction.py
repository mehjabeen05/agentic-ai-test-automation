"""Secret redaction utility.

Untrusted text (stdout, stderr, tracebacks, DOM snippets, ...) may contain
credentials or tokens that leaked into logs or error output. This module
provides a best-effort redaction pass to run over any such text before it
is sent to an LLM. It is not a guarantee — only a defense-in-depth measure
against the obvious cases.
"""

import re

REDACTED_PLACEHOLDER = "[REDACTED]"

# key=value / key: value / key "value" style secrets, case-insensitive key names.
_SECRET_KEY_VALUE_PATTERN = re.compile(
    r"(?i)\b(password|passwd|pwd|secret|api[_-]?key|access[_-]?token|"
    r"auth[_-]?token|refresh[_-]?token|token|client[_-]?secret)\b"
    r"(\s*[:=]\s*)"
    r"(\"[^\"]*\"|'[^']*'|\S+)"
)

# Well-known token/key shapes that appear without a preceding key name.
_BEARER_TOKEN_PATTERN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/]+=*")
_OPENAI_STYLE_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9]{16,}\b")


def redact_secrets(text: str | None) -> str | None:
    """Return `text` with obvious secrets replaced by a redaction marker.

    Best-effort: targets common `key=value`/`key: value` patterns for
    credential-like key names, plus a couple of well-known token shapes
    (Bearer tokens, OpenAI-style API keys). Returns `None` unchanged.
    """
    if not text:
        return text

    def _replace(match: re.Match) -> str:
        return f"{match.group(1)}{match.group(2)}{REDACTED_PLACEHOLDER}"

    redacted = _SECRET_KEY_VALUE_PATTERN.sub(_replace, text)
    redacted = _BEARER_TOKEN_PATTERN.sub(f"Bearer {REDACTED_PLACEHOLDER}", redacted)
    redacted = _OPENAI_STYLE_KEY_PATTERN.sub(REDACTED_PLACEHOLDER, redacted)
    return redacted
