"""Shared logging configuration for the whole framework."""

import logging
import sys

from core.config import PROJECT_ROOT, get_settings

_CONFIGURED = False


def _configure_root_logger() -> None:
    """Attach console + file handlers to the root logger exactly once."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    logs_dir = PROJECT_ROOT / settings.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(logs_dir / "app.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level.upper())
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with console + file handlers already attached."""
    _configure_root_logger()
    return logging.getLogger(name)
