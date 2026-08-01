"""Centralized logging configuration for the ML pipeline.

This module is the single source of truth for how the pipeline emits logs.
Every stage (`preprocess`, `train`, `evaluate`) configures logging through
:func:`configure_logging` at its entry point and obtains a module-scoped logger
via :func:`get_logger`.

Design goals:

- **Console + file output** — logs stream to the console for interactive runs
  and are persisted to a rotating file for later inspection.
- **Configurable level** — the ``LOG_LEVEL`` environment variable controls
  verbosity (default ``INFO``); no code change is needed to enable ``DEBUG``.
- **Consistent format** — a single formatter with timestamp, level, logger
  name, and message is applied to every handler.
- **Idempotent** — repeated calls do not attach duplicate handlers.
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

DEFAULT_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Log file location; kept under a git-ignored ``logs/`` directory at the repo root.
LOG_DIR = Path(os.environ.get("LOG_DIR", "logs"))
LOG_FILE = LOG_DIR / "pipeline.log"

# Rotating file handler bounds: 5 MB per file, 3 backups retained.
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 3


def configure_logging(level: str | None = None) -> None:
    """Configure the root logger with console and file handlers.

    Safe to call multiple times: if handlers are already attached, the call is a
    no-op so stages that share a process do not accumulate duplicate handlers.

    Args:
        level: Log level name (e.g. ``"DEBUG"``, ``"INFO"``). When ``None``,
            falls back to the ``LOG_LEVEL`` environment variable, then to
            ``INFO``.
    """
    root = logging.getLogger()

    resolved_level = (level or os.environ.get("LOG_LEVEL") or DEFAULT_LEVEL).upper()
    root.setLevel(resolved_level)

    if root.handlers:
        # Already configured in this process; keep the existing setup.
        return

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger.

    Args:
        name: Logger name, conventionally the module's ``__name__``.

    Returns:
        A :class:`logging.Logger` that inherits the root configuration set up by
        :func:`configure_logging`.
    """
    return logging.getLogger(name)
