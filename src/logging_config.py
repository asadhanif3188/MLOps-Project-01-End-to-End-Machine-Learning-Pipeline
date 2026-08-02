"""Centralized logging configuration for the ML pipeline.

This module is the single source of truth for how the pipeline emits logs.
Every stage (`preprocess`, `train`, `evaluate`) configures logging through
:func:`configure_logging` at its entry point and obtains a stage-scoped logger
via :func:`get_logger`.

Design goals:

- **Console + file output** — logs stream to the console for interactive runs
  and are persisted to a rotating file for later inspection.
- **Configurable level** — the ``LOG_LEVEL`` environment variable controls
  verbosity (default ``INFO``); no code change is needed to enable ``DEBUG``.
- **Consistent format** — a single formatter with timestamp, level, logger
  name, and message is applied to every handler.
- **Quiet by default** — known-noisy third-party loggers (e.g. ``botocore``,
  ``urllib3``) are capped at ``WARNING`` so they do not drown out pipeline logs,
  even when ``LOG_LEVEL=DEBUG``.
- **Idempotent** — repeated calls do not attach duplicate handlers.

Environment variables (read at :func:`configure_logging` call time, so they can
be provided via ``.env`` as long as ``load_dotenv()`` runs first):

- ``LOG_LEVEL`` — log level name (default ``INFO``).
- ``LOG_DIR`` — directory for the log file (default: ``logs/`` at the repo root).
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

DEFAULT_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE_NAME = "pipeline.log"

# Project root (this file lives at <root>/src/logging_config.py). Used to anchor
# the default log directory so it is stable regardless of the process CWD.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Rotating file handler bounds: 5 MB per file, 3 backups retained.
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 3

# Third-party loggers capped at WARNING so their INFO/DEBUG chatter (verbose at
# LOG_LEVEL=DEBUG) does not swamp the pipeline's own logs.
_NOISY_LOGGERS = ("urllib3", "botocore", "boto3", "s3transfer", "git", "matplotlib")

# Guards against attaching duplicate handlers when stages share a process.
_configured = False


def _resolve_log_dir() -> Path:
    """Resolve the log directory, honoring ``LOG_DIR`` at call time."""
    env_dir = os.environ.get("LOG_DIR")
    return Path(env_dir) if env_dir else _PROJECT_ROOT / "logs"


def configure_logging(level: str | None = None) -> None:
    """Configure the root logger with console and file handlers.

    Safe to call multiple times: handlers are attached only on the first call so
    stages that share a process do not accumulate duplicate handlers. The level
    is (re)applied on every call.

    Call this **after** ``load_dotenv()`` so that ``LOG_LEVEL`` / ``LOG_DIR``
    defined in ``.env`` are honored.

    Args:
        level: Log level name (e.g. ``"DEBUG"``, ``"INFO"``). When ``None``,
            falls back to the ``LOG_LEVEL`` environment variable, then to
            ``INFO``.
    """
    global _configured

    resolved_level = (level or os.environ.get("LOG_LEVEL") or DEFAULT_LEVEL).upper()
    root = logging.getLogger()
    root.setLevel(resolved_level)

    if _configured:
        return

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    log_dir = _resolve_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    # NOTE: RotatingFileHandler assumes a single writer. DVC runs stages
    # sequentially, so concurrent rotation of this file is not a concern today.
    file_handler = RotatingFileHandler(
        log_dir / LOG_FILE_NAME,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a stage-scoped logger.

    Args:
        name: Logger name. Stages pass a stable identifier (e.g. ``"train"``)
            rather than ``__name__``: since stages run as ``__main__``, a stable
            name keeps the ``%(name)s`` field meaningful in the shared log file.

    Returns:
        A :class:`logging.Logger` that inherits the root configuration set up by
        :func:`configure_logging`.
    """
    return logging.getLogger(name)
