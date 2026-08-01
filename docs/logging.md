# Logging Strategy

This document describes how the ML pipeline emits, formats, and persists logs.
It is the reference for both contributors adding new log statements and
operators consuming pipeline output.

---

## 1. Goals

- **One configuration, applied everywhere.** All stages log through a single
  module so format, level, and destinations are consistent.
- **Observable by default.** Every pipeline stage emits meaningful *lifecycle*
  logs (start / completion) without drowning them in noise.
- **Console and file.** Logs stream to the console for interactive runs and are
  persisted to a rotating file for later inspection.
- **Configurable verbosity.** Log level is controlled by an environment
  variable; no code change is needed to switch to `DEBUG`.

---

## 2. The logging module

All logging is configured by [`src/logging_config.py`](../src/logging_config.py),
which exposes two functions:

| Function | Purpose |
|----------|---------|
| `configure_logging(level=None)` | Configure the root logger with console + file handlers. Called **once** at each stage's entry point (`if __name__ == "__main__":`). Idempotent. |
| `get_logger(name)` | Return a named logger for a module. Called at module import time. |

### Usage pattern

```python
from logging_config import configure_logging, get_logger

logger = get_logger("preprocess")


def preprocess(input_path, output_path):
    logger.info("Preprocess stage started (input=%s, output=%s)", input_path, output_path)
    ...
    logger.info("Preprocess stage completed: %d rows written to %s", len(df), output_path)


if __name__ == "__main__":
    load_dotenv()        # load .env first so LOG_LEVEL / LOG_DIR are honored
    configure_logging()
    ...
```

> **Call order matters.** `configure_logging()` reads `LOG_LEVEL` / `LOG_DIR`
> from the environment at call time, so `load_dotenv()` must run first for
> values defined in `.env` to take effect.

> The stages run as standalone scripts (`python src/preprocess.py`), so `src/`
> is on `sys.path` and the sibling import `from logging_config import ...` is
> correct. Stable logger names (`"preprocess"`, `"train"`, `"evaluate"`) are
> used instead of `__name__` so the stage is always identifiable in a shared
> log file, even when a script runs as `__main__`.

---

## 3. Output destinations

`configure_logging()` attaches two handlers to the root logger:

1. **Console** (`StreamHandler`) — for interactive and CI runs.
2. **Rotating file** (`RotatingFileHandler`) — persisted to
   `logs/pipeline.log`, rotating at **5 MB** with **3** backups retained.

The `logs/` directory defaults to the **repository root** (anchored to the
module location, not the process CWD, so the path is stable however a stage is
launched). It is created automatically and is **git-ignored**
(see [`.gitignore`](../.gitignore)); log files are never committed. The location
can be overridden with the `LOG_DIR` environment variable.

### Third-party noise

`configure_logging()` caps known-chatty dependency loggers (`botocore`,
`urllib3`, `s3transfer`, `git`, …) at `WARNING`, so their output does not swamp
the pipeline's own logs — even when `LOG_LEVEL=DEBUG` is used to debug pipeline
code.

---

## 4. Log format

A single formatter is applied to every handler:

```
%(asctime)s | %(levelname)-8s | %(name)s | %(message)s
```

Example:

```
2026-08-01 18:53:38 | INFO     | preprocess | Preprocess stage started (input=data/raw/data.csv, output=data/processed/data.csv)
2026-08-01 18:53:39 | INFO     | preprocess | Preprocess stage completed: 768 rows written to data/processed/data.csv
```

Each line carries a timestamp, the severity, the emitting stage, and the
message — enough context to trace a run without external correlation.

> Log messages use lazy `%`-style interpolation (`logger.info("… %s", value)`)
> rather than f-strings, so argument formatting is skipped when the level is
> disabled.

---

## 5. Log levels

Configured via the `LOG_LEVEL` environment variable (default: `INFO`).

| Level | When to use | Examples in this pipeline |
|-------|-------------|---------------------------|
| `DEBUG` | Development-only detail. **Off by default.** | Fine-grained diagnostics while debugging locally. |
| `INFO` | Stage lifecycle and key outcomes. | Stage started / completed, best model accuracy, model saved. |
| `WARNING` | Recoverable issues that don't stop the run. | Falling back to a default, retrying a transient step. |
| `ERROR` | Failures. | An operation that cannot complete (paired with exception handling). |

To run with verbose output:

```bash
# Linux / macOS
LOG_LEVEL=DEBUG python src/train.py

# Windows PowerShell
$env:LOG_LEVEL = "DEBUG"; python src/train.py
```

The log **file location** can be overridden with the `LOG_DIR` environment
variable (default: `logs`).

---

## 6. Lifecycle logs per stage

Each stage emits a small, deliberate set of `INFO` logs — enough to follow the
run, not enough to bury the signal.

| Stage | Lifecycle logs (INFO) |
|-------|-----------------------|
| **preprocess** | stage started (input/output paths) → stage completed (row count, output path) |
| **train** | stage started → hyperparameter tuning started → tuning completed (best params) → best model accuracy → model saved → stage completed |
| **evaluate** | stage started → stage completed (model accuracy) |

---

## 7. Conventions ("do not over-log")

- **One start and one completion log per stage.** These bookend the work and
  make runs easy to follow.
- **Log outcomes, not iterations.** Record the *result* of a step (e.g. best
  accuracy), not per-loop progress.
- **No noisy `DEBUG` in committed code.** `DEBUG` is reserved for temporary,
  local diagnostics and stays off in normal runs.
- **Never use `print()` for diagnostics.** `print()` cannot be filtered,
  redirected, or timestamped; use a logger.
- **Log before raising**, with an actionable message, when exception handling is
  introduced (see the [engineering review](reviews/sprint-02-engineering-review.md)
  finding H-2).

> **Note on GridSearchCV.** `GridSearchCV(..., verbose=2)` in `train.py` produces
> its own progress output independent of this logging configuration. It is a
> tuning-time diagnostic and can be reduced by lowering `verbose` if that output
> becomes noisy.

---

## Related Documentation

- [Architecture](architecture.md) — see §7, Observability & Logging.
- [Engineering Review](reviews/sprint-02-engineering-review.md) — finding H-1.
