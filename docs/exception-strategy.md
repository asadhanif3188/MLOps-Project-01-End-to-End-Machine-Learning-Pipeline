# Exception Strategy

This document describes how the ML pipeline **classifies, raises, propagates, and
logs errors**. It is the reference for contributors adding new failure-handling
code and the companion to the [Logging Strategy](logging.md).

It implements engineering-review finding
[**H-2**](reviews/sprint-02-engineering-review.md) ("No exception handling
anywhere"): a small typed exception hierarchy, wrapped IO/config/network
boundaries, up-front validation of required configuration, and errors that are
logged exactly once with a preserved traceback.

---

## 1. Goals

- **Typed, not stringly.** Every expected failure is an instance of a small
  hierarchy, so callers can decide *what kind* of thing went wrong by type, not
  by parsing a message.
- **Actionable messages.** An error tells the operator what happened *and what
  to do next* (e.g. run `dvc pull`, copy `.env.example` to `.env`).
- **Never swallow.** No `except` clause silently discards an error. Boundaries
  either handle-and-re-raise (with context) or let the error propagate.
- **Preserve the cause.** Wrapping uses `raise NewError(...) from exc` so the
  original traceback is never lost.
- **Log once, at the boundary.** A failure is logged a single time — at the
  stage entry point — with the full chained traceback, rather than at every
  frame it passes through.
- **Fail loudly for automation.** A failed stage exits non-zero so `dvc repro`
  and CI stop instead of continuing on bad state.

---

## 2. Exception hierarchy

All pipeline-specific errors live in
[`src/exceptions.py`](../src/exceptions.py) and derive from a single base,
`PipelineError`. Each subclass marks one **failure boundary** the pipeline
crosses.

```text
Exception
└── PipelineError            # base: catch this to handle any expected failure
    ├── ConfigError          # missing/invalid params.yaml or env vars
    ├── DataError            # dataset cannot be read/written or has wrong shape
    ├── ModelError           # model cannot be (de)serialized or used
    └── TrackingError        # MLflow / DagsHub interaction failed
```

| Exception | Raised when | Example trigger |
|-----------|-------------|-----------------|
| `PipelineError` | Base class — not raised directly. | — |
| `ConfigError` | Configuration is missing or invalid. | `params.yaml` absent, missing key, or `MLFLOW_TRACKING_URI` unset. |
| `DataError` | A dataset can't be read/written or lacks a required column. | `data/raw/data.csv` missing; no `Outcome` column. |
| `ModelError` | A model artifact can't be (de)serialized or used. | `models/model.pkl` missing/corrupt; `predict` fails. |
| `TrackingError` | Experiment tracking fails at a network boundary. | DagsHub unreachable or credentials rejected. |

`PipelineError` is intentionally **behavior-free** (message only) and the module
is **dependency-free** (standard library only) so it can be imported anywhere
without risking import cycles.

---

## 3. Where errors are raised — the boundary helpers

The `try`/`except` blocks are **centralized** in
[`src/pipeline_io.py`](../src/pipeline_io.py), not scattered across the stages.
Each helper wraps one low-level boundary and re-raises a typed error. This is
what "standardized across modules" means in practice: `preprocess`, `train`, and
`evaluate` all read a CSV through the *same* `read_csv`, so a missing file
produces the *same* `DataError` with the *same* actionable message everywhere.

| Helper | Boundary wrapped | Raises |
|--------|------------------|--------|
| `load_params(path, stage, required)` | Open + parse YAML; validate section and keys. | `ConfigError` |
| `require_env(name)` | Read a required environment variable. | `ConfigError` |
| `read_csv(path)` | `pandas.read_csv`. | `DataError` |
| `write_csv(df, path)` | `makedirs` + `DataFrame.to_csv`. | `DataError` |
| `ensure_columns(df, required, source)` | Validate a DataFrame's columns. | `DataError` |
| `load_pickle(path)` | `open` + `pickle.load`. | `ModelError` |
| `save_pickle(obj, path)` | `makedirs` + `pickle.dump`. | `ModelError` |

MLflow calls are wrapped **in the stage** (they are interleaved with training
logic): `train` and `evaluate` catch `mlflow.exceptions.MlflowException` around
their tracking blocks and re-raise `TrackingError`. Because `MlflowException` is
specific, this never accidentally reclassifies a data or model error that occurs
in the same block.

### Catch narrow, wrap, and chain

Boundaries catch the **specific** low-level exceptions they expect and translate
them — they do not catch bare `Exception`:

```python
def read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError as exc:
        raise DataError(
            f"Dataset not found: {path!r}. If it is DVC-tracked, run "
            f"'dvc pull' to fetch it first."
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise DataError(f"Dataset {path!r} is empty.") from exc
```

The `from exc` clause is mandatory when wrapping: it sets `__cause__` so the
original traceback is chained and printed under
*"The above exception was the direct cause of the following exception."*

---

## 4. Error propagation

Errors flow **outward, unhandled, until the stage boundary**:

```text
pipeline_io helper           stage function              stage entry point
─────────────────            ──────────────              ─────────────────
raises ConfigError/          calls helpers; may add      run_stage(name, main)
DataError/ModelError    ──▶  TrackingError around   ──▶  catches PipelineError,
(chained from the low-            MLflow calls;                logs once + exit(1)
level cause)                 otherwise lets errors
                             propagate untouched
```

- **Boundary helpers** raise; they do **not** log (that would duplicate the log
  written at the top).
- **Stage functions** (`preprocess`, `train`, `evaluate`) add a `TrackingError`
  boundary around MLflow, and a narrow `ModelError` boundary around
  `model.predict`; otherwise they let typed errors propagate.
- **Up-front validation.** Required config is validated at the start of each
  stage (`load_params(..., required=...)`, `require_env(...)`) so a missing key
  or env var fails immediately with a clear message, before any expensive work.

### The one intentional broad catch

The single place that catches broadly is the stage entry point,
[`run_stage`](../src/stage_runner.py) — deliberately, at the **process
boundary**, and it never swallows:

```python
def run_stage(stage: str, main: Callable[[], None]) -> None:
    logger = get_logger(stage)
    try:
        main()
    except PipelineError as exc:
        logger.error("%s stage failed: %s", stage, exc, exc_info=True)
        sys.exit(1)
    except Exception:  # process-boundary safety net; logged, never swallowed
        logger.exception("%s stage failed with an unexpected error", stage)
        sys.exit(1)
```

- **Expected failures** (`PipelineError`) → a concise, actionable `ERROR` and
  exit `1`.
- **Unexpected failures** (bugs — e.g. a `sklearn` error) → logged as an
  *unexpected* error, distinguishing "known failure mode" from "something we did
  not anticipate", also with a full traceback and exit `1`.
- `KeyboardInterrupt` and `SystemExit` derive from `BaseException`, so they are
  **not** caught — Ctrl-C and explicit exits behave normally.

Each stage's `if __name__ == "__main__":` block is therefore just:

```python
if __name__ == "__main__":
    run_stage("train", main)
```

---

## 5. Logging strategy

Exception logging follows the [Logging Strategy](logging.md); the rules specific
to errors are:

- **Log once, at the stage boundary.** `run_stage` is the *only* place a failure
  is logged. Boundary helpers raise but do not log, so the log file contains one
  authoritative record per failure — not the same error repeated at every frame.
- **Preserve the stack trace.** The boundary logs with `exc_info=True` (via
  `logger.error(..., exc_info=True)` / `logger.exception(...)`), which emits the
  full **chained** traceback — including the original `__cause__` set by
  `raise ... from exc`.
- **`ERROR` for failures.** Failures use the `ERROR` level (see the
  [level table](logging.md#5-log-levels)). `WARNING` is reserved for recoverable
  issues that do not stop the run.
- **Actionable message, then traceback.** The first log line is the human
  message (what to do next); the traceback follows for diagnosis.

Example log record for a missing dataset:

```
2026-08-01 19:15:29 | ERROR    | train | train stage failed: Dataset not found: 'data/raw/data.csv'. If it is DVC-tracked, run 'dvc pull' to fetch it first.
Traceback (most recent call last):
  ...
FileNotFoundError: [Errno 2] No such file or directory: 'data/raw/data.csv'

The above exception was the direct cause of the following exception:
  ...
exceptions.DataError: Dataset not found: 'data/raw/data.csv'. ...
```

---

## 6. User-facing errors

The pipeline runs as command-line stages (directly or via `dvc repro`), so the
"user" is an operator reading the console and `logs/pipeline.log`. The contract:

- **No raw tracebacks as the headline.** The first thing the operator sees is a
  typed, actionable `ERROR` line — not a bare `KeyError` or `FileNotFoundError`.
- **Every message says what to do next.** Messages name the offending value
  (path, key, variable) and the remedy. Examples produced by the helpers:

  | Situation | Message |
  |-----------|---------|
  | Data file missing | `Dataset not found: 'data/raw/data.csv'. If it is DVC-tracked, run 'dvc pull' to fetch it first.` |
  | Env var unset | `Required environment variable MLFLOW_TRACKING_URI is not set. Copy .env.example to .env and set MLFLOW_TRACKING_URI (see the README), then re-run.` |
  | Model missing | `Model file not found: 'models/model.pkl'. Run the train stage first to produce it.` |
  | Wrong schema | `Dataset 'data/raw/data.csv' is missing required column(s): Outcome. Found columns: [...]` |
  | Tracking down | `MLflow tracking failed against '<uri>': <detail>. Check the tracking URI and your DagsHub credentials / network connection.` |

- **Non-zero exit code.** Every failure exits `1`, so `dvc repro` and CI treat a
  failed stage as failed instead of proceeding on partial or missing outputs.
- **Traceback stays available.** The full chained traceback is still logged
  (§5) for debugging — it is demoted below the actionable message, not removed.

---

## 7. Adding a new failure mode — checklist

When you write code that can fail:

1. **Pick the closest existing type** (`ConfigError`, `DataError`, `ModelError`,
   `TrackingError`). Add a new `PipelineError` subclass only for a genuinely new
   boundary.
2. **Catch narrowly.** Catch the specific low-level exception(s) you expect — not
   bare `Exception`.
3. **Wrap and chain.** `raise SomeError("actionable message") from exc`.
4. **Prefer a shared helper.** If the boundary is IO/config/model, add or reuse a
   helper in [`pipeline_io.py`](../src/pipeline_io.py) so every stage handles it
   identically.
5. **Do not log at the raise site.** Let `run_stage` log it once at the boundary.
6. **Validate early.** Check required config at the start of the stage so it
   fails before expensive work.

---

## Related Documentation

- [Logging Strategy](logging.md) — format, levels, and destinations.
- [Type Safety](type-safety.md) — typing conventions and the mypy configuration.
- [Architecture](architecture.md) — see §7, Observability & Logging.
- [Engineering Review](reviews/sprint-02-engineering-review.md) — finding H-2.
