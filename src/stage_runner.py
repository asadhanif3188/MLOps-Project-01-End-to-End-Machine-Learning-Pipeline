"""Uniform top-level error handling for pipeline stage entry points.

Each stage's ``if __name__ == "__main__":`` block delegates to :func:`run_stage`,
which runs the stage's ``main`` and turns any failure into a single, well-formed
log record plus a non-zero exit code. This gives every stage the *same* outermost
behavior: a failed stage stops ``dvc repro`` and CI with an actionable error
instead of leaking a raw traceback to stdout.

This is the one place the pipeline catches broadly — deliberately, at the process
boundary — and it never swallows: everything is logged with the full traceback
and re-surfaced as a failure exit code.
"""

import sys
from collections.abc import Callable

from exceptions import PipelineError
from logging_config import get_logger
from pipeline_metrics import time_stage


def run_stage(stage: str, main: Callable[[], None]) -> None:
    """Execute a stage's ``main`` callable with standardized error handling.

    Expected failures (:class:`~exceptions.PipelineError` subclasses) are logged
    as a concise, actionable ``ERROR``; anything else is logged as an unexpected
    error (a likely bug). Both preserve the full traceback via ``exc_info`` and
    exit with status ``1``. ``KeyboardInterrupt`` and ``SystemExit`` are *not*
    caught, so Ctrl-C and explicit exits behave normally.

    As a side effect, the stage's operational metrics (wall-clock duration and
    success/failure) are pushed to the Prometheus Pushgateway via
    :func:`pipeline_metrics.time_stage` — a no-op unless ``PUSHGATEWAY_URL`` is set,
    and best-effort so a monitoring outage never changes the outcome here
    (Sprint 8, PR 3 — ADR-030).

    Args:
        stage: Stage name, used as the logger name (e.g. ``"train"``).
        main: Zero-argument callable that runs the stage end to end.
    """
    logger = get_logger(stage)
    try:
        # time_stage wraps the run to emit the stage's operational metrics
        # (duration + success/failure) to the Pushgateway on exit — a no-op unless
        # PUSHGATEWAY_URL is set, and best-effort so it never changes this
        # function's failure behaviour (ADR-030). It re-raises any stage failure
        # unchanged, so the except arms below are reached exactly as before.
        with time_stage(stage):
            main()
    except PipelineError as exc:
        # Expected failure: the message is already actionable; log it once here
        # (with the chained cause) rather than at every boundary it passed.
        logger.error("%s stage failed: %s", stage, exc, exc_info=True)
        sys.exit(1)
    except Exception:
        logger.exception("%s stage failed with an unexpected error", stage)
        sys.exit(1)
