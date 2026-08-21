"""Bounded retry with exponential back-off — a small, auditable resilience primitive.

Sprint 8 PR 13 (reliability hardening) introduced this helper to absorb a
*transient* dependency blip — specifically a mid-run MLflow tracking-server
rolling restart — without discarding the pipeline's completed compute. The
motivating evidence is in
``docs/proof/sprint-08-mlflow-failure-tests-evidence.md`` § 8 (candidate #1); the
design of record is ``docs/decisions/ADR-037-pipeline-reliability-hardening.md``.

Why a hand-rolled, standard-library helper rather than ``tenacity``:

* **Supply-chain surface.** This module runs inside the hardened pipeline image
  whose dependencies are inventoried (SBOM) and whose provenance is verified
  (Sprint 8 PR 8/PR 9). A ~40-line, dependency-free primitive with an explicit,
  reviewable bound is cheaper to audit than a new transitive dependency for a
  single call site.
* **Explicit bounds are the whole point.** The task's reliability rules forbid
  *unbounded* retries and forbid *hiding* a persistent failure. This helper makes
  both guarantees structural: :func:`retry_call` runs a **finite** number of
  attempts and, once they are exhausted, **re-raises the last exception** — it
  never swallows a failure and never loops forever.

The primitive is deliberately generic (it knows nothing about MLflow) and takes
an injectable ``sleep`` so it is fully unit-testable offline with no clock, no
network, and no third-party dependency.
"""

import time
from collections.abc import Callable
from logging import Logger


def retry_call[T](
    operation: Callable[[], T],
    *,
    attempts: int,
    base_delay: float,
    max_delay: float,
    retry_on: tuple[type[BaseException], ...],
    factor: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
    logger: Logger | None = None,
    description: str = "operation",
) -> T:
    """Call ``operation`` up to ``attempts`` times, backing off between retries.

    A retry happens only when ``operation`` raises an exception that is an
    instance of one of the ``retry_on`` types. Any other exception propagates
    immediately (it is not a transient fault the caller asked to absorb). After
    the final attempt fails, the last exception is **re-raised unchanged** — the
    helper never hides a persistent failure and never retries without bound.

    The delay before retry *n* (1-indexed, i.e. before the 2nd attempt) is
    ``min(base_delay * factor ** (n - 1), max_delay)`` seconds, so the wait grows
    exponentially but is clamped to ``max_delay``. No delay is taken after the
    last attempt (there is nothing left to wait for).

    Args:
        operation: A zero-argument callable performing one attempt. It should be
            self-contained so that each retry is a clean, independent attempt
            (e.g. it opens its own MLflow run rather than reusing a half-built
            one).
        attempts: Maximum number of attempts (must be >= 1). ``attempts=1``
            disables retrying — ``operation`` runs exactly once.
        base_delay: The first back-off delay in seconds (used before attempt 2).
        max_delay: Upper bound on any single back-off delay in seconds.
        retry_on: Exception types that mark a *transient* failure worth retrying.
            Exceptions outside this set propagate on the first occurrence.
        factor: Exponential growth factor for the back-off (default ``2.0``).
        sleep: Injectable sleep function (defaults to :func:`time.sleep`); tests
            pass a spy so no real time passes.
        logger: Optional logger; when supplied, each retryable failure is logged
            at WARNING with the attempt number and the delay before the next try.
        description: Human-readable name of the operation for log messages.

    Returns:
        Whatever ``operation`` returns on its first successful attempt.

    Raises:
        ValueError: If ``attempts`` < 1.
        BaseException: The last exception raised by ``operation`` once all
            attempts are exhausted, or immediately for a non-``retry_on``
            exception.
    """
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}")

    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except retry_on as exc:
            last_exc = exc
            if attempt == attempts:
                # Attempts exhausted: do not sleep, do not swallow — let the
                # persistent failure surface to the caller.
                break
            delay = min(base_delay * factor ** (attempt - 1), max_delay)
            if logger is not None:
                logger.warning(
                    "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                    description,
                    attempt,
                    attempts,
                    exc,
                    delay,
                )
            sleep(delay)

    # Unreachable unless the loop broke on the final attempt, in which case
    # last_exc is the exhausting failure. Re-raise it so the caller sees the real,
    # persistent error (never a synthetic "gave up" message).
    assert last_exc is not None  # invariant: the loop only breaks holding an exc
    raise last_exc
