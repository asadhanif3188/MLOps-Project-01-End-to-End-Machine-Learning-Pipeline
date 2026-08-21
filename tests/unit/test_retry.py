"""Unit tests for :mod:`retry` — the bounded exponential-back-off primitive.

These pin the reliability guarantees Sprint 8 PR 13 depends on and that the
task's rules require the tracking retry to honour:

* a *transient* failure is absorbed within the bound (the improvement);
* a *persistent* failure is **re-raised**, never swallowed (rule: do not hide a
  persistent dependency failure);
* retrying is **finite** — a forever-failing operation stops after ``attempts``
  and does not loop without bound (rule: no infinite retries);
* an exception outside ``retry_on`` propagates immediately (a deterministic error
  is not retried).

The suite injects a spy ``sleep`` so no real time passes and the back-off
schedule can be asserted directly, and it uses no MLflow, network, or clock —
exactly the properties the pipeline's unit contract requires.
"""

import pytest

from retry import retry_call

pytestmark = pytest.mark.unit


class _Transient(Exception):
    """A retryable, transient-style failure."""


class _Permanent(Exception):
    """A non-retryable failure (outside ``retry_on``)."""


class _Op:
    """A callable that fails ``fail_times`` times, then returns ``result``.

    Records how many times it was invoked so tests can assert the exact attempt
    count (proving the bound and that retries stop on first success).
    """

    def __init__(
        self,
        *,
        fail_times: int,
        exc: Exception | None = None,
        result: str = "ok",
    ):
        self.fail_times = fail_times
        self.exc = exc or _Transient("boom")
        self.result = result
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc
        return self.result


@pytest.fixture
def sleeps() -> list[float]:
    """Collects the delays passed to the injected ``sleep`` spy."""
    return []


def _sleep_spy(store: list[float]):
    def _sleep(delay: float) -> None:
        store.append(delay)

    return _sleep


# --------------------------------------------------------------------------- #
# Success paths
# --------------------------------------------------------------------------- #
def test_returns_immediately_on_first_success(sleeps: list[float]) -> None:
    op = _Op(fail_times=0)
    result = retry_call(
        op,
        attempts=5,
        base_delay=5.0,
        max_delay=30.0,
        retry_on=(_Transient,),
        sleep=_sleep_spy(sleeps),
    )
    assert result == "ok"
    assert op.calls == 1  # no retry when the first attempt succeeds
    assert sleeps == []  # and therefore no back-off waits


def test_absorbs_transient_failures_then_succeeds(sleeps: list[float]) -> None:
    # THE IMPROVEMENT: two transient failures (a mid-run MLflow blip) are ridden
    # out and the operation ultimately succeeds — the caller never sees an error.
    op = _Op(fail_times=2)
    result = retry_call(
        op,
        attempts=5,
        base_delay=5.0,
        max_delay=30.0,
        retry_on=(_Transient,),
        sleep=_sleep_spy(sleeps),
    )
    assert result == "ok"
    assert op.calls == 3  # 2 failures + 1 success
    assert sleeps == [5.0, 10.0]  # backed off before each retry, exponential


# --------------------------------------------------------------------------- #
# Failure / bound paths
# --------------------------------------------------------------------------- #
def test_reraises_last_exception_after_exhausting_attempts(
    sleeps: list[float],
) -> None:
    # A PERSISTENT failure is NOT hidden: the underlying exception surfaces so the
    # caller (tracking.py) can convert it to a clear TrackingError and fail the run.
    sentinel = _Transient("still down")
    op = _Op(fail_times=99, exc=sentinel)
    with pytest.raises(_Transient) as excinfo:
        retry_call(
            op,
            attempts=3,
            base_delay=5.0,
            max_delay=30.0,
            retry_on=(_Transient,),
            sleep=_sleep_spy(sleeps),
        )
    assert excinfo.value is sentinel


def test_retrying_is_finite(sleeps: list[float]) -> None:
    # NO INFINITE RETRIES: a forever-failing op is attempted exactly ``attempts``
    # times and then stops — never an unbounded loop.
    op = _Op(fail_times=10**6)
    with pytest.raises(_Transient):
        retry_call(
            op,
            attempts=4,
            base_delay=5.0,
            max_delay=30.0,
            retry_on=(_Transient,),
            sleep=_sleep_spy(sleeps),
        )
    assert op.calls == 4  # exactly the bound
    assert len(sleeps) == 3  # no sleep after the final attempt


def test_non_retryable_exception_propagates_immediately(sleeps: list[float]) -> None:
    # A deterministic error outside ``retry_on`` is not retried at all.
    op = _Op(fail_times=1, exc=_Permanent("bad request"))
    with pytest.raises(_Permanent):
        retry_call(
            op,
            attempts=5,
            base_delay=5.0,
            max_delay=30.0,
            retry_on=(_Transient,),
            sleep=_sleep_spy(sleeps),
        )
    assert op.calls == 1  # failed once, not retried
    assert sleeps == []


# --------------------------------------------------------------------------- #
# Back-off schedule
# --------------------------------------------------------------------------- #
def test_backoff_is_exponential_and_clamped_to_max_delay(sleeps: list[float]) -> None:
    op = _Op(fail_times=10**6)
    with pytest.raises(_Transient):
        retry_call(
            op,
            attempts=6,
            base_delay=5.0,
            max_delay=30.0,
            factor=2.0,
            retry_on=(_Transient,),
            sleep=_sleep_spy(sleeps),
        )
    # 5, 10, 20, then clamped at 30 (would be 40, 80) — bounded growth.
    assert sleeps == [5.0, 10.0, 20.0, 30.0, 30.0]


def test_attempts_one_runs_once_without_retrying(sleeps: list[float]) -> None:
    op = _Op(fail_times=10**6)
    with pytest.raises(_Transient):
        retry_call(
            op,
            attempts=1,
            base_delay=5.0,
            max_delay=30.0,
            retry_on=(_Transient,),
            sleep=_sleep_spy(sleeps),
        )
    assert op.calls == 1
    assert sleeps == []


def test_invalid_attempts_raises_value_error() -> None:
    with pytest.raises(ValueError, match="attempts must be >= 1"):
        retry_call(
            lambda: "ok",
            attempts=0,
            base_delay=5.0,
            max_delay=30.0,
            retry_on=(_Transient,),
        )
