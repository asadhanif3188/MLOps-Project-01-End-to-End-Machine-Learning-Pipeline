"""Operational metrics for the ephemeral pipeline Job (Sprint 8, PR 3 — ADR-030).

This is the one place the pipeline emits **operational** metrics to Prometheus.
It exists because of a hard constraint the observability architecture calls out
(ADR-028 §§ 3 and 4): Prometheus **pulls** — it scrapes a live target each interval —
but the pipeline is a `batch/v1` Job whose pod **exits** seconds after it finishes,
and each `dvc repro` stage is its own short-lived process. A `/metrics` endpoint on
the Job has nothing to scrape between (or often *within*) runs. The Prometheus-
sanctioned way to get metrics *out of* a batch job is the **Pushgateway**: the job
pushes to a persistent gateway before exiting and Prometheus scrapes the gateway.

**The ownership boundary is deliberate and narrow (ADR-028 § 5, ADR-030).** This
module emits ONLY operational signals — *how long a stage took* and *whether it
succeeded*. It must NEVER emit model accuracy, hyper-parameters, or any ML-semantic
value: those belong to **MLflow** (:mod:`tracking`), which is built for run-indexed
experiment data. Prometheus is not a second experiment database. kube-state-metrics
already answers the *run-level* questions (did the Job succeed, when, how long, was
it OOMKilled — from the persistent Job/Pod object); the one genuinely operational
signal KSM cannot give is **per-stage** granularity, which is exactly what this
module adds.

**Bounded label cardinality (the sprint brief; ADR-030).** The only label is
``stage``, drawn from the fixed set :data:`PIPELINE_STAGES`. There is deliberately
**no** run UUID, model path, dataset filename, or timestamp label — those would
grow the series set without bound and are the classic Pushgateway cardinality
foot-gun. Unknown stage names are refused rather than emitted.

**Batch metric lifecycle (ADR-030 § "Pushgateway lifecycle").** Each stage is a
distinct Pushgateway *group* keyed by ``job=mlops_pipeline`` + ``stage=<name>``, so
one stage's push (a PUT, which replaces its whole group) never clobbers another's.
At the very start of a run :func:`reset_pipeline_metrics` DELETEs every stage group
so a shorter/failed run cannot leave a *previous* run's stage series behind as
stale data — the sticky-metric hazard ADR-028 warned about. A stage that does not
run in a given execution is therefore simply *absent*, which reads correctly ("the
pipeline never got there").

**Best-effort, never fatal.** Metrics are observability, not a pipeline output: a
push failure (gateway down, package absent, a malformed HTTP response mid-roll) is
logged at WARNING and swallowed so a monitoring hiccup can never fail a real
pipeline run. The catch is deliberately **broad** (any ``Exception``), not narrow:
``prometheus_client``'s push runs over ``urllib``/``http.client``, which raises not
only ``OSError`` (connect/timeout/DNS/TLS/4xx-5xx) and ``ImportError`` (client
absent) but also ``http.client.HTTPException`` subclasses like ``BadStatusLine``
(e.g. the gateway pod rolled mid-response) that are **not** ``OSError`` — a narrow
catch would let exactly that hiccup fail a stage whose real work already succeeded.
The broad catch always logs (never a silent ``pass``), the sanctioned form here.

**Disabled unless configured.** Emission is a no-op unless ``PUSHGATEWAY_URL`` is
set, so ``dvc repro`` on a workstation, the CI fixture run, and the unit tests do
no network I/O and need nothing installed. In-cluster the URL is injected by the
base ConfigMap (``k8s/base/configmap.yaml``); ``prometheus_client`` is imported
*lazily* at push time, so importing this module never requires the package —
matching how :mod:`fetch_dataset` treats boto3 and :mod:`train` treats MLflow.
"""

import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from logging_config import get_logger

logger = get_logger("pipeline_metrics")

# Environment variable carrying the Pushgateway address (e.g.
# "http://pushgateway.monitoring:9091"). Unset/empty ⇒ emission disabled.
PUSHGATEWAY_URL_ENV = "PUSHGATEWAY_URL"

# The Pushgateway ``job`` grouping label — the stable identity of this pipeline.
# Underscored (not "mlops-pipeline") because it becomes a Prometheus label value
# and keeps the series legible; there is exactly one such job.
JOB = "mlops_pipeline"

# The fixed, bounded set of stages that may appear as the ``stage`` label. Order
# is execution order: fetch-dataset (init container) then the four dvc repro
# stages. Emitting anything outside this set is refused to protect cardinality.
PIPELINE_STAGES: tuple[str, ...] = (
    "fetch_dataset",
    "preprocess",
    "split",
    "train",
    "evaluate",
)

# Metric names. Prefixed with the project namespace so they never collide with an
# exporter's series. Both are GAUGES: a batch job reports the final value of a run,
# not a monotonically increasing counter (a fresh process per stage cannot carry a
# running total, and Pushgateway replaces rather than accumulates — see ADR-030 for
# why the brief's ``*_total`` names are represented as last-run gauges instead).
STAGE_DURATION_METRIC = "mlops_pipeline_stage_duration_seconds"
STAGE_SUCCESS_METRIC = "mlops_pipeline_stage_success"

_STAGE_DURATION_HELP = "Wall-clock seconds the pipeline stage took to run."
_STAGE_SUCCESS_HELP = (
    "Whether the pipeline stage's last run succeeded (1) or failed (0)."
)

# Bound the push/delete so a wedged gateway cannot stall a stage for long: the call
# is best-effort and any timeout is caught and logged.
_GATEWAY_TIMEOUT_SECONDS = 5.0

# Injection seams (used by tests to capture calls without a real gateway). The
# defaults do the real prometheus_client I/O; a test passes its own callable.
GatewayPush = Callable[[str, str, float, bool], None]
GatewayDelete = Callable[[str, str], None]


def resolve_gateway_url(url: str | None = None) -> str | None:
    """Return the Pushgateway URL to use, or ``None`` when emission is disabled.

    An explicit ``url`` wins; otherwise the ``PUSHGATEWAY_URL`` environment
    variable is read. An unset/blank value means "metrics disabled" — the normal
    state for local ``dvc repro``, CI, and tests.

    Args:
        url: An explicit override, or ``None`` to read the environment.

    Returns:
        The stripped, non-empty gateway URL, or ``None`` if none is configured.
    """
    value = (
        url if url is not None else os.environ.get(PUSHGATEWAY_URL_ENV, "")
    ).strip()
    return value or None


def _default_push(url: str, stage: str, duration_seconds: float, success: bool) -> None:
    """Push one stage's operational metrics to the Pushgateway (real I/O).

    Builds a fresh registry with exactly the two gauges and PUTs it under the
    ``job``/``stage`` grouping key. PUT (``push_to_gateway``) replaces the whole
    group, so re-running a stage cleanly overwrites its prior series. The ``stage``
    label comes from the grouping key (not a metric label), which keeps cardinality
    bounded and lets Prometheus attach it as a target label on scrape.
    """
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    registry = CollectorRegistry()
    Gauge(STAGE_DURATION_METRIC, _STAGE_DURATION_HELP, registry=registry).set(
        duration_seconds
    )
    Gauge(STAGE_SUCCESS_METRIC, _STAGE_SUCCESS_HELP, registry=registry).set(
        1.0 if success else 0.0
    )
    push_to_gateway(
        url,
        job=JOB,
        registry=registry,
        grouping_key={"stage": stage},
        timeout=_GATEWAY_TIMEOUT_SECONDS,
    )


def _default_delete(url: str, stage: str) -> None:
    """Delete one stage's Pushgateway group (real I/O).

    Idempotent: the Pushgateway returns success even if the group does not exist,
    so clearing on the first run is harmless.
    """
    from prometheus_client import delete_from_gateway

    delete_from_gateway(
        url, job=JOB, grouping_key={"stage": stage}, timeout=_GATEWAY_TIMEOUT_SECONDS
    )


def push_stage_metrics(
    stage: str,
    duration_seconds: float,
    *,
    success: bool,
    url: str | None = None,
    push: GatewayPush | None = None,
) -> None:
    """Emit a stage's duration and success to the Pushgateway (best-effort).

    A no-op when no gateway is configured (:func:`resolve_gateway_url` returns
    ``None``). An unknown ``stage`` is refused rather than emitted, to keep label
    cardinality bounded. Any push failure — of any kind — is logged at WARNING and
    swallowed; metrics must never fail the pipeline (see the module docstring for why
    the catch is deliberately broad).

    Args:
        stage: The stage name; must be one of :data:`PIPELINE_STAGES`.
        duration_seconds: Wall-clock seconds the stage took.
        success: Whether the stage completed successfully.
        url: Optional explicit gateway URL (defaults to ``PUSHGATEWAY_URL``).
        push: Optional injected pusher (tests); defaults to the real push.
    """
    target = resolve_gateway_url(url)
    if target is None:
        logger.debug(
            "%s not set; skipping metrics for stage %s", PUSHGATEWAY_URL_ENV, stage
        )
        return
    if stage not in PIPELINE_STAGES:
        logger.warning(
            "Refusing to emit metrics for unknown stage %r (expected one of %s)",
            stage,
            ", ".join(PIPELINE_STAGES),
        )
        return

    pusher = push or _default_push
    try:
        pusher(target, stage, float(duration_seconds), bool(success))
    except Exception as exc:  # noqa: BLE001 — deliberate best-effort sink
        # Metrics are observability, not a pipeline output, so ANY failure here is
        # logged and swallowed — a monitoring hiccup must never fail a real run. The
        # catch is deliberately broad, not narrow: prometheus_client's push path runs
        # over urllib/http.client, which can raise not just OSError (connect/timeout/
        # DNS/TLS/4xx-5xx) but also http.client.HTTPException subclasses such as
        # BadStatusLine (e.g. the gateway pod rolled mid-response) that are NOT
        # OSError — narrowing the catch would let exactly that hiccup fail a stage
        # whose real work already succeeded. Logged (not silently passed), which is
        # the sanctioned form of a broad catch here.
        logger.warning(
            "Could not push metrics for stage %s to %s: %s", stage, target, exc
        )
    else:
        logger.info(
            "Pushed stage metrics: stage=%s duration=%.3fs success=%s",
            stage,
            duration_seconds,
            success,
        )


def reset_pipeline_metrics(
    *, url: str | None = None, delete: GatewayDelete | None = None
) -> None:
    """Clear every stage's Pushgateway group at the start of a run (best-effort).

    Deleting all known stage groups before a run guarantees the group set reflects
    *this* run only: a run that fails early, or a shorter pipeline, cannot leave a
    previous run's later-stage series behind as stale data. Called once, first
    thing, by the ``fetch-dataset`` init container (:func:`fetch_dataset.main`).

    A no-op when no gateway is configured. Each delete is independent and
    best-effort; a failure to clear one group is logged and does not stop the rest
    or the pipeline.

    Args:
        url: Optional explicit gateway URL (defaults to ``PUSHGATEWAY_URL``).
        delete: Optional injected deleter (tests); defaults to the real delete.
    """
    target = resolve_gateway_url(url)
    if target is None:
        return

    deleter = delete or _default_delete
    for stage in PIPELINE_STAGES:
        try:
            deleter(target, stage)
        except Exception as exc:  # noqa: BLE001 — deliberate best-effort sink
            # Same broad-catch rationale as push_stage_metrics: this reset runs
            # first thing in the pipeline (fetch_dataset.main), so an uncaught
            # exception here would abort the whole run before any work — exactly what
            # a best-effort clear must never do. Swallow-and-continue per stage.
            logger.warning(
                "Could not clear stale metrics for stage %s at %s: %s",
                stage,
                target,
                exc,
            )
    logger.info("Cleared prior-run stage metrics at %s", target)


@contextmanager
def time_stage(
    stage: str, *, url: str | None = None, push: GatewayPush | None = None
) -> Iterator[None]:
    """Time a stage and push its duration + success on exit (best-effort).

    Wraps a block so the stage is timed with a monotonic clock and, whether it
    succeeds or raises, its duration and outcome are pushed exactly once before the
    exception (if any) propagates. This is the single instrumentation point used by
    both :func:`stage_runner.run_stage` (the four dvc stages) and
    :func:`fetch_dataset.main` (the init container).

    ``KeyboardInterrupt`` / ``SystemExit`` are :class:`BaseException`, not
    ``Exception``, so they are not treated as a stage failure — they propagate
    without emitting a spurious ``success=0`` (matching ``run_stage``'s contract
    that interpreter control-flow signals pass through untouched).

    Args:
        stage: The stage name; must be one of :data:`PIPELINE_STAGES`.
        url: Optional explicit gateway URL (defaults to ``PUSHGATEWAY_URL``).
        push: Optional injected pusher (tests); defaults to the real push.
    """
    start = time.perf_counter()
    try:
        yield
    except Exception:
        # A stage failure: record it (success=0) and re-raise so the caller's
        # error handling (run_stage / fetch_dataset.main) is unchanged. Re-raising
        # is what keeps this a transparent timer, not a swallow.
        push_stage_metrics(
            stage, time.perf_counter() - start, success=False, url=url, push=push
        )
        raise
    else:
        # Normal completion. A KeyboardInterrupt/SystemExit is a BaseException, not
        # an Exception, so it is neither caught above nor reaches this branch — it
        # propagates without emitting a spurious failure, matching run_stage's
        # contract that interpreter control-flow signals pass through untouched.
        push_stage_metrics(
            stage, time.perf_counter() - start, success=True, url=url, push=push
        )
