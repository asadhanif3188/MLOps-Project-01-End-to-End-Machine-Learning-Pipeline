"""Contract tests: the Sprint 8 observability artifacts agree with each other.

The Kubernetes-render validator (``k8s/validate.py``) already proves the
monitoring *manifests* are well-formed, hardened, and wired (scrape config,
Grafana provisioning, the alert-rule ConfigMap), and ``promtool`` unit-tests the
alert *logic*. Those gates each see a single artifact type. This suite asserts
the property none of them can: that the observability pieces spread across
**code, manifests, and docs** still describe the *same* system.

The failure this guards is silent drift between artifacts that no single-file
gate notices — e.g. an alert added to ``alerts.yml`` but never documented, a
``runbook_url`` whose anchor was renamed in ``docs/alerting.md``, or the pipeline
code renaming the metric its dashboards and alerts query. Each of those keeps
every existing gate green while breaking the platform's observability.

Pure parsing (YAML/JSON/Markdown) plus one import of the pipeline's own metrics
module — no cluster, no network, no kustomize, no credentials. Runs in the
ordinary ``pytest`` gate alongside the other ``contract`` tests.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ALERTS_YML = _REPO_ROOT / "k8s/monitoring/base/prometheus/alerts.yml"
_ALERTING_MD = _REPO_ROOT / "docs/alerting.md"
_OBSERVABILITY_MD = _REPO_ROOT / "docs/observability.md"
_DASHBOARDS_DIR = _REPO_ROOT / "k8s/monitoring/base/grafana/dashboards"

# The canonical Sprint 8 alert set (docs/observability.md § 6 + ADR-033). It is
# duplicated in k8s/validate.py M11 (which checks it against the rendered
# ConfigMap); here it anchors the CROSS-FILE bindings below. Bump deliberately, in
# lock-step with alerts.yml, docs/alerting.md, and the runbooks.
EXPECTED_ALERTS = frozenset(
    {
        "PipelineJobFailed",
        "PipelineJobOOMKilled",
        "MLflowDown",
        "MLflowMemoryHigh",
        "PostgresDown",
        "PostgresPVCAlmostFull",
        "PostgresMemoryHigh",
        "KubePodCrashLooping",
    }
)

# The three purpose-built dashboards (ADR-032), by the stable uid the datasource +
# docs reference. Keyed by the on-disk file so a rename is caught too.
EXPECTED_DASHBOARD_UIDS = frozenset(
    {
        "mlops-eks-platform-health",
        "mlops-pipeline-operations",
        "mlops-mlflow-platform-health",
    }
)


def _github_slug(heading: str) -> str:
    """Slug GitHub derives from a Markdown heading (for ``#anchor`` links).

    Lower-cases, drops every character that is not a word char, space, or hyphen,
    then turns spaces into hyphens. Matches GitHub's own anchor generation closely
    enough for the ASCII headings this repo uses (e.g. ``PipelineJobFailed`` ->
    ``pipelinejobfailed``; ``Escalation / known limitations`` ->
    ``escalation--known-limitations``).
    """
    text = heading.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return text.replace(" ", "-")


def _headings(markdown: str) -> list[tuple[int, str]]:
    """Every ATX heading in a Markdown doc as ``(level, text)`` pairs.

    Fenced code blocks are skipped so a ``#`` comment inside a shell example is
    never mistaken for a heading.
    """
    out: list[tuple[int, str]] = []
    in_fence = False
    for line in markdown.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = re.match(r"^(#{1,6})\s+(.*?)\s*$", line)
        if m:
            out.append((len(m.group(1)), m.group(2)))
    return out


@pytest.fixture(scope="module")
def alert_rules() -> list[dict[str, Any]]:
    """Every alert rule parsed from the packaged ``alerts.yml``."""
    doc = yaml.safe_load(_ALERTS_YML.read_text(encoding="utf-8")) or {}
    rules = [
        rule
        for group in (doc.get("groups", []) or [])
        for rule in (group.get("rules", []) or [])
        if rule.get("alert")
    ]
    assert rules, "alerts.yml declares no alert rules"
    return rules


@pytest.fixture(scope="module")
def alerting_md() -> str:
    return _ALERTING_MD.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The alert set is one thing, described in three places.
# ---------------------------------------------------------------------------


@pytest.mark.contract
def test_alert_rule_names_match_the_canonical_set(
    alert_rules: list[dict[str, Any]],
) -> None:
    """``alerts.yml`` declares exactly the documented alert set — no more, no less.

    A new rule slipped into the file (or one quietly deleted) drifts from the
    documented contract and from the dashboards/runbooks keyed to it.
    """
    names = sorted(r["alert"] for r in alert_rules)
    assert len(names) == len(set(names)), (
        f"duplicate alert names in alerts.yml: {names}"
    )
    assert set(names) == EXPECTED_ALERTS, (
        f"alerts.yml set {names} != canonical {sorted(EXPECTED_ALERTS)}"
    )


@pytest.mark.contract
def test_alerting_md_documents_exactly_the_alert_rules(
    alert_rules: list[dict[str, Any]], alerting_md: str
) -> None:
    """Every alert has a ``### <Alert>`` section in ``docs/alerting.md`` § 3, and
    that per-alert section list is exactly the rule set.

    Binds the human runbook doc to the machine rule file: an alert without a
    documented section (or a stale section for a deleted alert) fails here.
    """
    # § 3 "Runbook — per alert" holds one level-3 heading per alert; other level-3
    # headings in the doc are alerts too (only §3 uses ### for alert names). Collect
    # the level-3 headings whose text is a single CamelCase token (an alert name).
    documented = {
        text
        for level, text in _headings(alerting_md)
        if level == 3 and re.fullmatch(r"[A-Za-z]+", text)
    }
    rule_names = {r["alert"] for r in alert_rules}
    assert documented == rule_names, (
        f"docs/alerting.md § 3 documents {sorted(documented)} but alerts.yml has "
        f"{sorted(rule_names)} — the two must list the same alerts"
    )


@pytest.mark.contract
def test_every_alert_runbook_url_anchor_resolves(
    alert_rules: list[dict[str, Any]], alerting_md: str
) -> None:
    """Each rule's ``runbook_url`` points at a real ``docs/alerting.md`` anchor.

    ``k8s/validate.py`` M11 checks the URL merely *contains* ``docs/alerting.md``;
    this checks the ``#fragment`` actually resolves to a heading in that file, so a
    renamed section can no longer leave every alert pointing at a dead link.
    """
    valid_anchors = {_github_slug(text) for _, text in _headings(alerting_md)}
    broken: list[str] = []
    for rule in alert_rules:
        url = (rule.get("annotations", {}) or {}).get("runbook_url", "")
        assert "docs/alerting.md#" in url, (
            f"{rule['alert']}: runbook_url {url!r} does not target docs/alerting.md#…"
        )
        fragment = url.split("docs/alerting.md#", 1)[1]
        if fragment not in valid_anchors:
            broken.append(f"{rule['alert']} -> #{fragment}")
    assert not broken, (
        "runbook_url anchors with no matching heading in docs/alerting.md: "
        + ", ".join(broken)
    )


# ---------------------------------------------------------------------------
# Pipeline metrics wiring: the name the code emits == the name everything queries.
# ---------------------------------------------------------------------------


@pytest.mark.contract
def test_pipeline_metric_names_are_consumed_by_alerts_and_dashboards() -> None:
    """The metric names ``pipeline_metrics`` emits are exactly what the alert file,
    the pipeline dashboard, and the observability doc query.

    The single highest-value binding here: rename ``STAGE_SUCCESS_METRIC`` in the
    code and this fails, instead of the dashboards/alerts silently going blank on
    the next deploy. Imports the pipeline's own module (``pythonpath = src``) so the
    name is read from the source of truth, never re-typed.
    """
    import pipeline_metrics as pm

    success = pm.STAGE_SUCCESS_METRIC
    duration = pm.STAGE_DURATION_METRIC
    assert success == "mlops_pipeline_stage_success"
    assert duration == "mlops_pipeline_stage_duration_seconds"

    alerts_text = _ALERTS_YML.read_text(encoding="utf-8")
    dashboard_text = (_DASHBOARDS_DIR / "mlops-pipeline-operations.json").read_text(
        encoding="utf-8"
    )
    observability_text = _OBSERVABILITY_MD.read_text(encoding="utf-8")

    # success feeds PipelineJobFailed triage + the stage-status tiles.
    assert success in alerts_text, (
        f"{success} (emitted by pipeline_metrics) is never referenced in alerts.yml"
    )
    assert success in dashboard_text, (
        f"{success} is not queried by the pipeline dashboard"
    )
    # duration drives the per-stage duration panels + the docs signal catalogue.
    assert duration in dashboard_text, (
        f"{duration} is not queried by the pipeline dashboard"
    )
    assert duration in observability_text, (
        f"{duration} is not documented in docs/observability.md"
    )


@pytest.mark.contract
def test_metric_stage_domain_matches_the_pipeline_stages() -> None:
    """The bounded ``stage`` label set equals the fetch step plus the DVC stages.

    ``pipeline_metrics.PIPELINE_STAGES`` is the cardinality allow-list (ADR-030); it
    must stay in step with the actual pipeline (``dvc.yaml`` stages + the
    ``fetch-dataset`` init step). Rename or add a DVC stage without updating the
    metric domain and a real stage would be refused emission — caught here.
    """
    import pipeline_metrics as pm

    dvc_stages = set(
        (yaml.safe_load((_REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8")) or {})
        .get("stages", {})
        .keys()
    )
    assert dvc_stages, "dvc.yaml declares no stages"
    # fetch_dataset is the init-container stage that has no DVC entry but does emit.
    expected_domain = {"fetch_dataset"} | dvc_stages
    assert set(pm.PIPELINE_STAGES) == expected_domain, (
        f"pipeline_metrics.PIPELINE_STAGES {sorted(pm.PIPELINE_STAGES)} != "
        f"fetch_dataset + dvc.yaml stages {sorted(expected_domain)}"
    )


# ---------------------------------------------------------------------------
# Grafana dashboards: the provisioned set is the documented set.
# ---------------------------------------------------------------------------


@pytest.mark.contract
def test_dashboard_uids_are_the_documented_three() -> None:
    """Exactly the three ADR-032 dashboards are present, each with its stable uid.

    The uid is the handle docs and deep links use (``/d/<uid>``); a changed or
    duplicated uid silently breaks every reference. ``k8s/validate.py`` M10
    parse-validates each file — this pins the *set of uids* the platform ships.
    """
    files = sorted(_DASHBOARDS_DIR.glob("*.json"))
    uids: list[str] = []
    for f in files:
        dash = json.loads(f.read_text(encoding="utf-8"))
        uid = dash.get("uid")
        assert uid, f"{f.name} has no uid"
        uids.append(uid)
    assert len(uids) == len(set(uids)), f"duplicate dashboard uids: {uids}"
    assert set(uids) == set(EXPECTED_DASHBOARD_UIDS), (
        f"dashboard uids {sorted(uids)} != documented {sorted(EXPECTED_DASHBOARD_UIDS)}"
    )


@pytest.mark.contract
def test_documented_dashboard_uids_exist_on_disk() -> None:
    """Every dashboard uid referenced in ``docs/`` resolves to a real dashboard.

    Guards the doc→dashboard direction: a runbook or ops guide that deep-links
    ``/d/<uid>`` must point at a dashboard the stack actually provisions.
    """
    on_disk = {
        json.loads(f.read_text(encoding="utf-8")).get("uid")
        for f in _DASHBOARDS_DIR.glob("*.json")
    }
    referenced: set[str] = set()
    for md in (_REPO_ROOT / "docs").rglob("*.md"):
        for uid in EXPECTED_DASHBOARD_UIDS:
            if uid in md.read_text(encoding="utf-8"):
                referenced.add(uid)
    missing = referenced - on_disk
    assert not missing, (
        f"docs reference dashboard uids with no on-disk dashboard: {sorted(missing)}"
    )
