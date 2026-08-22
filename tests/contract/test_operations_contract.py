"""Contract tests: the operational runbooks stay complete and cross-linked.

The Sprint 8 PR 14 runbooks (``docs/runbooks/``) turn real failure evidence into
recovery procedures, and ``docs/alerting.md`` § 4 maps each alert to the deep
runbook for it. Nothing enforced that structure — so a renamed runbook, a
half-templated new one, or an alert added without a procedure would all pass CI
while leaving an operator with a dead link at 3 a.m.

These checks assert the operations layer holds together:

* every expected runbook file exists and is listed in the index (both ways);
* every runbook follows the nine-section template (the promise the index makes);
* every alert in ``alerts.yml`` maps to a runbook that exists; and
* every *relative* Markdown link inside a runbook resolves — the file is present,
  and (for links into ``docs/alerting.md`` or a sibling runbook) the ``#anchor``
  resolves to a real heading.

Pure Markdown/YAML parsing — no cluster, no network. Runs in the ordinary
``pytest`` gate with the other ``contract`` tests.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNBOOKS_DIR = _REPO_ROOT / "docs/runbooks"
_README = _RUNBOOKS_DIR / "README.md"
_ALERTS_YML = _REPO_ROOT / "k8s/monitoring/base/prometheus/alerts.yml"
_ALERTING_MD = _REPO_ROOT / "docs/alerting.md"

# The failure-mode runbooks that must exist (README excluded — it is the index).
EXPECTED_RUNBOOKS = frozenset(
    {
        "platform-health.md",
        "pipeline-failure.md",
        "dataset-retrieval-failure.md",
        "dataset-integrity-failure.md",
        "mlflow-unavailable.md",
        "postgresql-failure.md",
        "oomkilled.md",
        "crash-restart.md",
    }
)

# The nine sections every runbook promises (docs/runbooks/README.md § "The
# template"). Matched as level-2 headings, case-insensitively.
REQUIRED_SECTIONS = (
    "Purpose",
    "Symptoms",
    "Detection",
    "Initial checks",
    "Diagnosis",
    "Likely causes",
    "Remediation",
    "Recovery verification",
    "Escalation / known limitations",
)

# A Markdown inline link: [text](target). Reference-style/auto links are not used
# in these docs, so the inline form is sufficient.
_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def _github_slug(heading: str) -> str:
    """The anchor GitHub derives from a heading (see the observability suite)."""
    text = heading.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return text.replace(" ", "-")


def _headings(markdown: str) -> list[str]:
    """ATX heading texts, skipping fenced code blocks."""
    out: list[str] = []
    in_fence = False
    for line in markdown.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = re.match(r"^#{1,6}\s+(.*?)\s*$", line)
        if m:
            out.append(m.group(1))
    return out


def _anchors(path: Path) -> set[str]:
    """The set of ``#anchor`` slugs a Markdown file exposes."""
    return {_github_slug(h) for h in _headings(path.read_text(encoding="utf-8"))}


@pytest.fixture(scope="module")
def alert_names() -> set[str]:
    doc = yaml.safe_load(_ALERTS_YML.read_text(encoding="utf-8")) or {}
    names = {
        rule["alert"]
        for group in (doc.get("groups", []) or [])
        for rule in (group.get("rules", []) or [])
        if rule.get("alert")
    }
    assert names, "alerts.yml declares no alerts"
    return names


# ---------------------------------------------------------------------------
# The runbook set exists and the index agrees with the filesystem.
# ---------------------------------------------------------------------------


@pytest.mark.contract
def test_expected_runbooks_exist() -> None:
    """Every runbook the platform promises is present on disk."""
    present = {p.name for p in _RUNBOOKS_DIR.glob("*.md")} - {"README.md"}
    missing = EXPECTED_RUNBOOKS - present
    unexpected = present - EXPECTED_RUNBOOKS
    assert not missing, (
        f"expected runbooks missing from docs/runbooks/: {sorted(missing)}"
    )
    assert not unexpected, (
        f"undocumented runbook file(s) present (add to the index + this contract): "
        f"{sorted(unexpected)}"
    )


@pytest.mark.contract
def test_readme_index_lists_every_runbook_both_ways() -> None:
    """The index links to exactly the runbooks that exist — no dead row, no orphan.

    A runbook missing from the index is undiscoverable; an index row pointing at a
    deleted runbook is a dead link. Both fail here.
    """
    linked = {
        target
        for target in _LINK.findall(_README.read_text(encoding="utf-8"))
        if target.endswith(".md") and "/" not in target and target != "README.md"
    }
    assert linked == set(EXPECTED_RUNBOOKS), (
        f"README index links {sorted(linked)} but expected {sorted(EXPECTED_RUNBOOKS)}"
    )


@pytest.mark.contract
@pytest.mark.parametrize("runbook", sorted(EXPECTED_RUNBOOKS))
def test_runbook_follows_the_nine_section_template(runbook: str) -> None:
    """Each runbook carries all nine template sections as level-2 headings.

    "A runbook is not complete if it ends at 'restart the pod'" — the template
    exists to force a **Recovery verification** section, so its absence (or any
    other section's) is a real regression, not a style nit.
    """
    headings = {
        h.lower() for h in _headings((_RUNBOOKS_DIR / runbook).read_text("utf-8"))
    }
    missing = [s for s in REQUIRED_SECTIONS if s.lower() not in headings]
    assert not missing, f"{runbook} is missing template section(s): {missing}"


# ---------------------------------------------------------------------------
# Every alert maps to a runbook that exists (docs/alerting.md § 4).
# ---------------------------------------------------------------------------


@pytest.mark.contract
def test_every_alert_maps_to_an_existing_runbook(alert_names: set[str]) -> None:
    """``docs/alerting.md`` § 4 gives every alert at least one deep runbook, and
    every runbook it names exists.

    This is the "alerts reference current runbooks" contract: the mapping table is
    the operator's index from *alert fired* to *procedure*, so a missing row or a
    link to a deleted runbook is an operational gap.
    """
    text = _ALERTING_MD.read_text(encoding="utf-8")
    # § 4 rows look like: | `AlertName` | … | [Label](runbooks/<file>.md) · … |
    mapped: dict[str, set[str]] = {}
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        alert_match = re.search(r"`([A-Za-z]+)`", line)
        if not alert_match or alert_match.group(1) not in alert_names:
            continue
        targets = {
            t.split("#", 1)[0]
            for t in _LINK.findall(line)
            if t.startswith("runbooks/") and t.split("#", 1)[0].endswith(".md")
        }
        if targets:
            mapped.setdefault(alert_match.group(1), set()).update(targets)

    unmapped = alert_names - set(mapped)
    assert not unmapped, (
        f"alerts with no runbook in docs/alerting.md § 4: {sorted(unmapped)}"
    )
    broken = {
        f"{alert} -> {target}"
        for alert, targets in mapped.items()
        for target in targets
        if not (_REPO_ROOT / "docs" / target).is_file()
    }
    assert not broken, (
        f"alerting.md § 4 links to non-existent runbooks: {sorted(broken)}"
    )


# ---------------------------------------------------------------------------
# No dead relative links inside the runbooks.
# ---------------------------------------------------------------------------


@pytest.mark.contract
@pytest.mark.parametrize("runbook", sorted(EXPECTED_RUNBOOKS | {"README.md"}))
def test_runbook_relative_links_resolve(runbook: str) -> None:
    """Every relative Markdown link in a runbook resolves to a real target.

    The file must exist; and for links into ``docs/alerting.md``, a sibling
    runbook, or the index, the ``#anchor`` must resolve to a real heading too
    (where the alert<->runbook cross-links live and rot). Links into the large
    evidence/decision docs are checked for file existence only — their headings
    carry emoji/markup that make slug matching unreliable, and file existence is
    the regression that actually matters there.
    """
    path = _RUNBOOKS_DIR / runbook
    base = path.parent
    # Files whose anchors we verify (stable, hand-authored headings).
    anchor_checked = (
        {_ALERTING_MD.resolve()}
        | {(_RUNBOOKS_DIR / r).resolve() for r in EXPECTED_RUNBOOKS}
        | {_README.resolve()}
    )

    dead_files: list[str] = []
    dead_anchors: list[str] = []
    for target in _LINK.findall(path.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        file_part, _, fragment = target.partition("#")
        if not file_part:
            # A pure in-page anchor (#section) — resolve against this file.
            if fragment and fragment not in _anchors(path):
                dead_anchors.append(target)
            continue
        resolved = (base / file_part).resolve()
        if not resolved.is_file():
            dead_files.append(target)
            continue
        if (
            fragment
            and resolved in anchor_checked
            and fragment not in _anchors(resolved)
        ):
            dead_anchors.append(target)

    assert not dead_files, f"{runbook} links to missing file(s): {dead_files}"
    assert not dead_anchors, f"{runbook} links to missing anchor(s): {dead_anchors}"
