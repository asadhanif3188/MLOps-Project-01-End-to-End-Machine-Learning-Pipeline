"""Contract tests: the CI workflow keeps the Sprint 8 supply-chain + gate steps.

Sprint 8 hardened the supply chain (PRs 8-9) and stood up the observability,
network, and IaC gates. All of that lives as *steps* in one file,
``.github/workflows/ci.yml`` — and a refactor of that file could quietly drop the
image scan, the SBOM, or the provenance check and still leave a green,
plausible-looking pipeline. Nothing else in the repo notices a *missing* CI step.

These tests parse the workflow **semantically** (as YAML — the parsed job/step
graph, not a raw ``grep`` over the text) and assert the load-bearing steps are
still there and still gating:

* the container-image vulnerability scan exists and *blocks* the build;
* a CycloneDX SBOM is generated;
* image provenance (git commit -> image) is verified;
* the alert-rule, manifest, IaC, and test gates that protect Sprint 8 remain
  wired; and
* CI keeps least privilege (read-only token, no publish scope).

Pure YAML parsing — no Docker, no network, no credentials. Runs in the ordinary
``pytest`` gate with the other ``contract`` tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CI_YML = _REPO_ROOT / ".github/workflows/ci.yml"


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    data = yaml.safe_load(_CI_YML.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "ci.yml did not parse to a mapping"
    return data


def _jobs(workflow: dict[str, Any]) -> dict[str, Any]:
    jobs = workflow.get("jobs")
    assert isinstance(jobs, dict) and jobs, "ci.yml declares no jobs"
    return jobs


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in (job.get("steps", []) or []) if isinstance(s, dict)]


def _run_scripts(job: dict[str, Any]) -> str:
    """All ``run:`` shell of a job concatenated — the semantic body to assert on."""
    return "\n".join(s.get("run", "") for s in _steps(job) if s.get("run"))


def _step_with(job: dict[str, Any], *needles: str) -> dict[str, Any] | None:
    """The first step whose ``run`` contains every needle (a single logical step)."""
    for step in _steps(job):
        run = step.get("run", "") or ""
        if all(n in run for n in needles):
            return step
    return None


# ---------------------------------------------------------------------------
# The job graph exists.
# ---------------------------------------------------------------------------


@pytest.mark.contract
def test_expected_ci_jobs_are_present(workflow: dict[str, Any]) -> None:
    """The Sprint 8 gate jobs all exist (renaming one silently drops its gate)."""
    jobs = set(_jobs(workflow))
    required = {
        "quality",  # lint + tests (runs these contracts)
        "docker",  # build + scan + SBOM + provenance
        "k8s-validate",  # manifest security/observability contract
        "prometheus-rules",  # alert-rule validity
        "terraform-validate",  # IaC + security-regression contract suite
    }
    missing = required - jobs
    assert not missing, f"ci.yml is missing required job(s): {sorted(missing)}"


# ---------------------------------------------------------------------------
# Supply chain: image scan (gating) + SBOM + provenance.
# ---------------------------------------------------------------------------


@pytest.mark.contract
def test_image_vulnerability_scan_exists_and_gates(workflow: dict[str, Any]) -> None:
    """The docker job scans the built image and *fails the build* on findings.

    Existence is not enough: a scan that only reports would let a vulnerable image
    through. The gate is proven by a ``trivy image`` invocation with ``--exit-code
    1`` (a non-zero exit fails the step) scanning HIGH/CRITICAL severities.
    """
    docker = _jobs(workflow).get("docker")
    assert docker, "no docker job in ci.yml"
    gate = _step_with(docker, "trivy image", "--exit-code 1")
    assert gate is not None, (
        "no gating `trivy image … --exit-code 1` step in the docker job "
        "(the image vulnerability scan must block the build, not just report)"
    )
    run = gate["run"]
    assert "--ignore-unfixed" in run, (
        "the image scan gate should key on FIXABLE findings (--ignore-unfixed) so "
        "it stays actionable (ADR-035)"
    )
    # Severity is set via the TRIVY_SEVERITY env (workflow/job/step level) or an
    # inline flag; fold all of them in so either form proves the scan is scoped to
    # the serious findings.
    body = (
        _run_scripts(docker)
        + str(workflow.get("env", {}))
        + str(docker.get("env", {}))
        + "".join(str(s.get("env", {})) for s in _steps(docker))
    )
    assert "HIGH,CRITICAL" in body or "CRITICAL,HIGH" in body, (
        "the image scan must target HIGH,CRITICAL severities"
    )


@pytest.mark.contract
def test_sbom_generation_exists(workflow: dict[str, Any]) -> None:
    """A CycloneDX SBOM is generated for the built image(s)."""
    docker = _jobs(workflow).get("docker")
    assert docker, "no docker job in ci.yml"
    sbom = _step_with(docker, "trivy image", "--format cyclonedx")
    assert sbom is not None, (
        "no SBOM step in the docker job (expected `trivy image --format cyclonedx …`)"
    )
    # An SBOM with zero components is a generation failure, not a clean result — the
    # step must guard against it rather than uploading an empty inventory.
    assert "components" in sbom["run"], (
        "the SBOM step should assert a non-empty component inventory "
        "(a zero-component SBOM is a silent generation failure)"
    )


@pytest.mark.contract
def test_image_provenance_is_verified(workflow: dict[str, Any]) -> None:
    """The git commit -> image binding is verified via the OCI revision label.

    The first link of the supply chain: the built image must embed the exact
    commit SHA in ``org.opencontainers.image.revision``. A Dockerfile change that
    dropped the label would break provenance — caught here in CI.
    """
    docker = _jobs(workflow).get("docker")
    assert docker, "no docker job in ci.yml"
    prov = _step_with(docker, "org.opencontainers.image.revision", "GITHUB_SHA")
    assert prov is not None, (
        "no provenance step verifying the image's OCI revision label equals the "
        "commit SHA (the git->image binding, ADR-036)"
    )


# ---------------------------------------------------------------------------
# The Sprint 8 gates that protect observability / manifests / IaC stay wired.
# ---------------------------------------------------------------------------


@pytest.mark.contract
def test_alert_rules_are_validated_and_unit_tested(workflow: dict[str, Any]) -> None:
    """The prometheus-rules job runs both promtool passes (structure + logic)."""
    job = _jobs(workflow).get("prometheus-rules")
    assert job, "no prometheus-rules job in ci.yml"
    body = _run_scripts(job)
    assert "promtool check rules" in body, (
        "prometheus-rules must run `promtool check rules` (structural validity)"
    )
    assert "promtool test rules" in body, (
        "prometheus-rules must run `promtool test rules` (the alert-logic unit tests)"
    )


@pytest.mark.contract
def test_manifest_security_contract_runs(workflow: dict[str, Any]) -> None:
    """The k8s-validate job runs the project manifest validator."""
    job = _jobs(workflow).get("k8s-validate")
    assert job, "no k8s-validate job in ci.yml"
    assert "k8s/validate.py" in _run_scripts(job), (
        "k8s-validate must run `python k8s/validate.py` (the manifest security + "
        "observability contract, incl. NetworkPolicy least-privilege)"
    )


@pytest.mark.contract
def test_terraform_security_contract_suite_runs(workflow: dict[str, Any]) -> None:
    """The terraform-validate job runs ``terraform test`` (the IaC contract suite).

    The EKS access-model, Pod-Identity/CNI, and KMS-encryption regressions are
    enforced by ``terraform/tests/*.tftest.hcl``; if this step were dropped, those
    contracts would stop running while CI stayed green.
    """
    job = _jobs(workflow).get("terraform-validate")
    assert job, "no terraform-validate job in ci.yml"
    assert "terraform test" in _run_scripts(job), (
        "terraform-validate must run `terraform test` (the offline, mocked-provider "
        "security/lifecycle contract suite)"
    )


@pytest.mark.contract
def test_pytest_gate_runs_these_contracts(workflow: dict[str, Any]) -> None:
    """The quality job runs ``pytest`` — the gate these contract tests ride in."""
    job = _jobs(workflow).get("quality")
    assert job, "no quality job in ci.yml"
    assert "pytest" in _run_scripts(job), (
        "the quality job must run `pytest` so the contract suite (these tests) gates CI"
    )


# ---------------------------------------------------------------------------
# CI keeps least privilege — a supply-chain regression in its own right.
# ---------------------------------------------------------------------------


@pytest.mark.contract
def test_ci_token_is_read_only(workflow: dict[str, Any]) -> None:
    """The workflow grants only ``contents: read`` — no publish/write scope.

    A CI token that gained ``packages: write`` or ``contents: write`` could push
    images or mutate the repo, the exact posture Sprint 8 keeps out (the build job
    never publishes). Enforced at the workflow level so it cannot drift.
    """
    perms = workflow.get("permissions")
    assert perms == {"contents": "read"}, (
        f"workflow-level permissions must be exactly {{contents: read}}; got {perms!r} "
        "— CI must not carry write/publish scope"
    )
    # No job may re-grant a write scope either.
    for name, job in _jobs(workflow).items():
        jp = job.get("permissions")
        if jp is None:
            continue
        writeable = {k: v for k, v in jp.items() if v == "write"}
        assert not writeable, (
            f"job {name!r} grants write scope {writeable} — CI stays read-only"
        )
