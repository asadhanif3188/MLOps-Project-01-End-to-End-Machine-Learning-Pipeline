"""Contract tests: the Sprint 7/8 security posture cannot silently regress.

The security model is enforced deeply elsewhere — ``k8s/validate.py`` proves the
rendered manifests are hardened (PSA restricted, non-root, dropped capabilities),
and ``terraform/tests/*.tftest.hcl`` prove the IaC posture (KMS secrets
encryption, EKS Pod-Identity/CNI, the EKS access model, no static keys) against
the real provider schema via ``terraform test``. Those are the right tools and
this suite does not re-implement them.

What this suite adds is the two regressions those gates cannot see themselves:

* **A static AWS credential committed anywhere we own.** The manifest secret-scan
  only sees the *rendered* ``k8s`` output; nothing scans the raw Terraform,
  workflow, or source tree for a hard-coded key. This does, over the whole owned
  source, semantically (a key *fingerprint* and the Terraform *resource* that
  mints one — not a blanket text grep).
* **A security gate being deleted.** ``terraform test`` still passes with fewer
  tests, so removing an ``*.tftest.hcl`` file would drop a security contract with
  CI staying green. This asserts the Sprint 7/8 security suites are still present.

Pure filesystem/text parsing — no AWS, no cluster, no network. Runs in the
ordinary ``pytest`` gate with the other ``contract`` tests.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TF_DIR = _REPO_ROOT / "terraform"

# An AWS access-key-id fingerprint. High-signal, effectively zero false positives
# (the same shape k8s/validate.py's secret-scan uses). Split so this literal is not
# itself a scannable key shape.
_AWS_KEY_ID = re.compile(r"AKIA" + r"[0-9A-Z]{16}")

# A Terraform provider/resource assignment of a literal (quoted) access/secret key.
# Matches `access_key = "…"` / `secret_key = "…"` with a non-empty, non-placeholder
# value — the way a static credential would be baked into the IaC.
_TF_INLINE_KEY = re.compile(
    r'\b(access_key|secret_key)\s*=\s*"(?!\s*$)([^"]+)"', re.IGNORECASE
)
_PLACEHOLDER = re.compile(
    r"^(<.*>|replace_with|changeme|example|dummy|xxx+|\$\{)", re.I
)

# The Sprint 7/8 Terraform security-contract suites. Each is a gate whose silent
# deletion would remove a security contract while `terraform test` stayed green.
REQUIRED_TF_TEST_SUITES = {
    "eks_access_control.tftest.hcl": "the EKS access model (access entries/auth)",
    "eks_api_security.tftest.hcl": "the private EKS API-server posture",
    "eks_cni_identity.tftest.hcl": "EKS Pod Identity + the VPC CNI identity",
    "eks_secrets_encryption.tftest.hcl": "KMS envelope encryption of EKS secrets",
}


def _owned_source_files() -> list[Path]:
    """The tracked source we author, restricted to text formats worth scanning.

    Deliberately excludes generated/binary state (``*.tfstate*``, ``tfplan``),
    caches, and the test tree itself (these tests carry key *patterns*).
    """
    roots_globs = (
        (_TF_DIR, ("*.tf", "*.tftest.hcl", "*.tfvars.example")),
        (_REPO_ROOT / "k8s", ("*.yaml", "*.yml")),
        (_REPO_ROOT / ".github", ("*.yml", "*.yaml")),
        (_REPO_ROOT / "src", ("*.py",)),
    )
    files: list[Path] = []
    for root, patterns in roots_globs:
        if not root.is_dir():
            continue
        for pattern in patterns:
            files.extend(root.rglob(pattern))
    # Never scan Terraform state/plan artifacts (not tracked, but be defensive).
    return [
        f
        for f in files
        if not (
            f.name.startswith("terraform.tfstate")
            or f.name == "tfplan"
            or "__pycache__" in f.parts
        )
    ]


# ---------------------------------------------------------------------------
# No static AWS credentials in the owned source tree.
# ---------------------------------------------------------------------------


@pytest.mark.contract
def test_no_aws_access_key_fingerprint_in_source() -> None:
    """No AWS access-key-id fingerprint appears in the Terraform/k8s/CI/src tree.

    The project authenticates with EKS Pod Identity and short-lived roles — never
    a static key (ADR-022 / the CI no-AWS boundary). A committed ``AKIA…`` key is
    the canonical static-credential leak; there must be none.
    """
    hits = [
        f.relative_to(_REPO_ROOT).as_posix()
        for f in _owned_source_files()
        if _AWS_KEY_ID.search(f.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert not hits, f"AWS access-key-id fingerprint found in: {hits}"


@pytest.mark.contract
def test_terraform_declares_no_static_key_resource_or_literal() -> None:
    """Terraform neither mints a static key nor inlines one.

    Two failure modes: an ``aws_iam_access_key`` resource (which *creates* a
    long-lived key), and an inline ``access_key``/``secret_key`` literal in a
    provider or resource block. Both are static credentials the posture forbids;
    the provider is configured by role/instance identity only.
    """
    resource_hits: list[str] = []
    literal_hits: list[str] = []
    for f in _TF_DIR.rglob("*.tf"):
        text = f.read_text(encoding="utf-8", errors="ignore")
        rel = f.relative_to(_REPO_ROOT).as_posix()
        if re.search(r'resource\s+"aws_iam_access_key"', text):
            resource_hits.append(rel)
        for key, value in _TF_INLINE_KEY.findall(text):
            if not _PLACEHOLDER.match(value.strip()):
                literal_hits.append(f"{rel}: {key}")
    assert not resource_hits, (
        f"Terraform mints a static key via aws_iam_access_key in: {resource_hits}"
    )
    assert not literal_hits, (
        f"Terraform inlines a static credential literal in: {literal_hits}"
    )


# ---------------------------------------------------------------------------
# The security-contract gates remain present.
# ---------------------------------------------------------------------------


@pytest.mark.contract
@pytest.mark.parametrize("suite", sorted(REQUIRED_TF_TEST_SUITES))
def test_terraform_security_contract_suite_present(suite: str) -> None:
    """Each Sprint 7/8 Terraform security suite exists and declares run blocks.

    ``terraform test`` (wired in CI) enforces these contracts; this guards against
    the suite file itself being deleted, which would drop a gate silently.
    """
    path = _TF_DIR / "tests" / suite
    assert path.is_file(), (
        f"missing Terraform security suite {suite} — it enforces "
        f"{REQUIRED_TF_TEST_SUITES[suite]}"
    )
    assert 'run "' in path.read_text(encoding="utf-8"), (
        f"{suite} declares no `run` blocks — the security contract is empty"
    )


@pytest.mark.contract
def test_iac_declares_kms_pod_identity_and_eks_access_model() -> None:
    """The IaC still *declares* the KMS/Pod-Identity/access-model building blocks.

    A presence check — the semantic contracts live in the ``*.tftest.hcl`` suites
    (run by ``terraform test``). This catches the coarsest regression: the
    encryption key, the Pod-Identity association, or the EKS access model being
    removed from the configuration wholesale.
    """
    tf_text = "\n".join(
        f.read_text(encoding="utf-8", errors="ignore") for f in _TF_DIR.rglob("*.tf")
    )
    assert re.search(r'resource\s+"aws_kms_key"', tf_text), (
        "no aws_kms_key resource in terraform/ (KMS envelope encryption)"
    )
    assert "encryption_config" in tf_text, (
        "EKS cluster declares no encryption_config (KMS-encrypted secrets)"
    )
    assert "aws_eks_pod_identity_association" in tf_text, (
        "no aws_eks_pod_identity_association in terraform/ (Pod Identity, not keys)"
    )
    assert "authentication_mode" in tf_text or "aws_eks_access_entry" in tf_text, (
        "EKS declares no access model (authentication_mode / access entries)"
    )
