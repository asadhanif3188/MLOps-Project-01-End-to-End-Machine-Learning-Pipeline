#!/usr/bin/env python3
"""Static validation for the Kubernetes manifests (Sprint 5, PR 6).

This is the project-owned half of the k8s validation gate. It does the checks
that are specific to *this* workload's contract and that a generic schema
validator cannot express — the security posture agreed in PRs 1-5 and the
required workload fields — and it emits an understandable pass/fail line for
every check so a CI failure names exactly what is wrong.

Division of labour in the gate (see docs/ci-cd.md and ADR-012):
  * kustomize        renders base + overlay            -> proves the Kustomize
                     build is valid (task: Kustomize validation).
  * kubeconform      validates the rendered objects against the upstream
                     Kubernetes OpenAPI schema          -> proves every field
                     name/type is a real API field (task: schema validation).
  * THIS SCRIPT      parses every manifest (YAML syntax), then asserts the
                     security controls, the required workload fields, and the
                     secret-hygiene rules on the *rendered* output.

It is deliberately dependency-light: the standard library plus PyYAML (already
a project dependency). No cluster is contacted; nothing is applied. This is
STATIC validation — it does not prove the workload deploys or runs, only that
the manifests are well-formed, schema-valid, hardened, and complete. Cluster
admission is a separate, opt-in check (docs/ci-cd.md § "Cluster integration").

Usage:
    python k8s/validate.py                     # validate k8s/overlays/local
    python k8s/validate.py k8s/overlays/local  # explicit overlay
Exit code 0 = all checks passed; 1 = at least one failure (or a render error).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

# Repository root is the parent of this file's directory (k8s/).
K8S_DIR = Path(__file__).resolve().parent
REPO_ROOT = K8S_DIR.parent
DEFAULT_OVERLAY = "k8s/overlays/local"
BASE_DIR = "k8s/base"

# The namespace this workload is designed around (PRs 1-5).
EXPECTED_NAMESPACE = "mlops"

# Kinds that must carry an explicit namespace (cluster-scoped kinds are exempt).
CLUSTER_SCOPED_KINDS = {"Namespace"}

# Env-var / data keys whose *values* must never be a real, inline credential.
CREDENTIAL_KEY = re.compile(
    r"(password|passwd|secret|token|api[_-]?key|credential)", re.I
)
# A value that is clearly a placeholder, not a real credential.
PLACEHOLDER_VALUE = re.compile(
    r"^(replace_with|changeme|<.*>|placeholder|example|dummy|xxx+)", re.I
)
# High-signal, low-false-positive credential fingerprints in raw manifest text.
SECRET_FINGERPRINTS = [
    ("AWS access key id", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("GitHub token", re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}")),
    ("Slack token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
    (
        "private key block",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ),
]


class Report:
    """Collects check results and prints an understandable grouped summary."""

    def __init__(self) -> None:
        self.results: list[tuple[str, str, bool, str]] = []

    def check(self, section: str, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append((section, name, bool(ok), detail))
        return bool(ok)

    def render(self) -> int:
        section = None
        for sec, name, ok, detail in self.results:
            if sec != section:
                print(f"\n{sec}")
                section = sec
            tag = "PASS" if ok else "FAIL"
            line = f"  [{tag}] {name}"
            # The `detail` strings explain *why a failure occurred*, so they are
            # only meaningful (and only shown) on a FAIL line.
            if not ok and detail:
                line += f" -> {detail}"
            print(line)
        passed = sum(1 for _, _, ok, _ in self.results if ok)
        total = len(self.results)
        failed = total - passed
        print(f"\n{passed}/{total} checks passed", end="")
        print(f", {failed} FAILED" if failed else "")
        return 1 if failed else 0


def kustomize_cmd() -> list[str]:
    """Prefer standalone `kustomize`; fall back to `kubectl kustomize`."""
    if shutil.which("kustomize"):
        return ["kustomize", "build"]
    if shutil.which("kubectl"):
        return ["kubectl", "kustomize"]
    print("ERROR: neither `kustomize` nor `kubectl` found on PATH.", file=sys.stderr)
    sys.exit(2)


def render(path: str) -> str:
    """Render a kustomize directory to a multi-doc YAML string, or exit non-zero."""
    cmd = [*kustomize_cmd(), path]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print(
            f"ERROR: `{' '.join(cmd)}` failed to render {path}:\n{proc.stderr}",
            file=sys.stderr,
        )
        sys.exit(1)
    return proc.stdout


def load_docs(rendered: str) -> list[dict]:
    return [d for d in yaml.safe_load_all(rendered) if isinstance(d, dict)]


def containers_of(job: dict) -> list[dict]:
    return (
        job.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
        or []
    )


def pod_spec_of(job: dict) -> dict:
    return job.get("spec", {}).get("template", {}).get("spec", {}) or {}


def walk_strings(node: object):
    """Yield (key, value) for every scalar string value under a nested structure."""
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, str):
                yield str(k), v
            else:
                yield from walk_strings(v)
    elif isinstance(node, list):
        for item in node:
            yield from walk_strings(item)


def validate() -> int:
    overlay = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OVERLAY
    r = Report()

    # ------------------------------------------------------------------ #
    # Section 1 — YAML syntax & Kustomize rendering.
    # ------------------------------------------------------------------ #
    sec = "1. YAML syntax & Kustomize rendering"
    # Every raw manifest under k8s/ must parse as YAML (the template Secret too,
    # even though it is never rendered).
    for f in sorted(K8S_DIR.rglob("*.yaml")):
        rel = f.relative_to(REPO_ROOT).as_posix()
        try:
            list(yaml.safe_load_all(f.read_text(encoding="utf-8")))
            r.check(sec, f"YAML parses: {rel}", True)
        except yaml.YAMLError as exc:
            r.check(sec, f"YAML parses: {rel}", False, f"invalid YAML: {exc}")

    base_rendered = render(BASE_DIR)
    r.check(sec, f"kustomize build {BASE_DIR}", bool(base_rendered.strip()))
    rendered = render(overlay)
    r.check(sec, f"kustomize build {overlay}", bool(rendered.strip()))

    docs = load_docs(rendered)
    by_kind: dict[str, list[dict]] = {}
    for d in docs:
        by_kind.setdefault(d.get("kind", "?"), []).append(d)

    jobs = by_kind.get("Job", [])
    if not jobs:
        r.check(sec, "rendered output contains a Job", False, "no Job found in render")
        return r.render()
    r.check(sec, "rendered output contains a Job", True)
    job = jobs[0]
    pod = pod_spec_of(job)
    containers = containers_of(job)

    # ------------------------------------------------------------------ #
    # Section 2 — Required workload fields.
    # ------------------------------------------------------------------ #
    sec = "2. Required workload fields"
    r.check(
        sec,
        "Namespace object present",
        any(
            n.get("metadata", {}).get("name") == EXPECTED_NAMESPACE
            for n in by_kind.get("Namespace", [])
        ),
        f"expected a Namespace/{EXPECTED_NAMESPACE}",
    )
    for d in docs:
        kind = d.get("kind", "?")
        name = d.get("metadata", {}).get("name", "?")
        if kind in CLUSTER_SCOPED_KINDS:
            continue
        ns = d.get("metadata", {}).get("namespace")
        r.check(
            sec,
            f"{kind}/{name} pinned to namespace",
            ns == EXPECTED_NAMESPACE,
            f"namespace={ns!r}, expected {EXPECTED_NAMESPACE!r}",
        )

    r.check(sec, "Job has at least one container", bool(containers))
    for c in containers:
        cname = c.get("name", "?")
        image = c.get("image", "")
        # Explicit, pinned image: has a tag (or digest) and it is not floating :latest.
        has_ref = bool(image) and (":" in image or "@" in image)
        tag = image.rsplit(":", 1)[-1] if ":" in image and "@" not in image else image
        pinned = has_ref and tag != "latest"
        r.check(
            sec,
            f"container {cname}: explicit pinned image",
            pinned,
            f"image={image!r} (must have an explicit non-latest tag/digest)",
        )
        reqs = c.get("resources", {}).get("requests", {})
        limits = c.get("resources", {}).get("limits", {})
        r.check(
            sec,
            f"container {cname}: CPU request set",
            bool(reqs.get("cpu")),
            f"requests.cpu={reqs.get('cpu')!r}",
        )
        r.check(
            sec,
            f"container {cname}: memory request set",
            bool(reqs.get("memory")),
            f"requests.memory={reqs.get('memory')!r}",
        )
        r.check(
            sec,
            f"container {cname}: CPU limit set",
            bool(limits.get("cpu")),
            f"limits.cpu={limits.get('cpu')!r}",
        )
        r.check(
            sec,
            f"container {cname}: memory limit set",
            bool(limits.get("memory")),
            f"limits.memory={limits.get('memory')!r}",
        )

    restart = pod.get("restartPolicy")
    r.check(
        sec,
        "Job restartPolicy is Never/OnFailure",
        restart in {"Never", "OnFailure"},
        f"restartPolicy={restart!r} (a Job must not use Always)",
    )

    # ------------------------------------------------------------------ #
    # Section 3 — Workload security controls (PRs 1-5 contract).
    # ------------------------------------------------------------------ #
    sec = "3. Security controls"
    pod_sc = pod.get("securityContext", {}) or {}
    r.check(
        sec,
        "pod runAsNonRoot: true",
        pod_sc.get("runAsNonRoot") is True,
        f"runAsNonRoot={pod_sc.get('runAsNonRoot')!r}",
    )
    run_as = pod_sc.get("runAsUser")
    r.check(
        sec,
        "pod runAsUser is non-root (uid != 0)",
        isinstance(run_as, int) and run_as != 0,
        f"runAsUser={run_as!r}",
    )
    r.check(
        sec,
        "pod seccompProfile RuntimeDefault",
        (pod_sc.get("seccompProfile", {}) or {}).get("type") == "RuntimeDefault",
        f"seccompProfile.type={(pod_sc.get('seccompProfile', {}) or {}).get('type')!r}",
    )

    sa_name = pod.get("serviceAccountName")
    r.check(
        sec,
        "explicit non-default ServiceAccount",
        bool(sa_name) and sa_name != "default",
        f"serviceAccountName={sa_name!r}",
    )
    r.check(
        sec,
        "referenced ServiceAccount object exists",
        any(
            sa.get("metadata", {}).get("name") == sa_name
            for sa in by_kind.get("ServiceAccount", [])
        ),
        f"no ServiceAccount/{sa_name} in render",
    )
    r.check(
        sec,
        "pod automountServiceAccountToken: false",
        pod.get("automountServiceAccountToken") is False,
        f"automountServiceAccountToken={pod.get('automountServiceAccountToken')!r}",
    )
    for sa in by_kind.get("ServiceAccount", []):
        n = sa.get("metadata", {}).get("name", "?")
        r.check(
            sec,
            f"ServiceAccount/{n} automount disabled",
            sa.get("automountServiceAccountToken") is False,
            f"automountServiceAccountToken={sa.get('automountServiceAccountToken')!r}",
        )

    for c in containers:
        cname = c.get("name", "?")
        csc = c.get("securityContext", {}) or {}
        r.check(
            sec,
            f"container {cname}: allowPrivilegeEscalation false",
            csc.get("allowPrivilegeEscalation") is False,
            f"allowPrivilegeEscalation={csc.get('allowPrivilegeEscalation')!r}",
        )
        drops = {
            str(x).upper() for x in (csc.get("capabilities", {}) or {}).get("drop", [])
        }
        r.check(
            sec,
            f"container {cname}: capabilities drop ALL",
            "ALL" in drops,
            f"capabilities.drop={sorted(drops)}",
        )

    # ------------------------------------------------------------------ #
    # Section 4 — Secret hygiene (no hardcoded credentials).
    # ------------------------------------------------------------------ #
    sec = "4. Secret hygiene"
    # The rendered workload must not carry a Secret: credentials are created
    # out-of-band (secret.example.yaml is a template, excluded from kustomize).
    secret_names = [
        m.get("metadata", {}).get("name") for m in by_kind.get("Secret", [])
    ]
    r.check(
        sec,
        "no Secret object in rendered workload",
        "Secret" not in by_kind,
        f"rendered kinds include Secret: {secret_names}",
    )

    # No credential-keyed value in the rendered output holds a real inline value.
    inline_creds: list[str] = []
    for d in docs:
        for key, value in walk_strings(d):
            if (
                CREDENTIAL_KEY.search(key)
                and value
                and not PLACEHOLDER_VALUE.match(value.strip())
            ):
                inline_creds.append(
                    f"{d.get('kind')}/{d.get('metadata', {}).get('name')}:{key}"
                )
    r.check(
        sec,
        "no inline credential values in rendered output",
        not inline_creds,
        f"credential-keyed values found: {inline_creds}",
    )

    # High-signal secret fingerprints anywhere in the raw k8s/ tree (one
    # aggregate check — names every offender so a failure is actionable).
    fingerprint_hits: list[str] = []
    for f in sorted(K8S_DIR.rglob("*.yaml")):
        text = f.read_text(encoding="utf-8")
        rel = f.relative_to(REPO_ROOT).as_posix()
        for label, pat in SECRET_FINGERPRINTS:
            if pat.search(text):
                fingerprint_hits.append(f"{rel}: {label}")
    r.check(
        sec,
        "no hardcoded-secret fingerprints in k8s/ tree",
        not fingerprint_hits,
        f"matches: {fingerprint_hits}",
    )

    # The Secret TEMPLATE must contain only placeholders (defence: a real token
    # must never be committed into the example).
    for f in sorted(K8S_DIR.rglob("*.yaml")):
        text = f.read_text(encoding="utf-8")
        for doc in yaml.safe_load_all(text):
            if isinstance(doc, dict) and doc.get("kind") == "Secret":
                rel = f.relative_to(REPO_ROOT).as_posix()
                bad = [
                    k
                    for k, v in (doc.get("stringData", {}) or {}).items()
                    if isinstance(v, str) and not PLACEHOLDER_VALUE.match(v.strip())
                ]
                r.check(
                    sec,
                    f"Secret template holds only placeholders ({rel})",
                    not bad,
                    f"non-placeholder keys: {bad}",
                )

    return r.render()


if __name__ == "__main__":
    print("Kubernetes static validation (k8s/validate.py)")
    code = validate()
    print(
        "\nRESULT:",
        "PASS - manifests are well-formed, hardened, and complete (STATIC checks only)."
        if code == 0
        else "FAIL - see the [FAIL] lines above.",
    )
    sys.exit(code)
