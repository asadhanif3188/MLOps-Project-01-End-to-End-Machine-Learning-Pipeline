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

import json
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


def init_containers_of(job: dict) -> list[dict]:
    return pod_spec_of(job).get("initContainers", []) or []


def volumes_of(job: dict) -> list[dict]:
    return pod_spec_of(job).get("volumes", []) or []


def volume_mounts_of(container: dict) -> list[dict]:
    return container.get("volumeMounts", []) or []


# Every pod-bearing (workload) kind whose template embeds a pod spec. These are
# the objects the uniform security-context contract (Section 7) applies to —
# their `spec.template.spec` is a PodSpec, so pod_spec_of/containers_of/
# init_containers_of all work on them unchanged.
WORKLOAD_KINDS = ("Job", "Deployment", "StatefulSet", "DaemonSet", "ReplicaSet")


def workloads_in(by_kind: dict[str, list[dict]]) -> list[tuple[str, str, dict]]:
    """Every pod-bearing workload in the render, as (kind, name, manifest)."""
    out: list[tuple[str, str, dict]] = []
    for kind in WORKLOAD_KINDS:
        for d in by_kind.get(kind, []):
            out.append((kind, d.get("metadata", {}).get("name", "?"), d))
    return out


def all_containers_of(workload: dict) -> list[tuple[str, dict]]:
    """Every container in a workload's pod, init and main alike, as (role, container).

    Both are ordinary containers as far as the security-context contract is
    concerned: an init container that could escalate privileges or keep the full
    capability set is exactly as much of a hole as a main one, so the contract
    must reach both (the pre-Sprint-7 checks only covered the main containers).
    """
    return [("init", c) for c in init_containers_of(workload)] + [
        ("main", c) for c in containers_of(workload)
    ]


def container_env(container: dict) -> dict[str, object]:
    """Map env var name -> its `value` (or the whole entry when valueFrom-based)."""
    out: dict[str, object] = {}
    for e in container.get("env", []) or []:
        name = e.get("name")
        if name:
            out[name] = e.get("value", e)
    return out


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
    # Select the PIPELINE Job by name. The local overlay now renders a second Job
    # (`minio-setup`, the artifact-bucket bootstrap), so `jobs[0]` is no longer
    # unambiguously the pipeline — the workload-contract checks below must target
    # `mlops-pipeline` explicitly (they assert its command, dataset mount, etc.).
    job = next(
        (j for j in jobs if j.get("metadata", {}).get("name") == "mlops-pipeline"),
        jobs[0],
    )
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

    # The observability queryability contract (Sprint 8, PR 2 — ADR-028 § 3 /
    # ADR-029). The pipeline pod exits in under a minute, but kube-state-metrics
    # reflects the persistent Job/Pod OBJECT — so the last-run success/duration/OOM
    # series stay scrapable ONLY while the finished Job lingers. A positive
    # ttlSecondsAfterFinished guarantees the finished Job outlives many scrape
    # intervals before it is auto-reaped. This is a static assertion that the
    # manifest declares the contract, not that a scrape happened.
    ttl = job.get("spec", {}).get("ttlSecondsAfterFinished")
    r.check(
        sec,
        "pipeline Job sets a positive ttlSecondsAfterFinished (queryability contract)",
        isinstance(ttl, int) and ttl > 0,
        f"ttlSecondsAfterFinished={ttl!r} (must be a positive integer so the "
        f"finished Job outlives a Prometheus scrape — ADR-028 § 3)",
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

    # ------------------------------------------------------------------ #
    # Section 5 — Runtime execution contract (PR 8).
    # ------------------------------------------------------------------ #
    # These assert the wiring that makes `dvc repro` actually RUN to completion
    # in-cluster (proven end-to-end on a live cluster; see ADR-013). They are
    # STATIC: they prove the manifest declares the contract, not that a run
    # succeeds. Checked on the rendered overlay (which includes the base), so both
    # the environment-independent DVC no-SCM config and the local dataset/MLflow
    # wiring are present.
    sec = "5. Runtime execution contract"
    c0 = containers[0] if containers else {}

    # (a) The workload invokes the DVC pipeline runner.
    command = c0.get("command", []) or []
    r.check(
        sec,
        "container command runs `dvc`",
        bool(command) and command[0] == "dvc",
        f"command={command!r} (expected it to start with 'dvc')",
    )

    # (b) DVC no-SCM contract: a ConfigMap carries `core.no_scm = true` in a
    # `config.local`, and the Job mounts it read-only at /app/.dvc/config.local
    # (subPath, so it does not shadow the image's /app/.dvc/config). This is what
    # lets `dvc repro` run without a Git repo in the image.
    no_scm_cms = [
        cm
        for cm in by_kind.get("ConfigMap", [])
        if "no_scm" in (cm.get("data", {}) or {}).get("config.local", "")
    ]
    scm_cm_ok = any(
        re.search(r"no_scm\s*=\s*true", cm["data"]["config.local"], re.I)
        for cm in no_scm_cms
    )
    r.check(
        sec,
        "DVC no-SCM ConfigMap present (config.local: core.no_scm = true)",
        scm_cm_ok,
        "no ConfigMap has data.'config.local' containing 'no_scm = true'",
    )
    no_scm_cm_names = {cm.get("metadata", {}).get("name") for cm in no_scm_cms}
    scm_vol_names = {
        v.get("name")
        for v in volumes_of(job)
        if (v.get("configMap", {}) or {}).get("name") in no_scm_cm_names
    }
    scm_mount_ok = any(
        m.get("name") in scm_vol_names
        and m.get("mountPath") == "/app/.dvc/config.local"
        and m.get("subPath") == "config.local"
        and m.get("readOnly") is True
        for m in volume_mounts_of(c0)
    )
    r.check(
        sec,
        "DVC config.local mounted read-only at /app/.dvc/config.local (subPath)",
        scm_mount_ok,
        "no read-only subPath mount of the no-SCM ConfigMap at /app/.dvc/config.local",
    )

    # (c) Runtime dataset via S3 init-container retrieval (Sprint 7, PR 8 — closes
    # M-04). The dataset is no longer delivered by a ConfigMap: a `fetch-dataset`
    # init container downloads it from S3 (real S3 on EKS via Pod Identity;
    # S3-compatible MinIO locally) into a shared emptyDir that the pipeline
    # container reads at /app/data/raw. These checks assert that wiring on the
    # rendered overlay. The dataset OBJECT lives in S3, created out-of-band, so it
    # is intentionally NOT in the render — only the retrieval contract is.
    init_containers = init_containers_of(job)
    fetch = next((c for c in init_containers if c.get("name") == "fetch-dataset"), None)
    r.check(
        sec,
        "fetch-dataset init container present",
        fetch is not None,
        "no initContainer named 'fetch-dataset' (the S3 dataset retrieval step)",
    )

    # The pipeline container consumes the dataset read-only at /app/data/raw.
    data_mounts = [
        m for m in volume_mounts_of(c0) if m.get("mountPath") == "/app/data/raw"
    ]
    r.check(
        sec,
        "runtime dataset mounted read-only at /app/data/raw (pipeline)",
        bool(data_mounts) and all(m.get("readOnly") is True for m in data_mounts),
        "the pipeline container must mount /app/data/raw read-only "
        "(the init container is the sole writer)",
    )

    # That mount must be an emptyDir (the init->main hand-off buffer), NOT a
    # ConfigMap (M-04) and NOT a hostPath (explicit sprint constraint).
    volumes_by_name = {v.get("name"): v for v in volumes_of(job)}
    data_vol_names = {m.get("name") for m in data_mounts}
    dataset_vols = [volumes_by_name.get(n, {}) for n in data_vol_names]
    r.check(
        sec,
        "dataset volume is an emptyDir (not ConfigMap, not hostPath)",
        bool(dataset_vols)
        and all(
            "emptyDir" in v and "configMap" not in v and "hostPath" not in v
            for v in dataset_vols
        ),
        f"dataset volume(s) must be emptyDir; got {dataset_vols}",
    )

    # The init container writes the SAME volume at /app/data/raw (writable — no
    # readOnly), so the download lands where the pipeline reads it.
    if fetch is not None:
        fetch_data_mounts = [
            m
            for m in volume_mounts_of(fetch)
            if m.get("mountPath") == "/app/data/raw" and m.get("name") in data_vol_names
        ]
        r.check(
            sec,
            "fetch-dataset writes the shared dataset volume at /app/data/raw",
            bool(fetch_data_mounts)
            and all(m.get("readOnly") is not True for m in fetch_data_mounts),
            "the fetch-dataset init container must mount the dataset volume "
            "writable at /app/data/raw",
        )
        # It must run the first-party retrieval script and know its S3 source.
        fcmd = fetch.get("command", []) or []
        fargs = fetch.get("args", []) or []
        runs_fetch = any("fetch_dataset" in str(t) for t in [*fcmd, *fargs])
        r.check(
            sec,
            "fetch-dataset runs the retrieval script (src/fetch_dataset.py)",
            runs_fetch,
            f"command/args do not reference fetch_dataset: {fcmd} {fargs}",
        )
        fetch_env = container_env(fetch)
        r.check(
            sec,
            "fetch-dataset has DATASET_S3_URI configured",
            "DATASET_S3_URI" in fetch_env,
            "the fetch-dataset init container must set DATASET_S3_URI "
            "(the S3 object to retrieve)",
        )

    # The old ConfigMap dataset mechanism must be fully gone: no volume may be
    # backed by a ConfigMap named 'mlops-pipeline-dataset', anywhere in the render.
    dataset_cm_refs = [
        v.get("name")
        for v in volumes_of(job)
        if (v.get("configMap", {}) or {}).get("name") == "mlops-pipeline-dataset"
    ]
    r.check(
        sec,
        "no ConfigMap-backed dataset volume (M-04 closed)",
        not dataset_cm_refs,
        f"dataset must not be delivered via ConfigMap; found volumes {dataset_cm_refs}",
    )

    # Defence-in-depth against the sprint's explicit "no hostPath" constraint: no
    # volume anywhere in the pipeline pod may be a hostPath.
    hostpath_vols = [v.get("name") for v in volumes_of(job) if "hostPath" in v]
    r.check(
        sec,
        "no hostPath volumes in the pipeline pod",
        not hostpath_vols,
        f"hostPath volumes are forbidden; found {hostpath_vols}",
    )

    # The pinned dataset identity is present so the init container can verify
    # integrity (documents dataset version/checksum in-manifest — requirement 11).
    dataset_sha_present = any(
        "DATASET_SHA256" in (cm.get("data", {}) or {})
        for cm in by_kind.get("ConfigMap", [])
    )
    r.check(
        sec,
        "DATASET_SHA256 pinned in a ConfigMap (dataset integrity/identity)",
        dataset_sha_present,
        "no ConfigMap carries DATASET_SHA256 (the pinned dataset checksum)",
    )

    # (d) MLflow tracking endpoint is configured for the container — either an
    # explicit env override (the local file store) or the base ConfigMap's
    # MLFLOW_TRACKING_URI injected via envFrom. Credentials stay in the Secret.
    env = container_env(c0)
    envfrom_cm_names = {
        (ef.get("configMapRef", {}) or {}).get("name")
        for ef in (c0.get("envFrom", []) or [])
    }
    envfrom_has_uri = any(
        "MLFLOW_TRACKING_URI" in (cm.get("data", {}) or {})
        for cm in by_kind.get("ConfigMap", [])
        if cm.get("metadata", {}).get("name") in envfrom_cm_names
    )
    r.check(
        sec,
        "MLFLOW_TRACKING_URI configured (env override or envFrom ConfigMap)",
        "MLFLOW_TRACKING_URI" in env or envfrom_has_uri,
        "no MLFLOW_TRACKING_URI in container env or referenced ConfigMap",
    )

    # ------------------------------------------------------------------ #
    # Section 6 — MLflow tracking platform (Sprint 7, PR 6; ADR-026).
    # ------------------------------------------------------------------ #
    # The persistent in-cluster MLflow platform: a stateless tracking-server
    # Deployment fronted by an INTERNAL Service, backed by a PostgreSQL StatefulSet
    # whose PVC makes metadata survive pod recreation. These checks assert the
    # platform's declared contract (probes, resources, hardening, internal-only
    # exposure, explicit persistence) on the rendered overlay. STILL static: they
    # prove the manifest declares the platform, not that it runs (that is the
    # persistence test in ADR-026).
    sec = "6. MLflow tracking platform"

    # (a) The tracking server Deployment exists and is hardened + probed + bounded.
    deployments = by_kind.get("Deployment", [])
    mlflow_dep = next(
        (d for d in deployments if d.get("metadata", {}).get("name") == "mlflow"),
        None,
    )
    r.check(
        sec,
        "MLflow tracking server Deployment present",
        mlflow_dep is not None,
        "no Deployment/mlflow in render",
    )
    if mlflow_dep is not None:
        mpod = pod_spec_of(mlflow_dep)
        mcs = containers_of(mlflow_dep)
        mc = mcs[0] if mcs else {}
        r.check(
            sec,
            "MLflow server: readiness + liveness probes",
            bool(mc.get("readinessProbe")) and bool(mc.get("livenessProbe")),
            "the tracking server must define both readiness and liveness probes",
        )
        mreqs = mc.get("resources", {}).get("requests", {})
        mlimits = mc.get("resources", {}).get("limits", {})
        r.check(
            sec,
            "MLflow server: CPU/memory requests + limits",
            bool(mreqs.get("cpu"))
            and bool(mreqs.get("memory"))
            and bool(mlimits.get("cpu"))
            and bool(mlimits.get("memory")),
            f"requests={mreqs}, limits={mlimits}",
        )
        mpod_sc = mpod.get("securityContext", {}) or {}
        r.check(
            sec,
            "MLflow server: pod runAsNonRoot true",
            mpod_sc.get("runAsNonRoot") is True,
            f"runAsNonRoot={mpod_sc.get('runAsNonRoot')!r}",
        )
        mc_sc = mc.get("securityContext", {}) or {}
        mcaps = (mc_sc.get("capabilities", {}) or {}).get("drop", [])
        mdrops = {str(x).upper() for x in mcaps}
        r.check(
            sec,
            "MLflow server: allowPrivilegeEscalation false + drop ALL",
            mc_sc.get("allowPrivilegeEscalation") is False and "ALL" in mdrops,
            f"allowPrivilegeEscalation={mc_sc.get('allowPrivilegeEscalation')!r}, "
            f"drop={sorted(mdrops)}",
        )
        mimage = mc.get("image", "")
        mtag = (
            mimage.rsplit(":", 1)[-1] if ":" in mimage and "@" not in mimage else mimage
        )
        r.check(
            sec,
            "MLflow server: explicit pinned image",
            bool(mimage) and (":" in mimage or "@" in mimage) and mtag != "latest",
            f"image={mimage!r} (must have an explicit non-latest tag/digest)",
        )

    # (b) The tracking Service exists and is INTERNAL ONLY (requirement 13): a
    # ClusterIP (or unset, which defaults to ClusterIP), never a NodePort or
    # LoadBalancer that would expose the no-auth server outside the cluster.
    mlflow_svc = next(
        (
            s
            for s in by_kind.get("Service", [])
            if s.get("metadata", {}).get("name") == "mlflow"
        ),
        None,
    )
    r.check(
        sec,
        "MLflow Service present",
        mlflow_svc is not None,
        "no Service/mlflow in render",
    )
    if mlflow_svc is not None:
        svc_type = mlflow_svc.get("spec", {}).get("type", "ClusterIP")
        r.check(
            sec,
            "MLflow Service is internal-only (ClusterIP)",
            svc_type == "ClusterIP",
            f"service type={svc_type!r} (must be ClusterIP — internal, requirement 13)",
        )

    # (c) The metadata backend is a StatefulSet with an EXPLICIT PVC template — the
    # single thing that makes MLflow metadata survive pod recreation.
    statefulsets = by_kind.get("StatefulSet", [])
    pg = next(
        (
            s
            for s in statefulsets
            if s.get("metadata", {}).get("name") == "mlflow-postgres"
        ),
        None,
    )
    r.check(
        sec,
        "PostgreSQL metadata StatefulSet present",
        pg is not None,
        "no StatefulSet/mlflow-postgres in render",
    )
    if pg is not None:
        vcts = pg.get("spec", {}).get("volumeClaimTemplates", []) or []
        r.check(
            sec,
            "PostgreSQL persistence is explicit (volumeClaimTemplate)",
            bool(vcts)
            and any(
                (v.get("spec", {}).get("resources", {}).get("requests", {}) or {}).get(
                    "storage"
                )
                for v in vcts
            ),
            "the Postgres StatefulSet must declare a volumeClaimTemplate "
            "with a storage request",
        )
        pg_c = containers_of(pg)[0] if containers_of(pg) else {}
        pg_reqs = pg_c.get("resources", {}).get("requests", {})
        pg_limits = pg_c.get("resources", {}).get("limits", {})
        r.check(
            sec,
            "PostgreSQL: CPU/memory requests + limits",
            bool(pg_reqs.get("cpu"))
            and bool(pg_reqs.get("memory"))
            and bool(pg_limits.get("cpu"))
            and bool(pg_limits.get("memory")),
            f"requests={pg_reqs}, limits={pg_limits}",
        )

    # (d) DagsHub is fully removed (ADR-026): no rendered manifest may reference it
    # (endpoint, credentials, or otherwise). Guards against a stray leftover.
    dagshub_hits = [
        f"{d.get('kind')}/{d.get('metadata', {}).get('name')}:{k}"
        for d in docs
        for k, v in walk_strings(d)
        if "dagshub" in v.lower()
    ]
    r.check(
        sec,
        "no DagsHub reference in rendered output",
        not dagshub_hits,
        f"DagsHub still referenced: {dagshub_hits}",
    )

    # (e) The pipeline's tracking URI now targets the in-cluster server over HTTP
    # (not a file store, not an external SaaS) — the platform is actually wired in.
    tracking_uri_vals = [
        cm.get("data", {}).get("MLFLOW_TRACKING_URI", "")
        for cm in by_kind.get("ConfigMap", [])
        if cm.get("metadata", {}).get("name") == "mlops-pipeline-config"
    ]
    r.check(
        sec,
        "pipeline MLFLOW_TRACKING_URI points at the in-cluster server (http)",
        any(uri.startswith("http") and "mlflow" in uri for uri in tracking_uri_vals),
        f"mlops-pipeline-config MLFLOW_TRACKING_URI values: {tracking_uri_vals}",
    )

    # (f) Layer 4 DB-health exporter (Sprint 8, PR 4; ADR-031). postgres-exporter
    # lives in the mlops namespace beside the DB so its credential Secret never
    # enters the monitoring namespace. Its pod hardening is already enforced by the
    # uniform contract (Section 7); the checks here are the PR-4 SECURITY contract
    # the brief calls out: the DB password must come from a Secret (never inline),
    # and the connection target must carry no embedded credential.
    pgx = next(
        (
            d
            for d in deployments
            if d.get("metadata", {}).get("name") == "postgres-exporter"
        ),
        None,
    )
    r.check(
        sec,
        "postgres-exporter Deployment present (Layer 4 DB health)",
        pgx is not None,
        "no Deployment/postgres-exporter in render",
    )
    r.check(
        sec,
        "postgres-exporter Service present",
        any(
            s.get("metadata", {}).get("name") == "postgres-exporter"
            for s in by_kind.get("Service", [])
        ),
        "no Service/postgres-exporter in render",
    )
    if pgx is not None:
        pgx_cs = containers_of(pgx)
        pgx_env = container_env(pgx_cs[0]) if pgx_cs else {}
        # The password must be a secretKeyRef, not an inline value — the whole point
        # of "do not expose database credentials in metrics/config".
        pass_entry = pgx_env.get("DATA_SOURCE_PASS")
        pass_from_secret = isinstance(pass_entry, dict) and "secretKeyRef" in (
            pass_entry.get("valueFrom", {}) or {}
        )
        r.check(
            sec,
            "postgres-exporter password sourced from a Secret (secretKeyRef)",
            pass_from_secret,
            "DATA_SOURCE_PASS must be a secretKeyRef, never an inline value "
            "(no DB credentials in config — ADR-031)",
        )
        # The non-secret connection target must not smuggle a credential in a DSN
        # (`user:pass@host`); the split DATA_SOURCE_URI form carries host/db only.
        uri_val = pgx_env.get("DATA_SOURCE_URI", "")
        uri_str = uri_val if isinstance(uri_val, str) else ""
        r.check(
            sec,
            "postgres-exporter DATA_SOURCE_URI carries no inline credential",
            "@" not in uri_str,
            f"DATA_SOURCE_URI must be host/db only (no 'user:pass@'); got {uri_str!r}",
        )

    # ------------------------------------------------------------------ #
    # Section 7 — Uniform workload security-context contract (Sprint 7 PR 11).
    # ------------------------------------------------------------------ #
    # Requirement 11 ("Kubernetes workload security context") applies to the WHOLE
    # fleet, not just the pipeline Job. Sections 3 and 6 spot-check individual
    # workloads (the pipeline's main containers; part of the MLflow platform), but
    # that left real holes a regression could slip through unnoticed:
    #   * the pipeline's INIT containers (fetch-dataset, wait-for-mlflow) — never
    #     checked, yet an init container that gained privilege escalation or kept
    #     the full capability set is exactly as much of a breakout surface as the
    #     main one (it runs in the same pod, with the same fsGroup-owned volumes);
    #   * the PostgreSQL StatefulSet and the local MinIO StatefulSet + minio-setup
    #     Job — their container hardening (allowPrivilegeEscalation/drop ALL) and
    #     seccomp profile were unverified;
    #   * seccompProfile RuntimeDefault was asserted only for the pipeline pod.
    # This section enforces ONE hardened baseline across EVERY pod-bearing workload
    # the overlay renders — Job, Deployment, StatefulSet (and DaemonSet/ReplicaSet
    # if ever added) — and across EVERY container in each, init and main alike. It
    # is semantic: it reads the rendered securityContext, not a keyword. A new
    # workload (or a new init container) that is not hardened to this baseline now
    # fails CI by construction, instead of silently shipping under-hardened because
    # no bespoke check was written for it. STILL static: it proves the manifest
    # declares the hardening, not that the kernel enforces it at runtime.
    sec = "7. Uniform workload security-context contract (all pods)"

    # (a) Namespace-level backstop: the mlops Namespace must make Pod Security
    # Admission ENFORCE the `restricted` standard, so the cluster itself rejects a
    # violating pod at admission — the standing counterpart to the static per-pod
    # checks below. The policy version must be pinned (not `latest`) so the admitted
    # ruleset is deterministic. `warn` (or `audit`) at `restricted` is also required
    # so a violation is surfaced even if `enforce` is ever dialled back.
    PSA = "pod-security.kubernetes.io"
    mlops_ns = next(
        (
            n
            for n in by_kind.get("Namespace", [])
            if n.get("metadata", {}).get("name") == EXPECTED_NAMESPACE
        ),
        None,
    )
    ns_labels = ((mlops_ns or {}).get("metadata", {}) or {}).get("labels", {}) or {}
    r.check(
        sec,
        f"Namespace/{EXPECTED_NAMESPACE} enforces the restricted Pod Security Standard",
        ns_labels.get(f"{PSA}/enforce") == "restricted",
        f"{PSA}/enforce={ns_labels.get(f'{PSA}/enforce')!r} (must be 'restricted')",
    )
    enforce_ver = ns_labels.get(f"{PSA}/enforce-version")
    r.check(
        sec,
        f"Namespace/{EXPECTED_NAMESPACE} pins the enforce policy version",
        bool(enforce_ver) and enforce_ver != "latest",
        f"{PSA}/enforce-version={enforce_ver!r} "
        f"(must be a pinned version, not 'latest')",
    )
    r.check(
        sec,
        f"Namespace/{EXPECTED_NAMESPACE} warns or audits at restricted",
        ns_labels.get(f"{PSA}/warn") == "restricted"
        or ns_labels.get(f"{PSA}/audit") == "restricted",
        f"neither {PSA}/warn nor {PSA}/audit is 'restricted' "
        f"(warn={ns_labels.get(f'{PSA}/warn')!r}, "
        f"audit={ns_labels.get(f'{PSA}/audit')!r})",
    )

    workloads = workloads_in(by_kind)
    r.check(
        sec,
        "at least one pod-bearing workload rendered",
        bool(workloads),
        "no Job/Deployment/StatefulSet found to apply the security contract to",
    )
    for kind, name, wl in workloads:
        wpod = pod_spec_of(wl)
        wsc = wpod.get("securityContext", {}) or {}

        # Pod-level: must refuse to run as root, pin a non-root numeric uid (a name
        # the kubelet cannot resolve would fail runAsNonRoot admission), apply the
        # runtime's default seccomp profile, and never auto-mount an API token.
        r.check(
            sec,
            f"{kind}/{name}: pod runAsNonRoot true",
            wsc.get("runAsNonRoot") is True,
            f"runAsNonRoot={wsc.get('runAsNonRoot')!r}",
        )
        wuid = wsc.get("runAsUser")
        r.check(
            sec,
            f"{kind}/{name}: pod runAsUser is a non-root uid",
            isinstance(wuid, int) and wuid != 0,
            f"runAsUser={wuid!r} (must be an explicit numeric uid != 0)",
        )
        r.check(
            sec,
            f"{kind}/{name}: pod seccompProfile RuntimeDefault",
            (wsc.get("seccompProfile", {}) or {}).get("type") == "RuntimeDefault",
            f"seccompProfile.type="
            f"{(wsc.get('seccompProfile', {}) or {}).get('type')!r}",
        )
        r.check(
            sec,
            f"{kind}/{name}: automountServiceAccountToken false",
            wpod.get("automountServiceAccountToken") is False,
            f"automountServiceAccountToken={wpod.get('automountServiceAccountToken')!r}",
        )

        # No host-namespace sharing. hostNetwork/hostPID/hostIPC dissolve the pod's
        # isolation from the node (a classic container-breakout surface, forbidden
        # by the restricted Pod Security Standard). Each is a boolean that defaults
        # to false when absent, so "unset" is compliant and only an explicit `true`
        # fails.
        for host_ns in ("hostNetwork", "hostPID", "hostIPC"):
            r.check(
                sec,
                f"{kind}/{name}: {host_ns} not enabled",
                wpod.get(host_ns) in (None, False),
                f"{host_ns}={wpod.get(host_ns)!r} "
                f"(host-namespace sharing is forbidden)",
            )

        # No hostPath volume. A hostPath mounts a node-filesystem path into the pod
        # (host access / breakout), also forbidden by the restricted PSS. Section 5
        # already guards the pipeline pod against this; here it is enforced across
        # every workload the overlay renders.
        hostpath_vols = [v.get("name") for v in volumes_of(wl) if "hostPath" in v]
        r.check(
            sec,
            f"{kind}/{name}: no hostPath volumes",
            not hostpath_vols,
            f"hostPath volumes are forbidden; found {hostpath_vols}",
        )

        # Container-level (every container, init AND main): no privilege escalation,
        # never privileged, and the entire default capability set dropped.
        every = all_containers_of(wl)
        r.check(
            sec,
            f"{kind}/{name}: has at least one container",
            bool(every),
            "the workload declares no containers",
        )
        for role, c in every:
            cname = c.get("name", "?")
            csc = c.get("securityContext", {}) or {}
            r.check(
                sec,
                f"{kind}/{name}: {role} container {cname} "
                f"allowPrivilegeEscalation false",
                csc.get("allowPrivilegeEscalation") is False,
                f"allowPrivilegeEscalation={csc.get('allowPrivilegeEscalation')!r}",
            )
            r.check(
                sec,
                f"{kind}/{name}: {role} container {cname} not privileged",
                csc.get("privileged") is not True,
                f"privileged={csc.get('privileged')!r}",
            )
            drops = {
                str(x).upper()
                for x in (csc.get("capabilities", {}) or {}).get("drop", [])
            }
            r.check(
                sec,
                f"{kind}/{name}: {role} container {cname} capabilities drop ALL",
                "ALL" in drops,
                f"capabilities.drop={sorted(drops)}",
            )

    return r.render()


# ------------------------------------------------------------------------- #
# Monitoring stack validation (Sprint 8, PR 2 — ADR-028 / ADR-029).
# ------------------------------------------------------------------------- #
# The observability foundation (k8s/monitoring) is a SEPARATE kustomize root with
# its own namespace and, deliberately, a DIFFERENT security contract from the
# mlops workload: Prometheus and kube-state-metrics genuinely need the Kubernetes
# API (so their tokens ARE mounted), and node-exporter needs read-only hostPath
# access to the node's /proc,/sys,/ (which forces the monitoring namespace to
# `privileged` Pod Security — the single documented exception, ADR-029). The
# fleet-wide contract in validate() (automount off everywhere, no hostPath,
# restricted PSA) would wrongly fail this stack, so it gets its OWN pass with the
# monitoring-appropriate invariants. STILL static — it proves the manifests
# declare the contract, not that the stack runs (runtime evidence is PR 6).
MONITORING_DIR = "k8s/monitoring/base"

# The one workload permitted the hostPath exception: node-exporter must read the
# node's kernel interfaces, which no other component may. Every hostPath it uses
# must still be read-only, and it stays otherwise fully hardened (ADR-029).
HOSTPATH_EXEMPT_WORKLOAD = "node-exporter"

# The only verbs a monitoring RBAC rule may use — these components OBSERVE the API,
# they never mutate it. Anything else (create/update/patch/delete/*, escalate,
# bind, impersonate) is a least-privilege violation.
READ_ONLY_VERBS = {"get", "list", "watch"}

# The namespace the monitoring stack is designed around.
MONITORING_NAMESPACE = "monitoring"


def _mounts_readonly(container: dict, vol_names: set[str]) -> bool:
    """True if every mount of a named volume in this container sets readOnly."""
    return all(
        m.get("readOnly") is True
        for m in volume_mounts_of(container)
        if m.get("name") in vol_names
    )


def validate_monitoring(path: str = MONITORING_DIR) -> int:
    """Validate the monitoring stack render against its own security contract."""
    r = Report()
    PSA = "pod-security.kubernetes.io"

    # -- M1. Render -- #
    sec = "M1. Monitoring render"
    rendered = render(path)
    r.check(sec, f"kustomize build {path}", bool(rendered.strip()))
    docs = load_docs(rendered)
    by_kind: dict[str, list[dict]] = {}
    for d in docs:
        by_kind.setdefault(d.get("kind", "?"), []).append(d)

    # -- M2. Namespace & the documented Pod Security exception -- #
    sec = "M2. Namespace & Pod Security exception"
    ns = next(
        (
            n
            for n in by_kind.get("Namespace", [])
            if n.get("metadata", {}).get("name") == MONITORING_NAMESPACE
        ),
        None,
    )
    r.check(
        sec,
        f"Namespace/{MONITORING_NAMESPACE} present",
        ns is not None,
        f"expected a Namespace/{MONITORING_NAMESPACE}",
    )
    ns_labels = ((ns or {}).get("metadata", {}) or {}).get("labels", {}) or {}
    r.check(
        sec,
        f"Namespace/{MONITORING_NAMESPACE} declares a PSA enforce level",
        bool(ns_labels.get(f"{PSA}/enforce")),
        f"{PSA}/enforce={ns_labels.get(f'{PSA}/enforce')!r} "
        f"(the node-exporter hostPath exception, ADR-029)",
    )
    # Every PSA level present must pin its version (deterministic ruleset).
    for level in ("enforce", "warn", "audit"):
        if not ns_labels.get(f"{PSA}/{level}"):
            continue
        ver = ns_labels.get(f"{PSA}/{level}-version")
        r.check(
            sec,
            f"Namespace/{MONITORING_NAMESPACE} pins the {level} policy version",
            bool(ver) and ver != "latest",
            f"{PSA}/{level}-version={ver!r} (must be pinned, not 'latest')",
        )
    # warn/audit should still catch regressions beyond the node-exporter exception.
    r.check(
        sec,
        f"Namespace/{MONITORING_NAMESPACE} warns or audits (regression backstop)",
        bool(ns_labels.get(f"{PSA}/warn")) or bool(ns_labels.get(f"{PSA}/audit")),
        "neither warn nor audit is set — a regression beyond node-exporter's "
        "hostPath would pass admission unnoticed",
    )

    # Namespaced objects pinned to `monitoring` (cluster-scoped kinds exempt).
    cluster_scoped = CLUSTER_SCOPED_KINDS | {"ClusterRole", "ClusterRoleBinding"}
    for d in docs:
        kind = d.get("kind", "?")
        if kind in cluster_scoped:
            continue
        name = d.get("metadata", {}).get("name", "?")
        nsval = d.get("metadata", {}).get("namespace")
        r.check(
            sec,
            f"{kind}/{name} pinned to {MONITORING_NAMESPACE}",
            nsval == MONITORING_NAMESPACE,
            f"namespace={nsval!r}, expected {MONITORING_NAMESPACE!r}",
        )

    workloads = workloads_in(by_kind)
    r.check(
        sec,
        "at least one monitoring workload rendered",
        bool(workloads),
        "no Deployment/DaemonSet/StatefulSet found in the monitoring render",
    )

    # -- M3. Required fields (pinned images, resources) -- #
    sec = "M3. Required fields (images, resources)"
    for kind, name, wl in workloads:
        for role, c in all_containers_of(wl):
            cname = c.get("name", "?")
            image = c.get("image", "")
            has_ref = bool(image) and (":" in image or "@" in image)
            tag = (
                image.rsplit(":", 1)[-1] if ":" in image and "@" not in image else image
            )
            pinned = has_ref and tag != "latest"
            r.check(
                sec,
                f"{kind}/{name}: {role} {cname} explicit pinned image",
                pinned,
                f"image={image!r} (needs an explicit non-latest tag/digest)",
            )
            reqs = c.get("resources", {}).get("requests", {}) or {}
            lims = c.get("resources", {}).get("limits", {}) or {}
            r.check(
                sec,
                f"{kind}/{name}: {role} {cname} CPU/memory requests+limits",
                bool(reqs.get("cpu"))
                and bool(reqs.get("memory"))
                and bool(lims.get("cpu"))
                and bool(lims.get("memory")),
                f"requests={reqs}, limits={lims}",
            )

    # -- M4. Hardening + the node-exporter hostPath exception -- #
    sec = "M4. Hardening & the hostPath exception"
    for kind, name, wl in workloads:
        wpod = pod_spec_of(wl)
        wsc = wpod.get("securityContext", {}) or {}
        r.check(
            sec,
            f"{kind}/{name}: pod runAsNonRoot true",
            wsc.get("runAsNonRoot") is True,
            f"runAsNonRoot={wsc.get('runAsNonRoot')!r}",
        )
        wuid = wsc.get("runAsUser")
        r.check(
            sec,
            f"{kind}/{name}: pod runAsUser is a non-root uid",
            isinstance(wuid, int) and wuid != 0,
            f"runAsUser={wuid!r} (must be an explicit numeric uid != 0)",
        )
        r.check(
            sec,
            f"{kind}/{name}: pod seccompProfile RuntimeDefault",
            (wsc.get("seccompProfile", {}) or {}).get("type") == "RuntimeDefault",
            f"seccompProfile.type="
            f"{(wsc.get('seccompProfile', {}) or {}).get('type')!r}",
        )
        # No host-namespace sharing — node-exporter deliberately avoids it too, so
        # the exception stays scoped to read-only hostPath alone.
        for host_ns in ("hostNetwork", "hostPID", "hostIPC"):
            r.check(
                sec,
                f"{kind}/{name}: {host_ns} not enabled",
                wpod.get(host_ns) in (None, False),
                f"{host_ns}={wpod.get(host_ns)!r} (host-namespace sharing forbidden)",
            )
        # Every container hardened.
        for role, c in all_containers_of(wl):
            cname = c.get("name", "?")
            csc = c.get("securityContext", {}) or {}
            r.check(
                sec,
                f"{kind}/{name}: {role} {cname} allowPrivilegeEscalation false",
                csc.get("allowPrivilegeEscalation") is False,
                f"allowPrivilegeEscalation={csc.get('allowPrivilegeEscalation')!r}",
            )
            r.check(
                sec,
                f"{kind}/{name}: {role} {cname} not privileged",
                csc.get("privileged") is not True,
                f"privileged={csc.get('privileged')!r}",
            )
            drops = {
                str(x).upper()
                for x in (csc.get("capabilities", {}) or {}).get("drop", [])
            }
            r.check(
                sec,
                f"{kind}/{name}: {role} {cname} capabilities drop ALL",
                "ALL" in drops,
                f"capabilities.drop={sorted(drops)}",
            )
            # readOnlyRootFilesystem: every monitoring container locks its root FS.
            # Unlike the mlops pipeline Job (which must keep it writable for
            # `dvc repro`, ADR-010), no monitoring component writes to its root:
            # Prometheus/pushgateway use dedicated volumes / in-memory state, KSM and
            # node-exporter write nothing. Asserting it here means a future monitoring
            # workload cannot silently drop the control and still pass — backing the
            # "hardened" claim the ADRs make (Sprint 8, PR 3 review follow-up).
            r.check(
                sec,
                f"{kind}/{name}: {role} {cname} readOnlyRootFilesystem true",
                csc.get("readOnlyRootFilesystem") is True,
                f"readOnlyRootFilesystem={csc.get('readOnlyRootFilesystem')!r} "
                f"(monitoring containers must lock the root filesystem)",
            )
        # hostPath: permitted ONLY on node-exporter, and only read-only.
        hostpath_vols = {v.get("name") for v in volumes_of(wl) if "hostPath" in v}
        if name == HOSTPATH_EXEMPT_WORKLOAD:
            if hostpath_vols:
                ro_ok = all(
                    _mounts_readonly(c, hostpath_vols) for _, c in all_containers_of(wl)
                )
                r.check(
                    sec,
                    f"{kind}/{name}: every hostPath mount is read-only (the exception)",
                    ro_ok,
                    "node-exporter's hostPath mounts must all set readOnly: true",
                )
        else:
            r.check(
                sec,
                f"{kind}/{name}: no hostPath volumes",
                not hostpath_vols,
                f"hostPath is permitted ONLY on {HOSTPATH_EXEMPT_WORKLOAD}; "
                f"found {sorted(hostpath_vols)}",
            )

    # -- M5. RBAC least privilege (read-only) -- #
    sec = "M5. RBAC least privilege"
    roles = by_kind.get("ClusterRole", []) + by_kind.get("Role", [])
    r.check(sec, "at least one monitoring RBAC role rendered", bool(roles))
    for role in roles:
        rname = role.get("metadata", {}).get("name", "?")
        offending: list[list[str]] = []
        for rule in role.get("rules", []) or []:
            verbs = {str(v).lower() for v in rule.get("verbs", [])}
            extra = verbs - READ_ONLY_VERBS
            if extra:
                offending.append(sorted(extra))
        r.check(
            sec,
            f"{role.get('kind')}/{rname}: read-only verbs only",
            not offending,
            f"non-read verbs present: {offending} "
            f"(only {sorted(READ_ONLY_VERBS)} permitted for monitoring)",
        )
    # Every ClusterRoleBinding binds only ServiceAccounts in the monitoring ns
    # (never a user/group, never a cross-namespace SA).
    for crb in by_kind.get("ClusterRoleBinding", []):
        cname = crb.get("metadata", {}).get("name", "?")
        subs = crb.get("subjects", []) or []
        ok = bool(subs) and all(
            s.get("kind") == "ServiceAccount"
            and s.get("namespace") == MONITORING_NAMESPACE
            for s in subs
        )
        r.check(
            sec,
            f"ClusterRoleBinding/{cname}: binds only monitoring ServiceAccounts",
            ok,
            f"subjects must be ServiceAccounts in {MONITORING_NAMESPACE!r}; got {subs}",
        )

    # -- M6. Token mounted IFF the SA has API access -- #
    sec = "M6. ServiceAccount token discipline"
    api_sas = {
        s.get("name")
        for crb in by_kind.get("ClusterRoleBinding", [])
        for s in (crb.get("subjects", []) or [])
        if s.get("kind") == "ServiceAccount"
    }
    for sa in by_kind.get("ServiceAccount", []):
        sname = sa.get("metadata", {}).get("name", "?")
        automount = sa.get("automountServiceAccountToken")
        if sname in api_sas:
            r.check(
                sec,
                f"ServiceAccount/{sname}: token mounted (genuinely needs the API)",
                automount is True,
                f"automountServiceAccountToken={automount!r} — {sname} is bound to a "
                f"ClusterRole, so it must mount its token (the documented exception)",
            )
        else:
            r.check(
                sec,
                f"ServiceAccount/{sname}: token NOT mounted (no API access)",
                automount is False,
                f"automountServiceAccountToken={automount!r} — {sname} has no RBAC, "
                f"so mounting a token would be an unused, exfiltratable credential",
            )

    # -- M7. Internal-only exposure -- #
    sec = "M7. Internal-only exposure"
    for svc in by_kind.get("Service", []):
        sname = svc.get("metadata", {}).get("name", "?")
        stype = svc.get("spec", {}).get("type", "ClusterIP")
        r.check(
            sec,
            f"Service/{sname} is ClusterIP (internal-only)",
            stype == "ClusterIP",
            f"type={stype!r} — Prometheus/exporters must never be exposed externally",
        )

    # -- M8. Prometheus scrape config wiring -- #
    sec = "M8. Prometheus scrape config"
    prom_cm = next(
        (
            cm
            for cm in by_kind.get("ConfigMap", [])
            if cm.get("metadata", {}).get("name") == "prometheus-config"
        ),
        None,
    )
    r.check(sec, "prometheus-config ConfigMap present", prom_cm is not None)
    if prom_cm is not None:
        cfg = (prom_cm.get("data", {}) or {}).get("prometheus.yml", "")
        scrape_needles = (
            "kube-state-metrics",
            "node-exporter",
            "cadvisor",
            "pushgateway",
            # Sprint 8, PR 4 (ADR-031) — Layer 3/4 depth:
            "blackbox-mlflow-health",  # MLflow /health availability (Layer 3)
            "postgres-exporter",  # PostgreSQL backend health (Layer 4)
            # The Postgres PVC-fill signal comes from the kubelet's volume stats;
            # the metric_relabel keep names the family, so this needle proves the
            # kubelet scrape exists AND is scoped to just those series (Layer 4).
            "kubelet_volume_stats_",
        )
        for needle in scrape_needles:
            r.check(
                sec,
                f"scrape config covers {needle}",
                needle in cfg,
                f"prometheus.yml has no scrape job referencing {needle}",
            )
        # The Pushgateway scrape MUST set honor_labels so the pipeline's pushed
        # job/stage labels survive the scrape instead of being overwritten with the
        # gateway's own job name (Sprint 8, PR 3; ADR-030). Without it, per-stage
        # attribution is silently lost.
        r.check(
            sec,
            "pushgateway scrape sets honor_labels",
            "honor_labels: true" in cfg,
            "the pushgateway scrape job must set 'honor_labels: true' so pushed "
            "job/stage labels are preserved (ADR-030)",
        )
        # The MLflow availability probe must target MLflow's STABLE /health endpoint
        # (exempt from the server's host allow-list, and a cheap "server up" handler)
        # — not the root or a heavier path (ADR-031). This asserts the blackbox job
        # actually probes the right thing, not merely that a job name exists.
        r.check(
            sec,
            "blackbox scrape targets MLflow /health",
            "mlflow.mlops.svc.cluster.local:5000/health" in cfg,
            "the blackbox scrape must probe the MLflow /health endpoint "
            "(the stable, load-free availability target — ADR-031)",
        )
        # Sprint 8, PR 6 (ADR-033): the config must LOAD the alert rules via
        # rule_files, and point at the path the prometheus-alerts ConfigMap is
        # mounted (/etc/prometheus/rules). Without this the rules ship in a
        # ConfigMap but are never evaluated.
        r.check(
            sec,
            "prometheus.yml wires rule_files for the alert rules",
            "rule_files:" in cfg and "/etc/prometheus/rules/alerts.yml" in cfg,
            "prometheus.yml must declare rule_files including "
            "/etc/prometheus/rules/alerts.yml (ADR-033)",
        )

    # -- M9. Layer 3 exporter (blackbox) present & wired -- #
    # postgres-exporter (Layer 4) is deliberately NOT here — it lives in the mlops
    # namespace beside the DB (so its credential Secret never enters `monitoring`),
    # and is validated by the mlops pass (Section 6f). blackbox-exporter carries no
    # credentials and is a pure monitoring component, so it lives in this stack.
    sec = "M9. Layer 3/4 exporters"
    bb_dep = next(
        (
            d
            for d in by_kind.get("Deployment", [])
            if d.get("metadata", {}).get("name") == "blackbox-exporter"
        ),
        None,
    )
    r.check(
        sec,
        "blackbox-exporter Deployment present (Layer 3 availability)",
        bb_dep is not None,
        "no Deployment/blackbox-exporter in the monitoring render",
    )
    r.check(
        sec,
        "blackbox-exporter Service present",
        any(
            s.get("metadata", {}).get("name") == "blackbox-exporter"
            for s in by_kind.get("Service", [])
        ),
        "no Service/blackbox-exporter in the monitoring render",
    )
    # Its module config must define an http prober (the /health probe recipe).
    bb_cfgs = [
        cm.get("data", {}).get("blackbox.yml", "")
        for cm in by_kind.get("ConfigMap", [])
        if cm.get("metadata", {}).get("name") == "blackbox-exporter-config"
    ]
    r.check(
        sec,
        "blackbox-exporter module config defines an http prober",
        any("prober: http" in c for c in bb_cfgs),
        "blackbox-exporter-config must define a module with 'prober: http'",
    )

    # -- M10. Grafana dashboards layer (Sprint 8, PR 5 — ADR-032) -- #
    # Grafana renders the three purpose-built dashboards over the Prometheus datasource;
    # everything (datasource, provider, dashboard JSON) is PROVISIONED from version-
    # controlled files. kustomize embeds the dashboard JSON as an opaque ConfigMap
    # string, so a malformed dashboard would render + schema-validate fine and only
    # break Grafana at runtime. This pass parse-validates each dashboard file, asserts
    # it points at the provisioned datasource, and enforces the brief's "no AWS account
    # IDs on a dashboard" rule — the checks a generic schema validator cannot express.
    sec = "M10. Grafana dashboards"
    grafana_dir = REPO_ROOT / "k8s/monitoring/base/grafana"
    dashboards_dir = grafana_dir / "dashboards"

    # The workload + provisioning must all be present in the render.
    r.check(
        sec,
        "Grafana Deployment present",
        any(
            d.get("metadata", {}).get("name") == "grafana"
            for d in by_kind.get("Deployment", [])
        ),
        "no Deployment/grafana in the monitoring render",
    )
    r.check(
        sec,
        "Grafana Service present",
        any(
            s.get("metadata", {}).get("name") == "grafana"
            for s in by_kind.get("Service", [])
        ),
        "no Service/grafana in the monitoring render",
    )
    cms_by_name = {
        cm.get("metadata", {}).get("name"): cm for cm in by_kind.get("ConfigMap", [])
    }
    for cm_name in (
        "grafana-datasources",
        "grafana-dashboard-provider",
        "grafana-dashboards",
    ):
        r.check(
            sec,
            f"{cm_name} ConfigMap present",
            cm_name in cms_by_name,
            f"no ConfigMap/{cm_name} in the monitoring render (dashboards)",
        )

    # The provisioned datasource must declare the fixed uid the dashboards reference,
    # and it must be internal proxy access (never a browser-direct/public URL).
    ds_cfg = (cms_by_name.get("grafana-datasources", {}).get("data", {}) or {}).get(
        "datasources.yaml", ""
    )
    r.check(
        sec,
        "datasource declares the fixed uid 'prometheus'",
        "uid: prometheus" in ds_cfg,
        "grafana-datasources must pin 'uid: prometheus' (the handle dashboards use)",
    )
    r.check(
        sec,
        "datasource uses server-side proxy access (internal-only)",
        "access: proxy" in ds_cfg,
        "the Prometheus datasource must use access: proxy so Prometheus stays internal",
    )

    # Parse-validate every dashboard JSON file on disk (the source of truth kustomize
    # packages into grafana-dashboards).
    dash_files = (
        sorted(dashboards_dir.glob("*.json")) if dashboards_dir.is_dir() else []
    )
    # The count is pinned to the three dashboards ADR-032 defines (EKS/Platform Health,
    # MLOps Pipeline Operations, MLflow Platform Health) — bump this deliberately when a
    # dashboard is added/removed, in lock-step with the configMapGenerator file list.
    # The per-file "packaged into the ConfigMap" check below is the redundant backstop.
    r.check(
        sec,
        "three dashboard JSON files present",
        len(dash_files) == 3,
        f"expected 3 dashboards in {dashboards_dir}, found {len(dash_files)}",
    )
    cm_data_keys = set(
        (cms_by_name.get("grafana-dashboards", {}).get("data", {}) or {}).keys()
    )
    # A 12-digit run of digits — an AWS account id shape the brief says must not appear.
    account_id = re.compile(r"\b\d{12}\b")
    for f in dash_files:
        rel = f.relative_to(REPO_ROOT).as_posix()
        raw = f.read_text(encoding="utf-8")
        try:
            dash = json.loads(raw)
            parsed = True
        except json.JSONDecodeError as exc:
            parsed = False
            r.check(
                sec, f"dashboard parses as JSON: {rel}", False, f"invalid JSON: {exc}"
            )
        if not parsed:
            continue
        r.check(sec, f"dashboard parses as JSON: {rel}", True)
        panels = dash.get("panels", [])
        r.check(
            sec,
            f"{rel}: has a uid and non-empty panels",
            bool(dash.get("uid")) and bool(panels),
            f"uid={dash.get('uid')!r}, panels={len(panels)}",
        )
        ids = [p.get("id") for p in panels]
        r.check(
            sec,
            f"{rel}: panel ids are unique",
            len(ids) == len(set(ids)),
            f"duplicate panel ids: {ids}",
        )
        r.check(
            sec,
            f"{rel}: every panel has title + type + gridPos",
            all(p.get("title") and p.get("type") and p.get("gridPos") for p in panels),
            "a panel is missing title/type/gridPos",
        )
        # Every query panel must target the provisioned datasource uid (so provisioning
        # is deterministic — no dangling ${DS_*} or a stale hard-coded id).
        bad_ds = [
            (p.get("id"))
            for p in panels
            for t in (p.get("targets", []) or [])
            if (t.get("datasource", {}) or {}).get("uid") not in (None, "prometheus")
        ]
        r.check(
            sec,
            f"{rel}: query panels use the 'prometheus' datasource uid",
            not bad_ds,
            f"panels with a non-prometheus datasource uid: {bad_ds}",
        )
        r.check(
            sec,
            f"{rel}: no AWS account id exposed",
            not account_id.search(raw),
            "a 12-digit account-id-shaped string is present (never on a dashboard)",
        )
        # The file must actually be packaged into the ConfigMap kustomize generates.
        r.check(
            sec,
            f"{rel}: packaged into the grafana-dashboards ConfigMap",
            f.name in cm_data_keys,
            f"{f.name} is not a key in the grafana-dashboards ConfigMap "
            f"(add it to the kustomization configMapGenerator)",
        )

    # -- M11. Alert rules layer (Sprint 8, PR 6 — ADR-033) -- #
    # A small, high-signal alert set encoding the docs/observability.md § 6
    # objectives + § 3 signal catalogue. Like the dashboards, the raw rule file is
    # a first-class, promtool-tested artifact packaged into a ConfigMap (kustomize
    # embeds it as an opaque string, so a malformed rule would render fine and only
    # break Prometheus at load). This pass asserts the file is wired end to end,
    # that every rule carries the required operator metadata (severity + human
    # summary/description + runbook_url), and — the "no arbitrary alerts" contract —
    # that the alert set is EXACTLY the documented one.
    sec = "M11. Alert rules"
    prom_dir = REPO_ROOT / "k8s/monitoring/base/prometheus"
    alerts_file = prom_dir / "alerts.yml"
    alerts_test = prom_dir / "alerts_test.yml"

    # The rules must be packaged into the prometheus-alerts ConfigMap kustomize
    # generates (key: alerts.yml), and Prometheus must mount it where rule_files
    # points (/etc/prometheus/rules).
    alerts_cm_keys = set(
        (cms_by_name.get("prometheus-alerts", {}).get("data", {}) or {}).keys()
    )
    r.check(
        sec,
        "prometheus-alerts ConfigMap present with alerts.yml",
        "alerts.yml" in alerts_cm_keys,
        "no ConfigMap/prometheus-alerts data key 'alerts.yml' in the render "
        "(add prometheus/alerts.yml to the kustomization configMapGenerator)",
    )
    prom_dep = next(
        (
            d
            for d in by_kind.get("Deployment", [])
            if d.get("metadata", {}).get("name") == "prometheus"
        ),
        None,
    )
    if prom_dep is not None:
        spec = prom_dep.get("spec", {}).get("template", {}).get("spec", {})
        mounts = [
            m
            for c in spec.get("containers", [])
            for m in (c.get("volumeMounts", []) or [])
            if m.get("mountPath") == "/etc/prometheus/rules"
        ]
        vols_by_name = {v.get("name"): v for v in spec.get("volumes", [])}
        mount_ok = bool(mounts) and all(
            (vols_by_name.get(m.get("name"), {}).get("configMap", {}) or {}).get("name")
            == "prometheus-alerts"
            for m in mounts
        )
        r.check(
            sec,
            "Prometheus mounts prometheus-alerts at /etc/prometheus/rules",
            mount_ok,
            "the prometheus Deployment must mount the prometheus-alerts ConfigMap "
            "read-only at /etc/prometheus/rules (where rule_files points)",
        )

    # Parse-validate the raw rule file on disk (the source of truth kustomize packs).
    # A 12-digit run of digits — an AWS account id shape the brief says must not leak.
    r.check(
        sec,
        "alerts.yml present on disk",
        alerts_file.is_file(),
        f"missing {alerts_file}",
    )
    r.check(
        sec,
        "alerts_test.yml (promtool unit tests) present on disk",
        alerts_test.is_file(),
        f"missing {alerts_test} — the alert rules must ship promtool unit tests",
    )
    if alerts_file.is_file():
        raw = alerts_file.read_text(encoding="utf-8")
        r.check(
            sec,
            "alerts.yml exposes no AWS account id",
            not account_id.search(raw),
            "a 12-digit account-id-shaped string is present in alerts.yml",
        )
        try:
            spec = yaml.safe_load(raw) or {}
            parsed = True
        except yaml.YAMLError as exc:
            parsed = False
            r.check(sec, "alerts.yml parses as YAML", False, f"invalid YAML: {exc}")
        if parsed:
            r.check(sec, "alerts.yml parses as YAML", True)
            rules = [
                rule
                for group in (spec.get("groups", []) or [])
                for rule in (group.get("rules", []) or [])
            ]
            names = [rule.get("alert") for rule in rules]
            # The "no arbitrary alerts" contract: the set is EXACTLY the documented
            # one (docs/observability.md § 6 + docs/alerting.md). Bump this set
            # deliberately, in lock-step with the docs and the promtool tests, when
            # an alert is added/removed.
            expected = {
                "PipelineJobFailed",
                "PipelineJobOOMKilled",
                "MLflowDown",
                "MLflowMemoryHigh",
                "PostgresDown",
                "PostgresPVCAlmostFull",
                "PostgresMemoryHigh",
                "KubePodCrashLooping",
            }
            r.check(
                sec,
                "alert set is exactly the documented one (no arbitrary alerts)",
                set(names) == expected and len(names) == len(expected),
                f"alerts {sorted(set(names))} != documented {sorted(expected)}",
            )
            # Every rule must carry the full operator contract: a stable expr, a
            # sensible for-duration, a severity label, and human-readable
            # summary/description plus a runbook_url pointing at docs/alerting.md.
            for rule in rules:
                name = rule.get("alert", "?")
                ann = rule.get("annotations", {}) or {}
                labels = rule.get("labels", {}) or {}
                ok = (
                    bool(rule.get("expr"))
                    and bool(rule.get("for"))
                    and labels.get("severity") in {"critical", "warning"}
                    and bool(ann.get("summary"))
                    and bool(ann.get("description"))
                    and "docs/alerting.md" in ann.get("runbook_url", "")
                )
                r.check(
                    sec,
                    f"{name}: has expr, for, severity, summary, "
                    "description, runbook_url",
                    ok,
                    f"{name} is missing a required field "
                    f"(severity={labels.get('severity')!r}, "
                    f"for={rule.get('for')!r}, runbook={ann.get('runbook_url')!r})",
                )

    return r.render()


if __name__ == "__main__":
    print("Kubernetes static validation (k8s/validate.py)")
    code = validate()

    # Also validate the monitoring stack when present (Sprint 8, PR 2). It is a
    # separate kustomize root with its own security contract, so it gets its own
    # pass; its failures fail the whole run.
    mon_code = 0
    if (REPO_ROOT / MONITORING_DIR).exists():
        print("\n\nMonitoring stack static validation (k8s/monitoring)")
        mon_code = validate_monitoring()

    code = code or mon_code
    print(
        "\nRESULT:",
        "PASS - manifests are well-formed, hardened, and complete (STATIC checks only)."
        if code == 0
        else "FAIL - see the [FAIL] lines above.",
    )
    sys.exit(code)
