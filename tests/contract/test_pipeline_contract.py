"""Contract tests: the pipeline *definition* obeys ``docs/pipeline-contract.md``.

Every other test exercises code. These assert a different, orthogonal property:
that the three declaration files the pipeline is wired from —
:file:`dvc.yaml`, :file:`params.yaml`, and the stage scripts under :file:`src/` —
agree with each other and with the engineering contract, *before* a single stage
runs. They are the CI-enforceable half of the contract (Sprint 4 PR 6): the
half a reviewer would otherwise have to re-check by eye on every change.

Why this is a pure, offline test with **no external dependency**:

* It parses YAML and inspects the filesystem. It never runs a stage, never reads
  ``data/``, never imports scikit-learn or MLflow, and never contacts DVC's
  remote (DagsHub S3) or MLflow. So it is deterministic and needs no
  credentials, no network, and no ``dvc pull`` — exactly the property the
  contract (§9) requires of CI-time validation.
* It is the deterministic complement to the ``dvc dag`` step in CI: ``dvc dag``
  proves DVC itself accepts and can build the graph; these tests pin the graph's
  *shape* and its cross-file consistency to the contract, which ``dvc dag`` does
  not check (DVC does not validate that a ``params:`` key actually exists in
  ``params.yaml`` until it runs a stage against real data).

Each failure below maps to a "CI must fail when …" clause: a broken stage
contract, an inconsistent configuration, a mis-owned artifact, or a graph that no
longer matches the declared lineage ``raw -> preprocess -> processed -> train ->
model -> evaluate -> metrics``.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

# Repo root: tests/contract/<this file> -> parents[2] is the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# A sentinel distinct from ``None`` so a param whose value is legitimately null is
# not mistaken for an absent one.
_MISSING = object()

# The declared artifact-lineage chain (contract §2, §7.1). Each tuple is
# (producing stage, the artifact it owns).
_RAW_DATA = "data/raw/data.csv"
_PROCESSED_DATA = "data/processed/data.csv"
_MODEL = "models/model.pkl"
_METRICS = "metrics/metrics.json"


def _load_yaml(name: str) -> dict[str, Any]:
    """Parse a YAML file at the repository root into a mapping."""
    text = (_REPO_ROOT / name).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    assert isinstance(data, dict), f"{name} did not parse to a mapping"
    return data


@pytest.fixture(scope="module")
def dvc_pipeline() -> dict[str, Any]:
    """The parsed ``dvc.yaml`` ``stages`` mapping."""
    stages = _load_yaml("dvc.yaml").get("stages")
    assert isinstance(stages, dict) and stages, "dvc.yaml has no stages"
    return stages


@pytest.fixture(scope="module")
def params() -> dict[str, Any]:
    """The parsed ``params.yaml`` mapping (the authoritative parameter source)."""
    return _load_yaml("params.yaml")


def _normalise_path(path: str) -> str:
    """Normalise a declared path so ``a/b`` and ``a\\b`` compare equal."""
    return path.replace("\\", "/")


def _deps(stage: dict[str, Any]) -> list[str]:
    return [_normalise_path(d) for d in stage.get("deps", [])]


def _param_keys(stage: dict[str, Any]) -> list[str]:
    """Dotted parameter keys a stage declares (only the plain ``section.key``
    string form this pipeline uses; a per-file dict form is not expected here)."""
    keys: list[str] = []
    for entry in stage.get("params", []):
        assert isinstance(entry, str), (
            f"unexpected non-string params entry {entry!r}; this pipeline declares "
            "params as 'section.key' strings"
        )
        keys.append(entry)
    return keys


def _outputs(stage: dict[str, Any]) -> list[str]:
    """Every artifact a stage produces: cached ``outs`` and ``metrics`` alike.

    ``metrics`` entries may be a bare string or a ``{path: {cache: false}}``
    mapping (as ``evaluate`` declares ``metrics/metrics.json``); both forms are
    flattened to their path here.
    """
    paths: list[str] = []
    for out in stage.get("outs", []):
        paths.append(_normalise_path(out if isinstance(out, str) else next(iter(out))))
    for metric in stage.get("metrics", []):
        paths.append(
            _normalise_path(metric if isinstance(metric, str) else next(iter(metric)))
        )
    return paths


def _resolve(mapping: dict[str, Any], dotted: str) -> Any:
    """Resolve a dotted ``a.b.c`` key against a nested mapping, or ``_MISSING``."""
    node: Any = mapping
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


# ---------------------------------------------------------------------------
# Configuration consistency: dvc.yaml <-> params.yaml (contract §4).
# ---------------------------------------------------------------------------


@pytest.mark.contract
def test_every_dvc_param_key_exists_in_params_yaml(
    dvc_pipeline: dict[str, Any], params: dict[str, Any]
) -> None:
    """Every ``params:`` key a stage references is defined in ``params.yaml``.

    This is the classic drift that silently breaks ``dvc repro`` at run time
    (contract inconsistencies C1/C2): ``dvc.yaml`` names ``train.data`` while
    ``params.yaml`` only defines ``train.input``. Caught here at parse time.
    """
    missing: list[str] = []
    for name, stage in dvc_pipeline.items():
        for key in _param_keys(stage):
            if _resolve(params, key) is _MISSING:
                missing.append(f"{name}: {key}")
    assert not missing, (
        "dvc.yaml references params absent from params.yaml: " + ", ".join(missing)
    )


@pytest.mark.contract
def test_no_orphaned_params(
    dvc_pipeline: dict[str, Any], params: dict[str, Any]
) -> None:
    """No parameter is defined for a stage yet referenced by no stage.

    Enforces contract rule §4.3(3) — "no orphaned params". For each section in
    ``params.yaml`` that corresponds to a pipeline stage, every leaf key must be
    referenced by that stage in ``dvc.yaml``, so configuration cannot rot into
    dead, unvalidated knobs.
    """
    referenced = {key for stage in dvc_pipeline.values() for key in _param_keys(stage)}
    orphans: list[str] = []
    for section, values in params.items():
        if section not in dvc_pipeline or not isinstance(values, dict):
            continue
        for leaf in values:
            dotted = f"{section}.{leaf}"
            if dotted not in referenced:
                orphans.append(dotted)
    assert not orphans, (
        "params.yaml defines keys no stage references (orphaned params): "
        + ", ".join(orphans)
    )


@pytest.mark.contract
def test_declared_outputs_match_params(
    dvc_pipeline: dict[str, Any], params: dict[str, Any]
) -> None:
    """Each stage's artifact path in ``dvc.yaml`` equals its ``params.yaml`` value.

    ``preprocess.output``, ``train.output`` and ``evaluate.metrics`` are declared
    in *both* files; if they disagree, the DVC graph and the code's own config
    have silently diverged. Pin them equal.
    """
    expected = {
        "preprocess": ("preprocess.output", _PROCESSED_DATA),
        "train": ("train.output", _MODEL),
        "evaluate": ("evaluate.metrics", _METRICS),
    }
    for stage_name, (param_key, artifact) in expected.items():
        assert artifact in _outputs(dvc_pipeline[stage_name]), (
            f"{stage_name} does not declare output {artifact} in dvc.yaml"
        )
        resolved = _resolve(params, param_key)
        assert _normalise_path(str(resolved)) == artifact, (
            f"params.yaml {param_key}={resolved!r} != dvc.yaml output {artifact}"
        )


# ---------------------------------------------------------------------------
# Graph shape & lineage (contract §2, §5, §11 D1).
# ---------------------------------------------------------------------------


@pytest.mark.contract
def test_pipeline_graph_is_acyclic(dvc_pipeline: dict[str, Any]) -> None:
    """The stage dependency graph has no cycle.

    A cycle means ``dvc repro`` could never terminate; the DAG would be invalid.
    Edges are inferred the way DVC infers them: a stage depends on whatever stage
    produces a file it lists as a dependency.
    """
    producer = {
        out: name for name, stage in dvc_pipeline.items() for out in _outputs(stage)
    }
    graph = {
        name: {producer[dep] for dep in _deps(stage) if dep in producer}
        for name, stage in dvc_pipeline.items()
    }

    visiting, done = set(), set()

    def has_cycle(node: str) -> bool:
        visiting.add(node)
        for nxt in graph[node]:
            if nxt in visiting or (nxt not in done and has_cycle(nxt)):
                return True
        visiting.discard(node)
        done.add(node)
        return False

    cyclic = [name for name in graph if name not in done and has_cycle(name)]
    assert not cyclic, f"pipeline graph contains a cycle involving: {cyclic}"


@pytest.mark.contract
def test_lineage_matches_contract(dvc_pipeline: dict[str, Any]) -> None:
    """The wired graph is the contract's linear chain, not the old raw-fed wiring.

    Pins the TARGET lineage (contract §2 / deviation D1):
    ``raw -> preprocess -> processed -> train -> model -> evaluate -> metrics``.
    In particular ``train`` must consume the *processed* dataset, never the raw
    file — the specific regression D1 was raised to prevent.
    """
    preprocess, train, evaluate = (
        dvc_pipeline["preprocess"],
        dvc_pipeline["train"],
        dvc_pipeline["evaluate"],
    )

    assert _RAW_DATA in _deps(preprocess), "preprocess must consume the raw dataset"
    assert _PROCESSED_DATA in _outputs(preprocess), (
        "preprocess must produce the processed dataset"
    )

    assert _PROCESSED_DATA in _deps(train), "train must consume the processed dataset"
    assert _RAW_DATA not in _deps(train), (
        "train must NOT read raw data directly (contract deviation D1)"
    )
    assert _MODEL in _outputs(train), "train must produce the model artifact"

    assert _MODEL in _deps(evaluate), "evaluate must consume the model"
    assert _METRICS in _outputs(evaluate), (
        "evaluate must produce the metrics artifact (contract deviation D4)"
    )


@pytest.mark.contract
def test_each_artifact_has_exactly_one_producer(
    dvc_pipeline: dict[str, Any],
) -> None:
    """No artifact is written by more than one stage (single-owner rule, §5).

    Two stages writing the same path is a lineage ambiguity DVC cannot resolve and
    a direct violation of artifact ownership.
    """
    owners: dict[str, list[str]] = {}
    for name, stage in dvc_pipeline.items():
        for out in _outputs(stage):
            owners.setdefault(out, []).append(name)
    shared = {path: names for path, names in owners.items() if len(names) > 1}
    assert not shared, f"artifacts with more than one producing stage: {shared}"


@pytest.mark.contract
def test_processed_data_is_consumed_not_orphaned(
    dvc_pipeline: dict[str, Any],
) -> None:
    """``preprocess``'s output is a real dependency of a downstream stage.

    Guards deviation D1 from the other side: it is not enough for ``preprocess``
    to *produce* the processed dataset — some later stage must *consume* it, or the
    artifact is orphaned exactly as it was before Sprint 4.
    """
    consumers = [
        name for name, stage in dvc_pipeline.items() if _PROCESSED_DATA in _deps(stage)
    ]
    assert consumers, f"{_PROCESSED_DATA} is produced but consumed by no stage"


# ---------------------------------------------------------------------------
# Stage command <-> source wiring (reproducibility: code is a tracked input).
# ---------------------------------------------------------------------------


@pytest.mark.contract
def test_stage_commands_reference_tracked_existing_scripts(
    dvc_pipeline: dict[str, Any],
) -> None:
    """Each stage runs a ``src/*.py`` that exists *and* is a declared dependency.

    Two failure modes in one check: a command pointing at a script that was
    renamed/deleted (the stage cannot run), and a script that runs but is not
    listed under ``deps`` (a code change would not invalidate the stage, breaking
    change-detection and reproducibility, contract §7).
    """
    problems: list[str] = []
    for name, stage in dvc_pipeline.items():
        cmd = stage.get("cmd", "")
        script = next((tok for tok in cmd.split() if tok.endswith(".py")), None)
        assert script, f"{name}: command {cmd!r} runs no .py script"
        script = _normalise_path(script)
        if not (_REPO_ROOT / script).is_file():
            problems.append(f"{name}: script {script} does not exist")
        elif script not in _deps(stage):
            problems.append(f"{name}: script {script} is not a declared dep")
    assert not problems, "; ".join(problems)
