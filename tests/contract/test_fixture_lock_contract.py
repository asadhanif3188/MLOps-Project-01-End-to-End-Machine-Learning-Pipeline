"""Contract test: the committed fixture ``dvc.lock`` is not *structurally* stale.

The fixture reproducibility pipeline (``tests/fixtures/pipeline/``) commits a
``dvc.lock`` so that lock-state reproduction is a checked-in fact, not a promise
(see ADR-008 and pipeline-contract §7). CI proves the lock reproduces by
*running* ``dvc repro`` against it; this test is the complementary offline,
platform-independent half — it proves the committed lock still describes the same
pipeline the fixture ``dvc.yaml`` declares.

Why a separate, hash-free check: DVC output hashes are platform-sensitive (a
model pickled on Linux need not be byte-identical to one pickled on Windows), so
asserting the lock's *output hashes* in the test suite would flake across
machines. Instead this pins the parts that must never silently diverge — the set
of stages and, per stage, the command, the declared dependency and output
*paths*, and the declared parameter *keys*. If someone edits the fixture
``dvc.yaml`` (adds a stage, retargets a dep, renames an output) without
regenerating the lock, the lock is stale in the way that matters and this fails —
in the ordinary ``pytest`` run, with no DVC, no network, and no data.

It parses YAML only; like the pipeline-contract tests it never runs a stage.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "pipeline"


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path} did not parse to a mapping"
    return data


@pytest.fixture(scope="module")
def fixture_dvc() -> dict[str, Any]:
    """The fixture ``dvc.yaml`` ``stages`` mapping (the declared pipeline)."""
    stages = _load_yaml(_FIXTURE_DIR / "dvc.yaml").get("stages")
    assert isinstance(stages, dict) and stages, "fixture dvc.yaml has no stages"
    return stages


@pytest.fixture(scope="module")
def fixture_lock() -> dict[str, Any]:
    """The committed fixture ``dvc.lock`` ``stages`` mapping (the resolved state)."""
    lock_path = _FIXTURE_DIR / "dvc.lock"
    assert lock_path.is_file(), (
        "fixture dvc.lock is missing — regenerate it with "
        "`dvc repro tests/fixtures/pipeline/dvc.yaml` and commit it (ADR-008)"
    )
    stages = _load_yaml(lock_path).get("stages")
    assert isinstance(stages, dict) and stages, "fixture dvc.lock has no stages"
    return stages


def _dep_paths(entry: dict[str, Any]) -> set[str]:
    return {d["path"] if isinstance(d, dict) else d for d in entry.get("deps", [])}


def _out_paths(entry: dict[str, Any]) -> set[str]:
    paths: set[str] = set()
    for out in entry.get("outs", []):
        paths.add(out["path"] if isinstance(out, dict) else out)
    return paths


@pytest.mark.contract
def test_fixture_lock_covers_every_stage(
    fixture_dvc: dict[str, Any], fixture_lock: dict[str, Any]
) -> None:
    """Every declared stage is locked, and the lock invents no extra stage."""
    assert set(fixture_lock) == set(fixture_dvc), (
        "fixture dvc.lock stages diverge from dvc.yaml — the lock is stale; "
        f"dvc.yaml={sorted(fixture_dvc)} lock={sorted(fixture_lock)}"
    )


@pytest.mark.contract
def test_fixture_lock_matches_stage_definitions(
    fixture_dvc: dict[str, Any], fixture_lock: dict[str, Any]
) -> None:
    """Per stage, the lock's command, dep/out *paths*, and param *keys* match.

    This is the structural "not stale" guarantee: the lock records the same
    command, the same set of declared dependency and output paths, and the same
    parameter keys the fixture ``dvc.yaml`` declares. Output *hashes* are
    intentionally not asserted here (they are platform-sensitive and are proven
    to reproduce by the CI ``dvc repro`` step instead).
    """
    problems: list[str] = []
    for name, declared in fixture_dvc.items():
        locked = fixture_lock[name]

        if locked.get("cmd") != declared.get("cmd"):
            problems.append(
                f"{name}: cmd differs (dvc.yaml {declared.get('cmd')!r} vs "
                f"lock {locked.get('cmd')!r})"
            )

        declared_deps = {p.replace("\\", "/") for p in declared.get("deps", [])}
        locked_deps = {p.replace("\\", "/") for p in _dep_paths(locked)}
        if declared_deps != locked_deps:
            problems.append(
                f"{name}: dep paths differ ({declared_deps} vs {locked_deps})"
            )

        declared_outs = {p.replace("\\", "/") for p in declared.get("outs", [])}
        locked_outs = {p.replace("\\", "/") for p in _out_paths(locked)}
        if declared_outs != locked_outs:
            problems.append(
                f"{name}: out paths differ ({declared_outs} vs {locked_outs})"
            )

        declared_params = set(declared.get("params", []))
        locked_params = {
            f"{file}:{key}".replace("params.yaml:", "")
            for file, keys in locked.get("params", {}).items()
            for key in keys
        }
        if declared_params != locked_params:
            problems.append(
                f"{name}: param keys differ ({declared_params} vs {locked_params})"
            )

    assert not problems, "fixture dvc.lock is stale:\n  " + "\n  ".join(problems)
