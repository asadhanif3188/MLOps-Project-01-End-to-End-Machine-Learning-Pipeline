"""Deterministically generate the fixture dataset for the reproducibility pipeline.

This produces ``data/raw/data.csv`` — the *committed* input the fixture pipeline
reproduces (see ``tests/fixtures/pipeline/dvc.yaml`` and
``docs/decisions/ADR-008-fixture-reproducibility.md``). It is **not** the
project's real dataset (the Pima Indians Diabetes set, remote-only via DVC); it
only mimics that set's *schema* so the unmodified ``src/`` stages run against it
unchanged.

Determinism is the whole point: the generator is seeded, so re-running it emits a
byte-identical CSV. The file it writes is committed to git and is what the
fixture pipeline (and the reproducibility tests) consume; regenerate it only by
running this script, so the fixture data always has a reproducible provenance.

    python tests/fixtures/pipeline/generate_data.py

Sizing rationale (kept deliberately small for CI, big enough to exercise the
real stages): ``train`` performs an internal ``train_test_split`` (0.2) and then
a 3-fold ``GridSearchCV``, and ``split``/``train`` stratify on ``Outcome``, so
every class needs enough members to survive both splits. 80 rows, balanced
40/40, clears that with margin while staying a ~4 KB file.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Schema of the real dataset, so the unmodified stages consume the fixture as-is.
_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

_N_PER_CLASS = 40
_SEED = 20240809


def generate() -> pd.DataFrame:
    """Return a seeded, class-balanced DataFrame shaped like the real dataset.

    The two classes are drawn from slightly separated distributions so a Random
    Forest can learn a non-trivial boundary (a genuine, if easy, signal), which
    keeps the evaluation metric meaningful rather than a coin flip. Fully seeded,
    so the output is reproducible.
    """
    rng = np.random.default_rng(_SEED)

    def _class(n: int, center: float) -> pd.DataFrame:
        # Feature means shifted by ``center`` so the classes are separable but
        # overlapping; integer-like columns are rounded to mirror the real data.
        data = {
            "Pregnancies": rng.integers(0, 12, n),
            "Glucose": np.round(rng.normal(100 + center, 20, n)).astype(int),
            "BloodPressure": np.round(rng.normal(70 + center / 3, 10, n)).astype(int),
            "SkinThickness": np.round(rng.normal(25 + center / 4, 8, n)).astype(int),
            "Insulin": np.round(rng.normal(120 + center, 40, n)).astype(int),
            "BMI": np.round(rng.normal(30 + center / 6, 5, n), 1),
            "DiabetesPedigreeFunction": np.round(rng.normal(0.5, 0.2, n), 3),
            "Age": rng.integers(21, 65, n),
        }
        return pd.DataFrame(data, columns=_COLUMNS)

    negatives = _class(_N_PER_CLASS, center=0.0)
    negatives["Outcome"] = 0
    positives = _class(_N_PER_CLASS, center=30.0)
    positives["Outcome"] = 1

    # Interleave then reset the index so the on-disk order is deterministic and
    # not grouped by class (which a naive split could otherwise exploit).
    frame = pd.concat([negatives, positives]).sort_index(kind="stable")
    return frame.reset_index(drop=True)


def main() -> None:
    out = Path(__file__).resolve().parent / "data" / "raw" / "data.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Force LF line endings on every platform (``newline=""`` disables the OS
    # translation; ``lineterminator="\n"`` makes pandas emit LF) so the committed
    # file — and therefore the dvc.lock hash of it — is byte-identical everywhere.
    with open(out, "w", newline="", encoding="utf-8") as fh:
        generate().to_csv(fh, index=False, lineterminator="\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
