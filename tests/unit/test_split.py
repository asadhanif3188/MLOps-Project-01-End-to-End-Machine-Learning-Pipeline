"""Unit tests for the train/evaluation split stage.

Two tiers, both requiring only scikit-learn (skipped when absent), never MLflow —
the split stage crosses no tracking boundary:

* *pure computation* — :func:`split.split_dataset`, exercised on in-memory data;
  and
* *the stage orchestrator* — :func:`split.split`, exercised through real files in
  a temp dir, so the read → compute → **persist two outputs** path is validated.

The behaviours pinned here are the held-out guarantee itself (see
pipeline-contract.md §8): the two partitions are **disjoint** (no row is used for
both training and evaluation) and **exhaustive** (no row is lost), the split is
**deterministic** given a seed, stratification preserves the class balance, both
artifacts are produced with headers, and invalid/missing data fails as a typed
:class:`~exceptions.DataError`.
"""

from pathlib import Path

import pytest

pytest.importorskip("sklearn")

import pandas as pd

from exceptions import DataError
from split import split, split_dataset


def _balanced_frame(n: int = 40) -> pd.DataFrame:
    """A deterministic, class-balanced dataset large enough to stratify.

    Each row carries a unique ``Glucose`` value, so a row's identity survives the
    round-trip to CSV and back — which is what lets a test assert two partitions
    share no row.
    """
    return pd.DataFrame(
        {
            "Glucose": list(range(n)),
            "BloodPressure": [x % 7 for x in range(n)],
            "Outcome": [0, 1] * (n // 2),
        }
    )


# --------------------------------------------------------------------------- #
# split_dataset — the pure computation
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_split_dataset_is_deterministic_given_seed() -> None:
    """Same data + same random_state => identical train/held-out partitions.

    This is the reproducibility guarantee for the boundary: the exact rows that
    are held out do not drift between runs."""
    df = _balanced_frame()

    first = split_dataset(df, target="Outcome", test_size=0.25, random_state=42)
    second = split_dataset(df, target="Outcome", test_size=0.25, random_state=42)

    pd.testing.assert_frame_equal(first.train, second.train)
    pd.testing.assert_frame_equal(first.test, second.test)


@pytest.mark.unit
def test_split_dataset_partitions_are_disjoint() -> None:
    """No row appears in both partitions — training and evaluation never overlap.

    The central held-out property: if any row were shared, the reported accuracy
    would be partly in-sample. Identity is checked by the unique ``Glucose``
    value each row carries."""
    df = _balanced_frame()

    result = split_dataset(df, target="Outcome", test_size=0.25, random_state=42)

    train_ids = set(result.train["Glucose"])
    test_ids = set(result.test["Glucose"])
    assert train_ids.isdisjoint(test_ids)


@pytest.mark.unit
def test_split_dataset_partitions_are_exhaustive() -> None:
    """Every row lands in exactly one partition — none lost, none duplicated."""
    df = _balanced_frame()

    result = split_dataset(df, target="Outcome", test_size=0.25, random_state=42)

    assert len(result.train) + len(result.test) == len(df)
    recovered = set(result.train["Glucose"]) | set(result.test["Glucose"])
    assert recovered == set(df["Glucose"])


@pytest.mark.unit
def test_split_dataset_holds_out_configured_fraction() -> None:
    """The held-out set is the configured fraction of the dataset (40 * 0.25)."""
    df = _balanced_frame(n=40)

    result = split_dataset(df, target="Outcome", test_size=0.25, random_state=42)

    assert len(result.test) == 10
    assert len(result.train) == 30


@pytest.mark.unit
def test_split_dataset_preserves_class_balance() -> None:
    """Stratification keeps both classes represented in each partition, so the
    held-out metric is measured against a class ratio like the training data's."""
    df = _balanced_frame(n=40)  # 20 of each class

    result = split_dataset(df, target="Outcome", test_size=0.25, random_state=42)

    # 50/50 source, stratified => each partition stays 50/50.
    assert set(result.train["Outcome"]) == {0, 1}
    assert set(result.test["Outcome"]) == {0, 1}
    assert result.test["Outcome"].sum() == len(result.test) / 2


@pytest.mark.unit
def test_split_dataset_too_imbalanced_to_stratify_raises_data_error() -> None:
    """A class with too few members to stratify fails as a typed ``DataError``,
    not a raw scikit-learn ``ValueError``."""
    df = pd.DataFrame(
        {
            "Glucose": [1, 2, 3, 4, 5],
            "BloodPressure": [1, 2, 3, 4, 5],
            "Outcome": [0, 0, 0, 0, 1],  # the '1' class has a single member
        }
    )

    with pytest.raises(DataError, match="stratify"):
        split_dataset(df, target="Outcome", test_size=0.25, random_state=42)


# --------------------------------------------------------------------------- #
# split — the stage orchestrator (read -> compute -> persist two outputs)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_split_writes_two_disjoint_headered_outputs(tmp_path: Path) -> None:
    """The stage's core contract: a valid processed dataset yields two artifacts
    that exist, are headered (columns selectable by name), are disjoint, and
    together recover the input — all on disk, as the next stages will read them."""
    processed = tmp_path / "processed" / "data.csv"
    processed.parent.mkdir(parents=True)
    df = _balanced_frame()
    df.to_csv(processed, index=False)

    train_out = tmp_path / "processed" / "train.csv"  # nested dir must be created
    test_out = tmp_path / "processed" / "test.csv"

    split(
        str(processed),
        str(train_out),
        str(test_out),
        target="Outcome",
        test_size=0.25,
        random_state=42,
    )

    assert train_out.exists()
    assert test_out.exists()

    train_df = pd.read_csv(train_out)
    test_df = pd.read_csv(test_out)

    # Headered: the target column is selectable by name in both outputs.
    assert "Outcome" in train_df.columns
    assert "Outcome" in test_df.columns

    # Disjoint and exhaustive across the two files on disk.
    train_ids = set(train_df["Glucose"])
    test_ids = set(test_df["Glucose"])
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == set(df["Glucose"])


@pytest.mark.unit
def test_split_missing_input_raises_data_error(tmp_path: Path) -> None:
    """A missing processed dataset fails predictably as a typed ``DataError``
    naming the path, and writes no partitions."""
    train_out = tmp_path / "train.csv"
    test_out = tmp_path / "test.csv"

    with pytest.raises(DataError, match="not found"):
        split(
            str(tmp_path / "absent.csv"),
            str(train_out),
            str(test_out),
            target="Outcome",
            test_size=0.25,
            random_state=42,
        )
    assert not train_out.exists()
    assert not test_out.exists()


@pytest.mark.unit
def test_split_missing_target_column_raises_data_error(tmp_path: Path) -> None:
    """Processed data lacking the configured target column fails as a
    ``DataError`` naming the column, and writes no partitions."""
    processed = tmp_path / "data.csv"
    pd.DataFrame({"Glucose": [1, 2, 3], "BloodPressure": [4, 5, 6]}).to_csv(
        processed, index=False
    )
    train_out = tmp_path / "train.csv"
    test_out = tmp_path / "test.csv"

    with pytest.raises(DataError, match="Outcome"):
        split(
            str(processed),
            str(train_out),
            str(test_out),
            target="Outcome",
            test_size=0.25,
            random_state=42,
        )
    assert not train_out.exists()
    assert not test_out.exists()
