"""Train/evaluation split stage: the pipeline's explicit held-out boundary.

Before this stage, ``preprocess`` produces a single processed dataset. This stage
divides that dataset **once** into two disjoint files — a training set consumed
only by ``train`` and a held-out evaluation set consumed only by ``evaluate`` —
so the model is never scored on rows it was fitted on. Making the split its own
stage (rather than a hidden step inside ``preprocess`` or ``evaluate``) puts the
train/evaluation boundary in the DVC graph, where ``dvc dag`` and the ``contract``
tests can see and enforce it.

Structured like the train/evaluate stages:

* **ML computation** — :func:`split_dataset` takes a DataFrame and returns two
  disjoint DataFrames. It performs no file IO and reads no environment, so it is
  deterministic given its inputs and unit-testable in isolation.
* **Orchestration** — :func:`split` / :func:`main` read config and data, invoke
  the computation, and persist the two owned outputs.

The split is **deterministic**: ``train_test_split`` is seeded from
``split.random_state`` and stratified on the target, so a given processed dataset
and seed always yield the same partition with the same class balance.
"""

from dataclasses import dataclass

import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from exceptions import DataError
from logging_config import configure_logging, get_logger
from pipeline_io import ensure_columns, load_params, read_csv, write_csv
from stage_runner import run_stage

logger = get_logger("split")


@dataclass(frozen=True)
class DatasetSplit:
    """The two disjoint partitions produced by :func:`split_dataset`.

    Attributes:
        train: Rows reserved for model fitting (consumed by ``train``).
        test: Held-out rows reserved for evaluation (consumed by ``evaluate``) —
            never seen by ``train``.
    """

    train: pd.DataFrame
    test: pd.DataFrame


def split_dataset(
    df: pd.DataFrame,
    *,
    target: str,
    test_size: float,
    random_state: int,
) -> DatasetSplit:
    """Partition ``df`` into disjoint training and held-out evaluation frames.

    Deterministic given ``df``, ``test_size``, and ``random_state``: the split is
    seeded, so repeated calls yield the same partition. It is stratified on
    ``target`` so both partitions preserve the dataset's class balance — a
    held-out set with a skewed class ratio would make the evaluation metric hard
    to interpret. Performs no IO.

    The returned partitions are **disjoint and exhaustive**: every row of ``df``
    lands in exactly one of ``train``/``test`` and no row appears in both. This
    invariant — the whole point of a held-out split — is checked before the frames
    are returned and raises :class:`~exceptions.DataError` if violated, so a future
    change that broke it fails loudly rather than silently leaking evaluation rows
    into training. (A raised error, not an ``assert``: the guard must survive
    ``python -O``, which strips ``assert`` statements.)

    Args:
        df: The processed dataset to partition (must contain ``target``).
        target: Name of the label column, used to stratify the split.
        test_size: Fraction of rows to hold out for evaluation (0 < x < 1).
        random_state: Seed for the split (reproducibility).

    Returns:
        A :class:`DatasetSplit` with the disjoint ``train`` and ``test`` frames.

    Raises:
        DataError: If ``df`` cannot be split as requested — e.g. ``test_size`` is
            out of range, or the data is too small / too class-imbalanced to
            stratify (a class with fewer members than the number of partitions).
    """
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[target],
        )
    except ValueError as exc:
        raise DataError(
            f"Cannot split dataset into train/held-out partitions: {exc}. Check "
            f"that 'split.test_size' is between 0 and 1 and that every class in "
            f"{target!r} has enough rows to stratify."
        ) from exc

    # The held-out guarantee, enforced rather than merely documented: the two
    # partitions must be disjoint (no shared row index) and exhaustive (together
    # they are the whole dataset). train_test_split satisfies this by
    # construction; checking it here makes any future regression that leaked rows
    # into both sides fail immediately, as a typed error the pipeline handles like
    # any other data failure (and one that -O cannot strip, unlike an assert).
    if not train_df.index.intersection(test_df.index).empty:
        raise DataError(
            "train/held-out partitions overlap — evaluation data leaked into "
            "training; the split is not disjoint"
        )
    if len(train_df) + len(test_df) != len(df):
        raise DataError(
            "train/held-out partitions do not cover the dataset exactly — "
            f"{len(train_df)} + {len(test_df)} != {len(df)} rows"
        )
    return DatasetSplit(train=train_df, test=test_df)


def split(
    input_path: str,
    train_output: str,
    test_output: str,
    target: str,
    test_size: float,
    random_state: int,
) -> None:
    """Orchestrate the split stage: read → compute → persist two outputs.

    Stage contract:
        * Input:  the processed dataset (``input_path``, the ``preprocess``
          output), containing ``target`` plus feature columns.
        * Outputs: the training dataset (``train_output``) and the held-out
          evaluation dataset (``test_output``) — both owned by this stage; the
          former is consumed only by ``train``, the latter only by ``evaluate``.
        * Configuration: ``target``, ``test_size``, and ``random_state`` from the
          ``split`` section of ``params.yaml``.

    Args:
        input_path: Path to the processed CSV dataset.
        train_output: Path to write the training split (headered CSV).
        test_output: Path to write the held-out evaluation split (headered CSV).
        target: Name of the label column; used to stratify the split.
        test_size: Fraction of rows to hold out for evaluation.
        random_state: Seed for the split (reproducibility).

    Raises:
        DataError: If the dataset cannot be read, lacks ``target``, cannot be
            split as requested, or an output cannot be written.
    """
    logger.info(
        "Split stage started (input=%s, train=%s, test=%s)",
        input_path,
        train_output,
        test_output,
    )

    data = read_csv(input_path)
    ensure_columns(data, [target], input_path)

    result = split_dataset(
        data, target=target, test_size=test_size, random_state=random_state
    )

    # Preserve the header so train/evaluate can select columns by name, exactly
    # like the preprocess output these two files replace as those stages' inputs.
    write_csv(result.train, train_output, header=True, index=False)
    write_csv(result.test, test_output, header=True, index=False)

    logger.info(
        "Split stage completed: %d train rows -> %s, %d held-out rows -> %s",
        len(result.train),
        train_output,
        len(result.test),
        test_output,
    )


def main() -> None:
    """Entry point: load environment, configure logging, run the stage."""
    load_dotenv()
    configure_logging()

    params = load_params(
        "params.yaml",
        "split",
        required=(
            "input",
            "train_output",
            "test_output",
            "target",
            "test_size",
            "random_state",
        ),
    )

    split(
        params["input"],
        params["train_output"],
        params["test_output"],
        params["target"],
        params["test_size"],
        params["random_state"],
    )


if __name__ == "__main__":
    run_stage("split", main)
