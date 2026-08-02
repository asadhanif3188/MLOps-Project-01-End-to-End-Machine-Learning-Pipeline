"""IO, configuration, and serialization helpers with consistent error handling.

Every filesystem, config, and model-serialization boundary in the pipeline goes
through one of these helpers, so failures surface as the pipeline's own typed
:mod:`exceptions` — with an actionable message and the original exception
chained (``raise ... from exc``) to preserve the traceback — instead of raw,
per-stage tracebacks that differ from one stage to the next.

Keeping the ``try``/``except`` blocks here (rather than inlined in each stage)
means the three stages catch the *same* low-level errors and emit the *same*
messages: exception handling is standardized in one place.
"""

import os
import pickle
from collections.abc import Sequence
from typing import Any

import pandas as pd
import yaml

from exceptions import ConfigError, DataError, ModelError


def load_params(path: str, stage: str, required: Sequence[str] = ()) -> dict[str, Any]:
    """Load one stage's parameters from a YAML config file.

    Args:
        path: Path to the YAML file (e.g. ``"params.yaml"``).
        stage: Top-level key to extract (e.g. ``"train"``).
        required: Keys that must be present under ``stage``.

    Returns:
        The parameter mapping for ``stage``. Values are typed ``Any``: YAML is a
        dynamic format, so the concrete type of each parameter is only known to
        the calling stage, which passes them into its own typed ``*(...)`` call
        where the real types are checked.

    Raises:
        ConfigError: If the file is missing or unparseable, the ``stage``
            section is absent or not a mapping, or a ``required`` key is missing.
    """
    try:
        with open(path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError as exc:
        raise ConfigError(
            f"Config file not found: {path!r}. Run the pipeline from the "
            f"repository root, where params.yaml lives."
        ) from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config file {path!r} is not valid YAML: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read config file {path!r}: {exc}") from exc

    if not isinstance(config, dict) or stage not in config:
        raise ConfigError(f"Config file {path!r} has no {stage!r} section.")

    section = config[stage]
    if not isinstance(section, dict):
        raise ConfigError(
            f"Config section {stage!r} in {path!r} must be a mapping, "
            f"got {type(section).__name__}."
        )

    # Presence-only: a required key may legitimately be null (e.g. a scikit-learn
    # ``max_depth`` of ``None`` means unbounded depth), so this validates that the
    # key exists, not that its value is truthy.
    missing = [key for key in required if key not in section]
    if missing:
        raise ConfigError(
            f"Config section {stage!r} in {path!r} is missing key(s): "
            f"{', '.join(missing)}."
        )
    return section


def require_env(name: str) -> str:
    """Return a required environment variable or raise a clear error.

    Args:
        name: Environment variable name.

    Returns:
        The variable's (non-empty) value.

    Raises:
        ConfigError: If the variable is unset or empty.
    """
    value = os.environ.get(name)
    if not value:
        raise ConfigError(
            f"Required environment variable {name} is not set. Copy "
            f".env.example to .env and set {name} (see the README), then re-run."
        )
    return value


def read_csv(path: str) -> pd.DataFrame:
    """Read a CSV dataset into a DataFrame.

    Args:
        path: Path to the CSV file.

    Returns:
        The loaded DataFrame.

    Raises:
        DataError: If the file is missing, empty, or not valid CSV.
    """
    try:
        return pd.read_csv(path)
    except FileNotFoundError as exc:
        raise DataError(
            f"Dataset not found: {path!r}. If it is DVC-tracked, run "
            f"'dvc pull' to fetch it first."
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise DataError(f"Dataset {path!r} is empty.") from exc
    except pd.errors.ParserError as exc:
        raise DataError(f"Dataset {path!r} is not valid CSV: {exc}") from exc
    except OSError as exc:
        raise DataError(f"Could not read dataset {path!r}: {exc}") from exc


def write_csv(
    df: pd.DataFrame, path: str, *, header: bool = True, index: bool = False
) -> None:
    """Write a DataFrame to CSV, creating parent directories.

    Args:
        df: The DataFrame to write.
        path: Destination path for the CSV file.
        header: Whether to write the column header row.
        index: Whether to write the DataFrame index.

    Raises:
        DataError: If the destination cannot be created or written.
    """
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        df.to_csv(path, header=header, index=index)
    except OSError as exc:
        raise DataError(f"Could not write dataset to {path!r}: {exc}") from exc


def ensure_columns(df: pd.DataFrame, required: Sequence[str], source: str) -> None:
    """Validate that a DataFrame contains the required columns.

    Args:
        df: The DataFrame to check.
        required: Column names that must be present.
        source: Human-readable origin of ``df`` (e.g. a file path), used in the
            error message.

    Raises:
        DataError: If any required column is missing.
    """
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise DataError(
            f"Dataset {source!r} is missing required column(s): "
            f"{', '.join(missing)}. Found columns: {list(df.columns)}."
        )


def load_pickle(path: str) -> Any:
    """Load a pickled object (e.g. a trained model) from disk.

    Note:
        ``pickle`` executes arbitrary code on load; only unpickle artifacts this
        pipeline produced. Hardening deserialization is tracked separately in the
        engineering review.

    Args:
        path: Path to the pickle file.

    Returns:
        The deserialized object, typed ``Any``: the on-disk type is not known
        statically, and callers use the result as the concrete artifact they
        expect (e.g. a fitted estimator with ``.predict``).

    Raises:
        ModelError: If the file is missing, or is corrupt / not a valid pickle.
    """
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError as exc:
        raise ModelError(
            f"Model file not found: {path!r}. Run the train stage first to produce it."
        ) from exc
    except (pickle.UnpicklingError, EOFError, ValueError) as exc:
        raise ModelError(
            f"Model file {path!r} is corrupt or not a valid pickle: {exc}"
        ) from exc
    except OSError as exc:
        raise ModelError(f"Could not read model file {path!r}: {exc}") from exc


def save_pickle(obj: object, path: str) -> None:
    """Serialize an object to disk with pickle, creating parent directories.

    ``obj`` is typed ``object`` rather than ``Any``: this helper makes no
    assumptions about — and calls nothing on — the value it serializes, so the
    stricter type documents that and still accepts anything picklable.

    Args:
        obj: The object to serialize (e.g. a fitted estimator).
        path: Destination path for the pickle file.

    Raises:
        ModelError: If the destination cannot be created/written, or ``obj`` is
            not picklable.
    """
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    except OSError as exc:
        raise ModelError(f"Could not write model to {path!r}: {exc}") from exc
    except (pickle.PicklingError, TypeError, AttributeError) as exc:
        raise ModelError(f"Object could not be pickled to {path!r}: {exc}") from exc
