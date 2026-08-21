"""Runtime dataset retrieval: fetch the raw dataset from S3 into ``data/raw``.

This is the code the ``fetch-dataset`` init container runs (Sprint 7, PR 8 — closes
finding M-04). It replaces the former ConfigMap dataset-delivery mechanism with a
professional cloud data path:

    S3 (or S3-compatible MinIO)  ->  workload identity  ->  THIS retrieval
                                  ->  /app/data/raw       ->  DVC pipeline

Why an init container running first-party Python, not application code or a shell
``aws s3 cp`` (ADR-027):

* **Not in a DVC stage / application code.** Preprocess and the other stages must
  stay pure computation over ``data/raw`` so ``dvc repro`` remains reproducible and
  the DVC DAG represents the *computation graph*, not data acquisition. Coupling a
  stage to S3 would also force the main container to hold S3 identity. The dataset
  is a stage *input*; acquiring it is a separate concern that runs *before* the DAG.
* **Not an entrypoint wrapper.** The image's ``CMD`` stays exactly ``dvc repro``;
  wrapping it would blur that contract and make failure isolation harder. The pod
  already uses an init container (``wait-for-mlflow``) as its pre-run gate, so this
  fits the established pattern.
* **First-party Python, not shell.** Keeping the logic here (rather than an opaque
  ``aws s3 cp``) makes it unit-testable with an injected client, gives it the
  project's typed :mod:`exceptions` and structured logging, and lets it verify the
  downloaded bytes against the recorded checksum.

Credentials are NEVER read from static keys in the AWS path: boto3's default
credential chain resolves the pod-scoped, short-lived credentials that EKS Pod
Identity serves to the pipeline's service account (terraform/datasets.tf). Locally
the same code runs against MinIO, whose throwaway credentials come from an
out-of-band Secret (k8s/overlays/local) — never committed, never on AWS.

Configuration (all via environment, injected by the K8s manifests; see
``.env.example`` for local runs):

* ``DATASET_S3_URI``      — REQUIRED. ``s3://<bucket>/<key>`` of the dataset object.
* ``DATASET_DEST``        — optional. Local destination path
  (default :data:`DEFAULT_DEST`, ``data/raw/data.csv``).
* ``DATASET_SHA256``      — optional. Expected SHA-256 of the object; when set, a
  mismatch is a hard failure (integrity gate + documents the dataset identity).
* ``AWS_S3_ENDPOINT_URL`` — optional. An S3-compatible endpoint (MinIO locally);
  when set, path-style addressing is used. Unset ⇒ real Amazon S3.

Failure behaviour (requirement 10): any misconfiguration, download failure, or
checksum mismatch raises a typed :class:`~exceptions.PipelineError`; :func:`main`
logs it and exits non-zero, so the init container fails and the Job surfaces a
clear error (and retries per ``backoffLimit``) instead of the pipeline starting
against missing or corrupt data.
"""

import hashlib
import os
from urllib.parse import urlparse

from dotenv import load_dotenv

from exceptions import ConfigError, DataError
from logging_config import configure_logging, get_logger
from pipeline_io import require_env
from pipeline_metrics import reset_pipeline_metrics, time_stage

logger = get_logger("fetch_dataset")

# Environment variables that make up the retrieval configuration. The names match
# exactly what the K8s manifests inject and what ``.env.example`` documents.
DATASET_URI_ENV = "DATASET_S3_URI"
DATASET_DEST_ENV = "DATASET_DEST"
DATASET_SHA256_ENV = "DATASET_SHA256"
S3_ENDPOINT_ENV = "AWS_S3_ENDPOINT_URL"

# Where the preprocess stage expects its raw input (params.yaml: preprocess.input).
DEFAULT_DEST = "data/raw/data.csv"

# Read the object in fixed-size chunks when hashing so a large file is not slurped
# into memory just to checksum it.
_HASH_CHUNK_BYTES = 1024 * 1024


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split an ``s3://bucket/key`` URI into its bucket and key.

    Args:
        uri: The dataset URI, e.g. ``"s3://my-bucket/data/raw.csv"``.

    Returns:
        A ``(bucket, key)`` tuple.

    Raises:
        ConfigError: If ``uri`` is not a well-formed ``s3://`` URI with both a
            bucket and a non-empty key.
    """
    parsed = urlparse(uri)
    key = parsed.path.lstrip("/")
    if parsed.scheme != "s3" or not parsed.netloc or not key:
        raise ConfigError(
            f"{DATASET_URI_ENV}={uri!r} is not a valid S3 URI. Expected "
            f"'s3://<bucket>/<key>' with both a bucket and an object key."
        )
    return parsed.netloc, key


def sha256_of(path: str) -> str:
    """Return the hex SHA-256 digest of a file, read in chunks.

    Args:
        path: Path to the file to hash.

    Returns:
        The lowercase hex digest.

    Raises:
        DataError: If the file cannot be read.
    """
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(_HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DataError(
            f"Could not read {path!r} to verify its checksum: {exc}"
        ) from exc
    return digest.hexdigest()


def build_s3_client(endpoint_url: str | None) -> object:
    """Construct a boto3 S3 client, honouring an optional custom endpoint.

    boto3 is imported here (not at module load) so this module stays importable —
    and unit-testable with an injected client — without the AWS SDK present, and so
    the import cost is only paid when a download actually happens.

    Credentials are resolved by boto3's default chain: EKS Pod Identity on AWS
    (short-lived, pod-scoped — no static keys), or the out-of-band MinIO Secret
    locally. When ``endpoint_url`` is set (MinIO), path-style addressing is forced,
    because virtual-hosted-style (``<bucket>.<host>``) does not resolve for an
    in-cluster MinIO Service.

    Args:
        endpoint_url: An S3-compatible endpoint URL, or ``None`` for Amazon S3.

    Returns:
        A configured boto3 S3 client.
    """
    import boto3
    from botocore.config import Config

    config = Config(s3={"addressing_style": "path"}) if endpoint_url else None
    return boto3.client("s3", endpoint_url=endpoint_url or None, config=config)


def download_object(client: object, bucket: str, key: str, dest: str) -> None:
    """Download ``s3://bucket/key`` to the local path ``dest``.

    Args:
        client: A boto3-compatible S3 client exposing ``download_file``.
        bucket: The S3 bucket name.
        key: The S3 object key.
        dest: Local destination path (parent directories are created).

    Raises:
        DataError: If the object cannot be downloaded (missing, access denied,
            endpoint unreachable, no credentials) or the destination cannot be
            written — each surfaced with an actionable message.
    """
    from botocore.exceptions import BotoCoreError, ClientError

    try:
        parent = os.path.dirname(dest)
        if parent:
            os.makedirs(parent, exist_ok=True)
        client.download_file(bucket, key, dest)  # type: ignore[attr-defined]
    except (BotoCoreError, ClientError) as exc:
        raise DataError(
            f"Failed to download the dataset s3://{bucket}/{key}: {exc}. Verify "
            f"{DATASET_URI_ENV} points at an existing object, the workload identity "
            f"grants read access to it, and the S3 endpoint is reachable."
        ) from exc
    except OSError as exc:
        raise DataError(
            f"Downloaded the dataset but could not write it to {dest!r}: {exc}."
        ) from exc


def verify_checksum(path: str, expected_sha256: str) -> None:
    """Verify a downloaded file matches an expected SHA-256, or fail hard.

    Args:
        path: Path to the downloaded file.
        expected_sha256: The expected lowercase hex SHA-256 digest.

    Raises:
        DataError: If the file's digest does not match ``expected_sha256``.
    """
    actual = sha256_of(path)
    if actual != expected_sha256.strip().lower():
        raise DataError(
            f"Dataset integrity check failed for {path!r}: expected SHA-256 "
            f"{expected_sha256.strip().lower()}, got {actual}. The object in S3 "
            f"does not match the pinned dataset version ({DATASET_SHA256_ENV})."
        )
    logger.info("Dataset checksum verified (sha256=%s)", actual)


def fetch_dataset(*, client: object | None = None) -> str:
    """Resolve configuration, download the dataset, and verify it.

    Args:
        client: An optional pre-built S3 client (used by tests). When ``None``, a
            client is built from the environment via :func:`build_s3_client`.

    Returns:
        The local path the dataset was written to.

    Raises:
        ConfigError: If ``DATASET_S3_URI`` is unset/empty or malformed.
        DataError: If the download fails or the checksum does not match.
    """
    uri = require_env(DATASET_URI_ENV)
    bucket, key = parse_s3_uri(uri)
    dest = os.environ.get(DATASET_DEST_ENV, "").strip() or DEFAULT_DEST
    expected_sha256 = os.environ.get(DATASET_SHA256_ENV, "").strip()
    endpoint_url = os.environ.get(S3_ENDPOINT_ENV, "").strip() or None

    logger.info(
        "Fetching dataset s3://%s/%s -> %s%s",
        bucket,
        key,
        dest,
        f" via {endpoint_url}" if endpoint_url else "",
    )

    if client is None:
        client = build_s3_client(endpoint_url)
    download_object(client, bucket, key, dest)

    # Log the successful RETRIEVAL before the integrity gate runs, so the init
    # container's logs cleanly separate "object retrieved" from "integrity
    # verified". On a checksum mismatch this line is the operational proof that the
    # download itself succeeded and that it was the integrity gate — not retrieval —
    # that rejected the object; without it, a mismatch and an unreachable/missing
    # object are only distinguishable by the *absence* of a later log line. This is
    # additive observability only: it changes no retrieval or failure behaviour (the
    # instrumentation the Sprint 8 PR 10 checksum-mismatch scenario needs to prove
    # retrieve-then-reject ordering from the logs alone).
    logger.info(
        "Dataset retrieved: %s (%d bytes); verifying integrity",
        dest,
        os.path.getsize(dest),
    )

    if expected_sha256:
        verify_checksum(dest, expected_sha256)
    else:
        logger.warning(
            "%s not set — skipping the dataset integrity check. Set it to pin the "
            "dataset version.",
            DATASET_SHA256_ENV,
        )

    size = os.path.getsize(dest)
    logger.info("Dataset ready: %s (%d bytes)", dest, size)
    return dest


def main() -> None:
    """Entry point: load environment, configure logging, retrieve the dataset.

    On any :class:`~exceptions.PipelineError` the failure is logged and the process
    exits non-zero, so the init container fails cleanly and the Job surfaces the
    error rather than the pipeline running against missing/corrupt data.
    """
    load_dotenv()
    configure_logging()

    # This init container is the FIRST thing that runs in a pipeline execution, so
    # it owns the once-per-run metric reset: clear every stage's Pushgateway group
    # up front so a shorter/failed run can never leave a previous run's later-stage
    # series behind as stale data (ADR-030). Best-effort and a no-op unless
    # PUSHGATEWAY_URL is set, so local runs and tests are unaffected.
    reset_pipeline_metrics()

    try:
        # Time the retrieval as the "fetch_dataset" stage; time_stage pushes its
        # duration + success/failure on exit and re-raises any failure unchanged.
        with time_stage("fetch_dataset"):
            fetch_dataset()
    except (ConfigError, DataError) as exc:
        logger.error("Dataset retrieval failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
