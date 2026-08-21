"""Unit tests for :mod:`fetch_dataset` — the runtime S3 dataset retrieval.

These exercise the retrieval logic with an INJECTED fake S3 client and a real
``botocore`` error type, so they need no network, no AWS credentials, and no
MinIO: the exact properties the pipeline contract requires of unit tests. They
pin the configuration parsing, the integrity gate, and the clear-failure
behaviour that make the init container a trustworthy replacement for the removed
ConfigMap dataset mechanism (M-04).
"""

import hashlib

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

import fetch_dataset
from exceptions import ConfigError, DataError

pytestmark = pytest.mark.unit

_CONTENT = b"col1,col2\n1,2\n3,4\n"
_CONTENT_SHA256 = hashlib.sha256(_CONTENT).hexdigest()


class _FakeS3Client:
    """A minimal boto3-compatible stub whose ``download_file`` writes fixed bytes."""

    def __init__(self, content: bytes = _CONTENT, error: Exception | None = None):
        self.content = content
        self.error = error
        self.calls: list[tuple[str, str, str]] = []

    def download_file(self, bucket: str, key: str, dest: str) -> None:
        self.calls.append((bucket, key, dest))
        if self.error is not None:
            raise self.error
        with open(dest, "wb") as f:
            f.write(self.content)


# --------------------------------------------------------------------------- #
# parse_s3_uri
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("s3://bucket/key.csv", ("bucket", "key.csv")),
        ("s3://my-bucket/a/b/c/data.csv", ("my-bucket", "a/b/c/data.csv")),
        ("s3://b/pima/v1/data.csv", ("b", "pima/v1/data.csv")),
    ],
)
def test_parse_s3_uri_valid(uri: str, expected: tuple[str, str]) -> None:
    assert fetch_dataset.parse_s3_uri(uri) == expected


@pytest.mark.parametrize(
    "uri",
    [
        "https://bucket/key.csv",  # wrong scheme
        "s3://bucket",  # no key
        "s3://bucket/",  # empty key
        "s3:///key.csv",  # no bucket
        "bucket/key.csv",  # no scheme
        "",  # empty
    ],
)
def test_parse_s3_uri_invalid_raises(uri: str) -> None:
    with pytest.raises(ConfigError):
        fetch_dataset.parse_s3_uri(uri)


# --------------------------------------------------------------------------- #
# sha256_of / verify_checksum
# --------------------------------------------------------------------------- #
def test_sha256_of_matches_hashlib(tmp_path) -> None:
    p = tmp_path / "d.csv"
    p.write_bytes(_CONTENT)
    assert fetch_dataset.sha256_of(str(p)) == _CONTENT_SHA256


def test_sha256_of_missing_file_raises(tmp_path) -> None:
    with pytest.raises(DataError):
        fetch_dataset.sha256_of(str(tmp_path / "nope.csv"))


def test_verify_checksum_ok(tmp_path) -> None:
    p = tmp_path / "d.csv"
    p.write_bytes(_CONTENT)
    # No exception on match; tolerant of surrounding whitespace and case.
    fetch_dataset.verify_checksum(str(p), f"  {_CONTENT_SHA256.upper()}  ")


def test_verify_checksum_mismatch_raises(tmp_path) -> None:
    p = tmp_path / "d.csv"
    p.write_bytes(_CONTENT)
    with pytest.raises(DataError, match="integrity check failed"):
        fetch_dataset.verify_checksum(str(p), "0" * 64)


# --------------------------------------------------------------------------- #
# download_object
# --------------------------------------------------------------------------- #
def test_download_object_writes_file_and_creates_parents(tmp_path) -> None:
    dest = tmp_path / "nested" / "raw" / "data.csv"
    client = _FakeS3Client()
    fetch_dataset.download_object(client, "bucket", "key.csv", str(dest))
    assert dest.read_bytes() == _CONTENT
    assert client.calls == [("bucket", "key.csv", str(dest))]


def test_download_object_client_error_becomes_dataerror(tmp_path) -> None:
    err = ClientError({"Error": {"Code": "NoSuchKey", "Message": "nope"}}, "GetObject")
    client = _FakeS3Client(error=err)
    with pytest.raises(DataError, match="Failed to download"):
        fetch_dataset.download_object(
            client, "bucket", "missing.csv", str(tmp_path / "d.csv")
        )


def test_download_object_endpoint_error_becomes_dataerror(tmp_path) -> None:
    err = EndpointConnectionError(endpoint_url="http://minio:9000")
    client = _FakeS3Client(error=err)
    with pytest.raises(DataError, match="Failed to download"):
        fetch_dataset.download_object(
            client, "bucket", "k.csv", str(tmp_path / "d.csv")
        )


# --------------------------------------------------------------------------- #
# fetch_dataset (orchestration)
# --------------------------------------------------------------------------- #
def test_fetch_dataset_downloads_and_verifies(tmp_path, monkeypatch) -> None:
    dest = tmp_path / "data" / "raw" / "data.csv"
    monkeypatch.setenv("DATASET_S3_URI", "s3://bucket/pima/v1/data.csv")
    monkeypatch.setenv("DATASET_DEST", str(dest))
    monkeypatch.setenv("DATASET_SHA256", _CONTENT_SHA256)
    client = _FakeS3Client()

    result = fetch_dataset.fetch_dataset(client=client)

    assert result == str(dest)
    assert dest.read_bytes() == _CONTENT
    assert client.calls == [("bucket", "pima/v1/data.csv", str(dest))]


def test_fetch_dataset_logs_retrieval_before_integrity(
    tmp_path, monkeypatch, caplog
) -> None:
    # The init container's logs must show the object was RETRIEVED before the
    # integrity gate runs, so a checksum mismatch is distinguishable from an
    # unreachable/missing object from the logs alone (Sprint 8 PR 10 observability).
    dest = tmp_path / "data.csv"
    monkeypatch.setenv("DATASET_S3_URI", "s3://bucket/k.csv")
    monkeypatch.setenv("DATASET_DEST", str(dest))
    monkeypatch.setenv("DATASET_SHA256", _CONTENT_SHA256)

    with caplog.at_level("INFO", logger="fetch_dataset"):
        fetch_dataset.fetch_dataset(client=_FakeS3Client())

    messages = [r.getMessage() for r in caplog.records]
    retrieved = next(i for i, m in enumerate(messages) if "Dataset retrieved" in m)
    verified = next(i for i, m in enumerate(messages) if "checksum verified" in m)
    assert retrieved < verified


def test_fetch_dataset_missing_uri_raises_configerror(monkeypatch) -> None:
    monkeypatch.delenv("DATASET_S3_URI", raising=False)
    with pytest.raises(ConfigError):
        fetch_dataset.fetch_dataset(client=_FakeS3Client())


def test_fetch_dataset_checksum_mismatch_raises(tmp_path, monkeypatch) -> None:
    dest = tmp_path / "data.csv"
    monkeypatch.setenv("DATASET_S3_URI", "s3://bucket/k.csv")
    monkeypatch.setenv("DATASET_DEST", str(dest))
    monkeypatch.setenv("DATASET_SHA256", "0" * 64)
    with pytest.raises(DataError, match="integrity check failed"):
        fetch_dataset.fetch_dataset(client=_FakeS3Client())


def test_fetch_dataset_without_checksum_warns_but_succeeds(
    tmp_path, monkeypatch
) -> None:
    dest = tmp_path / "data.csv"
    monkeypatch.setenv("DATASET_S3_URI", "s3://bucket/k.csv")
    monkeypatch.setenv("DATASET_DEST", str(dest))
    monkeypatch.delenv("DATASET_SHA256", raising=False)

    result = fetch_dataset.fetch_dataset(client=_FakeS3Client())

    assert result == str(dest)
    assert dest.read_bytes() == _CONTENT


def test_fetch_dataset_defaults_dest_when_unset(tmp_path, monkeypatch) -> None:
    # DATASET_DEST unset -> DEFAULT_DEST; run from a temp cwd so the relative
    # default path is created under the sandbox, not the repo.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATASET_S3_URI", "s3://bucket/k.csv")
    monkeypatch.delenv("DATASET_DEST", raising=False)
    monkeypatch.delenv("DATASET_SHA256", raising=False)

    result = fetch_dataset.fetch_dataset(client=_FakeS3Client())

    assert result == fetch_dataset.DEFAULT_DEST
    assert (tmp_path / fetch_dataset.DEFAULT_DEST).read_bytes() == _CONTENT


def test_main_exits_nonzero_on_failure(monkeypatch) -> None:
    monkeypatch.delenv("DATASET_S3_URI", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        fetch_dataset.main()
    assert excinfo.value.code == 1
