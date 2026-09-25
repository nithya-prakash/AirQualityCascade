from pathlib import Path

from aqcascade.tracking.mlflow_utils import dataset_version


def test_dataset_version_is_stable_for_identical_content(tmp_path: Path):
    f1 = tmp_path / "a.parquet"
    f2 = tmp_path / "b.parquet"
    f1.write_bytes(b"identical content")
    f2.write_bytes(b"identical content")
    assert dataset_version(f1) == dataset_version(f2)


def test_dataset_version_changes_when_content_changes(tmp_path: Path):
    f1 = tmp_path / "a.parquet"
    f1.write_bytes(b"version one")
    v1 = dataset_version(f1)
    f1.write_bytes(b"version two")
    v2 = dataset_version(f1)
    assert v1 != v2
