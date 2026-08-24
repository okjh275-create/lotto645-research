from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from lrp.operations import write_operation_artifact


def test_writer_signature_adds_optional_artifact_key() -> None:
    assert str(
        inspect.signature(
            write_operation_artifact
        )
    ) == (
        "(payload: 'Mapping[str, Any]', *, "
        "output_root: 'str | Path', "
        "artifact_type: 'str', "
        "round_no: 'int', "
        "filename: 'str', "
        "artifact_key: 'str | None' = None) "
        "-> 'dict[str, Any]'"
    )


def test_writer_legacy_path_is_unchanged(
    tmp_path: Path,
) -> None:
    result = write_operation_artifact(
        {"value": 1},
        output_root=tmp_path,
        artifact_type="sample",
        round_no=7,
        filename="data.json",
    )

    assert Path(result["data_path"]) == (
        tmp_path
        / "sample"
        / "round_0007"
        / "data.json"
    )
    assert Path(result["manifest_path"]) == (
        tmp_path
        / "sample"
        / "round_0007"
        / "manifest.json"
    )


def test_writer_keyed_path_is_exact(
    tmp_path: Path,
) -> None:
    result = write_operation_artifact(
        {"value": 1},
        output_root=tmp_path,
        artifact_type="sample",
        round_no=7,
        filename="data.json",
        artifact_key="candidate-a",
    )

    assert Path(result["data_path"]) == (
        tmp_path
        / "sample"
        / "round_0007"
        / "candidate-a"
        / "data.json"
    )
    assert Path(result["manifest_path"]) == (
        tmp_path
        / "sample"
        / "round_0007"
        / "candidate-a"
        / "manifest.json"
    )


def test_writer_keyed_directory_return_is_exact(
    tmp_path: Path,
) -> None:
    result = write_operation_artifact(
        {"value": 1},
        output_root=tmp_path,
        artifact_type="sample",
        round_no=7,
        filename="data.json",
        artifact_key="candidate-a",
    )

    assert Path(result["directory"]) == (
        tmp_path
        / "sample"
        / "round_0007"
        / "candidate-a"
    )


def test_writer_legacy_manifest_identity_is_unchanged(
    tmp_path: Path,
) -> None:
    result = write_operation_artifact(
        {"value": 1},
        output_root=tmp_path,
        artifact_type="sample",
        round_no=7,
        filename="data.json",
    )

    manifest = json.loads(
        Path(result["manifest_path"]).read_text(
            encoding="utf-8"
        )
    )

    assert manifest["schema_version"] == "1.0"
    assert manifest["artifact_type"] == "sample"
    assert manifest["round"] == 7
    assert "artifact_key" not in manifest


def test_writer_keyed_manifest_preserves_core_identity(
    tmp_path: Path,
) -> None:
    result = write_operation_artifact(
        {"value": 1},
        output_root=tmp_path,
        artifact_type="sample",
        round_no=7,
        filename="data.json",
        artifact_key="candidate-a",
    )

    manifest = json.loads(
        Path(result["manifest_path"]).read_text(
            encoding="utf-8"
        )
    )

    assert manifest["schema_version"] == "1.0"
    assert manifest["artifact_type"] == "sample"
    assert manifest["round"] == 7
    assert manifest["artifact_key"] == "candidate-a"


def test_writer_keyed_operation_log_stays_at_output_root(
    tmp_path: Path,
) -> None:
    write_operation_artifact(
        {"value": 1},
        output_root=tmp_path,
        artifact_type="sample",
        round_no=7,
        filename="data.json",
        artifact_key="candidate-a",
    )

    assert (
        tmp_path
        / "operation_log.jsonl"
    ).exists()


def test_writer_keyed_operation_log_records_keyed_path(
    tmp_path: Path,
) -> None:
    result = write_operation_artifact(
        {"value": 1},
        output_root=tmp_path,
        artifact_type="sample",
        round_no=7,
        filename="data.json",
        artifact_key="candidate-a",
    )

    record = json.loads(
        (
            tmp_path
            / "operation_log.jsonl"
        ).read_text(
            encoding="utf-8"
        ).splitlines()[-1]
    )

    assert record["artifact_type"] == "sample"
    assert record["round"] == 7
    assert record["status"] == "PASS"
    assert record["path"] == result["data_path"]


@pytest.mark.parametrize(
    "artifact_key",
    (
        "candidate-a",
        "baseline_01",
        "seed.20260823",
        "A1",
        "a-b_c.d",
    ),
)
def test_writer_accepts_valid_artifact_keys(
    tmp_path: Path,
    artifact_key: str,
) -> None:
    result = write_operation_artifact(
        {"value": 1},
        output_root=tmp_path,
        artifact_type="sample",
        round_no=7,
        filename="data.json",
        artifact_key=artifact_key,
    )

    assert Path(result["directory"]).name == artifact_key


@pytest.mark.parametrize(
    "artifact_key",
    (
        "",
        " ",
        "   ",
        ".",
        "..",
        "../x",
        "..\\x",
        "a/b",
        "a\\b",
        "/absolute",
        "\\absolute",
        "C:\\absolute",
        "a:b",
        "a b",
        "한글",
        "a@b",
        "a#b",
        "a" * 129,
    ),
)
def test_writer_rejects_invalid_artifact_keys(
    tmp_path: Path,
    artifact_key: str,
) -> None:
    with pytest.raises(ValueError):
        write_operation_artifact(
            {"value": 1},
            output_root=tmp_path,
            artifact_type="sample",
            round_no=7,
            filename="data.json",
            artifact_key=artifact_key,
        )


def test_writer_accepts_artifact_key_at_128_chars(
    tmp_path: Path,
) -> None:
    key = "a" * 128

    result = write_operation_artifact(
        {"value": 1},
        output_root=tmp_path,
        artifact_type="sample",
        round_no=7,
        filename="data.json",
        artifact_key=key,
    )

    assert Path(result["directory"]).name == key