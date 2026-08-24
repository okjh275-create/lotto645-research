"""Hardening contracts for durable replay result artifact consumer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lrp.operations.durable_replay_result_artifact_consumer import (
    DurableReplayResultArtifactConsumer,
    DurableReplayResultArtifactConsumerRequest,
)
from lrp.operations.runtime import write_operation_artifact


def _request(root: Path, round_no: int = 1231):
    return DurableReplayResultArtifactConsumerRequest(
        artifact_root=root,
        end_round=round_no,
    )


def _write_valid(root: Path, round_no: int = 1231) -> Path:
    payload = {
        "status": "PASS",
        "candidate_model_name": "candidate-model",
        "baseline_model_name": "baseline-model",
        "round_count": 1,
        "evaluation": {"marker": "au04"},
    }

    result = write_operation_artifact(
        payload,
        output_root=root,
        artifact_type="durable-replay-evaluations",
        round_no=round_no,
        filename="evaluation_result.json",
    )

    return Path(result["data_path"])


def test_valid_artifact_round_trip(tmp_path: Path) -> None:
    _write_valid(tmp_path)

    payload = DurableReplayResultArtifactConsumer().consume(
        request=_request(tmp_path)
    )

    assert payload["status"] == "PASS"
    assert payload["candidate_model_name"] == "candidate-model"


def test_missing_round_directory_fails(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        DurableReplayResultArtifactConsumer().consume(
            request=_request(tmp_path)
        )


def test_missing_manifest_fails(tmp_path: Path) -> None:
    result_path = _write_valid(tmp_path)
    manifest = result_path.parent / "manifest.json"
    manifest.unlink()

    with pytest.raises(Exception):
        DurableReplayResultArtifactConsumer().consume(
            request=_request(tmp_path)
        )


def test_missing_result_file_fails(tmp_path: Path) -> None:
    result_path = _write_valid(tmp_path)
    result_path.unlink()

    with pytest.raises(Exception):
        DurableReplayResultArtifactConsumer().consume(
            request=_request(tmp_path)
        )


def test_tampered_result_fails_manifest_verification(
    tmp_path: Path,
) -> None:
    result_path = _write_valid(tmp_path)
    result_path.write_text(
        json.dumps({"status": "TAMPERED"}),
        encoding="utf-8",
    )

    with pytest.raises(Exception):
        DurableReplayResultArtifactConsumer().consume(
            request=_request(tmp_path)
        )


def test_invalid_json_fails(tmp_path: Path) -> None:
    result_path = _write_valid(tmp_path)

    # Keep manifest valid for deliberately invalid JSON bytes by rewriting
    # artifact through the generic writer contract.
    bad_root = tmp_path / "bad-json"
    write_operation_artifact(
        {"placeholder": True},
        output_root=bad_root,
        artifact_type="durable-replay-evaluations",
        round_no=1231,
        filename="evaluation_result.json",
    )

    bad_result = (
        bad_root
        / "durable-replay-evaluations"
        / "round_1231"
        / "evaluation_result.json"
    )
    bad_manifest = bad_result.parent / "manifest.json"

    raw = b"{not-json"

    import hashlib

    bad_result.write_bytes(raw)
    manifest = json.loads(
        bad_manifest.read_text(encoding="utf-8-sig")
    )
    manifest["files"]["evaluation_result.json"]["sha256"] = (
        hashlib.sha256(raw).hexdigest()
    )
    manifest["files"]["evaluation_result.json"]["bytes"] = len(raw)
    bad_manifest.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    with pytest.raises(json.JSONDecodeError):
        DurableReplayResultArtifactConsumer().consume(
            request=_request(bad_root)
        )


@pytest.mark.parametrize(
    "value",
    (
        [],
        [1, 2, 3],
        "text",
        123,
        True,
        None,
    ),
)
def test_non_object_top_level_is_rejected(
    tmp_path: Path,
    value: object,
) -> None:
    root = tmp_path / ("case-" + str(type(value).__name__))

    result = write_operation_artifact(
        {"placeholder": True},
        output_root=root,
        artifact_type="durable-replay-evaluations",
        round_no=1231,
        filename="evaluation_result.json",
    )

    result_path = Path(result["data_path"])
    manifest_path = result_path.parent / "manifest.json"

    raw = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    import hashlib

    result_path.write_bytes(raw)

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8-sig")
    )
    manifest["files"]["evaluation_result.json"]["sha256"] = (
        hashlib.sha256(raw).hexdigest()
    )
    manifest["files"]["evaluation_result.json"]["bytes"] = len(raw)
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    with pytest.raises(TypeError):
        DurableReplayResultArtifactConsumer().consume(
            request=_request(root)
        )


def test_wrong_round_does_not_fallback_to_existing_round(
    tmp_path: Path,
) -> None:
    _write_valid(tmp_path, round_no=1231)

    with pytest.raises(Exception):
        DurableReplayResultArtifactConsumer().consume(
            request=_request(tmp_path, round_no=1232)
        )


def test_consumer_does_not_scan_sibling_rounds(
    tmp_path: Path,
) -> None:
    _write_valid(tmp_path, round_no=1230)
    _write_valid(tmp_path, round_no=1232)

    with pytest.raises(Exception):
        DurableReplayResultArtifactConsumer().consume(
            request=_request(tmp_path, round_no=1231)
        )


def test_result_artifact_is_not_modified_by_consume(
    tmp_path: Path,
) -> None:
    result_path = _write_valid(tmp_path)
    before = result_path.read_bytes()

    DurableReplayResultArtifactConsumer().consume(
        request=_request(tmp_path)
    )

    after = result_path.read_bytes()

    assert after == before