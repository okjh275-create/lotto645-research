from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import pytest

from lrp.operations.durable_replay_result_artifact_consumer import (
    DurableReplayResultArtifactConsumerRequest,
)
from lrp.operations.durable_replay_result_artifact_inspection import (
    DurableReplayResultArtifactInspectionService,
)
from lrp.operations.runtime import write_operation_artifact


ROUND = 1231


def _request(root: Path) -> DurableReplayResultArtifactConsumerRequest:
    return DurableReplayResultArtifactConsumerRequest(
        artifact_root=root,
        end_round=ROUND,
    )


def _write_payload(
    root: Path,
    payload: object,
) -> Path:
    write_operation_artifact(
        payload=payload,  # type: ignore[arg-type]
        output_root=root,
        artifact_type="durable-replay-evaluations",
        round_no=ROUND,
        filename="evaluation_result.json",
    )
    return (
        root
        / "durable-replay-evaluations"
        / f"round_{ROUND:04d}"
        / "evaluation_result.json"
    )


def _valid_payload() -> dict[str, object]:
    return {
        "status": "PASS",
        "round_count": 1,
        "candidate_model_name": "candidate-model",
        "baseline_model_name": "baseline-model",
        "evaluation": {
            "model_name": "candidate-model",
            "score": 0.75,
            "nested": {
                "window": "sample",
            },
        },
    }


def test_valid_payload_round_trip_is_exact(tmp_path: Path) -> None:
    payload = _valid_payload()
    _write_payload(tmp_path, payload)

    result = DurableReplayResultArtifactInspectionService().inspect(
        _request(tmp_path)
    )

    assert result.status == payload["status"]
    assert result.round_count == payload["round_count"]
    assert result.candidate_model_name == payload["candidate_model_name"]
    assert result.baseline_model_name == payload["baseline_model_name"]
    assert dict(result.evaluation) == payload["evaluation"]


@pytest.mark.parametrize(
    "missing_key",
    [
        "status",
        "round_count",
        "candidate_model_name",
        "baseline_model_name",
        "evaluation",
    ],
)
def test_missing_required_field_fails(
    tmp_path: Path,
    missing_key: str,
) -> None:
    payload = _valid_payload()
    del payload[missing_key]
    _write_payload(tmp_path, payload)

    with pytest.raises(KeyError):
        DurableReplayResultArtifactInspectionService().inspect(
            _request(tmp_path)
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("status", 1),
        ("round_count", "1"),
        ("candidate_model_name", 7),
        ("baseline_model_name", False),
        ("evaluation", []),
    ],
)
def test_wrong_required_field_type_fails(
    tmp_path: Path,
    field_name: str,
    invalid_value: object,
) -> None:
    payload = _valid_payload()
    payload[field_name] = invalid_value
    _write_payload(tmp_path, payload)

    with pytest.raises(TypeError):
        DurableReplayResultArtifactInspectionService().inspect(
            _request(tmp_path)
        )


def test_evaluation_mapping_is_top_level_read_only(tmp_path: Path) -> None:
    _write_payload(tmp_path, _valid_payload())

    result = DurableReplayResultArtifactInspectionService().inspect(
        _request(tmp_path)
    )

    assert isinstance(result.evaluation, MappingProxyType)

    with pytest.raises(TypeError):
        result.evaluation["x"] = 1  # type: ignore[index]


def test_result_dataclass_is_immutable(tmp_path: Path) -> None:
    _write_payload(tmp_path, _valid_payload())

    result = DurableReplayResultArtifactInspectionService().inspect(
        _request(tmp_path)
    )

    with pytest.raises(Exception):
        result.status = "FAIL"  # type: ignore[misc]


def test_manifest_verification_failure_propagates(tmp_path: Path) -> None:
    result_path = _write_payload(tmp_path, _valid_payload())

    result_path.write_text(
        json.dumps({"status": "TAMPERED"}),
        encoding="utf-8",
    )

    with pytest.raises(Exception):
        DurableReplayResultArtifactInspectionService().inspect(
            _request(tmp_path)
        )


def test_wrong_round_does_not_fallback(tmp_path: Path) -> None:
    _write_payload(tmp_path, _valid_payload())

    with pytest.raises(FileNotFoundError):
        DurableReplayResultArtifactInspectionService().inspect(
            DurableReplayResultArtifactConsumerRequest(
                artifact_root=tmp_path,
                end_round=ROUND + 1,
            )
        )


def test_inspection_does_not_modify_result_or_manifest(
    tmp_path: Path,
) -> None:
    result_path = _write_payload(tmp_path, _valid_payload())
    manifest_path = result_path.parent / "manifest.json"

    result_before = result_path.read_bytes()
    manifest_before = manifest_path.read_bytes()

    DurableReplayResultArtifactInspectionService().inspect(
        _request(tmp_path)
    )

    assert result_path.read_bytes() == result_before
    assert manifest_path.read_bytes() == manifest_before


def test_inspection_does_not_create_new_files(tmp_path: Path) -> None:
    _write_payload(tmp_path, _valid_payload())
    before = sorted(
        path.relative_to(tmp_path).as_posix()
        for path in tmp_path.rglob("*")
        if path.is_file()
    )

    DurableReplayResultArtifactInspectionService().inspect(
        _request(tmp_path)
    )

    after = sorted(
        path.relative_to(tmp_path).as_posix()
        for path in tmp_path.rglob("*")
        if path.is_file()
    )

    assert after == before
