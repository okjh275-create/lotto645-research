from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path

import pytest

from lrp.contracts.exceptions import ContractError
from lrp.operations.durable_replay_artifact_discovery import (
    DurableReplayArtifactDiscoveryRequest,
    DurableReplayArtifactDiscoveryService,
    DurableReplayArtifactSelector,
)


def _selector(
    *,
    artifact_key: str | None = None,
) -> DurableReplayArtifactSelector:
    return DurableReplayArtifactSelector(
        round_no=1231,
        model_name="model-a",
        regime_id="regime-a",
        strategy_name="strategy-a",
        artifact_key=artifact_key,
    )


def _discover(
    tmp_path: Path,
    *,
    candidate_key: str | None,
    baseline_key: str | None,
):
    return DurableReplayArtifactDiscoveryService().discover(
        request=DurableReplayArtifactDiscoveryRequest(
            artifact_root=tmp_path,
            candidate_selectors=(
                _selector(
                    artifact_key=candidate_key,
                ),
            ),
            baseline_selectors=(
                DurableReplayArtifactSelector(
                    round_no=1231,
                    model_name="model-b",
                    regime_id="regime-b",
                    strategy_name="strategy-b",
                    artifact_key=baseline_key,
                ),
            ),
        )
    )


def test_selector_fields_add_only_artifact_key() -> None:
    assert tuple(
        field.name
        for field in dataclasses.fields(
            DurableReplayArtifactSelector
        )
    ) == (
        "round_no",
        "model_name",
        "regime_id",
        "strategy_name",
        "artifact_key",
    )


def test_selector_public_signature_is_backward_compatible() -> None:
    assert str(
        inspect.signature(
            DurableReplayArtifactSelector
        )
    ) == (
        "(round_no: 'int', model_name: 'str', "
        "regime_id: 'str | None' = None, "
        "strategy_name: 'str | None' = None, "
        "artifact_key: 'str | None' = None) -> None"
    )


def test_selector_legacy_construction_preserves_none() -> None:
    selector = DurableReplayArtifactSelector(
        round_no=7,
        model_name="model",
    )

    assert selector.artifact_key is None


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
def test_selector_accepts_valid_artifact_keys(
    artifact_key: str,
) -> None:
    selector = _selector(
        artifact_key=artifact_key,
    )

    assert selector.artifact_key == artifact_key


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
    ),
)
def test_selector_rejects_invalid_artifact_keys(
    artifact_key: str,
) -> None:
    with pytest.raises(ContractError):
        _selector(
            artifact_key=artifact_key,
        )


def test_selector_rejects_artifact_key_over_128_chars() -> None:
    with pytest.raises(ContractError):
        _selector(
            artifact_key="a" * 129,
        )


def test_selector_accepts_artifact_key_at_128_chars() -> None:
    selector = _selector(
        artifact_key="a" * 128,
    )

    assert selector.artifact_key == "a" * 128


def test_legacy_candidate_path_is_unchanged(
    tmp_path: Path,
) -> None:
    candidate, _ = _discover(
        tmp_path,
        candidate_key=None,
        baseline_key=None,
    )

    assert candidate[0].artifact_path == (
        tmp_path
        / "prediction-evaluation-sources"
        / "round_1231"
        / "evaluation_source.json"
    )


def test_legacy_baseline_path_is_unchanged(
    tmp_path: Path,
) -> None:
    _, baseline = _discover(
        tmp_path,
        candidate_key=None,
        baseline_key=None,
    )

    assert baseline[0].artifact_path == (
        tmp_path
        / "prediction-evaluation-sources"
        / "round_1231"
        / "evaluation_source.json"
    )


def test_keyed_candidate_path_is_exact(
    tmp_path: Path,
) -> None:
    candidate, _ = _discover(
        tmp_path,
        candidate_key="candidate-a",
        baseline_key=None,
    )

    assert candidate[0].artifact_path == (
        tmp_path
        / "prediction-evaluation-sources"
        / "round_1231"
        / "candidate-a"
        / "evaluation_source.json"
    )


def test_keyed_baseline_path_is_exact(
    tmp_path: Path,
) -> None:
    _, baseline = _discover(
        tmp_path,
        candidate_key=None,
        baseline_key="baseline-a",
    )

    assert baseline[0].artifact_path == (
        tmp_path
        / "prediction-evaluation-sources"
        / "round_1231"
        / "baseline-a"
        / "evaluation_source.json"
    )


def test_same_round_different_keys_resolve_distinct_paths(
    tmp_path: Path,
) -> None:
    candidate, baseline = _discover(
        tmp_path,
        candidate_key="candidate-a",
        baseline_key="baseline-a",
    )

    assert (
        candidate[0].artifact_path
        != baseline[0].artifact_path
    )


def test_same_round_same_key_resolves_same_path(
    tmp_path: Path,
) -> None:
    candidate, baseline = _discover(
        tmp_path,
        candidate_key="same-a",
        baseline_key="same-a",
    )

    assert (
        candidate[0].artifact_path
        == baseline[0].artifact_path
    )


def test_artifact_key_does_not_change_model_name(
    tmp_path: Path,
) -> None:
    candidate, _ = _discover(
        tmp_path,
        candidate_key="physical-a",
        baseline_key=None,
    )

    assert candidate[0].model_name == "model-a"


def test_artifact_key_does_not_change_regime_id(
    tmp_path: Path,
) -> None:
    candidate, _ = _discover(
        tmp_path,
        candidate_key="physical-a",
        baseline_key=None,
    )

    assert candidate[0].regime_id == "regime-a"


def test_artifact_key_does_not_change_strategy_name(
    tmp_path: Path,
) -> None:
    candidate, _ = _discover(
        tmp_path,
        candidate_key="physical-a",
        baseline_key=None,
    )

    assert candidate[0].strategy_name == "strategy-a"


def test_discovery_does_not_require_keyed_artifact_to_exist(
    tmp_path: Path,
) -> None:
    candidate, _ = _discover(
        tmp_path,
        candidate_key="missing-but-valid",
        baseline_key=None,
    )

    assert candidate[0].artifact_path == (
        tmp_path
        / "prediction-evaluation-sources"
        / "round_1231"
        / "missing-but-valid"
        / "evaluation_source.json"
    )


def test_discovery_does_not_fallback_from_keyed_to_legacy(
    tmp_path: Path,
) -> None:
    legacy = (
        tmp_path
        / "prediction-evaluation-sources"
        / "round_1231"
        / "evaluation_source.json"
    )
    legacy.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    legacy.write_text(
        "{}",
        encoding="utf-8",
    )

    candidate, _ = _discover(
        tmp_path,
        candidate_key="candidate-a",
        baseline_key=None,
    )

    assert candidate[0].artifact_path != legacy
    assert candidate[0].artifact_path == (
        tmp_path
        / "prediction-evaluation-sources"
        / "round_1231"
        / "candidate-a"
        / "evaluation_source.json"
    )


def test_discovery_does_not_scan_keyed_children_for_legacy(
    tmp_path: Path,
) -> None:
    keyed = (
        tmp_path
        / "prediction-evaluation-sources"
        / "round_1231"
        / "candidate-a"
        / "evaluation_source.json"
    )
    keyed.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    keyed.write_text(
        "{}",
        encoding="utf-8",
    )

    candidate, _ = _discover(
        tmp_path,
        candidate_key=None,
        baseline_key=None,
    )

    assert candidate[0].artifact_path == (
        tmp_path
        / "prediction-evaluation-sources"
        / "round_1231"
        / "evaluation_source.json"
    )
    assert candidate[0].artifact_path != keyed


def test_artifact_key_is_not_in_execution_source_public_fields(
    tmp_path: Path,
) -> None:
    candidate, _ = _discover(
        tmp_path,
        candidate_key="physical-a",
        baseline_key=None,
    )

    assert tuple(
        field.name
        for field in dataclasses.fields(
            type(candidate[0])
        )
    ) == (
        "artifact_path",
        "round_no",
        "model_name",
        "regime_id",
        "strategy_name",
    )


def test_request_public_surface_remains_unchanged() -> None:
    assert tuple(
        field.name
        for field in dataclasses.fields(
            DurableReplayArtifactDiscoveryRequest
        )
    ) == (
        "artifact_root",
        "candidate_selectors",
        "baseline_selectors",
    )