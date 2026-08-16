from __future__ import annotations

import json
from pathlib import Path

import pytest

from lrp.production import (
    ProductionPredictionConfiguration,
)


def _write_active_decision(
    registry_root: Path,
    *,
    selected_model: object,
) -> Path:
    path = (
        registry_root
        / "active"
        / "champion_decision.json"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            {
                "selection": {
                    "selected_model": selected_model,
                },
            }
        ),
        encoding="utf-8",
    )

    return path


def test_from_registry_resolves_combined_model(
    tmp_path: Path,
) -> None:
    registry_root = (
        tmp_path
        / "registry"
    )

    snapshot_root = (
        tmp_path
        / "snapshots"
    )

    _write_active_decision(
        registry_root,
        selected_model="combined",
    )

    config = (
        ProductionPredictionConfiguration
        .from_registry(
            registry_root=registry_root,
            snapshot_root=snapshot_root,
        )
    )

    assert config.requested_model == "combined"
    assert config.resolved_model == "combined"
    assert config.fallback_applied is False
    assert config.fallback_reason is None

    assert (
        config.regime_calibration_snapshot_root
        == snapshot_root
        / "regime-calibration"
    )

    assert (
        config.regime_bayesian_snapshot_root
        == snapshot_root
        / "regime-bayesian"
    )


def test_from_registry_preserves_baseline_fallback(
    tmp_path: Path,
) -> None:
    registry_root = (
        tmp_path
        / "registry"
    )

    snapshot_root = (
        tmp_path
        / "snapshots"
    )

    _write_active_decision(
        registry_root,
        selected_model=None,
    )

    config = (
        ProductionPredictionConfiguration
        .from_registry(
            registry_root=registry_root,
            snapshot_root=snapshot_root,
        )
    )

    assert config.requested_model is None
    assert config.resolved_model == "baseline"
    assert config.fallback_applied is True

    assert (
        config.fallback_reason
        == "no_selected_model"
    )

    assert (
        config.regime_calibration_snapshot_root
        is None
    )

    assert (
        config.regime_bayesian_snapshot_root
        is None
    )


def test_from_registry_resolves_calibration_model(
    tmp_path: Path,
) -> None:
    registry_root = (
        tmp_path
        / "registry"
    )

    snapshot_root = (
        tmp_path
        / "snapshots"
    )

    _write_active_decision(
        registry_root,
        selected_model="calibration",
    )

    config = (
        ProductionPredictionConfiguration
        .from_registry(
            registry_root=registry_root,
            snapshot_root=snapshot_root,
        )
    )

    assert config.resolved_model == "calibration"

    assert (
        config.pipeline_kwargs()
        == {
            "regime_calibration_snapshot_root": (
                snapshot_root
                / "regime-calibration"
            ),
            "regime_bayesian_snapshot_root": None,
        }
    )


def test_from_registry_resolves_bayesian_model(
    tmp_path: Path,
) -> None:
    registry_root = (
        tmp_path
        / "registry"
    )

    snapshot_root = (
        tmp_path
        / "snapshots"
    )

    _write_active_decision(
        registry_root,
        selected_model="bayesian",
    )

    config = (
        ProductionPredictionConfiguration
        .from_registry(
            registry_root=registry_root,
            snapshot_root=snapshot_root,
        )
    )

    assert config.resolved_model == "bayesian"

    assert (
        config.pipeline_kwargs()
        == {
            "regime_calibration_snapshot_root": None,
            "regime_bayesian_snapshot_root": (
                snapshot_root
                / "regime-bayesian"
            ),
        }
    )


def test_from_registry_accepts_string_paths(
    tmp_path: Path,
) -> None:
    registry_root = (
        tmp_path
        / "registry"
    )

    snapshot_root = (
        tmp_path
        / "snapshots"
    )

    _write_active_decision(
        registry_root,
        selected_model="baseline",
    )

    config = (
        ProductionPredictionConfiguration
        .from_registry(
            registry_root=str(
                registry_root
            ),
            snapshot_root=str(
                snapshot_root
            ),
        )
    )

    assert config.requested_model == "baseline"
    assert config.resolved_model == "baseline"
    assert config.fallback_applied is False


def test_from_registry_propagates_missing_registry(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
    ):
        (
            ProductionPredictionConfiguration
            .from_registry(
                registry_root=(
                    tmp_path
                    / "missing"
                ),
                snapshot_root=(
                    tmp_path
                    / "snapshots"
                ),
            )
        )


def test_from_registry_propagates_invalid_decision(
    tmp_path: Path,
) -> None:
    registry_root = (
        tmp_path
        / "registry"
    )

    decision_path = (
        registry_root
        / "active"
        / "champion_decision.json"
    )

    decision_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    decision_path.write_text(
        "not-json\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
    ):
        (
            ProductionPredictionConfiguration
            .from_registry(
                registry_root=registry_root,
                snapshot_root=(
                    tmp_path
                    / "snapshots"
                ),
            )
        )


def test_from_registry_returns_same_contract_as_from_decision(
    tmp_path: Path,
) -> None:
    registry_root = (
        tmp_path
        / "registry"
    )

    snapshot_root = (
        tmp_path
        / "snapshots"
    )

    decision_path = _write_active_decision(
        registry_root,
        selected_model="combined",
    )

    registry_config = (
        ProductionPredictionConfiguration
        .from_registry(
            registry_root=registry_root,
            snapshot_root=snapshot_root,
        )
    )

    decision_config = (
        ProductionPredictionConfiguration
        .from_decision(
            decision_path=decision_path,
            snapshot_root=snapshot_root,
        )
    )

    assert registry_config == decision_config
