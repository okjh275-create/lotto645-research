from __future__ import annotations

import json
from pathlib import Path

import pytest

import lrp.cli.predict as cli
from lrp.production import (
    ProductionPredictionConfiguration,
)


def _write_registry(
    root: Path,
    *,
    selected_model: object,
) -> Path:
    decision_path = (
        root
        / "active"
        / "champion_decision.json"
    )

    decision_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    decision_path.write_text(
        json.dumps(
            {
                "selection": {
                    "selected_model": selected_model,
                },
            }
        ),
        encoding="utf-8",
    )

    return decision_path


def test_parser_accepts_production_registry() -> None:
    parser = cli._parser()

    args = parser.parse_args(
        [
            "--history",
            "history.json",
            "--round",
            "1232",
            "--seed",
            "20260816",
            "--production-registry",
            "production-registry",
            "--production-snapshot-root",
            "snapshots",
        ]
    )

    assert args.production_registry == Path(
        "production-registry"
    )

    assert args.production_snapshot_root == Path(
        "snapshots"
    )


def test_registry_requires_snapshot_root() -> None:
    with pytest.raises(
        ValueError,
        match="production_snapshot_root",
    ):
        cli._resolve_production_configuration(
            champion_decision=None,
            production_registry=Path(
                "production-registry"
            ),
            production_snapshot_root=None,
        )


def test_snapshot_root_requires_activation_source() -> None:
    with pytest.raises(
        ValueError,
        match="champion_decision.*production_registry|production_registry.*champion_decision",
    ):
        cli._resolve_production_configuration(
            champion_decision=None,
            production_registry=None,
            production_snapshot_root=Path(
                "snapshots"
            ),
        )


def test_decision_and_registry_are_mutually_exclusive() -> None:
    with pytest.raises(
        ValueError,
        match="mutually exclusive|cannot.*both|both.*cannot",
    ):
        cli._resolve_production_configuration(
            champion_decision=Path(
                "champion_decision.json"
            ),
            production_registry=Path(
                "production-registry"
            ),
            production_snapshot_root=Path(
                "snapshots"
            ),
        )


def test_no_options_preserve_default_pipeline() -> None:
    result = cli._resolve_production_configuration(
        champion_decision=None,
        production_registry=None,
        production_snapshot_root=None,
    )

    assert result is None


def test_direct_decision_path_remains_supported(
    tmp_path: Path,
) -> None:
    decision_path = (
        tmp_path
        / "champion_decision.json"
    )

    decision_path.write_text(
        json.dumps(
            {
                "selection": {
                    "selected_model": "calibration",
                },
            }
        ),
        encoding="utf-8",
    )

    result = cli._resolve_production_configuration(
        champion_decision=decision_path,
        production_registry=None,
        production_snapshot_root=(
            tmp_path
            / "snapshots"
        ),
    )

    assert isinstance(
        result,
        ProductionPredictionConfiguration,
    )

    assert result.resolved_model == "calibration"


def test_registry_resolves_configuration(
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

    _write_registry(
        registry_root,
        selected_model="combined",
    )

    result = cli._resolve_production_configuration(
        champion_decision=None,
        production_registry=registry_root,
        production_snapshot_root=snapshot_root,
    )

    assert isinstance(
        result,
        ProductionPredictionConfiguration,
    )

    assert result.requested_model == "combined"
    assert result.resolved_model == "combined"
    assert result.fallback_applied is False

    assert result.pipeline_kwargs() == {
        "regime_calibration_snapshot_root": (
            snapshot_root
            / "regime-calibration"
        ),
        "regime_bayesian_snapshot_root": (
            snapshot_root
            / "regime-bayesian"
        ),
    }


def test_registry_preserves_baseline_fallback(
    tmp_path: Path,
) -> None:
    registry_root = (
        tmp_path
        / "registry"
    )

    _write_registry(
        registry_root,
        selected_model=None,
    )

    result = cli._resolve_production_configuration(
        champion_decision=None,
        production_registry=registry_root,
        production_snapshot_root=(
            tmp_path
            / "snapshots"
        ),
    )

    assert result is not None

    assert result.requested_model is None
    assert result.resolved_model == "baseline"
    assert result.fallback_applied is True
    assert (
        result.fallback_reason
        == "no_selected_model"
    )


def test_registry_propagates_missing_registry(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
    ):
        cli._resolve_production_configuration(
            champion_decision=None,
            production_registry=(
                tmp_path
                / "missing"
            ),
            production_snapshot_root=(
                tmp_path
                / "snapshots"
            ),
        )


def test_existing_direct_decision_signature_contract() -> None:
    config = (
        ProductionPredictionConfiguration
        .from_decision
    )

    assert callable(config)