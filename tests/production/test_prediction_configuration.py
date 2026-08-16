import json
from pathlib import Path

from lrp.production import (
    ProductionPredictionConfiguration,
)


def _write_decision(
    path: Path,
    selected_model: str | None,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            {
                "selection": {
                    "selected_model": (
                        selected_model
                    ),
                },
            }
        ),
        encoding="utf-8",
    )


def test_baseline_configuration_has_no_regime_roots(
    tmp_path: Path,
) -> None:
    decision_path = (
        tmp_path
        / "champion_decision.json"
    )

    _write_decision(
        decision_path,
        "baseline",
    )

    config = (
        ProductionPredictionConfiguration
        .from_decision(
            decision_path=decision_path,
            snapshot_root=tmp_path / "snapshots",
        )
    )

    assert config.requested_model == "baseline"
    assert config.resolved_model == "baseline"

    assert (
        config.regime_calibration_snapshot_root
        is None
    )

    assert (
        config.regime_bayesian_snapshot_root
        is None
    )


def test_calibration_configuration_sets_only_calibration_root(
    tmp_path: Path,
) -> None:
    decision_path = (
        tmp_path
        / "champion_decision.json"
    )

    snapshot_root = (
        tmp_path
        / "snapshots"
    )

    _write_decision(
        decision_path,
        "calibration",
    )

    config = (
        ProductionPredictionConfiguration
        .from_decision(
            decision_path=decision_path,
            snapshot_root=snapshot_root,
        )
    )

    assert config.resolved_model == "calibration"

    assert (
        config.regime_calibration_snapshot_root
        == snapshot_root / "regime-calibration"
    )

    assert (
        config.regime_bayesian_snapshot_root
        is None
    )


def test_bayesian_configuration_sets_only_bayesian_root(
    tmp_path: Path,
) -> None:
    decision_path = (
        tmp_path
        / "champion_decision.json"
    )

    snapshot_root = (
        tmp_path
        / "snapshots"
    )

    _write_decision(
        decision_path,
        "bayesian",
    )

    config = (
        ProductionPredictionConfiguration
        .from_decision(
            decision_path=decision_path,
            snapshot_root=snapshot_root,
        )
    )

    assert config.resolved_model == "bayesian"

    assert (
        config.regime_calibration_snapshot_root
        is None
    )

    assert (
        config.regime_bayesian_snapshot_root
        == snapshot_root / "regime-bayesian"
    )


def test_combined_configuration_sets_both_roots(
    tmp_path: Path,
) -> None:
    decision_path = (
        tmp_path
        / "champion_decision.json"
    )

    snapshot_root = (
        tmp_path
        / "snapshots"
    )

    _write_decision(
        decision_path,
        "combined",
    )

    config = (
        ProductionPredictionConfiguration
        .from_decision(
            decision_path=decision_path,
            snapshot_root=snapshot_root,
        )
    )

    assert config.resolved_model == "combined"

    assert (
        config.regime_calibration_snapshot_root
        == snapshot_root / "regime-calibration"
    )

    assert (
        config.regime_bayesian_snapshot_root
        == snapshot_root / "regime-bayesian"
    )


def test_none_selected_model_falls_back_to_baseline(
    tmp_path: Path,
) -> None:
    decision_path = (
        tmp_path
        / "champion_decision.json"
    )

    _write_decision(
        decision_path,
        None,
    )

    config = (
        ProductionPredictionConfiguration
        .from_decision(
            decision_path=decision_path,
            snapshot_root=tmp_path / "snapshots",
        )
    )

    assert config.requested_model is None
    assert config.resolved_model == "baseline"
    assert config.fallback_applied is True

    assert (
        config.fallback_reason
        == "no_selected_model"
    )


def test_unknown_model_falls_back_to_baseline(
    tmp_path: Path,
) -> None:
    decision_path = (
        tmp_path
        / "champion_decision.json"
    )

    _write_decision(
        decision_path,
        "experimental",
    )

    config = (
        ProductionPredictionConfiguration
        .from_decision(
            decision_path=decision_path,
            snapshot_root=tmp_path / "snapshots",
        )
    )

    assert config.resolved_model == "baseline"
    assert config.fallback_applied is True

    assert (
        config.fallback_reason
        == "unsupported_selected_model"
    )


def test_pipeline_kwargs_contains_only_regime_activation(
    tmp_path: Path,
) -> None:
    decision_path = (
        tmp_path
        / "champion_decision.json"
    )

    snapshot_root = (
        tmp_path
        / "snapshots"
    )

    _write_decision(
        decision_path,
        "combined",
    )

    config = (
        ProductionPredictionConfiguration
        .from_decision(
            decision_path=decision_path,
            snapshot_root=snapshot_root,
        )
    )

    assert config.pipeline_kwargs() == {
        "regime_calibration_snapshot_root": (
            snapshot_root
            / "regime-calibration"
        ),
        "regime_bayesian_snapshot_root": (
            snapshot_root
            / "regime-bayesian"
        ),
    }


def test_baseline_pipeline_kwargs_are_explicit_none(
    tmp_path: Path,
) -> None:
    decision_path = (
        tmp_path
        / "champion_decision.json"
    )

    _write_decision(
        decision_path,
        "baseline",
    )

    config = (
        ProductionPredictionConfiguration
        .from_decision(
            decision_path=decision_path,
            snapshot_root=tmp_path / "snapshots",
        )
    )

    assert config.pipeline_kwargs() == {
        "regime_calibration_snapshot_root": None,
        "regime_bayesian_snapshot_root": None,
    }


def test_serialization_reports_activation_provenance(
    tmp_path: Path,
) -> None:
    decision_path = (
        tmp_path
        / "champion_decision.json"
    )

    _write_decision(
        decision_path,
        None,
    )

    config = (
        ProductionPredictionConfiguration
        .from_decision(
            decision_path=decision_path,
            snapshot_root="snapshots",
        )
    )

    assert config.as_dict() == {
        "requested_model": None,
        "resolved_model": "baseline",
        "fallback_applied": True,
        "fallback_reason": "no_selected_model",
        "regime_calibration_snapshot_root": None,
        "regime_bayesian_snapshot_root": None,
    }
