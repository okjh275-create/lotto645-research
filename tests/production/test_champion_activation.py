from pathlib import Path

from lrp.production import (
    ProductionChampionActivation,
)


def test_selected_baseline_resolves_baseline() -> None:
    result = ProductionChampionActivation.resolve(
        selected_model="baseline",
        snapshot_root=Path("snapshots"),
    )

    assert result.requested_model == "baseline"
    assert result.resolved_model == "baseline"
    assert result.fallback_applied is False
    assert result.fallback_reason is None

    assert result.activation.model == "baseline"


def test_selected_calibration_resolves_calibration() -> None:
    result = ProductionChampionActivation.resolve(
        selected_model="calibration",
        snapshot_root=Path("snapshots"),
    )

    assert result.requested_model == "calibration"
    assert result.resolved_model == "calibration"
    assert result.fallback_applied is False
    assert result.activation.model == "calibration"


def test_selected_bayesian_resolves_bayesian() -> None:
    result = ProductionChampionActivation.resolve(
        selected_model="bayesian",
        snapshot_root=Path("snapshots"),
    )

    assert result.requested_model == "bayesian"
    assert result.resolved_model == "bayesian"
    assert result.fallback_applied is False
    assert result.activation.model == "bayesian"


def test_selected_combined_resolves_combined() -> None:
    result = ProductionChampionActivation.resolve(
        selected_model="combined",
        snapshot_root=Path("snapshots"),
    )

    assert result.requested_model == "combined"
    assert result.resolved_model == "combined"
    assert result.fallback_applied is False
    assert result.activation.model == "combined"


def test_none_selected_model_falls_back_to_baseline() -> None:
    result = ProductionChampionActivation.resolve(
        selected_model=None,
        snapshot_root=Path("snapshots"),
    )

    assert result.requested_model is None
    assert result.resolved_model == "baseline"
    assert result.fallback_applied is True

    assert (
        result.fallback_reason
        == "no_selected_model"
    )

    assert result.activation.model == "baseline"

    assert (
        result.activation.regime_calibration_snapshot_root
        is None
    )

    assert (
        result.activation.regime_bayesian_snapshot_root
        is None
    )


def test_unknown_selected_model_falls_back_to_baseline() -> None:
    result = ProductionChampionActivation.resolve(
        selected_model="experimental",
        snapshot_root=Path("snapshots"),
    )

    assert result.requested_model == "experimental"
    assert result.resolved_model == "baseline"
    assert result.fallback_applied is True

    assert (
        result.fallback_reason
        == "unsupported_selected_model"
    )

    assert result.activation.model == "baseline"


def test_blank_selected_model_falls_back_to_baseline() -> None:
    result = ProductionChampionActivation.resolve(
        selected_model="",
        snapshot_root=Path("snapshots"),
    )

    assert result.requested_model == ""
    assert result.resolved_model == "baseline"
    assert result.fallback_applied is True

    assert (
        result.fallback_reason
        == "unsupported_selected_model"
    )


def test_result_serialization() -> None:
    result = ProductionChampionActivation.resolve(
        selected_model="combined",
        snapshot_root=Path("snapshots"),
    )

    assert result.as_dict() == {
        "requested_model": "combined",
        "resolved_model": "combined",
        "fallback_applied": False,
        "fallback_reason": None,
        "activation": {
            "model": "combined",
            "regime_calibration_snapshot_root": (
                "snapshots/regime-calibration"
            ),
            "regime_bayesian_snapshot_root": (
                "snapshots/regime-bayesian"
            ),
        },
    }


def test_fallback_serialization() -> None:
    result = ProductionChampionActivation.resolve(
        selected_model=None,
        snapshot_root=Path("snapshots"),
    )

    assert result.as_dict() == {
        "requested_model": None,
        "resolved_model": "baseline",
        "fallback_applied": True,
        "fallback_reason": "no_selected_model",
        "activation": {
            "model": "baseline",
            "regime_calibration_snapshot_root": None,
            "regime_bayesian_snapshot_root": None,
        },
    }
