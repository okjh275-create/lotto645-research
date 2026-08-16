from pathlib import Path

import pytest

from lrp.production import (
    ProductionModelActivation,
)


def test_baseline_disables_regime_learning() -> None:
    activation = ProductionModelActivation.resolve(
        model="baseline",
        snapshot_root=Path("snapshots"),
    )

    assert activation.model == "baseline"
    assert activation.regime_calibration_snapshot_root is None
    assert activation.regime_bayesian_snapshot_root is None


def test_calibration_enables_only_calibration() -> None:
    root = Path("snapshots")

    activation = ProductionModelActivation.resolve(
        model="calibration",
        snapshot_root=root,
    )

    assert activation.model == "calibration"

    assert (
        activation.regime_calibration_snapshot_root
        == root / "regime-calibration"
    )

    assert activation.regime_bayesian_snapshot_root is None


def test_bayesian_enables_only_bayesian() -> None:
    root = Path("snapshots")

    activation = ProductionModelActivation.resolve(
        model="bayesian",
        snapshot_root=root,
    )

    assert activation.model == "bayesian"

    assert activation.regime_calibration_snapshot_root is None

    assert (
        activation.regime_bayesian_snapshot_root
        == root / "regime-bayesian"
    )


def test_combined_enables_both_regime_models() -> None:
    root = Path("snapshots")

    activation = ProductionModelActivation.resolve(
        model="combined",
        snapshot_root=root,
    )

    assert activation.model == "combined"

    assert (
        activation.regime_calibration_snapshot_root
        == root / "regime-calibration"
    )

    assert (
        activation.regime_bayesian_snapshot_root
        == root / "regime-bayesian"
    )


@pytest.mark.parametrize(
    "model",
    (
        "",
        "unknown",
        "BASELINE",
        " baseline",
        "baseline ",
    ),
)
def test_invalid_model_is_rejected(
    model: str,
) -> None:
    with pytest.raises(ValueError):
        ProductionModelActivation.resolve(
            model=model,
            snapshot_root=Path("snapshots"),
        )


def test_snapshot_root_is_normalized_to_path() -> None:
    activation = ProductionModelActivation.resolve(
        model="combined",
        snapshot_root="snapshots",
    )

    assert (
        activation.regime_calibration_snapshot_root
        == Path("snapshots") / "regime-calibration"
    )

    assert (
        activation.regime_bayesian_snapshot_root
        == Path("snapshots") / "regime-bayesian"
    )
