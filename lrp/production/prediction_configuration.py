"""Production prediction configuration resolved from champion decisions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lrp.production.champion_activation import (
    ProductionChampionActivation,
)
from lrp.production.champion_decision_reader import (
    ProductionChampionDecisionReader,
)


@dataclass(frozen=True, slots=True)
class ProductionPredictionConfiguration:
    """Pipeline configuration derived from a production champion decision."""

    requested_model: str | None
    resolved_model: str
    fallback_applied: bool
    fallback_reason: str | None
    regime_calibration_snapshot_root: Path | None
    regime_bayesian_snapshot_root: Path | None

    @classmethod
    def from_decision(
        cls,
        *,
        decision_path: str | Path,
        snapshot_root: str | Path,
    ) -> "ProductionPredictionConfiguration":
        decision = (
            ProductionChampionDecisionReader()
            .read(decision_path)
        )

        resolved = (
            ProductionChampionActivation.resolve(
                selected_model=(
                    decision.selected_model
                ),
                snapshot_root=snapshot_root,
            )
        )

        activation = resolved.activation

        return cls(
            requested_model=(
                resolved.requested_model
            ),
            resolved_model=(
                resolved.resolved_model
            ),
            fallback_applied=(
                resolved.fallback_applied
            ),
            fallback_reason=(
                resolved.fallback_reason
            ),
            regime_calibration_snapshot_root=(
                activation
                .regime_calibration_snapshot_root
            ),
            regime_bayesian_snapshot_root=(
                activation
                .regime_bayesian_snapshot_root
            ),
        )

    @classmethod
    def from_registry(
        cls,
        *,
        registry_root: str | Path,
        snapshot_root: str | Path,
    ) -> "ProductionPredictionConfiguration":
        from lrp.production.champion_registry import (
            ProductionChampionRegistry,
        )

        registry = ProductionChampionRegistry(
            registry_root
        )

        return cls.from_decision(
            decision_path=registry.decision_path(),
            snapshot_root=snapshot_root,
        )


    def pipeline_kwargs(
        self,
    ) -> dict[str, Path | None]:
        return {
            "regime_calibration_snapshot_root": (
                self.regime_calibration_snapshot_root
            ),
            "regime_bayesian_snapshot_root": (
                self.regime_bayesian_snapshot_root
            ),
        }

    def as_dict(
        self,
    ) -> dict[str, object]:
        calibration_root = (
            self.regime_calibration_snapshot_root
        )

        bayesian_root = (
            self.regime_bayesian_snapshot_root
        )

        return {
            "requested_model": (
                self.requested_model
            ),
            "resolved_model": (
                self.resolved_model
            ),
            "fallback_applied": (
                self.fallback_applied
            ),
            "fallback_reason": (
                self.fallback_reason
            ),
            "regime_calibration_snapshot_root": (
                calibration_root.as_posix()
                if calibration_root is not None
                else None
            ),
            "regime_bayesian_snapshot_root": (
                bayesian_root.as_posix()
                if bayesian_root is not None
                else None
            ),
        }
