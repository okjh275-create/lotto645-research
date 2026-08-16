"""Safe production activation from a selected champion model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lrp.production.model_activation import (
    ProductionModelActivation,
    SUPPORTED_PRODUCTION_MODELS,
)


BASELINE_MODEL = "baseline"


@dataclass(frozen=True, slots=True)
class ProductionChampionActivation:
    """Resolved production activation with fail-safe fallback."""

    requested_model: str | None
    resolved_model: str
    fallback_applied: bool
    fallback_reason: str | None
    activation: ProductionModelActivation

    @classmethod
    def resolve(
        cls,
        *,
        selected_model: str | None,
        snapshot_root: str | Path,
    ) -> "ProductionChampionActivation":
        if selected_model is None:
            resolved_model = BASELINE_MODEL
            fallback_applied = True
            fallback_reason = "no_selected_model"

        elif selected_model not in SUPPORTED_PRODUCTION_MODELS:
            resolved_model = BASELINE_MODEL
            fallback_applied = True
            fallback_reason = (
                "unsupported_selected_model"
            )

        else:
            resolved_model = selected_model
            fallback_applied = False
            fallback_reason = None

        activation = ProductionModelActivation.resolve(
            model=resolved_model,
            snapshot_root=snapshot_root,
        )

        return cls(
            requested_model=selected_model,
            resolved_model=resolved_model,
            fallback_applied=fallback_applied,
            fallback_reason=fallback_reason,
            activation=activation,
        )

    def as_dict(self) -> dict[str, object]:
        calibration_root = (
            self.activation
            .regime_calibration_snapshot_root
        )

        bayesian_root = (
            self.activation
            .regime_bayesian_snapshot_root
        )

        return {
            "requested_model": self.requested_model,
            "resolved_model": self.resolved_model,
            "fallback_applied": (
                self.fallback_applied
            ),
            "fallback_reason": (
                self.fallback_reason
            ),
            "activation": {
                "model": self.activation.model,
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
            },
        }
