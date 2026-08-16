"""Production model activation configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_PRODUCTION_MODELS = (
    "baseline",
    "calibration",
    "bayesian",
    "combined",
)


@dataclass(frozen=True, slots=True)
class ProductionModelActivation:
    """Resolved production configuration for one evaluated model."""

    model: str
    regime_calibration_snapshot_root: Path | None
    regime_bayesian_snapshot_root: Path | None

    @classmethod
    def resolve(
        cls,
        *,
        model: str,
        snapshot_root: str | Path,
    ) -> "ProductionModelActivation":
        if not isinstance(model, str):
            raise TypeError(
                "model must be a string"
            )

        if model not in SUPPORTED_PRODUCTION_MODELS:
            raise ValueError(
                "unsupported production model: "
                f"{model!r}"
            )

        root = Path(snapshot_root)

        calibration_enabled = model in (
            "calibration",
            "combined",
        )

        bayesian_enabled = model in (
            "bayesian",
            "combined",
        )

        return cls(
            model=model,
            regime_calibration_snapshot_root=(
                root / "regime-calibration"
                if calibration_enabled
                else None
            ),
            regime_bayesian_snapshot_root=(
                root / "regime-bayesian"
                if bayesian_enabled
                else None
            ),
        )
