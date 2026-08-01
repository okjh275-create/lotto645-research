from __future__ import annotations

from pathlib import Path

from lrp.evolution.integration.noop_weight_adapter import (
    NoOpEvolutionWeightAdapter,
)
from lrp.evolution.integration.profile_provider import (
    SnapshotProfileProvider,
)
from lrp.evolution.integration.provider_weight_adapter import (
    ProviderEvolutionWeightAdapter,
)
from lrp.evolution.integration.weight_adapter import (
    EvolutionWeightAdapter,
)
from lrp.evolution.storage import (
    SnapshotRepository,
)


class EvolutionAdapterFactory:
    """Assemble prediction-time evolution adapters."""

    @classmethod
    def build(
        cls,
        *,
        evolution: (
            EvolutionWeightAdapter[object]
            | None
        ) = None,
        snapshot_root: (
            str | Path | None
        ) = None,
    ) -> EvolutionWeightAdapter[object]:
        if evolution is not None:
            cls._validate_adapter(evolution)
            return evolution

        if snapshot_root is None:
            return NoOpEvolutionWeightAdapter()

        normalized_root = cls._normalize_root(
            snapshot_root
        )
        repository = SnapshotRepository(
            normalized_root
        )
        provider = SnapshotProfileProvider(
            repository
        )

        return ProviderEvolutionWeightAdapter(
            provider
        )

    @staticmethod
    def _validate_adapter(
        evolution: EvolutionWeightAdapter[object],
    ) -> None:
        if not isinstance(
            evolution,
            EvolutionWeightAdapter,
        ):
            raise TypeError(
                "evolution must implement "
                "EvolutionWeightAdapter"
            )

    @staticmethod
    def _normalize_root(
        snapshot_root: str | Path,
    ) -> Path:
        if not isinstance(
            snapshot_root,
            (str, Path),
        ):
            raise TypeError(
                "snapshot_root must be a "
                "string or Path"
            )

        if isinstance(snapshot_root, str):
            normalized = snapshot_root.strip()

            if not normalized:
                raise ValueError(
                    "snapshot_root must not be empty"
                )

            path = Path(normalized)
        else:
            path = snapshot_root

        return path.expanduser().resolve()
