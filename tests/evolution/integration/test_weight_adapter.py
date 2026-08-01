from __future__ import annotations

from typing import cast

from lrp.evolution.integration import (
    EvolutionWeightAdapter,
    NoOpEvolutionWeightAdapter,
)


def test_noop_matches_adapter_protocol() -> None:
    adapter = cast(
        EvolutionWeightAdapter[object],
        NoOpEvolutionWeightAdapter(),
    )

    value = object()

    result = adapter.adjust(
        value,
        round_no=1220,
        seed=20260802,
    )

    assert result is value


def test_runtime_protocol_check() -> None:
    assert isinstance(
        NoOpEvolutionWeightAdapter(),
        EvolutionWeightAdapter,
    )


def test_integration_all_exports() -> None:
    import lrp.evolution.integration as integration

    assert integration.__all__ == [
        "AdaptiveEvolutionWeightAdapter",
        "AdaptiveWeightProfileProvider",
        "EvolutionAdapterFactory",
        "EvolutionWeightAdapter",
        "NoOpEvolutionWeightAdapter",
        "PredictionRewardMapper",
        "ProviderEvolutionWeightAdapter",
        "ReviewSignalExtractor",
        "SnapshotProfileProvider",
        "StaticProfileProvider",
    ]
