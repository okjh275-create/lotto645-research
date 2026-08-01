from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from lrp.evolution.contracts.models import (
    AdaptiveWeightProfile,
)
from lrp.evolution.integration import (
    NoOpEvolutionWeightAdapter,
    ProviderEvolutionWeightAdapter,
)
from lrp.evolution.storage import (
    SnapshotRepository,
)
from lrp.pipelines.prediction import (
    PredictionPipeline,
)


def make_profile(
    revision: int,
) -> AdaptiveWeightProfile:
    return AdaptiveWeightProfile.default(
        revision=revision,
        generated_at=datetime(
            2026,
            8,
            2,
            tzinfo=timezone.utc,
        ),
    )


def test_pipeline_load_uses_noop_by_default() -> None:
    pipeline = PredictionPipeline.load()

    assert isinstance(
        pipeline.evolution,
        NoOpEvolutionWeightAdapter,
    )


def test_pipeline_load_builds_provider_adapter(
    tmp_path: Path,
) -> None:
    pipeline = PredictionPipeline.load(
        evolution_snapshot_root=tmp_path
    )

    assert isinstance(
        pipeline.evolution,
        ProviderEvolutionWeightAdapter,
    )


def test_pipeline_load_reads_latest_profile(
    tmp_path: Path,
) -> None:
    repository = SnapshotRepository(
        tmp_path
    )
    repository.save(make_profile(1))
    repository.save(make_profile(2))

    pipeline = PredictionPipeline.load(
        evolution_snapshot_root=tmp_path
    )

    assert isinstance(
        pipeline.evolution,
        ProviderEvolutionWeightAdapter,
    )

    profile = (
        pipeline.evolution
        .provider
        .get_profile(
            round_no=1220
        )
    )

    assert profile is not None
    assert profile.revision == 2


def test_explicit_adapter_has_priority(
    tmp_path: Path,
) -> None:
    explicit = NoOpEvolutionWeightAdapter()

    pipeline = PredictionPipeline.load(
        evolution=explicit,
        evolution_snapshot_root=tmp_path,
    )

    assert pipeline.evolution is explicit
