from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from lrp.evolution import (
    AdaptivePolicyConfig,
    AdaptiveWeightPolicy,
    AdaptiveWeightProfile,
    CallableEvolutionPipeline,
    EvolutionCoordinator,
    EvolutionEngine,
    EvolutionPipelineRequest,
    EvolutionRunResult,
    SnapshotRepository,
)


FIXED_TIME = datetime(
    2026,
    7,
    31,
    14,
    0,
    0,
    tzinfo=timezone.utc,
)


def make_profile(
    *,
    revision: int,
    confidence: float = 0.80,
    sample_size: int = 40,
) -> AdaptiveWeightProfile:
    return AdaptiveWeightProfile(
        hot_weight=0.35,
        cold_weight=0.15,
        gap_weight=0.15,
        trend_weight=0.15,
        transition_weight=0.10,
        learning_weight=0.05,
        adaptive_weight=0.05,
        confidence=confidence,
        sample_size=sample_size,
        revision=revision,
        generated_at=FIXED_TIME,
    )


def make_request(
    *,
    revision: int = 1,
    confidence: float = 0.80,
    sample_size: int = 40,
    **changes: Any,
) -> EvolutionPipelineRequest:
    values: dict[str, Any] = {
        "signals": {
            "hot": 0.30,
            "gap": 0.20,
        },
        "confidence": confidence,
        "sample_size": sample_size,
        "revision": revision,
        "generated_at": FIXED_TIME,
        "previous_profile": None,
    }
    values.update(changes)

    return EvolutionPipelineRequest(**values)


def make_engine(
    tmp_path: Path,
    calculator,
    *,
    config: AdaptivePolicyConfig | None = None,
) -> tuple[
    EvolutionEngine,
    SnapshotRepository,
]:
    pipeline = CallableEvolutionPipeline(
        calculator
    )
    policy = AdaptiveWeightPolicy(config)
    repository = SnapshotRepository(tmp_path)

    coordinator = EvolutionCoordinator(
        pipeline=pipeline,
        policy=policy,
        repository=repository,
    )

    return (
        EvolutionEngine(coordinator),
        repository,
    )


def test_first_run_is_applied_and_persisted(
    tmp_path: Path,
) -> None:
    received: list[EvolutionPipelineRequest] = []

    def calculator(
        request: EvolutionPipelineRequest,
    ) -> AdaptiveWeightProfile:
        received.append(request)
        return make_profile(
            revision=request.revision,
            confidence=request.confidence,
            sample_size=request.sample_size,
        )

    engine, repository = make_engine(
        tmp_path,
        calculator,
    )

    result = engine.run(
        make_request(revision=1)
    )

    assert isinstance(result, EvolutionRunResult)
    assert result.applied is True
    assert result.persisted is True
    assert result.revision == 1
    assert result.previous_revision is None
    assert result.snapshot is not None
    assert repository.load_latest().revision == 1
    assert received[0].previous_profile is None


def test_latest_profile_is_injected_into_pipeline(
    tmp_path: Path,
) -> None:
    previous = make_profile(revision=1)
    repository = SnapshotRepository(tmp_path)
    repository.save(
        previous,
        saved_at=FIXED_TIME,
    )

    received: list[EvolutionPipelineRequest] = []

    def calculator(
        request: EvolutionPipelineRequest,
    ) -> AdaptiveWeightProfile:
        received.append(request)
        return make_profile(revision=2)

    coordinator = EvolutionCoordinator(
        pipeline=CallableEvolutionPipeline(
            calculator
        ),
        policy=AdaptiveWeightPolicy(),
        repository=repository,
    )
    engine = EvolutionEngine(coordinator)

    supplied_previous = make_profile(revision=99)

    result = engine.run(
        make_request(
            revision=2,
            previous_profile=supplied_previous,
        )
    )

    assert received[0].previous_profile == previous
    assert result.previous_profile == previous
    assert result.previous_revision == 1


def test_second_run_creates_next_snapshot(
    tmp_path: Path,
) -> None:
    def calculator(
        request: EvolutionPipelineRequest,
    ) -> AdaptiveWeightProfile:
        return make_profile(
            revision=request.revision
        )

    engine, repository = make_engine(
        tmp_path,
        calculator,
    )

    first = engine.run(make_request(revision=1))
    second = engine.run(make_request(revision=2))

    assert first.revision == 1
    assert second.revision == 2
    assert second.previous_revision == 1
    assert repository.revisions() == (1, 2)


def test_rejected_candidate_is_not_persisted(
    tmp_path: Path,
) -> None:
    def calculator(
        request: EvolutionPipelineRequest,
    ) -> AdaptiveWeightProfile:
        return make_profile(
            revision=request.revision,
            confidence=0.20,
            sample_size=40,
        )

    engine, repository = make_engine(
        tmp_path,
        calculator,
    )

    result = engine.run(
        make_request(revision=1)
    )

    assert result.rejected is True
    assert result.persisted is False
    assert result.snapshot is None
    assert "confidence_below_threshold" in (
        result.reasons
    )
    assert repository.revisions() == ()


def test_rejected_update_keeps_previous_snapshot(
    tmp_path: Path,
) -> None:
    repository = SnapshotRepository(tmp_path)
    previous = make_profile(revision=1)

    repository.save(
        previous,
        saved_at=FIXED_TIME,
    )

    def calculator(
        request: EvolutionPipelineRequest,
    ) -> AdaptiveWeightProfile:
        return make_profile(
            revision=2,
            confidence=0.20,
        )

    coordinator = EvolutionCoordinator(
        pipeline=CallableEvolutionPipeline(
            calculator
        ),
        policy=AdaptiveWeightPolicy(),
        repository=repository,
    )
    engine = EvolutionEngine(coordinator)

    result = engine.run(
        make_request(revision=2)
    )

    assert result.rejected is True
    assert result.profile == previous
    assert result.previous_profile == previous
    assert repository.revisions() == (1,)


def test_fail_open_candidate_is_persisted(
    tmp_path: Path,
) -> None:
    def calculator(
        request: EvolutionPipelineRequest,
    ) -> AdaptiveWeightProfile:
        return make_profile(
            revision=request.revision,
            confidence=0.20,
            sample_size=1,
        )

    engine, repository = make_engine(
        tmp_path,
        calculator,
        config=AdaptivePolicyConfig(
            fail_open=True
        ),
    )

    result = engine.run(
        make_request(revision=1)
    )

    assert result.applied is True
    assert result.persisted is True
    assert "fail_open_applied" in result.reasons
    assert repository.revisions() == (1,)


def test_candidate_revision_must_match_request(
    tmp_path: Path,
) -> None:
    def calculator(
        request: EvolutionPipelineRequest,
    ) -> AdaptiveWeightProfile:
        return make_profile(revision=999)

    engine, repository = make_engine(
        tmp_path,
        calculator,
    )

    with pytest.raises(
        ValueError,
        match="candidate revision must match",
    ):
        engine.run(make_request(revision=1))

    assert repository.revisions() == ()


def test_duplicate_revision_is_rejected_by_policy(
    tmp_path: Path,
) -> None:
    repository = SnapshotRepository(tmp_path)
    repository.save(
        make_profile(revision=1),
        saved_at=FIXED_TIME,
    )

    def calculator(
        request: EvolutionPipelineRequest,
    ) -> AdaptiveWeightProfile:
        return make_profile(revision=1)

    coordinator = EvolutionCoordinator(
        pipeline=CallableEvolutionPipeline(
            calculator
        ),
        policy=AdaptiveWeightPolicy(),
        repository=repository,
    )
    engine = EvolutionEngine(coordinator)

    result = engine.run(
        make_request(revision=1)
    )

    assert result.rejected is True
    assert "revision_not_newer" in result.reasons
    assert repository.revisions() == (1,)


def test_engine_exposes_coordinator(
    tmp_path: Path,
) -> None:
    pipeline = CallableEvolutionPipeline(
        lambda request: make_profile(
            revision=request.revision
        )
    )
    policy = AdaptiveWeightPolicy()
    repository = SnapshotRepository(tmp_path)

    coordinator = EvolutionCoordinator(
        pipeline=pipeline,
        policy=policy,
        repository=repository,
    )

    engine = EvolutionEngine(coordinator)

    assert engine.coordinator is coordinator
    assert coordinator.pipeline is pipeline
    assert coordinator.policy is policy
    assert coordinator.repository is repository


def test_engine_validates_request_type(
    tmp_path: Path,
) -> None:
    engine, _ = make_engine(
        tmp_path,
        lambda request: make_profile(
            revision=request.revision
        ),
    )

    with pytest.raises(
        TypeError,
        match="EvolutionPipelineRequest",
    ):
        engine.run({})  # type: ignore[arg-type]


def test_coordinator_validates_pipeline_type(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        TypeError,
        match="EvolutionPipeline",
    ):
        EvolutionCoordinator(
            pipeline=object(),  # type: ignore[arg-type]
            policy=AdaptiveWeightPolicy(),
            repository=SnapshotRepository(
                tmp_path
            ),
        )


def test_engine_validates_coordinator_type() -> None:
    with pytest.raises(
        TypeError,
        match="EvolutionCoordinator",
    ):
        EvolutionEngine(  # type: ignore[arg-type]
            object()
        )