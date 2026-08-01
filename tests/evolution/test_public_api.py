"""Root public API tests for lrp.evolution."""

from __future__ import annotations

import lrp.evolution as evolution

from lrp.evolution import (
    FileSnapshotRepository,
    JsonSnapshotSerializer,
    LearningContext,
    LearningCycle,
    PersistentLearningRunResult,
    PersistentLearningRunner,
    PersistentLearningService,
    RewardFeedback,
    SnapshotCodec,
    SnapshotFactory,
)


NEW_PUBLIC_EXPORTS = {
    "FileSnapshotRepository",
    "JsonSnapshotSerializer",
    "LearningContext",
    "LearningCycle",
    "PersistentLearningRunResult",
    "PersistentLearningRunner",
    "PersistentLearningService",
    "RewardFeedback",
    "SnapshotCodec",
    "SnapshotFactory",
}


EXISTING_PUBLIC_EXPORTS = {
    "AdaptivePolicyConfig",
    "AdaptivePolicyDecision",
    "AdaptiveWeightCalculator",
    "AdaptiveWeightPolicy",
    "AdaptiveWeightProfile",
    "CallableEvolutionPipeline",
    "EvolutionCoordinator",
    "EvolutionEngine",
    "EvolutionPipeline",
    "EvolutionPipelineRequest",
    "EvolutionRunResult",
    "EvolutionSnapshot",
    "EvolutionSnapshotSerializer",
    "SnapshotNotFoundError",
    "SnapshotRepository",
    "SnapshotSerializationError",
}


def test_new_root_public_imports() -> None:
    assert FileSnapshotRepository.__name__ == (
        "FileSnapshotRepository"
    )
    assert JsonSnapshotSerializer.__name__ == (
        "JsonSnapshotSerializer"
    )
    assert LearningContext.__name__ == (
        "LearningContext"
    )
    assert LearningCycle.__name__ == (
        "LearningCycle"
    )
    assert PersistentLearningRunResult.__name__ == (
        "PersistentLearningRunResult"
    )
    assert PersistentLearningRunner.__name__ == (
        "PersistentLearningRunner"
    )
    assert PersistentLearningService.__name__ == (
        "PersistentLearningService"
    )
    assert RewardFeedback.__name__ == (
        "RewardFeedback"
    )
    assert SnapshotCodec.__name__ == (
        "SnapshotCodec"
    )
    assert SnapshotFactory.__name__ == (
        "SnapshotFactory"
    )


def test_root_module_exposes_new_api() -> None:
    for name in NEW_PUBLIC_EXPORTS:
        assert hasattr(evolution, name)


def test_root_all_contains_new_api() -> None:
    assert NEW_PUBLIC_EXPORTS.issubset(
        set(evolution.__all__)
    )


def test_existing_root_api_is_preserved() -> None:
    for name in EXISTING_PUBLIC_EXPORTS:
        assert hasattr(evolution, name)


def test_root_all_contains_existing_api() -> None:
    assert EXISTING_PUBLIC_EXPORTS.issubset(
        set(evolution.__all__)
    )


def test_root_all_has_no_duplicates() -> None:
    assert len(evolution.__all__) == len(
        set(evolution.__all__)
    )


def test_root_all_is_sorted() -> None:
    assert evolution.__all__ == sorted(
        evolution.__all__
    )
