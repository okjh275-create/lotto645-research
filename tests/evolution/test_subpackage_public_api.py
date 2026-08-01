"""Public API tests for evolution subpackages."""

from __future__ import annotations

from lrp.evolution.repositories import (
    FileSnapshotRepository,
    SnapshotRepository,
)
from lrp.evolution.serialization import (
    JsonSnapshotSerializer,
    SnapshotCodec,
)
from lrp.evolution.services import (
    AdaptiveEvolutionPipeline,
    CallableEvolutionPipeline,
    EvolutionCoordinator,
    EvolutionEngine,
    EvolutionPipeline,
    LearningCycle,
    PersistentLearningRunner,
    PersistentLearningService,
    SnapshotFactory,
)


def test_repository_public_api() -> None:
    assert FileSnapshotRepository.__name__ == (
        "FileSnapshotRepository"
    )
    assert SnapshotRepository.__name__ == (
        "SnapshotRepository"
    )


def test_serialization_public_api() -> None:
    assert JsonSnapshotSerializer.__name__ == (
        "JsonSnapshotSerializer"
    )
    assert SnapshotCodec.__name__ == (
        "SnapshotCodec"
    )


def test_existing_service_public_api_is_preserved() -> None:
    assert AdaptiveEvolutionPipeline.__name__ == (
        "AdaptiveEvolutionPipeline"
    )
    assert CallableEvolutionPipeline.__name__ == (
        "CallableEvolutionPipeline"
    )
    assert EvolutionCoordinator.__name__ == (
        "EvolutionCoordinator"
    )
    assert EvolutionEngine.__name__ == (
        "EvolutionEngine"
    )
    assert EvolutionPipeline.__name__ == (
        "EvolutionPipeline"
    )


def test_learning_service_public_api() -> None:
    assert LearningCycle.__name__ == (
        "LearningCycle"
    )
    assert PersistentLearningRunner.__name__ == (
        "PersistentLearningRunner"
    )
    assert PersistentLearningService.__name__ == (
        "PersistentLearningService"
    )
    assert SnapshotFactory.__name__ == (
        "SnapshotFactory"
    )


def test_repository_all_exports() -> None:
    import lrp.evolution.repositories as repositories

    assert repositories.__all__ == [
        "FileSnapshotRepository",
        "SnapshotRepository",
    ]


def test_serialization_all_exports() -> None:
    import lrp.evolution.serialization as serialization

    assert serialization.__all__ == [
        "JsonSnapshotSerializer",
        "SnapshotCodec",
    ]


def test_services_all_exports() -> None:
    import lrp.evolution.services as services

    assert services.__all__ == [
        "AdaptiveEvolutionPipeline",
        "CallableEvolutionPipeline",
        "EvolutionCoordinator",
        "EvolutionEngine",
        "EvolutionPipeline",
        "LearningCycle",
        "PersistentLearningRunner",
        "PersistentLearningService",
        "ReviewLearningService",
        "SnapshotFactory",
    ]
