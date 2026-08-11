"""Public API for the evolution subsystem."""

from __future__ import annotations

from importlib import import_module
from typing import Final


__all__ = [
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
    "SnapshotNotFoundError",
    "SnapshotRepository",
    "SnapshotSerializationError",
]


_EXPORTS: Final[
    dict[str, tuple[str, str]]
] = {
    "AdaptivePolicyConfig": (
        "lrp.evolution.policies",
        "AdaptivePolicyConfig",
    ),
    "AdaptivePolicyDecision": (
        "lrp.evolution.policies",
        "AdaptivePolicyDecision",
    ),
    "AdaptiveWeightCalculator": (
        "lrp.evolution.algorithms",
        "AdaptiveWeightCalculator",
    ),
    "AdaptiveWeightPolicy": (
        "lrp.evolution.policies",
        "AdaptiveWeightPolicy",
    ),
    "AdaptiveWeightProfile": (
        "lrp.evolution.contracts",
        "AdaptiveWeightProfile",
    ),
    "CallableEvolutionPipeline": (
        "lrp.evolution.services.evolution_pipeline",
        "CallableEvolutionPipeline",
    ),
    "EvolutionCoordinator": (
        "lrp.evolution.services.coordinator",
        "EvolutionCoordinator",
    ),
    "EvolutionEngine": (
        "lrp.evolution.services.evolution_engine",
        "EvolutionEngine",
    ),
    "EvolutionPipeline": (
        "lrp.evolution.services.evolution_pipeline",
        "EvolutionPipeline",
    ),
    "EvolutionPipelineRequest": (
        "lrp.evolution.contracts",
        "EvolutionPipelineRequest",
    ),
    "EvolutionRunResult": (
        "lrp.evolution.contracts",
        "EvolutionRunResult",
    ),
    "EvolutionSnapshot": (
        "lrp.evolution.storage",
        "EvolutionSnapshot",
    ),
    "EvolutionSnapshotSerializer": (
        "lrp.evolution.storage",
        "EvolutionSnapshotSerializer",
    ),
    "FileSnapshotRepository": (
        "lrp.evolution.repositories",
        "FileSnapshotRepository",
    ),
    "JsonSnapshotSerializer": (
        "lrp.evolution.serialization",
        "JsonSnapshotSerializer",
    ),
    "LearningContext": (
        "lrp.evolution.contracts.learning_context",
        "LearningContext",
    ),
    "LearningCycle": (
        "lrp.evolution.services.learning_cycle",
        "LearningCycle",
    ),
    "PersistentLearningRunResult": (
        "lrp.evolution.contracts.persistent_learning",
        "PersistentLearningRunResult",
    ),
    "PersistentLearningRunner": (
        "lrp.evolution.services.persistent_learning_runner",
        "PersistentLearningRunner",
    ),
    "PersistentLearningService": (
        "lrp.evolution.services.persistent_learning_service",
        "PersistentLearningService",
    ),
    "RewardFeedback": (
        "lrp.evolution.contracts.reinforcement",
        "RewardFeedback",
    ),
    "SnapshotCodec": (
        "lrp.evolution.serialization",
        "SnapshotCodec",
    ),
    "SnapshotFactory": (
        "lrp.evolution.services.snapshot_factory",
        "SnapshotFactory",
    ),
    "SnapshotNotFoundError": (
        "lrp.evolution.storage",
        "SnapshotNotFoundError",
    ),
    "SnapshotRepository": (
        "lrp.evolution.storage",
        "SnapshotRepository",
    ),
    "SnapshotSerializationError": (
        "lrp.evolution.storage",
        "SnapshotSerializationError",
    ),
}


def __getattr__(name: str) -> object:
    """Resolve public exports only when first requested."""

    target = _EXPORTS.get(name)

    if target is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )

    module_name, attribute_name = target

    module = import_module(module_name)
    value = getattr(module, attribute_name)

    globals()[name] = value

    return value


def __dir__() -> list[str]:
    """Expose lazy public names to introspection."""

    return sorted(
        set(globals()) | set(__all__)
    )
