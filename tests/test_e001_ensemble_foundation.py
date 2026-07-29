"""Regression test for Project E E-001 foundation."""

from __future__ import annotations

from dataclasses import dataclass

from lrp.ensemble import (
    EmptyStrategyWeightRepository,
    EnsembleConfig,
    EnsembleEngine,
    InMemoryStrategyWeightRepository,
    StrategyWeight,
    __version__,
)


@dataclass(frozen=True, slots=True)
class CandidateFixture:
    name: str
    normalized_score: float


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_empty_repository() -> None:
    repository = EmptyStrategyWeightRepository()

    assert repository.load_weights(
        round_no=1220
    ) == ()


def test_strategy_repository() -> None:
    repository = InMemoryStrategyWeightRepository(
        (
            StrategyWeight(
                strategy_type="scenario",
                strategy_name="gap",
                current_weight=0.96,
                normalized_weight=0.45,
                confidence=0.20,
                stability=0.70,
                trend="FLAT",
                sample_count=10,
            ),
            StrategyWeight(
                strategy_type="model",
                strategy_name="GPT-v3.3",
                current_weight=1.08,
                normalized_weight=0.55,
                confidence=0.30,
                stability=0.80,
                trend="UP",
                sample_count=12,
            ),
        )
    )

    weights = repository.load_weights(
        round_no=1220
    )

    assert len(weights) == 2
    assert weights[0].strategy_type == "model"
    assert weights[0].strategy_name == "GPT-v3.3"
    assert weights[1].strategy_type == "scenario"


def test_base_score_ordering() -> None:
    candidates = (
        CandidateFixture("B", 0.72),
        CandidateFixture("A", 0.91),
        CandidateFixture("C", 0.83),
    )

    engine = EnsembleEngine()
    result = engine.evaluate(
        candidates,
        round_no=1220,
    )

    assert result.count == 3
    assert [
        item.source.name
        for item in result.items
    ] == [
        "A",
        "C",
        "B",
    ]

    assert [
        item.ensemble_score
        for item in result.items
    ] == [
        0.91,
        0.83,
        0.72,
    ]

    assert result.strategy_weights == ()
    assert result.engine_version == "0.1.0"


def test_weighted_foundation() -> None:
    repository = InMemoryStrategyWeightRepository(
        (
            StrategyWeight(
                strategy_type="model",
                strategy_name="GPT-v3.3",
                current_weight=1.05,
                normalized_weight=0.60,
                confidence=0.40,
                stability=0.80,
                trend="UP",
                sample_count=20,
            ),
            StrategyWeight(
                strategy_type="model",
                strategy_name="Gemini-v7.1",
                current_weight=0.95,
                normalized_weight=0.40,
                confidence=0.20,
                stability=0.60,
                trend="FLAT",
                sample_count=20,
            ),
        )
    )

    engine = EnsembleEngine(
        repository=repository
    )

    result = engine.evaluate(
        (
            CandidateFixture("A", 0.90),
            CandidateFixture("B", 0.80),
            CandidateFixture("C", 0.70),
        ),
        round_no=1220,
        config=EnsembleConfig(
            base_score_weight=0.70,
            adaptive_weight=0.10,
            confidence_weight=0.05,
            stability_weight=0.10,
            trend_weight=0.05,
            top_k=2,
        ),
    )

    assert result.count == 2
    assert len(result.strategy_weights) == 2

    assert [
        item.source.name
        for item in result.items
    ] == [
        "A",
        "B",
    ]

    assert result.items[0].ensemble_score > (
        result.items[1].ensemble_score
    )

    payload = result.to_dict()

    assert payload["round_no"] == 1220
    assert payload["count"] == 2
    assert payload["engine_version"] == "0.1.0"
    assert len(payload["strategy_weights"]) == 2
    assert len(payload["items"]) == 2


def main() -> None:
    test_version()
    test_empty_repository()
    test_strategy_repository()
    test_base_score_ordering()
    test_weighted_foundation()

    print("PASS: Project E E-001 ensemble foundation")
    print("version:", __version__)
    print("repository_contract: PASS")
    print("base_score_ordering: PASS")
    print("weighted_foundation: PASS")


if __name__ == "__main__":
    main()
