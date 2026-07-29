"""Regression tests for Project E E-004 pipeline integration."""

from __future__ import annotations

from dataclasses import dataclass

from lrp.ensemble import (
    CandidateRescorer,
    LearningSnapshot,
    PipelineRescoringBridge,
    RescoringConfig,
    StrategyRankingSnapshot,
    StrategyStatisticSnapshot,
    StrategyWeight,
    recursive_base_score_reader,
    recursive_strategy_resolver,
    replace_candidate_score,
)


@dataclass(frozen=True, slots=True)
class CandidateFixture:
    numbers: tuple[int, ...]
    model_name: str


@dataclass(frozen=True, slots=True)
class ScoredFixture:
    candidate: CandidateFixture
    normalized_score: float


@dataclass(frozen=True, slots=True)
class RankedFixture:
    scored: ScoredFixture
    rank: int


class SnapshotRepositoryFixture:
    def __init__(
        self,
        snapshot: LearningSnapshot,
    ) -> None:
        self.snapshot = snapshot
        self.calls: list[int] = []

    def load_snapshot(
        self,
        *,
        round_no: int,
    ) -> LearningSnapshot:
        self.calls.append(round_no)
        return self.snapshot


class FailingSnapshotRepository:
    def load_snapshot(
        self,
        *,
        round_no: int,
    ) -> LearningSnapshot:
        raise RuntimeError(
            f"snapshot unavailable: {round_no}"
        )


def build_snapshot() -> LearningSnapshot:
    return LearningSnapshot(
        round_no=1220,
        revision=(9, 9),
        statistics=(
            StrategyStatisticSnapshot(
                strategy_type="model",
                strategy_name="strong",
                sample_count=30,
                average_match_count=5.0,
                average_prediction_score=0.95,
                hit3_plus_rate=1.0,
                prize_rate=1.0,
                updated_at_kst=(
                    "2026-07-27T20:00:00+09:00"
                ),
            ),
            StrategyStatisticSnapshot(
                strategy_type="model",
                strategy_name="weak",
                sample_count=30,
                average_match_count=1.0,
                average_prediction_score=0.30,
                hit3_plus_rate=0.0,
                prize_rate=0.0,
                updated_at_kst=(
                    "2026-07-27T20:00:00+09:00"
                ),
            ),
        ),
        rankings=(
            StrategyRankingSnapshot(
                strategy_type="model",
                strategy_name="strong",
                rank_position=1,
                rank_score=0.98,
                confidence=0.95,
                stability=0.95,
                trend="UP",
                recent_gain=0.20,
                sample_count=30,
                average_match_count=5.0,
                average_prediction_score=0.95,
                prize_rate=1.0,
            ),
            StrategyRankingSnapshot(
                strategy_type="model",
                strategy_name="weak",
                rank_position=2,
                rank_score=0.10,
                confidence=0.20,
                stability=0.30,
                trend="DOWN",
                recent_gain=-0.20,
                sample_count=30,
                average_match_count=1.0,
                average_prediction_score=0.30,
                prize_rate=0.0,
            ),
        ),
        strategy_weights=(
            StrategyWeight(
                strategy_type="model",
                strategy_name="strong",
                current_weight=1.20,
                normalized_weight=0.90,
                confidence=0.95,
                stability=0.95,
                trend="UP",
                sample_count=30,
            ),
            StrategyWeight(
                strategy_type="model",
                strategy_name="weak",
                current_weight=0.80,
                normalized_weight=0.10,
                confidence=0.20,
                stability=0.30,
                trend="DOWN",
                sample_count=30,
            ),
        ),
    )


def build_candidates() -> tuple[
    ScoredFixture,
    ...,
]:
    return (
        ScoredFixture(
            candidate=CandidateFixture(
                numbers=(1, 7, 13, 24, 31, 42),
                model_name="weak",
            ),
            normalized_score=0.92,
        ),
        ScoredFixture(
            candidate=CandidateFixture(
                numbers=(2, 8, 16, 25, 34, 43),
                model_name="strong",
            ),
            normalized_score=0.86,
        ),
    )


def build_bridge(
    repository: object,
    *,
    fail_open: bool = True,
) -> PipelineRescoringBridge:
    return PipelineRescoringBridge(
        snapshot_repository=repository,
        rescorer=CandidateRescorer(
            config=RescoringConfig(
                base_score_weight=0.55,
                adaptive_weight=0.12,
                ranking_weight=0.10,
                confidence_weight=0.05,
                stability_weight=0.05,
                trend_weight=0.04,
                performance_weight=0.07,
                sample_weight=0.02,
            ),
            score_reader=(
                recursive_base_score_reader
            ),
            strategy_resolver=(
                recursive_strategy_resolver
            ),
        ),
        fail_open=fail_open,
    )


def test_recursive_contract_readers() -> None:
    source = RankedFixture(
        scored=ScoredFixture(
            candidate=CandidateFixture(
                numbers=(
                    1,
                    9,
                    17,
                    25,
                    33,
                    41,
                ),
                model_name="strong",
            ),
            normalized_score=0.81,
        ),
        rank=1,
    )

    assert (
        recursive_base_score_reader(source)
        == 0.81
    )

    assert (
        recursive_strategy_resolver(source)
        == (
            ("model", "strong"),
        )
    )


def test_immutable_score_replacement() -> None:
    source = build_candidates()[0]

    replaced = replace_candidate_score(
        source,
        0.44,
    )

    assert isinstance(
        replaced,
        ScoredFixture,
    )
    assert replaced is not source
    assert replaced.candidate is source.candidate
    assert replaced.normalized_score == 0.44
    assert source.normalized_score == 0.92


def test_pipeline_rescoring_applied() -> None:
    repository = SnapshotRepositoryFixture(
        build_snapshot()
    )

    bridge = build_bridge(repository)
    candidates = build_candidates()

    result = bridge.apply(
        candidates,
        round_no=1220,
    )

    assert result.enabled is True
    assert result.applied is True
    assert result.fallback_reason is None
    assert result.candidate_count == 2
    assert repository.calls == [1220]

    assert result.rescoring is not None
    assert (
        result.rescoring
        .metadata[
            "evidence_candidate_count"
        ]
        == 2
    )

    effective = result.effective_candidates

    assert all(
        isinstance(
            item,
            ScoredFixture,
        )
        for item in effective
    )

    # Strong candidate must move ahead after learning evidence.
    assert (
        effective[0].candidate.model_name
        == "strong"
    )
    assert (
        effective[0].normalized_score
        > effective[1].normalized_score
    )

    # Original Project D-style objects remain unchanged.
    assert candidates[0].normalized_score == 0.92
    assert candidates[1].normalized_score == 0.86

    payload = result.to_dict()

    assert payload["enabled"] is True
    assert payload["applied"] is True
    assert payload["candidate_count"] == 2
    assert payload[
        "evidence_candidate_count"
    ] == 2
    assert payload["rescoring"] is not None


def test_disabled_preserves_candidates() -> None:
    candidates = build_candidates()

    bridge = PipelineRescoringBridge(
        snapshot_repository=(
            SnapshotRepositoryFixture(
                build_snapshot()
            )
        ),
        enabled=False,
    )

    result = bridge.apply(
        candidates,
        round_no=1220,
    )

    assert result.enabled is False
    assert result.applied is False
    assert (
        result.fallback_reason
        == "disabled"
    )
    assert (
        result.effective_candidates
        == candidates
    )
    assert result.rescoring is None


def test_unconfigured_preserves_candidates() -> None:
    candidates = build_candidates()

    result = PipelineRescoringBridge().apply(
        candidates,
        round_no=1220,
    )

    assert result.enabled is True
    assert result.applied is False
    assert (
        result.fallback_reason
        == "snapshot_repository_unconfigured"
    )
    assert (
        result.effective_candidates
        == candidates
    )


def test_fail_open_preserves_candidates() -> None:
    candidates = build_candidates()

    bridge = build_bridge(
        FailingSnapshotRepository(),
        fail_open=True,
    )

    result = bridge.apply(
        candidates,
        round_no=1220,
    )

    assert result.applied is False
    assert (
        result.effective_candidates
        == candidates
    )
    assert result.rescoring is None
    assert result.fallback_reason is not None
    assert result.fallback_reason.startswith(
        "RuntimeError:"
    )


def test_deterministic_ordering() -> None:
    repository = SnapshotRepositoryFixture(
        build_snapshot()
    )
    bridge = build_bridge(repository)
    candidates = build_candidates()

    first = bridge.apply(
        candidates,
        round_no=1220,
    )
    second = bridge.apply(
        candidates,
        round_no=1220,
    )

    first_rows = [
        (
            item.candidate.model_name,
            item.normalized_score,
        )
        for item in first.effective_candidates
    ]

    second_rows = [
        (
            item.candidate.model_name,
            item.normalized_score,
        )
        for item in second.effective_candidates
    ]

    assert first_rows == second_rows


def main() -> None:
    test_recursive_contract_readers()
    test_immutable_score_replacement()
    test_pipeline_rescoring_applied()
    test_disabled_preserves_candidates()
    test_unconfigured_preserves_candidates()
    test_fail_open_preserves_candidates()
    test_deterministic_ordering()

    print(
        "PASS: Project E E-004 "
        "pipeline integration"
    )
    print("recursive_contract_readers: PASS")
    print("immutable_score_replacement: PASS")
    print("pipeline_rescoring: PASS")
    print("disabled_fallback: PASS")
    print("unconfigured_fallback: PASS")
    print("fail_open: PASS")
    print("deterministic_ordering: PASS")


if __name__ == "__main__":
    main()
