from __future__ import annotations

from pathlib import Path

import pytest

from lrp.evaluation import EvaluationWindow

from tools.validation.historical_replay_models import (
    ReplayConfig,
    ReplayRoundResult,
    summarize_replay,
)

from tools.validation.historical_replay_runner import (
    HistoricalReplayResult,
)

from tools.validation.model_replay_provider import (
    HistoricalModelReplayProvider,
)


def make_row(
    round_no: int,
) -> ReplayRoundResult:
    return ReplayRoundResult(
        round_no=round_no,
        seed=20260802 + round_no,
        history_draws=100,
        noop_best_hits=1,
        adaptive_best_hits=2,
        noop_practical_hits=1,
        adaptive_practical_hits=2,
        noop_avg_jaccard=0.20,
        adaptive_avg_jaccard=0.18,
        probability_l1_delta=0.10,
        probability_max_delta=0.05,
        changed_probability_count=10,
        changed_set_count=3,
        profile_applied=True,
        profile_revision=1,
        profile_sample_size=10,
    )



def make_replay_result(
    *,
    config: ReplayConfig,
    rounds: tuple[ReplayRoundResult, ...],
    output_root: Path,
) -> HistoricalReplayResult:
    return HistoricalReplayResult(
        config=config,
        rounds=rounds,
        summary=summarize_replay(rounds),
        final_state=None,
        rounds_path=(
            output_root
            / "replay_rounds.jsonl"
        ),
        summary_path=(
            output_root
            / "replay_summary.json"
        ),
    )


def test_provider_executes_requested_model_window(
    tmp_path: Path,
) -> None:
    window = EvaluationWindow(
        name="recent",
        start_round=1220,
        end_round=1222,
    )

    calls = []

    def execute(
        *,
        model_name: str,
        config: ReplayConfig,
        output_root: Path,
    ) -> HistoricalReplayResult:
        calls.append(
            (
                model_name,
                config,
                output_root,
            )
        )

        return make_replay_result(
            config=config,
            rounds=(
                make_row(1220),
                make_row(1221),
                make_row(1222),
            ),
            output_root=output_root,
        )

    provider = HistoricalModelReplayProvider(
        execute=execute,
        output_root=tmp_path,
    )

    rows = provider(
        "combined",
        window,
    )

    assert tuple(
        row.round_no
        for row in rows
    ) == (
        1220,
        1221,
        1222,
    )

    assert len(calls) == 1

    model_name, config, output_root = calls[0]

    assert model_name == "combined"
    assert config.start_round == 1220
    assert config.end_round == 1222

    assert output_root == (
        tmp_path
        / "combined"
        / "recent"
    )


def test_provider_preserves_base_replay_parameters(
    tmp_path: Path,
) -> None:
    seen = []

    def execute(
        *,
        model_name: str,
        config: ReplayConfig,
        output_root: Path,
    ) -> HistoricalReplayResult:
        seen.append(config)

        return make_replay_result(
            config=config,
            rounds=(make_row(1220),),
            output_root=output_root,
        )

    base_config = ReplayConfig(
        start_round=2,
        end_round=2,
        seed_base=12345,
        temperature=0.75,
        candidate_count=250,
        top_k=12,
        practical_k=4,
        long_gap_window=8,
        confidence=0.90,
        mode="fast",
    )

    provider = HistoricalModelReplayProvider(
        execute=execute,
        output_root=tmp_path,
        base_config=base_config,
    )

    provider(
        "baseline",
        EvaluationWindow(
            name="single",
            start_round=1220,
            end_round=1220,
        ),
    )

    config = seen[0]

    assert config.start_round == 1220
    assert config.end_round == 1220
    assert config.seed_base == 12345
    assert config.temperature == pytest.approx(0.75)
    assert config.candidate_count == 250
    assert config.top_k == 12
    assert config.practical_k == 4
    assert config.long_gap_window == 8
    assert config.confidence == pytest.approx(0.90)
    assert config.mode == "fast"


def test_provider_rejects_blank_model_name(
    tmp_path: Path,
) -> None:
    provider = HistoricalModelReplayProvider(
        execute=lambda **kwargs: None,
        output_root=tmp_path,
    )

    with pytest.raises(
        ValueError,
        match="model_name",
    ):
        provider(
            " ",
            EvaluationWindow(
                name="recent",
                start_round=1220,
                end_round=1222,
            ),
        )


def test_provider_requires_evaluation_window(
    tmp_path: Path,
) -> None:
    provider = HistoricalModelReplayProvider(
        execute=lambda **kwargs: None,
        output_root=tmp_path,
    )

    with pytest.raises(
        TypeError,
        match="EvaluationWindow",
    ):
        provider(
            "baseline",
            object(),
        )


def test_provider_requires_historical_replay_result(
    tmp_path: Path,
) -> None:
    provider = HistoricalModelReplayProvider(
        execute=lambda **kwargs: object(),
        output_root=tmp_path,
    )

    with pytest.raises(
        TypeError,
        match="HistoricalReplayResult",
    ):
        provider(
            "baseline",
            EvaluationWindow(
                name="recent",
                start_round=1220,
                end_round=1220,
            ),
        )
