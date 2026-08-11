from __future__ import annotations

import pytest

from lrp.contracts import ContractError
from tools.validation.historical_replay_models import (
    ReplayConfig,
    ReplayRoundResult,
    summarize_replay,
    validate_round_coverage,
)


def make_row(
    *,
    round_no: int,
    noop_hits: int,
    adaptive_hits: int,
    revision: int,
) -> ReplayRoundResult:
    return ReplayRoundResult(
        round_no=round_no,
        seed=20260802 + round_no,
        history_draws=600,
        noop_best_hits=noop_hits,
        adaptive_best_hits=adaptive_hits,
        noop_practical_hits=noop_hits,
        adaptive_practical_hits=adaptive_hits,
        noop_avg_jaccard=0.08,
        adaptive_avg_jaccard=0.09,
        probability_l1_delta=0.001,
        probability_max_delta=0.0001,
        changed_probability_count=45,
        changed_set_count=3,
        profile_applied=True,
        profile_revision=revision,
        profile_sample_size=20 * revision,
        elapsed_seconds=2.5,
    )


def test_replay_config_builds_rounds_and_seeds() -> None:
    config = ReplayConfig(
        start_round=1222,
        end_round=1231,
    )

    assert config.rounds == tuple(
        range(1222, 1232)
    )
    assert config.seed_for_round(
        1222
    ) == 20262024


def test_replay_config_rejects_invalid_range() -> None:
    with pytest.raises(
        ContractError,
        match="end_round",
    ):
        ReplayConfig(
            start_round=1231,
            end_round=1222,
        )


def test_round_result_exposes_deltas() -> None:
    row = make_row(
        round_no=1222,
        noop_hits=2,
        adaptive_hits=3,
        revision=1,
    )

    assert row.best_hit_delta == 1
    assert row.practical_hit_delta == 1

    payload = row.as_dict()

    assert payload["best_hit_delta"] == 1
    assert payload[
        "practical_hit_delta"
    ] == 1


def test_summarize_replay() -> None:
    rows = (
        make_row(
            round_no=1222,
            noop_hits=2,
            adaptive_hits=3,
            revision=1,
        ),
        make_row(
            round_no=1223,
            noop_hits=3,
            adaptive_hits=2,
            revision=2,
        ),
        make_row(
            round_no=1224,
            noop_hits=2,
            adaptive_hits=2,
            revision=3,
        ),
    )

    summary = summarize_replay(rows)

    assert summary.round_count == 3
    assert summary.adaptive_win_count == 1
    assert summary.noop_win_count == 1
    assert summary.tie_count == 1
    assert summary.final_profile_revision == 3
    assert summary.final_profile_sample_size == 60
    assert summary.profile_applied_count == 3
    assert summary.total_elapsed_seconds == 7.5


def test_validate_round_coverage() -> None:
    config = ReplayConfig(
        start_round=1222,
        end_round=1224,
    )

    validate_round_coverage(
        config=config,
        draw_by_round={
            1222: object(),
            1223: object(),
            1224: object(),
        },
    )

    with pytest.raises(
        ContractError,
        match="1224",
    ):
        validate_round_coverage(
            config=config,
            draw_by_round={
                1222: object(),
                1223: object(),
            },
        )


def test_replay_summary_exposes_regime_learning_provenance() -> None:
    rows = (
        ReplayRoundResult(
            round_no=1222,
            seed=20262024,
            history_draws=600,
            noop_best_hits=2,
            adaptive_best_hits=3,
            noop_practical_hits=2,
            adaptive_practical_hits=3,
            noop_avg_jaccard=0.08,
            adaptive_avg_jaccard=0.09,
            probability_l1_delta=0.001,
            probability_max_delta=0.0001,
            changed_probability_count=45,
            changed_set_count=3,
            profile_applied=True,
            profile_revision=1,
            profile_sample_size=20,
            regime_calibration_revision=None,
            regime_calibration_sample_size=None,
            regime_bayesian_revision=None,
            regime_bayesian_sample_size=None,
            elapsed_seconds=2.5,
        ),
        ReplayRoundResult(
            round_no=1223,
            seed=20262025,
            history_draws=601,
            noop_best_hits=3,
            adaptive_best_hits=2,
            noop_practical_hits=3,
            adaptive_practical_hits=2,
            noop_avg_jaccard=0.08,
            adaptive_avg_jaccard=0.09,
            probability_l1_delta=0.001,
            probability_max_delta=0.0001,
            changed_probability_count=45,
            changed_set_count=3,
            profile_applied=True,
            profile_revision=2,
            profile_sample_size=40,
            regime_calibration_revision=1,
            regime_calibration_sample_size=10,
            regime_bayesian_revision=1,
            regime_bayesian_sample_size=10,
            elapsed_seconds=2.5,
        ),
        ReplayRoundResult(
            round_no=1224,
            seed=20262026,
            history_draws=602,
            noop_best_hits=2,
            adaptive_best_hits=2,
            noop_practical_hits=2,
            adaptive_practical_hits=2,
            noop_avg_jaccard=0.08,
            adaptive_avg_jaccard=0.09,
            probability_l1_delta=0.001,
            probability_max_delta=0.0001,
            changed_probability_count=45,
            changed_set_count=3,
            profile_applied=True,
            profile_revision=3,
            profile_sample_size=60,
            regime_calibration_revision=2,
            regime_calibration_sample_size=20,
            regime_bayesian_revision=2,
            regime_bayesian_sample_size=20,
            elapsed_seconds=2.5,
        ),
    )

    summary = summarize_replay(rows)

    assert summary.regime_calibration_applied_count == 2
    assert summary.final_regime_calibration_revision == 2
    assert summary.final_regime_calibration_sample_size == 20

    assert summary.regime_bayesian_applied_count == 2
    assert summary.final_regime_bayesian_revision == 2
    assert summary.final_regime_bayesian_sample_size == 20
