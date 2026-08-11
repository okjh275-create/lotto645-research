from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.validation.historical_replay_models import (
    ReplayRoundResult,
)
from tools.validation.replay_effectiveness import (
    evaluate_effectiveness,
    exact_two_sided_sign_test,
    load_replay_rows,
    write_effectiveness_report,
)


def make_row(
    *,
    round_no: int,
    noop_best: int = 2,
    adaptive_best: int = 3,
    noop_practical: int = 1,
    adaptive_practical: int = 2,
    revision: int = 1,
    regime_calibration_revision: int | None = None,
    regime_calibration_sample_size: int | None = None,
    regime_bayesian_revision: int | None = None,
    regime_bayesian_sample_size: int | None = None,
) -> ReplayRoundResult:
    return ReplayRoundResult(
        round_no=round_no,
        seed=20260802 + round_no,
        history_draws=600,
        noop_best_hits=noop_best,
        adaptive_best_hits=adaptive_best,
        noop_practical_hits=noop_practical,
        adaptive_practical_hits=(
            adaptive_practical
        ),
        noop_avg_jaccard=0.08,
        adaptive_avg_jaccard=0.09,
        probability_l1_delta=0.002,
        probability_max_delta=0.0002,
        changed_probability_count=45,
        changed_set_count=5,
        profile_applied=True,
        profile_revision=revision,
        profile_sample_size=20 * revision,
        regime_calibration_revision=(
            regime_calibration_revision
        ),
        regime_calibration_sample_size=(
            regime_calibration_sample_size
        ),
        regime_bayesian_revision=(
            regime_bayesian_revision
        ),
        regime_bayesian_sample_size=(
            regime_bayesian_sample_size
        ),
        elapsed_seconds=2.0,
    )


def test_exact_sign_test() -> None:
    assert exact_two_sided_sign_test(
        positive=5,
        negative=0,
    ) == pytest.approx(0.0625)

    assert exact_two_sided_sign_test(
        positive=0,
        negative=0,
    ) == 1.0

    assert exact_two_sided_sign_test(
        positive=3,
        negative=3,
    ) == 1.0


def test_evaluate_effectiveness() -> None:
    rows = (
        make_row(
            round_no=1222,
            noop_best=2,
            adaptive_best=3,
            noop_practical=1,
            adaptive_practical=2,
            revision=1,
        ),
        make_row(
            round_no=1223,
            noop_best=3,
            adaptive_best=2,
            noop_practical=2,
            adaptive_practical=2,
            revision=2,
        ),
        make_row(
            round_no=1224,
            noop_best=2,
            adaptive_best=2,
            noop_practical=1,
            adaptive_practical=2,
            revision=3,
        ),
    )

    summary = evaluate_effectiveness(rows)

    assert summary.round_count == 3
    assert summary.best_adaptive_wins == 1
    assert summary.best_noop_wins == 1
    assert summary.best_ties == 1
    assert summary.practical_adaptive_wins == 2
    assert summary.practical_noop_wins == 0
    assert summary.practical_ties == 1
    assert summary.best_hit_mean_delta == 0
    assert summary.practical_hit_mean_delta == (
        pytest.approx(2 / 3)
    )
    assert summary.final_profile_revision == 3
    assert summary.final_profile_sample_size == 60
    assert summary.changed_portfolio_round_count == 3


def test_load_replay_rows(
    tmp_path: Path,
) -> None:
    row = make_row(
        round_no=1222,
        noop_best=2,
        adaptive_best=3,
        noop_practical=1,
        adaptive_practical=2,
        revision=1,
    )

    path = tmp_path / "replay.jsonl"
    path.write_text(
        json.dumps(
            row.as_dict()
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_replay_rows(path)

    assert loaded == (row,)


def test_write_effectiveness_report(
    tmp_path: Path,
) -> None:
    summary = evaluate_effectiveness(
        (
            make_row(
                round_no=1222,
                noop_best=2,
                adaptive_best=3,
                noop_practical=1,
                adaptive_practical=2,
                revision=1,
            ),
        )
    )

    output = write_effectiveness_report(
        summary=summary,
        output=(
            tmp_path
            / "effectiveness.json"
        ),
    )

    payload = json.loads(
        output.read_text(
            encoding="utf-8"
        )
    )

    assert payload["status"] == "PASS"
    assert payload["summary"][
        "round_count"
    ] == 1
    assert payload["interpretation"][
        "best_hit_direction"
    ] == "adaptive_better"


def test_duplicate_rounds_are_rejected() -> None:
    row = make_row(
        round_no=1222,
        noop_best=2,
        adaptive_best=3,
        noop_practical=1,
        adaptive_practical=2,
        revision=1,
    )

    with pytest.raises(
        Exception,
        match="duplicate",
    ):
        evaluate_effectiveness(
            (row, row)
        )


def test_load_replay_rows_preserves_regime_learning_provenance(
    tmp_path: Path,
) -> None:
    row = ReplayRoundResult(
        round_no=1223,
        seed=20262025,
        history_draws=600,
        noop_best_hits=2,
        adaptive_best_hits=3,
        noop_practical_hits=1,
        adaptive_practical_hits=2,
        noop_avg_jaccard=0.08,
        adaptive_avg_jaccard=0.09,
        probability_l1_delta=0.002,
        probability_max_delta=0.0002,
        changed_probability_count=45,
        changed_set_count=5,
        profile_applied=True,
        profile_revision=2,
        profile_sample_size=40,
        regime_calibration_revision=1,
        regime_calibration_sample_size=10,
        regime_bayesian_revision=1,
        regime_bayesian_sample_size=10,
        elapsed_seconds=2.0,
    )

    path = tmp_path / "replay.jsonl"
    path.write_text(
        json.dumps(row.as_dict()) + "\n",
        encoding="utf-8",
    )

    loaded = load_replay_rows(path)

    assert loaded == (row,)

    restored = loaded[0]

    assert restored.regime_calibration_revision == 1
    assert restored.regime_calibration_sample_size == 10
    assert restored.regime_bayesian_revision == 1
    assert restored.regime_bayesian_sample_size == 10


def test_effectiveness_summary_exposes_regime_learning_provenance() -> None:
    rows = (
        make_row(
            round_no=1222,
            regime_calibration_revision=None,
            regime_calibration_sample_size=None,
            regime_bayesian_revision=None,
            regime_bayesian_sample_size=None,
        ),
        make_row(
            round_no=1223,
            regime_calibration_revision=1,
            regime_calibration_sample_size=10,
            regime_bayesian_revision=1,
            regime_bayesian_sample_size=10,
        ),
        make_row(
            round_no=1224,
            regime_calibration_revision=2,
            regime_calibration_sample_size=20,
            regime_bayesian_revision=2,
            regime_bayesian_sample_size=20,
        ),
    )

    summary = evaluate_effectiveness(rows)

    assert summary.regime_calibration_applied_count == 2
    assert summary.final_regime_calibration_revision == 2
    assert summary.final_regime_calibration_sample_size == 20

    assert summary.regime_bayesian_applied_count == 2
    assert summary.final_regime_bayesian_revision == 2
    assert summary.final_regime_bayesian_sample_size == 20
