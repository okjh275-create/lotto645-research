from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.validation.historical_replay_models import (
    ReplayConfig,
    ReplayRoundResult,
)
from tools.validation.historical_replay_runner import (
    HistoricalReplayRunner,
)


def make_row(
    *,
    round_no: int,
    seed: int,
    revision: int,
) -> ReplayRoundResult:
    return ReplayRoundResult(
        round_no=round_no,
        seed=seed,
        history_draws=600,
        noop_best_hits=2,
        adaptive_best_hits=3,
        noop_practical_hits=1,
        adaptive_practical_hits=2,
        noop_avg_jaccard=0.08,
        adaptive_avg_jaccard=0.09,
        probability_l1_delta=0.001,
        probability_max_delta=0.0001,
        changed_probability_count=45,
        changed_set_count=4,
        profile_applied=True,
        profile_revision=revision,
        profile_sample_size=20 * revision,
        elapsed_seconds=1.5,
    )


def test_runner_preserves_round_order_and_state(
    tmp_path: Path,
) -> None:
    calls: list[
        tuple[int, int, object | None]
    ] = []

    def executor(
        round_no: int,
        seed: int,
        draw: object,
        state: object | None,
    ) -> tuple[
        ReplayRoundResult,
        object,
    ]:
        calls.append(
            (
                round_no,
                seed,
                state,
            )
        )

        revision = (
            1
            if state is None
            else int(state) + 1
        )

        return (
            make_row(
                round_no=round_no,
                seed=seed,
                revision=revision,
            ),
            revision,
        )

    runner = HistoricalReplayRunner(
        executor=executor
    )

    config = ReplayConfig(
        start_round=1222,
        end_round=1224,
    )

    result = runner.run(
        config=config,
        draw_by_round={
            1222: object(),
            1223: object(),
            1224: object(),
        },
        output_root=tmp_path,
    )

    assert [
        item[0]
        for item in calls
    ] == [
        1222,
        1223,
        1224,
    ]

    assert calls[0][2] is None
    assert calls[1][2] == 1
    assert calls[2][2] == 2

    assert result.final_state == 3
    assert result.summary.round_count == 3
    assert result.summary.final_profile_revision == 3


def test_runner_writes_jsonl_and_summary(
    tmp_path: Path,
) -> None:
    def executor(
        round_no: int,
        seed: int,
        draw: object,
        state: object | None,
    ):
        return (
            make_row(
                round_no=round_no,
                seed=seed,
                revision=round_no - 1221,
            ),
            round_no - 1221,
        )

    result = HistoricalReplayRunner(
        executor=executor
    ).run(
        config=ReplayConfig(
            start_round=1222,
            end_round=1223,
        ),
        draw_by_round={
            1222: object(),
            1223: object(),
        },
        output_root=tmp_path,
    )

    lines = result.rounds_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 2
    assert json.loads(lines[0])[
        "round_no"
    ] == 1222
    assert json.loads(lines[1])[
        "round_no"
    ] == 1223

    summary = json.loads(
        result.summary_path.read_text(
            encoding="utf-8"
        )
    )

    assert summary["status"] == "PASS"
    assert summary["summary"][
        "round_count"
    ] == 2


def test_runner_refuses_existing_outputs(
    tmp_path: Path,
) -> None:
    output = tmp_path / "replay"
    output.mkdir()
    (
        output / "replay_rounds.jsonl"
    ).write_text(
        "{}\n",
        encoding="utf-8",
    )

    runner = HistoricalReplayRunner(
        executor=lambda *args: (
            make_row(
                round_no=1222,
                seed=20262024,
                revision=1,
            ),
            None,
        )
    )

    with pytest.raises(FileExistsError):
        runner.run(
            config=ReplayConfig(
                start_round=1222,
                end_round=1222,
            ),
            draw_by_round={
                1222: object(),
            },
            output_root=output,
        )


def test_runner_writes_failure_artifact(
    tmp_path: Path,
) -> None:
    def executor(
        round_no: int,
        seed: int,
        draw: object,
        state: object | None,
    ):
        if round_no == 1223:
            raise RuntimeError("boom")

        return (
            make_row(
                round_no=round_no,
                seed=seed,
                revision=1,
            ),
            1,
        )

    with pytest.raises(
        RuntimeError,
        match="boom",
    ):
        HistoricalReplayRunner(
            executor=executor
        ).run(
            config=ReplayConfig(
                start_round=1222,
                end_round=1224,
            ),
            draw_by_round={
                1222: object(),
                1223: object(),
                1224: object(),
            },
            output_root=tmp_path,
        )

    failure = json.loads(
        (
            tmp_path
            / "replay_failure.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert failure["status"] == "ERROR"
    assert failure["completed_rounds"] == [
        1222
    ]

    lines = (
        tmp_path
        / "replay_rounds.jsonl"
    ).read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 1


def test_runner_validates_executor_round(
    tmp_path: Path,
) -> None:
    runner = HistoricalReplayRunner(
        executor=lambda *args: (
            make_row(
                round_no=9999,
                seed=20262024,
                revision=1,
            ),
            None,
        )
    )

    with pytest.raises(
        Exception,
        match="unexpected round",
    ):
        runner.run(
            config=ReplayConfig(
                start_round=1222,
                end_round=1222,
            ),
            draw_by_round={
                1222: object(),
            },
            output_root=tmp_path,
        )
