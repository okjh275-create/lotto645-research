from __future__ import annotations

from pathlib import Path

from lrp.operations import (
    RoundCompletionRepository,
    summarize_round_completions,
    write_operation_artifact,
)


def write_round(
    root: Path,
    *,
    round_no: int,
    feedback_count: int,
    applied: bool,
) -> None:
    payload = {
        "round_no": round_no,
        "learning": {
            "snapshot_id": f"review-{round_no}",
            "feedback_count": feedback_count,
            "final_context_version": 2,
        },
        "profile": {
            "applied": applied,
            "revision": 1 if applied else None,
            "snapshot_saved": applied,
            "reasons": [],
        },
    }

    write_operation_artifact(
        payload,
        output_root=root,
        artifact_type="round-completion",
        round_no=round_no,
        filename="round_completion.json",
    )


def test_repository_lists_and_loads_rounds(
    tmp_path: Path,
) -> None:
    write_round(
        tmp_path,
        round_no=1231,
        feedback_count=2,
        applied=True,
    )
    write_round(
        tmp_path,
        round_no=1232,
        feedback_count=4,
        applied=False,
    )

    repository = RoundCompletionRepository(
        tmp_path / "round-completion"
    )

    assert repository.list_rounds() == (
        1231,
        1232,
    )
    assert repository.latest()["round_no"] == 1232
    assert repository.load_round(1231)["round_no"] == 1231


def test_repository_recent_is_latest_first(
    tmp_path: Path,
) -> None:
    for round_no in (1230, 1231, 1232):
        write_round(
            tmp_path,
            round_no=round_no,
            feedback_count=2,
            applied=True,
        )

    repository = RoundCompletionRepository(
        tmp_path / "round-completion"
    )

    recent = repository.recent(2)

    assert tuple(
        item["round_no"] for item in recent
    ) == (1232, 1231)


def test_repository_verifies_manifest(
    tmp_path: Path,
) -> None:
    write_round(
        tmp_path,
        round_no=1232,
        feedback_count=2,
        applied=True,
    )

    repository = RoundCompletionRepository(
        tmp_path / "round-completion"
    )

    verification = repository.verify_round(1232)

    assert verification["status"] == "PASS"
    assert verification["round"] == 1232


def test_summary_aggregates_recent_rounds(
    tmp_path: Path,
) -> None:
    write_round(
        tmp_path,
        round_no=1231,
        feedback_count=2,
        applied=True,
    )
    write_round(
        tmp_path,
        round_no=1232,
        feedback_count=4,
        applied=False,
    )

    repository = RoundCompletionRepository(
        tmp_path / "round-completion"
    )

    summary = summarize_round_completions(
        repository,
        limit=20,
    )

    assert summary.completion_count == 2
    assert summary.latest_round == 1232
    assert summary.average_feedback_count == 3.0
    assert summary.profile_apply_rate == 0.5
    assert summary.manifest_pass_rate == 1.0
    assert summary.latest_snapshot_id == "review-1232"


def test_empty_repository_summary(
    tmp_path: Path,
) -> None:
    repository = RoundCompletionRepository(
        tmp_path / "round-completion"
    )

    summary = summarize_round_completions(repository)

    assert summary.completion_count == 0
    assert summary.latest_round is None
    assert summary.average_feedback_count == 0.0
    assert summary.profile_apply_rate == 0.0
    assert summary.manifest_pass_rate == 0.0
    assert summary.latest_snapshot_id is None
