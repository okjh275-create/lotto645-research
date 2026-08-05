from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.validation.run_adaptive_automation import (
    build_parser,
    run,
)


def write_profile(path: Path) -> None:
    payload = {
        "hot_weight": 0.30,
        "cold_weight": 0.17,
        "gap_weight": 0.17,
        "trend_weight": 0.14,
        "transition_weight": 0.12,
        "learning_weight": 0.05,
        "adaptive_weight": 0.05,
        "confidence": 0.80,
        "sample_size": 300,
        "revision": 12,
        "generated_at": (
            "2026-08-04T00:00:00+00:00"
        ),
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def write_report(path: Path) -> None:
    trends = {
        field: {
            "values": [0.05, 0.05],
            "first": 0.05,
            "last": 0.05,
            "net_change": 0.0,
            "mean": 0.05,
            "direction": "stable",
            "increase_steps": 0,
            "decrease_steps": 0,
            "stable_steps": 1,
        }
        for field in (
            "hot_weight",
            "cold_weight",
            "gap_weight",
            "trend_weight",
            "transition_weight",
            "learning_weight",
            "adaptive_weight",
        )
    }

    payload = {
        "policies": {
            "floor": {
                "window_count": 3,
                "total_round_count": 300,
                "best_hit_mean_delta": 0.04,
                "practical_hit_mean_delta": 0.03,
                "average_probability_l1_delta": 0.05,
                "average_changed_set_count": 15.0,
            }
        },
        "weight_trends": {
            "policies": {
                "floor": {
                    "weights": trends,
                }
            }
        },
        "significance": {
            "policies": {
                "floor": {
                    "best": {
                        "adaptive_wins": 30,
                        "noop_wins": 20,
                        "ties": 250,
                        "direction": (
                            "adaptive_better"
                        ),
                        "p_value": 0.40,
                        "significant": False,
                    },
                    "practical": {
                        "adaptive_wins": 32,
                        "noop_wins": 18,
                        "ties": 250,
                        "direction": (
                            "adaptive_better"
                        ),
                        "p_value": 0.40,
                        "significant": False,
                    },
                }
            }
        },
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_parser_requires_inputs() -> None:
    args = build_parser().parse_args(
        [
            "--report",
            "report.json",
            "--profile",
            "profile.json",
            "--repository",
            "repository",
            "--policy",
            "floor",
            "--recommendation-id",
            "auto-1",
        ]
    )

    assert args.policy == "floor"
    assert args.recommendation_id == (
        "auto-1"
    )


def test_run_creates_repository_records(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "report.json"
    profile_path = tmp_path / "profile.json"
    repository = tmp_path / "repository"

    write_report(report_path)
    write_profile(profile_path)

    exit_code = run(
        [
            "--report",
            str(report_path),
            "--profile",
            str(profile_path),
            "--repository",
            str(repository),
            "--policy",
            "floor",
            "--recommendation-id",
            "auto-13",
            "--approve",
            "--created-at-utc",
            "2026-08-05T00:00:00+00:00",
        ]
    )

    payload = json.loads(
        capsys.readouterr().out
    )

    assert exit_code == 0
    assert payload["status"] == "PASS"
    assert payload["approved"] is True
    assert payload["source_revision"] == 12
    assert payload["target_revision"] == 13
    assert payload[
        "repository_revision_before"
    ] is None
    assert payload[
        "repository_revision_after"
    ] == 13
    assert payload[
        "automation_created"
    ] is True
    assert payload[
        "profile_created"
    ] is True


def test_missing_report_exits_with_error(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "profile.json"
    write_profile(profile_path)

    with pytest.raises(
        SystemExit,
    ) as error:
        run(
            [
                "--report",
                str(tmp_path / "missing.json"),
                "--profile",
                str(profile_path),
                "--repository",
                str(tmp_path / "repository"),
                "--policy",
                "floor",
                "--recommendation-id",
                "auto-1",
            ]
        )

    assert error.value.code == 2


def test_naive_created_at_exits_with_error(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "report.json"
    profile_path = tmp_path / "profile.json"

    write_report(report_path)
    write_profile(profile_path)

    with pytest.raises(
        SystemExit,
    ) as error:
        run(
            [
                "--report",
                str(report_path),
                "--profile",
                str(profile_path),
                "--repository",
                str(tmp_path / "repository"),
                "--policy",
                "floor",
                "--recommendation-id",
                "auto-1",
                "--created-at-utc",
                "2026-08-05T00:00:00",
            ]
        )

    assert error.value.code == 2


def test_existing_repository_can_be_required(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "report.json"
    profile_path = tmp_path / "profile.json"

    write_report(report_path)
    write_profile(profile_path)

    with pytest.raises(
        SystemExit,
    ) as error:
        run(
            [
                "--report",
                str(report_path),
                "--profile",
                str(profile_path),
                "--repository",
                str(tmp_path / "repository"),
                "--policy",
                "floor",
                "--recommendation-id",
                "auto-1",
                "--require-existing-repository",
                "--approve",
            ]
        )

    assert error.value.code == 2


def test_parser_supports_dry_run() -> None:
    args = build_parser().parse_args(
        [
            "--report",
            "report.json",
            "--profile",
            "profile.json",
            "--repository",
            "repository",
            "--policy",
            "floor",
            "--recommendation-id",
            "dry-1",
            "--dry-run",
        ]
    )

    assert args.dry_run is True


def test_dry_run_does_not_create_repository(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "report.json"
    profile_path = tmp_path / "profile.json"
    repository = tmp_path / "repository"

    write_report(report_path)
    write_profile(profile_path)

    exit_code = run(
        [
            "--report",
            str(report_path),
            "--profile",
            str(profile_path),
            "--repository",
            str(repository),
            "--policy",
            "floor",
            "--recommendation-id",
            "dry-13",
            "--created-at-utc",
            "2026-08-05T00:00:00+00:00",
            "--dry-run",
        ]
    )

    payload = json.loads(
        capsys.readouterr().out
    )

    assert exit_code == 0
    assert payload["status"] == "PASS"
    assert payload["mode"] == "dry_run"
    assert payload["automation_created"] is False
    assert payload["profile_created"] is False
    assert payload["automation_path"] is None
    assert payload["profile_path"] is None
    assert payload["repository_exists"] is False
    assert not repository.exists()
    assert len(payload["decisions"]) == 7
    assert payload["planned_profile"][
        "revision"
    ] == 13


def test_persisted_run_reports_mode(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "report.json"
    profile_path = tmp_path / "profile.json"
    repository = tmp_path / "repository"

    write_report(report_path)
    write_profile(profile_path)

    assert run(
        [
            "--report",
            str(report_path),
            "--profile",
            str(profile_path),
            "--repository",
            str(repository),
            "--policy",
            "floor",
            "--recommendation-id",
            "persisted-13",
            "--approve",
        ]
    ) == 0

    payload = json.loads(
        capsys.readouterr().out
    )

    assert payload["mode"] == "persisted"
    assert payload["automation_created"] is True
    assert payload["profile_created"] is True
    assert payload["repository_exists"] is True
    assert repository.is_dir()


def test_dry_run_ignores_existing_repository_requirement(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "report.json"
    profile_path = tmp_path / "profile.json"
    repository = tmp_path / "missing-repository"

    write_report(report_path)
    write_profile(profile_path)

    assert run(
        [
            "--report",
            str(report_path),
            "--profile",
            str(profile_path),
            "--repository",
            str(repository),
            "--policy",
            "floor",
            "--recommendation-id",
            "dry-required",
            "--require-existing-repository",
            "--dry-run",
        ]
    ) == 0

    payload = json.loads(
        capsys.readouterr().out
    )

    assert payload["mode"] == "dry_run"
    assert repository.exists() is False


def test_parser_supports_approve() -> None:
    args = build_parser().parse_args(
        [
            "--report",
            "report.json",
            "--profile",
            "profile.json",
            "--repository",
            "repository",
            "--policy",
            "floor",
            "--recommendation-id",
            "approved-1",
            "--approve",
        ]
    )

    assert args.approve is True


def test_persisted_run_requires_approval(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "report.json"
    profile_path = tmp_path / "profile.json"
    repository = tmp_path / "repository"

    write_report(report_path)
    write_profile(profile_path)

    with pytest.raises(
        SystemExit,
    ) as error:
        run(
            [
                "--report",
                str(report_path),
                "--profile",
                str(profile_path),
                "--repository",
                str(repository),
                "--policy",
                "floor",
                "--recommendation-id",
                "not-approved",
            ]
        )

    assert error.value.code == 2
    assert not repository.exists()


def test_approved_run_persists_records(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "report.json"
    profile_path = tmp_path / "profile.json"
    repository = tmp_path / "repository"

    write_report(report_path)
    write_profile(profile_path)

    assert run(
        [
            "--report",
            str(report_path),
            "--profile",
            str(profile_path),
            "--repository",
            str(repository),
            "--policy",
            "floor",
            "--recommendation-id",
            "approved-13",
            "--approve",
        ]
    ) == 0

    payload = json.loads(
        capsys.readouterr().out
    )

    assert payload["mode"] == "persisted"
    assert payload["automation_created"] is True
    assert payload["profile_created"] is True
    assert repository.is_dir()


def test_dry_run_and_approve_are_mutually_exclusive(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "report.json"
    profile_path = tmp_path / "profile.json"
    repository = tmp_path / "repository"

    write_report(report_path)
    write_profile(profile_path)

    with pytest.raises(
        SystemExit,
    ) as error:
        run(
            [
                "--report",
                str(report_path),
                "--profile",
                str(profile_path),
                "--repository",
                str(repository),
                "--policy",
                "floor",
                "--recommendation-id",
                "invalid-mode",
                "--dry-run",
                "--approve",
            ]
        )

    assert error.value.code == 2
    assert not repository.exists()
