from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.validation.run_adaptive_automation import (
    run as run_adaptive_automation,
)
from tools.validation.run_adaptive_doctor import (
    run as run_adaptive_doctor,
)
from tools.validation.run_adaptive_rollback import (
    run as run_adaptive_rollback,
)


WEIGHT_FIELDS = (
    "hot_weight",
    "cold_weight",
    "gap_weight",
    "trend_weight",
    "transition_weight",
    "learning_weight",
    "adaptive_weight",
)


def write_cross_window_report(
    path: Path,
) -> None:
    trends = {
        field: {
            "values": [
                0.05,
                0.05,
                0.05,
            ],
            "first": 0.05,
            "last": 0.05,
            "net_change": 0.0,
            "mean": 0.05,
            "direction": "stable",
            "increase_steps": 0,
            "decrease_steps": 0,
            "stable_steps": 2,
        }
        for field in WEIGHT_FIELDS
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
            "window_count": 3,
            "policy_count": 1,
            "policies": {
                "floor": {
                    "weights": trends,
                }
            },
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
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_initial_profile(
    path: Path,
) -> None:
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
            "2026-08-01T00:00:00+00:00"
        ),
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def read_json(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert isinstance(payload, dict)

    return payload


def consume_stdout_json(
    capsys: pytest.CaptureFixture[str],
) -> dict[str, Any]:
    captured = capsys.readouterr()

    payload = json.loads(
        captured.out
    )

    assert isinstance(payload, dict)

    return payload


def automation_arguments(
    *,
    report_path: Path,
    profile_path: Path,
    repository: Path,
    recommendation_id: str,
    created_at_utc: str,
    dry_run: bool = False,
    approve: bool = False,
) -> list[str]:
    arguments = [
        "--report",
        str(report_path),
        "--profile",
        str(profile_path),
        "--repository",
        str(repository),
        "--policy",
        "floor",
        "--recommendation-id",
        recommendation_id,
        "--created-at-utc",
        created_at_utc,
    ]

    if dry_run:
        arguments.append("--dry-run")

    if approve:
        arguments.append("--approve")

    return arguments


def doctor_arguments(
    *,
    repository: Path,
    output: Path,
    stem: str,
    fail_on_issues: bool = False,
) -> list[str]:
    arguments = [
        "--repository",
        str(repository),
        "--output",
        str(output),
        "--stem",
        stem,
    ]

    if fail_on_issues:
        arguments.append(
            "--fail-on-issues"
        )

    return arguments


def rollback_arguments(
    *,
    profile_path: Path,
    repository: Path,
    rollback_revision: int,
    rollback_id: str,
    generated_at_utc: str,
    dry_run: bool = False,
    approve: bool = False,
) -> list[str]:
    arguments = [
        "--profile",
        str(profile_path),
        "--repository",
        str(repository),
        "--rollback-revision",
        str(rollback_revision),
        "--rollback-id",
        rollback_id,
        "--generated-at-utc",
        generated_at_utc,
    ]

    if dry_run:
        arguments.append("--dry-run")

    if approve:
        arguments.append(
            "--approve-rollback"
        )

    return arguments


def test_adaptive_automation_end_to_end(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = (
        tmp_path
        / "inputs"
        / "cross_window_report.json"
    )
    initial_profile_path = (
        tmp_path
        / "inputs"
        / "profile-12.json"
    )
    repository = (
        tmp_path
        / "adaptive_repository"
    )
    doctor_output = (
        tmp_path
        / "doctor_reports"
    )

    write_cross_window_report(
        report_path
    )
    write_initial_profile(
        initial_profile_path
    )

    # 1. Dry-run: 저장소를 생성하지 않아야 한다.
    assert run_adaptive_automation(
        automation_arguments(
            report_path=report_path,
            profile_path=(
                initial_profile_path
            ),
            repository=repository,
            recommendation_id=(
                "e2e-dry-run-13"
            ),
            created_at_utc=(
                "2026-08-02T00:00:00+00:00"
            ),
            dry_run=True,
        )
    ) == 0

    dry_payload = consume_stdout_json(
        capsys
    )

    assert dry_payload["mode"] == (
        "dry_run"
    )
    assert dry_payload[
        "source_revision"
    ] == 12
    assert dry_payload[
        "target_revision"
    ] == 13
    assert dry_payload[
        "automation_created"
    ] is False
    assert dry_payload[
        "profile_created"
    ] is False
    assert not repository.exists()

    # 2. 명시적 승인으로 revision 13 저장.
    assert run_adaptive_automation(
        automation_arguments(
            report_path=report_path,
            profile_path=(
                initial_profile_path
            ),
            repository=repository,
            recommendation_id=(
                "e2e-approved-13"
            ),
            created_at_utc=(
                "2026-08-02T00:00:00+00:00"
            ),
            approve=True,
        )
    ) == 0

    first_payload = consume_stdout_json(
        capsys
    )

    assert first_payload["mode"] == (
        "persisted"
    )
    assert first_payload[
        "source_revision"
    ] == 12
    assert first_payload[
        "target_revision"
    ] == 13
    assert first_payload[
        "repository_revision_before"
    ] is None
    assert first_payload[
        "repository_revision_after"
    ] == 13
    assert first_payload[
        "automation_created"
    ] is True
    assert first_payload[
        "profile_created"
    ] is True

    revision_13_path = (
        repository
        / "profiles"
        / "revision-00000013.json"
    )
    automation_13_path = (
        repository
        / "automation"
        / "e2e-approved-13.json"
    )

    assert revision_13_path.is_file()
    assert automation_13_path.is_file()

    revision_13 = read_json(
        revision_13_path
    )

    assert revision_13[
        "source_revision"
    ] == 12
    assert revision_13[
        "target_revision"
    ] == 13
    assert revision_13[
        "profile"
    ]["revision"] == 13

    # 3. Doctor는 최초 저장소를 PASS로 진단해야 한다.
    assert run_adaptive_doctor(
        doctor_arguments(
            repository=repository,
            output=doctor_output,
            stem="doctor_after_13",
            fail_on_issues=True,
        )
    ) == 0

    doctor_13_summary = (
        consume_stdout_json(capsys)
    )

    assert doctor_13_summary[
        "status"
    ] == "PASS"
    assert doctor_13_summary[
        "overall_ok"
    ] is True
    assert doctor_13_summary[
        "latest_revision"
    ] == 13
    assert doctor_13_summary[
        "recommendation_count"
    ] == 1

    doctor_13_json = read_json(
        doctor_output
        / "doctor_after_13.json"
    )

    assert doctor_13_json[
        "overall_ok"
    ] is True
    assert doctor_13_json[
        "latest_revision"
    ] == 13

    doctor_13_markdown = (
        doctor_output
        / "doctor_after_13.md"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "- Overall status: PASS"
        in doctor_13_markdown
    )
    assert doctor_13_markdown.endswith(
        "## Conclusion\n\nPASS\n"
    )

    # 4. 저장된 revision 13을 현재 프로필로 사용해 revision 14 생성.
    assert run_adaptive_automation(
        automation_arguments(
            report_path=report_path,
            profile_path=(
                revision_13_path
            ),
            repository=repository,
            recommendation_id=(
                "e2e-approved-14"
            ),
            created_at_utc=(
                "2026-08-03T00:00:00+00:00"
            ),
            approve=True,
        )
    ) == 0

    second_payload = consume_stdout_json(
        capsys
    )

    assert second_payload[
        "repository_revision_before"
    ] == 13
    assert second_payload[
        "repository_revision_after"
    ] == 14
    assert second_payload[
        "target_revision"
    ] == 14

    revision_14_path = (
        repository
        / "profiles"
        / "revision-00000014.json"
    )

    assert revision_14_path.is_file()

    revision_14 = read_json(
        revision_14_path
    )

    assert revision_14[
        "source_revision"
    ] == 13
    assert revision_14[
        "target_revision"
    ] == 14

    # 5. Rollback dry-run: revision 15는 생성되지 않아야 한다.
    revision_15_path = (
        repository
        / "profiles"
        / "revision-00000015.json"
    )

    assert run_adaptive_rollback(
        rollback_arguments(
            profile_path=revision_14_path,
            repository=repository,
            rollback_revision=13,
            rollback_id=(
                "e2e-rollback-dry-15"
            ),
            generated_at_utc=(
                "2026-08-04T00:00:00+00:00"
            ),
            dry_run=True,
        )
    ) == 0

    rollback_dry_payload = (
        consume_stdout_json(capsys)
    )

    assert rollback_dry_payload[
        "mode"
    ] == "dry_run"
    assert rollback_dry_payload[
        "created"
    ] is False
    assert rollback_dry_payload[
        "plan"
    ]["source_revision"] == 14
    assert rollback_dry_payload[
        "plan"
    ]["rollback_revision"] == 13
    assert rollback_dry_payload[
        "plan"
    ]["target_revision"] == 15
    assert not revision_15_path.exists()

    # 6. 승인 rollback으로 revision 15 생성.
    assert run_adaptive_rollback(
        rollback_arguments(
            profile_path=revision_14_path,
            repository=repository,
            rollback_revision=13,
            rollback_id=(
                "e2e-rollback-15"
            ),
            generated_at_utc=(
                "2026-08-04T00:00:00+00:00"
            ),
            approve=True,
        )
    ) == 0

    rollback_payload = (
        consume_stdout_json(capsys)
    )

    assert rollback_payload[
        "mode"
    ] == "persisted"
    assert rollback_payload[
        "created"
    ] is True
    assert revision_15_path.is_file()

    revision_15 = read_json(
        revision_15_path
    )

    assert revision_15[
        "record_type"
    ] == "rollback"
    assert revision_15[
        "source_revision"
    ] == 14
    assert revision_15[
        "rollback_revision"
    ] == 13
    assert revision_15[
        "target_revision"
    ] == 15
    assert revision_15[
        "profile"
    ]["revision"] == 15

    # rollback 결과의 가중치는 revision 13의 가중치와 같아야 한다.
    for field in WEIGHT_FIELDS:
        assert revision_15[
            "profile"
        ][field] == pytest.approx(
            revision_13[
                "profile"
            ][field]
        )

    # 7. Rollback 이후 Doctor 재진단.
    assert run_adaptive_doctor(
        doctor_arguments(
            repository=repository,
            output=doctor_output,
            stem="doctor_after_rollback",
            fail_on_issues=True,
        )
    ) == 0

    final_doctor_summary = (
        consume_stdout_json(capsys)
    )

    assert final_doctor_summary[
        "status"
    ] == "PASS"
    assert final_doctor_summary[
        "overall_ok"
    ] is True
    assert final_doctor_summary[
        "latest_revision"
    ] == 15
    assert final_doctor_summary[
        "rollback_count"
    ] == 1
    assert final_doctor_summary[
        "recommendation_count"
    ] == 2
    assert final_doctor_summary[
        "error_count"
    ] == 0

    final_doctor = read_json(
        doctor_output
        / "doctor_after_rollback.json"
    )

    assert final_doctor[
        "latest_revision"
    ] == 15
    assert final_doctor[
        "rollback_count"
    ] == 1
    assert final_doctor[
        "repository"
    ]["rollback_count"] == 1
    assert final_doctor[
        "profile"
    ]["revision_count"] == 3
    assert final_doctor[
        "profile"
    ]["first_revision"] == 13
    assert final_doctor[
        "profile"
    ]["latest_revision"] == 15

    final_markdown = (
        doctor_output
        / "doctor_after_rollback.md"
    ).read_text(
        encoding="utf-8"
    )

    assert "- Latest revision: 15" in (
        final_markdown
    )
    assert "- Rollback count: 1" in (
        final_markdown
    )
    assert final_markdown.endswith(
        "## Conclusion\n\nPASS\n"
    )


def test_end_to_end_rejects_stale_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = (
        tmp_path / "report.json"
    )
    profile_path = (
        tmp_path / "profile-12.json"
    )
    repository = (
        tmp_path / "repository"
    )

    write_cross_window_report(
        report_path
    )
    write_initial_profile(
        profile_path
    )

    assert run_adaptive_automation(
        automation_arguments(
            report_path=report_path,
            profile_path=profile_path,
            repository=repository,
            recommendation_id=(
                "first-13"
            ),
            created_at_utc=(
                "2026-08-02T00:00:00+00:00"
            ),
            approve=True,
        )
    ) == 0

    consume_stdout_json(capsys)

    with pytest.raises(
        SystemExit,
    ) as error:
        run_adaptive_automation(
            automation_arguments(
                report_path=report_path,
                profile_path=profile_path,
                repository=repository,
                recommendation_id=(
                    "stale-second"
                ),
                created_at_utc=(
                    "2026-08-03T00:00:00+00:00"
                ),
                approve=True,
            )
        )

    assert error.value.code == 2

    assert not (
        repository
        / "profiles"
        / "revision-00000014.json"
    ).exists()


def test_doctor_reports_are_idempotent(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = (
        tmp_path / "repository"
    )
    output = tmp_path / "reports"

    profile_path = (
        repository
        / "profiles"
        / "revision-00000012.json"
    )

    write_initial_profile(
        profile_path
    )

    nested = read_json(
        profile_path
    )

    repository_payload = {
        "schema_version": 1,
        "source_revision": 11,
        "target_revision": 12,
        "profile": nested,
    }

    profile_path.write_text(
        json.dumps(
            repository_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    arguments = doctor_arguments(
        repository=repository,
        output=output,
        stem="doctor",
    )

    assert run_adaptive_doctor(
        arguments
    ) == 0

    first = consume_stdout_json(
        capsys
    )

    json_bytes = (
        output / "doctor.json"
    ).read_bytes()
    markdown_bytes = (
        output / "doctor.md"
    ).read_bytes()

    assert run_adaptive_doctor(
        arguments
    ) == 0

    second = consume_stdout_json(
        capsys
    )

    assert first["json_created"] is True
    assert first[
        "markdown_created"
    ] is True

    assert second[
        "json_created"
    ] is False
    assert second[
        "json_changed"
    ] is False
    assert second[
        "markdown_created"
    ] is False
    assert second[
        "markdown_changed"
    ] is False

    assert (
        output / "doctor.json"
    ).read_bytes() == json_bytes

    assert (
        output / "doctor.md"
    ).read_bytes() == markdown_bytes
