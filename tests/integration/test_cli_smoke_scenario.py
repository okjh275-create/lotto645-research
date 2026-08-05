from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
            "values": [0.05, 0.05, 0.05],
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
                        "direction": "adaptive_better",
                        "p_value": 0.40,
                        "significant": False,
                    },
                    "practical": {
                        "adaptive_wins": 32,
                        "noop_wins": 18,
                        "ties": 250,
                        "direction": "adaptive_better",
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


def run_module(
    module: str,
    arguments: list[str],
    *,
    expected_returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()

    existing_python_path = environment.get(
        "PYTHONPATH"
    )

    environment["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        if not existing_python_path
        else (
            str(PROJECT_ROOT)
            + os.pathsep
            + existing_python_path
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            module,
            *arguments,
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        check=False,
        timeout=30,
    )

    assert result.returncode == (
        expected_returncode
    ), (
        f"module={module}\n"
        f"expected_returncode="
        f"{expected_returncode}\n"
        f"actual_returncode="
        f"{result.returncode}\n"
        f"stdout={result.stdout}\n"
        f"stderr={result.stderr}"
    )

    return result


def stdout_json(
    result: subprocess.CompletedProcess[str],
) -> dict[str, Any]:
    assert result.stdout.strip()
    assert result.stderr == ""

    payload = json.loads(
        result.stdout
    )

    assert isinstance(payload, dict)

    return payload


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


def automation_args(
    *,
    report: Path,
    profile: Path,
    repository: Path,
    recommendation_id: str,
    timestamp: str,
    dry_run: bool = False,
    approve: bool = False,
) -> list[str]:
    arguments = [
        "--report",
        str(report),
        "--profile",
        str(profile),
        "--repository",
        str(repository),
        "--policy",
        "floor",
        "--recommendation-id",
        recommendation_id,
        "--created-at-utc",
        timestamp,
    ]

    if dry_run:
        arguments.append("--dry-run")

    if approve:
        arguments.append("--approve")

    return arguments


def doctor_args(
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


def rollback_args(
    *,
    profile: Path,
    repository: Path,
    rollback_revision: int,
    rollback_id: str,
    timestamp: str,
    dry_run: bool = False,
    approve: bool = False,
) -> list[str]:
    arguments = [
        "--profile",
        str(profile),
        "--repository",
        str(repository),
        "--rollback-revision",
        str(rollback_revision),
        "--rollback-id",
        rollback_id,
        "--generated-at-utc",
        timestamp,
    ]

    if dry_run:
        arguments.append("--dry-run")

    if approve:
        arguments.append(
            "--approve-rollback"
        )

    return arguments


def test_cli_operational_smoke_scenario(
    tmp_path: Path,
) -> None:
    report_path = (
        tmp_path
        / "input"
        / "cross_window_report.json"
    )
    profile_12_path = (
        tmp_path
        / "input"
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
        profile_12_path
    )

    # 1. 실제 CLI dry-run.
    dry_result = run_module(
        "tools.validation."
        "run_adaptive_automation",
        automation_args(
            report=report_path,
            profile=profile_12_path,
            repository=repository,
            recommendation_id=(
                "cli-dry-13"
            ),
            timestamp=(
                "2026-08-02T00:00:00+00:00"
            ),
            dry_run=True,
        ),
    )

    dry_payload = stdout_json(
        dry_result
    )

    assert dry_payload["mode"] == "dry_run"
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

    # 2. 실제 CLI 승인 저장.
    approved_result = run_module(
        "tools.validation."
        "run_adaptive_automation",
        automation_args(
            report=report_path,
            profile=profile_12_path,
            repository=repository,
            recommendation_id=(
                "cli-approved-13"
            ),
            timestamp=(
                "2026-08-02T00:00:00+00:00"
            ),
            approve=True,
        ),
    )

    approved_payload = stdout_json(
        approved_result
    )

    assert approved_payload[
        "mode"
    ] == "persisted"
    assert approved_payload[
        "repository_revision_after"
    ] == 13
    assert approved_payload[
        "automation_created"
    ] is True
    assert approved_payload[
        "profile_created"
    ] is True

    profile_13_path = (
        repository
        / "profiles"
        / "revision-00000013.json"
    )

    assert profile_13_path.is_file()

    # 3. 실제 Doctor CLI 검사.
    doctor_result = run_module(
        "tools.validation."
        "run_adaptive_doctor",
        doctor_args(
            repository=repository,
            output=doctor_output,
            stem="after_13",
            fail_on_issues=True,
        ),
    )

    doctor_payload = stdout_json(
        doctor_result
    )

    assert doctor_payload[
        "status"
    ] == "PASS"
    assert doctor_payload[
        "overall_ok"
    ] is True
    assert doctor_payload[
        "latest_revision"
    ] == 13

    doctor_json_path = (
        doctor_output
        / "after_13.json"
    )
    doctor_markdown_path = (
        doctor_output
        / "after_13.md"
    )

    assert doctor_json_path.is_file()
    assert doctor_markdown_path.is_file()

    assert not doctor_json_path.read_bytes().startswith(
        b"\xef\xbb\xbf"
    )
    assert not doctor_markdown_path.read_bytes().startswith(
        b"\xef\xbb\xbf"
    )

    # 4. 실제 CLI로 revision 14 생성.
    revision_14_result = run_module(
        "tools.validation."
        "run_adaptive_automation",
        automation_args(
            report=report_path,
            profile=profile_13_path,
            repository=repository,
            recommendation_id=(
                "cli-approved-14"
            ),
            timestamp=(
                "2026-08-03T00:00:00+00:00"
            ),
            approve=True,
        ),
    )

    revision_14_payload = stdout_json(
        revision_14_result
    )

    assert revision_14_payload[
        "repository_revision_before"
    ] == 13
    assert revision_14_payload[
        "repository_revision_after"
    ] == 14

    profile_14_path = (
        repository
        / "profiles"
        / "revision-00000014.json"
    )

    assert profile_14_path.is_file()

    # 5. 실제 rollback dry-run.
    rollback_dry_result = run_module(
        "tools.validation."
        "run_adaptive_rollback",
        rollback_args(
            profile=profile_14_path,
            repository=repository,
            rollback_revision=13,
            rollback_id=(
                "cli-rollback-dry-15"
            ),
            timestamp=(
                "2026-08-04T00:00:00+00:00"
            ),
            dry_run=True,
        ),
    )

    rollback_dry_payload = stdout_json(
        rollback_dry_result
    )

    assert rollback_dry_payload[
        "mode"
    ] == "dry_run"
    assert rollback_dry_payload[
        "plan"
    ]["target_revision"] == 15

    profile_15_path = (
        repository
        / "profiles"
        / "revision-00000015.json"
    )

    assert not profile_15_path.exists()

    # 6. 실제 rollback 승인 저장.
    rollback_result = run_module(
        "tools.validation."
        "run_adaptive_rollback",
        rollback_args(
            profile=profile_14_path,
            repository=repository,
            rollback_revision=13,
            rollback_id=(
                "cli-rollback-15"
            ),
            timestamp=(
                "2026-08-04T00:00:00+00:00"
            ),
            approve=True,
        ),
    )

    rollback_payload = stdout_json(
        rollback_result
    )

    assert rollback_payload[
        "mode"
    ] == "persisted"
    assert rollback_payload[
        "created"
    ] is True
    assert profile_15_path.is_file()

    profile_13 = read_json(
        profile_13_path
    )
    profile_15 = read_json(
        profile_15_path
    )

    assert profile_15[
        "record_type"
    ] == "rollback"
    assert profile_15[
        "source_revision"
    ] == 14
    assert profile_15[
        "rollback_revision"
    ] == 13
    assert profile_15[
        "target_revision"
    ] == 15

    for field in WEIGHT_FIELDS:
        assert profile_15[
            "profile"
        ][field] == pytest.approx(
            profile_13[
                "profile"
            ][field]
        )

    # 7. rollback 이후 실제 Doctor CLI 재검사.
    final_doctor_result = run_module(
        "tools.validation."
        "run_adaptive_doctor",
        doctor_args(
            repository=repository,
            output=doctor_output,
            stem="after_rollback",
            fail_on_issues=True,
        ),
    )

    final_payload = stdout_json(
        final_doctor_result
    )

    assert final_payload[
        "status"
    ] == "PASS"
    assert final_payload[
        "latest_revision"
    ] == 15
    assert final_payload[
        "rollback_count"
    ] == 1
    assert final_payload[
        "recommendation_count"
    ] == 2

    final_report = read_json(
        doctor_output
        / "after_rollback.json"
    )

    assert final_report[
        "overall_ok"
    ] is True
    assert final_report[
        "profile"
    ]["revision_count"] == 3
    assert final_report[
        "profile"
    ]["first_revision"] == 13
    assert final_report[
        "profile"
    ]["latest_revision"] == 15

    markdown = (
        doctor_output
        / "after_rollback.md"
    ).read_text(
        encoding="utf-8"
    )

    assert "- Latest revision: 15" in (
        markdown
    )
    assert "- Rollback count: 1" in (
        markdown
    )
    assert markdown.endswith(
        "## Conclusion\n\nPASS\n"
    )


def test_cli_fail_on_issues_exit_code(
    tmp_path: Path,
) -> None:
    output = tmp_path / "reports"

    result = run_module(
        "tools.validation."
        "run_adaptive_doctor",
        doctor_args(
            repository=(
                tmp_path
                / "empty_repository"
            ),
            output=output,
            stem="empty",
            fail_on_issues=True,
        ),
        expected_returncode=1,
    )

    payload = stdout_json(result)

    assert payload["status"] == "FAIL"
    assert payload["overall_ok"] is False

    assert (
        output / "empty.json"
    ).is_file()
    assert (
        output / "empty.md"
    ).is_file()


def test_cli_rejects_unapproved_write(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "report.json"
    profile_path = tmp_path / "profile.json"
    repository = tmp_path / "repository"

    write_cross_window_report(
        report_path
    )
    write_initial_profile(
        profile_path
    )

    result = run_module(
        "tools.validation."
        "run_adaptive_automation",
        automation_args(
            report=report_path,
            profile=profile_path,
            repository=repository,
            recommendation_id=(
                "not-approved"
            ),
            timestamp=(
                "2026-08-02T00:00:00+00:00"
            ),
        ),
        expected_returncode=2,
    )

    assert result.stdout == ""
    assert (
        "explicit --approve"
        in result.stderr
    )
    assert not repository.exists()
