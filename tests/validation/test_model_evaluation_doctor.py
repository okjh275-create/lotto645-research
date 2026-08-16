from __future__ import annotations

from pathlib import Path

import pytest

from tools.validation.model_evaluation_run_audit import (
    ModelEvaluationRunAuditResult,
)
from tools.validation.model_evaluation_run_audit_service import (
    ModelEvaluationRunAuditReport,
)


def _result(
    *,
    run_id: str,
    status: str,
    issues: tuple[str, ...] = (),
) -> ModelEvaluationRunAuditResult:
    return ModelEvaluationRunAuditResult(
        run_id=run_id,
        status=status,
        issues=issues,
        evaluation_run=Path(
            f"{run_id}/evaluation_run.json"
        ),
        champion_artifact=Path(
            f"{run_id}/champion_decision.json"
        ),
    )


class _FakeAuditService:
    def __init__(
        self,
        report: ModelEvaluationRunAuditReport,
    ) -> None:
        self._report = report
        self.calls: list[Path] = []

    def run(
        self,
        root: str | Path,
    ) -> ModelEvaluationRunAuditReport:
        self.calls.append(
            Path(root)
        )
        return self._report


def test_doctor_passes_when_all_runs_pass(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_doctor import (
        ModelEvaluationDoctor,
    )

    report = ModelEvaluationRunAuditReport(
        results=(
            _result(
                run_id="aaaaaaaaaaaaaaaa",
                status="PASS",
            ),
            _result(
                run_id="bbbbbbbbbbbbbbbb",
                status="PASS",
            ),
        ),
    )

    service = _FakeAuditService(report)

    result = ModelEvaluationDoctor(
        audit_service=service,
    ).inspect(tmp_path)

    assert service.calls == [tmp_path]

    assert result.status == "PASS"
    assert result.overall_ok is True
    assert result.total_count == 2
    assert result.pass_count == 2
    assert result.fail_count == 0
    assert result.incomplete_count == 0
    assert result.issues == ()


def test_doctor_fails_when_a_run_fails(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_doctor import (
        ModelEvaluationDoctor,
    )

    report = ModelEvaluationRunAuditReport(
        results=(
            _result(
                run_id="aaaaaaaaaaaaaaaa",
                status="PASS",
            ),
            _result(
                run_id="bbbbbbbbbbbbbbbb",
                status="FAIL",
                issues=(
                    "ranking_champion_mismatch",
                ),
            ),
        ),
    )

    result = ModelEvaluationDoctor(
        audit_service=_FakeAuditService(
            report
        ),
    ).inspect(tmp_path)

    assert result.status == "FAIL"
    assert result.overall_ok is False
    assert result.total_count == 2
    assert result.pass_count == 1
    assert result.fail_count == 1
    assert result.incomplete_count == 0
    assert result.issues == (
        "run_failed:bbbbbbbbbbbbbbbb",
    )


def test_doctor_fails_when_a_run_is_incomplete(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_doctor import (
        ModelEvaluationDoctor,
    )

    report = ModelEvaluationRunAuditReport(
        results=(
            _result(
                run_id="aaaaaaaaaaaaaaaa",
                status="INCOMPLETE",
                issues=(
                    "champion_artifact_missing",
                ),
            ),
        ),
    )

    result = ModelEvaluationDoctor(
        audit_service=_FakeAuditService(
            report
        ),
    ).inspect(tmp_path)

    assert result.status == "FAIL"
    assert result.overall_ok is False
    assert result.total_count == 1
    assert result.pass_count == 0
    assert result.fail_count == 0
    assert result.incomplete_count == 1
    assert result.issues == (
        "run_incomplete:aaaaaaaaaaaaaaaa",
    )


def test_doctor_fails_when_no_runs_are_discovered(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_doctor import (
        ModelEvaluationDoctor,
    )

    report = ModelEvaluationRunAuditReport(
        results=(),
    )

    result = ModelEvaluationDoctor(
        audit_service=_FakeAuditService(
            report
        ),
    ).inspect(tmp_path)

    assert result.status == "FAIL"
    assert result.overall_ok is False
    assert result.total_count == 0
    assert result.pass_count == 0
    assert result.fail_count == 0
    assert result.incomplete_count == 0
    assert result.issues == (
        "no_evaluation_runs",
    )


def test_doctor_result_serialization(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_doctor import (
        ModelEvaluationDoctor,
    )

    report = ModelEvaluationRunAuditReport(
        results=(
            _result(
                run_id="aaaaaaaaaaaaaaaa",
                status="PASS",
            ),
            _result(
                run_id="bbbbbbbbbbbbbbbb",
                status="FAIL",
                issues=(
                    "selected_model_mismatch",
                ),
            ),
        ),
    )

    result = ModelEvaluationDoctor(
        audit_service=_FakeAuditService(
            report
        ),
    ).inspect(tmp_path)

    assert result.as_dict() == {
        "status": "FAIL",
        "overall_ok": False,
        "total_count": 2,
        "pass_count": 1,
        "fail_count": 1,
        "incomplete_count": 0,
        "issues": [
            "run_failed:bbbbbbbbbbbbbbbb",
        ],
        "runs": [
            {
                "run_id": "aaaaaaaaaaaaaaaa",
                "status": "PASS",
                "issues": [],
            },
            {
                "run_id": "bbbbbbbbbbbbbbbb",
                "status": "FAIL",
                "issues": [
                    "selected_model_mismatch",
                ],
            },
        ],
    }


def test_doctor_rejects_missing_root(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_doctor import (
        ModelEvaluationDoctor,
    )

    root = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        ModelEvaluationDoctor().inspect(root)


def test_doctor_rejects_file_root(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_doctor import (
        ModelEvaluationDoctor,
    )

    root = tmp_path / "artifact.txt"

    root.write_text(
        "not a directory\n",
        encoding="utf-8",
    )

    with pytest.raises(NotADirectoryError):
        ModelEvaluationDoctor().inspect(root)
