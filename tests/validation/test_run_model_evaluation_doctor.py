from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _pass_result() -> object:
    return SimpleNamespace(
        status="PASS",
        overall_ok=True,
        as_dict=lambda: {
            "status": "PASS",
            "overall_ok": True,
            "total_count": 1,
            "pass_count": 1,
            "fail_count": 0,
            "incomplete_count": 0,
            "issues": [],
            "runs": [
                {
                    "run_id": "aaaaaaaaaaaaaaaa",
                    "status": "PASS",
                    "issues": [],
                }
            ],
        },
    )


def _fail_result() -> object:
    return SimpleNamespace(
        status="FAIL",
        overall_ok=False,
        as_dict=lambda: {
            "status": "FAIL",
            "overall_ok": False,
            "total_count": 1,
            "pass_count": 0,
            "fail_count": 1,
            "incomplete_count": 0,
            "issues": [
                "run_failed:bbbbbbbbbbbbbbbb",
            ],
            "runs": [
                {
                    "run_id": "bbbbbbbbbbbbbbbb",
                    "status": "FAIL",
                    "issues": [
                        "ranking_champion_mismatch",
                    ],
                }
            ],
        },
    )


def test_cli_outputs_json_for_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import tools.validation.run_model_evaluation_doctor as cli

    class FakeDoctor:
        def inspect(
            self,
            root: str | Path,
        ) -> object:
            assert Path(root) == tmp_path
            return _pass_result()

    monkeypatch.setattr(
        cli,
        "ModelEvaluationDoctor",
        FakeDoctor,
    )

    exit_code = cli.run(
        [
            "--root",
            str(tmp_path),
        ]
    )

    assert exit_code == 0

    payload = json.loads(
        capsys.readouterr().out
    )

    assert payload["status"] == "PASS"
    assert payload["overall_ok"] is True
    assert payload["total_count"] == 1


def test_cli_fail_result_is_zero_without_fail_on_issues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import tools.validation.run_model_evaluation_doctor as cli

    class FakeDoctor:
        def inspect(
            self,
            root: str | Path,
        ) -> object:
            return _fail_result()

    monkeypatch.setattr(
        cli,
        "ModelEvaluationDoctor",
        FakeDoctor,
    )

    exit_code = cli.run(
        [
            "--root",
            str(tmp_path),
        ]
    )

    assert exit_code == 0

    payload = json.loads(
        capsys.readouterr().out
    )

    assert payload["status"] == "FAIL"
    assert payload["overall_ok"] is False


def test_cli_fail_result_is_one_with_fail_on_issues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.validation.run_model_evaluation_doctor as cli

    class FakeDoctor:
        def inspect(
            self,
            root: str | Path,
        ) -> object:
            return _fail_result()

    monkeypatch.setattr(
        cli,
        "ModelEvaluationDoctor",
        FakeDoctor,
    )

    exit_code = cli.run(
        [
            "--root",
            str(tmp_path),
            "--fail-on-issues",
        ]
    )

    assert exit_code == 1


def test_cli_returns_two_for_missing_root(
    tmp_path: Path,
) -> None:
    import tools.validation.run_model_evaluation_doctor as cli

    exit_code = cli.run(
        [
            "--root",
            str(
                tmp_path
                / "missing"
            ),
        ]
    )

    assert exit_code == 2


def test_cli_returns_two_for_file_root(
    tmp_path: Path,
) -> None:
    import tools.validation.run_model_evaluation_doctor as cli

    root = (
        tmp_path
        / "artifact.txt"
    )

    root.write_text(
        "x",
        encoding="utf-8",
    )

    exit_code = cli.run(
        [
            "--root",
            str(root),
        ]
    )

    assert exit_code == 2


def test_parser_requires_root() -> None:
    import tools.validation.run_model_evaluation_doctor as cli

    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])
