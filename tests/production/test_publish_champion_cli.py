from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_decision(
    path: Path,
    *,
    selected_model: object,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            {
                "selection": {
                    "selected_model": selected_model,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_parser_accepts_publication_arguments() -> None:
    import lrp.cli.publish_champion as cli

    parser = cli._parser()

    arguments = parser.parse_args(
        [
            "--champion-decision",
            "decision.json",
            "--production-registry",
            "registry",
        ]
    )

    assert (
        arguments.champion_decision
        == Path("decision.json")
    )

    assert (
        arguments.production_registry
        == Path("registry")
    )


def test_run_publish_returns_result(
    tmp_path: Path,
) -> None:
    import lrp.cli.publish_champion as cli

    source = (
        tmp_path
        / "champion_decision.json"
    )

    registry = (
        tmp_path
        / "registry"
    )

    _write_decision(
        source,
        selected_model="combined",
    )

    summary = cli.run_publish(
        champion_decision=source,
        production_registry=registry,
    )

    assert summary[
        "status"
    ] == "PASS"

    assert summary[
        "selected_model"
    ] == "combined"

    assert Path(
        summary[
            "published_path"
        ]
    ).is_file()

    assert (
        registry
        / "active"
        / "publication.json"
    ).is_file()


def test_run_publish_preserves_none_model(
    tmp_path: Path,
) -> None:
    import lrp.cli.publish_champion as cli

    source = (
        tmp_path
        / "champion_decision.json"
    )

    registry = (
        tmp_path
        / "registry"
    )

    _write_decision(
        source,
        selected_model=None,
    )

    summary = cli.run_publish(
        champion_decision=source,
        production_registry=registry,
    )

    assert summary[
        "selected_model"
    ] is None


def test_main_prints_json_on_success(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import lrp.cli.publish_champion as cli

    source = (
        tmp_path
        / "champion_decision.json"
    )

    registry = (
        tmp_path
        / "registry"
    )

    _write_decision(
        source,
        selected_model="baseline",
    )

    exit_code = cli.main(
        [
            "--champion-decision",
            str(source),
            "--production-registry",
            str(registry),
        ]
    )

    assert exit_code == 0

    captured = capsys.readouterr()

    payload = json.loads(
        captured.out
    )

    assert payload[
        "status"
    ] == "PASS"

    assert payload[
        "selected_model"
    ] == "baseline"

    assert captured.err == ""


def test_main_returns_one_for_missing_source(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import lrp.cli.publish_champion as cli

    exit_code = cli.main(
        [
            "--champion-decision",
            str(
                tmp_path
                / "missing.json"
            ),
            "--production-registry",
            str(
                tmp_path
                / "registry"
            ),
        ]
    )

    assert exit_code == 1

    captured = capsys.readouterr()

    payload = json.loads(
        captured.err
    )

    assert payload[
        "status"
    ] == "ERROR"

    assert payload[
        "error_type"
    ] == "FileNotFoundError"


def test_main_returns_one_for_invalid_schema(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import lrp.cli.publish_champion as cli

    source = (
        tmp_path
        / "invalid.json"
    )

    source.write_text(
        json.dumps(
            {
                "selection": {
                    "selected_model": 123,
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "--champion-decision",
            str(source),
            "--production-registry",
            str(
                tmp_path
                / "registry"
            ),
        ]
    )

    assert exit_code == 1

    captured = capsys.readouterr()

    payload = json.loads(
        captured.err
    )

    assert payload[
        "status"
    ] == "ERROR"

    assert payload[
        "error_type"
    ] == "ValueError"


def test_success_summary_contains_provenance(
    tmp_path: Path,
) -> None:
    import lrp.cli.publish_champion as cli

    source = (
        tmp_path
        / "champion_decision.json"
    )

    registry = (
        tmp_path
        / "registry"
    )

    _write_decision(
        source,
        selected_model="bayesian",
    )

    summary = cli.run_publish(
        champion_decision=source,
        production_registry=registry,
    )

    assert isinstance(
        summary[
            "source_sha256"
        ],
        str,
    )

    assert len(
        summary[
            "source_sha256"
        ]
    ) == 64

    assert isinstance(
        summary[
            "published_at_kst"
        ],
        str,
    )

    assert (
        summary[
            "source_path"
        ]
        == str(source)
    )


def test_parser_requires_champion_decision() -> None:
    import lrp.cli.publish_champion as cli

    parser = cli._parser()

    with pytest.raises(
        SystemExit,
    ):
        parser.parse_args(
            [
                "--production-registry",
                "registry",
            ]
        )


def test_parser_requires_production_registry() -> None:
    import lrp.cli.publish_champion as cli

    parser = cli._parser()

    with pytest.raises(
        SystemExit,
    ):
        parser.parse_args(
            [
                "--champion-decision",
                "decision.json",
            ]
        )