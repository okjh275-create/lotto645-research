from __future__ import annotations

import json
from pathlib import Path

import pytest


def _parser():
    import lrp.cli.predict as cli

    return cli._parser()


def test_parser_accepts_production_activation_options() -> None:
    parser = _parser()

    args = parser.parse_args(
        [
            "--history",
            "history.json",
            "--round",
            "1232",
            "--seed",
            "20260816",
            "--champion-decision",
            "champion_decision.json",
            "--production-snapshot-root",
            "snapshots",
        ]
    )

    assert args.champion_decision == Path(
        "champion_decision.json"
    )

    assert args.production_snapshot_root == Path(
        "snapshots"
    )


def test_champion_decision_requires_snapshot_root() -> None:
    import lrp.cli.predict as cli

    with pytest.raises(
        ValueError,
        match="production_snapshot_root",
    ):
        cli._resolve_production_configuration(
            champion_decision=Path(
                "champion_decision.json"
            ),
            production_snapshot_root=None,
        )


def test_snapshot_root_requires_champion_decision() -> None:
    import lrp.cli.predict as cli

    with pytest.raises(
        ValueError,
        match="champion_decision",
    ):
        cli._resolve_production_configuration(
            champion_decision=None,
            production_snapshot_root=Path(
                "snapshots"
            ),
        )


def test_no_options_preserve_default_pipeline_kwargs() -> None:
    import lrp.cli.predict as cli

    result = cli._resolve_production_configuration(
        champion_decision=None,
        production_snapshot_root=None,
    )

    assert result is None


def test_resolver_returns_production_configuration(
    tmp_path: Path,
) -> None:
    import lrp.cli.predict as cli

    decision_path = (
        tmp_path
        / "champion_decision.json"
    )

    decision_path.write_text(
        json.dumps(
            {
                "selection": {
                    "selected_model": "combined",
                },
            }
        ),
        encoding="utf-8",
    )

    result = cli._resolve_production_configuration(
        champion_decision=decision_path,
        production_snapshot_root=(
            tmp_path
            / "snapshots"
        ),
    )

    assert result is not None
    assert result.requested_model == "combined"
    assert result.resolved_model == "combined"

    assert result.pipeline_kwargs() == {
        "regime_calibration_snapshot_root": (
            tmp_path
            / "snapshots"
            / "regime-calibration"
        ),
        "regime_bayesian_snapshot_root": (
            tmp_path
            / "snapshots"
            / "regime-bayesian"
        ),
    }


def test_resolver_preserves_baseline_fallback(
    tmp_path: Path,
) -> None:
    import lrp.cli.predict as cli

    decision_path = (
        tmp_path
        / "champion_decision.json"
    )

    decision_path.write_text(
        json.dumps(
            {
                "selection": {
                    "selected_model": None,
                },
            }
        ),
        encoding="utf-8",
    )

    result = cli._resolve_production_configuration(
        champion_decision=decision_path,
        production_snapshot_root=(
            tmp_path
            / "snapshots"
        ),
    )

    assert result is not None
    assert result.resolved_model == "baseline"
    assert result.fallback_applied is True

    assert result.pipeline_kwargs() == {
        "regime_calibration_snapshot_root": None,
        "regime_bayesian_snapshot_root": None,
    }
