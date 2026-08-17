from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def test_audit_champion_cli_module_exists() -> None:
    importlib.import_module(
        "lrp.cli.audit_champion"
    )


def test_parser_accepts_audit_arguments() -> None:
    import lrp.cli.audit_champion as cli

    parser = cli._parser()

    arguments = parser.parse_args(
        [
            "--production-registry",
            "registry",
            "--snapshot-root",
            "snapshots",
        ]
    )

    assert (
        arguments.production_registry
        == Path("registry")
    )

    assert (
        arguments.snapshot_root
        == Path("snapshots")
    )


def test_parser_requires_production_registry() -> None:
    import lrp.cli.audit_champion as cli

    parser = cli._parser()

    with pytest.raises(
        SystemExit,
    ):
        parser.parse_args(
            [
                "--snapshot-root",
                "snapshots",
            ]
        )


def test_parser_requires_snapshot_root() -> None:
    import lrp.cli.audit_champion as cli

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
