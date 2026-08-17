from __future__ import annotations

import pytest

import lrp.cli


def test_root_commands_include_audit_champion() -> None:
    assert (
        "audit-champion"
        in lrp.cli._COMMANDS
    )


def test_root_help_lists_audit_champion(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(
        SystemExit,
    ) as exc_info:
        lrp.cli.main(
            [
                "audit-champion",
                "--help",
            ]
        )

    assert exc_info.value.code == 0

    captured = capsys.readouterr()

    assert (
        "audit-champion"
        in captured.out
    )
