from __future__ import annotations

from pathlib import Path


def test_audit_imports_active_snapshot_reader() -> None:
    source = Path(
        "lrp/production/champion_audit.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    assert (
        "ProductionChampionActiveSnapshotReader"
        in source
    )


def test_audit_uses_active_snapshot_reader() -> None:
    source = Path(
        "lrp/production/champion_audit.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    assert (
        "ProductionChampionActiveSnapshotReader("
        in source
    )

    assert ".read(" in source


def test_audit_no_longer_uses_registry_reader_for_active_pair() -> None:
    source = Path(
        "lrp/production/champion_audit.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    assert (
        "ProductionChampionRegistryReader("
        not in source
    )


def test_audit_does_not_reread_active_decision_bytes() -> None:
    source = Path(
        "lrp/production/champion_audit.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        "champion_decision.json"
        ").read_bytes()"
    )

    assert forbidden not in source
