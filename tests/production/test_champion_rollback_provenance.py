from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lrp.production.champion_registry_publisher import (
    ProductionChampionRegistryPublisher,
)
from lrp.production.champion_rollback import (
    ChampionRollbackService,
)


def _write_decision(
    path: Path,
    *,
    selected_model: str,
) -> Path:
    payload = {
        "selection": {
            "selected_model": selected_model,
        },
    }

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return path


def _publish(
    registry_root: Path,
    source: Path,
):
    return (
        ProductionChampionRegistryPublisher()
        .publish(
            source_decision=source,
            registry_root=registry_root,
        )
    )


def _publication_revision_id(
    registry_root: Path,
    *,
    source_sha256: str,
) -> str:
    history_root = (
        registry_root
        / "history"
    )

    matches = []

    for path in history_root.glob(
        "*.json"
    ):
        if not path.is_file():
            continue

        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if (
            payload.get(
                "source_sha256"
            )
            == source_sha256
        ):
            matches.append(
                path.stem
            )

    if len(matches) != 1:
        raise AssertionError(
            "expected exactly one "
            "publication revision "
            f"for {source_sha256}; "
            f"found {len(matches)}"
        )

    return matches[0]


def _rollback_records(
    registry_root: Path,
) -> list[Path]:
    root = (
        registry_root
        / "history"
        / "rollbacks"
    )

    if not root.exists():
        return []

    return sorted(
        path
        for path in root.glob("*.json")
        if path.is_file()
    )


def _snapshot_existing_history(
    registry_root: Path,
) -> dict[str, bytes]:
    history_root = registry_root / "history"

    if not history_root.exists():
        return {}

    return {
        str(path.relative_to(history_root)):
            path.read_bytes()
        for path in history_root.rglob("*")
        if (
            path.is_file()
            and "rollbacks"
            not in path.parts
        )
    }


def _prepare_registry(
    tmp_path: Path,
):
    registry_root = tmp_path / "registry"

    source_a = _write_decision(
        tmp_path / "decision-a.json",
        selected_model="model-a",
    )
    source_b = _write_decision(
        tmp_path / "decision-b.json",
        selected_model="model-b",
    )

    result_a = _publish(
        registry_root,
        source_a,
    )
    result_b = _publish(
        registry_root,
        source_b,
    )

    return (
        registry_root,
        source_a,
        source_b,
        result_a,
        result_b,
    )


def test_successful_rollback_persists_provenance_record(
    tmp_path: Path,
) -> None:
    (
        registry_root,
        _,
        _,
        result_a,
        result_b,
    ) = _prepare_registry(tmp_path)

    service = ChampionRollbackService(
        registry_root=registry_root,
    )

    plan = service.plan(
        _publication_revision_id(
            registry_root,
            source_sha256=(
                result_a.source_sha256
            ),
        )
    )

    service.execute(plan)

    records = _rollback_records(
        registry_root
    )

    assert len(records) == 1

    payload = json.loads(
        records[0].read_text(
            encoding="utf-8"
        )
    )

    assert (
        payload["from_source_sha256"]
        == result_b.source_sha256
    )
    assert (
        payload["to_source_sha256"]
        == result_a.source_sha256
    )
    assert (
        payload["target_revision_id"]
        == _publication_revision_id(
            registry_root,
            source_sha256=(
                result_a.source_sha256
            ),
        )
    )
    assert (
        payload["selected_model"]
        == "model-a"
    )

    assert isinstance(
        payload["executed_at"],
        str,
    )
    assert payload["executed_at"]


def test_rollback_provenance_is_immutable_and_not_overwritten(
    tmp_path: Path,
) -> None:
    (
        registry_root,
        _,
        _,
        result_a,
        result_b,
    ) = _prepare_registry(tmp_path)

    service = ChampionRollbackService(
        registry_root=registry_root,
    )

    first_plan = service.plan(
        _publication_revision_id(
            registry_root,
            source_sha256=(
                result_a.source_sha256
            ),
        )
    )
    service.execute(first_plan)

    first_records = _rollback_records(
        registry_root
    )

    assert len(first_records) == 1

    first_path = first_records[0]
    first_bytes = first_path.read_bytes()

    second_plan = service.plan(
        _publication_revision_id(
            registry_root,
            source_sha256=(
                result_b.source_sha256
            ),
        )
    )
    service.execute(second_plan)

    records = _rollback_records(
        registry_root
    )

    assert len(records) == 2
    assert first_path.exists()
    assert first_path.read_bytes() == first_bytes
    assert records[0] != records[1]


def test_failed_stale_rollback_does_not_persist_success_record(
    tmp_path: Path,
) -> None:
    (
        registry_root,
        _,
        _,
        result_a,
        _,
    ) = _prepare_registry(tmp_path)

    service = ChampionRollbackService(
        registry_root=registry_root,
    )

    stale_plan = service.plan(
        _publication_revision_id(
            registry_root,
            source_sha256=(
                result_a.source_sha256
            ),
        )
    )

    source_c = _write_decision(
        tmp_path / "decision-c.json",
        selected_model="model-c",
    )

    _publish(
        registry_root,
        source_c,
    )

    before = _rollback_records(
        registry_root
    )

    with pytest.raises(
        ValueError,
        match="stale rollback plan",
    ):
        service.execute(
            stale_plan
        )

    after = _rollback_records(
        registry_root
    )

    assert after == before


def test_rollback_provenance_preserves_existing_history_bytes(
    tmp_path: Path,
) -> None:
    (
        registry_root,
        _,
        _,
        result_a,
        _,
    ) = _prepare_registry(tmp_path)

    before = _snapshot_existing_history(
        registry_root
    )

    service = ChampionRollbackService(
        registry_root=registry_root,
    )

    plan = service.plan(
        _publication_revision_id(
            registry_root,
            source_sha256=(
                result_a.source_sha256
            ),
        )
    )
    service.execute(plan)

    after = _snapshot_existing_history(
        registry_root
    )

    assert after == before


def test_rollback_provenance_identity_matches_record_bytes(
    tmp_path: Path,
) -> None:
    (
        registry_root,
        _,
        _,
        result_a,
        _,
    ) = _prepare_registry(tmp_path)

    service = ChampionRollbackService(
        registry_root=registry_root,
    )

    plan = service.plan(
        _publication_revision_id(
            registry_root,
            source_sha256=(
                result_a.source_sha256
            ),
        )
    )
    service.execute(plan)

    records = _rollback_records(
        registry_root
    )

    assert len(records) == 1

    record = records[0]
    digest = hashlib.sha256(
        record.read_bytes()
    ).hexdigest()

    assert record.stem == digest
