from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


MODULE_NAME = (
    "lrp.production.champion_rollback_history"
)


def _load_api():
    module = __import__(
        MODULE_NAME,
        fromlist=[
            "ChampionRollbackHistoryReader",
            "ChampionRollbackTarget",
        ],
    )

    return (
        module.ChampionRollbackHistoryReader,
        module.ChampionRollbackTarget,
    )


def _write_json(
    path: Path,
    payload: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _decision_bytes(
    selected_model: str,
) -> bytes:
    return (
        json.dumps(
            {
                "selection": {
                    "selected_model":
                    selected_model,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    ).encode(
        "utf-8"
    )


def _seed_revision(
    registry_root: Path,
    *,
    selected_model: str,
    published_at_kst: str,
) -> tuple[str, str]:
    decision = _decision_bytes(
        selected_model
    )

    source_sha256 = hashlib.sha256(
        decision
    ).hexdigest()

    decision_path = (
        registry_root
        / "history"
        / "decisions"
        / f"{source_sha256}.json"
    )

    decision_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    decision_path.write_bytes(
        decision
    )

    publication = {
        "published_at_kst":
            published_at_kst,
        "selected_model":
            selected_model,
        "source_path":
            f"/fixture/{selected_model}.json",
        "source_sha256":
            source_sha256,
        "published_path":
            str(
                registry_root
                / "active"
                / "champion_decision.json"
            ),
    }

    publication_bytes = (
        json.dumps(
            publication,
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    ).encode(
        "utf-8"
    )

    revision_id = hashlib.sha256(
        publication_bytes
    ).hexdigest()

    revision_path = (
        registry_root
        / "history"
        / f"{revision_id}.json"
    )

    revision_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    revision_path.write_bytes(
        publication_bytes
    )

    return (
        revision_id,
        source_sha256,
    )


def _write_active(
    registry_root: Path,
    *,
    selected_model: str,
) -> str:
    decision = _decision_bytes(
        selected_model
    )

    source_sha256 = hashlib.sha256(
        decision
    ).hexdigest()

    active_root = (
        registry_root
        / "active"
    )

    active_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        active_root
        / "champion_decision.json"
    ).write_bytes(
        decision
    )

    _write_json(
        active_root
        / "publication.json",
        {
            "published_at_kst":
                "2026-08-19T00:00:00+09:00",
            "selected_model":
                selected_model,
            "source_path":
                f"/fixture/{selected_model}.json",
            "source_sha256":
                source_sha256,
            "published_path":
                str(
                    active_root
                    / "champion_decision.json"
                ),
        },
    )

    return source_sha256


def test_reader_lists_valid_revisions_in_publish_order(
    tmp_path: Path,
) -> None:
    Reader, Target = _load_api()

    registry = (
        tmp_path
        / "registry"
    )

    first_id, first_sha = _seed_revision(
        registry,
        selected_model="model-a",
        published_at_kst=(
            "2026-08-18T10:00:00+09:00"
        ),
    )

    second_id, second_sha = _seed_revision(
        registry,
        selected_model="model-b",
        published_at_kst=(
            "2026-08-18T11:00:00+09:00"
        ),
    )

    reader = Reader(
        registry_root=registry
    )

    revisions = (
        reader.list_revisions()
    )

    assert [
        item.revision_id
        for item in revisions
    ] == [
        first_id,
        second_id,
    ]

    assert isinstance(
        revisions[0],
        Target,
    )

    assert (
        revisions[0].source_sha256
        == first_sha
    )

    assert (
        revisions[1].source_sha256
        == second_sha
    )


def test_resolve_returns_verified_decision_snapshot(
    tmp_path: Path,
) -> None:
    Reader, _ = _load_api()

    registry = (
        tmp_path
        / "registry"
    )

    revision_id, source_sha = (
        _seed_revision(
            registry,
            selected_model="model-a",
            published_at_kst=(
                "2026-08-18T10:00:00+09:00"
            ),
        )
    )

    reader = Reader(
        registry_root=registry
    )

    target = reader.resolve(
        revision_id
    )

    assert (
        target.revision_id
        == revision_id
    )

    assert (
        target.source_sha256
        == source_sha
    )

    assert (
        target.selected_model
        == "model-a"
    )

    assert target.decision_path.exists()

    assert (
        hashlib.sha256(
            target.decision_path
            .read_bytes()
        ).hexdigest()
        == source_sha
    )


def test_resolve_rejects_missing_revision(
    tmp_path: Path,
) -> None:
    Reader, _ = _load_api()

    reader = Reader(
        registry_root=(
            tmp_path
            / "registry"
        )
    )

    with pytest.raises(
        ValueError,
        match="revision",
    ):
        reader.resolve(
            "f" * 64
        )


def test_resolve_rejects_missing_decision_snapshot(
    tmp_path: Path,
) -> None:
    Reader, _ = _load_api()

    registry = (
        tmp_path
        / "registry"
    )

    revision_id, source_sha = (
        _seed_revision(
            registry,
            selected_model="model-a",
            published_at_kst=(
                "2026-08-18T10:00:00+09:00"
            ),
        )
    )

    (
        registry
        / "history"
        / "decisions"
        / f"{source_sha}.json"
    ).unlink()

    reader = Reader(
        registry_root=registry
    )

    with pytest.raises(
        ValueError,
        match="decision",
    ):
        reader.resolve(
            revision_id
        )


def test_resolve_rejects_corrupted_decision_snapshot(
    tmp_path: Path,
) -> None:
    Reader, _ = _load_api()

    registry = (
        tmp_path
        / "registry"
    )

    revision_id, source_sha = (
        _seed_revision(
            registry,
            selected_model="model-a",
            published_at_kst=(
                "2026-08-18T10:00:00+09:00"
            ),
        )
    )

    decision_path = (
        registry
        / "history"
        / "decisions"
        / f"{source_sha}.json"
    )

    decision_path.write_bytes(
        b'{"tampered": true}\n'
    )

    reader = Reader(
        registry_root=registry
    )

    with pytest.raises(
        ValueError,
        match="sha256",
    ):
        reader.resolve(
            revision_id
        )


def test_resolve_rejects_corrupted_publication_revision(
    tmp_path: Path,
) -> None:
    Reader, _ = _load_api()

    registry = (
        tmp_path
        / "registry"
    )

    revision_id, _ = _seed_revision(
        registry,
        selected_model="model-a",
        published_at_kst=(
            "2026-08-18T10:00:00+09:00"
        ),
    )

    revision_path = (
        registry
        / "history"
        / f"{revision_id}.json"
    )

    revision_path.write_bytes(
        b'{"tampered": true}\n'
    )

    reader = Reader(
        registry_root=registry
    )

    with pytest.raises(
        ValueError,
        match="revision",
    ):
        reader.resolve(
            revision_id
        )


def test_resolve_rejects_current_active_target(
    tmp_path: Path,
) -> None:
    Reader, _ = _load_api()

    registry = (
        tmp_path
        / "registry"
    )

    revision_id, _ = _seed_revision(
        registry,
        selected_model="model-a",
        published_at_kst=(
            "2026-08-18T10:00:00+09:00"
        ),
    )

    _write_active(
        registry,
        selected_model="model-a",
    )

    reader = Reader(
        registry_root=registry
    )

    with pytest.raises(
        ValueError,
        match="active",
    ):
        reader.resolve(
            revision_id,
            reject_active=True,
        )


def test_reader_does_not_mutate_registry(
    tmp_path: Path,
) -> None:
    Reader, _ = _load_api()

    registry = (
        tmp_path
        / "registry"
    )

    revision_id, _ = _seed_revision(
        registry,
        selected_model="model-a",
        published_at_kst=(
            "2026-08-18T10:00:00+09:00"
        ),
    )

    before = {
        path.relative_to(
            registry
        ).as_posix():
        path.read_bytes()
        for path in registry.rglob(
            "*"
        )
        if path.is_file()
    }

    reader = Reader(
        registry_root=registry
    )

    reader.list_revisions()
    reader.resolve(
        revision_id
    )

    after = {
        path.relative_to(
            registry
        ).as_posix():
        path.read_bytes()
        for path in registry.rglob(
            "*"
        )
        if path.is_file()
    }

    assert after == before
