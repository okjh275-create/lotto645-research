from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lrp.production.champion_rollback_history import (
    ChampionRollbackHistoryReader,
)


MODULE_NAME = (
    "lrp.production.champion_rollback"
)


def _load_api():
    module = __import__(
        MODULE_NAME,
        fromlist=[
            "ChampionRollbackPlan",
            "ChampionRollbackResult",
            "ChampionRollbackService",
        ],
    )

    return (
        module.ChampionRollbackPlan,
        module.ChampionRollbackResult,
        module.ChampionRollbackService,
    )


def _decision_bytes(
    model: str,
) -> bytes:
    return (
        json.dumps(
            {
                "selection": {
                    "selected_model": model,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    ).encode(
        "utf-8"
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


def _seed_revision(
    registry: Path,
    *,
    model: str,
    timestamp: str,
) -> tuple[str, str]:
    decision = _decision_bytes(
        model
    )

    source_sha256 = hashlib.sha256(
        decision
    ).hexdigest()

    decision_path = (
        registry
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
            timestamp,
        "selected_model":
            model,
        "source_path":
            f"/fixture/{model}.json",
        "source_sha256":
            source_sha256,
        "published_path":
            str(
                registry
                / "active"
                / "champion_decision.json"
            ),
    }

    raw = (
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
        raw
    ).hexdigest()

    (
        registry
        / "history"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        registry
        / "history"
        / f"{revision_id}.json"
    ).write_bytes(
        raw
    )

    return (
        revision_id,
        source_sha256,
    )


def _write_active(
    registry: Path,
    *,
    model: str,
    timestamp: str,
) -> str:
    decision = _decision_bytes(
        model
    )

    source_sha256 = hashlib.sha256(
        decision
    ).hexdigest()

    active_root = (
        registry
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
                timestamp,
            "selected_model":
                model,
            "source_path":
                f"/fixture/{model}.json",
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


def _file_snapshot(
    root: Path,
) -> dict[str, bytes]:
    return {
        path.relative_to(
            root
        ).as_posix():
        path.read_bytes()
        for path in root.rglob(
            "*"
        )
        if path.is_file()
    }


def test_plan_is_read_only_and_captures_current_active_identity(
    tmp_path: Path,
) -> None:
    Plan, _, Service = _load_api()

    registry = (
        tmp_path
        / "registry"
    )

    target_revision, target_sha = (
        _seed_revision(
            registry,
            model="model-a",
            timestamp=(
                "2026-08-18T10:00:00+09:00"
            ),
        )
    )

    current_sha = _write_active(
        registry,
        model="model-b",
        timestamp=(
            "2026-08-18T11:00:00+09:00"
        ),
    )

    before = _file_snapshot(
        registry
    )

    service = Service(
        registry_root=registry
    )

    plan = service.plan(
        target_revision
    )

    after = _file_snapshot(
        registry
    )

    assert after == before

    assert isinstance(
        plan,
        Plan,
    )

    assert (
        plan.target_revision_id
        == target_revision
    )

    assert (
        plan.target_source_sha256
        == target_sha
    )

    assert (
        plan.active_source_sha256
        == current_sha
    )

    assert (
        plan.target_selected_model
        == "model-a"
    )


def test_plan_rejects_current_active_target(
    tmp_path: Path,
) -> None:
    _, _, Service = _load_api()

    registry = (
        tmp_path
        / "registry"
    )

    revision_id, _ = _seed_revision(
        registry,
        model="model-a",
        timestamp=(
            "2026-08-18T10:00:00+09:00"
        ),
    )

    _write_active(
        registry,
        model="model-a",
        timestamp=(
            "2026-08-18T11:00:00+09:00"
        ),
    )

    service = Service(
        registry_root=registry
    )

    with pytest.raises(
        ValueError,
        match="active",
    ):
        service.plan(
            revision_id
        )


def test_execute_restores_target_decision_bytes(
    tmp_path: Path,
) -> None:
    _, Result, Service = _load_api()

    registry = (
        tmp_path
        / "registry"
    )

    target_revision, target_sha = (
        _seed_revision(
            registry,
            model="model-a",
            timestamp=(
                "2026-08-18T10:00:00+09:00"
            ),
        )
    )

    _write_active(
        registry,
        model="model-b",
        timestamp=(
            "2026-08-18T11:00:00+09:00"
        ),
    )

    service = Service(
        registry_root=registry
    )

    plan = service.plan(
        target_revision
    )

    result = service.execute(
        plan
    )

    assert isinstance(
        result,
        Result,
    )

    assert result.status == "PASS"

    active_decision = (
        registry
        / "active"
        / "champion_decision.json"
    )

    actual_sha = hashlib.sha256(
        active_decision.read_bytes()
    ).hexdigest()

    assert actual_sha == target_sha

    payload = json.loads(
        active_decision.read_text(
            encoding="utf-8"
        )
    )

    assert (
        payload[
            "selection"
        ][
            "selected_model"
        ]
        == "model-a"
    )


def test_execute_updates_active_publication_to_target(
    tmp_path: Path,
) -> None:
    _, _, Service = _load_api()

    registry = (
        tmp_path
        / "registry"
    )

    target_revision, target_sha = (
        _seed_revision(
            registry,
            model="model-a",
            timestamp=(
                "2026-08-18T10:00:00+09:00"
            ),
        )
    )

    _write_active(
        registry,
        model="model-b",
        timestamp=(
            "2026-08-18T11:00:00+09:00"
        ),
    )

    service = Service(
        registry_root=registry
    )

    result = service.execute(
        service.plan(
            target_revision
        )
    )

    active_publication = json.loads(
        (
            registry
            / "active"
            / "publication.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        active_publication[
            "source_sha256"
        ]
        == target_sha
    )

    assert (
        active_publication[
            "selected_model"
        ]
        == "model-a"
    )

    assert (
        result.target_revision_id
        == target_revision
    )


def test_execute_rejects_stale_plan_when_active_changes(
    tmp_path: Path,
) -> None:
    _, _, Service = _load_api()

    registry = (
        tmp_path
        / "registry"
    )

    target_revision, _ = _seed_revision(
        registry,
        model="model-a",
        timestamp=(
            "2026-08-18T10:00:00+09:00"
        ),
    )

    _write_active(
        registry,
        model="model-b",
        timestamp=(
            "2026-08-18T11:00:00+09:00"
        ),
    )

    service = Service(
        registry_root=registry
    )

    plan = service.plan(
        target_revision
    )

    _write_active(
        registry,
        model="model-c",
        timestamp=(
            "2026-08-18T12:00:00+09:00"
        ),
    )

    with pytest.raises(
        ValueError,
        match="stale",
    ):
        service.execute(
            plan
        )


def test_execute_revalidates_target_snapshot_before_write(
    tmp_path: Path,
) -> None:
    _, _, Service = _load_api()

    registry = (
        tmp_path
        / "registry"
    )

    target_revision, target_sha = (
        _seed_revision(
            registry,
            model="model-a",
            timestamp=(
                "2026-08-18T10:00:00+09:00"
            ),
        )
    )

    _write_active(
        registry,
        model="model-b",
        timestamp=(
            "2026-08-18T11:00:00+09:00"
        ),
    )

    service = Service(
        registry_root=registry
    )

    plan = service.plan(
        target_revision
    )

    (
        registry
        / "history"
        / "decisions"
        / f"{target_sha}.json"
    ).write_bytes(
        b'{"tampered": true}\n'
    )

    with pytest.raises(
        ValueError,
        match="sha256",
    ):
        service.execute(
            plan
        )


def test_failed_execute_does_not_partially_modify_active_state(
    tmp_path: Path,
) -> None:
    _, _, Service = _load_api()

    registry = (
        tmp_path
        / "registry"
    )

    target_revision, target_sha = (
        _seed_revision(
            registry,
            model="model-a",
            timestamp=(
                "2026-08-18T10:00:00+09:00"
            ),
        )
    )

    _write_active(
        registry,
        model="model-b",
        timestamp=(
            "2026-08-18T11:00:00+09:00"
        ),
    )

    service = Service(
        registry_root=registry
    )

    plan = service.plan(
        target_revision
    )

    before = _file_snapshot(
        registry
    )

    (
        registry
        / "history"
        / "decisions"
        / f"{target_sha}.json"
    ).write_bytes(
        b'{"tampered": true}\n'
    )

    active_before = {
        key: value
        for key, value
        in before.items()
        if key.startswith(
            "active/"
        )
    }

    with pytest.raises(
        ValueError
    ):
        service.execute(
            plan
        )

    active_after = {
        path.relative_to(
            registry
        ).as_posix():
        path.read_bytes()
        for path in (
            registry
            / "active"
        ).rglob(
            "*"
        )
        if path.is_file()
    }

    assert active_after == (
        active_before
    )


def test_execute_does_not_mutate_history(
    tmp_path: Path,
) -> None:
    _, _, Service = _load_api()

    registry = (
        tmp_path
        / "registry"
    )

    target_revision, _ = _seed_revision(
        registry,
        model="model-a",
        timestamp=(
            "2026-08-18T10:00:00+09:00"
        ),
    )

    _write_active(
        registry,
        model="model-b",
        timestamp=(
            "2026-08-18T11:00:00+09:00"
        ),
    )

    history_before = {
        path.relative_to(
            registry
        ).as_posix():
        path.read_bytes()
        for path in (
            registry
            / "history"
        ).rglob(
            "*"
        )
        if (
            path.is_file()
            and "rollbacks"
            not in path.parts
        )
    }

    service = Service(
        registry_root=registry
    )

    service.execute(
        service.plan(
            target_revision
        )
    )

    history_after = {
        path.relative_to(
            registry
        ).as_posix():
        path.read_bytes()
        for path in (
            registry
            / "history"
        ).rglob(
            "*"
        )
        if (
            path.is_file()
            and "rollbacks"
            not in path.parts
        )
    }

    assert history_after == (
        history_before
    )
