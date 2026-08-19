from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading

import pytest

from lrp.production.champion_registry_publisher import (
    ProductionChampionRegistryPublisher,
)
from lrp.production.production_registry_lock import (
    ProductionRegistryWriterLock,
)


def _load_api():
    from lrp.production.champion_history_retention import (
        ChampionHistoryRetentionExecutor,
        ChampionHistoryRetentionPlanner,
        ChampionHistoryRetentionPolicy,
        ChampionHistoryRetentionResult,
    )

    return (
        ChampionHistoryRetentionPolicy,
        ChampionHistoryRetentionPlanner,
        ChampionHistoryRetentionExecutor,
        ChampionHistoryRetentionResult,
    )


def _write_decision(
    path: Path,
    model: str,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "selection": {
                    "selected_model": model,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return path


def _publish(
    root: Path,
    model: str,
):
    source = _write_decision(
        root / f"{model}.json",
        model,
    )

    return (
        ProductionChampionRegistryPublisher()
        .publish(
            source_decision=source,
            registry_root=root / "registry",
        )
    )


def _tree(
    root: Path,
) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix():
            path.read_bytes()
        for path in sorted(
            root.rglob("*")
        )
        if (
            path.is_file()
            and path.name != ".writer.lock"
        )
    }


def test_executor_requires_exact_plan_type(
    tmp_path: Path,
) -> None:
    (
        _,
        _,
        Executor,
        _,
    ) = _load_api()

    executor = Executor(
        registry_root=tmp_path,
    )

    with pytest.raises(
        (TypeError, ValueError),
    ):
        executor.execute(
            object()
        )


def test_executor_deletes_only_prunable_publication_revisions(
    tmp_path: Path,
) -> None:
    (
        Policy,
        Planner,
        Executor,
        _,
    ) = _load_api()

    for index in range(1, 5):
        _publish(
            tmp_path,
            f"model-{index}",
        )

    registry = tmp_path / "registry"

    plan = Planner(
        registry
    ).plan(
        Policy(
            keep_recent=2
        )
    )

    retained = set(
        plan.retained_revision_ids
    )

    prunable = set(
        plan.prunable_revision_ids
    )

    result = Executor(
        registry
    ).execute(
        plan
    )

    history = registry / "history"

    remaining = {
        path.stem
        for path in history.glob(
            "*.json"
        )
        if path.is_file()
        and len(path.stem) == 64
    }

    assert retained <= remaining
    assert prunable.isdisjoint(
        remaining
    )

    assert set(
        result.deleted_revision_ids
    ) == prunable


def test_executor_deletes_only_zero_retained_reference_decisions(
    tmp_path: Path,
) -> None:
    (
        Policy,
        Planner,
        Executor,
        _,
    ) = _load_api()

    for index in range(1, 5):
        _publish(
            tmp_path,
            f"decision-{index}",
        )

    registry = tmp_path / "registry"

    plan = Planner(
        registry
    ).plan(
        Policy(
            keep_recent=2
        )
    )

    retained = set(
        plan.retained_decision_sha256s
    )

    prunable = set(
        plan.prunable_decision_sha256s
    )

    result = Executor(
        registry
    ).execute(
        plan
    )

    decision_root = (
        registry
        / "history"
        / "decisions"
    )

    remaining = {
        path.stem
        for path in decision_root.glob(
            "*.json"
        )
        if path.is_file()
        and len(path.stem) == 64
    }

    assert retained <= remaining
    assert prunable.isdisjoint(
        remaining
    )

    assert set(
        result.deleted_decision_sha256s
    ) == prunable


def test_executor_preserves_active_pair(
    tmp_path: Path,
) -> None:
    (
        Policy,
        Planner,
        Executor,
        _,
    ) = _load_api()

    for index in range(1, 4):
        _publish(
            tmp_path,
            f"active-{index}",
        )

    registry = tmp_path / "registry"

    active_root = (
        registry / "active"
    )

    before_decision = (
        active_root
        / "champion_decision.json"
    ).read_bytes()

    before_publication = (
        active_root
        / "publication.json"
    ).read_bytes()

    plan = Planner(
        registry
    ).plan(
        Policy(
            keep_recent=1
        )
    )

    Executor(
        registry
    ).execute(
        plan
    )

    assert (
        active_root
        / "champion_decision.json"
    ).read_bytes() == before_decision

    assert (
        active_root
        / "publication.json"
    ).read_bytes() == before_publication


def test_executor_preserves_rollback_provenance(
    tmp_path: Path,
) -> None:
    (
        Policy,
        Planner,
        Executor,
        _,
    ) = _load_api()

    for index in range(1, 4):
        _publish(
            tmp_path,
            f"rollback-{index}",
        )

    registry = tmp_path / "registry"

    rollback_root = (
        registry
        / "history"
        / "rollbacks"
    )

    rollback_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    provenance = (
        rollback_root
        / "provenance.json"
    )

    provenance.write_text(
        '{"preserve": true}\n',
        encoding="utf-8",
    )

    before = provenance.read_bytes()

    plan = Planner(
        registry
    ).plan(
        Policy(
            keep_recent=1
        )
    )

    Executor(
        registry
    ).execute(
        plan
    )

    assert provenance.read_bytes() == before


def test_executor_preserves_unknown_history_files(
    tmp_path: Path,
) -> None:
    (
        Policy,
        Planner,
        Executor,
        _,
    ) = _load_api()

    for index in range(1, 4):
        _publish(
            tmp_path,
            f"unknown-{index}",
        )

    registry = tmp_path / "registry"

    unknown = (
        registry
        / "history"
        / "operator-note.txt"
    )

    unknown.write_text(
        "preserve\n",
        encoding="utf-8",
    )

    before = unknown.read_bytes()

    plan = Planner(
        registry
    ).plan(
        Policy(
            keep_recent=1
        )
    )

    Executor(
        registry
    ).execute(
        plan
    )

    assert unknown.read_bytes() == before


def test_executor_rejects_stale_plan_fail_closed(
    tmp_path: Path,
) -> None:
    (
        Policy,
        Planner,
        Executor,
        _,
    ) = _load_api()

    for index in range(1, 4):
        _publish(
            tmp_path,
            f"stale-{index}",
        )

    registry = tmp_path / "registry"

    plan = Planner(
        registry
    ).plan(
        Policy(
            keep_recent=1
        )
    )

    # Change the active generation after planning.
    _publish(
        tmp_path,
        "stale-new-generation",
    )

    before = _tree(
        registry
    )

    with pytest.raises(
        (ValueError, RuntimeError),
    ):
        Executor(
            registry
        ).execute(
            plan
        )

    after = _tree(
        registry
    )

    assert after == before


def test_executor_uses_registry_writer_lock(
    tmp_path: Path,
) -> None:
    (
        Policy,
        Planner,
        Executor,
        _,
    ) = _load_api()

    for index in range(1, 4):
        _publish(
            tmp_path,
            f"lock-{index}",
        )

    registry = tmp_path / "registry"

    plan = Planner(
        registry
    ).plan(
        Policy(
            keep_recent=1
        )
    )

    entered = threading.Event()
    finished = threading.Event()
    error: list[BaseException] = []

    def run() -> None:
        entered.set()

        try:
            Executor(
                registry
            ).execute(
                plan
            )
        except BaseException as exc:
            error.append(exc)
        finally:
            finished.set()

    with ProductionRegistryWriterLock(
        registry
    ):
        thread = threading.Thread(
            target=run,
            daemon=True,
        )

        thread.start()

        assert entered.wait(
            timeout=2.0
        )

        assert not finished.wait(
            timeout=0.25
        )

    assert finished.wait(
        timeout=5.0
    )

    thread.join(
        timeout=5.0
    )

    assert not thread.is_alive()
    assert error == []


def test_executor_revalidates_plan_inside_writer_lock(
    tmp_path: Path,
) -> None:
    (
        Policy,
        Planner,
        Executor,
        _,
    ) = _load_api()

    for index in range(1, 4):
        _publish(
            tmp_path,
            f"revalidate-{index}",
        )

    registry = tmp_path / "registry"

    plan = Planner(
        registry
    ).plan(
        Policy(
            keep_recent=1
        )
    )

    # This represents a plan that can no longer be trusted
    # because the registry generation has moved.
    _publish(
        tmp_path,
        "revalidate-new",
    )

    before = _tree(
        registry
    )

    with pytest.raises(
        (ValueError, RuntimeError),
    ):
        Executor(
            registry
        ).execute(
            plan
        )

    assert _tree(
        registry
    ) == before


def test_executor_fails_closed_before_deletion_if_planned_file_missing(
    tmp_path: Path,
) -> None:
    (
        Policy,
        Planner,
        Executor,
        _,
    ) = _load_api()

    for index in range(1, 5):
        _publish(
            tmp_path,
            f"missing-{index}",
        )

    registry = tmp_path / "registry"

    plan = Planner(
        registry
    ).plan(
        Policy(
            keep_recent=2
        )
    )

    victim = (
        registry
        / "history"
        / (
            plan.prunable_revision_ids[0]
            + ".json"
        )
    )

    victim.unlink()

    before = _tree(
        registry
    )

    with pytest.raises(
        (
            FileNotFoundError,
            ValueError,
            RuntimeError,
        ),
    ):
        Executor(
            registry
        ).execute(
            plan
        )

    after = _tree(
        registry
    )

    # Executor itself must not partially delete anything
    # after detecting the inconsistent precondition.
    assert after == before


def test_executor_result_is_deterministic(
    tmp_path: Path,
) -> None:
    (
        Policy,
        Planner,
        Executor,
        Result,
    ) = _load_api()

    for index in range(1, 5):
        _publish(
            tmp_path,
            f"result-{index}",
        )

    registry = tmp_path / "registry"

    plan = Planner(
        registry
    ).plan(
        Policy(
            keep_recent=2
        )
    )

    result = Executor(
        registry
    ).execute(
        plan
    )

    assert isinstance(
        result,
        Result,
    )

    assert tuple(
        sorted(
            result.deleted_revision_ids
        )
    ) == result.deleted_revision_ids

    assert tuple(
        sorted(
            result.deleted_decision_sha256s
        )
    ) == result.deleted_decision_sha256s


def test_executor_is_idempotent_only_via_fresh_plan(
    tmp_path: Path,
) -> None:
    (
        Policy,
        Planner,
        Executor,
        _,
    ) = _load_api()

    for index in range(1, 5):
        _publish(
            tmp_path,
            f"idempotent-{index}",
        )

    registry = tmp_path / "registry"

    policy = Policy(
        keep_recent=2
    )

    first_plan = Planner(
        registry
    ).plan(
        policy
    )

    first = Executor(
        registry
    ).execute(
        first_plan
    )

    assert (
        len(
            first.deleted_revision_ids
        )
        > 0
    )

    second_plan = Planner(
        registry
    ).plan(
        policy
    )

    second = Executor(
        registry
    ).execute(
        second_plan
    )

    assert (
        second.deleted_revision_ids
        == ()
    )

    assert (
        second.deleted_decision_sha256s
        == ()
    )


def test_executor_never_deletes_active_source_snapshot(
    tmp_path: Path,
) -> None:
    (
        Policy,
        Planner,
        Executor,
        _,
    ) = _load_api()

    for index in range(1, 5):
        _publish(
            tmp_path,
            f"source-{index}",
        )

    registry = tmp_path / "registry"

    plan = Planner(
        registry
    ).plan(
        Policy(
            keep_recent=1
        )
    )

    active_source = (
        plan.active_source_sha256
    )

    assert (
        active_source
        not in
        plan.prunable_decision_sha256s
    )

    Executor(
        registry
    ).execute(
        plan
    )

    active_snapshot = (
        registry
        / "history"
        / "decisions"
        / f"{active_source}.json"
    )

    assert active_snapshot.is_file()

    assert (
        hashlib.sha256(
            active_snapshot.read_bytes()
        ).hexdigest()
        == active_source
    )
