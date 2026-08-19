from __future__ import annotations

import json
from pathlib import Path

import pytest

from lrp.production.champion_history_retention import (
    ChampionHistoryRetentionExecutor,
    ChampionHistoryRetentionPlanner,
    ChampionHistoryRetentionPolicy,
)
from lrp.production.champion_registry_publisher import (
    ProductionChampionRegistryPublisher,
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


def _publish_many(
    tmp_path: Path,
    count: int = 5,
) -> Path:

    registry = (
        tmp_path
        / "registry"
    )

    publisher = (
        ProductionChampionRegistryPublisher()
    )

    for index in range(count):

        publisher.publish(
            source_decision=(
                _write_decision(
                    tmp_path
                    / f"d{index}.json",
                    f"w10-model-{index}",
                )
            ),
            registry_root=registry,
        )

    return registry


def _protected_tree(
    registry: Path,
) -> dict[str, bytes]:

    return {
        path.relative_to(
            registry
        ).as_posix():
            path.read_bytes()
        for path in sorted(
            registry.rglob("*")
        )
        if (
            path.is_file()
            and path.name != ".writer.lock"
        )
    }


def _history_revision_files(
    registry: Path,
) -> set[str]:

    root = (
        registry
        / "history"
    )

    return {
        path.stem
        for path in root.glob("*.json")
        if (
            path.is_file()
            and len(path.stem) == 64
        )
    }


def _decision_files(
    registry: Path,
) -> set[str]:

    root = (
        registry
        / "history"
        / "decisions"
    )

    return {
        path.stem
        for path in root.glob("*.json")
        if (
            path.is_file()
            and len(path.stem) == 64
        )
    }


def test_first_unlink_failure_leaves_protected_state_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    registry = _publish_many(
        tmp_path,
        count=5,
    )

    plan = (
        ChampionHistoryRetentionPlanner(
            registry
        )
        .plan(
            ChampionHistoryRetentionPolicy(
                keep_recent=2
            )
        )
    )

    before = _protected_tree(
        registry
    )

    original_unlink = (
        Path.unlink
    )

    calls = 0

    def fail_first(
        self: Path,
        *args,
        **kwargs,
    ):
        nonlocal calls

        calls += 1

        if calls == 1:
            raise OSError(
                "injected first unlink failure"
            )

        return original_unlink(
            self,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        Path,
        "unlink",
        fail_first,
    )

    with pytest.raises(
        OSError,
        match="injected first unlink failure",
    ):
        ChampionHistoryRetentionExecutor(
            registry
        ).execute(
            plan
        )

    after = _protected_tree(
        registry
    )

    assert after == before



def test_second_revision_unlink_failure_is_compensated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    registry = _publish_many(
        tmp_path,
        count=5,
    )

    plan = (
        ChampionHistoryRetentionPlanner(
            registry
        )
        .plan(
            ChampionHistoryRetentionPolicy(
                keep_recent=2
            )
        )
    )

    assert (
        len(
            plan.prunable_revision_ids
        )
        >= 2
    )

    before_revisions = (
        _history_revision_files(
            registry
        )
    )

    original_unlink = (
        Path.unlink
    )

    revision_delete_calls = 0

    def fail_second_revision(
        self: Path,
        *args,
        **kwargs,
    ):
        nonlocal revision_delete_calls

        is_revision = (
            self.parent.name
            == "history"
            and self.suffix == ".json"
            and len(self.stem) == 64
        )

        if is_revision:
            revision_delete_calls += 1

            if revision_delete_calls == 2:
                raise OSError(
                    "injected second revision failure"
                )

        return original_unlink(
            self,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        Path,
        "unlink",
        fail_second_revision,
    )

    with pytest.raises(
        OSError,
        match="injected second revision failure",
    ):
        ChampionHistoryRetentionExecutor(
            registry
        ).execute(
            plan
        )

    after_revisions = (
        _history_revision_files(
            registry
        )
    )

    deleted = (
        before_revisions
        - after_revisions
    )

    print(
        "REVISION FILES MISSING AFTER COMPENSATION:",
        sorted(deleted),
    )

    # The injected mid-revision failure occurs after
    # at least one successful unlink, but W-10C
    # compensation restores every prior deletion.
    assert deleted == set()



def test_decision_unlink_failure_restores_revision_deletions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    registry = _publish_many(
        tmp_path,
        count=5,
    )

    plan = (
        ChampionHistoryRetentionPlanner(
            registry
        )
        .plan(
            ChampionHistoryRetentionPolicy(
                keep_recent=2
            )
        )
    )

    assert (
        plan.prunable_revision_ids
    )

    assert (
        plan.prunable_decision_sha256s
    )

    before_revisions = (
        _history_revision_files(
            registry
        )
    )

    before_decisions = (
        _decision_files(
            registry
        )
    )

    original_unlink = (
        Path.unlink
    )

    decision_calls = 0

    def fail_first_decision(
        self: Path,
        *args,
        **kwargs,
    ):
        nonlocal decision_calls

        is_decision = (
            self.parent.name
            == "decisions"
            and self.suffix == ".json"
            and len(self.stem) == 64
        )

        if is_decision:
            decision_calls += 1

            if decision_calls == 1:
                raise OSError(
                    "injected decision delete failure"
                )

        return original_unlink(
            self,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        Path,
        "unlink",
        fail_first_decision,
    )

    with pytest.raises(
        OSError,
        match="injected decision delete failure",
    ):
        ChampionHistoryRetentionExecutor(
            registry
        ).execute(
            plan
        )

    after_revisions = (
        _history_revision_files(
            registry
        )
    )

    after_decisions = (
        _decision_files(
            registry
        )
    )

    deleted_revisions = (
        before_revisions
        - after_revisions
    )

    deleted_decisions = (
        before_decisions
        - after_decisions
    )

    print(
        "REVISION FILES MISSING AFTER COMPENSATION:",
        sorted(
            deleted_revisions
        ),
    )

    print(
        "DECISION FILES MISSING AFTER COMPENSATION:",
        sorted(
            deleted_decisions
        ),
    )

    # All publication revisions had already been
    # unlinked before the injected decision failure.
    # Compensation must restore them, and no decision
    # snapshot may remain deleted.
    assert deleted_revisions == set()
    assert deleted_decisions == set()


def test_partial_failure_does_not_mutate_active_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    registry = _publish_many(
        tmp_path,
        count=5,
    )

    plan = (
        ChampionHistoryRetentionPlanner(
            registry
        )
        .plan(
            ChampionHistoryRetentionPolicy(
                keep_recent=2
            )
        )
    )

    active_decision = (
        registry
        / "active"
        / "champion_decision.json"
    )

    active_publication = (
        registry
        / "active"
        / "publication.json"
    )

    decision_before = (
        active_decision.read_bytes()
    )

    publication_before = (
        active_publication.read_bytes()
    )

    original_unlink = (
        Path.unlink
    )

    calls = 0

    def fail_second(
        self: Path,
        *args,
        **kwargs,
    ):
        nonlocal calls

        calls += 1

        if calls == 2:
            raise OSError(
                "injected partial failure"
            )

        return original_unlink(
            self,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        Path,
        "unlink",
        fail_second,
    )

    with pytest.raises(
        OSError,
        match="injected partial failure",
    ):
        ChampionHistoryRetentionExecutor(
            registry
        ).execute(
            plan
        )

    assert (
        active_decision.read_bytes()
        == decision_before
    )

    assert (
        active_publication.read_bytes()
        == publication_before
    )


def test_partial_failure_preserves_rollback_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    registry = _publish_many(
        tmp_path,
        count=5,
    )

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
        / "audit.json"
    )

    provenance.write_text(
        '{"preserve": true}\n',
        encoding="utf-8",
    )

    before = provenance.read_bytes()

    plan = (
        ChampionHistoryRetentionPlanner(
            registry
        )
        .plan(
            ChampionHistoryRetentionPolicy(
                keep_recent=2
            )
        )
    )

    original_unlink = (
        Path.unlink
    )

    calls = 0

    def fail_second(
        self: Path,
        *args,
        **kwargs,
    ):
        nonlocal calls

        calls += 1

        if calls == 2:
            raise OSError(
                "injected partial failure"
            )

        return original_unlink(
            self,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        Path,
        "unlink",
        fail_second,
    )

    with pytest.raises(
        OSError,
        match="injected partial failure",
    ):
        ChampionHistoryRetentionExecutor(
            registry
        ).execute(
            plan
        )

    assert provenance.read_bytes() == before


def test_lock_releases_after_mid_delete_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    from lrp.production.production_registry_lock import (
        ProductionRegistryWriterLock,
    )

    registry = _publish_many(
        tmp_path,
        count=5,
    )

    plan = (
        ChampionHistoryRetentionPlanner(
            registry
        )
        .plan(
            ChampionHistoryRetentionPolicy(
                keep_recent=2
            )
        )
    )

    original_unlink = (
        Path.unlink
    )

    calls = 0

    def fail_second(
        self: Path,
        *args,
        **kwargs,
    ):
        nonlocal calls

        calls += 1

        if calls == 2:
            raise OSError(
                "injected lock release failure"
            )

        return original_unlink(
            self,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        Path,
        "unlink",
        fail_second,
    )

    with pytest.raises(
        OSError,
        match="injected lock release failure",
    ):
        ChampionHistoryRetentionExecutor(
            registry
        ).execute(
            plan
        )

    # Restore unlink before acquiring the lock again.
    monkeypatch.setattr(
        Path,
        "unlink",
        original_unlink,
    )

    with ProductionRegistryWriterLock(
        registry,
        timeout=1.0,
    ):
        pass
