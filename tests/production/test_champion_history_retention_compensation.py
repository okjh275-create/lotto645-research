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


def _load_atomicity_error():
    from lrp.production.champion_history_retention import (
        ChampionHistoryRetentionAtomicityError,
    )

    return ChampionHistoryRetentionAtomicityError


def _write_decision(
    path: Path,
    model: str,
) -> Path:

    path.write_text(
        json.dumps(
            {
                "selection": {
                    "selected_model":
                        model,
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
                    f"w10b-model-{index}",
                )
            ),
            registry_root=registry,
        )

    return registry


def _plan(
    registry: Path,
):
    return (
        ChampionHistoryRetentionPlanner(
            registry
        )
        .plan(
            ChampionHistoryRetentionPolicy(
                keep_recent=2
            )
        )
    )


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


def test_mid_revision_delete_failure_restores_deleted_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    registry = _publish_many(
        tmp_path
    )

    plan = _plan(
        registry
    )

    assert (
        len(
            plan.prunable_revision_ids
        )
        >= 2
    )

    before = _protected_tree(
        registry
    )

    original_unlink = Path.unlink
    revision_calls = 0

    def fail_second_revision(
        self: Path,
        *args,
        **kwargs,
    ):
        nonlocal revision_calls

        is_revision = (
            self.parent.name == "history"
            and self.suffix == ".json"
            and len(self.stem) == 64
        )

        if is_revision:
            revision_calls += 1

            if revision_calls == 2:
                raise OSError(
                    "injected revision failure"
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
        match="injected revision failure",
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


def test_decision_delete_failure_restores_all_deleted_revisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    registry = _publish_many(
        tmp_path
    )

    plan = _plan(
        registry
    )

    before = _protected_tree(
        registry
    )

    original_unlink = Path.unlink
    decision_calls = 0

    def fail_first_decision(
        self: Path,
        *args,
        **kwargs,
    ):
        nonlocal decision_calls

        is_decision = (
            self.parent.name == "decisions"
            and self.suffix == ".json"
            and len(self.stem) == 64
        )

        if is_decision:
            decision_calls += 1

            if decision_calls == 1:
                raise OSError(
                    "injected decision failure"
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
        match="injected decision failure",
    ):
        ChampionHistoryRetentionExecutor(
            registry
        ).execute(
            plan
        )

    assert (
        _protected_tree(
            registry
        )
        == before
    )


def test_mid_decision_failure_restores_decisions_and_revisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    registry = _publish_many(
        tmp_path,
        count=6,
    )

    plan = _plan(
        registry
    )

    assert (
        len(
            plan.prunable_decision_sha256s
        )
        >= 2
    )

    before = _protected_tree(
        registry
    )

    original_unlink = Path.unlink
    decision_calls = 0

    def fail_second_decision(
        self: Path,
        *args,
        **kwargs,
    ):
        nonlocal decision_calls

        is_decision = (
            self.parent.name == "decisions"
            and self.suffix == ".json"
            and len(self.stem) == 64
        )

        if is_decision:
            decision_calls += 1

            if decision_calls == 2:
                raise OSError(
                    "injected second decision failure"
                )

        return original_unlink(
            self,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        Path,
        "unlink",
        fail_second_decision,
    )

    with pytest.raises(
        OSError,
        match="injected second decision failure",
    ):
        ChampionHistoryRetentionExecutor(
            registry
        ).execute(
            plan
        )

    assert (
        _protected_tree(
            registry
        )
        == before
    )


def test_compensation_restores_exact_original_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    registry = _publish_many(
        tmp_path
    )

    plan = _plan(
        registry
    )

    targets = {}

    for revision_id in (
        plan.prunable_revision_ids
    ):
        path = (
            registry
            / "history"
            / f"{revision_id}.json"
        )

        targets[
            path.relative_to(
                registry
            ).as_posix()
        ] = path.read_bytes()

    for source_sha in (
        plan.prunable_decision_sha256s
    ):
        path = (
            registry
            / "history"
            / "decisions"
            / f"{source_sha}.json"
        )

        targets[
            path.relative_to(
                registry
            ).as_posix()
        ] = path.read_bytes()

    original_unlink = Path.unlink
    calls = 0

    def fail_later(
        self: Path,
        *args,
        **kwargs,
    ):
        nonlocal calls

        calls += 1

        if calls == 2:
            raise OSError(
                "injected byte-identity failure"
            )

        return original_unlink(
            self,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        Path,
        "unlink",
        fail_later,
    )

    with pytest.raises(
        OSError,
        match="injected byte-identity failure",
    ):
        ChampionHistoryRetentionExecutor(
            registry
        ).execute(
            plan
        )

    for relative, expected in (
        targets.items()
    ):

        path = (
            registry
            / relative
        )

        assert path.is_file()

        assert (
            path.read_bytes()
            == expected
        )


def test_compensated_failure_leaves_protected_tree_byte_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    registry = _publish_many(
        tmp_path
    )

    plan = _plan(
        registry
    )

    before = _protected_tree(
        registry
    )

    original_unlink = Path.unlink
    calls = 0

    def fail_midway(
        self: Path,
        *args,
        **kwargs,
    ):
        nonlocal calls

        calls += 1

        if calls == 3:
            raise OSError(
                "injected protected-tree failure"
            )

        return original_unlink(
            self,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        Path,
        "unlink",
        fail_midway,
    )

    with pytest.raises(
        OSError,
        match="injected protected-tree failure",
    ):
        ChampionHistoryRetentionExecutor(
            registry
        ).execute(
            plan
        )

    assert (
        _protected_tree(
            registry
        )
        == before
    )


def test_compensation_runs_while_writer_lock_is_still_held(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    from lrp.production.production_registry_lock import (
        ProductionRegistryLockTimeout,
        ProductionRegistryWriterLock,
    )

    registry = _publish_many(
        tmp_path
    )

    plan = _plan(
        registry
    )

    original_unlink = Path.unlink
    calls = 0

    def fail_second(
        self: Path,
        *args,
        **kwargs,
    ):
        nonlocal calls

        calls += 1

        if calls == 2:

            with pytest.raises(
                ProductionRegistryLockTimeout
            ):
                with ProductionRegistryWriterLock(
                    registry,
                    timeout=0.0,
                ):
                    pass

            raise OSError(
                "injected locked compensation failure"
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
        match="injected locked compensation failure",
    ):
        ChampionHistoryRetentionExecutor(
            registry
        ).execute(
            plan
        )

    # Byte restoration succeeding implies the
    # compensation completed before execute exited.
    fresh = (
        ChampionHistoryRetentionPlanner(
            registry
        )
        .plan(
            ChampionHistoryRetentionPolicy(
                keep_recent=2
            )
        )
    )

    assert fresh == plan


def test_writer_lock_releases_after_successful_compensation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    from lrp.production.production_registry_lock import (
        ProductionRegistryWriterLock,
    )

    registry = _publish_many(
        tmp_path
    )

    plan = _plan(
        registry
    )

    original_unlink = Path.unlink
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
                "injected compensation lock release"
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
        match="injected compensation lock release",
    ):
        ChampionHistoryRetentionExecutor(
            registry
        ).execute(
            plan
        )

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


def test_active_pair_is_unchanged_during_compensated_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    registry = _publish_many(
        tmp_path
    )

    plan = _plan(
        registry
    )

    decision = (
        registry
        / "active"
        / "champion_decision.json"
    )

    publication = (
        registry
        / "active"
        / "publication.json"
    )

    decision_before = (
        decision.read_bytes()
    )

    publication_before = (
        publication.read_bytes()
    )

    original_unlink = Path.unlink
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
                "injected active-pair failure"
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
        match="injected active-pair failure",
    ):
        ChampionHistoryRetentionExecutor(
            registry
        ).execute(
            plan
        )

    assert (
        decision.read_bytes()
        == decision_before
    )

    assert (
        publication.read_bytes()
        == publication_before
    )


def test_rollback_provenance_is_unchanged_during_compensated_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    registry = _publish_many(
        tmp_path
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

    provenance.write_bytes(
        b'{"immutable": true}\n'
    )

    before = provenance.read_bytes()

    plan = _plan(
        registry
    )

    original_unlink = Path.unlink
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
                "injected provenance failure"
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
        match="injected provenance failure",
    ):
        ChampionHistoryRetentionExecutor(
            registry
        ).execute(
            plan
        )

    assert provenance.read_bytes() == before


def test_unknown_history_file_is_unchanged_during_compensated_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    registry = _publish_many(
        tmp_path
    )

    unknown = (
        registry
        / "history"
        / "operator-note.txt"
    )

    unknown.write_bytes(
        b"preserve exactly\n"
    )

    before = unknown.read_bytes()

    plan = _plan(
        registry
    )

    original_unlink = Path.unlink
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
                "injected unknown-file failure"
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
        match="injected unknown-file failure",
    ):
        ChampionHistoryRetentionExecutor(
            registry
        ).execute(
            plan
        )

    assert unknown.read_bytes() == before


def test_restore_failure_raises_atomicity_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    AtomicityError = (
        _load_atomicity_error()
    )

    registry = _publish_many(
        tmp_path
    )

    plan = _plan(
        registry
    )

    original_unlink = Path.unlink
    original_write_bytes = Path.write_bytes

    unlink_calls = 0
    restore_calls = 0

    def fail_second_unlink(
        self: Path,
        *args,
        **kwargs,
    ):
        nonlocal unlink_calls

        unlink_calls += 1

        if unlink_calls == 2:
            raise OSError(
                "original deletion failure"
            )

        return original_unlink(
            self,
            *args,
            **kwargs,
        )

    def fail_restore(
        self: Path,
        data: bytes,
    ):
        nonlocal restore_calls

        restore_calls += 1

        if restore_calls == 1:
            raise OSError(
                "injected restore failure"
            )

        return original_write_bytes(
            self,
            data,
        )

    monkeypatch.setattr(
        Path,
        "unlink",
        fail_second_unlink,
    )

    monkeypatch.setattr(
        Path,
        "write_bytes",
        fail_restore,
    )

    with pytest.raises(
        AtomicityError,
    ) as captured:

        ChampionHistoryRetentionExecutor(
            registry
        ).execute(
            plan
        )

    message = str(
        captured.value
    ).lower()

    assert (
        "atomic"
        in message
        or "restore"
        in message
        or "compensation"
        in message
    )


def test_restore_failure_is_never_converted_to_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    AtomicityError = (
        _load_atomicity_error()
    )

    registry = _publish_many(
        tmp_path
    )

    plan = _plan(
        registry
    )

    original_unlink = Path.unlink

    unlink_calls = 0

    def fail_second_unlink(
        self: Path,
        *args,
        **kwargs,
    ):
        nonlocal unlink_calls

        unlink_calls += 1

        if unlink_calls == 2:
            raise OSError(
                "delete failure"
            )

        return original_unlink(
            self,
            *args,
            **kwargs,
        )

    def always_fail_restore(
        self: Path,
        data: bytes,
    ):
        raise OSError(
            "restore failure"
        )

    monkeypatch.setattr(
        Path,
        "unlink",
        fail_second_unlink,
    )

    monkeypatch.setattr(
        Path,
        "write_bytes",
        always_fail_restore,
    )

    with pytest.raises(
        AtomicityError,
    ):
        result = (
            ChampionHistoryRetentionExecutor(
                registry
            ).execute(
                plan
            )
        )

        raise AssertionError(
            f"unexpected PASS result: {result!r}"
        )
