from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lrp.production.champion_registry_publisher import (
    ProductionChampionRegistryPublisher,
)


def _load_api():
    from lrp.production.champion_history_retention import (
        ChampionHistoryRetentionPlanner,
        ChampionHistoryRetentionPolicy,
    )

    return (
        ChampionHistoryRetentionPolicy,
        ChampionHistoryRetentionPlanner,
    )


def _write_decision(
    path: Path,
    *,
    model: str,
) -> Path:

    payload = {
        "selection": {
            "selected_model":
                model,
        },
    }

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return path


def _publish(
    registry: Path,
    source: Path,
):
    return (
        ProductionChampionRegistryPublisher()
        .publish(
            source_decision=source,
            registry_root=registry,
        )
    )


def _history_revision_ids(
    registry: Path,
) -> list[str]:

    result = []

    history = (
        registry
        / "history"
    )

    if not history.is_dir():
        return result

    for path in history.glob(
        "*.json"
    ):

        result.append(
            path.stem
        )

    return sorted(
        result
    )


def _decision_ids(
    registry: Path,
) -> list[str]:

    root = (
        registry
        / "history"
        / "decisions"
    )

    if not root.is_dir():
        return []

    return sorted(
        path.stem
        for path in root.glob(
            "*.json"
        )
    )


def _snapshot_tree(
    registry: Path,
) -> dict[str, bytes]:

    if not registry.exists():
        return {}

    result = {}

    for path in sorted(
        registry.rglob(
            "*"
        )
    ):

        if not path.is_file():
            continue

        result[
            path.relative_to(
                registry
            ).as_posix()
        ] = path.read_bytes()

    return result


def _active_source_sha(
    registry: Path,
) -> str:

    data = (
        registry
        / "active"
        / "champion_decision.json"
    ).read_bytes()

    return hashlib.sha256(
        data
    ).hexdigest()


def _make_three_publications(
    tmp_path: Path,
) -> Path:

    registry = (
        tmp_path
        / "registry"
    )

    for index in range(
        1,
        4,
    ):

        _publish(
            registry,
            _write_decision(
                tmp_path
                / f"d{index}.json",
                model=(
                    f"model-{index}"
                ),
            ),
        )

    return registry


def test_policy_rejects_keep_recent_less_than_one() -> None:
    (
        Policy,
        _,
    ) = _load_api()

    with pytest.raises(
        ValueError,
    ):
        Policy(
            keep_recent=0
        )

    with pytest.raises(
        ValueError,
    ):
        Policy(
            keep_recent=-1
        )


def test_planner_is_read_only(
    tmp_path: Path,
) -> None:

    (
        Policy,
        Planner,
    ) = _load_api()

    registry = (
        _make_three_publications(
            tmp_path
        )
    )

    before = (
        _snapshot_tree(
            registry
        )
    )

    planner = Planner(
        registry
    )

    planner.plan(
        Policy(
            keep_recent=2
        )
    )

    after = (
        _snapshot_tree(
            registry
        )
    )

    assert after == before


def test_planner_keeps_newest_n_publication_revisions(
    tmp_path: Path,
) -> None:

    (
        Policy,
        Planner,
    ) = _load_api()

    registry = (
        _make_three_publications(
            tmp_path
        )
    )

    plan = Planner(
        registry
    ).plan(
        Policy(
            keep_recent=2
        )
    )

    assert (
        len(
            plan.retained_revision_ids
        )
        == 2
    )

    assert (
        len(
            plan.prunable_revision_ids
        )
        == 1
    )


def test_planner_always_preserves_active_revision(
    tmp_path: Path,
) -> None:

    (
        Policy,
        Planner,
    ) = _load_api()

    registry = (
        _make_three_publications(
            tmp_path
        )
    )

    active_sha = (
        _active_source_sha(
            registry
        )
    )

    plan = Planner(
        registry
    ).plan(
        Policy(
            keep_recent=1
        )
    )

    assert (
        plan.active_source_sha256
        == active_sha
    )

    assert (
        active_sha
        in plan.retained_decision_sha256s
    )


def test_planner_preserves_decisions_referenced_by_retained_revisions(
    tmp_path: Path,
) -> None:

    (
        Policy,
        Planner,
    ) = _load_api()

    registry = (
        _make_three_publications(
            tmp_path
        )
    )

    plan = Planner(
        registry
    ).plan(
        Policy(
            keep_recent=2
        )
    )

    assert (
        len(
            plan.retained_decision_sha256s
        )
        >= 2
    )

    for source_sha in (
        plan.retained_decision_sha256s
    ):

        assert (
            registry
            / "history"
            / "decisions"
            / f"{source_sha}.json"
        ).is_file()


def test_planner_supports_multiple_revisions_sharing_one_decision_snapshot(
    tmp_path: Path,
) -> None:

    (
        Policy,
        Planner,
    ) = _load_api()

    registry = (
        tmp_path
        / "registry"
    )

    shared = (
        _write_decision(
            tmp_path
            / "shared.json",
            model="shared-model",
        )
    )

    first = _publish(
        registry,
        shared,
    )

    second = _publish(
        registry,
        shared,
    )

    third = _publish(
        registry,
        _write_decision(
            tmp_path
            / "third.json",
            model="third-model",
        ),
    )

    assert (
        first.source_sha256
        == second.source_sha256
    )

    plan = Planner(
        registry
    ).plan(
        Policy(
            keep_recent=2
        )
    )

    shared_sha = (
        first.source_sha256
    )

    assert (
        shared_sha
        in plan.retained_decision_sha256s
    )

    assert (
        shared_sha
        not in plan.prunable_decision_sha256s
    )


def test_planner_marks_only_zero_retained_reference_decisions_prunable(
    tmp_path: Path,
) -> None:

    (
        Policy,
        Planner,
    ) = _load_api()

    registry = (
        _make_three_publications(
            tmp_path
        )
    )

    plan = Planner(
        registry
    ).plan(
        Policy(
            keep_recent=1
        )
    )

    retained = set(
        plan.retained_decision_sha256s
    )

    prunable = set(
        plan.prunable_decision_sha256s
    )

    assert retained
    assert prunable

    assert (
        retained
        .isdisjoint(
            prunable
        )
    )

    assert (
        set(
            _decision_ids(
                registry
            )
        )
        ==
        retained
        | prunable
    )


def test_planner_ignores_rollback_provenance_for_pruning(
    tmp_path: Path,
) -> None:

    (
        Policy,
        Planner,
    ) = _load_api()

    registry = (
        _make_three_publications(
            tmp_path
        )
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

    rollback_file = (
        rollback_root
        / "abc.json"
    )

    rollback_file.write_text(
        '{"audit": true}\n',
        encoding="utf-8",
    )

    plan = Planner(
        registry
    ).plan(
        Policy(
            keep_recent=1
        )
    )

    assert (
        rollback_file.is_file()
    )

    serialized = repr(
        plan
    )

    assert (
        "rollbacks/abc.json"
        not in serialized
    )


def test_planner_preserves_unknown_history_files(
    tmp_path: Path,
) -> None:

    (
        Policy,
        Planner,
    ) = _load_api()

    registry = (
        _make_three_publications(
            tmp_path
        )
    )

    unknown = (
        registry
        / "history"
        / "operator-note.txt"
    )

    unknown.write_text(
        "preserve",
        encoding="utf-8",
    )

    plan = Planner(
        registry
    ).plan(
        Policy(
            keep_recent=1
        )
    )

    assert unknown.is_file()

    assert (
        "operator-note.txt"
        not in repr(
            plan
        )
    )


def test_planner_fails_closed_on_malformed_publication_revision(
    tmp_path: Path,
) -> None:

    (
        Policy,
        Planner,
    ) = _load_api()

    registry = (
        _make_three_publications(
            tmp_path
        )
    )

    revision_ids = (
        _history_revision_ids(
            registry
        )
    )

    assert revision_ids

    target = (
        registry
        / "history"
        / f"{revision_ids[0]}.json"
    )

    target.write_text(
        "{not-json",
        encoding="utf-8",
    )

    with pytest.raises(
        (
            ValueError,
            json.JSONDecodeError,
        ),
    ):
        Planner(
            registry
        ).plan(
            Policy(
                keep_recent=1
            )
        )


def test_planner_fails_closed_on_missing_required_decision_snapshot(
    tmp_path: Path,
) -> None:

    (
        Policy,
        Planner,
    ) = _load_api()

    registry = (
        _make_three_publications(
            tmp_path
        )
    )

    decision_ids = (
        _decision_ids(
            registry
        )
    )

    assert decision_ids

    target = (
        registry
        / "history"
        / "decisions"
        / f"{decision_ids[0]}.json"
    )

    target.unlink()

    with pytest.raises(
        (
            FileNotFoundError,
            ValueError,
        ),
    ):
        Planner(
            registry
        ).plan(
            Policy(
                keep_recent=1
            )
        )


def test_planner_output_is_deterministic(
    tmp_path: Path,
) -> None:

    (
        Policy,
        Planner,
    ) = _load_api()

    registry = (
        _make_three_publications(
            tmp_path
        )
    )

    planner = Planner(
        registry
    )

    policy = Policy(
        keep_recent=2
    )

    first = planner.plan(
        policy
    )

    second = planner.plan(
        policy
    )

    assert first == second
