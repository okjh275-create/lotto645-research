from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def _api():
    from lrp.production.production_reconciliation import (
        ProductionReconciliationService,
    )

    return ProductionReconciliationService


def _write_json(
    path: Path,
    payload: object,
) -> bytes:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    path.write_bytes(raw)

    return raw


def _registry(
    root: Path,
) -> Path:
    registry = root / "registry"

    active = registry / "active"
    history = registry / "history"
    decisions = history / "decisions"
    rollbacks = history / "rollbacks"

    active.mkdir(
        parents=True,
        exist_ok=True,
    )

    decisions.mkdir(
        parents=True,
        exist_ok=True,
    )

    rollbacks.mkdir(
        parents=True,
        exist_ok=True,
    )

    decision = {
        "selection": {
            "selected_model": "baseline",
        },
    }

    decision_raw = _write_json(
        active / "champion_decision.json",
        decision,
    )

    source_sha = hashlib.sha256(
        decision_raw
    ).hexdigest()

    revision_id = "b" * 64

    _write_json(
        active / "publication.json",
        {
            "source_path": "source.json",
            "source_sha256": source_sha,
            "published_path": str(
                active / "champion_decision.json"
            ),
            "published_at_kst":
                "2026-08-20T00:00:00+09:00",
            "selected_model": "baseline",
            "revision_id": revision_id,
        },
    )

    _write_json(
        history / f"{revision_id}.json",
        {
            "revision_id": revision_id,
            "source_sha256": source_sha,
            "selected_model": "baseline",
        },
    )

    _write_json(
        decisions / f"{source_sha}.json",
        decision,
    )

    return registry


def _reconcile(
    registry: Path,
):
    Service = _api()

    return Service(
        registry_root=registry
    ).reconcile()


def test_reconcile_clean_registry_passes(
    tmp_path: Path,
) -> None:
    result = _reconcile(
        _registry(tmp_path)
    )

    assert result.status == "PASS"


def test_reconcile_missing_active_decision_fails(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)

    (
        registry
        / "active"
        / "champion_decision.json"
    ).unlink()

    result = _reconcile(registry)

    assert result.status == "FAIL"

    assert any(
        issue.code
        == "ACTIVE_PAIR_MISSING"
        for issue in result.issues
    )


def test_reconcile_missing_active_publication_fails(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)

    (
        registry
        / "active"
        / "publication.json"
    ).unlink()

    result = _reconcile(registry)

    assert result.status == "FAIL"


def test_reconcile_active_pair_hash_mismatch_fails(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)

    publication = (
        registry
        / "active"
        / "publication.json"
    )

    payload = json.loads(
        publication.read_text(
            encoding="utf-8"
        )
    )

    payload["source_sha256"] = "f" * 64

    _write_json(
        publication,
        payload,
    )

    result = _reconcile(registry)

    assert result.status == "FAIL"

    assert any(
        issue.code
        == "ACTIVE_PAIR_MISMATCH"
        for issue in result.issues
    )


def test_reconcile_missing_active_revision_fails(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)

    (
        registry
        / "history"
        / (("b" * 64) + ".json")
    ).unlink()

    result = _reconcile(registry)

    assert result.status == "FAIL"

    assert any(
        issue.code
        == "REVISION_MISSING"
        for issue in result.issues
    )


def test_reconcile_revision_hash_mismatch_fails(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)

    path = (
        registry
        / "history"
        / (("b" * 64) + ".json")
    )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    payload["source_sha256"] = "e" * 64

    _write_json(
        path,
        payload,
    )

    result = _reconcile(registry)

    assert result.status == "FAIL"

    assert any(
        issue.code
        == "REVISION_HASH_MISMATCH"
        for issue in result.issues
    )


def test_reconcile_orphan_revision_warns(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)

    _write_json(
        registry
        / "history"
        / (("c" * 64) + ".json"),
        {
            "revision_id": "c" * 64,
            "source_sha256": "d" * 64,
            "selected_model": "baseline",
        },
    )

    result = _reconcile(registry)

    assert result.status == "WARN"

    assert any(
        issue.code
        == "ORPHAN_REVISION"
        for issue in result.issues
    )


def test_reconcile_missing_decision_history_fails(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)

    publication = json.loads(
        (
            registry
            / "active"
            / "publication.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    (
        registry
        / "history"
        / "decisions"
        / (
            publication[
                "source_sha256"
            ]
            + ".json"
        )
    ).unlink()

    result = _reconcile(registry)

    assert result.status == "FAIL"

    assert any(
        issue.code
        == "DECISION_HISTORY_MISSING"
        for issue in result.issues
    )


def test_reconcile_decision_history_hash_mismatch_fails(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)

    publication = json.loads(
        (
            registry
            / "active"
            / "publication.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    path = (
        registry
        / "history"
        / "decisions"
        / (
            publication[
                "source_sha256"
            ]
            + ".json"
        )
    )

    _write_json(
        path,
        {
            "selection": {
                "selected_model":
                    "combined",
            },
        },
    )

    result = _reconcile(registry)

    assert result.status == "FAIL"

    assert any(
        issue.code
        == "DECISION_HISTORY_HASH_MISMATCH"
        for issue in result.issues
    )


def test_reconcile_orphan_decision_warns(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)

    payload = {
        "selection": {
            "selected_model":
                "combined",
        },
    }

    raw = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    orphan_sha = hashlib.sha256(
        raw
    ).hexdigest()

    (
        registry
        / "history"
        / "decisions"
        / f"{orphan_sha}.json"
    ).write_bytes(raw)

    result = _reconcile(registry)

    assert result.status == "WARN"

    assert any(
        issue.code
        == "ORPHAN_DECISION"
        for issue in result.issues
    )


def test_reconcile_valid_rollback_provenance_passes(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)

    _write_json(
        registry
        / "history"
        / "rollbacks"
        / "rollback-001.json",
        {
            "source_revision_id":
                "b" * 64,

            "target_revision_id":
                "b" * 64,

            "target_source_sha256":
                json.loads(
                    (
                        registry
                        / "active"
                        / "publication.json"
                    ).read_text(
                        encoding="utf-8"
                    )
                )["source_sha256"],
        },
    )

    result = _reconcile(registry)

    assert result.status == "PASS"


def test_reconcile_missing_rollback_target_fails(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)

    _write_json(
        registry
        / "history"
        / "rollbacks"
        / "rollback-001.json",
        {
            "source_revision_id":
                "b" * 64,

            "target_revision_id":
                "d" * 64,

            "target_source_sha256":
                "e" * 64,
        },
    )

    result = _reconcile(registry)

    assert result.status == "FAIL"

    assert any(
        issue.code
        == "ROLLBACK_TARGET_MISSING"
        for issue in result.issues
    )


def test_reconcile_orphan_rollback_warns(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)

    _write_json(
        registry
        / "history"
        / "rollbacks"
        / "rollback-001.json",
        {
            "source_revision_id":
                None,

            "target_revision_id":
                "b" * 64,

            "target_source_sha256":
                "d" * 64,
        },
    )

    result = _reconcile(registry)

    assert result.status == "WARN"

    assert any(
        issue.code
        == "ORPHAN_ROLLBACK"
        for issue in result.issues
    )


def test_reconcile_cross_store_identity_mismatch_fails(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)

    revision = (
        registry
        / "history"
        / (("b" * 64) + ".json")
    )

    payload = json.loads(
        revision.read_text(
            encoding="utf-8"
        )
    )

    payload["selected_model"] = "combined"

    _write_json(
        revision,
        payload,
    )

    result = _reconcile(registry)

    assert result.status == "FAIL"

    assert any(
        issue.code
        == "CROSS_STORE_IDENTITY_MISMATCH"
        for issue in result.issues
    )


def test_reconcile_aggregate_warn(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)

    _write_json(
        registry
        / "history"
        / (("c" * 64) + ".json"),
        {
            "revision_id": "c" * 64,
            "source_sha256": "d" * 64,
            "selected_model": "baseline",
        },
    )

    result = _reconcile(registry)

    assert result.status == "WARN"


def test_reconcile_aggregate_fail(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)

    (
        registry
        / "active"
        / "publication.json"
    ).unlink()

    result = _reconcile(registry)

    assert result.status == "FAIL"
