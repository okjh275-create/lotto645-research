from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _api():
    from lrp.production.production_reconciliation import (
        ProductionReconciliationService,
    )
    return ProductionReconciliationService


def _write_json(path: Path, payload: object) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)

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


def _canonical_registry(root: Path) -> Path:
    registry = root / "registry"

    active = registry / "active"
    history = registry / "history"
    decisions = history / "decisions"
    rollbacks = history / "rollbacks"

    active.mkdir(parents=True, exist_ok=True)
    decisions.mkdir(parents=True, exist_ok=True)
    rollbacks.mkdir(parents=True, exist_ok=True)

    decision_payload = {
        "selection": {
            "selected_model": "baseline",
        },
    }

    decision_raw = _write_json(
        active / "champion_decision.json",
        decision_payload,
    )

    source_sha = hashlib.sha256(
        decision_raw
    ).hexdigest()

    revision_id = "a" * 64

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

    (
        decisions
        / f"{source_sha}.json"
    ).write_bytes(decision_raw)

    return registry


def _managed_bytes(
    registry: Path,
) -> dict[str, bytes]:
    return {
        path.relative_to(
            registry
        ).as_posix(): path.read_bytes()
        for path in registry.rglob("*")
        if (
            path.is_file()
            and path.name != ".writer.lock"
        )
    }


def _reconcile(registry: Path):
    Service = _api()

    return Service(
        registry_root=registry
    ).reconcile()


def _semantic_payload(result):
    payload = result.to_payload()
    payload.pop("generated_at")
    return payload


def test_repeated_reconciliation_semantics_are_stable(
    tmp_path: Path,
) -> None:
    registry = _canonical_registry(tmp_path)

    first = _reconcile(registry)
    second = _reconcile(registry)

    assert (
        _semantic_payload(first)
        == _semantic_payload(second)
    )


def test_generated_at_is_non_semantic(
    tmp_path: Path,
) -> None:
    registry = _canonical_registry(tmp_path)

    first = _reconcile(registry)
    second = _reconcile(registry)

    assert first.status == second.status
    assert first.checks == second.checks
    assert first.issues == second.issues
    assert first.domains == second.domains


def test_canonical_registry_fixture_passes(
    tmp_path: Path,
) -> None:
    result = _reconcile(
        _canonical_registry(tmp_path)
    )

    assert result.status == "PASS"
    assert result.issues == ()


def test_malformed_active_decision_fails_without_exception(
    tmp_path: Path,
) -> None:
    registry = _canonical_registry(tmp_path)

    (
        registry
        / "active"
        / "champion_decision.json"
    ).write_bytes(b"{not-json")

    result = _reconcile(registry)

    assert result.status == "FAIL"

    assert any(
        issue.domain == "active_pair"
        and issue.severity == "FAIL"
        for issue in result.issues
    )


def test_malformed_active_publication_fails_without_exception(
    tmp_path: Path,
) -> None:
    registry = _canonical_registry(tmp_path)

    (
        registry
        / "active"
        / "publication.json"
    ).write_bytes(b"{not-json")

    result = _reconcile(registry)

    assert result.status == "FAIL"

    assert any(
        issue.domain == "active_pair"
        and issue.severity == "FAIL"
        for issue in result.issues
    )


def test_malformed_revision_fails_without_exception(
    tmp_path: Path,
) -> None:
    registry = _canonical_registry(tmp_path)

    path = (
        registry
        / "history"
        / (("a" * 64) + ".json")
    )

    path.write_bytes(b"{not-json")

    result = _reconcile(registry)

    assert result.status == "FAIL"

    assert any(
        issue.domain == "revision_history"
        and issue.severity == "FAIL"
        for issue in result.issues
    )


def test_malformed_decision_history_fails_without_exception(
    tmp_path: Path,
) -> None:
    registry = _canonical_registry(tmp_path)

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

    path.write_bytes(b"{not-json")

    result = _reconcile(registry)

    assert result.status == "FAIL"

    assert any(
        issue.code
        == "DECISION_HISTORY_HASH_MISMATCH"
        for issue in result.issues
    )


def test_malformed_rollback_warns_without_exception(
    tmp_path: Path,
) -> None:
    registry = _canonical_registry(tmp_path)

    (
        registry
        / "history"
        / "rollbacks"
        / "rollback-malformed.json"
    ).write_bytes(b"{not-json")

    result = _reconcile(registry)

    assert result.status == "WARN"

    assert any(
        issue.code
        == "ROLLBACK_RECORD_MALFORMED"
        for issue in result.issues
    )


def test_orphan_issue_order_is_deterministic(
    tmp_path: Path,
) -> None:
    registry = _canonical_registry(tmp_path)

    _write_json(
        registry
        / "history"
        / (("d" * 64) + ".json"),
        {
            "revision_id": "d" * 64,
            "source_sha256": "e" * 64,
            "selected_model": "baseline",
        },
    )

    _write_json(
        registry
        / "history"
        / (("c" * 64) + ".json"),
        {
            "revision_id": "c" * 64,
            "source_sha256": "f" * 64,
            "selected_model": "baseline",
        },
    )

    payload_a = {
        "selection": {
            "selected_model": "combined",
        },
    }

    raw_a = (
        json.dumps(
            payload_a,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    sha_a = hashlib.sha256(
        raw_a
    ).hexdigest()

    payload_b = {
        "selection": {
            "selected_model": "experimental",
        },
    }

    raw_b = (
        json.dumps(
            payload_b,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    sha_b = hashlib.sha256(
        raw_b
    ).hexdigest()

    decisions = (
        registry
        / "history"
        / "decisions"
    )

    (
        decisions
        / f"{sha_b}.json"
    ).write_bytes(raw_b)

    (
        decisions
        / f"{sha_a}.json"
    ).write_bytes(raw_a)

    first = _reconcile(registry)
    second = _reconcile(registry)

    first_order = [
        (
            issue.domain,
            issue.code,
            issue.path or "",
        )
        for issue in first.issues
    ]

    second_order = [
        (
            issue.domain,
            issue.code,
            issue.path or "",
        )
        for issue in second.issues
    ]

    assert first_order == second_order

    domain_order = [
        "active_pair",
        "revision_history",
        "decision_history",
        "rollback_provenance",
        "cross_store_identity",
    ]

    assert first_order == sorted(
        first_order,
        key=lambda item: (
            domain_order.index(item[0]),
            item[1],
            item[2],
        ),
    )


def test_managed_registry_bytes_unchanged_after_repeated_reconcile(
    tmp_path: Path,
) -> None:
    registry = _canonical_registry(tmp_path)

    before = _managed_bytes(registry)

    _reconcile(registry)
    _reconcile(registry)

    after = _managed_bytes(registry)

    assert after == before
