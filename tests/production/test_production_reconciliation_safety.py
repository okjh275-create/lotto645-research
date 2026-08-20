from __future__ import annotations

import ast
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
            "selected_model":
                "baseline",
        },
    }

    raw = _write_json(
        active
        / "champion_decision.json",
        decision,
    )

    source_sha = hashlib.sha256(
        raw
    ).hexdigest()

    revision_id = "b" * 64

    _write_json(
        active
        / "publication.json",
        {
            "source_path":
                "source.json",

            "source_sha256":
                source_sha,

            "published_path":
                str(
                    active
                    / "champion_decision.json"
                ),

            "published_at_kst":
                "2026-08-20T00:00:00+09:00",

            "selected_model":
                "baseline",

            "revision_id":
                revision_id,
        },
    )

    _write_json(
        history
        / f"{revision_id}.json",
        {
            "revision_id":
                revision_id,

            "source_sha256":
                source_sha,

            "selected_model":
                "baseline",
        },
    )

    _write_json(
        decisions
        / f"{source_sha}.json",
        decision,
    )

    return registry


def _managed_bytes(
    registry: Path,
) -> dict[str, bytes]:
    return {
        path.relative_to(
            registry
        ).as_posix():
            path.read_bytes()
        for path
        in registry.rglob("*")
        if (
            path.is_file()
            and path.name
            != ".writer.lock"
        )
    }


def test_reconcile_is_read_only(
    tmp_path: Path,
) -> None:
    Service = _api()

    registry = _registry(
        tmp_path
    )

    before = _managed_bytes(
        registry
    )

    Service(
        registry_root=registry
    ).reconcile()

    after = _managed_bytes(
        registry
    )

    assert after == before


def test_reconcile_does_not_directly_acquire_writer_lock() -> None:
    path = Path(
        "lrp/production/"
        "production_reconciliation.py"
    )

    source = path.read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    direct_calls = []

    for node in ast.walk(tree):
        if isinstance(
            node,
            ast.Call,
        ):
            func = node.func

            if (
                isinstance(
                    func,
                    ast.Name,
                )
                and func.id
                == "ProductionRegistryWriterLock"
            ):
                direct_calls.append(
                    node.lineno
                )

    assert direct_calls == []


def test_reconcile_does_not_execute_mutating_services() -> None:
    source = Path(
        "lrp/production/"
        "production_reconciliation.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    forbidden = [
        ".backup(",
        ".restore(",
        ".rollback(",
        ".publish(",
        ".execute(",
        ".unlink(",
        ".write_text(",
        ".write_bytes(",
    ]

    violations = [
        token
        for token in forbidden
        if token in source
    ]

    assert violations == []


def test_reconciliation_order_is_deterministic(
    tmp_path: Path,
) -> None:
    Service = _api()

    registry = _registry(
        tmp_path
    )

    _write_json(
        registry
        / "history"
        / (("c" * 64) + ".json"),
        {
            "revision_id":
                "c" * 64,

            "source_sha256":
                "d" * 64,

            "selected_model":
                "baseline",
        },
    )

    first = Service(
        registry_root=registry
    ).reconcile()

    second = Service(
        registry_root=registry
    ).reconcile()

    expected_domains = [
        "active_pair",
        "revision_history",
        "decision_history",
        "rollback_provenance",
        "cross_store_identity",
    ]

    assert [
        check.domain
        for check in first.checks
    ] == expected_domains

    assert [
        (
            issue.domain,
            issue.code,
            issue.path or "",
        )
        for issue in first.issues
    ] == [
        (
            issue.domain,
            issue.code,
            issue.path or "",
        )
        for issue in second.issues
    ]

    first_payload = first.to_payload()
    second_payload = second.to_payload()

    first_payload.pop(
        "generated_at"
    )

    second_payload.pop(
        "generated_at"
    )

    assert (
        first_payload
        == second_payload
    )
