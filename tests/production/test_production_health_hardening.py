from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import lrp.production.production_health as health_mod

from lrp.production.production_health import (
    ProductionHealthService,
)


DOMAINS = [
    "active_champion",
    "writer_lock",
    "recovery_readiness",
    "history_safety",
    "lifecycle_readiness",
]


def _write_registry(
    root: Path,
    *,
    selected_model: str | None = "baseline",
) -> Path:
    registry = root / "registry"
    active = registry / "active"

    active.mkdir(
        parents=True,
        exist_ok=True,
    )

    decision_raw = (
        json.dumps(
            {
                "selection": {
                    "selected_model":
                        selected_model,
                },
            },
            indent=2,
        )
        + "\n"
    ).encode("utf-8")

    decision_path = (
        active
        / "champion_decision.json"
    )

    decision_path.write_bytes(
        decision_raw
    )

    source_path = (
        root
        / "source.json"
    )

    source_path.write_bytes(
        decision_raw
    )

    source_sha256 = hashlib.sha256(
        decision_raw
    ).hexdigest()

    (
        active
        / "publication.json"
    ).write_text(
        json.dumps(
            {
                "source_path":
                    str(source_path),

                "source_sha256":
                    source_sha256,

                "published_path":
                    str(decision_path),

                "published_at_kst":
                    "2026-08-20T00:00:00+09:00",

                "selected_model":
                    selected_model,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return registry


def _service(
    tmp_path: Path,
) -> ProductionHealthService:
    registry = _write_registry(
        tmp_path
    )

    return ProductionHealthService(
        registry_root=registry,
        snapshot_root=(
            tmp_path
            / "snapshots"
        ),
    )


def _managed_bytes(
    registry: Path,
) -> dict[str, bytes]:
    return {
        path.relative_to(
            registry
        ).as_posix():
            path.read_bytes()
        for path in registry.rglob("*")
        if (
            path.is_file()
            and path.name
            != ".writer.lock"
        )
    }


def test_repeated_snapshot_semantics_are_stable(
    tmp_path: Path,
) -> None:
    service = _service(
        tmp_path
    )

    first = service.snapshot()
    second = service.snapshot()

    assert first.status == second.status

    assert [
        (
            check.name,
            check.domain,
            check.status,
            check.detail,
        )
        for check in first.checks
    ] == [
        (
            check.name,
            check.domain,
            check.status,
            check.detail,
        )
        for check in second.checks
    ]

    assert [
        (
            issue.code,
            issue.domain,
            issue.severity,
            issue.message,
        )
        for issue in first.issues
    ] == [
        (
            issue.code,
            issue.domain,
            issue.severity,
            issue.message,
        )
        for issue in second.issues
    ]

    assert first.domains == second.domains


def test_check_order_is_exact(
    tmp_path: Path,
) -> None:
    snapshot = _service(
        tmp_path
    ).snapshot()

    assert [
        check.domain
        for check in snapshot.checks
    ] == DOMAINS


def test_issue_order_is_domain_then_code(
    tmp_path: Path,
) -> None:
    registry = _write_registry(
        tmp_path,
        selected_model=None,
    )

    (
        registry
        / ".writer.lock"
    ).write_bytes(
        b"\x00not-json"
    )

    snapshot = ProductionHealthService(
        registry_root=registry,
        snapshot_root=(
            tmp_path
            / "snapshots"
        ),
    ).snapshot()

    domain_index = {
        domain: index
        for index, domain
        in enumerate(DOMAINS)
    }

    actual = [
        (
            domain_index[
                issue.domain
            ],
            issue.code,
        )
        for issue in snapshot.issues
    ]

    assert actual == sorted(
        actual
    )


def test_unreadable_writer_lock_warns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(
        tmp_path
    )

    lock_path = (
        service._registry_root
        / ".writer.lock"
    )

    lock_path.write_bytes(
        b"\x00{}"
    )

    original_read_text = Path.read_text

    def fail_read_text(
        self: Path,
        *args: object,
        **kwargs: object,
    ) -> str:
        if self == lock_path:
            raise PermissionError(
                "synthetic lock read denial"
            )

        return original_read_text(
            self,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        Path,
        "read_text",
        fail_read_text,
    )

    domain, check, issues = (
        service._writer_lock_domain()
    )

    assert domain["status"] == "WARN"
    assert domain["metadata_valid"] is False

    assert check.status == "WARN"

    assert any(
        issue.code
        == "WRITER_LOCK_MALFORMED"
        for issue in issues
    )


def test_active_champion_exception_isolated_to_domain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(
        tmp_path
    )

    def fail_audit(
        self: object,
        *,
        registry_root: object,
        snapshot_root: object,
    ) -> object:
        raise RuntimeError(
            "synthetic audit failure"
        )

    monkeypatch.setattr(
        health_mod.ProductionChampionAudit,
        "audit",
        fail_audit,
    )

    snapshot = service.snapshot()

    assert snapshot.status == "FAIL"

    assert (
        snapshot.domains[
            "active_champion"
        ]["status"]
        == "FAIL"
    )

    assert (
        snapshot.domains[
            "recovery_readiness"
        ]["status"]
        == "PASS"
    )

    assert (
        snapshot.domains[
            "history_safety"
        ]["status"]
        == "PASS"
    )

    assert (
        snapshot.domains[
            "lifecycle_readiness"
        ]["status"]
        == "PASS"
    )

    assert any(
        issue.code
        == "ACTIVE_CHAMPION_FAIL"
        for issue in snapshot.issues
    )


def test_managed_registry_bytes_unchanged_after_repeated_snapshot(
    tmp_path: Path,
) -> None:
    registry = _write_registry(
        tmp_path
    )

    service = ProductionHealthService(
        registry_root=registry,
        snapshot_root=(
            tmp_path
            / "snapshots"
        ),
    )

    before = _managed_bytes(
        registry
    )

    service.snapshot()
    service.snapshot()
    service.snapshot()

    after = _managed_bytes(
        registry
    )

    assert after == before


def test_generated_at_does_not_change_semantic_status(
    tmp_path: Path,
) -> None:
    service = _service(
        tmp_path
    )

    first = service.snapshot()
    second = service.snapshot()

    assert first.status == second.status

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


def test_domains_payload_order_is_exact(
    tmp_path: Path,
) -> None:
    snapshot = _service(
        tmp_path
    ).snapshot()

    assert list(
        snapshot.domains
    ) == DOMAINS

    payload = snapshot.to_payload()

    assert list(
        payload["domains"]
    ) == DOMAINS
