from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lrp.production import (
    ProductionChampionAudit,
    ProductionChampionAuditResult,
)


def _write_decision(
    path: Path,
    *,
    selected_model: object,
) -> bytes:
    raw = (
        json.dumps(
            {
                "selection": {
                    "selected_model": (
                        selected_model
                    ),
                },
            },
            indent=2,
        )
        + "\n"
    ).encode("utf-8")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_bytes(raw)

    return raw


def _write_publication(
    path: Path,
    *,
    source_path: str,
    source_sha256: str,
    published_path: str,
    selected_model: object,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            {
                "source_path": source_path,
                "source_sha256": source_sha256,
                "published_path": published_path,
                "published_at_kst": (
                    "2026-08-16T22:00:00+09:00"
                ),
                "selected_model": (
                    selected_model
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _registry_paths(
    root: Path,
) -> tuple[
    Path,
    Path,
]:
    active = (
        root
        / "active"
    )

    return (
        active
        / "champion_decision.json",
        active
        / "publication.json",
    )


def test_audit_returns_result_for_valid_baseline_fallback(
    tmp_path: Path,
) -> None:
    registry = (
        tmp_path
        / "registry"
    )

    decision_path, publication_path = (
        _registry_paths(
            registry
        )
    )

    raw = _write_decision(
        decision_path,
        selected_model=None,
    )

    _write_publication(
        publication_path,
        source_path="source.json",
        source_sha256=(
            hashlib.sha256(
                raw
            ).hexdigest()
        ),
        published_path=str(
            decision_path
        ),
        selected_model=None,
    )

    result = (
        ProductionChampionAudit()
        .audit(
            registry_root=registry,
            snapshot_root=(
                tmp_path
                / "snapshots"
            ),
        )
    )

    assert isinstance(
        result,
        ProductionChampionAuditResult,
    )

    assert result.status == "WARN"

    assert result.selected_model is None

    assert (
        result.resolved_model
        == "baseline"
    )

    assert (
        result.fallback_applied
        is True
    )

    assert (
        result.fallback_reason
        == "no_selected_model"
    )

    assert any(
        issue.code
        == "baseline_fallback"
        for issue in result.issues
    )


def test_audit_passes_for_supported_selected_model(
    tmp_path: Path,
) -> None:
    registry = (
        tmp_path
        / "registry"
    )

    decision_path, publication_path = (
        _registry_paths(
            registry
        )
    )

    raw = _write_decision(
        decision_path,
        selected_model="combined",
    )

    _write_publication(
        publication_path,
        source_path="source.json",
        source_sha256=(
            hashlib.sha256(
                raw
            ).hexdigest()
        ),
        published_path=str(
            decision_path
        ),
        selected_model="combined",
    )

    snapshot_root = (
        tmp_path
        / "snapshots"
    )

    (
        snapshot_root
        / "regime-calibration"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        snapshot_root
        / "regime-bayesian"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    result = (
        ProductionChampionAudit()
        .audit(
            registry_root=registry,
            snapshot_root=(
                snapshot_root
            ),
        )
    )

    assert result.status == "PASS"

    assert (
        result.selected_model
        == "combined"
    )

    assert (
        result.resolved_model
        == "combined"
    )

    assert (
        result.fallback_applied
        is False
    )

    assert result.fallback_reason is None

    assert result.issues == ()


def test_audit_fails_when_registry_root_missing(
    tmp_path: Path,
) -> None:
    result = (
        ProductionChampionAudit()
        .audit(
            registry_root=(
                tmp_path
                / "missing"
            ),
            snapshot_root=(
                tmp_path
                / "snapshots"
            ),
        )
    )

    assert result.status == "FAIL"

    assert any(
        issue.code
        == "registry_missing"
        for issue in result.issues
    )


def test_audit_fails_when_active_decision_missing(
    tmp_path: Path,
) -> None:
    registry = (
        tmp_path
        / "registry"
    )

    (
        registry
        / "active"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    result = (
        ProductionChampionAudit()
        .audit(
            registry_root=registry,
            snapshot_root=(
                tmp_path
                / "snapshots"
            ),
        )
    )

    assert result.status == "FAIL"

    assert any(
        issue.code
        == "active_decision_missing"
        for issue in result.issues
    )


def test_audit_fails_when_publication_missing(
    tmp_path: Path,
) -> None:
    registry = (
        tmp_path
        / "registry"
    )

    decision_path, _ = (
        _registry_paths(
            registry
        )
    )

    _write_decision(
        decision_path,
        selected_model="baseline",
    )

    result = (
        ProductionChampionAudit()
        .audit(
            registry_root=registry,
            snapshot_root=(
                tmp_path
                / "snapshots"
            ),
        )
    )

    assert result.status == "FAIL"

    assert any(
        issue.code
        == "publication_missing"
        for issue in result.issues
    )


def test_audit_fails_when_decision_invalid(
    tmp_path: Path,
) -> None:
    registry = (
        tmp_path
        / "registry"
    )

    decision_path, publication_path = (
        _registry_paths(
            registry
        )
    )

    decision_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    decision_path.write_text(
        "{invalid",
        encoding="utf-8",
    )

    _write_publication(
        publication_path,
        source_path="source.json",
        source_sha256="0" * 64,
        published_path=str(
            decision_path
        ),
        selected_model=None,
    )

    result = (
        ProductionChampionAudit()
        .audit(
            registry_root=registry,
            snapshot_root=(
                tmp_path
                / "snapshots"
            ),
        )
    )

    assert result.status == "FAIL"

    assert any(
        issue.code
        == "active_decision_invalid"
        for issue in result.issues
    )


def test_audit_fails_when_publication_json_invalid(
    tmp_path: Path,
) -> None:
    registry = (
        tmp_path
        / "registry"
    )

    decision_path, publication_path = (
        _registry_paths(
            registry
        )
    )

    _write_decision(
        decision_path,
        selected_model="baseline",
    )

    publication_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    publication_path.write_text(
        "{invalid",
        encoding="utf-8",
    )

    result = (
        ProductionChampionAudit()
        .audit(
            registry_root=registry,
            snapshot_root=(
                tmp_path
                / "snapshots"
            ),
        )
    )

    assert result.status == "FAIL"

    assert any(
        issue.code
        == "publication_invalid"
        for issue in result.issues
    )


def test_audit_fails_when_models_disagree(
    tmp_path: Path,
) -> None:
    registry = (
        tmp_path
        / "registry"
    )

    decision_path, publication_path = (
        _registry_paths(
            registry
        )
    )

    raw = _write_decision(
        decision_path,
        selected_model="combined",
    )

    _write_publication(
        publication_path,
        source_path="source.json",
        source_sha256=(
            hashlib.sha256(
                raw
            ).hexdigest()
        ),
        published_path=str(
            decision_path
        ),
        selected_model="calibration",
    )

    result = (
        ProductionChampionAudit()
        .audit(
            registry_root=registry,
            snapshot_root=(
                tmp_path
                / "snapshots"
            ),
        )
    )

    assert result.status == "FAIL"

    assert any(
        issue.code
        == "selected_model_mismatch"
        for issue in result.issues
    )


def test_audit_fails_when_source_hash_invalid_format(
    tmp_path: Path,
) -> None:
    registry = (
        tmp_path
        / "registry"
    )

    decision_path, publication_path = (
        _registry_paths(
            registry
        )
    )

    _write_decision(
        decision_path,
        selected_model="baseline",
    )

    _write_publication(
        publication_path,
        source_path="source.json",
        source_sha256="not-a-sha256",
        published_path=str(
            decision_path
        ),
        selected_model="baseline",
    )

    result = (
        ProductionChampionAudit()
        .audit(
            registry_root=registry,
            snapshot_root=(
                tmp_path
                / "snapshots"
            ),
        )
    )

    assert result.status == "FAIL"

    assert any(
        issue.code
        == "source_sha256_invalid"
        for issue in result.issues
    )


def test_audit_fails_when_active_hash_does_not_match(
    tmp_path: Path,
) -> None:
    registry = (
        tmp_path
        / "registry"
    )

    decision_path, publication_path = (
        _registry_paths(
            registry
        )
    )

    _write_decision(
        decision_path,
        selected_model="baseline",
    )

    _write_publication(
        publication_path,
        source_path="source.json",
        source_sha256="0" * 64,
        published_path=str(
            decision_path
        ),
        selected_model="baseline",
    )

    result = (
        ProductionChampionAudit()
        .audit(
            registry_root=registry,
            snapshot_root=(
                tmp_path
                / "snapshots"
            ),
        )
    )

    assert result.status == "FAIL"

    assert any(
        issue.code
        == "active_hash_mismatch"
        for issue in result.issues
    )


@pytest.mark.parametrize(
    (
        "selected_model",
        "expected_resolved",
        "expected_fallback",
    ),
    (
        (
            "baseline",
            "baseline",
            False,
        ),
        (
            "calibration",
            "calibration",
            False,
        ),
        (
            "bayesian",
            "bayesian",
            False,
        ),
        (
            "combined",
            "combined",
            False,
        ),
        (
            None,
            "baseline",
            True,
        ),
    ),
)
def test_audit_reuses_activation_policy(
    tmp_path: Path,
    selected_model: str | None,
    expected_resolved: str,
    expected_fallback: bool,
) -> None:
    registry = (
        tmp_path
        / "registry"
    )

    decision_path, publication_path = (
        _registry_paths(
            registry
        )
    )

    raw = _write_decision(
        decision_path,
        selected_model=selected_model,
    )

    _write_publication(
        publication_path,
        source_path="source.json",
        source_sha256=(
            hashlib.sha256(
                raw
            ).hexdigest()
        ),
        published_path=str(
            decision_path
        ),
        selected_model=selected_model,
    )

    snapshot_root = (
        tmp_path
        / "snapshots"
    )

    if selected_model in {
        "calibration",
        "combined",
    }:
        (
            snapshot_root
            / "regime-calibration"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    if selected_model in {
        "bayesian",
        "combined",
    }:
        (
            snapshot_root
            / "regime-bayesian"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    result = (
        ProductionChampionAudit()
        .audit(
            registry_root=registry,
            snapshot_root=(
                snapshot_root
            ),
        )
    )

    assert (
        result.resolved_model
        == expected_resolved
    )

    assert (
        result.fallback_applied
        is expected_fallback
    )
