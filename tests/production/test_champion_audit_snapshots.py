from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lrp.production import (
    ProductionChampionAudit,
)


def _build_registry(
    root: Path,
    *,
    selected_model: str | None,
) -> Path:
    registry = (
        root
        / "registry"
    )

    active = (
        registry
        / "active"
    )

    active.mkdir(
        parents=True,
        exist_ok=True,
    )

    decision_path = (
        active
        / "champion_decision.json"
    )

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
    ).encode(
        "utf-8"
    )

    decision_path.write_bytes(
        raw
    )

    publication = {
        "source_path": "source.json",
        "source_sha256": (
            hashlib.sha256(
                raw
            ).hexdigest()
        ),
        "published_path": str(
            decision_path
        ),
        "published_at_kst": (
            "2026-08-16T22:00:00+09:00"
        ),
        "selected_model": selected_model,
    }

    (
        active
        / "publication.json"
    ).write_text(
        json.dumps(
            publication,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return registry


def _issue_codes(
    result: object,
) -> set[str]:
    return {
        issue.code
        for issue in result.issues
    }


def _check_names(
    result: object,
) -> set[str]:
    return {
        check.name
        for check in result.checks
    }


def test_baseline_does_not_require_snapshots(
    tmp_path: Path,
) -> None:
    registry = _build_registry(
        tmp_path,
        selected_model="baseline",
    )

    result = (
        ProductionChampionAudit()
        .audit(
            registry_root=registry,
            snapshot_root=(
                tmp_path
                / "missing_snapshots"
            ),
        )
    )

    assert result.status == "PASS"

    assert (
        "calibration_snapshot_missing"
        not in _issue_codes(result)
    )

    assert (
        "bayesian_snapshot_missing"
        not in _issue_codes(result)
    )


def test_fallback_baseline_does_not_require_snapshots(
    tmp_path: Path,
) -> None:
    registry = _build_registry(
        tmp_path,
        selected_model=None,
    )

    result = (
        ProductionChampionAudit()
        .audit(
            registry_root=registry,
            snapshot_root=(
                tmp_path
                / "missing_snapshots"
            ),
        )
    )

    assert result.status == "WARN"

    assert (
        _issue_codes(result)
        == {
            "baseline_fallback",
        }
    )


def test_calibration_requires_calibration_snapshot(
    tmp_path: Path,
) -> None:
    registry = _build_registry(
        tmp_path,
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

    assert (
        "calibration_snapshot_missing"
        in _issue_codes(result)
    )


def test_bayesian_requires_bayesian_snapshot(
    tmp_path: Path,
) -> None:
    registry = _build_registry(
        tmp_path,
        selected_model="bayesian",
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

    assert (
        "bayesian_snapshot_missing"
        in _issue_codes(result)
    )


def test_combined_requires_both_snapshots(
    tmp_path: Path,
) -> None:
    registry = _build_registry(
        tmp_path,
        selected_model="combined",
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

    codes = _issue_codes(
        result
    )

    assert (
        "calibration_snapshot_missing"
        in codes
    )

    assert (
        "bayesian_snapshot_missing"
        in codes
    )


def test_calibration_passes_with_required_snapshot(
    tmp_path: Path,
) -> None:
    registry = _build_registry(
        tmp_path,
        selected_model="calibration",
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
    )

    result = (
        ProductionChampionAudit()
        .audit(
            registry_root=registry,
            snapshot_root=snapshot_root,
        )
    )

    assert result.status == "PASS"

    assert (
        "regime_calibration_snapshot"
        in _check_names(result)
    )


def test_bayesian_passes_with_required_snapshot(
    tmp_path: Path,
) -> None:
    registry = _build_registry(
        tmp_path,
        selected_model="bayesian",
    )

    snapshot_root = (
        tmp_path
        / "snapshots"
    )

    (
        snapshot_root
        / "regime-bayesian"
    ).mkdir(
        parents=True,
    )

    result = (
        ProductionChampionAudit()
        .audit(
            registry_root=registry,
            snapshot_root=snapshot_root,
        )
    )

    assert result.status == "PASS"

    assert (
        "regime_bayesian_snapshot"
        in _check_names(result)
    )


def test_combined_passes_with_both_snapshots(
    tmp_path: Path,
) -> None:
    registry = _build_registry(
        tmp_path,
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
    )

    (
        snapshot_root
        / "regime-bayesian"
    ).mkdir(
        parents=True,
    )

    result = (
        ProductionChampionAudit()
        .audit(
            registry_root=registry,
            snapshot_root=snapshot_root,
        )
    )

    assert result.status == "PASS"

    names = _check_names(
        result
    )

    assert (
        "regime_calibration_snapshot"
        in names
    )

    assert (
        "regime_bayesian_snapshot"
        in names
    )


def test_snapshot_path_must_be_directory(
    tmp_path: Path,
) -> None:
    registry = _build_registry(
        tmp_path,
        selected_model="calibration",
    )

    snapshot_root = (
        tmp_path
        / "snapshots"
    )

    snapshot_root.mkdir(
        parents=True,
    )

    (
        snapshot_root
        / "regime-calibration"
    ).write_text(
        "not a directory\n",
        encoding="utf-8",
    )

    result = (
        ProductionChampionAudit()
        .audit(
            registry_root=registry,
            snapshot_root=snapshot_root,
        )
    )

    assert result.status == "FAIL"

    assert (
        "calibration_snapshot_invalid"
        in _issue_codes(result)
    )


def test_unused_snapshot_is_not_required(
    tmp_path: Path,
) -> None:
    registry = _build_registry(
        tmp_path,
        selected_model="calibration",
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
    )

    # Bayesian directory intentionally absent.
    result = (
        ProductionChampionAudit()
        .audit(
            registry_root=registry,
            snapshot_root=snapshot_root,
        )
    )

    assert result.status == "PASS"

    assert (
        "bayesian_snapshot_missing"
        not in _issue_codes(result)
    )