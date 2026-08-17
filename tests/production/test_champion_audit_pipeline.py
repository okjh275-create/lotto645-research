from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lrp.pipelines import (
    PredictionPipeline,
)
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

    (
        active
        / "publication.json"
    ).write_text(
        json.dumps(
            {
                "source_path": (
                    "source.json"
                ),
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
                "selected_model": (
                    selected_model
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return registry


def _prepare_snapshots(
    root: Path,
    *,
    selected_model: str | None,
) -> Path:
    snapshot_root = (
        root
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

    return snapshot_root


def _check_names(
    result: object,
) -> set[str]:
    return {
        check.name
        for check in result.checks
    }


def _issue_codes(
    result: object,
) -> set[str]:
    return {
        issue.code
        for issue in result.issues
    }


def test_pipeline_load_check_is_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _build_registry(
        tmp_path,
        selected_model="baseline",
    )

    calls: list[
        dict[str, object]
    ] = []

    original = (
        PredictionPipeline.load
    )

    def fake_load(
        **kwargs: object,
    ) -> object:
        calls.append(
            dict(kwargs)
        )

        return object()

    monkeypatch.setattr(
        PredictionPipeline,
        "load",
        fake_load,
    )

    try:
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
    finally:
        monkeypatch.setattr(
            PredictionPipeline,
            "load",
            original,
        )

    assert result.status == "PASS"

    assert (
        "pipeline_load"
        in _check_names(result)
    )

    assert len(calls) == 1

    assert calls[0] == {
        "regime_calibration_snapshot_root": (
            None
        ),
        "regime_bayesian_snapshot_root": (
            None
        ),
    }


def test_pipeline_load_failure_returns_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _build_registry(
        tmp_path,
        selected_model="baseline",
    )

    def failing_load(
        **kwargs: object,
    ) -> object:
        raise RuntimeError(
            "synthetic pipeline failure"
        )

    monkeypatch.setattr(
        PredictionPipeline,
        "load",
        failing_load,
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
        "pipeline_load_failed"
        in _issue_codes(result)
    )

    pipeline_checks = [
        check
        for check in result.checks
        if check.name
        == "pipeline_load"
    ]

    assert len(
        pipeline_checks
    ) == 1

    assert (
        pipeline_checks[0].status
        == "FAIL"
    )


def test_fallback_baseline_preserves_warn_after_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _build_registry(
        tmp_path,
        selected_model=None,
    )

    calls = 0

    def fake_load(
        **kwargs: object,
    ) -> object:
        nonlocal calls

        calls += 1

        return object()

    monkeypatch.setattr(
        PredictionPipeline,
        "load",
        fake_load,
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

    assert calls == 1

    assert result.status == "WARN"

    assert result.resolved_model == "baseline"

    assert result.fallback_applied is True

    assert (
        "baseline_fallback"
        in _issue_codes(result)
    )

    assert (
        "pipeline_load"
        in _check_names(result)
    )


@pytest.mark.parametrize(
    "selected_model",
    (
        "calibration",
        "bayesian",
        "combined",
    ),
)
def test_pipeline_receives_configuration_kwargs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected_model: str,
) -> None:
    registry = _build_registry(
        tmp_path,
        selected_model=selected_model,
    )

    snapshot_root = (
        _prepare_snapshots(
            tmp_path,
            selected_model=(
                selected_model
            ),
        )
    )

    calls: list[
        dict[str, object]
    ] = []

    def fake_load(
        **kwargs: object,
    ) -> object:
        calls.append(
            dict(kwargs)
        )

        return object()

    monkeypatch.setattr(
        PredictionPipeline,
        "load",
        fake_load,
    )

    result = (
        ProductionChampionAudit()
        .audit(
            registry_root=registry,
            snapshot_root=snapshot_root,
        )
    )

    assert result.status == "PASS"

    assert len(calls) == 1

    kwargs = calls[0]

    if selected_model in {
        "calibration",
        "combined",
    }:
        assert (
            kwargs[
                "regime_calibration_snapshot_root"
            ]
            == (
                snapshot_root
                / "regime-calibration"
            )
        )
    else:
        assert (
            kwargs[
                "regime_calibration_snapshot_root"
            ]
            is None
        )

    if selected_model in {
        "bayesian",
        "combined",
    }:
        assert (
            kwargs[
                "regime_bayesian_snapshot_root"
            ]
            == (
                snapshot_root
                / "regime-bayesian"
            )
        )
    else:
        assert (
            kwargs[
                "regime_bayesian_snapshot_root"
            ]
            is None
        )


def test_pipeline_not_loaded_when_snapshot_audit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _build_registry(
        tmp_path,
        selected_model="combined",
    )

    calls = 0

    def fake_load(
        **kwargs: object,
    ) -> object:
        nonlocal calls

        calls += 1

        return object()

    monkeypatch.setattr(
        PredictionPipeline,
        "load",
        fake_load,
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

    assert result.status == "FAIL"

    assert calls == 0

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

    assert (
        "pipeline_load"
        not in _check_names(result)
    )


def test_pipeline_failure_preserves_activation_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _build_registry(
        tmp_path,
        selected_model="combined",
    )

    snapshot_root = (
        _prepare_snapshots(
            tmp_path,
            selected_model="combined",
        )
    )

    def failing_load(
        **kwargs: object,
    ) -> object:
        raise RuntimeError(
            "synthetic pipeline failure"
        )

    monkeypatch.setattr(
        PredictionPipeline,
        "load",
        failing_load,
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


def test_pipeline_failure_on_fallback_preserves_fallback_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _build_registry(
        tmp_path,
        selected_model=None,
    )

    def failing_load(
        **kwargs: object,
    ) -> object:
        raise RuntimeError(
            "synthetic pipeline failure"
        )

    monkeypatch.setattr(
        PredictionPipeline,
        "load",
        failing_load,
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

    assert result.selected_model is None

    assert result.resolved_model == "baseline"

    assert result.fallback_applied is True

    assert (
        result.fallback_reason
        == "no_selected_model"
    )

    codes = _issue_codes(
        result
    )

    assert (
        "pipeline_load_failed"
        in codes
    )