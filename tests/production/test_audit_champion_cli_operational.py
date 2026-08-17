from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def _write_registry(
    root: Path,
    *,
    selected_model: str | None,
) -> Path:
    registry = root / "registry"
    active = registry / "active"

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
    ).encode("utf-8")

    decision_path.write_bytes(
        raw
    )

    (
        active
        / "publication.json"
    ).write_text(
        json.dumps(
            {
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
                    "2026-08-17T22:00:00+09:00"
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


def test_main_pass_returns_zero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import lrp.cli.audit_champion as cli

    registry = _write_registry(
        tmp_path,
        selected_model="baseline",
    )

    exit_code = cli.main(
        [
            "--production-registry",
            str(registry),
            "--snapshot-root",
            str(tmp_path / "snapshots"),
        ]
    )

    assert exit_code == 0

    captured = capsys.readouterr()

    payload = json.loads(
        captured.out
    )

    assert payload["status"] == "PASS"
    assert captured.err == ""


def test_main_warn_returns_zero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import lrp.cli.audit_champion as cli

    registry = _write_registry(
        tmp_path,
        selected_model=None,
    )

    exit_code = cli.main(
        [
            "--production-registry",
            str(registry),
            "--snapshot-root",
            str(tmp_path / "snapshots"),
        ]
    )

    assert exit_code == 0

    payload = json.loads(
        capsys.readouterr().out
    )

    assert payload["status"] == "WARN"
    assert payload["fallback_applied"] is True


def test_main_fail_returns_one(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import lrp.cli.audit_champion as cli

    registry = _write_registry(
        tmp_path,
        selected_model="combined",
    )

    exit_code = cli.main(
        [
            "--production-registry",
            str(registry),
            "--snapshot-root",
            str(
                tmp_path
                / "missing_snapshots"
            ),
        ]
    )

    assert exit_code == 1

    payload = json.loads(
        capsys.readouterr().out
    )

    assert payload["status"] == "FAIL"


def test_unexpected_error_uses_stderr(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lrp.cli.audit_champion as cli

    def fail(
        **kwargs: object,
    ) -> dict[str, object]:
        raise RuntimeError(
            "synthetic CLI failure"
        )

    monkeypatch.setattr(
        cli,
        "run_audit",
        fail,
    )

    exit_code = cli.main(
        [
            "--production-registry",
            str(tmp_path / "registry"),
            "--snapshot-root",
            str(tmp_path / "snapshots"),
        ]
    )

    assert exit_code == 1

    captured = capsys.readouterr()

    assert captured.out == ""

    payload = json.loads(
        captured.err
    )

    assert payload["status"] == "ERROR"
    assert (
        payload["error_type"]
        == "RuntimeError"
    )