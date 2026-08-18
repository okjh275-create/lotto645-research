from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _history_path() -> Path:
    return Path(
        "artifacts/validation/"
        "project_r_release_readiness/"
        "r04_real_model_evaluation_e2e/"
        "history.json"
    )


def _registry_path() -> Path:
    return Path(
        "artifacts/validation/"
        "project_r_release_readiness/"
        "r05_real_release_lifecycle_e2e/"
        "registry"
    )


def _run_prediction(
    *,
    tmp_path: Path,
    production: bool,
) -> Path:
    history = _history_path()

    assert history.is_file()

    output = tmp_path / "prediction"

    command = [
        sys.executable,
        "-m",
        "lrp",
        "predict",
        "--history",
        str(history),
        "--round",
        "1232",
        "--seed",
        "20260817",
        "--mode",
        "fast",
        "--candidate-count",
        "100",
        "--top-k",
        "10",
        "--practical-k",
        "5",
        "--output",
        str(output),
    ]

    if production:
        registry = _registry_path()

        assert registry.is_dir()

        command.extend(
            [
                "--production-registry",
                str(registry),
                "--production-snapshot-root",
                "artifacts/production/snapshots",
            ]
        )

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        result.stdout
        + "\n"
        + result.stderr
    )

    return output


def test_production_activation_is_persisted_in_prediction_artifact(
    tmp_path: Path,
) -> None:
    output = _run_prediction(
        tmp_path=tmp_path,
        production=True,
    )

    prediction_path = next(
        output.rglob("prediction.json")
    )

    payload = json.loads(
        prediction_path.read_text(
            encoding="utf-8-sig"
        )
    )

    activation = payload.get(
        "production_activation"
    )

    assert activation == {
        "enabled": True,
        "requested_model": None,
        "resolved_model": "baseline",
        "fallback_applied": True,
        "fallback_reason": "no_selected_model",
    }


def test_production_activation_is_persisted_in_manifest_metadata(
    tmp_path: Path,
) -> None:
    output = _run_prediction(
        tmp_path=tmp_path,
        production=True,
    )

    manifest_path = next(
        output.rglob("manifest.json")
    )

    payload = json.loads(
        manifest_path.read_text(
            encoding="utf-8-sig"
        )
    )

    activation = (
        payload
        .get("metadata", {})
        .get("production_activation")
    )

    assert activation == {
        "enabled": True,
        "requested_model": None,
        "resolved_model": "baseline",
        "fallback_applied": True,
        "fallback_reason": "no_selected_model",
    }


def test_non_production_prediction_persists_disabled_activation(
    tmp_path: Path,
) -> None:
    output = _run_prediction(
        tmp_path=tmp_path,
        production=False,
    )

    prediction_path = next(
        output.rglob("prediction.json")
    )

    payload = json.loads(
        prediction_path.read_text(
            encoding="utf-8-sig"
        )
    )

    activation = payload.get(
        "production_activation"
    )

    assert activation == {
        "enabled": False,
        "requested_model": None,
        "resolved_model": None,
        "fallback_applied": False,
        "fallback_reason": None,
    }
