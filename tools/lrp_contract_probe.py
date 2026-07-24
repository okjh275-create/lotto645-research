"""Probe public contracts required by the LRP integration pipeline."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import importlib
import inspect
import json
from pathlib import Path
from typing import Any


OUTPUT_PATH = Path("logs/lrp_contract_probe.json")

PACKAGES = {
    "foundation": {
        "functions": (
            "check_foundation_compatibility",
        ),
        "classes": (
            "ExecutionContext",
            "PluginRegistry",
        ),
    },
    "lotto645_statistics": {
        "functions": (
            "analyze_all",
            "analyze_uncertainty",
            "rolling_backtest",
            "build_feature_matrix",
            "feature_matrix_to_dict",
            "snapshot_to_dict",
        ),
        "classes": (
            "AnalysisConfig",
            "AnalysisReport",
            "StatisticsEngine",
        ),
    },
    "lotto645_candidates": {
        "functions": (
            "validate_statistics_contract",
            "number_signals_from_statistics",
            "generate_candidates",
            "score_candidates",
            "score_candidates_fast",
            "rank_candidates",
            "select_diverse_candidates",
            "select_practical_sets",
        ),
        "classes": (
            "CandidateConfig",
            "RiskFilterConfig",
            "ScoreWeights",
            "RankingConfig",
            "DiversityConfig",
            "PracticalSelectionConfig",
        ),
    },
}


def safe_signature(value: object) -> str | None:
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return None


def describe_class(value: object) -> dict[str, Any]:
    result: dict[str, Any] = {
        "signature": safe_signature(value),
        "module": getattr(value, "__module__", None),
        "qualname": getattr(value, "__qualname__", None),
        "is_dataclass": is_dataclass(value),
    }

    if is_dataclass(value):
        result["fields"] = [
            {
                "name": field.name,
                "type": str(field.type),
                "default": (
                    repr(field.default)
                    if repr(field.default) != "<dataclasses._MISSING_TYPE object"
                    else None
                ),
                "default_factory": (
                    repr(field.default_factory)
                    if repr(field.default_factory)
                    != "<dataclasses._MISSING_TYPE object"
                    else None
                ),
            }
            for field in fields(value)
        ]

    annotations = getattr(value, "__annotations__", None)
    if isinstance(annotations, dict):
        result["annotations"] = {
            str(name): str(annotation)
            for name, annotation in annotations.items()
        }

    return result


def inspect_package(
    package_name: str,
    spec: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    try:
        module = importlib.import_module(package_name)
    except Exception as exc:
        return {
            "installed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    exports = getattr(module, "__all__", ())
    stable_exports = getattr(module, "STABLE_EXPORTS", ())

    result: dict[str, Any] = {
        "installed": True,
        "module_file": getattr(module, "__file__", None),
        "version": getattr(module, "__version__", None),
        "stable_api_version": getattr(
            module,
            "STABLE_API_VERSION",
            None,
        ),
        "exports": (
            sorted(str(item) for item in exports)
            if isinstance(exports, (tuple, list, set))
            else []
        ),
        "stable_exports": (
            sorted(str(item) for item in stable_exports)
            if isinstance(stable_exports, (tuple, list, set))
            else []
        ),
        "functions": {},
        "classes": {},
    }

    for name in spec["functions"]:
        value = getattr(module, name, None)
        result["functions"][name] = {
            "present": value is not None,
            "signature": (
                safe_signature(value)
                if value is not None
                else None
            ),
            "module": getattr(value, "__module__", None),
        }

    for name in spec["classes"]:
        value = getattr(module, name, None)
        result["classes"][name] = (
            {
                "present": False,
            }
            if value is None
            else {
                "present": True,
                **describe_class(value),
            }
        )

    return result


def main() -> int:
    report = {
        "generated_at_utc": (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        ),
        "packages": {
            name: inspect_package(name, spec)
            for name, spec in PACKAGES.items()
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(f"PASS: wrote {OUTPUT_PATH}")

    for package_name, package in report["packages"].items():
        if not package["installed"]:
            print(
                f"PENDING: {package_name}: "
                f"{package.get('error', 'not importable')}"
            )
            continue

        print(
            f"PASS: {package_name} "
            f"version={package.get('version')} "
            f"api={package.get('stable_api_version')}"
        )

        missing_functions = [
            name
            for name, item in package["functions"].items()
            if not item["present"]
        ]
        missing_classes = [
            name
            for name, item in package["classes"].items()
            if not item["present"]
        ]

        if missing_functions:
            print(
                "  missing functions:",
                ", ".join(missing_functions),
            )
        if missing_classes:
            print(
                "  missing classes:",
                ", ".join(missing_classes),
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
