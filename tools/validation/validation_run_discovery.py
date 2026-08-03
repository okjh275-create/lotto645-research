"""Discover and validate local validation-run artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar


@dataclass(frozen=True, slots=True)
class ValidationRunRecord:
    """Discovered validation-run metadata."""

    run_type: str
    root: Path
    start_round: int
    end_round: int
    round_count: int
    policy_name: str | None
    files: dict[str, str]
    missing_files: tuple[str, ...]
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_type": self.run_type,
            "root": str(self.root),
            "start_round": self.start_round,
            "end_round": self.end_round,
            "round_count": self.round_count,
            "policy_name": self.policy_name,
            "files": dict(self.files),
            "missing_files": list(
                self.missing_files
            ),
            "status": self.status,
        }


class ValidationRunDiscovery:
    """Discover replay and policy-comparison validation runs."""

    REPLAY_REQUIRED_FILES: ClassVar[
        tuple[str, ...]
    ] = (
        "replay_rounds.jsonl",
        "replay_summary.json",
    )

    POLICY_REQUIRED_FILES: ClassVar[
        tuple[str, ...]
    ] = (
        "policy_comparison.json",
    )

    OPTIONAL_FILES: ClassVar[
        tuple[str, ...]
    ] = (
        "effectiveness_report.json",
        "feature_attribution_effectiveness.json",
        "feature_attribution_stability.json",
    )

    def discover(
        self,
        root: Path,
    ) -> tuple[ValidationRunRecord, ...]:
        root = Path(root)

        if not root.exists():
            raise FileNotFoundError(root)

        if not root.is_dir():
            raise NotADirectoryError(root)

        records: list[
            ValidationRunRecord
        ] = []

        policy_files = sorted(
            root.rglob(
                "policy_comparison.json"
            )
        )

        policy_roots = {
            path.parent.resolve()
            for path in policy_files
        }

        for path in policy_files:
            records.append(
                self._policy_record(path)
            )

        for path in sorted(
            root.rglob(
                "replay_summary.json"
            )
        ):
            run_root = path.parent

            if any(
                policy_root
                in run_root.resolve().parents
                or run_root.resolve()
                == policy_root
                for policy_root in policy_roots
            ):
                continue

            records.append(
                self._replay_record(path)
            )

        records.sort(
            key=lambda item: (
                item.start_round,
                item.end_round,
                item.run_type,
                item.policy_name or "",
                str(item.root),
            )
        )

        return tuple(records)

    def _replay_record(
        self,
        summary_path: Path,
    ) -> ValidationRunRecord:
        payload = self._load_object(
            summary_path
        )
        summary = self._mapping(
            payload,
            "summary",
        )
        config = self._mapping(
            payload,
            "config",
        )

        start_round = self._integer(
            config,
            "start_round",
        )
        end_round = self._integer(
            config,
            "end_round",
        )
        round_count = self._integer(
            summary,
            "round_count",
        )

        self._validate_window(
            start_round=start_round,
            end_round=end_round,
            round_count=round_count,
        )

        return self._build_record(
            run_type="replay",
            run_root=summary_path.parent,
            start_round=start_round,
            end_round=end_round,
            round_count=round_count,
            policy_name=None,
            required_files=(
                self.REPLAY_REQUIRED_FILES
            ),
        )

    def _policy_record(
        self,
        comparison_path: Path,
    ) -> ValidationRunRecord:
        payload = self._load_object(
            comparison_path
        )
        config = self._mapping(
            payload,
            "config",
        )

        start_round = self._integer(
            config,
            "start_round",
        )
        end_round = self._integer(
            config,
            "end_round",
        )
        round_count = (
            end_round - start_round + 1
        )

        scenario_count = self._integer(
            payload,
            "scenario_count",
        )

        if scenario_count < 1:
            raise ValueError(
                "scenario_count must be "
                "greater than or equal to 1"
            )

        return self._build_record(
            run_type="policy_comparison",
            run_root=comparison_path.parent,
            start_round=start_round,
            end_round=end_round,
            round_count=round_count,
            policy_name=None,
            required_files=(
                self.POLICY_REQUIRED_FILES
            ),
        )

    def _build_record(
        self,
        *,
        run_type: str,
        run_root: Path,
        start_round: int,
        end_round: int,
        round_count: int,
        policy_name: str | None,
        required_files: tuple[str, ...],
    ) -> ValidationRunRecord:
        files: dict[str, str] = {}
        missing: list[str] = []

        for name in (
            *required_files,
            *self.OPTIONAL_FILES,
        ):
            path = run_root / name

            if path.is_file():
                files[name] = str(path)
            elif name in required_files:
                missing.append(name)

        return ValidationRunRecord(
            run_type=run_type,
            root=run_root,
            start_round=start_round,
            end_round=end_round,
            round_count=round_count,
            policy_name=policy_name,
            files=files,
            missing_files=tuple(missing),
            status=(
                "PASS"
                if not missing
                else "INCOMPLETE"
            ),
        )

    @staticmethod
    def _validate_window(
        *,
        start_round: int,
        end_round: int,
        round_count: int,
    ) -> None:
        if start_round < 1:
            raise ValueError(
                "start_round must be "
                "greater than or equal to 1"
            )

        if end_round < start_round:
            raise ValueError(
                "end_round must be greater "
                "than or equal to start_round"
            )

        expected = (
            end_round - start_round + 1
        )

        if round_count != expected:
            raise ValueError(
                "round_count does not match "
                "the configured round window"
            )

    @staticmethod
    def _load_object(
        path: Path,
    ) -> dict[str, Any]:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(payload, dict):
            raise TypeError(
                f"{path.name} must contain "
                "a JSON object"
            )

        return payload

    @staticmethod
    def _mapping(
        values: dict[str, Any],
        key: str,
    ) -> dict[str, Any]:
        value = values.get(key)

        if not isinstance(value, dict):
            raise TypeError(
                f"{key} must be an object"
            )

        return value

    @staticmethod
    def _integer(
        values: dict[str, Any],
        key: str,
    ) -> int:
        if key not in values:
            raise ValueError(
                f"missing field: {key}"
            )

        value = values[key]

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise TypeError(
                f"{key} must be an integer"
            )

        return value
