"""Read-only access to persisted round-completion artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .runtime import verify_manifest


_ROUND_DIRECTORY = re.compile(r"^round_(\d+)$")


class RoundCompletionRepository:
    """Read persisted round-completion operational artifacts."""

    def __init__(
        self,
        root: str | Path,
    ) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def list_rounds(self) -> tuple[int, ...]:
        if not self.root.is_dir():
            return ()

        rounds: list[int] = []

        for path in self.root.iterdir():
            if not path.is_dir():
                continue

            match = _ROUND_DIRECTORY.fullmatch(
                path.name
            )

            if match is None:
                continue

            data_path = (
                path / "round_completion.json"
            )

            if not data_path.is_file():
                continue

            rounds.append(int(match.group(1)))

        return tuple(sorted(rounds))

    def load_round(
        self,
        round_no: int,
    ) -> dict[str, Any] | None:
        round_no = self._round_no(round_no)

        path = (
            self.root
            / f"round_{round_no:04d}"
            / "round_completion.json"
        )

        if not path.is_file():
            return None

        payload = json.loads(
            path.read_text(
                encoding="utf-8-sig"
            )
        )

        if not isinstance(payload, dict):
            raise TypeError(
                "round completion artifact must be "
                "a JSON object"
            )

        artifact_round = payload.get(
            "round_no"
        )

        if artifact_round != round_no:
            raise ValueError(
                "round completion artifact round "
                "does not match its directory"
            )

        return payload

    def latest(self) -> dict[str, Any] | None:
        rounds = self.list_rounds()

        if not rounds:
            return None

        return self.load_round(rounds[-1])

    def recent(
        self,
        limit: int = 20,
    ) -> tuple[dict[str, Any], ...]:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
        ):
            raise TypeError(
                "limit must be an integer"
            )

        if limit < 1:
            raise ValueError(
                "limit must be greater than or equal to 1"
            )

        selected = self.list_rounds()[-limit:]

        records: list[dict[str, Any]] = []

        for round_no in reversed(selected):
            record = self.load_round(round_no)

            if record is not None:
                records.append(record)

        return tuple(records)

    def verify_round(
        self,
        round_no: int,
    ) -> dict[str, Any]:
        round_no = self._round_no(round_no)

        manifest = (
            self.root
            / f"round_{round_no:04d}"
            / "manifest.json"
        )

        if not manifest.is_file():
            return {
                "status": "FAIL",
                "round": round_no,
                "reason": "manifest_missing",
                "manifest": str(
                    manifest.resolve()
                ),
            }

        result = verify_manifest(manifest)

        return {
            "round": round_no,
            **result,
        }

    @staticmethod
    def _round_no(value: int) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise TypeError(
                "round_no must be an integer"
            )

        if value < 1:
            raise ValueError(
                "round_no must be greater than or equal to 1"
            )

        return value
