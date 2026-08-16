"""Production champion registry path contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProductionChampionRegistry:
    """Resolve the deterministic active champion decision path."""

    root: Path

    def __init__(
        self,
        root: str | Path,
    ) -> None:
        object.__setattr__(
            self,
            "root",
            Path(root),
        )

    @property
    def active_decision_path(self) -> Path:
        return (
            self.root
            / "active"
            / "champion_decision.json"
        )

    def decision_path(self) -> Path:
        root = self.root

        if not root.exists():
            raise FileNotFoundError(root)

        if not root.is_dir():
            raise NotADirectoryError(root)

        active_root = (
            root
            / "active"
        )

        if not active_root.exists():
            raise FileNotFoundError(
                active_root
            )

        if not active_root.is_dir():
            raise NotADirectoryError(
                active_root
            )

        decision = (
            self.active_decision_path
        )

        if not decision.exists():
            raise FileNotFoundError(
                decision
            )

        if decision.is_dir():
            raise IsADirectoryError(
                decision
            )

        if not decision.is_file():
            raise FileNotFoundError(
                decision
            )

        return decision
