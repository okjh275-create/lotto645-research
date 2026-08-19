from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lrp.production.production_registry_lock import (
    ProductionRegistryWriterLock,
)


@dataclass(frozen=True)
class ProductionChampionActiveSnapshot:
    decision_path: Path
    publication_path: Path
    decision_bytes: bytes
    publication_bytes: bytes


class ProductionChampionActiveSnapshotReader:
    def __init__(
        self,
        *,
        timeout: float = 5.0,
    ) -> None:
        self._timeout = timeout

    def read(
        self,
        registry_root: str | Path,
    ) -> ProductionChampionActiveSnapshot:
        root = Path(
            registry_root
        )

        decision_path = (
            root
            / "active"
            / "champion_decision.json"
        )

        publication_path = (
            root
            / "active"
            / "publication.json"
        )

        with ProductionRegistryWriterLock(
            root,
            timeout=self._timeout,
        ):
            if not decision_path.is_file():
                raise FileNotFoundError(
                    decision_path
                )

            if not publication_path.is_file():
                raise FileNotFoundError(
                    publication_path
                )

            decision_bytes = (
                decision_path.read_bytes()
            )

            publication_bytes = (
                publication_path.read_bytes()
            )

        return ProductionChampionActiveSnapshot(
            decision_path=decision_path,
            publication_path=publication_path,
            decision_bytes=decision_bytes,
            publication_bytes=publication_bytes,
        )
