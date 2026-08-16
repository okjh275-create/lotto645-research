"""Read the active production champion decision from a registry."""

from __future__ import annotations

from pathlib import Path

from lrp.production.champion_decision import (
    ProductionChampionDecision,
)
from lrp.production.champion_decision_reader import (
    ProductionChampionDecisionReader,
)
from lrp.production.champion_registry import (
    ProductionChampionRegistry,
)


class ProductionChampionRegistryReader:
    """Resolve and read the active champion decision."""

    def read(
        self,
        root: str | Path,
    ) -> ProductionChampionDecision:
        registry = ProductionChampionRegistry(
            root=root,
        )

        decision_path = (
            registry.decision_path()
        )

        return (
            ProductionChampionDecisionReader()
            .read(
                decision_path
            )
        )
