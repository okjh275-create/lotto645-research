"""Reader for production champion decision artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from lrp.production.champion_decision import (
    ProductionChampionDecision,
)


class ProductionChampionDecisionReader:
    """Read a champion decision artifact into the production contract."""

    def read(
        self,
        path: str | Path,
    ) -> ProductionChampionDecision:
        resolved = Path(path)

        if not resolved.exists():
            raise FileNotFoundError(
                resolved
            )

        if resolved.is_dir():
            raise IsADirectoryError(
                resolved
            )

        payload = json.loads(
            resolved.read_text(
                encoding="utf-8"
            )
        )

        return (
            ProductionChampionDecision
            .from_payload(
                payload
            )
        )
