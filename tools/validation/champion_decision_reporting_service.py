"""Reporting service for historical champion decisions."""

from __future__ import annotations

from pathlib import Path

from tools.validation.champion_decision_report_writer import (
    ChampionDecisionReportWriter,
)
from tools.validation.model_evaluation_champion import (
    HistoricalChampionSelection,
)


CHAMPION_DECISION_JSON = "champion_decision.json"


class ChampionDecisionReportingService:
    """Write historical champion decisions to standard artifacts."""

    def __init__(
        self,
        *,
        writer: ChampionDecisionReportWriter | None = None,
    ) -> None:
        self._writer = (
            writer
            if writer is not None
            else ChampionDecisionReportWriter()
        )

    def write(
        self,
        *,
        report: HistoricalChampionSelection,
        output_root: Path,
    ) -> Path:
        if not isinstance(
            report,
            HistoricalChampionSelection,
        ):
            raise TypeError(
                "report must be a "
                "HistoricalChampionSelection"
            )

        root = Path(output_root)

        if root.exists() and not root.is_dir():
            raise NotADirectoryError(root)

        root.mkdir(
            parents=True,
            exist_ok=True,
        )

        output = (
            root
            / CHAMPION_DECISION_JSON
        )

        return self._writer.write_json(
            report=report,
            output=output,
        )
