"""JSON writer for historical champion decisions."""

from __future__ import annotations

import json
from pathlib import Path

from tools.validation.model_evaluation_champion import (
    HistoricalChampionSelection,
)


class ChampionDecisionReportWriter:
    """Write historical champion decisions as deterministic JSON."""

    def write_json(
        self,
        *,
        report: HistoricalChampionSelection,
        output: Path,
    ) -> Path:
        if not isinstance(
            report,
            HistoricalChampionSelection,
        ):
            raise TypeError(
                "report must be a "
                "HistoricalChampionSelection"
            )

        output = Path(output)

        if output.is_dir():
            raise IsADirectoryError(output)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = (
            json.dumps(
                report.as_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

        output.write_text(
            payload,
            encoding="utf-8",
        )

        return output
