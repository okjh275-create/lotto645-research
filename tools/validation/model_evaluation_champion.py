from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lrp.evaluation import (
    ChampionPromotionPolicy,
    ChampionSelection,
    select_champion,
)

from tools.validation.model_evaluation_matrix import (
    HistoricalEvaluationMatrix,
)


@dataclass(frozen=True)
class HistoricalChampionSelection:
    matrix: HistoricalEvaluationMatrix
    selection: ChampionSelection

    @property
    def ranking_champion(self) -> str:
        return self.selection.ranking_champion

    @property
    def selected_model(self) -> str:
        return self.selection.selected_model

    def as_dict(self) -> dict[str, Any]:
        return {
            "matrix": self.matrix.as_dict(),
            "selection": self.selection.as_dict(),
            "ranking_champion": self.ranking_champion,
            "selected_model": self.selected_model,
        }


def select_historical_champion(
    *,
    matrix: HistoricalEvaluationMatrix,
    policy: ChampionPromotionPolicy | None = None,
) -> HistoricalChampionSelection:
    if not isinstance(
        matrix,
        HistoricalEvaluationMatrix,
    ):
        raise TypeError(
            "matrix must be a HistoricalEvaluationMatrix"
        )

    selection = select_champion(
        entries=matrix.ranking.entries,
        policy=policy,
    )

    return HistoricalChampionSelection(
        matrix=matrix,
        selection=selection,
    )
