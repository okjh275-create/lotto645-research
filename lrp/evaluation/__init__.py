"""Prediction-model evaluation public API."""

from .contracts import (
    EvaluationWindow,
    ModelEvaluation,
    WindowEvaluation,
    build_model_evaluation,
)
from .promotion import (
    ChampionPromotionDecision,
    ChampionPromotionPolicy,
    evaluate_champion_promotion,
)
from .selection import (
    ChampionSelection,
    select_champion,
)
from .ranking import (
    ChampionRanking,
    ModelRankingEntry,
    rank_model_evaluations,
)

__all__ = [
    "ChampionPromotionDecision",
    "ChampionPromotionPolicy",
    "ChampionRanking",
    "ChampionSelection",
    "EvaluationWindow",
    "ModelEvaluation",
    "ModelRankingEntry",
    "WindowEvaluation",
    "build_model_evaluation",
    "evaluate_champion_promotion",
    "rank_model_evaluations",
    "select_champion",
]
