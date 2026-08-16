"""Champion selection integration for Project M."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .promotion import (
    ChampionPromotionDecision,
    ChampionPromotionPolicy,
    evaluate_champion_promotion,
)
from .ranking import ModelRankingEntry


@dataclass(frozen=True)
class ChampionSelection:
    """Result of ranking-to-promotion champion selection."""

    ranking_champion: str | None
    promotion: ChampionPromotionDecision
    selected_model: str | None

    def __post_init__(self) -> None:
        if not isinstance(
            self.promotion,
            ChampionPromotionDecision,
        ):
            raise TypeError(
                "promotion must be ChampionPromotionDecision"
            )

        if self.ranking_champion is not None:
            if not isinstance(
                self.ranking_champion,
                str,
            ):
                raise TypeError(
                    "ranking_champion must be str or None"
                )

            if not self.ranking_champion:
                raise ValueError(
                    "ranking_champion must not be empty"
                )

        if self.selected_model is not None:
            if not isinstance(
                self.selected_model,
                str,
            ):
                raise TypeError(
                    "selected_model must be str or None"
                )

            if not self.selected_model:
                raise ValueError(
                    "selected_model must not be empty"
                )

        if self.promotion.promoted:
            if (
                self.selected_model
                != self.promotion.promoted_model
            ):
                raise ValueError(
                    "selected_model must match promoted_model"
                )
        elif self.selected_model is not None:
            raise ValueError(
                "rejected promotion cannot select a model"
            )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation."""

        return {
            "ranking_champion": self.ranking_champion,
            "promotion": self.promotion.as_dict(),
            "selected_model": self.selected_model,
        }


def select_champion(
    *,
    entries: tuple[ModelRankingEntry, ...],
    policy: ChampionPromotionPolicy | None = None,
) -> ChampionSelection:
    """Apply promotion policy to ranked model entries."""

    normalized = tuple(entries)

    promotion = evaluate_champion_promotion(
        entries=normalized,
        policy=policy,
    )

    ranking_champion = next(
        (
            entry.model_name
            for entry in normalized
            if entry.eligible
        ),
        None,
    )

    selected_model = (
        promotion.promoted_model
        if promotion.promoted
        else None
    )

    return ChampionSelection(
        ranking_champion=ranking_champion,
        promotion=promotion,
        selected_model=selected_model,
    )
