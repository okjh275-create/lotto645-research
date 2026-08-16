"""Champion promotion policy for Project M."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Iterable

from lrp.contracts import ContractError

from .ranking import ModelRankingEntry


@dataclass(frozen=True, slots=True)
class ChampionPromotionPolicy:
    """Thresholds required to promote a ranked model."""

    minimum_composite_score: float = 0.0
    minimum_practical_score: float = 0.0
    minimum_significance_score: float = 1.0
    minimum_composite_margin: float = 0.01

    def __post_init__(self) -> None:
        for field_name in (
            "minimum_composite_score",
            "minimum_practical_score",
            "minimum_significance_score",
            "minimum_composite_margin",
        ):
            value = getattr(self, field_name)

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
            ):
                raise ContractError(
                    f"{field_name} must be finite numeric"
                )

        if self.minimum_significance_score < 0.0:
            raise ContractError(
                "minimum_significance_score must be non-negative"
            )

        if self.minimum_composite_margin < 0.0:
            raise ContractError(
                "minimum_composite_margin must be non-negative"
            )


@dataclass(frozen=True, slots=True)
class ChampionPromotionDecision:
    """Result of applying promotion policy to ranking entries."""

    candidate: str | None
    promoted: bool
    promoted_model: str | None
    composite_margin: float | None
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.candidate is not None
            and (
                not isinstance(self.candidate, str)
                or not self.candidate.strip()
            )
        ):
            raise ContractError(
                "candidate must be None or a non-empty string"
            )

        if not isinstance(self.promoted, bool):
            raise ContractError(
                "promoted must be boolean"
            )

        if (
            self.promoted_model is not None
            and (
                not isinstance(self.promoted_model, str)
                or not self.promoted_model.strip()
            )
        ):
            raise ContractError(
                "promoted_model must be None or a non-empty string"
            )

        if self.composite_margin is not None:
            if (
                isinstance(self.composite_margin, bool)
                or not isinstance(
                    self.composite_margin,
                    (int, float),
                )
                or not isfinite(
                    float(self.composite_margin)
                )
            ):
                raise ContractError(
                    "composite_margin must be None or finite numeric"
                )

        if any(
            not isinstance(reason, str)
            or not reason.strip()
            for reason in self.rejection_reasons
        ):
            raise ContractError(
                "rejection_reasons must contain non-empty strings"
            )

        if self.promoted:
            if self.candidate is None:
                raise ContractError(
                    "promoted decision requires candidate"
                )

            if self.promoted_model != self.candidate:
                raise ContractError(
                    "promoted_model must equal candidate"
                )

            if self.rejection_reasons:
                raise ContractError(
                    "promoted decision must not have rejection reasons"
                )
        else:
            if self.promoted_model is not None:
                raise ContractError(
                    "rejected decision must not have promoted_model"
                )

            if not self.rejection_reasons:
                raise ContractError(
                    "rejected decision requires rejection reasons"
                )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rejection_reasons"] = list(
            self.rejection_reasons
        )
        return payload


def evaluate_champion_promotion(
    *,
    entries: Iterable[ModelRankingEntry],
    policy: ChampionPromotionPolicy | None = None,
) -> ChampionPromotionDecision:
    """Evaluate whether the highest-ranked eligible model is promotable."""

    normalized = tuple(entries)

    if not normalized:
        raise ContractError(
            "entries must not be empty"
        )

    if any(
        not isinstance(entry, ModelRankingEntry)
        for entry in normalized
    ):
        raise TypeError(
            "entries must contain ModelRankingEntry values"
        )

    actual_ranks = tuple(
        entry.rank
        for entry in normalized
    )

    expected_ranks = tuple(
        range(1, len(normalized) + 1)
    )

    if actual_ranks != expected_ranks:
        raise ContractError(
            "entries must be ordered by contiguous rank"
        )

    active_policy = (
        ChampionPromotionPolicy()
        if policy is None
        else policy
    )

    if not isinstance(
        active_policy,
        ChampionPromotionPolicy,
    ):
        raise TypeError(
            "policy must be ChampionPromotionPolicy"
        )

    eligible = tuple(
        entry
        for entry in normalized
        if entry.eligible
    )

    if not eligible:
        return ChampionPromotionDecision(
            candidate=None,
            promoted=False,
            promoted_model=None,
            composite_margin=None,
            rejection_reasons=(
                "no_eligible_candidate",
            ),
        )

    candidate = eligible[0]

    composite_margin = None

    if len(eligible) > 1:
        composite_margin = (
            candidate.composite_score
            - eligible[1].composite_score
        )

    reasons: list[str] = []

    if candidate.composite_score <= 0.0:
        reasons.append(
            "composite_score_not_positive"
        )

    if (
        candidate.composite_score
        < active_policy.minimum_composite_score
    ):
        reasons.append(
            "composite_score_below_minimum"
        )

    if (
        candidate.practical_score
        < active_policy.minimum_practical_score
    ):
        reasons.append(
            "practical_score_below_minimum"
        )

    if (
        candidate.significance_score
        < active_policy.minimum_significance_score
    ):
        reasons.append(
            "significance_below_minimum"
        )

    if (
        composite_margin is not None
        and composite_margin
        < active_policy.minimum_composite_margin
    ):
        reasons.append(
            "composite_margin_below_minimum"
        )

    rejection_reasons = tuple(reasons)

    if rejection_reasons:
        return ChampionPromotionDecision(
            candidate=candidate.model_name,
            promoted=False,
            promoted_model=None,
            composite_margin=composite_margin,
            rejection_reasons=rejection_reasons,
        )

    return ChampionPromotionDecision(
        candidate=candidate.model_name,
        promoted=True,
        promoted_model=candidate.model_name,
        composite_margin=composite_margin,
        rejection_reasons=(),
    )
