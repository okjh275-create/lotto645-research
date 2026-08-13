"""Champion/challenger ranking for Project M."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Iterable

from lrp.contracts import ContractError

from .contracts import ModelEvaluation


@dataclass(frozen=True, slots=True)
class ModelRankingEntry:
    """One ranked model-evaluation result."""

    rank: int
    model_name: str

    practical_score: float
    best_score: float
    stability_score: float
    significance_score: float

    composite_score: float

    eligible: bool
    exclusion_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank < 1
        ):
            raise ContractError(
                "rank must be a positive integer"
            )

        if (
            not isinstance(self.model_name, str)
            or not self.model_name.strip()
        ):
            raise ContractError(
                "model_name must not be empty"
            )

        for field_name in (
            "practical_score",
            "best_score",
            "stability_score",
            "significance_score",
            "composite_score",
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

        if not isinstance(self.eligible, bool):
            raise ContractError(
                "eligible must be boolean"
            )

        if any(
            not isinstance(reason, str)
            or not reason.strip()
            for reason in self.exclusion_reasons
        ):
            raise ContractError(
                "exclusion_reasons must contain non-empty strings"
            )

        if self.eligible and self.exclusion_reasons:
            raise ContractError(
                "eligible model must not have exclusion reasons"
            )

        if (
            not self.eligible
            and not self.exclusion_reasons
        ):
            raise ContractError(
                "ineligible model requires exclusion reasons"
            )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["exclusion_reasons"] = list(
            self.exclusion_reasons
        )
        return payload


@dataclass(frozen=True, slots=True)
class ChampionRanking:
    """Ordered Project M champion/challenger evaluation."""

    entries: tuple[ModelRankingEntry, ...]
    champion: str | None

    def __post_init__(self) -> None:
        if not self.entries:
            raise ContractError(
                "ranking requires at least one entry"
            )

        expected_ranks = tuple(
            range(1, len(self.entries) + 1)
        )

        actual_ranks = tuple(
            entry.rank
            for entry in self.entries
        )

        if actual_ranks != expected_ranks:
            raise ContractError(
                "ranking entries must have contiguous ranks"
            )

        names = tuple(
            entry.model_name
            for entry in self.entries
        )

        if len(names) != len(set(names)):
            raise ContractError(
                "ranking model names must be unique"
            )

        eligible_names = tuple(
            entry.model_name
            for entry in self.entries
            if entry.eligible
        )

        if self.champion is None:
            if eligible_names:
                raise ContractError(
                    "champion is required when eligible models exist"
                )
        else:
            if self.champion not in eligible_names:
                raise ContractError(
                    "champion must refer to an eligible model"
                )

            if self.champion != eligible_names[0]:
                raise ContractError(
                    "champion must be highest-ranked eligible model"
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "entries": [
                entry.as_dict()
                for entry in self.entries
            ],
            "champion": self.champion,
        }


def rank_model_evaluations(
    evaluations: Iterable[ModelEvaluation],
    *,
    minimum_worst_practical_delta: float = -0.25,
    minimum_worst_best_delta: float = -0.50,
) -> ChampionRanking:
    """Rank models with practical performance and stability first."""

    normalized = tuple(evaluations)

    if not normalized:
        raise ContractError(
            "evaluations must not be empty"
        )

    if any(
        not isinstance(item, ModelEvaluation)
        for item in normalized
    ):
        raise TypeError(
            "evaluations must contain ModelEvaluation values"
        )

    names = tuple(
        item.model_name
        for item in normalized
    )

    if len(names) != len(set(names)):
        raise ContractError(
            "model names must be unique"
        )

    for name, value in (
        (
            "minimum_worst_practical_delta",
            minimum_worst_practical_delta,
        ),
        (
            "minimum_worst_best_delta",
            minimum_worst_best_delta,
        ),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
        ):
            raise ContractError(
                f"{name} must be finite numeric"
            )

    rows = []

    for evaluation in normalized:
        reasons = []

        if (
            evaluation.worst_practical_hit_delta
            < minimum_worst_practical_delta
        ):
            reasons.append(
                "worst_practical_delta_below_floor"
            )

        if (
            evaluation.worst_best_hit_delta
            < minimum_worst_best_delta
        ):
            reasons.append(
                "worst_best_delta_below_floor"
            )

        practical_score = (
            evaluation.mean_practical_hit_delta
        )

        best_score = (
            evaluation.mean_best_hit_delta
        )

        stability_score = (
            evaluation.worst_practical_hit_delta
            + 0.5 * evaluation.worst_best_hit_delta
        )

        significance_score = (
            evaluation.significant_practical_window_count
            + 0.5
            * evaluation.significant_best_window_count
        )

        composite_score = (
            0.45 * practical_score
            + 0.25 * best_score
            + 0.20 * stability_score
            + 0.10 * significance_score
        )

        rows.append(
            {
                "evaluation": evaluation,
                "practical_score": practical_score,
                "best_score": best_score,
                "stability_score": stability_score,
                "significance_score": significance_score,
                "composite_score": composite_score,
                "eligible": not reasons,
                "reasons": tuple(reasons),
            }
        )

    rows.sort(
        key=lambda row: (
            not row["eligible"],
            -float(row["composite_score"]),
            -float(row["practical_score"]),
            -float(row["best_score"]),
            -float(row["stability_score"]),
            row["evaluation"].model_name,
        )
    )

    entries = tuple(
        ModelRankingEntry(
            rank=index,
            model_name=row["evaluation"].model_name,
            practical_score=row["practical_score"],
            best_score=row["best_score"],
            stability_score=row["stability_score"],
            significance_score=row["significance_score"],
            composite_score=row["composite_score"],
            eligible=row["eligible"],
            exclusion_reasons=row["reasons"],
        )
        for index, row in enumerate(
            rows,
            start=1,
        )
    )

    champion = next(
        (
            entry.model_name
            for entry in entries
            if entry.eligible
        ),
        None,
    )

    return ChampionRanking(
        entries=entries,
        champion=champion,
    )
