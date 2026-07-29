"""Project E ensemble engine foundation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Iterable, Mapping

from lrp.contracts import ContractError

from .models import (
    EnsembleCandidateScore,
    EnsembleConfig,
    EnsembleResult,
)
from .repository import (
    EmptyStrategyWeightRepository,
    StrategyWeightRepository,
)
from .version import __version__


ScoreReader = Callable[[object], float]


def _read(value: object, *names: str) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]

        if hasattr(value, name):
            return getattr(value, name)

    return None


def _default_score_reader(value: object) -> float:
    """Read a normalized Project D score from nested objects."""

    current = value

    for _ in range(6):
        score = _read(
            current,
            "normalized_score",
            "ranking_score",
            "final_score",
            "score",
        )

        if (
            isinstance(score, (int, float))
            and not isinstance(score, bool)
        ):
            normalized = float(score)

            if not math.isfinite(normalized):
                raise ContractError(
                    "candidate score must be finite"
                )

            return min(
                1.0,
                max(0.0, normalized),
            )

        nested = _read(
            current,
            "ranked",
            "scored",
            "scored_candidate",
            "candidate",
            "item",
        )

        if nested is None or nested is current:
            break

        current = nested

    raise ContractError(
        "unable to locate candidate score"
    )


def _trend_factor(trend: str) -> float:
    if trend == "UP":
        return 1.0

    if trend == "DOWN":
        return 0.0

    return 0.5


@dataclass(slots=True)
class EnsembleEngine:
    """Stable Project E entry point."""

    repository: StrategyWeightRepository = field(
        default_factory=EmptyStrategyWeightRepository
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.repository,
            StrategyWeightRepository,
        ):
            raise ContractError(
                "repository must implement "
                "StrategyWeightRepository"
            )

    def evaluate(
        self,
        candidates: Iterable[object],
        *,
        round_no: int,
        config: EnsembleConfig | None = None,
        score_reader: ScoreReader | None = None,
    ) -> EnsembleResult:
        if (
            isinstance(round_no, bool)
            or not isinstance(round_no, int)
            or round_no <= 0
        ):
            raise ContractError(
                "round_no must be a positive integer"
            )

        effective_config = config or EnsembleConfig()

        if not isinstance(
            effective_config,
            EnsembleConfig,
        ):
            raise ContractError(
                "config must be an EnsembleConfig"
            )

        reader = score_reader or _default_score_reader

        if not callable(reader):
            raise ContractError(
                "score_reader must be callable"
            )

        sources = tuple(candidates)
        strategy_weights = self.repository.load_weights(
            round_no=round_no
        )

        adaptive_signal = (
            sum(
                weight.normalized_weight
                for weight in strategy_weights
            )
            / len(strategy_weights)
            if strategy_weights
            else 0.0
        )

        confidence_signal = (
            sum(
                weight.confidence
                for weight in strategy_weights
            )
            / len(strategy_weights)
            if strategy_weights
            else 0.0
        )

        stability_signal = (
            sum(
                weight.stability
                for weight in strategy_weights
            )
            / len(strategy_weights)
            if strategy_weights
            else 0.0
        )

        trend_signal = (
            sum(
                _trend_factor(weight.trend)
                for weight in strategy_weights
            )
            / len(strategy_weights)
            if strategy_weights
            else 0.5
        )

        items: list[EnsembleCandidateScore] = []

        for index, source in enumerate(sources):
            try:
                base_score = float(reader(source))
            except ContractError:
                raise
            except (TypeError, ValueError) as exc:
                raise ContractError(
                    "score_reader returned an invalid score"
                ) from exc

            if not math.isfinite(base_score):
                raise ContractError(
                    "score_reader returned a non-finite score"
                )

            base_score = min(
                1.0,
                max(0.0, base_score),
            )

            contributions = {
                "base": (
                    effective_config.base_score_weight
                    * base_score
                ),
                "adaptive": (
                    effective_config.adaptive_weight
                    * adaptive_signal
                ),
                "confidence": (
                    effective_config.confidence_weight
                    * confidence_signal
                ),
                "stability": (
                    effective_config.stability_weight
                    * stability_signal
                ),
                "trend": (
                    effective_config.trend_weight
                    * trend_signal
                ),
            }

            ensemble_score = (
                sum(contributions.values())
                / effective_config.total_weight
            )
            ensemble_score = min(
                1.0,
                max(0.0, ensemble_score),
            )

            items.append(
                EnsembleCandidateScore(
                    source=source,
                    source_index=index,
                    base_score=base_score,
                    ensemble_score=ensemble_score,
                    contributions=contributions,
                )
            )

        ordered = tuple(
            sorted(
                items,
                key=lambda item: (
                    -item.ensemble_score,
                    -item.base_score,
                    item.source_index,
                ),
            )
        )

        if effective_config.top_k is not None:
            ordered = ordered[:effective_config.top_k]

        return EnsembleResult(
            round_no=round_no,
            items=ordered,
            strategy_weights=tuple(strategy_weights),
            config=effective_config,
            engine_version=__version__,
        )
