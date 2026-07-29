"""Repository contracts and M6 adapters for Project E."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, runtime_checkable

from lrp.contracts import ContractError

from .adapters import (
    rankings_from_m6,
    statistics_from_m6,
    weights_from_m6,
    weights_from_rankings,
)
from .models import (
    StrategyWeight,
    normalize_strategy_weights,
)
from .snapshot import LearningSnapshot


@runtime_checkable
class StrategyWeightRepository(Protocol):
    """Read-only source of Project E strategy weights."""

    def load_weights(
        self,
        *,
        round_no: int,
    ) -> tuple[StrategyWeight, ...]:
        """Return weights available for the requested round."""


@runtime_checkable
class SnapshotRepository(Protocol):
    """Read-only source of Project E learning snapshots."""

    def load_snapshot(
        self,
        *,
        round_no: int,
    ) -> LearningSnapshot:
        """Return one immutable learning snapshot."""


def _validate_round(round_no: int) -> int:
    if (
        isinstance(round_no, bool)
        or not isinstance(round_no, int)
        or round_no <= 0
    ):
        raise ContractError(
            "round_no must be a positive integer"
        )

    return round_no


@dataclass(slots=True)
class EmptyStrategyWeightRepository:
    """Repository used when no learning data is configured."""

    def load_weights(
        self,
        *,
        round_no: int,
    ) -> tuple[StrategyWeight, ...]:
        _validate_round(round_no)
        return ()


@dataclass(slots=True)
class InMemoryStrategyWeightRepository:
    """Deterministic repository for tests and local execution."""

    weights: Iterable[StrategyWeight] = field(
        default_factory=tuple
    )
    _normalized: tuple[StrategyWeight, ...] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._normalized = normalize_strategy_weights(
            tuple(self.weights)
        )

    def load_weights(
        self,
        *,
        round_no: int,
    ) -> tuple[StrategyWeight, ...]:
        _validate_round(round_no)
        return self._normalized


@dataclass(slots=True)
class LearningSnapshotWeightRepository:
    """Expose a SnapshotRepository through the E-001 weight contract."""

    snapshot_repository: SnapshotRepository

    def __post_init__(self) -> None:
        if not isinstance(
            self.snapshot_repository,
            SnapshotRepository,
        ):
            raise ContractError(
                "snapshot_repository must implement "
                "SnapshotRepository"
            )

    def load_weights(
        self,
        *,
        round_no: int,
    ) -> tuple[StrategyWeight, ...]:
        return (
            self.snapshot_repository
            .load_snapshot(
                round_no=round_no
            )
            .strategy_weights
        )


@dataclass(slots=True)
class M6LearningSnapshotRepository:
    """Build Project E snapshots from the existing M6 service."""

    service: object
    strategy_types: tuple[str, ...] = (
        "model",
        "scenario",
    )
    history_limit: int = 100
    _cache_key: tuple[
        int,
        tuple[int, int],
    ] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _cache_value: LearningSnapshot | None = field(
        init=False,
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.service is None:
            raise ContractError(
                "service must not be None"
            )

        if (
            isinstance(self.history_limit, bool)
            or not isinstance(
                self.history_limit,
                int,
            )
            or self.history_limit <= 0
        ):
            raise ContractError(
                "history_limit must be a positive integer"
            )

        normalized_types = tuple(
            str(value).strip().lower()
            for value in self.strategy_types
        )

        if (
            not normalized_types
            or any(
                not value
                for value in normalized_types
            )
        ):
            raise ContractError(
                "strategy_types must not be empty"
            )

        if len(normalized_types) != len(
            set(normalized_types)
        ):
            raise ContractError(
                "strategy_types contains duplicates"
            )

        self.strategy_types = normalized_types

    def _ranking_repository(self) -> object | None:
        return getattr(
            self.service,
            "ranking_repository",
            None,
        )

    def _revision(self) -> tuple[int, int]:
        repository = self._ranking_repository()

        if repository is None:
            return (0, 0)

        method = getattr(
            repository,
            "repository_revision",
            None,
        )

        if not callable(method):
            return (0, 0)

        value = tuple(method())

        if (
            len(value) != 2
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or item < 0
                for item in value
            )
        ):
            raise ContractError(
                "M6 repository_revision returned "
                "an invalid value"
            )

        return (
            int(value[0]),
            int(value[1]),
        )

    def _statistics(
        self,
        strategy_type: str,
    ) -> tuple[object, ...]:
        method = getattr(
            self.service,
            "get_strategy_statistics",
            None,
        )

        if not callable(method):
            repository = getattr(
                self.service,
                "repository",
                None,
            )
            method = getattr(
                repository,
                "get_strategy_statistics",
                None,
            )

        if not callable(method):
            raise ContractError(
                "M6 service does not expose "
                "get_strategy_statistics"
            )

        return tuple(
            method(
                strategy_type=strategy_type
            )
        )

    def _rankings(
        self,
        strategy_type: str,
    ) -> tuple[object, ...]:
        method = getattr(
            self.service,
            "rank_strategies",
            None,
        )

        if not callable(method):
            raise ContractError(
                "M6 service does not expose "
                "rank_strategies"
            )

        return tuple(
            method(
                strategy_type=strategy_type,
                history_limit=self.history_limit,
            )
        )

    def _adaptive_weights(
        self,
        strategy_type: str,
    ) -> tuple[object, ...] | None:
        candidate_names = (
            "adaptive_weights",
            "calculate_adaptive_weights",
            "build_adaptive_weights",
            "get_adaptive_weights",
            "update_adaptive_weights",
        )

        for name in candidate_names:
            method = getattr(
                self.service,
                name,
                None,
            )

            if not callable(method):
                continue

            attempts = (
                {
                    "strategy_type": (
                        strategy_type
                    ),
                    "history_limit": (
                        self.history_limit
                    ),
                },
                {
                    "strategy_type": (
                        strategy_type
                    ),
                },
                {},
            )

            for arguments in attempts:
                try:
                    result = method(**arguments)
                except TypeError:
                    continue

                if result is None:
                    return ()

                return tuple(result)

        return None

    def load_snapshot(
        self,
        *,
        round_no: int,
    ) -> LearningSnapshot:
        round_no = _validate_round(
            round_no
        )

        revision = self._revision()
        cache_key = (
            round_no,
            revision,
        )

        if (
            self._cache_key == cache_key
            and self._cache_value is not None
        ):
            return self._cache_value

        raw_statistics: list[object] = []
        raw_rankings: list[object] = []
        raw_weights: list[object] = []
        adaptive_available = True

        for strategy_type in self.strategy_types:
            raw_statistics.extend(
                self._statistics(
                    strategy_type
                )
            )

            raw_rankings.extend(
                self._rankings(
                    strategy_type
                )
            )

            adaptive = self._adaptive_weights(
                strategy_type
            )

            if adaptive is None:
                adaptive_available = False
            else:
                raw_weights.extend(
                    adaptive
                )

        statistics = statistics_from_m6(
            raw_statistics
        )
        rankings = rankings_from_m6(
            raw_rankings
        )

        if adaptive_available and raw_weights:
            weights = weights_from_m6(
                raw_weights
            )
            weight_source = "m6_adaptive"
        else:
            weights = weights_from_rankings(
                rankings
            )
            weight_source = (
                "m6_ranking_fallback"
            )

        snapshot = LearningSnapshot(
            round_no=round_no,
            revision=revision,
            statistics=statistics,
            rankings=rankings,
            strategy_weights=weights,
            source="lrp.learning",
            metadata={
                "history_limit": (
                    self.history_limit
                ),
                "strategy_types": list(
                    self.strategy_types
                ),
                "weight_source": (
                    weight_source
                ),
            },
        )

        self._cache_key = cache_key
        self._cache_value = snapshot

        return snapshot

    def invalidate_cache(self) -> None:
        self._cache_key = None
        self._cache_value = None
