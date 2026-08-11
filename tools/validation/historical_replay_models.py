"""Contracts for Project G historical replay validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Mapping

from lrp.contracts import ContractError


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    """Immutable historical-replay configuration."""

    start_round: int
    end_round: int
    seed_base: int = 20260802
    temperature: float = 0.85
    candidate_count: int = 1_000
    top_k: int = 20
    practical_k: int = 5
    long_gap_window: int = 5
    confidence: float = 0.80
    mode: str = "fast"

    def __post_init__(self) -> None:
        for name in (
            "start_round",
            "end_round",
            "seed_base",
            "candidate_count",
            "top_k",
            "practical_k",
            "long_gap_window",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
            ):
                raise ContractError(
                    f"{name} must be an integer"
                )

        if self.start_round <= 1:
            raise ContractError(
                "start_round must be greater than one"
            )

        if self.end_round < self.start_round:
            raise ContractError(
                "end_round must be greater than "
                "or equal to start_round"
            )

        if self.candidate_count < 1:
            raise ContractError(
                "candidate_count must be positive"
            )

        if self.top_k < 1:
            raise ContractError(
                "top_k must be positive"
            )

        if self.practical_k < 1:
            raise ContractError(
                "practical_k must be positive"
            )

        if self.practical_k > self.top_k:
            raise ContractError(
                "practical_k must not exceed top_k"
            )

        if self.long_gap_window < 1:
            raise ContractError(
                "long_gap_window must be positive"
            )

        for name in (
            "temperature",
            "confidence",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
            ):
                raise ContractError(
                    f"{name} must be finite numeric"
                )

        if self.temperature <= 0.0:
            raise ContractError(
                "temperature must be positive"
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ContractError(
                "confidence must be between 0 and 1"
            )

        if self.mode not in ("fast", "full"):
            raise ContractError(
                "mode must be fast or full"
            )

    @property
    def rounds(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.start_round,
                self.end_round + 1,
            )
        )

    def seed_for_round(
        self,
        round_no: int,
    ) -> int:
        if round_no not in self.rounds:
            raise ContractError(
                "round_no is outside replay range"
            )

        return self.seed_base + round_no

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReplayRoundResult:
    """One leak-free NoOp versus Adaptive replay result."""

    round_no: int
    seed: int
    history_draws: int

    noop_best_hits: int
    adaptive_best_hits: int
    noop_practical_hits: int
    adaptive_practical_hits: int

    noop_avg_jaccard: float
    adaptive_avg_jaccard: float

    probability_l1_delta: float
    probability_max_delta: float
    changed_probability_count: int
    changed_set_count: int

    profile_applied: bool
    profile_revision: int | None
    profile_sample_size: int | None

    regime_calibration_revision: int | None = None
    regime_calibration_sample_size: int | None = None
    regime_bayesian_revision: int | None = None
    regime_bayesian_sample_size: int | None = None

    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.round_no <= 1:
            raise ContractError(
                "round_no must be greater than one"
            )

        if self.history_draws < 1:
            raise ContractError(
                "history_draws must be positive"
            )

        for name in (
            "noop_best_hits",
            "adaptive_best_hits",
            "noop_practical_hits",
            "adaptive_practical_hits",
        ):
            value = getattr(self, name)

            if not 0 <= value <= 6:
                raise ContractError(
                    f"{name} must be between 0 and 6"
                )

        for name in (
            "noop_avg_jaccard",
            "adaptive_avg_jaccard",
            "probability_l1_delta",
            "probability_max_delta",
            "elapsed_seconds",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ContractError(
                    f"{name} must be finite and non-negative"
                )

        if self.changed_probability_count not in range(
            0,
            46,
        ):
            raise ContractError(
                "changed_probability_count "
                "must be between 0 and 45"
            )

        if self.changed_set_count < 0:
            raise ContractError(
                "changed_set_count must be non-negative"
            )

        for name in (
            "profile_revision",
            "profile_sample_size",
            "regime_calibration_revision",
            "regime_calibration_sample_size",
            "regime_bayesian_revision",
            "regime_bayesian_sample_size",
        ):
            value = getattr(self, name)

            if value is not None and value < 1:
                raise ContractError(
                    f"{name} must be positive or None"
                )

    @property
    def best_hit_delta(self) -> int:
        return (
            self.adaptive_best_hits
            - self.noop_best_hits
        )

    @property
    def practical_hit_delta(self) -> int:
        return (
            self.adaptive_practical_hits
            - self.noop_practical_hits
        )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["best_hit_delta"] = (
            self.best_hit_delta
        )
        payload["practical_hit_delta"] = (
            self.practical_hit_delta
        )
        return payload


@dataclass(frozen=True, slots=True)
class ReplaySummary:
    """Aggregate historical replay metrics."""

    round_count: int

    noop_average_best_hits: float
    adaptive_average_best_hits: float

    noop_average_practical_hits: float
    adaptive_average_practical_hits: float

    adaptive_win_count: int
    noop_win_count: int
    tie_count: int

    average_probability_l1_delta: float
    average_changed_set_count: float

    profile_applied_count: int
    final_profile_revision: int | None
    final_profile_sample_size: int | None

    total_elapsed_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_replay(
    rows: tuple[ReplayRoundResult, ...],
) -> ReplaySummary:
    """Aggregate deterministic replay rows."""

    if not rows:
        raise ContractError(
            "replay rows must not be empty"
        )

    count = len(rows)

    adaptive_wins = sum(
        row.adaptive_best_hits
        > row.noop_best_hits
        for row in rows
    )
    noop_wins = sum(
        row.noop_best_hits
        > row.adaptive_best_hits
        for row in rows
    )
    ties = count - adaptive_wins - noop_wins

    final_revision = next(
        (
            row.profile_revision
            for row in reversed(rows)
            if row.profile_revision is not None
        ),
        None,
    )

    final_sample_size = next(
        (
            row.profile_sample_size
            for row in reversed(rows)
            if row.profile_sample_size is not None
        ),
        None,
    )

    return ReplaySummary(
        round_count=count,
        noop_average_best_hits=sum(
            row.noop_best_hits
            for row in rows
        ) / count,
        adaptive_average_best_hits=sum(
            row.adaptive_best_hits
            for row in rows
        ) / count,
        noop_average_practical_hits=sum(
            row.noop_practical_hits
            for row in rows
        ) / count,
        adaptive_average_practical_hits=sum(
            row.adaptive_practical_hits
            for row in rows
        ) / count,
        adaptive_win_count=adaptive_wins,
        noop_win_count=noop_wins,
        tie_count=ties,
        average_probability_l1_delta=sum(
            row.probability_l1_delta
            for row in rows
        ) / count,
        average_changed_set_count=sum(
            row.changed_set_count
            for row in rows
        ) / count,
        profile_applied_count=sum(
            row.profile_applied
            for row in rows
        ),
        final_profile_revision=final_revision,
        final_profile_sample_size=final_sample_size,
        total_elapsed_seconds=sum(
            row.elapsed_seconds
            for row in rows
        ),
    )


def validate_round_coverage(
    *,
    config: ReplayConfig,
    draw_by_round: Mapping[int, object],
) -> None:
    """Require actual draw results for all replay rounds."""

    missing = tuple(
        round_no
        for round_no in config.rounds
        if round_no not in draw_by_round
    )

    if missing:
        raise ContractError(
            "history is missing replay rounds: "
            + ", ".join(
                str(round_no)
                for round_no in missing
            )
        )
