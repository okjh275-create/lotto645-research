"""Public request and result models for the prediction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
from typing import Any, Mapping, Sequence

from lrp.contracts import ContractError


_DEFAULT_WEIGHTS = {
    "recency": 0.35,
    "frequency": 0.20,
    "gap_reversion": 0.15,
    "pair_graph": 0.10,
    "terminal_dispersion": 0.08,
    "sum_band": 0.07,
    "parity_balance": 0.05,
}


def _normalize_numbers(
    values: Sequence[int] | frozenset[int],
    *,
    field_name: str,
) -> frozenset[int]:
    try:
        normalized = frozenset(int(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ContractError(
            f"{field_name} must contain integers"
        ) from exc

    invalid = tuple(
        sorted(
            number
            for number in normalized
            if not 1 <= number <= 45
        )
    )
    if invalid:
        raise ContractError(
            f"{field_name} contains invalid lotto numbers: {invalid}"
        )

    return normalized


@dataclass(frozen=True, slots=True)
class PredictionRequest:
    """Immutable Project A prediction request."""

    round_no: int
    seed: int
    temperature: float = 0.85
    candidate_count: int = 10_000
    max_attempts_multiplier: int = 50
    top_k: int = 10
    practical_k: int = 5

    previous_numbers: frozenset[int] = field(
        default_factory=frozenset
    )
    long_gap_numbers: frozenset[int] = field(
        default_factory=frozenset
    )

    weights: Mapping[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_WEIGHTS)
    )

    sum_min: int = 90
    sum_max: int = 200
    preferred_sum_min: int = 110
    preferred_sum_max: int = 180

    jaccard_max: float = 0.33
    max_overlap_between_sets: int = 3
    mmr_lambda: float = 0.75

    def __post_init__(self) -> None:
        if (
            isinstance(self.round_no, bool)
            or not isinstance(self.round_no, int)
            or self.round_no <= 0
        ):
            raise ContractError(
                "round_no must be a positive integer"
            )

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ContractError("seed must be an integer")

        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(float(self.temperature))
            or self.temperature <= 0.0
        ):
            raise ContractError(
                "temperature must be a finite positive number"
            )

        for name, value in (
            ("candidate_count", self.candidate_count),
            (
                "max_attempts_multiplier",
                self.max_attempts_multiplier,
            ),
            ("top_k", self.top_k),
            ("practical_k", self.practical_k),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ContractError(
                    f"{name} must be a positive integer"
                )

        if self.practical_k > self.top_k:
            raise ContractError(
                "practical_k must not exceed top_k"
            )

        previous = _normalize_numbers(
            self.previous_numbers,
            field_name="previous_numbers",
        )
        if previous and len(previous) != 6:
            raise ContractError(
                "previous_numbers must contain exactly six numbers"
            )

        long_gap = _normalize_numbers(
            self.long_gap_numbers,
            field_name="long_gap_numbers",
        )

        if not long_gap:
            raise ContractError(
                "long_gap_numbers must contain at least one number"
            )

        if not isinstance(self.weights, Mapping):
            raise ContractError("weights must be a mapping")

        missing = tuple(
            name
            for name in _DEFAULT_WEIGHTS
            if name not in self.weights
        )
        if missing:
            raise ContractError(
                f"weights are missing fields: {missing}"
            )

        normalized_weights: dict[str, float] = {}

        for name in _DEFAULT_WEIGHTS:
            value = self.weights[name]

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ContractError(
                    f"weight {name} must be finite and non-negative"
                )

            normalized_weights[name] = float(value)

        if sum(normalized_weights.values()) <= 0.0:
            raise ContractError(
                "at least one weight must be positive"
            )

        if self.sum_min > self.sum_max:
            raise ContractError(
                "sum_min must not exceed sum_max"
            )

        if self.preferred_sum_min > self.preferred_sum_max:
            raise ContractError(
                "preferred_sum_min must not exceed preferred_sum_max"
            )

        if not 0.0 <= float(self.jaccard_max) <= 1.0:
            raise ContractError(
                "jaccard_max must be between 0 and 1"
            )

        if not 0.0 <= float(self.mmr_lambda) <= 1.0:
            raise ContractError(
                "mmr_lambda must be between 0 and 1"
            )

        object.__setattr__(
            self,
            "temperature",
            float(self.temperature),
        )
        object.__setattr__(
            self,
            "previous_numbers",
            previous,
        )
        object.__setattr__(
            self,
            "long_gap_numbers",
            long_gap,
        )
        object.__setattr__(
            self,
            "weights",
            normalized_weights,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_no": self.round_no,
            "seed": self.seed,
            "temperature": self.temperature,
            "candidate_count": self.candidate_count,
            "max_attempts_multiplier": (
                self.max_attempts_multiplier
            ),
            "top_k": self.top_k,
            "practical_k": self.practical_k,
            "previous_numbers": sorted(self.previous_numbers),
            "long_gap_numbers": sorted(self.long_gap_numbers),
            "weights": dict(self.weights),
            "sum_min": self.sum_min,
            "sum_max": self.sum_max,
            "preferred_sum_min": self.preferred_sum_min,
            "preferred_sum_max": self.preferred_sum_max,
            "jaccard_max": self.jaccard_max,
            "max_overlap_between_sets": (
                self.max_overlap_between_sets
            ),
            "mmr_lambda": self.mmr_lambda,
        }


@dataclass(frozen=True, slots=True)
class PredictionGenerationResult:
    """Output of candidate generation."""

    request: PredictionRequest
    windows: tuple[int, int, int]
    probabilities: Mapping[int, float]
    statistics_contract: object
    number_signals: Mapping[int, object]
    candidates: tuple[object, ...]
    statistics_version: str
    candidate_version: str

    @property
    def generated_count(self) -> int:
        return len(self.candidates)

    @property
    def complete(self) -> bool:
        return self.generated_count >= self.request.candidate_count


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Complete Project A prediction output."""

    generation: PredictionGenerationResult
    scored_candidates: tuple[object, ...]
    ranking: object
    diversity: object
    practical: object
    generated_at_kst: datetime

    @property
    def request(self) -> PredictionRequest:
        return self.generation.request

    @property
    def generated_count(self) -> int:
        return self.generation.generated_count
