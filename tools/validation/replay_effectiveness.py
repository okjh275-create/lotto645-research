"""Effectiveness analysis for historical replay results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import comb, isfinite
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

from lrp.contracts import ContractError

from .historical_replay_models import (
    ReplayRoundResult,
)


@dataclass(frozen=True, slots=True)
class EffectivenessSummary:
    """NoOp versus Adaptive paired replay statistics."""

    round_count: int

    best_hit_mean_delta: float
    best_hit_delta_stddev: float
    best_adaptive_wins: int
    best_noop_wins: int
    best_ties: int
    best_sign_test_pvalue: float

    practical_hit_mean_delta: float
    practical_hit_delta_stddev: float
    practical_adaptive_wins: int
    practical_noop_wins: int
    practical_ties: int
    practical_sign_test_pvalue: float

    average_probability_l1_delta: float
    average_probability_max_delta: float
    average_changed_set_count: float
    changed_portfolio_round_count: int

    final_profile_revision: int | None
    final_profile_sample_size: int | None

    regime_calibration_applied_count: int
    final_regime_calibration_revision: int | None
    final_regime_calibration_sample_size: int | None

    regime_bayesian_applied_count: int
    final_regime_bayesian_revision: int | None
    final_regime_bayesian_sample_size: int | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def exact_two_sided_sign_test(
    *,
    positive: int,
    negative: int,
) -> float:
    """Return an exact two-sided sign-test p-value."""

    for name, value in (
        ("positive", positive),
        ("negative", negative),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ContractError(
                f"{name} must be a non-negative integer"
            )

    sample_size = positive + negative

    if sample_size == 0:
        return 1.0

    smaller = min(positive, negative)

    lower_tail = sum(
        comb(sample_size, index)
        for index in range(smaller + 1)
    ) / (2 ** sample_size)

    return min(1.0, 2.0 * lower_tail)


def evaluate_effectiveness(
    rows: Iterable[ReplayRoundResult],
) -> EffectivenessSummary:
    """Aggregate paired NoOp versus Adaptive outcomes."""

    normalized = tuple(rows)

    if not normalized:
        raise ContractError(
            "replay rows must not be empty"
        )

    if any(
        not isinstance(row, ReplayRoundResult)
        for row in normalized
    ):
        raise TypeError(
            "every replay row must be ReplayRoundResult"
        )

    rounds = tuple(
        row.round_no
        for row in normalized
    )

    if len(rounds) != len(set(rounds)):
        raise ContractError(
            "replay rows contain duplicate rounds"
        )

    best_deltas = tuple(
        row.best_hit_delta
        for row in normalized
    )
    practical_deltas = tuple(
        row.practical_hit_delta
        for row in normalized
    )

    best_positive = sum(
        delta > 0
        for delta in best_deltas
    )
    best_negative = sum(
        delta < 0
        for delta in best_deltas
    )

    practical_positive = sum(
        delta > 0
        for delta in practical_deltas
    )
    practical_negative = sum(
        delta < 0
        for delta in practical_deltas
    )

    final_revision = next(
        (
            row.profile_revision
            for row in reversed(normalized)
            if row.profile_revision is not None
        ),
        None,
    )

    final_sample_size = next(
        (
            row.profile_sample_size
            for row in reversed(normalized)
            if row.profile_sample_size is not None
        ),
        None,
    )

    calibration_applied_count = sum(
        row.regime_calibration_revision is not None
        for row in normalized
    )
    final_calibration_revision = next(
        (
            row.regime_calibration_revision
            for row in reversed(normalized)
            if row.regime_calibration_revision is not None
        ),
        None,
    )
    final_calibration_sample_size = next(
        (
            row.regime_calibration_sample_size
            for row in reversed(normalized)
            if row.regime_calibration_sample_size is not None
        ),
        None,
    )

    bayesian_applied_count = sum(
        row.regime_bayesian_revision is not None
        for row in normalized
    )
    final_bayesian_revision = next(
        (
            row.regime_bayesian_revision
            for row in reversed(normalized)
            if row.regime_bayesian_revision is not None
        ),
        None,
    )
    final_bayesian_sample_size = next(
        (
            row.regime_bayesian_sample_size
            for row in reversed(normalized)
            if row.regime_bayesian_sample_size is not None
        ),
        None,
    )

    return EffectivenessSummary(
        round_count=len(normalized),

        best_hit_mean_delta=mean(best_deltas),
        best_hit_delta_stddev=(
            stdev(best_deltas)
            if len(best_deltas) > 1
            else 0.0
        ),
        best_adaptive_wins=best_positive,
        best_noop_wins=best_negative,
        best_ties=(
            len(normalized)
            - best_positive
            - best_negative
        ),
        best_sign_test_pvalue=(
            exact_two_sided_sign_test(
                positive=best_positive,
                negative=best_negative,
            )
        ),

        practical_hit_mean_delta=mean(
            practical_deltas
        ),
        practical_hit_delta_stddev=(
            stdev(practical_deltas)
            if len(practical_deltas) > 1
            else 0.0
        ),
        practical_adaptive_wins=(
            practical_positive
        ),
        practical_noop_wins=(
            practical_negative
        ),
        practical_ties=(
            len(normalized)
            - practical_positive
            - practical_negative
        ),
        practical_sign_test_pvalue=(
            exact_two_sided_sign_test(
                positive=practical_positive,
                negative=practical_negative,
            )
        ),

        average_probability_l1_delta=mean(
            row.probability_l1_delta
            for row in normalized
        ),
        average_probability_max_delta=mean(
            row.probability_max_delta
            for row in normalized
        ),
        average_changed_set_count=mean(
            row.changed_set_count
            for row in normalized
        ),
        changed_portfolio_round_count=sum(
            row.changed_set_count > 0
            for row in normalized
        ),

        final_profile_revision=final_revision,
        final_profile_sample_size=(
            final_sample_size
        ),

        regime_calibration_applied_count=(
            calibration_applied_count
        ),
        final_regime_calibration_revision=(
            final_calibration_revision
        ),
        final_regime_calibration_sample_size=(
            final_calibration_sample_size
        ),

        regime_bayesian_applied_count=(
            bayesian_applied_count
        ),
        final_regime_bayesian_revision=(
            final_bayesian_revision
        ),
        final_regime_bayesian_sample_size=(
            final_bayesian_sample_size
        ),
    )


def load_replay_rows(
    path: str | Path,
) -> tuple[ReplayRoundResult, ...]:
    """Load ReplayRoundResult rows from JSONL."""

    source = Path(path)

    if not source.is_file():
        raise FileNotFoundError(source)

    rows: list[ReplayRoundResult] = []

    for line_number, raw_line in enumerate(
        source.read_text(
            encoding="utf-8-sig"
        ).splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue

        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ContractError(
                "invalid replay JSONL at line "
                f"{line_number}"
            ) from exc

        if not isinstance(payload, dict):
            raise ContractError(
                "every replay JSONL row must "
                "be an object"
            )

        rows.append(
            ReplayRoundResult(
                round_no=int(payload["round_no"]),
                seed=int(payload["seed"]),
                history_draws=int(
                    payload["history_draws"]
                ),
                noop_best_hits=int(
                    payload["noop_best_hits"]
                ),
                adaptive_best_hits=int(
                    payload["adaptive_best_hits"]
                ),
                noop_practical_hits=int(
                    payload["noop_practical_hits"]
                ),
                adaptive_practical_hits=int(
                    payload["adaptive_practical_hits"]
                ),
                noop_avg_jaccard=float(
                    payload["noop_avg_jaccard"]
                ),
                adaptive_avg_jaccard=float(
                    payload[
                        "adaptive_avg_jaccard"
                    ]
                ),
                probability_l1_delta=float(
                    payload[
                        "probability_l1_delta"
                    ]
                ),
                probability_max_delta=float(
                    payload[
                        "probability_max_delta"
                    ]
                ),
                changed_probability_count=int(
                    payload[
                        "changed_probability_count"
                    ]
                ),
                changed_set_count=int(
                    payload["changed_set_count"]
                ),
                profile_applied=bool(
                    payload["profile_applied"]
                ),
                profile_revision=(
                    None
                    if payload.get(
                        "profile_revision"
                    ) is None
                    else int(
                        payload[
                            "profile_revision"
                        ]
                    )
                ),
                profile_sample_size=(
                    None
                    if payload.get(
                        "profile_sample_size"
                    ) is None
                    else int(
                        payload[
                            "profile_sample_size"
                        ]
                    )
                ),
                regime_calibration_revision=(
                    None
                    if payload.get(
                        "regime_calibration_revision"
                    ) is None
                    else int(
                        payload[
                            "regime_calibration_revision"
                        ]
                    )
                ),
                regime_calibration_sample_size=(
                    None
                    if payload.get(
                        "regime_calibration_sample_size"
                    ) is None
                    else int(
                        payload[
                            "regime_calibration_sample_size"
                        ]
                    )
                ),
                regime_bayesian_revision=(
                    None
                    if payload.get(
                        "regime_bayesian_revision"
                    ) is None
                    else int(
                        payload[
                            "regime_bayesian_revision"
                        ]
                    )
                ),
                regime_bayesian_sample_size=(
                    None
                    if payload.get(
                        "regime_bayesian_sample_size"
                    ) is None
                    else int(
                        payload[
                            "regime_bayesian_sample_size"
                        ]
                    )
                ),
                elapsed_seconds=float(
                    payload["elapsed_seconds"]
                ),
            )
        )

    if not rows:
        raise ContractError(
            "replay JSONL contains no rows"
        )

    return tuple(rows)


def write_effectiveness_report(
    *,
    summary: EffectivenessSummary,
    output: str | Path,
) -> Path:
    """Write a reproducible effectiveness report."""

    if not isinstance(
        summary,
        EffectivenessSummary,
    ):
        raise TypeError(
            "summary must be EffectivenessSummary"
        )

    target = Path(output)
    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "status": "PASS",
        "summary": summary.as_dict(),
        "interpretation": {
            "best_hit_direction": _direction(
                summary.best_hit_mean_delta
            ),
            "practical_hit_direction": _direction(
                summary.practical_hit_mean_delta
            ),
            "significance_threshold": 0.05,
            "best_significant": (
                summary.best_sign_test_pvalue
                < 0.05
            ),
            "practical_significant": (
                summary.practical_sign_test_pvalue
                < 0.05
            ),
        },
    }

    target.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return target


def _direction(value: float) -> str:
    if not isfinite(value):
        raise ContractError(
            "effectiveness value must be finite"
        )

    if value > 0:
        return "adaptive_better"

    if value < 0:
        return "noop_better"

    return "tie"
