"""Final prediction JSON serializer."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from typing import Any, Iterable, Mapping

from lrp.contracts import ContractError

from .models import PredictionResult


def _read(value: object, *names: str) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]

        if hasattr(value, name):
            return getattr(value, name)

    return None


def _candidate_object(value: object) -> object:
    current = value

    for _ in range(5):
        numbers = _read(current, "numbers")
        if numbers is not None:
            return current

        nested = _read(
            current,
            "candidate",
            "scored_candidate",
            "scored",
            "ranked",
            "item",
        )
        if nested is None or nested is current:
            break

        current = nested

    raise ContractError(
        "unable to locate candidate numbers"
    )


def _numbers(value: object) -> tuple[int, ...]:
    candidate = _candidate_object(value)
    raw = _read(candidate, "numbers")

    try:
        numbers = tuple(sorted(int(number) for number in raw))
    except (TypeError, ValueError) as exc:
        raise ContractError(
            "candidate numbers are invalid"
        ) from exc

    if len(numbers) != 6:
        raise ContractError(
            "candidate must contain six numbers"
        )

    return numbers


def _normalized_score(value: object) -> float:
    current = value

    for _ in range(5):
        score = _read(
            current,
            "normalized_score",
            "score",
            "ranking_score",
            "final_score",
        )
        if isinstance(score, (int, float)) and not isinstance(
            score,
            bool,
        ):
            return min(1.0, max(0.0, float(score)))

        nested = _read(
            current,
            "candidate",
            "scored_candidate",
            "scored",
            "ranked",
            "item",
        )
        if nested is None or nested is current:
            break

        current = nested

    return 0.0


def _consecutive_runs(
    numbers: tuple[int, ...],
) -> list[list[int]]:
    runs: list[list[int]] = []
    current: list[int] = [numbers[0]]

    for previous, number in zip(numbers, numbers[1:]):
        if number == previous + 1:
            current.append(number)
        else:
            if len(current) >= 2:
                runs.append(current)
            current = [number]

    if len(current) >= 2:
        runs.append(current)

    return runs


def _risk_flags(value: object) -> list[str]:
    candidate = _candidate_object(value)
    risk = _read(candidate, "risk")

    if risk is None:
        return []

    flags = _read(
        risk,
        "risk_flags",
        "flags",
        "reasons",
        "violations",
    )

    if flags is None:
        passed = _read(risk, "passed", "accepted", "valid")
        return [] if passed is not False else ["risk_rejected"]

    if isinstance(flags, str):
        return [flags]

    try:
        return [str(flag) for flag in flags]
    except TypeError:
        return [str(flags)]


def _features(numbers: tuple[int, ...]) -> dict[str, Any]:
    odd = sum(number % 2 for number in numbers)
    low = sum(number <= 22 for number in numbers)

    return {
        "sum": sum(numbers),
        "odd_even": f"{odd}:{6 - odd}",
        "low_high": f"{low}:{6 - low}",
        "consecutives": _consecutive_runs(numbers),
        "end_digits": [number % 10 for number in numbers],
    }


def _jaccard(
    left: Iterable[int],
    right: Iterable[int],
) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set

    if not union:
        return 0.0

    return len(left_set & right_set) / len(union)


def _diversity_metrics(
    number_sets: list[tuple[int, ...]],
) -> dict[str, Any]:
    similarities: list[float] = []

    for index, left in enumerate(number_sets):
        for right in number_sets[index + 1:]:
            similarities.append(_jaccard(left, right))

    average = (
        sum(similarities) / len(similarities)
        if similarities
        else 0.0
    )

    unique = len(
        {
            number
            for numbers in number_sets
            for number in numbers
        }
    )

    return {
        "avg_jaccard": round(average, 8),
        "unique_numbers": unique,
    }


def _selected_items(value: object) -> tuple[object, ...]:
    selected = _read(value, "selected", "items", "candidates")

    if selected is None:
        return ()

    return tuple(selected)


def prediction_to_dict(
    result: PredictionResult,
) -> dict[str, Any]:
    if not isinstance(result, PredictionResult):
        raise ContractError(
            "result must be a PredictionResult"
        )

    request = result.request
    diversity_items = _selected_items(result.diversity)

    sets: list[dict[str, Any]] = []
    id_by_numbers: dict[tuple[int, ...], str] = {}

    for index, item in enumerate(diversity_items, start=1):
        numbers = _numbers(item)
        set_id = f"S{index}"

        id_by_numbers[numbers] = set_id

        sets.append(
            {
                "id": set_id,
                "numbers": list(numbers),
                "score": round(_normalized_score(item), 8),
                "risk_flags": _risk_flags(item),
                "features": _features(numbers),
            }
        )

    practical_ids: list[str] = []

    for item in _selected_items(result.practical):
        numbers = _numbers(item)
        set_id = id_by_numbers.get(numbers)

        if set_id is not None and set_id not in practical_ids:
            practical_ids.append(set_id)

    if len(practical_ids) < request.practical_k:
        fallback = sorted(
            sets,
            key=lambda item: item["score"],
            reverse=True,
        )

        for item in fallback:
            if item["id"] not in practical_ids:
                practical_ids.append(item["id"])

            if len(practical_ids) >= request.practical_k:
                break

    filters = {
        "sum_range": [request.sum_min, request.sum_max],
        "odd_even": "2:4~4:2",
        "low_high_min_each": 1,
        "max_consecutives": 2,
        "max_same_ending": 2,
        "max_overlap_prev_round": 1,
        "min_long_gap_inclusion": 1,
        "max_same_decade": 3,
        "jaccard_max": request.jaccard_max,
        "max_overlap_between_sets": (
            request.max_overlap_between_sets
        ),
    }

    return {
        "round": request.round_no,
        "generated_at_kst": result.generated_at_kst.strftime(
            "%Y-%m-%d %H:%M"
        ),
        "seed": request.seed,
        "params": {
            "temperature": request.temperature,
            "weights": dict(request.weights),
            "filters": filters,
            "windows": {
                "short": result.generation.windows[0],
                "mid": result.generation.windows[1],
                "long": result.generation.windows[2],
            },
            "K": request.top_k,
        },
        "sets": sets,
        "probability_vector": (
            result.generation
            .probability_vector
            .as_dict()
        ),
        "diversity": _diversity_metrics(
            [
                tuple(item["numbers"])
                for item in sets
            ]
        ),
        "top5_practical": practical_ids[:request.practical_k],
        "metadata": {
            "generated_candidates": result.generated_count,
            "statistics_version": (
                result.generation.statistics_version
            ),
            "candidate_version": (
                result.generation.candidate_version
            ),
            "practical_complete": bool(
                _read(result.practical, "complete")
            ),
            "global_regime": (
                {
                    **result.generation
                    .global_regime_context
                    .as_dict(),
                    "mode": (
                        result.generation
                        .global_regime_mode
                    ),
                }
                if (
                    result.generation
                    .global_regime_context
                    is not None
                )
                else None
            ),
        },
    }


def prediction_to_json(
    result: PredictionResult,
    *,
    indent: int | None = 2,
) -> str:
    return json.dumps(
        prediction_to_dict(result),
        ensure_ascii=False,
        indent=indent,
    )
