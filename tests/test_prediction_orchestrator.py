"""M2 prediction orchestrator integration smoke test."""

from __future__ import annotations

import json

from lrp.pipelines import (
    PredictionPipeline,
    PredictionRequest,
    prediction_to_dict,
)


def build_draws(module: object) -> tuple[object, ...]:
    draw_type = getattr(module, "DrawRecord")
    draws: list[object] = []

    for round_no in range(1, 81):
        start = ((round_no - 1) * 7) % 45

        numbers = tuple(
            sorted(
                {
                    ((start + offset * 6) % 45) + 1
                    for offset in range(6)
                }
            )
        )

        if len(numbers) != 6:
            raise AssertionError(
                f"invalid synthetic draw: {numbers}"
            )

        draws.append(
            draw_type(
                round=round_no,
                numbers=numbers,
                bonus=None,
            )
        )

    return tuple(draws)


def main() -> None:
    pipeline = PredictionPipeline.load()
    draws = build_draws(pipeline.statistics.module)

    previous_numbers = frozenset(draws[-1].numbers)

    request = PredictionRequest(
        round_no=81,
        seed=20260721,
        temperature=0.85,
        candidate_count=500,
        max_attempts_multiplier=100,
        top_k=10,
        practical_k=5,
        previous_numbers=previous_numbers,
        long_gap_numbers=frozenset(range(1, 46)),
    )

    first = pipeline.run(draws, request)
    second = pipeline.run(draws, request)

    first_payload = prediction_to_dict(first)
    second_payload = prediction_to_dict(second)

    first_sets = [
        item["numbers"]
        for item in first_payload["sets"]
    ]
    second_sets = [
        item["numbers"]
        for item in second_payload["sets"]
    ]

    assert first_sets == second_sets, (
        "same seed must reproduce identical selected sets"
    )

    assert first.generated_count == 500
    assert len(first_payload["sets"]) <= 10
    assert len(first_payload["sets"]) > 0
    assert len(first_payload["top5_practical"]) <= 5

    probability_vector = first_payload[
        "probability_vector"
    ]

    assert probability_vector[
        "probability_count"
    ] == 45
    assert len(
        probability_vector["probabilities"]
    ) == 45

    first_probability = probability_vector[
        "probabilities"
    ][0]

    assert first_probability["number"] == 1
    assert set(
        first_probability["components"]
    ) == {
        "hot",
        "cold",
        "gap",
        "trend",
        "transition",
        "learning",
        "adaptive",
    }

    for item in first_payload["sets"]:
        assert len(item["numbers"]) == 6
        assert item["numbers"] == sorted(item["numbers"])
        assert 90 <= item["features"]["sum"] <= 200

    print(
        json.dumps(
            {
                "status": "PASS",
                "generated_candidates": first.generated_count,
                "selected_sets": len(first_payload["sets"]),
                "top5_practical": (
                    first_payload["top5_practical"]
                ),
                "diversity": first_payload["diversity"],
                "statistics_version": (
                    first_payload["metadata"][
                        "statistics_version"
                    ]
                ),
                "candidate_version": (
                    first_payload["metadata"][
                        "candidate_version"
                    ]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
