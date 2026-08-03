"""Run comparable adaptive-policy replay scenarios."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from lrp.evolution.algorithms.adaptive import (
    AdaptiveWeightCalculator,
)
from lrp.io import load_history
from tools.validation.historical_replay_executor import (
    HistoricalReplayExecutor,
)
from tools.validation.historical_replay_models import (
    ReplayConfig,
)
from tools.validation.historical_replay_runner import (
    HistoricalReplayRunner,
)
from tools.validation.replay_effectiveness import (
    evaluate_effectiveness,
)


@dataclass(frozen=True, slots=True)
class PolicyScenario:
    """One adaptive-weight policy configuration."""

    name: str
    adjustment_scale: float
    minimum_weight: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "adjustment_scale": (
                self.adjustment_scale
            ),
            "minimum_weight": (
                self.minimum_weight
            ),
        }


DEFAULT_SCENARIOS = (
    PolicyScenario(
        name="baseline",
        adjustment_scale=0.25,
        minimum_weight=0.01,
    ),
    PolicyScenario(
        name="damped",
        adjustment_scale=0.0625,
        minimum_weight=0.01,
    ),
    PolicyScenario(
        name="floor",
        adjustment_scale=0.25,
        minimum_weight=0.03,
    ),
    PolicyScenario(
        name="conservative",
        adjustment_scale=0.0625,
        minimum_weight=0.03,
    ),
)


def run_policy_comparison(
    *,
    history_path: Path,
    output_root: Path,
    config: ReplayConfig,
    scenarios: tuple[
        PolicyScenario,
        ...,
    ] = DEFAULT_SCENARIOS,
) -> dict[str, Any]:
    """Execute all policy scenarios under identical replay inputs."""

    history_path = Path(history_path)
    output_root = Path(output_root)

    if not history_path.is_file():
        raise FileNotFoundError(
            history_path
        )

    if not scenarios:
        raise ValueError(
            "scenarios must not be empty"
        )

    names = [
        scenario.name
        for scenario in scenarios
    ]

    if len(names) != len(set(names)):
        raise ValueError(
            "scenario names must be unique"
        )

    history = load_history(
        history_path
    )

    draw_by_round = {
        row.round_no: row
        for row in history
    }

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    started = perf_counter()
    results = []

    for scenario in scenarios:
        scenario_root = (
            output_root / scenario.name
        )

        if scenario_root.exists():
            raise FileExistsError(
                "scenario output already exists: "
                f"{scenario_root}"
            )

        calculator = (
            AdaptiveWeightCalculator(
                adjustment_scale=(
                    scenario.adjustment_scale
                ),
                minimum_weight=(
                    scenario.minimum_weight
                ),
            )
        )

        executor = HistoricalReplayExecutor(
            history=history,
            config=config,
            learning_root=(
                scenario_root / "learning"
            ),
            profile_root=(
                scenario_root / "profiles"
            ),
            adaptive_calculator=calculator,
        )

        replay = HistoricalReplayRunner(
            executor=executor
        ).run(
            config=config,
            draw_by_round=draw_by_round,
            output_root=scenario_root,
        )

        effectiveness = (
            evaluate_effectiveness(
                replay.rounds
            )
        )

        final_profile = (
            _load_final_profile(
                scenario_root / "profiles"
            )
        )

        result = {
            "scenario": scenario.as_dict(),
            "replay_summary": (
                replay.summary.as_dict()
            ),
            "effectiveness": (
                effectiveness.as_dict()
            ),
            "final_profile": final_profile,
        }

        (
            scenario_root
            / "scenario_result.json"
        ).write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        results.append(result)

    comparison = {
        "schema_version": 1,
        "config": config.as_dict(),
        "scenario_count": len(
            results
        ),
        "elapsed_seconds": (
            perf_counter() - started
        ),
        "results": results,
        "ranking": _rank_results(
            results
        ),
    }

    (
        output_root
        / "policy_comparison.json"
    ).write_text(
        json.dumps(
            comparison,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return comparison


def _load_final_profile(
    profile_root: Path,
) -> dict[str, Any]:
    files = sorted(
        profile_root.glob(
            "revision-*.json"
        )
    )

    if not files:
        raise ValueError(
            "no profile snapshots found"
        )

    payload = json.loads(
        files[-1].read_text(
            encoding="utf-8"
        )
    )

    profile = payload.get(
        "profile",
        payload,
    )

    if not isinstance(
        profile,
        dict,
    ):
        raise TypeError(
            "profile payload must be an object"
        )

    weights = profile.get(
        "weights"
    )

    if isinstance(weights, dict):
        profile = {
            **profile,
            **weights,
        }

    keys = (
        "hot_weight",
        "cold_weight",
        "gap_weight",
        "trend_weight",
        "transition_weight",
        "learning_weight",
        "adaptive_weight",
    )

    normalized = {
        "revision": profile["revision"],
        "sample_size": (
            profile["sample_size"]
        ),
        "weights": {
            key: float(profile[key])
            for key in keys
        },
    }

    normalized[
        "total_weight"
    ] = sum(
        normalized["weights"].values()
    )

    return normalized


def _rank_results(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rank practical performance first, then best-hit preservation."""

    rows = []

    for result in results:
        summary = result[
            "effectiveness"
        ]["summary"]

        rows.append(
            {
                "scenario": result[
                    "scenario"
                ]["name"],
                "practical_hit_mean_delta": (
                    summary[
                        "practical_hit_mean_delta"
                    ]
                ),
                "best_hit_mean_delta": (
                    summary[
                        "best_hit_mean_delta"
                    ]
                ),
                "practical_adaptive_wins": (
                    summary[
                        "practical_adaptive_wins"
                    ]
                ),
                "practical_noop_wins": (
                    summary[
                        "practical_noop_wins"
                    ]
                ),
                "average_probability_l1_delta": (
                    summary[
                        "average_probability_l1_delta"
                    ]
                ),
                "average_changed_set_count": (
                    summary[
                        "average_changed_set_count"
                    ]
                ),
            }
        )

    rows.sort(
        key=lambda row: (
            -float(
                row[
                    "practical_hit_mean_delta"
                ]
            ),
            -float(
                row[
                    "best_hit_mean_delta"
                ]
            ),
            -int(
                row[
                    "practical_adaptive_wins"
                ]
            ),
            float(
                row[
                    "average_probability_l1_delta"
                ]
            ),
        )
    )

    return [
        {
            "rank": index,
            **row,
        }
        for index, row in enumerate(
            rows,
            start=1,
        )
    ]
