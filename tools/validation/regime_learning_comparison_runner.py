"""Compare regime-learning replay scenarios."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RegimeLearningScenario:
    """One regime-learning configuration."""

    name: str
    calibration_enabled: bool
    bayesian_enabled: bool

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError(
                "name must be a string"
            )

        normalized = self.name.strip()

        if not normalized:
            raise ValueError(
                "name must not be empty"
            )

        if normalized != self.name:
            raise ValueError(
                "name must not have "
                "leading or trailing whitespace"
            )

        if not isinstance(
            self.calibration_enabled,
            bool,
        ):
            raise TypeError(
                "calibration_enabled "
                "must be a boolean"
            )

        if not isinstance(
            self.bayesian_enabled,
            bool,
        ):
            raise TypeError(
                "bayesian_enabled "
                "must be a boolean"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "calibration_enabled": (
                self.calibration_enabled
            ),
            "bayesian_enabled": (
                self.bayesian_enabled
            ),
        }


DEFAULT_REGIME_LEARNING_SCENARIOS = (
    RegimeLearningScenario(
        name="baseline",
        calibration_enabled=False,
        bayesian_enabled=False,
    ),
    RegimeLearningScenario(
        name="calibration",
        calibration_enabled=True,
        bayesian_enabled=False,
    ),
    RegimeLearningScenario(
        name="bayesian",
        calibration_enabled=False,
        bayesian_enabled=True,
    ),
    RegimeLearningScenario(
        name="combined",
        calibration_enabled=True,
        bayesian_enabled=True,
    ),
)
from pathlib import Path

from lrp.io import load_history

from tools.validation.historical_replay_executor import (
    HistoricalReplayExecutor,
)
from tools.validation.historical_replay_models import (
    ReplayConfig,
    ReplayRoundResult,
)
from tools.validation.historical_replay_runner import (
    HistoricalReplayResult,
    HistoricalReplayRunner,
)
from tools.validation.replay_effectiveness import (
    EffectivenessSummary,
    evaluate_effectiveness,
    exact_two_sided_sign_test,
)


def default_regime_learning_scenarios(
) -> tuple[RegimeLearningScenario, ...]:
    """Return the fixed regime-learning scenarios."""
    return DEFAULT_REGIME_LEARNING_SCENARIOS


@dataclass(frozen=True, slots=True)
class RegimeLearningScenarioResult:
    """Result of one regime-learning replay scenario."""

    scenario: RegimeLearningScenario
    replay: HistoricalReplayResult
    effectiveness: EffectivenessSummary

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario.as_dict(),
            "replay_summary": (
                self.replay.summary.as_dict()
            ),
            "effectiveness": (
                self.effectiveness.as_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class RegimeLearningComparisonResult:
    """Completed regime-learning scenario comparison."""

    schema_version: int
    config: ReplayConfig
    scenarios: tuple[
        RegimeLearningScenarioResult,
        ...,
    ]
    pairwise: tuple[
        RegimeLearningPairwiseResult,
        ...,
    ]
    ranking: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config": self.config.as_dict(),
            "scenario_count": len(
                self.scenarios
            ),
            "results": [
                item.as_dict()
                for item in self.scenarios
            ],
            "pairwise_count": len(
                self.pairwise
            ),
            "pairwise": [
                item.as_dict()
                for item in self.pairwise
            ],
            "ranking": list(
                self.ranking
            ),
        }


DEFAULT_REGIME_LEARNING_PAIRS = (
    ("baseline", "calibration"),
    ("baseline", "bayesian"),
    ("baseline", "combined"),
    ("calibration", "combined"),
)


@dataclass(frozen=True, slots=True)
class RegimeLearningPairwiseResult:
    """Pairwise effectiveness delta between scenarios."""

    left: str
    right: str
    practical_hit_mean_delta: float
    best_hit_mean_delta: float
    practical_win_delta: int
    practical_loss_delta: int
    round_statistics: RegimeLearningRoundComparison

    def as_dict(self) -> dict[str, Any]:
        return {
            "left": self.left,
            "right": self.right,
            "practical_hit_mean_delta": (
                self.practical_hit_mean_delta
            ),
            "best_hit_mean_delta": (
                self.best_hit_mean_delta
            ),
            "practical_win_delta": (
                self.practical_win_delta
            ),
            "practical_loss_delta": (
                self.practical_loss_delta
            ),
            "round_statistics": (
                self.round_statistics.as_dict()
            ),
        }


def _compare_scenarios(
    *,
    left: RegimeLearningScenarioResult,
    right: RegimeLearningScenarioResult,
) -> RegimeLearningPairwiseResult:
    round_statistics = _compare_rounds(
        left=left.replay.rounds,
        right=right.replay.rounds,
    )

    return RegimeLearningPairwiseResult(
        left=left.scenario.name,
        right=right.scenario.name,
        practical_hit_mean_delta=(
            right.effectiveness
            .practical_hit_mean_delta
            - left.effectiveness
            .practical_hit_mean_delta
        ),
        best_hit_mean_delta=(
            right.effectiveness
            .best_hit_mean_delta
            - left.effectiveness
            .best_hit_mean_delta
        ),
        practical_win_delta=(
            right.effectiveness
            .practical_adaptive_wins
            - left.effectiveness
            .practical_adaptive_wins
        ),
        practical_loss_delta=(
            right.effectiveness
            .practical_noop_wins
            - left.effectiveness
            .practical_noop_wins
        ),
        round_statistics=round_statistics,
    )


@dataclass(frozen=True, slots=True)
class RegimeLearningRoundComparison:
    """Paired round-level comparison between scenarios."""

    round_count: int

    practical_right_wins: int
    practical_left_wins: int
    practical_ties: int
    practical_sign_test_p_value: float

    best_right_wins: int
    best_left_wins: int
    best_ties: int
    best_sign_test_p_value: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "round_count": self.round_count,
            "practical_right_wins": (
                self.practical_right_wins
            ),
            "practical_left_wins": (
                self.practical_left_wins
            ),
            "practical_ties": (
                self.practical_ties
            ),
            "practical_sign_test_p_value": (
                self.practical_sign_test_p_value
            ),
            "best_right_wins": (
                self.best_right_wins
            ),
            "best_left_wins": (
                self.best_left_wins
            ),
            "best_ties": self.best_ties,
            "best_sign_test_p_value": (
                self.best_sign_test_p_value
            ),
        }


def _compare_rounds(
    *,
    left: tuple[ReplayRoundResult, ...],
    right: tuple[ReplayRoundResult, ...],
) -> RegimeLearningRoundComparison:
    left_rounds = tuple(
        row.round_no
        for row in left
    )
    right_rounds = tuple(
        row.round_no
        for row in right
    )

    if left_rounds != right_rounds:
        raise ValueError(
            "round alignment mismatch"
        )

    practical_right_wins = 0
    practical_left_wins = 0
    practical_ties = 0

    best_right_wins = 0
    best_left_wins = 0
    best_ties = 0

    for left_row, right_row in zip(
        left,
        right,
        strict=True,
    ):
        if (
            right_row.adaptive_practical_hits
            > left_row.adaptive_practical_hits
        ):
            practical_right_wins += 1
        elif (
            right_row.adaptive_practical_hits
            < left_row.adaptive_practical_hits
        ):
            practical_left_wins += 1
        else:
            practical_ties += 1

        if (
            right_row.adaptive_best_hits
            > left_row.adaptive_best_hits
        ):
            best_right_wins += 1
        elif (
            right_row.adaptive_best_hits
            < left_row.adaptive_best_hits
        ):
            best_left_wins += 1
        else:
            best_ties += 1

    return RegimeLearningRoundComparison(
        round_count=len(left),
        practical_right_wins=(
            practical_right_wins
        ),
        practical_left_wins=(
            practical_left_wins
        ),
        practical_ties=practical_ties,
        practical_sign_test_p_value=(
            exact_two_sided_sign_test(
                positive=practical_right_wins,
                negative=practical_left_wins,
            )
        ),
        best_right_wins=best_right_wins,
        best_left_wins=best_left_wins,
        best_ties=best_ties,
        best_sign_test_p_value=(
            exact_two_sided_sign_test(
                positive=best_right_wins,
                negative=best_left_wins,
            )
        ),
    )


def _rank_scenarios(
    scenarios: tuple[
        RegimeLearningScenarioResult,
        ...,
    ]
    | list[RegimeLearningScenarioResult],
) -> tuple[str, ...]:
    """Rank scenario results by effectiveness and perturbation."""

    ranked = sorted(
        scenarios,
        key=lambda item: (
            -float(
                item.effectiveness
                .practical_hit_mean_delta
            ),
            -float(
                item.effectiveness
                .best_hit_mean_delta
            ),
            -int(
                item.effectiveness
                .practical_adaptive_wins
                - item.effectiveness
                .practical_noop_wins
            ),
            float(
                item.effectiveness
                .average_probability_l1_delta
            ),
            item.scenario.name,
        ),
    )

    return tuple(
        item.scenario.name
        for item in ranked
    )


def _scenario_roots(
    *,
    scenario: RegimeLearningScenario,
    scenario_root: Path,
) -> tuple[
    Path | None,
    Path | None,
]:
    scenario_root = Path(scenario_root)

    calibration_root = (
        scenario_root / "regime-calibration"
        if scenario.calibration_enabled
        else None
    )

    bayesian_root = (
        scenario_root / "regime-bayesian"
        if scenario.bayesian_enabled
        else None
    )

    return (
        calibration_root,
        bayesian_root,
    )


def run_regime_learning_comparison(
    *,
    output_root: Path,
    config: ReplayConfig,
    scenarios: tuple[
        RegimeLearningScenario,
        ...,
    ] = DEFAULT_REGIME_LEARNING_SCENARIOS,
    history: tuple[object, ...] | None = None,
    history_path: Path | None = None,
) -> RegimeLearningComparisonResult:
    output_root = Path(output_root)

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

    if not isinstance(config, ReplayConfig):
        raise TypeError(
            "config must be a ReplayConfig"
        )

    if (
        history is None
        and history_path is None
    ):
        raise ValueError(
            "exactly one of history or "
            "history_path must be provided"
        )

    if (
        history is not None
        and history_path is not None
    ):
        raise ValueError(
            "exactly one of history or "
            "history_path must be provided"
        )

    if history is not None:
        normalized_history = tuple(history)

        if not normalized_history:
            raise ValueError(
                "history must not be empty"
            )
    else:
        assert history_path is not None

        normalized_history_path = Path(
            history_path
        )

        if not normalized_history_path.is_file():
            raise FileNotFoundError(
                normalized_history_path
            )

        normalized_history = tuple(
            load_history(
                normalized_history_path
            )
        )

    draw_by_round = {
        row.round_no: row
        for row in normalized_history
    }

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    results: list[
        RegimeLearningScenarioResult
    ] = []

    for scenario in scenarios:
        scenario_root = (
            output_root / scenario.name
        )

        if scenario_root.exists():
            raise FileExistsError(
                "scenario output already exists: "
                f"{scenario_root}"
            )

        (
            calibration_root,
            bayesian_root,
        ) = _scenario_roots(
            scenario=scenario,
            scenario_root=scenario_root,
        )

        executor = HistoricalReplayExecutor(
            history=normalized_history,
            config=config,
            learning_root=(
                scenario_root / "learning"
            ),
            profile_root=(
                scenario_root / "profiles"
            ),
            regime_calibration_root=(
                calibration_root
            ),
            regime_bayesian_root=(
                bayesian_root
            ),
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

        results.append(
            RegimeLearningScenarioResult(
                scenario=scenario,
                replay=replay,
                effectiveness=effectiveness,
            )
        )

    by_name = {
        item.scenario.name: item
        for item in results
    }

    pairwise = tuple(
        _compare_scenarios(
            left=by_name[left_name],
            right=by_name[right_name],
        )
        for left_name, right_name
        in DEFAULT_REGIME_LEARNING_PAIRS
    )

    ranking = _rank_scenarios(
        results
    )

    result = RegimeLearningComparisonResult(
        schema_version=1,
        config=config,
        scenarios=tuple(results),
        pairwise=pairwise,
        ranking=ranking,
    )

    artifact_path = (
        output_root
        / "regime_learning_comparison.json"
    )

    artifact_path.write_text(
        json.dumps(
            result.as_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return result
