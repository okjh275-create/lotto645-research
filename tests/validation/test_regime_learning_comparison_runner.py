from __future__ import annotations

import pytest

from tools.validation.regime_learning_comparison_runner import (
    DEFAULT_REGIME_LEARNING_SCENARIOS,
    RegimeLearningScenario,
)


def test_default_regime_learning_scenarios_are_fixed() -> None:
    assert tuple(
        scenario.name
        for scenario in DEFAULT_REGIME_LEARNING_SCENARIOS
    ) == (
        "baseline",
        "calibration",
        "bayesian",
        "combined",
    )


def test_baseline_disables_regime_learning() -> None:
    scenario = DEFAULT_REGIME_LEARNING_SCENARIOS[0]

    assert scenario.calibration_enabled is False
    assert scenario.bayesian_enabled is False


def test_calibration_scenario_enables_only_calibration() -> None:
    scenario = DEFAULT_REGIME_LEARNING_SCENARIOS[1]

    assert scenario.calibration_enabled is True
    assert scenario.bayesian_enabled is False


def test_bayesian_scenario_enables_only_bayesian() -> None:
    scenario = DEFAULT_REGIME_LEARNING_SCENARIOS[2]

    assert scenario.calibration_enabled is False
    assert scenario.bayesian_enabled is True


def test_combined_scenario_enables_both_learning_paths() -> None:
    scenario = DEFAULT_REGIME_LEARNING_SCENARIOS[3]

    assert scenario.calibration_enabled is True
    assert scenario.bayesian_enabled is True


def test_regime_learning_scenario_serialization() -> None:
    scenario = RegimeLearningScenario(
        name="custom",
        calibration_enabled=True,
        bayesian_enabled=False,
    )

    assert scenario.as_dict() == {
        "name": "custom",
        "calibration_enabled": True,
        "bayesian_enabled": False,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("calibration_enabled", 1),
        ("bayesian_enabled", 1),
    ),
)
def test_regime_learning_scenario_requires_boolean_flags(
    field: str,
    value: object,
) -> None:
    kwargs = {
        "name": "invalid",
        "calibration_enabled": False,
        "bayesian_enabled": False,
    }
    kwargs[field] = value

    with pytest.raises(
        TypeError,
        match="must be a boolean",
    ):
        RegimeLearningScenario(**kwargs)


def test_regime_learning_scenario_requires_non_empty_name() -> None:
    with pytest.raises(
        ValueError,
        match="name",
    ):
        RegimeLearningScenario(
            name="",
            calibration_enabled=False,
            bayesian_enabled=False,
        )


def test_scenario_roots_match_learning_flags(
    tmp_path,
) -> None:
    from tools.validation.regime_learning_comparison_runner import (
        _scenario_roots,
    )

    baseline = _scenario_roots(
        scenario=DEFAULT_REGIME_LEARNING_SCENARIOS[0],
        scenario_root=tmp_path / "baseline",
    )
    calibration = _scenario_roots(
        scenario=DEFAULT_REGIME_LEARNING_SCENARIOS[1],
        scenario_root=tmp_path / "calibration",
    )
    bayesian = _scenario_roots(
        scenario=DEFAULT_REGIME_LEARNING_SCENARIOS[2],
        scenario_root=tmp_path / "bayesian",
    )
    combined = _scenario_roots(
        scenario=DEFAULT_REGIME_LEARNING_SCENARIOS[3],
        scenario_root=tmp_path / "combined",
    )

    assert baseline == (None, None)

    assert calibration == (
        tmp_path / "calibration" / "regime-calibration",
        None,
    )

    assert bayesian == (
        None,
        tmp_path / "bayesian" / "regime-bayesian",
    )

    assert combined == (
        tmp_path / "combined" / "regime-calibration",
        tmp_path / "combined" / "regime-bayesian",
    )


def test_comparison_requires_unique_scenario_names(
    tmp_path,
) -> None:
    from tools.validation.regime_learning_comparison_runner import (
        run_regime_learning_comparison,
    )

    scenarios = (
        RegimeLearningScenario(
            name="same",
            calibration_enabled=False,
            bayesian_enabled=False,
        ),
        RegimeLearningScenario(
            name="same",
            calibration_enabled=True,
            bayesian_enabled=False,
        ),
    )

    with pytest.raises(
        ValueError,
        match="scenario names must be unique",
    ):
        run_regime_learning_comparison(
            history_path=tmp_path / "history.json",
            output_root=tmp_path / "output",
            config=object(),
            scenarios=scenarios,
        )

def test_four_scenario_replay_end_to_end(
    tmp_path,
    monkeypatch,
) -> None:
    import lrp.pipelines.prediction as prediction_module

    from lrp.io.draws import HistoryRow
    from lrp.regimes import (
        RegimeDecision,
        RegimeFeatureSnapshot,
    )

    from tools.validation.historical_replay_models import (
        ReplayConfig,
    )
    from tools.validation.regime_learning_comparison_runner import (
        default_regime_learning_scenarios,
        run_regime_learning_comparison,
    )

    class FixedFeatureExtractor:
        def extract(self, snapshot):
            return snapshot

    class FixedStabilityPolicy:
        def decide(self, features):
            return RegimeDecision(
                primary="high_band_expansion",
                confidence=1.0,
                features=RegimeFeatureSnapshot(
                    average_recency=0.5,
                    average_frequency=0.5,
                    average_gap_reversion=0.5,
                    pair_density=0.5,
                    frequency_dispersion=0.5,
                    recency_variance=0.5,
                    pair_variance=0.5,
                    low_band_ratio=0.3,
                    high_band_ratio=0.7,
                ),
                scores={
                    "neutral": 0.0,
                    "mixed": 0.0,
                    "gap_recovery": 0.0,
                    "cluster_rotation": 0.0,
                    "high_band_expansion": 1.0,
                    "low_band_expansion": 0.0,
                },
            )

    monkeypatch.setattr(
        prediction_module,
        "GlobalRegimeFeatureExtractor",
        lambda: FixedFeatureExtractor(),
    )
    monkeypatch.setattr(
        prediction_module,
        "GlobalRegimeStabilityPolicy",
        lambda: FixedStabilityPolicy(),
    )

    history = []

    for round_no in range(1142, 1224):
        start = ((round_no - 1) * 7) % 45

        numbers = tuple(
            sorted(
                {
                    ((start + offset * 6) % 45) + 1
                    for offset in range(6)
                }
            )
        )

        assert len(numbers) == 6

        bonus = next(
            number
            for number in range(1, 46)
            if number not in numbers
        )

        history.append(
            HistoryRow(
                round_no=round_no,
                numbers=numbers,
                bonus=bonus,
            )
        )

    result = run_regime_learning_comparison(
        history=tuple(history),
        config=ReplayConfig(
            start_round=1222,
            end_round=1223,
            candidate_count=100,
            top_k=10,
            practical_k=5,
            mode="fast",
        ),
        output_root=tmp_path / "comparison",
        scenarios=default_regime_learning_scenarios(),
    )

    assert len(result.scenarios) == 4

    by_name = {
        item.scenario.name: item
        for item in result.scenarios
    }

    assert set(by_name) == {
        "baseline",
        "calibration",
        "bayesian",
        "combined",
    }

    baseline = by_name["baseline"]
    calibration = by_name["calibration"]
    bayesian = by_name["bayesian"]
    combined = by_name["combined"]

    assert (
        baseline.replay.summary
        .regime_calibration_applied_count
        == 0
    )
    assert (
        baseline.replay.summary
        .regime_bayesian_applied_count
        == 0
    )

    assert (
        calibration.replay.summary
        .regime_calibration_applied_count
        == 1
    )
    assert (
        calibration.replay.summary
        .regime_bayesian_applied_count
        == 0
    )

    assert (
        bayesian.replay.summary
        .regime_calibration_applied_count
        == 0
    )
    assert (
        bayesian.replay.summary
        .regime_bayesian_applied_count
        == 1
    )

    assert (
        combined.replay.summary
        .regime_calibration_applied_count
        == 1
    )
    assert (
        combined.replay.summary
        .regime_bayesian_applied_count
        == 1
    )

    for scenario_result in result.scenarios:
        assert scenario_result.replay.rounds_path.exists()
        assert scenario_result.replay.summary_path.exists()



def test_default_pairwise_comparisons_are_fixed() -> None:
    from tools.validation.regime_learning_comparison_runner import (
        DEFAULT_REGIME_LEARNING_PAIRS,
    )

    assert DEFAULT_REGIME_LEARNING_PAIRS == (
        ("baseline", "calibration"),
        ("baseline", "bayesian"),
        ("baseline", "combined"),
        ("calibration", "combined"),
    )


def test_pairwise_comparison_uses_scenario_effectiveness() -> None:
    from tools.validation.regime_learning_comparison_runner import (
        RegimeLearningPairwiseResult,
        _compare_scenarios,
    )

    class Effectiveness:
        def __init__(
            self,
            *,
            practical: float,
            best: float,
            practical_wins: int,
            practical_losses: int,
        ) -> None:
            self.practical_hit_mean_delta = practical
            self.best_hit_mean_delta = best
            self.practical_adaptive_wins = practical_wins
            self.practical_noop_wins = practical_losses

    class Scenario:
        def __init__(self, name: str) -> None:
            self.name = name

    class Replay:
        def __init__(self) -> None:
            self.rounds = ()

    class ScenarioResult:
        def __init__(
            self,
            name: str,
            effectiveness,
        ) -> None:
            self.scenario = Scenario(name)
            self.effectiveness = effectiveness
            self.replay = Replay()

    left = ScenarioResult(
        "baseline",
        Effectiveness(
            practical=0.10,
            best=0.05,
            practical_wins=4,
            practical_losses=2,
        ),
    )
    right = ScenarioResult(
        "combined",
        Effectiveness(
            practical=0.30,
            best=0.15,
            practical_wins=7,
            practical_losses=1,
        ),
    )

    result = _compare_scenarios(
        left=left,
        right=right,
    )

    assert isinstance(
        result,
        RegimeLearningPairwiseResult,
    )
    assert result.left == "baseline"
    assert result.right == "combined"
    assert result.practical_hit_mean_delta == pytest.approx(0.20)
    assert result.best_hit_mean_delta == pytest.approx(0.10)
    assert result.practical_win_delta == 3
    assert result.practical_loss_delta == -1

def test_comparison_result_includes_default_pairwise_results(
    tmp_path,
    monkeypatch,
) -> None:
    import lrp.pipelines.prediction as prediction_module

    from lrp.io.draws import HistoryRow
    from lrp.regimes import (
        RegimeDecision,
        RegimeFeatureSnapshot,
    )
    from tools.validation.historical_replay_models import (
        ReplayConfig,
    )
    from tools.validation.regime_learning_comparison_runner import (
        run_regime_learning_comparison,
    )

    class FixedFeatureExtractor:
        def extract(self, snapshot):
            return snapshot

    class FixedStabilityPolicy:
        def decide(self, features):
            return RegimeDecision(
                primary="high_band_expansion",
                confidence=1.0,
                features=RegimeFeatureSnapshot(
                    average_recency=0.5,
                    average_frequency=0.5,
                    average_gap_reversion=0.5,
                    pair_density=0.5,
                    frequency_dispersion=0.5,
                    recency_variance=0.5,
                    pair_variance=0.5,
                    low_band_ratio=0.3,
                    high_band_ratio=0.7,
                ),
                scores={
                    "neutral": 0.0,
                    "mixed": 0.0,
                    "gap_recovery": 0.0,
                    "cluster_rotation": 0.0,
                    "high_band_expansion": 1.0,
                    "low_band_expansion": 0.0,
                },
            )

    monkeypatch.setattr(
        prediction_module,
        "GlobalRegimeFeatureExtractor",
        lambda: FixedFeatureExtractor(),
    )
    monkeypatch.setattr(
        prediction_module,
        "GlobalRegimeStabilityPolicy",
        lambda: FixedStabilityPolicy(),
    )

    history = []

    for round_no in range(1142, 1224):
        start = ((round_no - 1) * 7) % 45

        numbers = tuple(
            sorted(
                {
                    ((start + offset * 6) % 45) + 1
                    for offset in range(6)
                }
            )
        )

        bonus = next(
            number
            for number in range(1, 46)
            if number not in numbers
        )

        history.append(
            HistoryRow(
                round_no=round_no,
                numbers=numbers,
                bonus=bonus,
            )
        )

    result = run_regime_learning_comparison(
        history=tuple(history),
        config=ReplayConfig(
            start_round=1222,
            end_round=1223,
            candidate_count=100,
            top_k=10,
            practical_k=5,
            mode="fast",
        ),
        output_root=tmp_path / "comparison",
    )

    assert tuple(
        (item.left, item.right)
        for item in result.pairwise
    ) == (
        ("baseline", "calibration"),
        ("baseline", "bayesian"),
        ("baseline", "combined"),
        ("calibration", "combined"),
    )

    payload = result.as_dict()

    assert payload["pairwise_count"] == 4
    assert len(payload["pairwise"]) == 4

    assert payload["pairwise"][0]["left"] == "baseline"
    assert payload["pairwise"][0]["right"] == "calibration"



def test_pairwise_round_level_statistics() -> None:
    from tools.validation.historical_replay_models import (
        ReplayRoundResult,
    )
    from tools.validation.regime_learning_comparison_runner import (
        _compare_rounds,
    )

    def row(
        round_no: int,
        *,
        practical: int,
        best: int,
    ) -> ReplayRoundResult:
        return ReplayRoundResult(
            round_no=round_no,
            seed=20260000 + round_no,
            history_draws=600,
            noop_best_hits=0,
            adaptive_best_hits=best,
            noop_practical_hits=0,
            adaptive_practical_hits=practical,
            noop_avg_jaccard=0.0,
            adaptive_avg_jaccard=0.0,
            probability_l1_delta=0.0,
            probability_max_delta=0.0,
            changed_probability_count=0,
            changed_set_count=0,
            profile_applied=False,
            profile_revision=None,
            profile_sample_size=None,
            elapsed_seconds=0.1,
        )

    left = (
        row(100, practical=1, best=2),
        row(101, practical=2, best=2),
        row(102, practical=3, best=1),
        row(103, practical=1, best=2),
        row(104, practical=1, best=2),
    )

    right = (
        row(100, practical=2, best=3),
        row(101, practical=3, best=2),
        row(102, practical=3, best=2),
        row(103, practical=2, best=1),
        row(104, practical=1, best=2),
    )

    stats = _compare_rounds(
        left=left,
        right=right,
    )

    assert stats.round_count == 5

    assert stats.practical_right_wins == 3
    assert stats.practical_left_wins == 0
    assert stats.practical_ties == 2

    assert stats.best_right_wins == 2
    assert stats.best_left_wins == 1
    assert stats.best_ties == 2

    assert stats.practical_sign_test_p_value == pytest.approx(
        0.25
    )
    assert stats.best_sign_test_p_value == pytest.approx(
        1.0
    )


def test_pairwise_round_statistics_reject_misaligned_rounds() -> None:
    from tools.validation.historical_replay_models import (
        ReplayRoundResult,
    )
    from tools.validation.regime_learning_comparison_runner import (
        _compare_rounds,
    )

    def row(round_no: int) -> ReplayRoundResult:
        return ReplayRoundResult(
            round_no=round_no,
            seed=20260000 + round_no,
            history_draws=600,
            noop_best_hits=0,
            adaptive_best_hits=1,
            noop_practical_hits=0,
            adaptive_practical_hits=1,
            noop_avg_jaccard=0.0,
            adaptive_avg_jaccard=0.0,
            probability_l1_delta=0.0,
            probability_max_delta=0.0,
            changed_probability_count=0,
            changed_set_count=0,
            profile_applied=False,
            profile_revision=None,
            profile_sample_size=None,
            elapsed_seconds=0.1,
        )

    with pytest.raises(
        ValueError,
        match="round alignment",
    ):
        _compare_rounds(
            left=(row(100), row(101)),
            right=(row(100), row(102)),
        )




def test_rank_scenarios_prefers_effectiveness_then_lower_perturbation() -> None:
    from tools.validation.regime_learning_comparison_runner import (
        _rank_scenarios,
    )

    class Scenario:
        def __init__(self, name: str) -> None:
            self.name = name

    class Effectiveness:
        def __init__(
            self,
            *,
            practical: float,
            best: float,
            wins: int,
            losses: int,
            probability_l1: float,
        ) -> None:
            self.practical_hit_mean_delta = practical
            self.best_hit_mean_delta = best
            self.practical_adaptive_wins = wins
            self.practical_noop_wins = losses
            self.average_probability_l1_delta = probability_l1

    class ScenarioResult:
        def __init__(
            self,
            name: str,
            effectiveness,
        ) -> None:
            self.scenario = Scenario(name)
            self.effectiveness = effectiveness

    rows = (
        ScenarioResult(
            "baseline",
            Effectiveness(
                practical=0.10,
                best=0.10,
                wins=4,
                losses=2,
                probability_l1=0.01,
            ),
        ),
        ScenarioResult(
            "calibration",
            Effectiveness(
                practical=0.20,
                best=0.10,
                wins=5,
                losses=2,
                probability_l1=0.03,
            ),
        ),
        ScenarioResult(
            "bayesian",
            Effectiveness(
                practical=0.20,
                best=0.15,
                wins=5,
                losses=2,
                probability_l1=0.04,
            ),
        ),
        ScenarioResult(
            "combined",
            Effectiveness(
                practical=0.20,
                best=0.15,
                wins=5,
                losses=2,
                probability_l1=0.02,
            ),
        ),
    )

    ranking = _rank_scenarios(rows)

    assert ranking == (
        "combined",
        "bayesian",
        "calibration",
        "baseline",
    )


def test_pairwise_result_includes_round_statistics(
    tmp_path,
    monkeypatch,
) -> None:
    import lrp.pipelines.prediction as prediction_module

    from lrp.io.draws import HistoryRow
    from lrp.regimes import (
        RegimeDecision,
        RegimeFeatureSnapshot,
    )
    from tools.validation.historical_replay_models import (
        ReplayConfig,
    )
    from tools.validation.regime_learning_comparison_runner import (
        run_regime_learning_comparison,
    )

    class FixedFeatureExtractor:
        def extract(self, snapshot):
            return snapshot

    class FixedStabilityPolicy:
        def decide(self, features):
            return RegimeDecision(
                primary="high_band_expansion",
                confidence=1.0,
                features=RegimeFeatureSnapshot(
                    average_recency=0.5,
                    average_frequency=0.5,
                    average_gap_reversion=0.5,
                    pair_density=0.5,
                    frequency_dispersion=0.5,
                    recency_variance=0.5,
                    pair_variance=0.5,
                    low_band_ratio=0.3,
                    high_band_ratio=0.7,
                ),
                scores={
                    "neutral": 0.0,
                    "mixed": 0.0,
                    "gap_recovery": 0.0,
                    "cluster_rotation": 0.0,
                    "high_band_expansion": 1.0,
                    "low_band_expansion": 0.0,
                },
            )

    monkeypatch.setattr(
        prediction_module,
        "GlobalRegimeFeatureExtractor",
        lambda: FixedFeatureExtractor(),
    )
    monkeypatch.setattr(
        prediction_module,
        "GlobalRegimeStabilityPolicy",
        lambda: FixedStabilityPolicy(),
    )

    history = []

    for round_no in range(1142, 1224):
        start = ((round_no - 1) * 7) % 45

        numbers = tuple(
            sorted(
                {
                    ((start + offset * 6) % 45) + 1
                    for offset in range(6)
                }
            )
        )

        bonus = next(
            number
            for number in range(1, 46)
            if number not in numbers
        )

        history.append(
            HistoryRow(
                round_no=round_no,
                numbers=numbers,
                bonus=bonus,
            )
        )

    result = run_regime_learning_comparison(
        history=tuple(history),
        config=ReplayConfig(
            start_round=1222,
            end_round=1223,
            candidate_count=100,
            top_k=10,
            practical_k=5,
            mode="fast",
        ),
        output_root=tmp_path / "comparison",
    )

    assert len(result.pairwise) == 4

    for item in result.pairwise:
        assert item.round_statistics.round_count == 2

        assert (
            item.round_statistics.practical_right_wins
            + item.round_statistics.practical_left_wins
            + item.round_statistics.practical_ties
            == 2
        )

        assert (
            item.round_statistics.best_right_wins
            + item.round_statistics.best_left_wins
            + item.round_statistics.best_ties
            == 2
        )

        assert (
            0.0
            <= item.round_statistics.practical_sign_test_p_value
            <= 1.0
        )

        assert (
            0.0
            <= item.round_statistics.best_sign_test_p_value
            <= 1.0
        )

    payload = result.as_dict()

    assert "round_statistics" in payload["pairwise"][0]
    assert (
        payload["pairwise"][0]["round_statistics"]["round_count"]
        == 2
    )


def test_comparison_result_includes_ranking(
    tmp_path,
    monkeypatch,
) -> None:
    import lrp.pipelines.prediction as prediction_module

    from lrp.io.draws import HistoryRow
    from lrp.regimes import (
        RegimeDecision,
        RegimeFeatureSnapshot,
    )
    from tools.validation.historical_replay_models import (
        ReplayConfig,
    )
    from tools.validation.regime_learning_comparison_runner import (
        run_regime_learning_comparison,
    )

    class FixedFeatureExtractor:
        def extract(self, snapshot):
            return snapshot

    class FixedStabilityPolicy:
        def decide(self, features):
            return RegimeDecision(
                primary="high_band_expansion",
                confidence=1.0,
                features=RegimeFeatureSnapshot(
                    average_recency=0.5,
                    average_frequency=0.5,
                    average_gap_reversion=0.5,
                    pair_density=0.5,
                    frequency_dispersion=0.5,
                    recency_variance=0.5,
                    pair_variance=0.5,
                    low_band_ratio=0.3,
                    high_band_ratio=0.7,
                ),
                scores={
                    "neutral": 0.0,
                    "mixed": 0.0,
                    "gap_recovery": 0.0,
                    "cluster_rotation": 0.0,
                    "high_band_expansion": 1.0,
                    "low_band_expansion": 0.0,
                },
            )

    monkeypatch.setattr(
        prediction_module,
        "GlobalRegimeFeatureExtractor",
        lambda: FixedFeatureExtractor(),
    )
    monkeypatch.setattr(
        prediction_module,
        "GlobalRegimeStabilityPolicy",
        lambda: FixedStabilityPolicy(),
    )

    history = []

    for round_no in range(1142, 1224):
        start = ((round_no - 1) * 7) % 45

        numbers = tuple(
            sorted(
                {
                    ((start + offset * 6) % 45) + 1
                    for offset in range(6)
                }
            )
        )

        bonus = next(
            number
            for number in range(1, 46)
            if number not in numbers
        )

        history.append(
            HistoryRow(
                round_no=round_no,
                numbers=numbers,
                bonus=bonus,
            )
        )

    result = run_regime_learning_comparison(
        history=tuple(history),
        config=ReplayConfig(
            start_round=1222,
            end_round=1223,
            candidate_count=100,
            top_k=10,
            practical_k=5,
            mode="fast",
        ),
        output_root=tmp_path / "comparison",
    )

    assert len(result.ranking) == 4
    assert set(result.ranking) == {
        "baseline",
        "calibration",
        "bayesian",
        "combined",
    }

    payload = result.as_dict()

    assert payload["ranking"] == list(
        result.ranking
    )


def test_comparison_writes_final_artifact(
    tmp_path,
    monkeypatch,
) -> None:
    import json
    import lrp.pipelines.prediction as prediction_module

    from lrp.io.draws import HistoryRow
    from lrp.regimes import (
        RegimeDecision,
        RegimeFeatureSnapshot,
    )
    from tools.validation.historical_replay_models import (
        ReplayConfig,
    )
    from tools.validation.regime_learning_comparison_runner import (
        run_regime_learning_comparison,
    )

    class FixedFeatureExtractor:
        def extract(self, snapshot):
            return snapshot

    class FixedStabilityPolicy:
        def decide(self, features):
            return RegimeDecision(
                primary="high_band_expansion",
                confidence=1.0,
                features=RegimeFeatureSnapshot(
                    average_recency=0.5,
                    average_frequency=0.5,
                    average_gap_reversion=0.5,
                    pair_density=0.5,
                    frequency_dispersion=0.5,
                    recency_variance=0.5,
                    pair_variance=0.5,
                    low_band_ratio=0.3,
                    high_band_ratio=0.7,
                ),
                scores={
                    "neutral": 0.0,
                    "mixed": 0.0,
                    "gap_recovery": 0.0,
                    "cluster_rotation": 0.0,
                    "high_band_expansion": 1.0,
                    "low_band_expansion": 0.0,
                },
            )

    monkeypatch.setattr(
        prediction_module,
        "GlobalRegimeFeatureExtractor",
        lambda: FixedFeatureExtractor(),
    )
    monkeypatch.setattr(
        prediction_module,
        "GlobalRegimeStabilityPolicy",
        lambda: FixedStabilityPolicy(),
    )

    history = []

    for round_no in range(1142, 1224):
        start = ((round_no - 1) * 7) % 45

        numbers = tuple(
            sorted(
                {
                    ((start + offset * 6) % 45) + 1
                    for offset in range(6)
                }
            )
        )

        bonus = next(
            number
            for number in range(1, 46)
            if number not in numbers
        )

        history.append(
            HistoryRow(
                round_no=round_no,
                numbers=numbers,
                bonus=bonus,
            )
        )

    output_root = tmp_path / "comparison"

    result = run_regime_learning_comparison(
        history=tuple(history),
        config=ReplayConfig(
            start_round=1222,
            end_round=1223,
            candidate_count=100,
            top_k=10,
            practical_k=5,
            mode="fast",
        ),
        output_root=output_root,
    )

    artifact = (
        output_root
        / "regime_learning_comparison.json"
    )

    assert artifact.exists()

    payload = json.loads(
        artifact.read_text(
            encoding="utf-8"
        )
    )

    assert payload == result.as_dict()
    assert payload["schema_version"] == 1
    assert payload["scenario_count"] == 4
    assert payload["pairwise_count"] == 4
    assert len(payload["ranking"]) == 4
