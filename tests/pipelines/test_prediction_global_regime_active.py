from __future__ import annotations

import lrp.pipelines.prediction as prediction_module

from lrp.pipelines import (
    PredictionPipeline,
    PredictionRequest,
)
from lrp.regimes import (
    RegimeDecision,
    RegimeFeatureSnapshot,
)
from lrp.regimes.integration import (
    ActiveGlobalRegimeAdjustmentAdapter,
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

        draws.append(
            draw_type(
                round=round_no,
                numbers=numbers,
                bonus=None,
            )
        )

    return tuple(draws)


def make_request(
    previous_numbers: frozenset[int],
) -> PredictionRequest:
    return PredictionRequest(
        round_no=81,
        seed=20260721,
        temperature=0.85,
        candidate_count=200,
        max_attempts_multiplier=100,
        top_k=10,
        practical_k=5,
        previous_numbers=previous_numbers,
        long_gap_numbers=frozenset(range(1, 46)),
    )


def make_high_band_regime() -> RegimeDecision:
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


class FixedFeatureExtractor:
    def extract(self, snapshot: object) -> object:
        return snapshot


class FixedStabilityPolicy:
    def decide(self, features: object) -> RegimeDecision:
        return make_high_band_regime()


def test_active_pipeline_changes_probability_vector(
    monkeypatch,
) -> None:
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

    baseline = PredictionPipeline.load()
    active = PredictionPipeline.load(
        global_regime_adjustment=(
            ActiveGlobalRegimeAdjustmentAdapter()
        ),
    )

    draws = build_draws(
        baseline.statistics.module
    )
    request = make_request(
        frozenset(draws[-1].numbers)
    )

    baseline_result = baseline.run(
        draws,
        request,
    )
    active_result = active.run(
        draws,
        request,
    )

    baseline_vector = (
        baseline_result.generation.probability_vector
    )
    active_vector = (
        active_result.generation.probability_vector
    )

    assert baseline_vector is not None
    assert active_vector is not None

    baseline_probabilities = tuple(
        item.probability
        for item in baseline_vector.probabilities
    )
    active_probabilities = tuple(
        item.probability
        for item in active_vector.probabilities
    )

    assert (
        active_probabilities
        != baseline_probabilities
    )

    assert (
        active_vector.metadata[
            "global_regime_adjusted"
        ]
        is True
    )

    assert (
        "global_regime_adjusted"
        not in baseline_vector.metadata
    )


def test_repository_calibration_strengthens_active_pipeline(
    tmp_path,
    monkeypatch,
) -> None:
    from datetime import datetime, timezone

    from lrp.evolution.contracts.regime_calibration import (
        RegimeCalibration,
    )
    from lrp.regimes.calibration_repository import (
        RegimeCalibrationRepository,
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

    root = tmp_path / "regime-calibration"

    repository = RegimeCalibrationRepository(
        root
    )

    repository.save(
        RegimeCalibration(
            gap_recovery=1.0,
            cluster_rotation=1.0,
            high_band_expansion=1.20,
            low_band_expansion=1.0,
        ),
        revision=1,
        sample_size=20,
        saved_at=datetime(
            2026,
            8,
            10,
            tzinfo=timezone.utc,
        ),
    )

    baseline = PredictionPipeline.load(
        global_regime_adjustment=(
            ActiveGlobalRegimeAdjustmentAdapter()
        ),
    )

    learned = PredictionPipeline.load(
        regime_calibration_snapshot_root=root,
    )

    draws = build_draws(
        baseline.statistics.module
    )
    request = make_request(
        frozenset(draws[-1].numbers)
    )

    baseline_result = baseline.run(
        draws,
        request,
    )
    learned_result = learned.run(
        draws,
        request,
    )

    baseline_vector = (
        baseline_result.generation
        .probability_vector
    )
    learned_vector = (
        learned_result.generation
        .probability_vector
    )

    assert baseline_vector is not None
    assert learned_vector is not None

    assert (
        learned_vector.get(45).raw_score
        > baseline_vector.get(45).raw_score
    )

    baseline_probabilities = tuple(
        item.probability
        for item in baseline_vector.probabilities
    )
    learned_probabilities = tuple(
        item.probability
        for item in learned_vector.probabilities
    )

    assert (
        learned_probabilities
        != baseline_probabilities
    )

    assert abs(
        sum(
            item.probability
            for item in learned_vector.probabilities
        )
        - 1.0
    ) < 1e-9

    assert (
        learned_result.generation
        .global_regime_mode
        == "active"
    )

    assert (
        learned_vector.metadata[
            "global_regime_adjusted"
        ]
        is True
    )