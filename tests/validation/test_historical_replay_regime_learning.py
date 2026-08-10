from __future__ import annotations

from pathlib import Path

import pytest

import lrp.pipelines.prediction as prediction_module
from lrp.io.draws import HistoryRow
from lrp.regimes import (
    RegimeDecision,
    RegimeFeatureSnapshot,
)
from lrp.regimes.calibration_repository import (
    RegimeCalibrationNotFoundError,
    RegimeCalibrationRepository,
)
from tools.validation.historical_replay_executor import (
    HistoricalReplayExecutor,
)
from tools.validation.historical_replay_models import (
    ReplayConfig,
)


def make_history() -> tuple[HistoryRow, ...]:
    rows: list[HistoryRow] = []

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

        if len(numbers) != 6:
            raise AssertionError(
                f"invalid synthetic draw: {numbers}"
            )

        bonus = next(
            number
            for number in range(1, 46)
            if number not in numbers
        )

        rows.append(
            HistoryRow(
                round_no=round_no,
                numbers=numbers,
                bonus=bonus,
            )
        )

    return tuple(rows)


def make_regime() -> RegimeDecision:
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
        return make_regime()


def test_two_round_replay_applies_previous_round_calibration_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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

    load_observations: list[int | None] = []

    original_load_latest = (
        RegimeCalibrationRepository.load_latest
    )

    def observed_load_latest(
        repository: RegimeCalibrationRepository,
        *,
        skip_corrupt: bool = True,
    ):
        try:
            snapshot = original_load_latest(
                repository,
                skip_corrupt=skip_corrupt,
            )
        except RegimeCalibrationNotFoundError:
            load_observations.append(None)
            raise

        load_observations.append(
            snapshot.revision
        )
        return snapshot

    monkeypatch.setattr(
        RegimeCalibrationRepository,
        "load_latest",
        observed_load_latest,
    )

    history = make_history()
    by_round = {
        row.round_no: row
        for row in history
    }

    regime_root = (
        tmp_path / "regime-calibration"
    )

    executor = HistoricalReplayExecutor(
        history=history,
        config=ReplayConfig(
            start_round=1222,
            end_round=1223,
            candidate_count=100,
            top_k=10,
            practical_k=5,
            mode="fast",
        ),
        learning_root=tmp_path / "learning",
        profile_root=tmp_path / "profiles",
        regime_calibration_root=regime_root,
    )

    first_row, first_state = executor(
        1222,
        executor.config.seed_for_round(1222),
        by_round[1222],
        None,
    )

    repository = RegimeCalibrationRepository(
        regime_root
    )

    assert first_row.round_no == 1222
    assert repository.revisions() == (1,)

    second_row, second_state = executor(
        1223,
        executor.config.seed_for_round(1223),
        by_round[1223],
        first_state,
    )

    assert second_row.round_no == 1223
    assert second_state is not None
    assert repository.revisions() == (
        1,
        2,
    )

    assert load_observations[:2] == [
        None,
        None,
    ]

    assert 1 in load_observations[2:]