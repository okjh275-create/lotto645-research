from __future__ import annotations

import pytest

from lrp.contracts import ContractError
from lrp.pipelines.prediction import PredictionPipeline
from lrp.pipelines.models import (
    PredictionRequest,
)

def make_request() -> PredictionRequest:
    return PredictionRequest(
        round_no=1220,
        seed=20260802,
        candidate_count=10,
        top_k=5,
        practical_k=3,
        previous_numbers=frozenset(
            {1, 2, 3, 4, 5, 6}
        ),
        long_gap_numbers=frozenset(
            {7, 8, 9}
        ),
    )

from lrp.regimes.integration import (
    ActiveGlobalRegimeAdjustmentAdapter,
    GlobalRegimeAdjustmentAdapter,
    NoOpGlobalRegimeAdjustmentAdapter,
)


class RecordingGlobalRegimeAdapter:
    def adjust(
        self,
        probability_vector: object,
        *,
        global_regime: object | None,
        round_no: int,
        seed: int,
    ) -> object:
        return probability_vector


def test_pipeline_uses_noop_global_regime_adapter_by_default() -> None:
    pipeline = PredictionPipeline(
        statistics=object(),  # type: ignore[arg-type]
        candidate=object(),  # type: ignore[arg-type]
    )

    assert isinstance(
        pipeline.global_regime_adjustment,
        NoOpGlobalRegimeAdjustmentAdapter,
    )


def test_pipeline_accepts_structural_global_regime_adapter() -> None:
    adapter = RecordingGlobalRegimeAdapter()

    pipeline = PredictionPipeline(
        statistics=object(),  # type: ignore[arg-type]
        candidate=object(),  # type: ignore[arg-type]
        global_regime_adjustment=adapter,
    )

    assert pipeline.global_regime_adjustment is adapter
    assert isinstance(
        adapter,
        GlobalRegimeAdjustmentAdapter,
    )


def test_pipeline_rejects_invalid_global_regime_adapter() -> None:
    with pytest.raises(
        ContractError,
        match="GlobalRegimeAdjustmentAdapter",
    ):
        PredictionPipeline(
            statistics=object(),  # type: ignore[arg-type]
            candidate=object(),  # type: ignore[arg-type]
            global_regime_adjustment=object(),
        )


class RecordingAdjustmentAdapter:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[
            tuple[object, object | None, int, int]
        ] = []

    def adjust(
        self,
        probability_vector: object,
        *,
        global_regime: object | None,
        round_no: int,
        seed: int,
    ) -> object:
        self.calls.append(
            (
                probability_vector,
                global_regime,
                round_no,
                seed,
            )
        )
        return self.result


class NoneAdjustmentAdapter:
    def adjust(
        self,
        probability_vector: object,
        *,
        global_regime: object | None,
        round_no: int,
        seed: int,
    ) -> object:
        return None


def test_global_regime_adjustment_receives_context_round_and_seed() -> None:
    original = object()
    adjusted = object()
    regime = object()

    adapter = RecordingAdjustmentAdapter(
        adjusted
    )

    pipeline = PredictionPipeline(
        statistics=object(),  # type: ignore[arg-type]
        candidate=object(),  # type: ignore[arg-type]
        global_regime_adjustment=adapter,
    )

    request = make_request()

    result = (
        pipeline
        ._adjust_global_regime_probability_vector(
            original,
            global_regime=regime,
            request=request,
        )
    )

    assert result is adjusted
    assert adapter.calls == [
        (
            original,
            regime,
            1220,
            20260802,
        )
    ]


def test_none_global_regime_adjustment_is_rejected() -> None:
    pipeline = PredictionPipeline(
        statistics=object(),  # type: ignore[arg-type]
        candidate=object(),  # type: ignore[arg-type]
        global_regime_adjustment=(
            NoneAdjustmentAdapter()
        ),
    )

    with pytest.raises(
        ContractError,
        match="returned None",
    ):
        pipeline._adjust_global_regime_probability_vector(
            object(),
            global_regime=object(),
            request=make_request(),
        )

def test_pipeline_load_accepts_active_global_regime_adapter() -> None:
    adapter = ActiveGlobalRegimeAdjustmentAdapter()

    pipeline = PredictionPipeline.load(
        global_regime_adjustment=adapter,
    )

    assert (
        pipeline.global_regime_adjustment
        is adapter
    )

def test_pipeline_load_defaults_to_noop_global_regime_adjustment() -> None:
    from lrp.regimes.integration import (
        NoOpGlobalRegimeAdjustmentAdapter,
    )

    pipeline = PredictionPipeline.load()

    assert isinstance(
        pipeline.global_regime_adjustment,
        NoOpGlobalRegimeAdjustmentAdapter,
    )


def test_pipeline_load_builds_repository_backed_active_adjustment(
    tmp_path,
) -> None:
    from lrp.regimes.integration import (
        ActiveGlobalRegimeAdjustmentAdapter,
        RepositoryRegimeCalibrationProvider,
    )

    root = tmp_path / "regime-calibration"

    pipeline = PredictionPipeline.load(
        regime_calibration_snapshot_root=root,
    )

    adapter = pipeline.global_regime_adjustment

    assert isinstance(
        adapter,
        ActiveGlobalRegimeAdjustmentAdapter,
    )
    assert isinstance(
        adapter.calibration_provider,
        RepositoryRegimeCalibrationProvider,
    )
    assert (
        adapter.calibration_provider.repository.root
        == root
    )


def test_explicit_global_regime_adjustment_overrides_calibration_root(
    tmp_path,
) -> None:
    from lrp.regimes.integration import (
        NoOpGlobalRegimeAdjustmentAdapter,
    )

    explicit = NoOpGlobalRegimeAdjustmentAdapter()

    pipeline = PredictionPipeline.load(
        global_regime_adjustment=explicit,
        regime_calibration_snapshot_root=(
            tmp_path / "regime-calibration"
        ),
    )

    assert (
        pipeline.global_regime_adjustment
        is explicit
    )

def test_pipeline_load_builds_repository_backed_bayesian_adjustment(
    tmp_path,
) -> None:
    from lrp.regimes.integration import (
        ActiveGlobalRegimeAdjustmentAdapter,
        RepositoryRegimeBayesianProvider,
    )

    root = tmp_path / "regime-bayesian"

    pipeline = PredictionPipeline.load(
        regime_bayesian_snapshot_root=root,
    )

    adapter = pipeline.global_regime_adjustment

    assert isinstance(
        adapter,
        ActiveGlobalRegimeAdjustmentAdapter,
    )
    assert isinstance(
        adapter.bayesian_provider,
        RepositoryRegimeBayesianProvider,
    )
    assert (
        adapter.bayesian_provider.repository.root
        == root
    )


def test_pipeline_load_combines_calibration_and_bayesian_providers(
    tmp_path,
) -> None:
    from lrp.regimes.integration import (
        ActiveGlobalRegimeAdjustmentAdapter,
        RepositoryRegimeBayesianProvider,
        RepositoryRegimeCalibrationProvider,
    )

    calibration_root = (
        tmp_path / "regime-calibration"
    )
    bayesian_root = (
        tmp_path / "regime-bayesian"
    )

    pipeline = PredictionPipeline.load(
        regime_calibration_snapshot_root=(
            calibration_root
        ),
        regime_bayesian_snapshot_root=(
            bayesian_root
        ),
    )

    adapter = pipeline.global_regime_adjustment

    assert isinstance(
        adapter,
        ActiveGlobalRegimeAdjustmentAdapter,
    )

    assert isinstance(
        adapter.calibration_provider,
        RepositoryRegimeCalibrationProvider,
    )
    assert isinstance(
        adapter.bayesian_provider,
        RepositoryRegimeBayesianProvider,
    )

    assert (
        adapter.calibration_provider.repository.root
        == calibration_root
    )
    assert (
        adapter.bayesian_provider.repository.root
        == bayesian_root
    )


def test_explicit_global_regime_adjustment_overrides_bayesian_root(
    tmp_path,
) -> None:
    from lrp.regimes.integration import (
        NoOpGlobalRegimeAdjustmentAdapter,
    )

    explicit = NoOpGlobalRegimeAdjustmentAdapter()

    pipeline = PredictionPipeline.load(
        global_regime_adjustment=explicit,
        regime_bayesian_snapshot_root=(
            tmp_path / "regime-bayesian"
        ),
    )

    assert (
        pipeline.global_regime_adjustment
        is explicit
    )
