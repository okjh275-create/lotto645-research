from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from lrp.evolution.algorithms.signals import (
    SignalAggregator,
    WeightedSignalAggregator,
)
from lrp.evolution.contracts.signals import (
    SIGNAL_COMPONENTS,
    SignalAggregationResult,
    SignalFrame,
    SignalSource,
)


class ExampleSignalSource:
    def produce(self) -> SignalFrame:
        return SignalFrame(
            source="example",
            signals={"hot": 0.5},
        )


def test_signal_source_protocol() -> None:
    source = ExampleSignalSource()

    assert isinstance(source, SignalSource)
    assert source.produce().source == "example"


def test_signal_frame_normalizes_source() -> None:
    frame = SignalFrame(
        source=" bayesian ",
        signals={"hot": 0.5},
    )

    assert frame.source == "bayesian"


def test_signal_frame_signals_are_immutable() -> None:
    frame = SignalFrame(
        source="bayesian",
        signals={"hot": 0.5},
    )

    assert isinstance(
        frame.signals,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        frame.signals["hot"] = 0.2  # type: ignore[index]


def test_signal_frame_is_frozen() -> None:
    frame = SignalFrame(
        source="bayesian",
        signals={"hot": 0.5},
    )

    with pytest.raises(FrozenInstanceError):
        frame.source = "changed"  # type: ignore[misc]


def test_signal_frame_rejects_empty_source() -> None:
    with pytest.raises(
        ValueError,
        match="source must not be empty",
    ):
        SignalFrame(
            source=" ",
            signals={"hot": 0.5},
        )


def test_signal_frame_rejects_empty_signals() -> None:
    with pytest.raises(
        ValueError,
        match="at least one item",
    ):
        SignalFrame(
            source="bayesian",
            signals={},
        )


def test_signal_frame_rejects_unknown_component() -> None:
    with pytest.raises(
        ValueError,
        match="unknown signal component",
    ):
        SignalFrame(
            source="bayesian",
            signals={"unknown": 0.5},
        )


@pytest.mark.parametrize(
    "value",
    [-1.01, 1.01],
)
def test_signal_frame_rejects_out_of_range_signal(
    value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="between -1.0 and 1.0",
    ):
        SignalFrame(
            source="bayesian",
            signals={"hot": value},
        )


@pytest.mark.parametrize(
    "reliability",
    [-0.01, 1.01],
)
def test_signal_frame_rejects_invalid_reliability(
    reliability: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="reliability must be between",
    ):
        SignalFrame(
            source="bayesian",
            signals={"hot": 0.5},
            reliability=reliability,
        )


def test_signal_for_returns_existing_signal() -> None:
    frame = SignalFrame(
        source="bayesian",
        signals={"hot": 0.5},
    )

    assert frame.signal_for("hot") == 0.5
    assert frame.signal_for("cold") is None


def test_equal_aggregation_averages_signals() -> None:
    frames = [
        SignalFrame(
            source="bayesian",
            signals={
                "hot": 1.0,
                "cold": -1.0,
            },
        ),
        SignalFrame(
            source="scenario",
            signals={
                "hot": 0.0,
                "cold": 1.0,
            },
        ),
    ]

    result = SignalAggregator().aggregate(
        frames
    )

    assert result.signals["hot"] == pytest.approx(
        0.5
    )
    assert result.signals["cold"] == pytest.approx(
        0.0
    )


def test_missing_component_does_not_vote() -> None:
    frames = [
        SignalFrame(
            source="bayesian",
            signals={"hot": 1.0},
        ),
        SignalFrame(
            source="scenario",
            signals={"cold": -1.0},
        ),
    ]

    result = SignalAggregator().aggregate(
        frames
    )

    assert result.signals["hot"] == pytest.approx(
        1.0
    )
    assert result.signals["cold"] == pytest.approx(
        -1.0
    )


def test_unrepresented_component_is_neutral() -> None:
    result = SignalAggregator().aggregate(
        [
            SignalFrame(
                source="bayesian",
                signals={"hot": 0.5},
            )
        ]
    )

    for component in SIGNAL_COMPONENTS:
        if component == "hot":
            assert result.signals[component] == 0.5
        else:
            assert result.signals[component] == 0.0


def test_reliability_affects_aggregation() -> None:
    frames = [
        SignalFrame(
            source="bayesian",
            signals={"hot": 1.0},
            reliability=1.0,
        ),
        SignalFrame(
            source="manual",
            signals={"hot": -1.0},
            reliability=0.5,
        ),
    ]

    result = SignalAggregator().aggregate(
        frames
    )

    assert result.signals["hot"] == pytest.approx(
        1.0 / 3.0
    )
    assert result.total_weight == pytest.approx(
        1.5
    )


def test_source_weight_affects_aggregation() -> None:
    frames = [
        SignalFrame(
            source="bayesian",
            signals={"hot": 1.0},
        ),
        SignalFrame(
            source="scenario",
            signals={"hot": -1.0},
        ),
    ]

    result = WeightedSignalAggregator().aggregate(
        frames,
        source_weights={
            "bayesian": 3.0,
            "scenario": 1.0,
        },
    )

    assert result.signals["hot"] == pytest.approx(
        0.5
    )
    assert result.total_weight == pytest.approx(
        4.0
    )


def test_zero_weight_source_does_not_vote() -> None:
    frames = [
        SignalFrame(
            source="bayesian",
            signals={"hot": 1.0},
        ),
        SignalFrame(
            source="manual",
            signals={"hot": -1.0},
        ),
    ]

    result = WeightedSignalAggregator().aggregate(
        frames,
        source_weights={
            "manual": 0.0,
        },
    )

    assert result.signals["hot"] == pytest.approx(
        1.0
    )


def test_all_zero_effective_weights_are_rejected() -> None:
    frames = [
        SignalFrame(
            source="bayesian",
            signals={"hot": 1.0},
            reliability=0.0,
        )
    ]

    with pytest.raises(
        ValueError,
        match="effective source weight total",
    ):
        SignalAggregator().aggregate(frames)


def test_duplicate_sources_are_rejected() -> None:
    frames = [
        SignalFrame(
            source="bayesian",
            signals={"hot": 1.0},
        ),
        SignalFrame(
            source="bayesian",
            signals={"cold": -1.0},
        ),
    ]

    with pytest.raises(
        ValueError,
        match="duplicate signal source",
    ):
        SignalAggregator().aggregate(frames)


def test_unknown_source_weight_is_rejected() -> None:
    frames = [
        SignalFrame(
            source="bayesian",
            signals={"hot": 1.0},
        )
    ]

    with pytest.raises(
        ValueError,
        match="unknown source weights",
    ):
        WeightedSignalAggregator().aggregate(
            frames,
            source_weights={"unknown": 1.0},
        )


def test_negative_source_weight_is_rejected() -> None:
    frames = [
        SignalFrame(
            source="bayesian",
            signals={"hot": 1.0},
        )
    ]

    with pytest.raises(
        ValueError,
        match="greater than or equal to 0",
    ):
        WeightedSignalAggregator().aggregate(
            frames,
            source_weights={"bayesian": -1.0},
        )


def test_empty_frames_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="at least one item",
    ):
        SignalAggregator().aggregate([])


def test_invalid_frame_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="SignalFrame",
    ):
        SignalAggregator().aggregate(
            [object()]  # type: ignore[list-item]
        )


def test_result_contains_all_components() -> None:
    result = SignalAggregator().aggregate(
        [
            SignalFrame(
                source="bayesian",
                signals={"hot": 0.5},
            )
        ]
    )

    assert isinstance(
        result,
        SignalAggregationResult,
    )
    assert tuple(result.signals) == SIGNAL_COMPONENTS
    assert result.source_count == 1


def test_custom_weighted_aggregator_is_preserved() -> None:
    weighted = WeightedSignalAggregator()
    aggregator = SignalAggregator(weighted)

    assert aggregator.weighted_aggregator is weighted


def test_invalid_weighted_aggregator_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="WeightedSignalAggregator or None",
    ):
        SignalAggregator(
            object()  # type: ignore[arg-type]
        )
