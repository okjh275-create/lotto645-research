from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from lrp.evolution.contracts.learning_context import (
    LearningContext,
)


def test_minimal_context() -> None:
    context = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
    )

    assert context.cycle_id == "cycle-1220"
    assert context.round_no == 1220
    assert context.version == 1
    assert context.signals == {}
    assert context.rewards == {}
    assert context.weights == {}
    assert context.metadata == {}


def test_context_normalizes_cycle_id() -> None:
    context = LearningContext(
        cycle_id=" cycle-1220 ",
        round_no=1220,
    )

    assert context.cycle_id == "cycle-1220"


def test_context_is_frozen() -> None:
    context = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
    )

    with pytest.raises(FrozenInstanceError):
        context.version = 2  # type: ignore[misc]


def test_nested_mappings_are_read_only() -> None:
    context = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
        signals={"hot": 0.7},
    )

    assert isinstance(
        context.signals,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        context.signals["hot"] = 0.1  # type: ignore[index]


def test_input_mapping_is_detached() -> None:
    signals = {
        "hot": 0.7,
    }
    context = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
        signals=signals,
    )

    signals["hot"] = 0.1

    assert context.signals["hot"] == pytest.approx(
        0.7
    )


def test_with_signals_returns_new_context() -> None:
    original = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
    )

    updated = original.with_signals(
        {
            "hot": 0.6,
            "gap": -0.2,
        }
    )

    assert original.signals == {}
    assert updated.signals["hot"] == pytest.approx(
        0.6
    )
    assert updated.signals["gap"] == pytest.approx(
        -0.2
    )


def test_with_rewards_returns_new_context() -> None:
    original = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
    )

    updated = original.with_rewards(
        {
            "strategy_a": 0.8,
        }
    )

    assert original.rewards == {}
    assert updated.rewards == {
        "strategy_a": 0.8,
    }


def test_with_selection_returns_new_context() -> None:
    original = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
    )

    updated = original.with_selection(
        policy="ucb1",
        arm="strategy_a",
    )

    assert original.selected_policy is None
    assert original.selected_arm is None
    assert updated.selected_policy == "ucb1"
    assert updated.selected_arm == "strategy_a"


def test_with_weights_returns_new_context() -> None:
    original = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
    )

    updated = original.with_weights(
        {
            "recency": 0.35,
            "frequency": 0.20,
        }
    )

    assert original.weights == {}
    assert updated.weights["recency"] == pytest.approx(
        0.35
    )


def test_with_metadata_returns_new_context() -> None:
    original = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
    )

    updated = original.with_metadata(
        {
            "seed": 20260731,
            "mode": "training",
            "persisted": False,
        }
    )

    assert original.metadata == {}
    assert updated.metadata["seed"] == 20260731
    assert updated.metadata["mode"] == "training"
    assert updated.metadata["persisted"] is False


def test_advance_version_returns_new_context() -> None:
    original = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
        version=3,
    )

    updated = original.advance_version()

    assert original.version == 3
    assert updated.version == 4


def test_snapshot_is_detached() -> None:
    context = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
        signals={"hot": 0.7},
        rewards={"strategy_a": 0.8},
        selected_policy="ucb1",
        selected_arm="strategy_a",
        weights={"recency": 0.35},
        metadata={"seed": 20260731},
    )

    snapshot = context.snapshot()

    assert snapshot == {
        "cycle_id": "cycle-1220",
        "round_no": 1220,
        "version": 1,
        "signals": {"hot": 0.7},
        "rewards": {"strategy_a": 0.8},
        "selected_policy": "ucb1",
        "selected_arm": "strategy_a",
        "weights": {"recency": 0.35},
        "metadata": {"seed": 20260731},
    }

    signals = snapshot["signals"]
    assert isinstance(signals, dict)

    signals["hot"] = 0.1

    assert context.signals["hot"] == pytest.approx(
        0.7
    )


@pytest.mark.parametrize(
    "cycle_id",
    ["", "   "],
)
def test_empty_cycle_id_is_rejected(
    cycle_id: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="cycle_id must not be empty",
    ):
        LearningContext(
            cycle_id=cycle_id,
            round_no=1220,
        )


@pytest.mark.parametrize(
    "round_no",
    [0, -1],
)
def test_invalid_round_number_is_rejected(
    round_no: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="round_no must be greater",
    ):
        LearningContext(
            cycle_id="cycle-1220",
            round_no=round_no,
        )


@pytest.mark.parametrize(
    "version",
    [0, -1],
)
def test_invalid_version_is_rejected(
    version: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="version must be greater",
    ):
        LearningContext(
            cycle_id="cycle-1220",
            round_no=1220,
            version=version,
        )


def test_non_finite_signal_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        LearningContext(
            cycle_id="cycle-1220",
            round_no=1220,
            signals={
                "hot": float("inf"),
            },
        )


def test_boolean_numeric_value_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="must be numeric",
    ):
        LearningContext(
            cycle_id="cycle-1220",
            round_no=1220,
            rewards={
                "strategy_a": True,
            },
        )


def test_empty_mapping_key_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="signals key must not be empty",
    ):
        LearningContext(
            cycle_id="cycle-1220",
            round_no=1220,
            signals={
                " ": 0.5,
            },
        )


def test_duplicate_normalized_key_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="duplicate normalized key",
    ):
        LearningContext(
            cycle_id="cycle-1220",
            round_no=1220,
            signals={
                "hot": 0.5,
                " hot ": 0.7,
            },
        )


def test_complex_metadata_value_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="scalar context values",
    ):
        LearningContext(
            cycle_id="cycle-1220",
            round_no=1220,
            metadata={
                "nested": {
                    "value": 1,
                },
            },  # type: ignore[arg-type]
        )


def test_non_finite_metadata_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        LearningContext(
            cycle_id="cycle-1220",
            round_no=1220,
            metadata={
                "score": float("nan"),
            },
        )


def test_empty_selected_policy_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="selected_policy must not be empty",
    ):
        LearningContext(
            cycle_id="cycle-1220",
            round_no=1220,
            selected_policy=" ",
        )


def test_empty_selected_arm_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="selected_arm must not be empty",
    ):
        LearningContext(
            cycle_id="cycle-1220",
            round_no=1220,
            selected_arm=" ",
        )
