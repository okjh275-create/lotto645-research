from __future__ import annotations

import pytest

from lrp.evolution.integration import (
    FeatureAttributionMapper,
)


def make_payload():

    probabilities = []

    for number in range(1, 46):

        probabilities.append(
            {
                "number": number,
                "components": {
                    "hot": number / 45,
                    "cold": 0.5,
                    "gap": 0.5,
                    "trend": 0.5,
                    "transition": 0.5,
                },
            }
        )

    return {
        "probability_vector": {
            "probabilities": probabilities,
        }
    }


def test_mapper_returns_all_components():

    result = FeatureAttributionMapper().map(
        make_payload(),
        (40,41,42,43,44,45),
    )

    assert set(result) == {
        "hot",
        "cold",
        "gap",
        "trend",
        "transition",
    }


def test_hot_signal_positive():

    result = FeatureAttributionMapper().map(
        make_payload(),
        (40,41,42,43,44,45),
    )

    assert result["hot"] > 0


def test_cold_signal_zero():

    result = FeatureAttributionMapper().map(
        make_payload(),
        (40,41,42,43,44,45),
    )

    assert result["cold"] == pytest.approx(0.0)
