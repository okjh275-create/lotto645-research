import pytest

from lrp.production import (
    ProductionChampionDecision,
)


def test_extracts_selected_model_from_selection() -> None:
    payload = {
        "selection": {
            "ranking_champion": "combined",
            "selected_model": "combined",
            "promotion": {
                "promoted": True,
            },
        },
        "matrix": {
            "ignored": True,
        },
    }

    decision = (
        ProductionChampionDecision.from_payload(
            payload
        )
    )

    assert decision.selected_model == "combined"


def test_none_selected_model_is_preserved() -> None:
    payload = {
        "selection": {
            "ranking_champion": "calibration",
            "selected_model": None,
            "promotion": {
                "promoted": False,
            },
        },
    }

    decision = (
        ProductionChampionDecision.from_payload(
            payload
        )
    )

    assert decision.selected_model is None


def test_extra_fields_are_ignored() -> None:
    payload = {
        "selection": {
            "ranking_champion": "bayesian",
            "selected_model": "bayesian",
            "promotion": {
                "promoted": True,
                "composite_margin": 0.123,
                "rejection_reasons": [],
            },
            "extra_selection_field": "ignored",
        },
        "matrix": {
            "large": "ignored",
        },
        "top_level_extra": {
            "anything": True,
        },
    }

    decision = (
        ProductionChampionDecision.from_payload(
            payload
        )
    )

    assert decision.selected_model == "bayesian"


def test_missing_selection_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="selection",
    ):
        ProductionChampionDecision.from_payload(
            {}
        )


def test_non_mapping_selection_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="selection",
    ):
        ProductionChampionDecision.from_payload(
            {
                "selection": "invalid",
            }
        )


def test_missing_selected_model_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="selected_model",
    ):
        ProductionChampionDecision.from_payload(
            {
                "selection": {
                    "ranking_champion": "baseline",
                },
            }
        )


def test_non_string_selected_model_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="selected_model",
    ):
        ProductionChampionDecision.from_payload(
            {
                "selection": {
                    "selected_model": 123,
                },
            }
        )


def test_non_mapping_payload_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="payload",
    ):
        ProductionChampionDecision.from_payload(
            []
        )


def test_serialization_contains_only_production_contract() -> None:
    decision = (
        ProductionChampionDecision.from_payload(
            {
                "selection": {
                    "ranking_champion": "combined",
                    "selected_model": "combined",
                    "promotion": {
                        "promoted": True,
                    },
                },
                "matrix": {
                    "must_not_leak": True,
                },
            }
        )
    )

    assert decision.as_dict() == {
        "selected_model": "combined",
    }
