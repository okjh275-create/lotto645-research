from __future__ import annotations

from lrp.evaluation import (
    ChampionPromotionPolicy,
    ModelEvaluation,
    ModelRankingEntry,
)

from lrp.evaluation.selection import (
    ChampionSelection,
    select_champion,
)


def ranking_entry(
    *,
    rank: int,
    name: str,
    composite: float,
    practical: float = 0.10,
    significance: float = 1.0,
    eligible: bool = True,
) -> ModelRankingEntry:
    return ModelRankingEntry(
        rank=rank,
        model_name=name,
        practical_score=practical,
        best_score=0.10,
        stability_score=0.10,
        significance_score=significance,
        composite_score=composite,
        eligible=eligible,
        exclusion_reasons=(
            ()
            if eligible
            else ("ranking_ineligible",)
        ),
    )


def test_selects_promoted_champion_from_ranking() -> None:
    entries = (
        ranking_entry(
            rank=1,
            name="combined",
            composite=0.20,
        ),
        ranking_entry(
            rank=2,
            name="calibration",
            composite=0.10,
        ),
    )

    result = select_champion(
        entries=entries,
        policy=ChampionPromotionPolicy(),
    )

    assert isinstance(
        result,
        ChampionSelection,
    )

    assert result.ranking_champion == "combined"
    assert result.promotion.candidate == "combined"
    assert result.promotion.promoted is True
    assert result.selected_model == "combined"


def test_holds_when_ranking_champion_fails_promotion() -> None:
    entries = (
        ranking_entry(
            rank=1,
            name="calibration",
            composite=0.016667,
            practical=0.0,
            significance=0.0,
        ),
        ranking_entry(
            rank=2,
            name="bayesian",
            composite=0.016250,
            practical=-0.016667,
            significance=0.0,
        ),
    )

    result = select_champion(
        entries=entries,
        policy=ChampionPromotionPolicy(),
    )

    assert result.ranking_champion == "calibration"
    assert result.promotion.candidate == "calibration"

    assert result.promotion.promoted is False
    assert result.selected_model is None

    assert (
        "significance_below_minimum"
        in result.promotion.rejection_reasons
    )

    assert (
        "composite_margin_below_minimum"
        in result.promotion.rejection_reasons
    )


def test_ranking_champion_can_differ_from_selected_model_state() -> None:
    entries = (
        ranking_entry(
            rank=1,
            name="candidate",
            composite=0.20,
            significance=0.0,
        ),
        ranking_entry(
            rank=2,
            name="runner_up",
            composite=0.10,
        ),
    )

    result = select_champion(
        entries=entries,
    )

    assert result.ranking_champion == "candidate"
    assert result.selected_model is None
    assert result.promotion.promoted is False


def test_selection_serialization_is_json_ready() -> None:
    entries = (
        ranking_entry(
            rank=1,
            name="combined",
            composite=0.20,
        ),
        ranking_entry(
            rank=2,
            name="baseline",
            composite=0.10,
        ),
    )

    result = select_champion(
        entries=entries,
    )

    payload = result.as_dict()

    assert payload["ranking_champion"] == "combined"
    assert payload["selected_model"] == "combined"

    assert payload["promotion"]["candidate"] == "combined"
    assert payload["promotion"]["promoted"] is True

    assert payload["promotion"]["rejection_reasons"] == []


def test_no_eligible_model_produces_hold() -> None:
    entries = (
        ranking_entry(
            rank=1,
            name="first",
            composite=0.20,
            eligible=False,
        ),
        ranking_entry(
            rank=2,
            name="second",
            composite=0.10,
            eligible=False,
        ),
    )

    result = select_champion(
        entries=entries,
    )

    assert result.ranking_champion is None
    assert result.selected_model is None

    assert result.promotion.candidate is None
    assert result.promotion.promoted is False

    assert result.promotion.rejection_reasons == (
        "no_eligible_candidate",
    )


def test_selection_does_not_require_model_evaluation_objects() -> None:
    entries = (
        ranking_entry(
            rank=1,
            name="combined",
            composite=0.20,
        ),
    )

    result = select_champion(
        entries=entries,
    )

    assert result.selected_model == "combined"


def test_model_evaluation_import_remains_available() -> None:
    assert ModelEvaluation.__name__ == "ModelEvaluation"
