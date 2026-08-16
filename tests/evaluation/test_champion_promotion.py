from __future__ import annotations

from dataclasses import replace

import pytest

from lrp.contracts import ContractError
from lrp.evaluation import (
    ChampionPromotionPolicy,
    ModelRankingEntry,
    evaluate_champion_promotion,
)


def make_entry(
    *,
    rank: int,
    name: str,
    practical: float,
    best: float,
    stability: float,
    significance: float,
    composite: float,
    eligible: bool = True,
) -> ModelRankingEntry:
    return ModelRankingEntry(
        rank=rank,
        model_name=name,
        practical_score=practical,
        best_score=best,
        stability_score=stability,
        significance_score=significance,
        composite_score=composite,
        eligible=eligible,
        exclusion_reasons=(
            ()
            if eligible
            else ("ranking_ineligible",)
        ),
    )


def default_policy() -> ChampionPromotionPolicy:
    return ChampionPromotionPolicy(
        minimum_composite_score=0.0,
        minimum_practical_score=0.0,
        minimum_significance_score=1.0,
        minimum_composite_margin=0.01,
    )


def test_strong_leader_is_promoted() -> None:
    entries = (
        make_entry(
            rank=1,
            name="challenger",
            practical=0.08,
            best=0.12,
            stability=0.02,
            significance=1.0,
            composite=0.08,
        ),
        make_entry(
            rank=2,
            name="runner_up",
            practical=0.04,
            best=0.08,
            stability=0.01,
            significance=1.0,
            composite=0.05,
        ),
    )

    result = evaluate_champion_promotion(
        entries=entries,
        policy=default_policy(),
    )

    assert result.candidate == "challenger"
    assert result.promoted is True
    assert result.promoted_model == "challenger"
    assert result.rejection_reasons == ()
    assert result.composite_margin == pytest.approx(
        0.03
    )


def test_zero_significance_blocks_promotion() -> None:
    entries = (
        make_entry(
            rank=1,
            name="calibration",
            practical=0.02,
            best=0.12,
            stability=0.01,
            significance=0.0,
            composite=0.05,
        ),
        make_entry(
            rank=2,
            name="bayesian",
            practical=0.01,
            best=0.10,
            stability=0.00,
            significance=0.0,
            composite=0.02,
        ),
    )

    result = evaluate_champion_promotion(
        entries=entries,
        policy=default_policy(),
    )

    assert result.promoted is False
    assert result.promoted_model is None
    assert (
        "significance_below_minimum"
        in result.rejection_reasons
    )


def test_negative_practical_score_blocks_promotion() -> None:
    entries = (
        make_entry(
            rank=1,
            name="challenger",
            practical=-0.01,
            best=0.20,
            stability=0.05,
            significance=1.0,
            composite=0.08,
        ),
        make_entry(
            rank=2,
            name="runner_up",
            practical=-0.02,
            best=0.10,
            stability=0.00,
            significance=1.0,
            composite=0.04,
        ),
    )

    result = evaluate_champion_promotion(
        entries=entries,
        policy=default_policy(),
    )

    assert result.promoted is False
    assert (
        "practical_score_below_minimum"
        in result.rejection_reasons
    )


def test_nonpositive_composite_blocks_promotion() -> None:
    entries = (
        make_entry(
            rank=1,
            name="challenger",
            practical=0.02,
            best=0.02,
            stability=0.00,
            significance=1.0,
            composite=0.0,
        ),
        make_entry(
            rank=2,
            name="runner_up",
            practical=0.01,
            best=0.01,
            stability=0.00,
            significance=1.0,
            composite=-0.02,
        ),
    )

    result = evaluate_champion_promotion(
        entries=entries,
        policy=default_policy(),
    )

    assert result.promoted is False
    assert (
        "composite_score_not_positive"
        in result.rejection_reasons
    )


def test_small_lead_blocks_promotion() -> None:
    entries = (
        make_entry(
            rank=1,
            name="calibration",
            practical=0.03,
            best=0.12,
            stability=0.01,
            significance=1.0,
            composite=0.050,
        ),
        make_entry(
            rank=2,
            name="bayesian",
            practical=0.02,
            best=0.11,
            stability=0.01,
            significance=1.0,
            composite=0.045,
        ),
    )

    result = evaluate_champion_promotion(
        entries=entries,
        policy=default_policy(),
    )

    assert result.promoted is False
    assert result.composite_margin == pytest.approx(
        0.005
    )
    assert (
        "composite_margin_below_minimum"
        in result.rejection_reasons
    )


def test_ineligible_rank_one_is_not_candidate() -> None:
    entries = (
        make_entry(
            rank=1,
            name="ineligible",
            practical=1.0,
            best=1.0,
            stability=1.0,
            significance=2.0,
            composite=1.0,
            eligible=False,
        ),
        make_entry(
            rank=2,
            name="eligible",
            practical=0.10,
            best=0.10,
            stability=0.10,
            significance=1.0,
            composite=0.10,
        ),
    )

    result = evaluate_champion_promotion(
        entries=entries,
        policy=default_policy(),
    )

    assert result.candidate == "eligible"
    assert result.promoted is True
    assert result.promoted_model == "eligible"


def test_no_eligible_candidate_returns_no_promotion() -> None:
    entries = (
        make_entry(
            rank=1,
            name="first",
            practical=-1.0,
            best=-1.0,
            stability=-1.0,
            significance=0.0,
            composite=-1.0,
            eligible=False,
        ),
        make_entry(
            rank=2,
            name="second",
            practical=-1.0,
            best=-1.0,
            stability=-1.0,
            significance=0.0,
            composite=-1.0,
            eligible=False,
        ),
    )

    result = evaluate_champion_promotion(
        entries=entries,
        policy=default_policy(),
    )

    assert result.candidate is None
    assert result.promoted is False
    assert result.promoted_model is None
    assert result.composite_margin is None
    assert result.rejection_reasons == (
        "no_eligible_candidate",
    )


def test_single_eligible_candidate_does_not_require_margin() -> None:
    entries = (
        make_entry(
            rank=1,
            name="only",
            practical=0.10,
            best=0.10,
            stability=0.10,
            significance=1.0,
            composite=0.10,
        ),
    )

    result = evaluate_champion_promotion(
        entries=entries,
        policy=default_policy(),
    )

    assert result.promoted is True
    assert result.composite_margin is None


def test_policy_rejects_invalid_thresholds() -> None:
    policy = default_policy()

    with pytest.raises(ContractError):
        replace(
            policy,
            minimum_composite_margin=-0.01,
        )

    with pytest.raises(ContractError):
        replace(
            policy,
            minimum_significance_score=-1.0,
        )


def test_long120_like_result_is_not_promoted() -> None:
    entries = (
        make_entry(
            rank=1,
            name="calibration",
            practical=0.000000,
            best=0.116667,
            stability=-0.062500,
            significance=0.000000,
            composite=0.016667,
        ),
        make_entry(
            rank=2,
            name="bayesian",
            practical=-0.016667,
            best=0.125000,
            stability=-0.037500,
            significance=0.000000,
            composite=0.016250,
        ),
        make_entry(
            rank=3,
            name="combined",
            practical=-0.033333,
            best=0.108333,
            stability=-0.037500,
            significance=0.000000,
            composite=0.004583,
        ),
        make_entry(
            rank=4,
            name="baseline",
            practical=-0.033333,
            best=0.066667,
            stability=-0.162500,
            significance=0.000000,
            composite=-0.030833,
        ),
    )

    result = evaluate_champion_promotion(
        entries=entries,
        policy=default_policy(),
    )

    assert result.candidate == "calibration"
    assert result.promoted is False
    assert result.promoted_model is None

    assert result.composite_margin == pytest.approx(
        0.000417
    )

    assert (
        "significance_below_minimum"
        in result.rejection_reasons
    )

    assert (
        "composite_margin_below_minimum"
        in result.rejection_reasons
    )
