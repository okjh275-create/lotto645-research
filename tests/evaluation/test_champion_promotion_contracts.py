from __future__ import annotations

from dataclasses import replace
from math import inf, nan

import pytest

from lrp.contracts import ContractError
from lrp.evaluation import (
    ChampionPromotionDecision,
    ChampionPromotionPolicy,
    ModelRankingEntry,
    evaluate_champion_promotion,
)


def make_entry(
    *,
    rank: int,
    name: str,
    practical: float = 0.10,
    best: float = 0.10,
    stability: float = 0.10,
    significance: float = 1.0,
    composite: float = 0.10,
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


def test_default_policy_contract() -> None:
    policy = ChampionPromotionPolicy()

    assert policy.minimum_composite_score == 0.0
    assert policy.minimum_practical_score == 0.0
    assert policy.minimum_significance_score == 1.0
    assert policy.minimum_composite_margin == 0.01


@pytest.mark.parametrize(
    "field_name",
    (
        "minimum_composite_score",
        "minimum_practical_score",
        "minimum_significance_score",
        "minimum_composite_margin",
    ),
)
@pytest.mark.parametrize(
    "value",
    (
        nan,
        inf,
        -inf,
        True,
        False,
    ),
)
def test_policy_rejects_nonfinite_or_boolean_thresholds(
    field_name: str,
    value: float | bool,
) -> None:
    policy = ChampionPromotionPolicy()

    with pytest.raises(ContractError):
        replace(
            policy,
            **{
                field_name: value,
            },
        )


def test_policy_allows_negative_score_thresholds() -> None:
    policy = ChampionPromotionPolicy(
        minimum_composite_score=-0.10,
        minimum_practical_score=-0.10,
    )

    assert policy.minimum_composite_score == -0.10
    assert policy.minimum_practical_score == -0.10


def test_empty_entries_are_rejected() -> None:
    with pytest.raises(
        ContractError,
        match="entries must not be empty",
    ):
        evaluate_champion_promotion(
            entries=(),
        )


def test_non_entry_value_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="ModelRankingEntry",
    ):
        evaluate_champion_promotion(
            entries=(object(),),
        )


@pytest.mark.parametrize(
    "entries",
    (
        (
            make_entry(
                rank=2,
                name="second",
            ),
            make_entry(
                rank=1,
                name="first",
            ),
        ),
        (
            make_entry(
                rank=1,
                name="first",
            ),
            make_entry(
                rank=3,
                name="third",
            ),
        ),
    ),
)
def test_entries_must_be_contiguous_rank_order(
    entries: tuple[ModelRankingEntry, ...],
) -> None:
    with pytest.raises(
        ContractError,
        match="contiguous rank",
    ):
        evaluate_champion_promotion(
            entries=entries,
        )


def test_wrong_policy_type_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="ChampionPromotionPolicy",
    ):
        evaluate_champion_promotion(
            entries=(
                make_entry(
                    rank=1,
                    name="candidate",
                ),
            ),
            policy=object(),
        )


def test_zero_composite_is_blocked_even_when_threshold_zero() -> None:
    decision = evaluate_champion_promotion(
        entries=(
            make_entry(
                rank=1,
                name="candidate",
                practical=0.10,
                significance=1.0,
                composite=0.0,
            ),
        ),
        policy=ChampionPromotionPolicy(
            minimum_composite_score=0.0,
            minimum_practical_score=0.0,
            minimum_significance_score=1.0,
            minimum_composite_margin=0.0,
        ),
    )

    assert decision.promoted is False
    assert decision.rejection_reasons == (
        "composite_score_not_positive",
    )


def test_exact_thresholds_are_accepted() -> None:
    decision = evaluate_champion_promotion(
        entries=(
            make_entry(
                rank=1,
                name="candidate",
                practical=0.0,
                significance=1.0,
                composite=0.10,
            ),
            make_entry(
                rank=2,
                name="runner_up",
                practical=0.0,
                significance=1.0,
                composite=0.09,
            ),
        ),
        policy=ChampionPromotionPolicy(
            minimum_composite_score=0.10,
            minimum_practical_score=0.0,
            minimum_significance_score=1.0,
            minimum_composite_margin=0.01,
        ),
    )

    assert decision.promoted is True
    assert decision.promoted_model == "candidate"
    assert decision.composite_margin == pytest.approx(
        0.01
    )


def test_margin_uses_next_eligible_model() -> None:
    decision = evaluate_champion_promotion(
        entries=(
            make_entry(
                rank=1,
                name="candidate",
                composite=0.10,
            ),
            make_entry(
                rank=2,
                name="excluded",
                composite=0.099,
                eligible=False,
            ),
            make_entry(
                rank=3,
                name="runner_up",
                composite=0.08,
            ),
        ),
        policy=ChampionPromotionPolicy(
            minimum_composite_margin=0.01,
        ),
    )

    assert decision.promoted is True
    assert decision.composite_margin == pytest.approx(
        0.02
    )


def test_decision_serialization_is_json_ready() -> None:
    decision = evaluate_champion_promotion(
        entries=(
            make_entry(
                rank=1,
                name="candidate",
            ),
        ),
    )

    payload = decision.as_dict()

    assert payload == {
        "candidate": "candidate",
        "promoted": True,
        "promoted_model": "candidate",
        "composite_margin": None,
        "rejection_reasons": [],
    }


def test_rejected_decision_serialization() -> None:
    decision = evaluate_champion_promotion(
        entries=(
            make_entry(
                rank=1,
                name="candidate",
                significance=0.0,
            ),
        ),
    )

    payload = decision.as_dict()

    assert payload["candidate"] == "candidate"
    assert payload["promoted"] is False
    assert payload["promoted_model"] is None
    assert payload["composite_margin"] is None
    assert payload["rejection_reasons"] == [
        "significance_below_minimum"
    ]


def test_decision_rejects_promoted_without_candidate() -> None:
    with pytest.raises(ContractError):
        ChampionPromotionDecision(
            candidate=None,
            promoted=True,
            promoted_model=None,
            composite_margin=None,
            rejection_reasons=(),
        )


def test_decision_rejects_promoted_model_mismatch() -> None:
    with pytest.raises(ContractError):
        ChampionPromotionDecision(
            candidate="candidate",
            promoted=True,
            promoted_model="other",
            composite_margin=0.02,
            rejection_reasons=(),
        )


def test_decision_rejects_rejected_with_promoted_model() -> None:
    with pytest.raises(ContractError):
        ChampionPromotionDecision(
            candidate="candidate",
            promoted=False,
            promoted_model="candidate",
            composite_margin=0.02,
            rejection_reasons=(
                "blocked",
            ),
        )


def test_decision_rejects_rejection_without_reason() -> None:
    with pytest.raises(ContractError):
        ChampionPromotionDecision(
            candidate="candidate",
            promoted=False,
            promoted_model=None,
            composite_margin=0.02,
            rejection_reasons=(),
        )
