from __future__ import annotations

import pytest

from lrp.evolution.contracts.bayesian import (
    BayesianPosterior,
)
from lrp.regimes.bayesian_state import (
    RegimeBayesianState,
)


def test_default_state_uses_uniform_priors() -> None:
    state = RegimeBayesianState.default()

    for posterior in state.posteriors.values():
        assert posterior.alpha == 1.0
        assert posterior.beta == 1.0
        assert posterior.mean == 0.5
        assert posterior.adaptive_signal == 0.0


def test_state_exposes_exact_regime_set() -> None:
    state = RegimeBayesianState.default()

    assert tuple(state.posteriors) == (
        "gap_recovery",
        "cluster_rotation",
        "high_band_expansion",
        "low_band_expansion",
    )


def test_from_posteriors_preserves_values() -> None:
    posteriors = {
        "gap_recovery": BayesianPosterior(
            alpha=3.0,
            beta=1.0,
        ),
        "cluster_rotation": BayesianPosterior(
            alpha=1.0,
            beta=2.0,
        ),
        "high_band_expansion": BayesianPosterior(
            alpha=4.0,
            beta=2.0,
        ),
        "low_band_expansion": BayesianPosterior(
            alpha=2.0,
            beta=5.0,
        ),
    }

    state = RegimeBayesianState.from_posteriors(
        posteriors
    )

    assert state.posteriors == posteriors


def test_signals_map_posterior_to_regime() -> None:
    state = RegimeBayesianState.from_posteriors(
        {
            "gap_recovery": BayesianPosterior(
                alpha=3.0,
                beta=1.0,
            ),
            "cluster_rotation": BayesianPosterior(
                alpha=1.0,
                beta=3.0,
            ),
            "high_band_expansion": BayesianPosterior(
                alpha=1.0,
                beta=1.0,
            ),
            "low_band_expansion": BayesianPosterior(
                alpha=2.0,
                beta=2.0,
            ),
        }
    )

    signals = state.to_signals()

    assert signals["gap_recovery"] == pytest.approx(
        0.5
    )
    assert signals["cluster_rotation"] == pytest.approx(
        -0.5
    )
    assert signals["high_band_expansion"] == pytest.approx(
        0.0
    )
    assert signals["low_band_expansion"] == pytest.approx(
        0.0
    )


def test_missing_regime_is_rejected() -> None:
    posteriors = {
        "gap_recovery": BayesianPosterior(
            alpha=1.0,
            beta=1.0,
        ),
        "cluster_rotation": BayesianPosterior(
            alpha=1.0,
            beta=1.0,
        ),
        "high_band_expansion": BayesianPosterior(
            alpha=1.0,
            beta=1.0,
        ),
    }

    with pytest.raises(
        ValueError,
        match="missing",
    ):
        RegimeBayesianState.from_posteriors(
            posteriors
        )


def test_unknown_regime_is_rejected() -> None:
    posteriors = {
        "gap_recovery": BayesianPosterior(
            alpha=1.0,
            beta=1.0,
        ),
        "cluster_rotation": BayesianPosterior(
            alpha=1.0,
            beta=1.0,
        ),
        "high_band_expansion": BayesianPosterior(
            alpha=1.0,
            beta=1.0,
        ),
        "low_band_expansion": BayesianPosterior(
            alpha=1.0,
            beta=1.0,
        ),
        "unknown": BayesianPosterior(
            alpha=1.0,
            beta=1.0,
        ),
    }

    with pytest.raises(
        ValueError,
        match="unknown",
    ):
        RegimeBayesianState.from_posteriors(
            posteriors
        )


def test_invalid_posterior_type_is_rejected() -> None:
    posteriors = {
        "gap_recovery": BayesianPosterior(
            alpha=1.0,
            beta=1.0,
        ),
        "cluster_rotation": BayesianPosterior(
            alpha=1.0,
            beta=1.0,
        ),
        "high_band_expansion": BayesianPosterior(
            alpha=1.0,
            beta=1.0,
        ),
        "low_band_expansion": object(),
    }

    with pytest.raises(
        TypeError,
        match="BayesianPosterior",
    ):
        RegimeBayesianState.from_posteriors(
            posteriors
        )