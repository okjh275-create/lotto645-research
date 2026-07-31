from __future__ import annotations

import pytest

from lrp.evolution.algorithms.bayesian import (
    BayesianPosteriorUpdater,
    BayesianSignalCalculator,
)
from lrp.evolution.contracts.bayesian import (
    BayesianComponentState,
    BayesianEvidence,
    BayesianPosterior,
    BayesianState,
)


COMPONENTS = BayesianState.COMPONENTS


def make_evidence(
    *,
    successes: int = 1,
    failures: int = 1,
) -> dict[str, BayesianEvidence]:
    return {
        name: BayesianEvidence(
            successes=successes,
            failures=failures,
        )
        for name in COMPONENTS
    }


def test_component_state_exposes_signal() -> None:
    component = BayesianComponentState(
        name="hot",
        posterior=BayesianPosterior(
            alpha=3.0,
            beta=1.0,
        ),
    )

    assert component.signal == pytest.approx(0.5)


def test_component_name_is_trimmed() -> None:
    component = BayesianComponentState(
        name=" hot ",
        posterior=BayesianPosterior(
            alpha=1.0,
            beta=1.0,
        ),
    )

    assert component.name == "hot"


def test_unknown_component_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="unsupported Bayesian component",
    ):
        BayesianComponentState(
            name="unknown",
            posterior=BayesianPosterior(
                alpha=1.0,
                beta=1.0,
            ),
        )


def test_invalid_component_posterior_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="BayesianPosterior",
    ):
        BayesianComponentState(
            name="hot",
            posterior=object(),  # type: ignore[arg-type]
        )


def test_default_state_contains_all_components() -> None:
    state = BayesianState.default()

    assert tuple(state.components) == COMPONENTS
    assert tuple(state.posteriors) == COMPONENTS
    assert tuple(state.to_signals()) == COMPONENTS


def test_default_state_is_neutral() -> None:
    state = BayesianState.default()

    assert state.to_signals() == pytest.approx(
        {
            name: 0.0
            for name in COMPONENTS
        }
    )


def test_state_uses_configured_prior() -> None:
    state = BayesianState.default(
        alpha=3.0,
        beta=1.0,
    )

    assert state.to_signals() == pytest.approx(
        {
            name: 0.5
            for name in COMPONENTS
        }
    )


def test_state_from_posteriors_preserves_values() -> None:
    posteriors = {
        name: BayesianPosterior(
            alpha=float(index + 2),
            beta=1.0,
        )
        for index, name in enumerate(COMPONENTS)
    }

    state = BayesianState.from_posteriors(
        posteriors
    )

    assert state.posteriors == posteriors


def test_state_rejects_missing_posterior() -> None:
    posteriors = {
        name: BayesianPosterior(
            alpha=1.0,
            beta=1.0,
        )
        for name in COMPONENTS
        if name != "adaptive"
    }

    with pytest.raises(
        ValueError,
        match="missing Bayesian components",
    ):
        BayesianState.from_posteriors(
            posteriors
        )


def test_state_rejects_unknown_posterior() -> None:
    posteriors = {
        name: BayesianPosterior(
            alpha=1.0,
            beta=1.0,
        )
        for name in COMPONENTS
    }
    posteriors["unknown"] = BayesianPosterior(
        alpha=1.0,
        beta=1.0,
    )

    with pytest.raises(
        ValueError,
        match="unknown Bayesian components",
    ):
        BayesianState.from_posteriors(
            posteriors
        )


def test_signal_calculator_creates_default_updater() -> None:
    calculator = BayesianSignalCalculator()

    assert isinstance(
        calculator.updater,
        BayesianPosteriorUpdater,
    )


def test_signal_calculator_preserves_custom_updater() -> None:
    updater = BayesianPosteriorUpdater(
        prior_alpha=2.0,
        prior_beta=3.0,
    )
    calculator = BayesianSignalCalculator(
        updater
    )

    assert calculator.updater is updater


def test_invalid_updater_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="BayesianPosteriorUpdater or None",
    ):
        BayesianSignalCalculator(
            object()  # type: ignore[arg-type]
        )


def test_initial_state_uses_updater_prior() -> None:
    calculator = BayesianSignalCalculator(
        BayesianPosteriorUpdater(
            prior_alpha=3.0,
            prior_beta=1.0,
        )
    )

    state = calculator.initial_state()

    assert state.to_signals() == pytest.approx(
        {
            name: 0.5
            for name in COMPONENTS
        }
    )


def test_update_applies_evidence_to_all_components() -> None:
    calculator = BayesianSignalCalculator()

    state = calculator.update(
        make_evidence(
            successes=3,
            failures=1,
        )
    )

    for posterior in state.posteriors.values():
        assert posterior.alpha == 4.0
        assert posterior.beta == 2.0


def test_update_uses_previous_state() -> None:
    calculator = BayesianSignalCalculator()
    previous = BayesianState.default(
        alpha=5.0,
        beta=3.0,
    )

    state = calculator.update(
        make_evidence(
            successes=2,
            failures=1,
        ),
        previous=previous,
    )

    for posterior in state.posteriors.values():
        assert posterior.alpha == 7.0
        assert posterior.beta == 4.0


def test_update_does_not_mutate_previous_state() -> None:
    calculator = BayesianSignalCalculator()
    previous = BayesianState.default(
        alpha=5.0,
        beta=3.0,
    )

    calculator.update(
        make_evidence(
            successes=2,
            failures=1,
        ),
        previous=previous,
    )

    for posterior in previous.posteriors.values():
        assert posterior.alpha == 5.0
        assert posterior.beta == 3.0


def test_components_update_independently() -> None:
    calculator = BayesianSignalCalculator()
    evidence = make_evidence(
        successes=0,
        failures=0,
    )
    evidence["hot"] = BayesianEvidence(
        successes=4,
        failures=0,
    )
    evidence["cold"] = BayesianEvidence(
        successes=0,
        failures=4,
    )

    state = calculator.update(evidence)
    signals = calculator.calculate(state)

    assert signals["hot"] > 0.0
    assert signals["cold"] < 0.0

    for name in COMPONENTS:
        if name not in {"hot", "cold"}:
            assert signals[name] == pytest.approx(0.0)


def test_calculate_returns_all_signals() -> None:
    calculator = BayesianSignalCalculator()
    state = calculator.update(
        make_evidence(
            successes=3,
            failures=1,
        )
    )

    signals = calculator.calculate(state)

    assert tuple(signals) == COMPONENTS

    for signal in signals.values():
        assert -1.0 <= signal <= 1.0


def test_calculate_rejects_invalid_state() -> None:
    calculator = BayesianSignalCalculator()

    with pytest.raises(
        TypeError,
        match="BayesianState",
    ):
        calculator.calculate(  # type: ignore[arg-type]
            object()
        )


def test_update_rejects_missing_evidence() -> None:
    evidence = make_evidence()
    del evidence["adaptive"]

    with pytest.raises(
        ValueError,
        match="missing Bayesian evidence",
    ):
        BayesianSignalCalculator().update(
            evidence
        )


def test_update_rejects_unknown_evidence() -> None:
    evidence = make_evidence()
    evidence["unknown"] = BayesianEvidence(
        successes=1,
        failures=0,
    )

    with pytest.raises(
        ValueError,
        match="unknown Bayesian evidence",
    ):
        BayesianSignalCalculator().update(
            evidence
        )


def test_update_rejects_invalid_component_evidence() -> None:
    evidence: dict[str, object] = make_evidence()
    evidence["hot"] = object()

    with pytest.raises(
        TypeError,
        match="evidence for 'hot'",
    ):
        BayesianSignalCalculator().update(
            evidence  # type: ignore[arg-type]
        )


def test_update_rejects_invalid_previous_state() -> None:
    with pytest.raises(
        TypeError,
        match="BayesianState or None",
    ):
        BayesianSignalCalculator().update(
            make_evidence(),
            previous=object(),  # type: ignore[arg-type]
        )


def test_public_api_exports_bayesian_types() -> None:
    calculator = BayesianSignalCalculator()
    state = BayesianState.default()

    assert isinstance(
        calculator.calculate(state),
        dict,
    )
