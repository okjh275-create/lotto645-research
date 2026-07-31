from __future__ import annotations

import pytest

from lrp.evolution.algorithms.bayesian import (
    BayesianPosteriorUpdater,
)
from lrp.evolution.contracts.bayesian import (
    BayesianEvidence,
    BayesianPosterior,
)


def test_evidence_exposes_observation_count() -> None:
    evidence = BayesianEvidence(
        successes=7,
        failures=3,
    )

    assert evidence.observations == 10


def test_evidence_exposes_success_rate() -> None:
    evidence = BayesianEvidence(
        successes=7,
        failures=3,
    )

    assert evidence.success_rate == pytest.approx(0.7)


def test_empty_evidence_has_zero_success_rate() -> None:
    evidence = BayesianEvidence(
        successes=0,
        failures=0,
    )

    assert evidence.observations == 0
    assert evidence.success_rate == 0.0


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    [
        ("successes", True, TypeError),
        ("successes", 1.5, TypeError),
        ("successes", -1, ValueError),
        ("failures", False, TypeError),
        ("failures", "1", TypeError),
        ("failures", -1, ValueError),
    ],
)
def test_invalid_evidence_is_rejected(
    field_name: str,
    value: object,
    error_type: type[Exception],
) -> None:
    values: dict[str, object] = {
        "successes": 1,
        "failures": 1,
    }
    values[field_name] = value

    with pytest.raises(error_type):
        BayesianEvidence(
            successes=values["successes"],  # type: ignore[arg-type]
            failures=values["failures"],  # type: ignore[arg-type]
        )


def test_posterior_exposes_distribution_metrics() -> None:
    posterior = BayesianPosterior(
        alpha=3.0,
        beta=2.0,
    )

    assert posterior.concentration == 5.0
    assert posterior.mean == pytest.approx(0.6)
    assert posterior.variance == pytest.approx(0.04)
    assert posterior.adaptive_signal == pytest.approx(0.2)


def test_balanced_posterior_has_neutral_signal() -> None:
    posterior = BayesianPosterior(
        alpha=2.0,
        beta=2.0,
    )

    assert posterior.mean == pytest.approx(0.5)
    assert posterior.adaptive_signal == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    [
        ("alpha", True, TypeError),
        ("alpha", "1.0", TypeError),
        ("alpha", 0.0, ValueError),
        ("alpha", -1.0, ValueError),
        ("alpha", float("inf"), ValueError),
        ("beta", False, TypeError),
        ("beta", None, TypeError),
        ("beta", 0.0, ValueError),
        ("beta", -1.0, ValueError),
        ("beta", float("nan"), ValueError),
    ],
)
def test_invalid_posterior_is_rejected(
    field_name: str,
    value: object,
    error_type: type[Exception],
) -> None:
    values: dict[str, object] = {
        "alpha": 1.0,
        "beta": 1.0,
    }
    values[field_name] = value

    with pytest.raises(error_type):
        BayesianPosterior(
            alpha=values["alpha"],  # type: ignore[arg-type]
            beta=values["beta"],  # type: ignore[arg-type]
        )


def test_posterior_update_adds_evidence() -> None:
    posterior = BayesianPosterior(
        alpha=2.0,
        beta=3.0,
    )

    updated = posterior.updated(
        BayesianEvidence(
            successes=4,
            failures=1,
        )
    )

    assert updated.alpha == 6.0
    assert updated.beta == 4.0
    assert updated.mean == pytest.approx(0.6)


def test_posterior_is_immutable_during_update() -> None:
    posterior = BayesianPosterior(
        alpha=2.0,
        beta=3.0,
    )

    posterior.updated(
        BayesianEvidence(
            successes=4,
            failures=1,
        )
    )

    assert posterior.alpha == 2.0
    assert posterior.beta == 3.0


def test_posterior_rejects_invalid_evidence() -> None:
    posterior = BayesianPosterior(
        alpha=1.0,
        beta=1.0,
    )

    with pytest.raises(
        TypeError,
        match="BayesianEvidence",
    ):
        posterior.updated(  # type: ignore[arg-type]
            object()
        )


def test_updater_uses_default_uniform_prior() -> None:
    updater = BayesianPosteriorUpdater()

    posterior = updater.update(
        BayesianEvidence(
            successes=3,
            failures=1,
        )
    )

    assert posterior.alpha == 4.0
    assert posterior.beta == 2.0


def test_updater_uses_configured_prior() -> None:
    updater = BayesianPosteriorUpdater(
        prior_alpha=2.5,
        prior_beta=1.5,
    )

    posterior = updater.update(
        BayesianEvidence(
            successes=2,
            failures=3,
        )
    )

    assert updater.prior_alpha == 2.5
    assert updater.prior_beta == 1.5
    assert posterior.alpha == 4.5
    assert posterior.beta == 4.5


def test_updater_uses_previous_posterior() -> None:
    updater = BayesianPosteriorUpdater(
        prior_alpha=1.0,
        prior_beta=1.0,
    )
    previous = BayesianPosterior(
        alpha=8.0,
        beta=4.0,
    )

    posterior = updater.update(
        BayesianEvidence(
            successes=2,
            failures=1,
        ),
        previous=previous,
    )

    assert posterior.alpha == 10.0
    assert posterior.beta == 5.0


def test_zero_evidence_preserves_previous_posterior() -> None:
    updater = BayesianPosteriorUpdater()
    previous = BayesianPosterior(
        alpha=8.0,
        beta=4.0,
    )

    posterior = updater.update(
        BayesianEvidence(
            successes=0,
            failures=0,
        ),
        previous=previous,
    )

    assert posterior == previous
    assert posterior is not previous


def test_updater_rejects_invalid_evidence() -> None:
    updater = BayesianPosteriorUpdater()

    with pytest.raises(
        TypeError,
        match="BayesianEvidence",
    ):
        updater.update(  # type: ignore[arg-type]
            object()
        )


def test_updater_rejects_invalid_previous() -> None:
    updater = BayesianPosteriorUpdater()

    with pytest.raises(
        TypeError,
        match="BayesianPosterior or None",
    ):
        updater.update(
            BayesianEvidence(
                successes=1,
                failures=0,
            ),
            previous=object(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    [
        ("prior_alpha", True, TypeError),
        ("prior_alpha", "1.0", TypeError),
        ("prior_alpha", 0.0, ValueError),
        ("prior_alpha", -1.0, ValueError),
        ("prior_alpha", float("inf"), ValueError),
        ("prior_beta", False, TypeError),
        ("prior_beta", None, TypeError),
        ("prior_beta", 0.0, ValueError),
        ("prior_beta", -1.0, ValueError),
        ("prior_beta", float("nan"), ValueError),
    ],
)
def test_invalid_updater_prior_is_rejected(
    field_name: str,
    value: object,
    error_type: type[Exception],
) -> None:
    values: dict[str, object] = {
        "prior_alpha": 1.0,
        "prior_beta": 1.0,
    }
    values[field_name] = value

    with pytest.raises(error_type):
        BayesianPosteriorUpdater(
            prior_alpha=values["prior_alpha"],  # type: ignore[arg-type]
            prior_beta=values["prior_beta"],  # type: ignore[arg-type]
        )
