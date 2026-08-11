from __future__ import annotations

from lrp.evolution.algorithms.bayesian import (
    BayesianPosteriorUpdater,
)
from lrp.regimes.bayesian_evidence import (
    RegimeBayesianEvidenceAdapter,
)
from lrp.regimes.bayesian_state import (
    RegimeBayesianState,
)
from lrp.regimes.reward import RegimeReward


class RegimeBayesianUpdater:
    """Update one global-regime Bayesian posterior from regime reward."""

    def __init__(
        self,
        *,
        evidence_adapter: (
            RegimeBayesianEvidenceAdapter | None
        ) = None,
        posterior_updater: (
            BayesianPosteriorUpdater | None
        ) = None,
    ) -> None:
        if (
            evidence_adapter is not None
            and not isinstance(
                evidence_adapter,
                RegimeBayesianEvidenceAdapter,
            )
        ):
            raise TypeError(
                "evidence_adapter must be a "
                "RegimeBayesianEvidenceAdapter or None"
            )

        if (
            posterior_updater is not None
            and not isinstance(
                posterior_updater,
                BayesianPosteriorUpdater,
            )
        ):
            raise TypeError(
                "posterior_updater must be a "
                "BayesianPosteriorUpdater or None"
            )

        self._evidence_adapter = (
            evidence_adapter
            if evidence_adapter is not None
            else RegimeBayesianEvidenceAdapter()
        )
        self._posterior_updater = (
            posterior_updater
            if posterior_updater is not None
            else BayesianPosteriorUpdater()
        )

    @property
    def evidence_adapter(
        self,
    ) -> RegimeBayesianEvidenceAdapter:
        return self._evidence_adapter

    @property
    def posterior_updater(
        self,
    ) -> BayesianPosteriorUpdater:
        return self._posterior_updater

    def update(
        self,
        state: RegimeBayesianState,
        reward: RegimeReward,
    ) -> RegimeBayesianState:
        if not isinstance(
            state,
            RegimeBayesianState,
        ):
            raise TypeError(
                "state must be a RegimeBayesianState"
            )

        if not isinstance(
            reward,
            RegimeReward,
        ):
            raise TypeError(
                "reward must be a RegimeReward"
            )

        regime = reward.regime

        if regime not in RegimeBayesianState.REGIMES:
            return RegimeBayesianState.from_posteriors(
                state.posteriors
            )

        evidence = (
            self.evidence_adapter.convert(
                reward
            )
        )

        posteriors = dict(
            state.posteriors
        )

        posteriors[regime] = (
            self.posterior_updater.update(
                evidence,
                previous=posteriors[regime],
            )
        )

        return RegimeBayesianState.from_posteriors(
            posteriors
        )