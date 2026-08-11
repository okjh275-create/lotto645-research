from __future__ import annotations

from lrp.evolution.contracts.bayesian import (
    BayesianEvidence,
)
from lrp.regimes.reward import RegimeReward


class RegimeBayesianEvidenceAdapter:
    """Convert continuous regime rewards into discrete Bayesian evidence."""

    def __init__(
        self,
        *,
        evidence_scale: int = 10,
    ) -> None:
        if isinstance(evidence_scale, bool):
            raise TypeError(
                "evidence_scale must be an integer"
            )

        if not isinstance(evidence_scale, int):
            raise TypeError(
                "evidence_scale must be an integer"
            )

        if evidence_scale < 1:
            raise ValueError(
                "evidence_scale must be greater than or equal to 1"
            )

        self._evidence_scale = evidence_scale

    @property
    def evidence_scale(self) -> int:
        return self._evidence_scale

    def convert(
        self,
        reward: RegimeReward,
    ) -> BayesianEvidence:
        if not isinstance(
            reward,
            RegimeReward,
        ):
            raise TypeError(
                "reward must be a RegimeReward"
            )

        effective_reward = reward.effective_reward

        if effective_reward == 0.0:
            return BayesianEvidence(
                successes=0,
                failures=0,
            )

        magnitude = int(
            round(
                abs(effective_reward)
                * self.evidence_scale
            )
        )

        magnitude = max(
            1,
            magnitude,
        )

        if effective_reward > 0.0:
            return BayesianEvidence(
                successes=magnitude,
                failures=0,
            )

        return BayesianEvidence(
            successes=0,
            failures=magnitude,
        )