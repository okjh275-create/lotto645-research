"""Aggregate cross-window sign-test statistics."""

from __future__ import annotations

from collections.abc import Mapping
from math import comb
from typing import Any


class CrossWindowSignificanceAnalyzer:
    """Calculate exact two-sided sign tests from aggregated wins."""

    def analyze(
        self,
        report: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(
            report,
            Mapping,
        ):
            raise TypeError(
                "report must be a mapping"
            )

        policies = report.get(
            "policies"
        )

        if not isinstance(
            policies,
            Mapping,
        ):
            raise TypeError(
                "policies must be an object"
            )

        analyzed = {}

        for policy_name, values in sorted(
            policies.items()
        ):
            if not isinstance(
                policy_name,
                str,
            ):
                raise TypeError(
                    "policy names must be strings"
                )

            if not isinstance(
                values,
                Mapping,
            ):
                raise TypeError(
                    "policy values must be objects"
                )

            analyzed[policy_name] = {
                "best": self._outcome(
                    adaptive_wins=self._integer(
                        values,
                        "best_adaptive_wins",
                    ),
                    noop_wins=self._integer(
                        values,
                        "best_noop_wins",
                    ),
                    ties=self._integer(
                        values,
                        "best_ties",
                    ),
                ),
                "practical": self._outcome(
                    adaptive_wins=self._integer(
                        values,
                        "practical_adaptive_wins",
                    ),
                    noop_wins=self._integer(
                        values,
                        "practical_noop_wins",
                    ),
                    ties=self._integer(
                        values,
                        "practical_ties",
                    ),
                ),
            }

        return {
            "schema_version": 1,
            "policy_count": len(analyzed),
            "significance_threshold": 0.05,
            "policies": analyzed,
        }

    def _outcome(
        self,
        *,
        adaptive_wins: int,
        noop_wins: int,
        ties: int,
    ) -> dict[str, Any]:
        if (
            adaptive_wins < 0
            or noop_wins < 0
            or ties < 0
        ):
            raise ValueError(
                "win and tie counts must be "
                "greater than or equal to 0"
            )

        non_ties = (
            adaptive_wins + noop_wins
        )

        p_value = self._two_sided_sign_test(
            adaptive_wins=adaptive_wins,
            noop_wins=noop_wins,
        )

        if adaptive_wins > noop_wins:
            direction = "adaptive_better"
        elif adaptive_wins < noop_wins:
            direction = "noop_better"
        else:
            direction = "tie"

        return {
            "adaptive_wins": adaptive_wins,
            "noop_wins": noop_wins,
            "ties": ties,
            "non_tie_count": non_ties,
            "direction": direction,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }

    @staticmethod
    def _two_sided_sign_test(
        *,
        adaptive_wins: int,
        noop_wins: int,
    ) -> float:
        sample_size = (
            adaptive_wins + noop_wins
        )

        if sample_size == 0:
            return 1.0

        smaller = min(
            adaptive_wins,
            noop_wins,
        )

        tail = sum(
            comb(sample_size, count)
            for count in range(
                smaller + 1
            )
        ) / (2 ** sample_size)

        return min(
            1.0,
            2.0 * tail,
        )

    @staticmethod
    def _integer(
        values: Mapping[str, Any],
        key: str,
    ) -> int:
        value = values.get(key)

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise TypeError(
                f"{key} must be an integer"
            )

        return value
