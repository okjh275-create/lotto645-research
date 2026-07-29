"""Human- and machine-readable E-003 score explanations."""

from __future__ import annotations

from typing import Any

from .rescoring import (
    RescoredCandidate,
    RescoringResult,
)


def explain_candidate(
    item: RescoredCandidate,
) -> dict[str, Any]:
    contributions = (
        item.contributions.to_dict()
    )

    return {
        "base_score": round(
            item.base_score,
            9,
        ),
        "ensemble_score": round(
            item.ensemble_score,
            9,
        ),
        "rank_before": item.rank_before,
        "rank_after": item.rank_after,
        "rank_change": item.rank_change,
        "strategy_evidence": [
            {
                "strategy_type": (
                    vector.strategy_type
                ),
                "strategy_name": (
                    vector.strategy_name
                ),
                "evidence_count": (
                    vector.evidence_count
                ),
            }
            for vector in item.feature_vectors
        ],
        "contributions": {
            name: round(value, 9)
            for name, value
            in contributions.items()
        },
    }


def explain_result(
    result: RescoringResult,
) -> dict[str, Any]:
    return {
        "round_no": result.round_no,
        "snapshot_revision": list(
            result.snapshot_revision
        ),
        "candidate_count": result.count,
        "changed_rank_count": (
            result.changed_rank_count
        ),
        "items": [
            explain_candidate(item)
            for item in result.items
        ],
        "metadata": dict(result.metadata),
    }
