from __future__ import annotations

import json
from typing import Any

from lrp.contracts.exceptions import ContractError
from lrp.evaluation.topk_live_evaluation_runtime import (
    TopKLiveEvaluationRuntimeResult,
)
from lrp.evaluation.topk_live_evaluation_source_snapshot import (
    snapshot_to_dict,
)


_SCHEMA_VERSION = "1.0"


def runtime_result_to_dict(
    result: TopKLiveEvaluationRuntimeResult,
) -> dict[str, Any]:
    if not isinstance(
        result,
        TopKLiveEvaluationRuntimeResult,
    ):
        raise ContractError(
            "result must be "
            "TopKLiveEvaluationRuntimeResult"
        )

    try:
        candidate_snapshot = (
            result.source_pair.candidate
        )

        baseline_snapshot = (
            result.source_pair.baseline
        )

        replay_evaluation = (
            result.evaluation.evaluation
        )

        candidate_round = (
            candidate_snapshot.round_no
        )

        baseline_round = (
            baseline_snapshot.round_no
        )

        candidate_model_name = (
            replay_evaluation.candidate_model_name
        )

        baseline_model_name = (
            replay_evaluation.baseline_model_name
        )

        round_count = (
            replay_evaluation.round_count
        )

    except (
        AttributeError,
        TypeError,
    ) as exc:
        raise ContractError(
            "result contains malformed "
            "runtime evaluation structure"
        ) from exc

    if candidate_round != baseline_round:
        raise ContractError(
            "candidate and baseline snapshot "
            "rounds must match"
        )

    if (
        candidate_model_name
        != candidate_snapshot.model_name
    ):
        raise ContractError(
            "candidate evaluation model identity "
            "must match candidate snapshot"
        )

    if (
        baseline_model_name
        != baseline_snapshot.model_name
    ):
        raise ContractError(
            "baseline evaluation model identity "
            "must match baseline snapshot"
        )

    try:
        candidate_payload = snapshot_to_dict(
            candidate_snapshot
        )

        baseline_payload = snapshot_to_dict(
            baseline_snapshot
        )

        walk_forward_payload = (
            replay_evaluation
            .evaluation
            .as_dict()
        )

    except ContractError:
        raise

    except (
        AttributeError,
        TypeError,
    ) as exc:
        raise ContractError(
            "result contains malformed "
            "runtime serialization structure"
        ) from exc

    evaluation_payload = {
        "candidate_model_name": (
            candidate_model_name
        ),
        "baseline_model_name": (
            baseline_model_name
        ),
        "round_count": (
            round_count
        ),
        "walk_forward": (
            walk_forward_payload
        ),
    }

    return {
        "schema_version": _SCHEMA_VERSION,
        "round": candidate_round,
        "candidate": candidate_payload,
        "baseline": baseline_payload,
        "evaluation": evaluation_payload,
    }


def runtime_result_to_json(
    result: TopKLiveEvaluationRuntimeResult,
) -> str:
    return json.dumps(
        runtime_result_to_dict(
            result
        ),
        ensure_ascii=False,
        separators=(
            ",",
            ":",
        ),
    )