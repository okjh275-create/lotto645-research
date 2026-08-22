from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from typing import Sequence

from lrp.operations.durable_replay_execution import (
    DurableReplayExecutionRequest,
    DurableReplayExecutionService,
    DurableReplayExecutionSource,
)


def _parse_source(
    value: str,
) -> DurableReplayExecutionSource:
    fields = value.split("|")

    if not 3 <= len(fields) <= 5:
        raise argparse.ArgumentTypeError(
            "source descriptor must contain 3 to 5 fields"
        )

    artifact_path = fields[0]

    round_text = fields[1]

    if not round_text.isdecimal():
        raise argparse.ArgumentTypeError(
            "source round_no must be an integer"
        )

    round_no = int(round_text)

    model_name = fields[2]

    regime_id = (
        fields[3]
        if len(fields) >= 4 and fields[3] != ""
        else None
    )

    strategy_name = (
        fields[4]
        if len(fields) >= 5 and fields[4] != ""
        else None
    )

    return DurableReplayExecutionSource(
        artifact_path=artifact_path,
        round_no=round_no,
        model_name=model_name,
        regime_id=regime_id,
        strategy_name=strategy_name,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="durable-replay-evaluation"
    )

    parser.add_argument(
        "--history",
        required=True,
    )

    parser.add_argument(
        "--window-name",
        required=True,
    )

    parser.add_argument(
        "--start-round",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--end-round",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--candidate",
        required=True,
        action="append",
        type=_parse_source,
    )

    parser.add_argument(
        "--baseline",
        required=True,
        action="append",
        type=_parse_source,
    )

    return parser


def _json_compatible(
    value: object,
) -> object:
    if is_dataclass(value):
        return _json_compatible(
            asdict(value)
        )

    if isinstance(value, dict):
        return {
            str(key): _json_compatible(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (tuple, list),
    ):
        return [
            _json_compatible(item)
            for item in value
        ]

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ) or value is None:
        return value

    if hasattr(
        value,
        "__dict__",
    ):
        return {
            str(key): _json_compatible(item)
            for key, item in vars(value).items()
        }

    return str(value)


def _result_to_dict(
    result: object,
) -> dict[str, object]:
    return {
        "status": "PASS",
        "candidate_model_name": getattr(
            result,
            "candidate_model_name",
        ),
        "baseline_model_name": getattr(
            result,
            "baseline_model_name",
        ),
        "round_count": getattr(
            result,
            "round_count",
        ),
        "evaluation": _json_compatible(
            getattr(
                result,
                "evaluation",
            )
        ),
    }


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = _parser().parse_args(argv)

    request = DurableReplayExecutionRequest(
        history_path=args.history,
        window_name=args.window_name,
        start_round=args.start_round,
        end_round=args.end_round,
        candidate_sources=tuple(
            args.candidate
        ),
        baseline_sources=tuple(
            args.baseline
        ),
    )

    result = (
        DurableReplayExecutionService()
        .execute(
            request=request
        )
    )

    payload = _result_to_dict(
        result
    )

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    return 0
