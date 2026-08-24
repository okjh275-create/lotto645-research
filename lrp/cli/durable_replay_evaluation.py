from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from typing import Sequence

import re
from lrp.operations.durable_replay_execution import (
    DurableReplayExecutionRequest,
    DurableReplayExecutionService,
    DurableReplayExecutionSource,
)


from lrp.operations.durable_replay_artifact_discovery import DurableReplayArtifactSelector
from lrp.operations.durable_replay_composition import (
    DurableReplayCompositionRequest,
    DurableReplayCompositionService,
)
from lrp.operations import write_operation_artifact

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

def _parse_selector(value: str) -> DurableReplayArtifactSelector:
    fields = value.split("|")

    if not 2 <= len(fields) <= 5:
        raise argparse.ArgumentTypeError(
            "selector descriptor must contain 2 to 5 fields"
        )

    round_text = fields[0]

    if not round_text.isdecimal():
        raise argparse.ArgumentTypeError(
            "selector round_no must be an integer"
        )

    round_no = int(round_text)
    model_name = fields[1]
    regime_id = (
        fields[2]
        if len(fields) >= 3 and fields[2] != ""
        else None
    )
    strategy_name = (
        fields[3]
        if len(fields) >= 4 and fields[3] != ""
        else None
    )
    artifact_key = fields[4] if len(fields) >= 5 else None

    if artifact_key is not None:
        if not artifact_key:
            raise argparse.ArgumentTypeError(
                "artifact_key must not be empty"
            )
        if len(artifact_key) > 128:
            raise argparse.ArgumentTypeError(
                "artifact_key must be at most 128 characters"
            )
        if (
            re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._-]*",
                artifact_key,
            )
            is None
        ):
            raise argparse.ArgumentTypeError(
                "artifact_key contains invalid characters"
            )
        if artifact_key in {".", ".."}:
            raise argparse.ArgumentTypeError(
                "artifact_key must not be dot path"
            )

    return DurableReplayArtifactSelector(
        round_no=round_no,
        model_name=model_name,
        regime_id=regime_id,
        strategy_name=strategy_name,
        artifact_key=artifact_key,
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
        required=False,
        action="append",
        type=_parse_source,
    )

    parser.add_argument(
        "--baseline",
        required=False,
        action="append",
        type=_parse_source,
    )

    parser.add_argument("--artifact-root", required=False)
    parser.add_argument("--candidate-selector", action="append", type=_parse_selector, required=False)
    parser.add_argument("--baseline-selector", action="append", type=_parse_selector, required=False)
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output root for durable replay "
            "evaluation result artifact"
        ),
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    candidate_sources = tuple(args.candidate or ())
    baseline_sources = tuple(args.baseline or ())
    candidate_selectors = tuple(args.candidate_selector or ())
    baseline_selectors = tuple(args.baseline_selector or ())

    has_explicit_sources = bool(candidate_sources or baseline_sources)
    has_selectors = bool(candidate_selectors or baseline_selectors)
    has_artifact_root = args.artifact_root is not None

    if has_explicit_sources and (has_selectors or has_artifact_root):
        parser.error(
            "explicit source mode and selector mode cannot be mixed"
        )

    if has_selectors and not has_artifact_root:
        parser.error(
            "--artifact-root is required with selector mode"
        )

    if has_artifact_root and not has_selectors:
        parser.error(
            "--artifact-root requires selector mode"
        )

    if not has_explicit_sources and not has_selectors:
        parser.error(
            "one explicit source or selector source is required"
        )

    if has_selectors:
        request = DurableReplayCompositionRequest(
            artifact_root=args.artifact_root,
            history_path=args.history,
            window_name=args.window_name,
            start_round=args.start_round,
            end_round=args.end_round,
            candidate_selectors=candidate_selectors,
            baseline_selectors=baseline_selectors,
        )

        result = DurableReplayCompositionService().execute(
            request=request
        )
    else:
        request = DurableReplayExecutionRequest(
            history_path=args.history,
            window_name=args.window_name,
            start_round=args.start_round,
            end_round=args.end_round,
            candidate_sources=candidate_sources,
            baseline_sources=baseline_sources,
        )

        result = DurableReplayExecutionService().execute(
            request=request
        )

    payload = _result_to_dict(result)

    if args.output is not None:
        write_operation_artifact(
            payload,
            output_root=args.output,
            artifact_type="durable-replay-evaluations",
            round_no=args.end_round,
            filename="evaluation_result.json",
        )

    print(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    return 0
