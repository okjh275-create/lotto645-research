"""CLI for planning and persisting adaptive rollbacks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from lrp.evolution.contracts import (
    AdaptiveWeightProfile,
)
from lrp.evolution.feedback import (
    AdaptiveAutomationRepository,
    AdaptiveRollbackManager,
    AdaptiveRollbackRepository,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adaptive-rollback",
        description=(
            "Plan or persist an adaptive profile rollback."
        ),
    )

    parser.add_argument(
        "--profile",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--repository",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--rollback-revision",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--rollback-id",
        required=True,
    )
    parser.add_argument(
        "--generated-at-utc",
        default=None,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    parser.add_argument(
        "--approve-rollback",
        action="store_true",
    )

    return parser


def run(
    argv: Sequence[str] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if (
            args.dry_run
            and args.approve_rollback
        ):
            raise ValueError(
                "--dry-run and --approve-rollback "
                "cannot be used together"
            )

        if (
            not args.dry_run
            and not args.approve_rollback
        ):
            raise ValueError(
                "rollback persistence requires "
                "explicit --approve-rollback"
            )

        profile = _load_profile(
            args.profile
        )
        timestamp = _optional_datetime(
            args.generated_at_utc
        )

        repository = (
            AdaptiveAutomationRepository(
                args.repository
            )
        )

        plan = AdaptiveRollbackManager(
            repository=repository
        ).plan(
            current_profile=profile,
            rollback_revision=(
                args.rollback_revision
            ),
            generated_at=timestamp,
        )

        if args.dry_run:
            payload = {
                "status": "PASS",
                "mode": "dry_run",
                "rollback_id": args.rollback_id,
                "created": False,
                "path": None,
                "plan": plan.as_dict(),
            }
        else:
            saved = AdaptiveRollbackRepository(
                repository=repository
            ).save(
                plan,
                rollback_id=args.rollback_id,
            )

            payload = {
                "status": "PASS",
                "mode": "persisted",
                "rollback_id": args.rollback_id,
                "created": saved.created,
                "path": str(saved.path),
                "plan": plan.as_dict(),
            }

    except (
        FileNotFoundError,
        FileExistsError,
        NotADirectoryError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        parser.exit(
            status=2,
            message=f"error: {exc}\n",
        )

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    return 0


def _load_profile(
    path: Path,
) -> AdaptiveWeightProfile:
    if not path.is_file():
        raise FileNotFoundError(path)

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(payload, dict):
        raise TypeError(
            "profile file must contain "
            "a JSON object"
        )

    nested = payload.get("profile")

    values = (
        nested
        if isinstance(nested, Mapping)
        else payload
    )

    generated_at_raw = values.get(
        "generated_at"
    )

    if not isinstance(
        generated_at_raw,
        str,
    ):
        raise TypeError(
            "profile generated_at must "
            "be a string"
        )

    generated_at = datetime.fromisoformat(
        generated_at_raw
    )

    if generated_at.tzinfo is None:
        raise ValueError(
            "profile generated_at must be "
            "timezone-aware"
        )

    return AdaptiveWeightProfile(
        hot_weight=_number(
            values,
            "hot_weight",
        ),
        cold_weight=_number(
            values,
            "cold_weight",
        ),
        gap_weight=_number(
            values,
            "gap_weight",
        ),
        trend_weight=_number(
            values,
            "trend_weight",
        ),
        transition_weight=_number(
            values,
            "transition_weight",
        ),
        learning_weight=_number(
            values,
            "learning_weight",
        ),
        adaptive_weight=_number(
            values,
            "adaptive_weight",
        ),
        confidence=_number(
            values,
            "confidence",
        ),
        sample_size=_integer(
            values,
            "sample_size",
        ),
        revision=_integer(
            values,
            "revision",
        ),
        generated_at=generated_at,
    )


def _optional_datetime(
    value: str | None,
) -> datetime | None:
    if value is None:
        return None

    timestamp = datetime.fromisoformat(
        value
    )

    if timestamp.tzinfo is None:
        raise ValueError(
            "generated_at_utc must be "
            "timezone-aware"
        )

    return timestamp


def _number(
    values: Mapping[str, Any],
    key: str,
) -> float:
    value = values.get(key)

    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, float),
        )
    ):
        raise TypeError(
            f"{key} must be numeric"
        )

    return float(value)


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


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
