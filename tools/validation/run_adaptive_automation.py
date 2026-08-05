"""Command-line entry point for adaptive automation."""

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
    AdaptiveAutomationService,
    RevisionAwareAutomationRunner,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adaptive-automation",
        description=(
            "Run adaptive automation from a "
            "cross-window validation report."
        ),
    )

    parser.add_argument(
        "--report",
        required=True,
        type=Path,
        help="Cross-window validation report JSON.",
    )
    parser.add_argument(
        "--profile",
        required=True,
        type=Path,
        help="Current AdaptiveWeightProfile JSON.",
    )
    parser.add_argument(
        "--repository",
        required=True,
        type=Path,
        help="Adaptive automation repository root.",
    )
    parser.add_argument(
        "--policy",
        required=True,
        help="Policy name to analyze.",
    )
    parser.add_argument(
        "--recommendation-id",
        required=True,
        help="Unique recommendation identifier.",
    )
    parser.add_argument(
        "--created-at-utc",
        default=None,
        help=(
            "Optional timezone-aware ISO-8601 "
            "timestamp."
        ),
    )
    parser.add_argument(
        "--target-confidence",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--target-sample-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--require-existing-repository",
        action="store_true",
        help=(
            "Reject persisted execution when no "
            "profile revision exists."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Analyze and plan without writing "
            "repository records."
        ),
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        help=(
            "Explicitly approve repository writes. "
            "Required for persisted execution."
        ),
    )

    return parser


def run(
    argv: Sequence[str] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        report = _load_object(
            args.report
        )
        current_profile = _load_profile(
            args.profile
        )
        created_at = _optional_datetime(
            args.created_at_utc
        )

        if (
            args.dry_run
            and args.approve
        ):
            raise ValueError(
                "--dry-run and --approve "
                "cannot be used together"
            )

        if (
            not args.dry_run
            and not args.approve
        ):
            raise ValueError(
                "persisted execution requires "
                "explicit --approve"
            )

        if args.dry_run:
            payload = _run_dry(
                report=report,
                policy_name=args.policy,
                recommendation_id=(
                    args.recommendation_id
                ),
                current_profile=current_profile,
                created_at_utc=created_at,
                target_confidence=(
                    args.target_confidence
                ),
                target_sample_size=(
                    args.target_sample_size
                ),
                repository_path=args.repository,
            )
        else:
            payload = _run_persisted(
                report=report,
                policy_name=args.policy,
                recommendation_id=(
                    args.recommendation_id
                ),
                current_profile=current_profile,
                created_at_utc=created_at,
                target_confidence=(
                    args.target_confidence
                ),
                target_sample_size=(
                    args.target_sample_size
                ),
                repository_path=args.repository,
                require_existing_repository=(
                    args.require_existing_repository
                ),
            )
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


def _run_dry(
    *,
    report: Mapping[str, Any],
    policy_name: str,
    recommendation_id: str,
    current_profile: AdaptiveWeightProfile,
    created_at_utc: datetime | None,
    target_confidence: float | None,
    target_sample_size: int | None,
    repository_path: Path,
) -> dict[str, Any]:
    result = AdaptiveAutomationService().run(
        report=report,
        policy_name=policy_name,
        recommendation_id=recommendation_id,
        current_profile=current_profile,
        created_at_utc=created_at_utc,
        target_confidence=target_confidence,
        target_sample_size=target_sample_size,
    )

    return {
        "status": "PASS",
        "mode": "dry_run",
        "recommendation_id": (
            result.recommendation
            .recommendation_id
        ),
        "approved": (
            result.update_plan.approved
        ),
        "source_revision": (
            result.update_plan.source_revision
        ),
        "target_revision": (
            result.update_plan.target_revision
        ),
        "repository_revision_before": None,
        "repository_revision_after": None,
        "automation_created": False,
        "profile_created": False,
        "automation_path": None,
        "profile_path": None,
        "repository_path": str(
            repository_path
        ),
        "repository_exists": (
            repository_path.exists()
        ),
        "violations": list(
            result.safety_result.violations
        ),
        "decisions": [
            decision.as_dict()
            for decision
            in result.recommendation.decisions
        ],
        "planned_profile": (
            result.update_plan
            .as_dict()["profile"]
        ),
    }


def _run_persisted(
    *,
    report: Mapping[str, Any],
    policy_name: str,
    recommendation_id: str,
    current_profile: AdaptiveWeightProfile,
    created_at_utc: datetime | None,
    target_confidence: float | None,
    target_sample_size: int | None,
    repository_path: Path,
    require_existing_repository: bool,
) -> dict[str, Any]:
    repository = AdaptiveAutomationRepository(
        repository_path
    )

    result = RevisionAwareAutomationRunner(
        repository=repository,
        allow_empty_repository=(
            not require_existing_repository
        ),
    ).run(
        report=report,
        policy_name=policy_name,
        recommendation_id=recommendation_id,
        current_profile=current_profile,
        created_at_utc=created_at_utc,
        target_confidence=target_confidence,
        target_sample_size=target_sample_size,
    )

    return {
        "status": "PASS",
        "mode": "persisted",
        "recommendation_id": (
            result.automation_result
            .recommendation
            .recommendation_id
        ),
        "approved": (
            result.automation_result
            .update_plan
            .approved
        ),
        "source_revision": (
            result.automation_result
            .update_plan
            .source_revision
        ),
        "target_revision": (
            result.automation_result
            .update_plan
            .target_revision
        ),
        "repository_revision_before": (
            result.repository_revision_before
        ),
        "repository_revision_after": (
            result.repository_revision_after
        ),
        "automation_created": (
            result.save_result
            .automation_created
        ),
        "profile_created": (
            result.save_result
            .profile_created
        ),
        "automation_path": str(
            result.save_result
            .automation_path
        ),
        "profile_path": (
            str(result.save_result.profile_path)
            if result.save_result.profile_path
            is not None
            else None
        ),
        "repository_path": str(
            repository_path
        ),
        "repository_exists": (
            repository_path.exists()
        ),
        "violations": list(
            result.automation_result
            .safety_result
            .violations
        ),
        "decisions": [
            decision.as_dict()
            for decision
            in result.automation_result
            .recommendation.decisions
        ],
        "planned_profile": (
            result.automation_result
            .update_plan
            .as_dict()["profile"]
        ),
    }


def _load_object(
    path: Path,
) -> dict[str, Any]:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(payload, dict):
        raise TypeError(
            f"{path.name} must contain "
            "a JSON object"
        )

    return payload


def _load_profile(
    path: Path,
) -> AdaptiveWeightProfile:
    payload = _load_object(path)

    nested = payload.get("profile")

    profile = (
        nested
        if isinstance(nested, Mapping)
        else payload
    )

    generated_at_raw = profile.get(
        "generated_at",
        profile.get("generated_at_utc"),
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
            profile,
            "hot_weight",
        ),
        cold_weight=_number(
            profile,
            "cold_weight",
        ),
        gap_weight=_number(
            profile,
            "gap_weight",
        ),
        trend_weight=_number(
            profile,
            "trend_weight",
        ),
        transition_weight=_number(
            profile,
            "transition_weight",
        ),
        learning_weight=_number(
            profile,
            "learning_weight",
        ),
        adaptive_weight=_number(
            profile,
            "adaptive_weight",
        ),
        confidence=_number(
            profile,
            "confidence",
        ),
        sample_size=_integer(
            profile,
            "sample_size",
        ),
        revision=_integer(
            profile,
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
            "created_at_utc must be "
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
