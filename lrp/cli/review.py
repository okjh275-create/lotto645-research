"""Review command."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lrp.evolution.contracts.learning_context import (
    LearningContext,
)
from lrp.evolution.policies import (
    AdaptiveWeightPolicy,
)
from lrp.evolution.repositories.file_snapshot_repository import (
    FileSnapshotRepository,
)
from lrp.evolution.services.adaptive_pipeline import (
    AdaptiveEvolutionPipeline,
)
from lrp.evolution.services.coordinator import (
    EvolutionCoordinator,
)
from lrp.evolution.services.persistent_learning_runner import (
    PersistentLearningRunner,
)
from lrp.evolution.services.persistent_learning_service import (
    PersistentLearningService,
)
from lrp.evolution.services.review_learning_service import (
    ReviewLearningService,
)
from lrp.evolution.services.review_profile_evolution_service import (
    ReviewProfileEvolutionService,
)
from lrp.evolution.storage import (
    SnapshotRepository,
)
from lrp.operations import (
    review_prediction,
    write_operation_artifact,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lrp review"
    )
    parser.add_argument(
        "--prediction",
        required=True,
    )
    parser.add_argument(
        "--numbers",
        nargs=6,
        required=True,
        type=int,
    )
    parser.add_argument(
        "--bonus",
        type=int,
    )
    parser.add_argument(
        "--output",
        default="snapshots",
    )

    parser.add_argument(
        "--learn",
        action="store_true",
        help=(
            "Run review learning and adaptive "
            "profile evolution."
        ),
    )
    parser.add_argument(
        "--learning-snapshots",
        help=(
            "Learning snapshot directory. "
            "Default: <output>/learning"
        ),
    )
    parser.add_argument(
        "--profile-snapshots",
        help=(
            "Adaptive profile snapshot directory. "
            "Default: <output>/profiles"
        ),
    )
    parser.add_argument(
        "--learning-policy",
        help=(
            "Policy label stored in reward keys, "
            "for example thompson or ucb1."
        ),
    )
    parser.add_argument(
        "--learning-snapshot-id",
        help=(
            "Learning snapshot ID. "
            "Default: review-<round>"
        ),
    )
    parser.add_argument(
        "--overwrite-learning",
        action="store_true",
        help="Overwrite an existing learning snapshot.",
    )
    parser.add_argument(
        "--learning-confidence",
        type=float,
        default=0.80,
        help=(
            "Adaptive profile confidence from "
            "0.0 to 1.0. Default: 0.80"
        ),
    )

    return parser


def _resolve_learning_root(
    arguments: argparse.Namespace,
) -> Path:
    if arguments.learning_snapshots:
        return Path(
            arguments.learning_snapshots
        )

    return Path(arguments.output) / "learning"


def _resolve_profile_root(
    arguments: argparse.Namespace,
) -> Path:
    if arguments.profile_snapshots:
        return Path(
            arguments.profile_snapshots
        )

    return Path(arguments.output) / "profiles"


def _build_review_learning_service(
    root: Path,
) -> ReviewLearningService:
    persistence = PersistentLearningService(
        FileSnapshotRepository(root)
    )
    runner = PersistentLearningRunner(
        persistence
    )

    return ReviewLearningService(runner)


def _build_profile_evolution_service(
    root: Path,
) -> ReviewProfileEvolutionService:
    coordinator = EvolutionCoordinator(
        pipeline=AdaptiveEvolutionPipeline(),
        policy=AdaptiveWeightPolicy(),
        repository=SnapshotRepository(root),
    )

    return ReviewProfileEvolutionService(
        coordinator
    )


def _run_learning(
    *,
    payload: Mapping[str, Any],
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    round_no = int(payload["round"])

    snapshot_id = (
        arguments.learning_snapshot_id
        or f"review-{round_no}"
    )

    context = LearningContext(
        cycle_id=f"review-{round_no}",
        round_no=round_no,
    )

    learning_service = (
        _build_review_learning_service(
            _resolve_learning_root(arguments)
        )
    )

    learning = learning_service.learn(
        context=context,
        review_payload=payload,
        snapshot_id=snapshot_id,
        policy=arguments.learning_policy,
        metadata={
            "round": round_no,
            "prediction": arguments.prediction,
        },
        overwrite=(
            arguments.overwrite_learning
        ),
    )

    profile_service = (
        _build_profile_evolution_service(
            _resolve_profile_root(arguments)
        )
    )

    evolution = profile_service.evolve(
        context=learning.final_context,
        generated_at=datetime.now(
            timezone.utc
        ),
        confidence=(
            arguments.learning_confidence
        ),
    )

    profile = evolution.decision.profile

    return {
        "learning_snapshot_id": (
            learning.snapshot_id
        ),
        "feedback_count": (
            learning.feedback_count
        ),
        "step_count": learning.step_count,
        "final_context_version": (
            learning.final_context.version
        ),
        "profile_applied": (
            evolution.decision.applied
        ),
        "profile_revision": (
            profile.revision
            if profile is not None
            else None
        ),
        "profile_snapshot_saved": (
            evolution.snapshot is not None
        ),
        "profile_reasons": list(
            evolution.decision.reasons
        ),
        "learning_snapshot_root": str(
            _resolve_learning_root(arguments)
        ),
        "profile_snapshot_root": str(
            _resolve_profile_root(arguments)
        ),
    }


def main(
    argv: Sequence[str] | None = None,
) -> int:
    arguments = _parser().parse_args(argv)

    try:
        payload = review_prediction(
            arguments.prediction,
            winning_numbers=arguments.numbers,
            bonus=arguments.bonus,
        )

        artifact = write_operation_artifact(
            payload,
            output_root=arguments.output,
            artifact_type="reviews",
            round_no=int(payload["round"]),
            filename="review.json",
        )

        response: dict[str, Any] = {
            "status": "PASS",
            "summary": payload["summary"],
            "artifact": artifact,
        }

        if arguments.learn:
            response["learning"] = _run_learning(
                payload=payload,
                arguments=arguments,
            )

        print(
            json.dumps(
                response,
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "ERROR",
                    "error_type": (
                        type(exc).__name__
                    ),
                    "message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
