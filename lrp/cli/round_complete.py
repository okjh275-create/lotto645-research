"""Round-completion command."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Sequence

from lrp import __version__
from lrp.evolution.policies import AdaptiveWeightPolicy
from lrp.evolution.repositories.file_snapshot_repository import (
    FileSnapshotRepository,
)
from lrp.evolution.services.adaptive_pipeline import (
    AdaptiveEvolutionPipeline,
)
from lrp.evolution.services.coordinator import EvolutionCoordinator
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
from lrp.evolution.storage import SnapshotRepository
from lrp.learning import LearningRepository
from lrp.operations import verify_manifest, write_operation_artifact
from lrp.outcomes import OutcomeBridge, OutcomeLearningBridge
from lrp.pipelines.round_completion import RoundCompletionPipeline


_KST = ZoneInfo("Asia/Seoul")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lrp round-complete"
    )

    parser.add_argument("--prediction", required=True)
    parser.add_argument(
        "--numbers",
        nargs=6,
        type=int,
        required=True,
    )
    parser.add_argument(
        "--bonus",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--output",
        default="snapshots",
    )
    parser.add_argument("--policy")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.80,
    )
    parser.add_argument("--snapshot-id")
    parser.add_argument(
        "--overwrite-learning",
        action="store_true",
    )
    parser.add_argument(
        "--model-name",
        default=f"lrp-v{__version__}",
    )

    return parser


def _build_pipeline(
    *,
    output: Path,
    model_name: str,
) -> RoundCompletionPipeline:
    learning_root = output / "learning"
    profile_root = output / "profiles"

    repository = LearningRepository(
        learning_root / "learning.db"
    )

    outcome_bridge = OutcomeBridge(
        repository=repository,
        model_name=model_name,
    )

    persistence = PersistentLearningService(
        FileSnapshotRepository(learning_root)
    )

    learning_bridge = OutcomeLearningBridge(
        service=ReviewLearningService(
            PersistentLearningRunner(
                persistence
            )
        )
    )

    profile_service = ReviewProfileEvolutionService(
        EvolutionCoordinator(
            pipeline=AdaptiveEvolutionPipeline(),
            policy=AdaptiveWeightPolicy(),
            repository=SnapshotRepository(
                profile_root
            ),
        )
    )

    return RoundCompletionPipeline(
        outcome_bridge=outcome_bridge,
        learning_bridge=learning_bridge,
        profile_service=profile_service,
    )


def main(
    argv: Sequence[str] | None = None,
) -> int:
    arguments = _parser().parse_args(argv)

    try:
        output = Path(arguments.output)

        pipeline = _build_pipeline(
            output=output,
            model_name=arguments.model_name,
        )

        result = pipeline.run(
            arguments.prediction,
            winning_numbers=tuple(arguments.numbers),
            bonus=arguments.bonus,
            snapshot_id=arguments.snapshot_id,
            policy=arguments.policy,
            confidence=arguments.confidence,
            recorded_at_kst=(
                "round-completion"
            ),
            overwrite_learning=(
                arguments.overwrite_learning
            ),
        )

        report_payload = {
            **result.as_dict(),
            "platform_version": __version__,
            "completed_at_kst": datetime.now(
                _KST
            ).isoformat(
                timespec="seconds"
            ),
        }

        artifact = write_operation_artifact(
            report_payload,
            output_root=output,
            artifact_type="round-completion",
            round_no=result.round_no,
            filename="round_completion.json",
        )

        verification = verify_manifest(
            artifact["manifest_path"]
        )

        if verification["status"] != "PASS":
            raise RuntimeError(
                "round completion artifact "
                "verification failed"
            )

        response = {
            "schema_version": "1.0",
            "status": "PASS",
            **report_payload,
            "artifact": artifact,
            "verification": verification,
            "warnings": [],
        }

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
                    "schema_version": "1.0",
                    "status": "ERROR",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "warnings": [],
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
