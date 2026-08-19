"""Read-only production champion history retention planning."""

from __future__ import annotations

from lrp.production.production_registry_lock import ProductionRegistryWriterLock

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from lrp.production.champion_rollback_history import (
    ChampionRollbackHistoryReader,
    ChampionRollbackTarget,
)


@dataclass(frozen=True)
class ChampionHistoryRetentionPolicy:
    """Retention policy for verified publication history."""

    keep_recent: int

    def __post_init__(
        self,
    ) -> None:
        if (
            isinstance(
                self.keep_recent,
                bool,
            )
            or not isinstance(
                self.keep_recent,
                int,
            )
            or self.keep_recent < 1
        ):
            raise ValueError(
                "keep_recent must be an integer >= 1"
            )


@dataclass(frozen=True)
class ChampionHistoryRetentionPlan:
    """Immutable read-only retention plan."""

    keep_recent: int
    retained_revision_ids: tuple[str, ...]
    prunable_revision_ids: tuple[str, ...]
    retained_decision_sha256s: tuple[str, ...]
    prunable_decision_sha256s: tuple[str, ...]
    active_source_sha256: str


class ChampionHistoryRetentionPlanner:
    """Build a verified non-mutating history retention plan."""

    def __init__(
        self,
        registry_root: str | Path,
    ) -> None:
        self._registry_root = Path(
            registry_root
        )

        self._history_reader = (
            ChampionRollbackHistoryReader(
                registry_root=(
                    self._registry_root
                )
            )
        )

    def plan(
        self,
        policy: ChampionHistoryRetentionPolicy,
    ) -> ChampionHistoryRetentionPlan:
        if not isinstance(
            policy,
            ChampionHistoryRetentionPolicy,
        ):
            raise TypeError(
                "policy must be ChampionHistoryRetentionPolicy"
            )

        targets = (
            self._history_reader
            .list_revisions()
        )

        active_revision_id, active_source_sha256 = (
            self._read_active_identity()
        )

        if not targets:
            raise ValueError(
                "publication history is empty"
            )

        target_by_revision = {
            target.revision_id:
                target
            for target in targets
        }

        if (
            active_revision_id
            not in target_by_revision
        ):
            raise ValueError(
                "active publication revision "
                "not found in history"
            )

        if (
            target_by_revision[
                active_revision_id
            ].source_sha256
            != active_source_sha256
        ):
            raise ValueError(
                "active publication history "
                "source mismatch"
            )

        keep_count = min(
            policy.keep_recent,
            len(targets),
        )

        newest = list(
            targets[
                len(targets)
                - keep_count:
            ]
        )

        retained_revision_set = {
            target.revision_id
            for target in newest
        }

        # The active generation is an independent
        # preservation root. Rollback can make an older
        # publication revision active, so KEEP_RECENT_N
        # alone is not sufficient.
        retained_revision_set.add(
            active_revision_id
        )

        retained_targets = [
            target
            for target in targets
            if (
                target.revision_id
                in retained_revision_set
            )
        ]

        prunable_targets = [
            target
            for target in targets
            if (
                target.revision_id
                not in retained_revision_set
            )
        ]

        retained_decision_set = {
            target.source_sha256
            for target in retained_targets
        }

        retained_decision_set.add(
            active_source_sha256
        )

        all_decision_ids = (
            self._read_all_decision_snapshot_ids()
        )

        all_referenced_decision_ids = {
            target.source_sha256
            for target in targets
        }

        # Every publication revision has already been
        # fully resolved by ChampionRollbackHistoryReader,
        # so a missing/corrupt referenced snapshot fails
        # before planning reaches this point.
        if not (
            all_referenced_decision_ids
            <= all_decision_ids
        ):
            raise FileNotFoundError(
                "required decision snapshot not found"
            )

        if (
            active_source_sha256
            not in all_decision_ids
        ):
            raise FileNotFoundError(
                "active decision snapshot not found"
            )

        prunable_decision_set = (
            all_decision_ids
            - retained_decision_set
        )

        retained_revision_ids = tuple(
            target.revision_id
            for target in targets
            if (
                target.revision_id
                in retained_revision_set
            )
        )

        prunable_revision_ids = tuple(
            target.revision_id
            for target in prunable_targets
        )

        retained_decision_sha256s = tuple(
            sorted(
                retained_decision_set
            )
        )

        prunable_decision_sha256s = tuple(
            sorted(
                prunable_decision_set
            )
        )

        return ChampionHistoryRetentionPlan(
            keep_recent=(
                policy.keep_recent
            ),
            retained_revision_ids=(
                retained_revision_ids
            ),
            prunable_revision_ids=(
                prunable_revision_ids
            ),
            retained_decision_sha256s=(
                retained_decision_sha256s
            ),
            prunable_decision_sha256s=(
                prunable_decision_sha256s
            ),
            active_source_sha256=(
                active_source_sha256
            ),
        )

    def _read_active_identity(
        self,
    ) -> tuple[str, str]:
        active_root = (
            self._registry_root
            / "active"
        )

        decision_path = (
            active_root
            / "champion_decision.json"
        )

        publication_path = (
            active_root
            / "publication.json"
        )

        if not decision_path.exists():
            raise FileNotFoundError(
                decision_path
            )

        if decision_path.is_dir():
            raise IsADirectoryError(
                decision_path
            )

        if not decision_path.is_file():
            raise FileNotFoundError(
                decision_path
            )

        if not publication_path.exists():
            raise FileNotFoundError(
                publication_path
            )

        if publication_path.is_dir():
            raise IsADirectoryError(
                publication_path
            )

        if not publication_path.is_file():
            raise FileNotFoundError(
                publication_path
            )

        decision_bytes = (
            decision_path.read_bytes()
        )

        publication_bytes = (
            publication_path.read_bytes()
        )

        source_sha256 = (
            hashlib.sha256(
                decision_bytes
            )
            .hexdigest()
        )

        revision_id = (
            hashlib.sha256(
                publication_bytes
            )
            .hexdigest()
        )

        try:
            publication = json.loads(
                publication_bytes.decode(
                    "utf-8-sig"
                )
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise ValueError(
                "invalid active publication"
            ) from exc

        if not isinstance(
            publication,
            dict,
        ):
            raise ValueError(
                "invalid active publication"
            )

        recorded_source_sha256 = (
            publication.get(
                "source_sha256"
            )
        )

        if (
            not isinstance(
                recorded_source_sha256,
                str,
            )
            or recorded_source_sha256
            != source_sha256
        ):
            raise ValueError(
                "active decision sha256 mismatch"
            )

        history_path = (
            self._registry_root
            / "history"
            / f"{revision_id}.json"
        )

        if not history_path.is_file():
            raise ValueError(
                "active publication revision "
                "not found in history"
            )

        if (
            history_path.read_bytes()
            != publication_bytes
        ):
            raise ValueError(
                "active publication revision mismatch"
            )

        return (
            revision_id,
            source_sha256,
        )

    def _read_all_decision_snapshot_ids(
        self,
    ) -> set[str]:
        root = (
            self._registry_root
            / "history"
            / "decisions"
        )

        if not root.exists():
            raise FileNotFoundError(
                root
            )

        if not root.is_dir():
            raise NotADirectoryError(
                root
            )

        result: set[str] = set()

        for path in root.glob(
            "*.json"
        ):
            if not path.is_file():
                continue

            source_sha256 = (
                path.stem
            )

            if (
                len(source_sha256) != 64
                or any(
                    character
                    not in "0123456789abcdef"
                    for character
                    in source_sha256
                )
            ):
                # Unknown decision-history files are
                # outside the known pruning population.
                continue

            data = (
                path.read_bytes()
            )

            actual_sha256 = (
                hashlib.sha256(
                    data
                )
                .hexdigest()
            )

            if (
                actual_sha256
                != source_sha256
            ):
                raise ValueError(
                    "decision snapshot sha256 mismatch"
                )

            result.add(
                source_sha256
            )

        return result
@dataclass(frozen=True)
class ChampionHistoryRetentionAtomicityError(
    RuntimeError
):
    """Retention mutation could not be fully compensated."""

    def __init__(
        self,
        *,
        mutation_error: BaseException,
        restoration_error: BaseException,
    ) -> None:
        object.__setattr__(
            self,
            "mutation_error",
            mutation_error,
        )
        object.__setattr__(
            self,
            "restoration_error",
            restoration_error,
        )

        super().__init__(
            "retention atomicity compensation failed: "
            f"mutation={type(mutation_error).__name__}: "
            f"{mutation_error}; "
            f"restore={type(restoration_error).__name__}: "
            f"{restoration_error}"
        )


@dataclass(frozen=True)
class ChampionHistoryRetentionResult:
    """Result of one retention execution."""

    deleted_revision_ids: tuple[str, ...]
    deleted_decision_sha256s: tuple[str, ...]


class ChampionHistoryRetentionExecutor:
    """Execute an already-computed retention plan safely."""

    def __init__(
        self,
        registry_root: str | Path,
    ) -> None:
        self._registry_root = Path(
            registry_root
        )

    def execute(
        self,
        plan: ChampionHistoryRetentionPlan,
    ) -> ChampionHistoryRetentionResult:

        if type(plan) is not ChampionHistoryRetentionPlan:
            raise TypeError(
                "plan must be "
                "ChampionHistoryRetentionPlan"
            )

        with ProductionRegistryWriterLock(
            self._registry_root
        ):
            self._validate_plan_locked(
                plan
            )

            revision_paths, decision_paths = (
                self._resolve_deletion_paths_locked(
                    plan
                )
            )

            deletion_paths = (
                revision_paths
                + decision_paths
            )

            original_bytes = (
                self._capture_original_bytes_locked(
                    deletion_paths
                )
            )

            deleted_paths: list[Path] = []

            try:
                for path in deletion_paths:
                    path.unlink()
                    deleted_paths.append(
                        path
                    )

            except Exception as mutation_error:

                try:
                    self._restore_deleted_locked(
                        deleted_paths=(
                            deleted_paths
                        ),
                        original_bytes=(
                            original_bytes
                        ),
                    )

                except Exception as restoration_error:
                    raise (
                        ChampionHistoryRetentionAtomicityError(
                            mutation_error=(
                                mutation_error
                            ),
                            restoration_error=(
                                restoration_error
                            ),
                        )
                    ) from restoration_error

                # Compensation succeeded. Preserve the
                # original mutation failure contract.
                raise

            return ChampionHistoryRetentionResult(
                deleted_revision_ids=tuple(
                    sorted(
                        plan.prunable_revision_ids
                    )
                ),
                deleted_decision_sha256s=tuple(
                    sorted(
                        plan.prunable_decision_sha256s
                    )
                ),
            )

    def _resolve_deletion_paths_locked(
        self,
        plan: ChampionHistoryRetentionPlan,
    ) -> tuple[
        tuple[Path, ...],
        tuple[Path, ...],
    ]:

        registry_root = (
            self._registry_root
            .resolve()
        )

        history_root = (
            registry_root
            / "history"
        )

        decision_root = (
            history_root
            / "decisions"
        )

        revision_paths = tuple(
            history_root
            / f"{revision_id}.json"
            for revision_id
            in plan.prunable_revision_ids
        )

        decision_paths = tuple(
            decision_root
            / f"{sha256}.json"
            for sha256
            in plan.prunable_decision_sha256s
        )

        for path in revision_paths:
            resolved = path.resolve()

            if (
                resolved.parent
                != history_root.resolve()
            ):
                raise ValueError(
                    "invalid retention revision path"
                )

        for path in decision_paths:
            resolved = path.resolve()

            if (
                resolved.parent
                != decision_root.resolve()
            ):
                raise ValueError(
                    "invalid retention decision path"
                )

        return (
            revision_paths,
            decision_paths,
        )

    def _capture_original_bytes_locked(
        self,
        deletion_paths: tuple[Path, ...],
    ) -> dict[Path, bytes]:

        original_bytes: dict[
            Path,
            bytes,
        ] = {}

        missing: list[Path] = []

        for path in deletion_paths:

            if not path.is_file():
                missing.append(
                    path
                )
                continue

            original_bytes[
                path
            ] = path.read_bytes()

        if missing:
            raise FileNotFoundError(
                "planned retention file missing: "
                + ", ".join(
                    str(path)
                    for path in missing
                )
            )

        return original_bytes

    def _restore_deleted_locked(
        self,
        *,
        deleted_paths: list[Path],
        original_bytes: dict[Path, bytes],
    ) -> None:

        for path in reversed(
            deleted_paths
        ):
            data = original_bytes[
                path
            ]

            path.write_bytes(
                data
            )

            restored = (
                path.read_bytes()
            )

            if restored != data:
                raise OSError(
                    "retention compensation "
                    "byte verification failed: "
                    f"{path}"
                )

    def _validate_plan_locked(
        self,
        plan: ChampionHistoryRetentionPlan,
    ) -> None:

        # Reconstruct the effective policy represented
        # by the existing plan. This behavior is kept
        # unchanged in W-10C; policy provenance can be
        # strengthened separately if required.
        keep_recent = len(
            plan.retained_revision_ids
        )

        if keep_recent < 1:
            raise ValueError(
                "retention plan has no retained "
                "publication revision"
            )

        current = (
            ChampionHistoryRetentionPlanner(
                self._registry_root
            )
            .plan(
                ChampionHistoryRetentionPolicy(
                    keep_recent=keep_recent
                )
            )
        )

        if current != plan:
            raise RuntimeError(
                "stale retention plan"
            )
