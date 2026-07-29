"""Safe integration between Project E rescoring and Project D candidates."""

from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
    is_dataclass,
    replace,
)
import math
from typing import (
    Any,
    Callable,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

from lrp.contracts import ContractError

from .rescoring import (
    CandidateRescorer,
    RescoredCandidate,
    RescoringResult,
    StrategyKey,
)
from .snapshot import LearningSnapshot


SnapshotLoader = Callable[[int], LearningSnapshot]


@runtime_checkable
class LearningSnapshotRepository(Protocol):
    """Minimal repository contract required by pipeline integration."""

    def load_snapshot(
        self,
        *,
        round_no: int,
    ) -> LearningSnapshot:
        ...


@dataclass(frozen=True, slots=True)
class PipelineRescoringResult:
    """Output of one Project E pipeline integration pass."""

    original_candidates: tuple[object, ...]
    effective_candidates: tuple[object, ...]
    rescoring: RescoringResult | None
    enabled: bool
    applied: bool
    fallback_reason: str | None = None
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        original = tuple(self.original_candidates)
        effective = tuple(self.effective_candidates)

        if len(original) != len(effective):
            raise ContractError(
                "original_candidates and effective_candidates "
                "must have equal lengths"
            )

        if self.rescoring is not None:
            if not isinstance(
                self.rescoring,
                RescoringResult,
            ):
                raise ContractError(
                    "rescoring must be a RescoringResult or None"
                )

            if self.rescoring.count != len(original):
                raise ContractError(
                    "rescoring candidate count does not match "
                    "pipeline candidate count"
                )

        if not isinstance(self.metadata, Mapping):
            raise ContractError(
                "metadata must be a mapping"
            )

        object.__setattr__(
            self,
            "original_candidates",
            original,
        )
        object.__setattr__(
            self,
            "effective_candidates",
            effective,
        )
        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )

    @property
    def candidate_count(self) -> int:
        return len(self.effective_candidates)

    @property
    def changed_rank_count(self) -> int:
        if self.rescoring is None:
            return 0

        return self.rescoring.changed_rank_count

    @property
    def evidence_candidate_count(self) -> int:
        if self.rescoring is None:
            return 0

        value = self.rescoring.metadata.get(
            "evidence_candidate_count",
            0,
        )

        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "applied": self.applied,
            "fallback_reason": self.fallback_reason,
            "candidate_count": self.candidate_count,
            "changed_rank_count": (
                self.changed_rank_count
            ),
            "evidence_candidate_count": (
                self.evidence_candidate_count
            ),
            "rescoring": (
                self.rescoring.to_dict()
                if self.rescoring is not None
                else None
            ),
            "metadata": dict(self.metadata),
        }


def _read(
    value: object,
    *names: str,
) -> Any:
    for name in names:
        if isinstance(value, Mapping):
            if name in value:
                return value[name]

        if hasattr(value, name):
            return getattr(value, name)

    return None


def _nested_values(
    value: object,
) -> tuple[object, ...]:
    """Return known nested Project D wrapper objects."""

    result: list[object] = []

    for name in (
        "source",
        "ranked",
        "scored",
        "scored_candidate",
        "candidate",
        "item",
    ):
        nested = _read(value, name)

        if (
            nested is not None
            and nested is not value
            and nested not in result
        ):
            result.append(nested)

    return tuple(result)


def recursive_strategy_resolver(
    candidate: object,
) -> tuple[StrategyKey, ...]:
    """Resolve model/scenario provenance through nested wrappers."""

    pending: list[tuple[object, int]] = [
        (candidate, 0)
    ]
    visited: set[int] = set()
    keys: list[StrategyKey] = []

    while pending:
        current, depth = pending.pop(0)

        identity = id(current)

        if identity in visited:
            continue

        visited.add(identity)

        strategy_type = _read(
            current,
            "strategy_type",
        )
        strategy_name = _read(
            current,
            "strategy_name",
        )

        if strategy_type and strategy_name:
            keys.append(
                (
                    str(strategy_type)
                    .strip()
                    .lower(),
                    str(strategy_name).strip(),
                )
            )

        model_name = _read(
            current,
            "model_name",
        )

        if model_name:
            keys.append(
                (
                    "model",
                    str(model_name).strip(),
                )
            )

        scenario_name = _read(
            current,
            "scenario_name",
        )

        if scenario_name:
            keys.append(
                (
                    "scenario",
                    str(scenario_name).strip(),
                )
            )

        scenario_names = _read(
            current,
            "scenario_names",
        )

        if scenario_names:
            try:
                for name in scenario_names:
                    keys.append(
                        (
                            "scenario",
                            str(name).strip(),
                        )
                    )
            except TypeError:
                pass

        features = _read(
            current,
            "features",
        )

        if isinstance(features, Mapping):
            feature_model = features.get(
                "model_name"
            )

            if feature_model:
                keys.append(
                    (
                        "model",
                        str(feature_model).strip(),
                    )
                )

            feature_scenarios = features.get(
                "scenario_names",
                (),
            )

            try:
                for name in (
                    feature_scenarios or ()
                ):
                    keys.append(
                        (
                            "scenario",
                            str(name).strip(),
                        )
                    )
            except TypeError:
                pass

        if depth < 8:
            pending.extend(
                (
                    nested,
                    depth + 1,
                )
                for nested in _nested_values(
                    current
                )
            )

    return tuple(
        dict.fromkeys(
            key
            for key in keys
            if key[0] and key[1]
        )
    )


def recursive_base_score_reader(
    candidate: object,
) -> float:
    """Locate a normalized score through Project D wrappers."""

    pending: list[tuple[object, int]] = [
        (candidate, 0)
    ]
    visited: set[int] = set()

    while pending:
        current, depth = pending.pop(0)

        identity = id(current)

        if identity in visited:
            continue

        visited.add(identity)

        for name in (
            "normalized_score",
            "ensemble_score",
            "ranking_score",
            "final_score",
            "score",
            "base_score",
        ):
            raw = _read(current, name)

            if raw is None:
                continue

            if isinstance(raw, bool):
                continue

            try:
                score = float(raw)
            except (TypeError, ValueError):
                continue

            if math.isfinite(score):
                return min(
                    1.0,
                    max(0.0, score),
                )

        if depth < 8:
            pending.extend(
                (
                    nested,
                    depth + 1,
                )
                for nested in _nested_values(
                    current
                )
            )

    raise ContractError(
        "unable to locate a finite candidate score"
    )


def _replace_mapping_score(
    candidate: Mapping[Any, Any],
    score: float,
) -> Mapping[Any, Any]:
    result = dict(candidate)

    if "normalized_score" in result:
        result["normalized_score"] = score
        return result

    for name in (
        "scored",
        "scored_candidate",
        "ranked",
        "candidate",
        "item",
        "source",
    ):
        nested = result.get(name)

        if nested is None:
            continue

        try:
            result[name] = replace_candidate_score(
                nested,
                score,
            )
            return result
        except ContractError:
            continue

    raise ContractError(
        "mapping candidate does not expose "
        "a replaceable normalized score"
    )


def replace_candidate_score(
    candidate: object,
    score: float,
) -> object:
    """Return an immutable copy with normalized_score replaced.

    Project D dataclasses remain Project D dataclasses. No wrapper is
    passed into ranking, diversity, or practical selection.
    """

    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
    ):
        raise ContractError(
            "score must be a finite number"
        )

    normalized = min(
        1.0,
        max(0.0, float(score)),
    )

    if isinstance(candidate, Mapping):
        return _replace_mapping_score(
            candidate,
            normalized,
        )

    if not is_dataclass(candidate):
        raise ContractError(
            "candidate must be a dataclass or mapping "
            "to replace its score safely"
        )

    dataclass_fields = getattr(
        candidate,
        "__dataclass_fields__",
        {},
    )

    if "normalized_score" in dataclass_fields:
        return replace(
            candidate,
            normalized_score=normalized,
        )

    for name in (
        "scored",
        "scored_candidate",
        "ranked",
        "candidate",
        "item",
        "source",
    ):
        if name not in dataclass_fields:
            continue

        nested = getattr(candidate, name)

        try:
            replaced_nested = (
                replace_candidate_score(
                    nested,
                    normalized,
                )
            )
        except ContractError:
            continue

        return replace(
            candidate,
            **{
                name: replaced_nested,
            },
        )

    raise ContractError(
        "candidate dataclass does not expose "
        "a replaceable normalized score"
    )


@dataclass(slots=True)
class PipelineRescoringBridge:
    """Inject E-003 rescoring before Project D ranking/diversity."""

    snapshot_repository: (
        LearningSnapshotRepository
        | SnapshotLoader
        | None
    ) = None

    rescorer: CandidateRescorer = field(
        default_factory=lambda: CandidateRescorer(
            score_reader=recursive_base_score_reader,
            strategy_resolver=(
                recursive_strategy_resolver
            ),
        )
    )

    enabled: bool = True
    fail_open: bool = True

    def __post_init__(self) -> None:
        if not isinstance(
            self.rescorer,
            CandidateRescorer,
        ):
            raise ContractError(
                "rescorer must be a CandidateRescorer"
            )

        if not isinstance(self.enabled, bool):
            raise ContractError(
                "enabled must be boolean"
            )

        if not isinstance(self.fail_open, bool):
            raise ContractError(
                "fail_open must be boolean"
            )

        repository = self.snapshot_repository

        if repository is None:
            return

        if callable(repository):
            return

        if isinstance(
            repository,
            LearningSnapshotRepository,
        ):
            return

        raise ContractError(
            "snapshot_repository must be callable, "
            "implement load_snapshot(), or be None"
        )

    def _load_snapshot(
        self,
        *,
        round_no: int,
    ) -> LearningSnapshot:
        repository = self.snapshot_repository

        if repository is None:
            raise ContractError(
                "learning snapshot repository is not configured"
            )

        if callable(repository):
            snapshot = repository(round_no)
        else:
            snapshot = repository.load_snapshot(
                round_no=round_no
            )

        if not isinstance(
            snapshot,
            LearningSnapshot,
        ):
            raise ContractError(
                "snapshot repository returned "
                "an invalid LearningSnapshot"
            )

        if snapshot.round_no != round_no:
            raise ContractError(
                "learning snapshot round does not "
                "match prediction round"
            )

        return snapshot

    def _fallback(
        self,
        candidates: tuple[object, ...],
        *,
        reason: str,
    ) -> PipelineRescoringResult:
        return PipelineRescoringResult(
            original_candidates=candidates,
            effective_candidates=candidates,
            rescoring=None,
            enabled=self.enabled,
            applied=False,
            fallback_reason=reason,
            metadata={
                "fail_open": self.fail_open,
                "score_replacement": False,
            },
        )

    def apply(
        self,
        candidates: Sequence[object],
        *,
        round_no: int,
    ) -> PipelineRescoringResult:
        sources = tuple(candidates)

        if (
            isinstance(round_no, bool)
            or not isinstance(round_no, int)
            or round_no <= 0
        ):
            raise ContractError(
                "round_no must be a positive integer"
            )

        if not sources:
            raise ContractError(
                "candidates must not be empty"
            )

        if not self.enabled:
            return self._fallback(
                sources,
                reason="disabled",
            )

        if self.snapshot_repository is None:
            return self._fallback(
                sources,
                reason="snapshot_repository_unconfigured",
            )

        try:
            snapshot = self._load_snapshot(
                round_no=round_no
            )

            rescoring = self.rescorer.evaluate(
                sources,
                snapshot=snapshot,
            )

            effective = tuple(
                replace_candidate_score(
                    item.source,
                    item.ensemble_score,
                )
                for item in rescoring.items
            )

        except Exception as exc:
            if not self.fail_open:
                raise

            return self._fallback(
                sources,
                reason=(
                    f"{exc.__class__.__name__}: {exc}"
                ),
            )

        return PipelineRescoringResult(
            original_candidates=sources,
            effective_candidates=effective,
            rescoring=rescoring,
            enabled=True,
            applied=True,
            fallback_reason=None,
            metadata={
                "fail_open": self.fail_open,
                "score_replacement": True,
                "snapshot_revision": list(
                    snapshot.revision
                ),
                "snapshot_source": snapshot.source,
            },
        )
