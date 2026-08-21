from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from typing import Any

from lrp.contracts.exceptions import ContractError


SelectedSet = tuple[int, int, int, int, int, int]


def _is_strict_positive_int(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
    )


def _normalize_required_text(
    value: Any,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise ContractError(
            f"{field_name} must be a non-empty string"
        )

    normalized = value.strip()

    if not normalized:
        raise ContractError(
            f"{field_name} must be a non-empty string"
        )

    return normalized


def _normalize_optional_text(
    value: Any,
    *,
    field_name: str,
) -> str | None:
    if value is None:
        return None

    return _normalize_required_text(
        value,
        field_name=field_name,
    )


def _normalize_selected_sets(
    value: Any,
) -> tuple[SelectedSet, ...]:
    try:
        rows = tuple(
            tuple(row)
            for row in value
        )
    except (TypeError, ValueError):
        raise ContractError(
            "selected_sets must be an iterable of sets"
        ) from None

    normalized: list[SelectedSet] = []

    for row in rows:
        if len(row) != 6:
            raise ContractError(
                "each selected set must contain six numbers"
            )

        numbers: list[int] = []

        for number in row:
            if (
                not isinstance(number, int)
                or isinstance(number, bool)
                or not 1 <= number <= 45
            ):
                raise ContractError(
                    "selected set numbers must be integers "
                    "between 1 and 45"
                )

            numbers.append(number)

        if len(set(numbers)) != 6:
            raise ContractError(
                "selected set numbers must be unique"
            )

        normalized.append(
            tuple(
                sorted(numbers)
            )
        )

    result = tuple(normalized)

    if len(set(result)) != len(result):
        raise ContractError(
            "selected_sets must not contain duplicates"
        )

    return result


def _normalize_history_rounds(
    value: Any,
    *,
    round_no: int,
) -> tuple[int, ...]:
    try:
        rows = tuple(value)
    except TypeError:
        raise ContractError(
            "history_rounds must be iterable"
        ) from None

    normalized: list[int] = []

    for history_round in rows:
        if (
            not isinstance(history_round, int)
            or isinstance(history_round, bool)
            or history_round <= 0
        ):
            raise ContractError(
                "history rounds must be positive integers"
            )

        if history_round >= round_no:
            raise ContractError(
                "history rounds must precede prediction round"
            )

        normalized.append(
            history_round
        )

    if not normalized:
        raise ContractError(
            "history rounds must not be empty"
        )

    if len(set(normalized)) != len(normalized):
        raise ContractError(
            "history rounds must be unique"
        )

    if any(
        left >= right
        for left, right in zip(
            normalized,
            normalized[1:],
        )
    ):
        raise ContractError(
            "history rounds must be strictly increasing"
        )

    return tuple(normalized)


@dataclass(frozen=True)
class TopKLiveEvaluationSourceSnapshot:
    schema_version: str
    round_no: int
    top_k: int
    selected_sets: tuple[SelectedSet, ...]
    model_name: str
    history_rounds: tuple[int, ...]
    regime_id: str | None
    strategy_name: str | None
    generated_at_kst: datetime
    source_artifact_sha256: str

    def __post_init__(self) -> None:
        schema_version = _normalize_required_text(
            self.schema_version,
            field_name="schema_version",
        )

        if schema_version != "1.0":
            raise ContractError(
                "unsupported schema_version"
            )

        if not _is_strict_positive_int(
            self.round_no
        ):
            raise ContractError(
                "round_no must be a positive integer"
            )

        if not _is_strict_positive_int(
            self.top_k
        ):
            raise ContractError(
                "top_k must be a positive integer"
            )

        selected_sets = _normalize_selected_sets(
            self.selected_sets
        )

        if len(selected_sets) != self.top_k:
            raise ContractError(
                "selected set count must equal top_k"
            )

        model_name = _normalize_required_text(
            self.model_name,
            field_name="model_name",
        )

        history_rounds = _normalize_history_rounds(
            self.history_rounds,
            round_no=self.round_no,
        )

        regime_id = _normalize_optional_text(
            self.regime_id,
            field_name="regime_id",
        )

        strategy_name = _normalize_optional_text(
            self.strategy_name,
            field_name="strategy_name",
        )

        if not isinstance(
            self.generated_at_kst,
            datetime,
        ):
            raise ContractError(
                "generated_at_kst must be datetime"
            )

        offset = self.generated_at_kst.utcoffset()

        if offset != timedelta(hours=9):
            raise ContractError(
                "generated_at_kst must use +09:00 offset"
            )

        source_artifact_sha256 = (
            _normalize_required_text(
                self.source_artifact_sha256,
                field_name="source_artifact_sha256",
            )
        )

        if (
            len(source_artifact_sha256) != 64
            or any(
                char not in "0123456789abcdef"
                for char in source_artifact_sha256
            )
        ):
            raise ContractError(
                "source_artifact_sha256 must be "
                "64 lowercase hexadecimal characters"
            )

        object.__setattr__(
            self,
            "schema_version",
            schema_version,
        )
        object.__setattr__(
            self,
            "selected_sets",
            selected_sets,
        )
        object.__setattr__(
            self,
            "model_name",
            model_name,
        )
        object.__setattr__(
            self,
            "history_rounds",
            history_rounds,
        )
        object.__setattr__(
            self,
            "regime_id",
            regime_id,
        )
        object.__setattr__(
            self,
            "strategy_name",
            strategy_name,
        )
        object.__setattr__(
            self,
            "source_artifact_sha256",
            source_artifact_sha256,
        )


@dataclass(frozen=True)
class TopKLiveEvaluationSourcePair:
    candidate: TopKLiveEvaluationSourceSnapshot
    baseline: TopKLiveEvaluationSourceSnapshot

    def __post_init__(self) -> None:
        if not isinstance(
            self.candidate,
            TopKLiveEvaluationSourceSnapshot,
        ):
            raise ContractError(
                "candidate must be source snapshot"
            )

        if not isinstance(
            self.baseline,
            TopKLiveEvaluationSourceSnapshot,
        ):
            raise ContractError(
                "baseline must be source snapshot"
            )

        if (
            self.candidate.round_no
            != self.baseline.round_no
        ):
            raise ContractError(
                "candidate and baseline rounds must match"
            )

        if (
            self.candidate.model_name
            == self.baseline.model_name
        ):
            raise ContractError(
                "candidate and baseline model identities "
                "must differ"
            )


def snapshot_to_dict(
    snapshot: TopKLiveEvaluationSourceSnapshot,
) -> dict[str, Any]:
    if not isinstance(
        snapshot,
        TopKLiveEvaluationSourceSnapshot,
    ):
        raise ContractError(
            "snapshot has invalid type"
        )

    return {
        "schema_version": snapshot.schema_version,
        "round_no": snapshot.round_no,
        "top_k": snapshot.top_k,
        "selected_sets": [
            list(row)
            for row in snapshot.selected_sets
        ],
        "model_name": snapshot.model_name,
        "history_rounds": list(
            snapshot.history_rounds
        ),
        "regime_id": snapshot.regime_id,
        "strategy_name": snapshot.strategy_name,
        "generated_at_kst": (
            snapshot.generated_at_kst.isoformat()
        ),
        "source_artifact_sha256": (
            snapshot.source_artifact_sha256
        ),
    }


def snapshot_from_dict(
    payload: dict[str, Any],
) -> TopKLiveEvaluationSourceSnapshot:
    if not isinstance(payload, dict):
        raise ContractError(
            "snapshot payload must be a dict"
        )

    expected_keys = {
        "schema_version",
        "round_no",
        "top_k",
        "selected_sets",
        "model_name",
        "history_rounds",
        "regime_id",
        "strategy_name",
        "generated_at_kst",
        "source_artifact_sha256",
    }

    actual_keys = set(payload)

    if actual_keys != expected_keys:
        raise ContractError(
            "snapshot payload keys do not match schema"
        )

    try:
        generated_at_kst = datetime.fromisoformat(
            payload["generated_at_kst"]
        )

        return TopKLiveEvaluationSourceSnapshot(
            schema_version=payload["schema_version"],
            round_no=payload["round_no"],
            top_k=payload["top_k"],
            selected_sets=payload["selected_sets"],
            model_name=payload["model_name"],
            history_rounds=payload["history_rounds"],
            regime_id=payload["regime_id"],
            strategy_name=payload["strategy_name"],
            generated_at_kst=generated_at_kst,
            source_artifact_sha256=(
                payload["source_artifact_sha256"]
            ),
        )
    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise ContractError(
            "invalid snapshot payload"
        ) from exc


def snapshot_to_json(
    snapshot: TopKLiveEvaluationSourceSnapshot,
) -> str:
    return json.dumps(
        snapshot_to_dict(snapshot),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def snapshot_from_json(
    payload: str,
) -> TopKLiveEvaluationSourceSnapshot:
    if not isinstance(payload, str):
        raise ContractError(
            "snapshot JSON must be a string"
        )

    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ContractError(
            "invalid snapshot JSON"
        ) from exc

    return snapshot_from_dict(
        decoded
    )
