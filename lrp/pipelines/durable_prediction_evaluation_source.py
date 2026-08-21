from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from typing import Any

from lrp.contracts.exceptions import ContractError


_SUPPORTED_SCHEMA_VERSION = "1.0"

_EXACT_PAYLOAD_KEYS = frozenset(
    {
        "schema_version",
        "round_no",
        "top_k",
        "selected_sets",
        "generated_at_kst",
    }
)


def _is_strict_positive_int(
    value: Any,
) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
    )


def _normalize_selected_sets(
    value: Any,
) -> tuple[tuple[int, ...], ...]:
    try:
        raw_sets = tuple(
            value
        )
    except TypeError as exc:
        raise ContractError(
            "selected_sets must be iterable"
        ) from exc

    normalized: list[
        tuple[int, ...]
    ] = []

    for raw_numbers in raw_sets:
        try:
            numbers = tuple(
                raw_numbers
            )
        except TypeError as exc:
            raise ContractError(
                "each selected set must be iterable"
            ) from exc

        if len(numbers) != 6:
            raise ContractError(
                "each selected set must contain exactly 6 numbers"
            )

        for number in numbers:
            if (
                not isinstance(
                    number,
                    int,
                )
                or isinstance(
                    number,
                    bool,
                )
            ):
                raise ContractError(
                    "selected-set numbers must be integers"
                )

            if not 1 <= number <= 45:
                raise ContractError(
                    "selected-set numbers must be between 1 and 45"
                )

        if len(set(numbers)) != 6:
            raise ContractError(
                "selected-set numbers must be unique"
            )

        normalized.append(
            tuple(
                sorted(
                    numbers
                )
            )
        )

    result = tuple(
        normalized
    )

    if len(set(result)) != len(result):
        raise ContractError(
            "selected_sets must not contain duplicates"
        )

    return result


def _validate_generated_at_kst(
    value: Any,
) -> datetime:
    if not isinstance(
        value,
        datetime,
    ):
        raise ContractError(
            "generated_at_kst must be datetime"
        )

    offset = value.utcoffset()

    if offset is None:
        raise ContractError(
            "generated_at_kst must be timezone-aware"
        )

    if offset != timedelta(hours=9):
        raise ContractError(
            "generated_at_kst must use +09:00 offset"
        )

    return value


@dataclass(
    frozen=True,
)
class DurablePredictionEvaluationSource:
    schema_version: str
    round_no: int
    top_k: int
    selected_sets: tuple[
        tuple[int, ...],
        ...
    ]
    generated_at_kst: datetime

    def __post_init__(
        self,
    ) -> None:
        if (
            self.schema_version
            != _SUPPORTED_SCHEMA_VERSION
        ):
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

        normalized_sets = (
            _normalize_selected_sets(
                self.selected_sets
            )
        )

        if len(
            normalized_sets
        ) != self.top_k:
            raise ContractError(
                "selected_sets count must equal top_k"
            )

        generated_at = (
            _validate_generated_at_kst(
                self.generated_at_kst
            )
        )

        object.__setattr__(
            self,
            "selected_sets",
            normalized_sets,
        )

        object.__setattr__(
            self,
            "generated_at_kst",
            generated_at,
        )


def source_to_dict(
    source: DurablePredictionEvaluationSource,
) -> dict[str, Any]:
    if not isinstance(
        source,
        DurablePredictionEvaluationSource,
    ):
        raise ContractError(
            "source must be DurablePredictionEvaluationSource"
        )

    return {
        "schema_version": (
            source.schema_version
        ),
        "round_no": (
            source.round_no
        ),
        "top_k": (
            source.top_k
        ),
        "selected_sets": [
            list(
                numbers
            )
            for numbers
            in source.selected_sets
        ],
        "generated_at_kst": (
            source.generated_at_kst.isoformat()
        ),
    }


def source_from_dict(
    payload: dict[str, Any],
) -> DurablePredictionEvaluationSource:
    if not isinstance(
        payload,
        dict,
    ):
        raise ContractError(
            "payload must be dict"
        )

    keys = frozenset(
        payload.keys()
    )

    if keys != _EXACT_PAYLOAD_KEYS:
        raise ContractError(
            "payload keys must match exact schema"
        )

    timestamp_raw = payload[
        "generated_at_kst"
    ]

    if not isinstance(
        timestamp_raw,
        str,
    ):
        raise ContractError(
            "generated_at_kst payload must be string"
        )

    try:
        generated_at = datetime.fromisoformat(
            timestamp_raw
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ContractError(
            "generated_at_kst payload is invalid"
        ) from exc

    try:
        return DurablePredictionEvaluationSource(
            schema_version=payload[
                "schema_version"
            ],
            round_no=payload[
                "round_no"
            ],
            top_k=payload[
                "top_k"
            ],
            selected_sets=payload[
                "selected_sets"
            ],
            generated_at_kst=generated_at,
        )
    except ContractError:
        raise
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ContractError(
            "payload is invalid"
        ) from exc


def source_to_json(
    source: DurablePredictionEvaluationSource,
) -> str:
    return json.dumps(
        source_to_dict(
            source
        ),
        ensure_ascii=False,
        separators=(
            ",",
            ":",
        ),
    )


def source_from_json(
    payload: str,
) -> DurablePredictionEvaluationSource:
    if not isinstance(
        payload,
        str,
    ):
        raise ContractError(
            "payload must be string"
        )

    try:
        decoded = json.loads(
            payload
        )
    except (
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise ContractError(
            "payload must contain valid JSON"
        ) from exc

    if not isinstance(
        decoded,
        dict,
    ):
        raise ContractError(
            "JSON payload must be an object"
        )

    return source_from_dict(
        decoded
    )