"""Prediction artifact import into the existing learning domain."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any
import json

from lrp.learning.models import PredictionRecord


class OutcomeImportError(ValueError):
    """Raised when a prediction artifact cannot be imported."""


def _required_text(
    value: object,
    *,
    field: str,
) -> str:
    if not isinstance(value, str):
        raise OutcomeImportError(
            f"{field} must be a string"
        )

    normalized = value.strip()

    if not normalized:
        raise OutcomeImportError(
            f"{field} must not be empty"
        )

    return normalized


def _positive_int(
    value: object,
    *,
    field: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise OutcomeImportError(
            f"{field} must be a positive integer"
        )

    return value


def _integer(
    value: object,
    *,
    field: str,
) -> int:
    if isinstance(value, bool) or not isinstance(
        value,
        int,
    ):
        raise OutcomeImportError(
            f"{field} must be an integer"
        )

    return value


def _mapping(
    value: object,
    *,
    field: str,
) -> dict[str, Any]:
    if value is None:
        return {}

    if not isinstance(value, Mapping):
        raise OutcomeImportError(
            f"{field} must be an object"
        )

    return {
        str(key): item
        for key, item in value.items()
    }


def _load_payload(
    source: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return {
            str(key): value
            for key, value in source.items()
        }

    path = Path(source)

    if not path.is_file():
        raise FileNotFoundError(path)

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8-sig"
            )
        )
    except json.JSONDecodeError as exc:
        raise OutcomeImportError(
            "prediction artifact must contain valid JSON"
        ) from exc

    if not isinstance(payload, dict):
        raise OutcomeImportError(
            "prediction artifact must be a JSON object"
        )

    return payload


def _sets(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    value = payload.get("sets")

    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
    ):
        raise OutcomeImportError(
            "sets must be an array"
        )

    result: list[Mapping[str, Any]] = []

    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise OutcomeImportError(
                f"sets[{index}] must be an object"
            )

        result.append(item)

    if not result:
        raise OutcomeImportError(
            "sets must not be empty"
        )

    return tuple(result)


def _prediction_id(
    *,
    round_no: int,
    seed: int,
    model_name: str,
    set_id: str,
    numbers: Sequence[object],
) -> str:
    canonical = json.dumps(
        {
            "round": round_no,
            "seed": seed,
            "model_name": model_name,
            "set_id": set_id,
            "numbers": list(numbers),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    digest = sha256(
        canonical.encode("utf-8")
    ).hexdigest()[:16]

    return (
        f"prediction-{round_no}-"
        f"{set_id}-{digest}"
    )


class OutcomeImporter:
    """Convert serialized predictions to learning PredictionRecord items."""

    def __init__(
        self,
        *,
        model_name: str,
    ) -> None:
        self._model_name = _required_text(
            model_name,
            field="model_name",
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def import_predictions(
        self,
        source: str | Path | Mapping[str, Any],
    ) -> tuple[PredictionRecord, ...]:
        payload = _load_payload(source)

        round_no = _positive_int(
            payload.get("round"),
            field="round",
        )
        seed = _integer(
            payload.get("seed"),
            field="seed",
        )
        generated_at_kst = _required_text(
            payload.get("generated_at_kst"),
            field="generated_at_kst",
        )

        parameters = _mapping(
            payload.get("params"),
            field="params",
        )
        metadata = _mapping(
            payload.get("metadata"),
            field="metadata",
        )

        practical_value = payload.get(
            "top5_practical",
            (),
        )

        if (
            isinstance(
                practical_value,
                (str, bytes),
            )
            or not isinstance(
                practical_value,
                Sequence,
            )
        ):
            raise OutcomeImportError(
                "top5_practical must be an array"
            )

        practical_ids = {
            str(value)
            for value in practical_value
        }

        records: list[PredictionRecord] = []
        seen_set_ids: set[str] = set()

        for index, item in enumerate(
            _sets(payload)
        ):
            set_id = _required_text(
                item.get("id"),
                field=f"sets[{index}].id",
            )

            if set_id in seen_set_ids:
                raise OutcomeImportError(
                    f"duplicate set id: {set_id}"
                )

            seen_set_ids.add(set_id)

            numbers_value = item.get(
                "numbers",
            )

            if (
                isinstance(
                    numbers_value,
                    (str, bytes),
                )
                or not isinstance(
                    numbers_value,
                    Sequence,
                )
            ):
                raise OutcomeImportError(
                    f"sets[{index}].numbers "
                    "must be an array"
                )

            try:
                score = float(
                    item.get("score")
                )
            except (TypeError, ValueError) as exc:
                raise OutcomeImportError(
                    f"sets[{index}].score "
                    "must be numeric"
                ) from exc

            features = _mapping(
                item.get("features"),
                field=(
                    f"sets[{index}].features"
                ),
            )

            risk_flags_value = item.get(
                "risk_flags",
                (),
            )

            if (
                isinstance(
                    risk_flags_value,
                    (str, bytes),
                )
                or not isinstance(
                    risk_flags_value,
                    Sequence,
                )
            ):
                raise OutcomeImportError(
                    f"sets[{index}].risk_flags "
                    "must be an array"
                )

            enriched_features = {
                **features,
                "risk_flags": tuple(
                    str(value)
                    for value
                    in risk_flags_value
                ),
                "is_practical": (
                    set_id in practical_ids
                ),
            }

            record_parameters = {
                **parameters,
                "prediction_metadata": (
                    metadata
                ),
            }

            prediction_id = _prediction_id(
                round_no=round_no,
                seed=seed,
                model_name=self.model_name,
                set_id=set_id,
                numbers=numbers_value,
            )

            try:
                record = PredictionRecord(
                    prediction_id=prediction_id,
                    round_no=round_no,
                    set_id=set_id,
                    numbers=tuple(
                        int(value)
                        for value
                        in numbers_value
                    ),
                    score=score,
                    model_name=self.model_name,
                    seed=seed,
                    generated_at_kst=(
                        generated_at_kst
                    ),
                    features=enriched_features,
                    parameters=record_parameters,
                )
            except (
                TypeError,
                ValueError,
            ) as exc:
                raise OutcomeImportError(
                    f"invalid prediction set "
                    f"{set_id}: {exc}"
                ) from exc

            records.append(record)

        return tuple(records)
