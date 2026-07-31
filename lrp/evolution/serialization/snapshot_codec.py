from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from lrp.evolution.contracts.learning_context import (
    LearningContext,
)
from lrp.evolution.contracts.learning_cycle import (
    LearningCycleResult,
    LearningCycleStep,
)
from lrp.evolution.contracts.snapshot_schema import (
    LearningCycleSnapshot,
)


class SnapshotCodec:
    """Encode and decode learning-cycle snapshot payloads."""

    SUPPORTED_SCHEMA_VERSION = 1

    def encode(
        self,
        snapshot: LearningCycleSnapshot,
    ) -> dict[str, object]:
        if not isinstance(
            snapshot,
            LearningCycleSnapshot,
        ):
            raise TypeError(
                "snapshot must be a LearningCycleSnapshot"
            )

        return snapshot.to_payload()

    def decode(
        self,
        payload: Mapping[str, object],
    ) -> LearningCycleSnapshot:
        normalized = self._require_mapping(
            payload,
            field_name="payload",
        )

        schema_version = self._require_integer(
            normalized,
            "schema_version",
            minimum=1,
        )

        if (
            schema_version
            != self.SUPPORTED_SCHEMA_VERSION
        ):
            raise ValueError(
                "unsupported snapshot schema version: "
                f"{schema_version}"
            )

        snapshot_id = self._require_string(
            normalized,
            "snapshot_id",
        )
        created_at_utc = self._parse_datetime(
            self._require_string(
                normalized,
                "created_at_utc",
            )
        )
        source = self._require_string(
            normalized,
            "source",
        )
        metadata = self._optional_scalar_mapping(
            normalized,
            "metadata",
        )

        result_payload = self._require_mapping_field(
            normalized,
            "result",
        )
        result = self._decode_result(
            result_payload
        )

        snapshot = LearningCycleSnapshot(
            snapshot_id=snapshot_id,
            result=result,
            created_at_utc=created_at_utc,
            schema_version=schema_version,
            source=source,
            metadata=metadata,
        )

        self._validate_identity_fields(
            payload=normalized,
            snapshot=snapshot,
        )

        return snapshot

    def _decode_result(
        self,
        payload: Mapping[str, object],
    ) -> LearningCycleResult:
        initial_context = self._decode_context(
            self._require_mapping_field(
                payload,
                "initial_context",
            )
        )
        final_context = self._decode_context(
            self._require_mapping_field(
                payload,
                "final_context",
            )
        )
        steps = self._decode_steps(
            self._require_sequence_field(
                payload,
                "steps",
            )
        )
        metadata = self._optional_scalar_mapping(
            payload,
            "metadata",
        )

        result = LearningCycleResult(
            initial_context=initial_context,
            final_context=final_context,
            steps=steps,
            metadata=metadata,
        )

        if "step_count" in payload:
            step_count = self._require_integer(
                payload,
                "step_count",
                minimum=0,
            )

            if step_count != result.step_count:
                raise ValueError(
                    "result step_count does not match "
                    "decoded steps"
                )

        if "version_delta" in payload:
            version_delta = self._require_integer(
                payload,
                "version_delta",
                minimum=0,
            )

            if (
                version_delta
                != result.version_delta
            ):
                raise ValueError(
                    "result version_delta does not match "
                    "decoded contexts"
                )

        return result

    def _decode_context(
        self,
        payload: Mapping[str, object],
    ) -> LearningContext:
        cycle_id = self._require_string(
            payload,
            "cycle_id",
        )
        round_no = self._require_integer(
            payload,
            "round_no",
            minimum=1,
        )
        version = self._require_integer(
            payload,
            "version",
            minimum=1,
        )

        signals = self._optional_numeric_mapping(
            payload,
            "signals",
        )
        rewards = self._optional_numeric_mapping(
            payload,
            "rewards",
        )
        weights = self._optional_numeric_mapping(
            payload,
            "weights",
        )
        metadata = self._optional_scalar_mapping(
            payload,
            "metadata",
        )

        selected_policy = self._optional_string(
            payload,
            "selected_policy",
        )
        selected_arm = self._optional_string(
            payload,
            "selected_arm",
        )

        return LearningContext(
            cycle_id=cycle_id,
            round_no=round_no,
            version=version,
            signals=signals,
            rewards=rewards,
            selected_policy=selected_policy,
            selected_arm=selected_arm,
            weights=weights,
            metadata=metadata,
        )

    def _decode_steps(
        self,
        payload: Sequence[object],
    ) -> tuple[LearningCycleStep, ...]:
        steps: list[LearningCycleStep] = []

        for expected_index, item in enumerate(
            payload,
            start=1,
        ):
            step_payload = self._require_mapping(
                item,
                field_name=(
                    f"steps[{expected_index - 1}]"
                ),
            )

            step = LearningCycleStep(
                index=self._require_integer(
                    step_payload,
                    "index",
                    minimum=1,
                ),
                name=self._require_string(
                    step_payload,
                    "name",
                ),
                version_before=(
                    self._require_integer(
                        step_payload,
                        "version_before",
                        minimum=1,
                    )
                ),
                version_after=(
                    self._require_integer(
                        step_payload,
                        "version_after",
                        minimum=1,
                    )
                ),
                reward_key=self._require_string(
                    step_payload,
                    "reward_key",
                ),
            )

            if step.index != expected_index:
                raise ValueError(
                    "learning-cycle step indexes "
                    "must be contiguous"
                )

            steps.append(step)

        return tuple(steps)

    def _validate_identity_fields(
        self,
        *,
        payload: Mapping[str, object],
        snapshot: LearningCycleSnapshot,
    ) -> None:
        expected_fields = {
            "cycle_id": snapshot.cycle_id,
            "round_no": snapshot.round_no,
            "context_version": (
                snapshot.context_version
            ),
            "step_count": snapshot.step_count,
        }

        for field_name, expected in (
            expected_fields.items()
        ):
            if field_name not in payload:
                continue

            actual = payload[field_name]

            if actual != expected:
                raise ValueError(
                    f"{field_name} does not match "
                    "decoded snapshot"
                )

    @staticmethod
    def _parse_datetime(
        value: str,
    ) -> datetime:
        normalized = value.strip()

        if normalized.endswith("Z"):
            normalized = (
                normalized[:-1] + "+00:00"
            )

        try:
            parsed = datetime.fromisoformat(
                normalized
            )
        except ValueError as exc:
            raise ValueError(
                "created_at_utc must be a valid "
                "ISO-8601 datetime"
            ) from exc

        if (
            parsed.tzinfo is None
            or parsed.utcoffset() is None
        ):
            raise ValueError(
                "created_at_utc must be "
                "timezone-aware"
            )

        return parsed.astimezone(
            timezone.utc
        )

    @staticmethod
    def _require_mapping(
        value: object,
        *,
        field_name: str,
    ) -> Mapping[str, object]:
        if not isinstance(value, Mapping):
            raise TypeError(
                f"{field_name} must be a mapping"
            )

        for key in value:
            if not isinstance(key, str):
                raise TypeError(
                    f"{field_name} keys must be strings"
                )

        return value

    @classmethod
    def _require_mapping_field(
        cls,
        payload: Mapping[str, object],
        field_name: str,
    ) -> Mapping[str, object]:
        if field_name not in payload:
            raise ValueError(
                f"missing required field: {field_name}"
            )

        return cls._require_mapping(
            payload[field_name],
            field_name=field_name,
        )

    @staticmethod
    def _require_sequence_field(
        payload: Mapping[str, object],
        field_name: str,
    ) -> Sequence[object]:
        if field_name not in payload:
            raise ValueError(
                f"missing required field: {field_name}"
            )

        value = payload[field_name]

        if (
            isinstance(value, (str, bytes))
            or not isinstance(value, Sequence)
        ):
            raise TypeError(
                f"{field_name} must be a sequence"
            )

        return value

    @staticmethod
    def _require_string(
        payload: Mapping[str, object],
        field_name: str,
    ) -> str:
        if field_name not in payload:
            raise ValueError(
                f"missing required field: {field_name}"
            )

        value = payload[field_name]

        if not isinstance(value, str):
            raise TypeError(
                f"{field_name} must be a string"
            )

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                f"{field_name} must not be empty"
            )

        return normalized

    @staticmethod
    def _optional_string(
        payload: Mapping[str, object],
        field_name: str,
    ) -> str | None:
        if (
            field_name not in payload
            or payload[field_name] is None
        ):
            return None

        value = payload[field_name]

        if not isinstance(value, str):
            raise TypeError(
                f"{field_name} must be a string "
                "or None"
            )

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                f"{field_name} must not be empty"
            )

        return normalized

    @staticmethod
    def _require_integer(
        payload: Mapping[str, object],
        field_name: str,
        *,
        minimum: int,
    ) -> int:
        if field_name not in payload:
            raise ValueError(
                f"missing required field: {field_name}"
            )

        value = payload[field_name]

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise TypeError(
                f"{field_name} must be an integer"
            )

        if value < minimum:
            raise ValueError(
                f"{field_name} must be greater than "
                f"or equal to {minimum}"
            )

        return value

    @classmethod
    def _optional_numeric_mapping(
        cls,
        payload: Mapping[str, object],
        field_name: str,
    ) -> dict[str, float]:
        if (
            field_name not in payload
            or payload[field_name] is None
        ):
            return {}

        mapping = cls._require_mapping(
            payload[field_name],
            field_name=field_name,
        )

        result: dict[str, float] = {}

        for key, value in mapping.items():
            if (
                isinstance(value, bool)
                or not isinstance(
                    value,
                    (int, float),
                )
            ):
                raise TypeError(
                    f"{field_name}[{key}] must be "
                    "numeric"
                )

            result[key] = float(value)

        return result

    @classmethod
    def _optional_scalar_mapping(
        cls,
        payload: Mapping[str, object],
        field_name: str,
    ) -> dict[
        str,
        str | int | float | bool | None,
    ]:
        if (
            field_name not in payload
            or payload[field_name] is None
        ):
            return {}

        mapping = cls._require_mapping(
            payload[field_name],
            field_name=field_name,
        )

        result: dict[
            str,
            str | int | float | bool | None,
        ] = {}

        for key, value in mapping.items():
            if not (
                value is None
                or isinstance(
                    value,
                    (str, int, float, bool),
                )
            ):
                raise TypeError(
                    f"{field_name}[{key}] must be "
                    "a scalar value"
                )

            result[key] = value

        return result
