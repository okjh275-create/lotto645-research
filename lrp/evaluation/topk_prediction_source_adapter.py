"""Translate canonical prediction results into replay predictions.

Project AD owns only the read-only source-binding boundary between the
prediction-domain result and the replay-domain prediction contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from lrp.contracts.exceptions import ContractError
from lrp.evaluation.topk_replay_adapter import TopKReplayPrediction
from lrp.pipelines.models import PredictionResult


@dataclass(frozen=True, slots=True)
class TopKPredictionSourceRecord:
    """Immutable provenance required to bind one prediction result to replay."""

    prediction_result: PredictionResult
    model_name: str
    history_rounds: tuple[int, ...]
    regime_id: str | None = None
    strategy_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.prediction_result,
            PredictionResult,
        ):
            raise ContractError(
                "prediction_result must be a PredictionResult"
            )

        model_name = _required_text(
            self.model_name,
            field_name="model_name",
        )

        regime_id = _optional_text(
            self.regime_id,
            field_name="regime_id",
        )

        strategy_name = _optional_text(
            self.strategy_name,
            field_name="strategy_name",
        )

        history_rounds = _history_rounds(
            self.history_rounds,
            target_round=self.prediction_result.request.round_no,
        )

        object.__setattr__(
            self,
            "model_name",
            model_name,
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
            "history_rounds",
            history_rounds,
        )


class TopKPredictionSourceAdapter:
    """Convert one explicit source record into TopKReplayPrediction."""

    def adapt(
        self,
        *,
        source: TopKPredictionSourceRecord,
    ) -> TopKReplayPrediction:
        if not isinstance(
            source,
            TopKPredictionSourceRecord,
        ):
            raise ContractError(
                "source must be a TopKPredictionSourceRecord"
            )

        prediction_result = source.prediction_result

        target_round = prediction_result.request.round_no

        selected = self._selected_predictions(
            prediction_result
        )

        expected_count = prediction_result.request.top_k

        if len(selected) != expected_count:
            raise ContractError(
                "selected prediction count must equal request.top_k"
            )

        predictions = tuple(
            self._normalize_prediction_numbers(
                item
            )
            for item in selected
        )

        if len(
            set(predictions)
        ) != len(predictions):
            raise ContractError(
                "duplicate prediction sets are forbidden"
            )

        return TopKReplayPrediction(
            round_no=target_round,
            history_rounds=source.history_rounds,
            predictions=predictions,
            model_name=source.model_name,
            regime_id=source.regime_id,
            strategy_name=source.strategy_name,
        )

    @staticmethod
    def _selected_predictions(
        prediction_result: PredictionResult,
    ) -> tuple[object, ...]:
        diversity = prediction_result.diversity

        try:
            selected = getattr(
                diversity,
                "selected",
            )
        except Exception as exc:
            raise ContractError(
                "diversity selected output is unavailable"
            ) from exc

        if selected is None:
            raise ContractError(
                "diversity selected output is unavailable"
            )

        if isinstance(
            selected,
            (
                str,
                bytes,
                bytearray,
            ),
        ):
            raise ContractError(
                "diversity selected output must be iterable"
            )

        try:
            return tuple(
                selected
            )
        except TypeError as exc:
            raise ContractError(
                "diversity selected output must be iterable"
            ) from exc

    @classmethod
    def _normalize_prediction_numbers(
        cls,
        candidate: object,
    ) -> tuple[int, ...]:
        numbers = cls._candidate_numbers(
            candidate
        )

        if isinstance(
            numbers,
            (
                str,
                bytes,
                bytearray,
            ),
        ):
            raise ContractError(
                "candidate numbers must be iterable integers"
            )

        try:
            values = tuple(
                numbers
            )
        except TypeError as exc:
            raise ContractError(
                "candidate numbers must be iterable"
            ) from exc

        if len(values) != 6:
            raise ContractError(
                "prediction set must contain exactly six numbers"
            )

        normalized: list[int] = []

        for value in values:
            if (
                isinstance(
                    value,
                    bool,
                )
                or not isinstance(
                    value,
                    int,
                )
            ):
                raise ContractError(
                    "prediction numbers must be integers"
                )

            if not 1 <= value <= 45:
                raise ContractError(
                    "prediction numbers must be between 1 and 45"
                )

            normalized.append(
                value
            )

        if len(
            set(normalized)
        ) != 6:
            raise ContractError(
                "prediction set numbers must be distinct"
            )

        return tuple(
            sorted(
                normalized
            )
        )

    @staticmethod
    def _candidate_numbers(
        candidate: object,
    ) -> Iterable[Any]:
        if candidate is None:
            raise ContractError(
                "candidate numbers cannot be located"
            )

        if hasattr(
            candidate,
            "numbers",
        ):
            try:
                value = getattr(
                    candidate,
                    "numbers",
                )
            except Exception as exc:
                raise ContractError(
                    "candidate numbers cannot be located"
                ) from exc

            if value is None:
                raise ContractError(
                    "candidate numbers cannot be located"
                )

            return value

        if isinstance(
            candidate,
            dict,
        ):
            if "numbers" not in candidate:
                raise ContractError(
                    "candidate numbers cannot be located"
                )

            value = candidate[
                "numbers"
            ]

            if value is None:
                raise ContractError(
                    "candidate numbers cannot be located"
                )

            return value

        raise ContractError(
            "candidate numbers cannot be located"
        )


def _required_text(
    value: object,
    *,
    field_name: str,
) -> str:
    if not isinstance(
        value,
        str,
    ):
        raise ContractError(
            f"{field_name} must be a string"
        )

    normalized = value.strip()

    if not normalized:
        raise ContractError(
            f"{field_name} must not be blank"
        )

    return normalized


def _optional_text(
    value: object,
    *,
    field_name: str,
) -> str | None:
    if value is None:
        return None

    return _required_text(
        value,
        field_name=field_name,
    )


def _history_rounds(
    value: object,
    *,
    target_round: int,
) -> tuple[int, ...]:
    if isinstance(
        value,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        raise ContractError(
            "history_rounds must be an iterable of integers"
        )

    try:
        rounds = tuple(
            value  # type: ignore[arg-type]
        )
    except TypeError as exc:
        raise ContractError(
            "history_rounds must be an iterable of integers"
        ) from exc

    if not rounds:
        raise ContractError(
            "history_rounds must not be empty"
        )

    normalized: list[int] = []

    for round_no in rounds:
        if (
            isinstance(
                round_no,
                bool,
            )
            or not isinstance(
                round_no,
                int,
            )
        ):
            raise ContractError(
                "history_rounds must contain integers"
            )

        if round_no <= 0:
            raise ContractError(
                "history rounds must be positive"
            )

        if round_no >= target_round:
            raise ContractError(
                "history rounds must precede target round"
            )

        normalized.append(
            round_no
        )

    normalized_tuple = tuple(
        normalized
    )

    if len(
        set(
            normalized_tuple
        )
    ) != len(
        normalized_tuple
    ):
        raise ContractError(
            "history_rounds must not contain duplicates"
        )

    if normalized_tuple != tuple(
        sorted(
            normalized_tuple
        )
    ):
        raise ContractError(
            "history_rounds must be strictly increasing"
        )

    return normalized_tuple


__all__ = (
    "TopKPredictionSourceRecord",
    "TopKPredictionSourceAdapter",
)
