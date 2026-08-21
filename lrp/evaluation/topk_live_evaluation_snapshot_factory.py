from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lrp.contracts.exceptions import ContractError
from lrp.io.draws import HistoryRow
from lrp.pipelines.models import PredictionResult
from lrp.evaluation.topk_live_evaluation_source_snapshot import (
    TopKLiveEvaluationSourceSnapshot,
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


def _normalize_sha256(
    value: Any,
) -> str:
    normalized = _normalize_required_text(
        value,
        field_name="source_artifact_sha256",
    )

    if (
        len(normalized) != 64
        or any(
            character not in "0123456789abcdef"
            for character in normalized
        )
    ):
        raise ContractError(
            "source_artifact_sha256 must be "
            "64 lowercase hexadecimal characters"
        )

    return normalized


def _normalize_history_rows(
    value: Any,
    *,
    prediction_round: int,
) -> tuple[HistoryRow, ...]:
    if not isinstance(value, tuple):
        raise ContractError(
            "history_rows must be a tuple"
        )

    if not value:
        raise ContractError(
            "history_rows must not be empty"
        )

    rounds: list[int] = []

    for row in value:
        if not isinstance(row, HistoryRow):
            raise ContractError(
                "history_rows must contain HistoryRow"
            )

        if row.round_no >= prediction_round:
            raise ContractError(
                "history_rows must contain only prior rounds"
            )

        rounds.append(row.round_no)

    if len(set(rounds)) != len(rounds):
        raise ContractError(
            "history_rows must contain unique rounds"
        )

    return value


@dataclass(frozen=True)
class TopKLiveEvaluationSnapshotBuildRequest:
    prediction_result: PredictionResult
    history_rows: tuple[HistoryRow, ...]
    model_name: str
    source_artifact_sha256: str
    regime_id: str | None = None
    strategy_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.prediction_result,
            PredictionResult,
        ):
            raise ContractError(
                "prediction_result must be PredictionResult"
            )

        try:
            prediction_round = (
                self.prediction_result
                .generation
                .request
                .round_no
            )
        except (AttributeError, TypeError):
            raise ContractError(
                "prediction_result must expose prediction round"
            ) from None

        history_rows = _normalize_history_rows(
            self.history_rows,
            prediction_round=prediction_round,
        )

        model_name = _normalize_required_text(
            self.model_name,
            field_name="model_name",
        )

        source_artifact_sha256 = _normalize_sha256(
            self.source_artifact_sha256
        )

        regime_id = _normalize_optional_text(
            self.regime_id,
            field_name="regime_id",
        )

        strategy_name = _normalize_optional_text(
            self.strategy_name,
            field_name="strategy_name",
        )

        object.__setattr__(
            self,
            "history_rows",
            history_rows,
        )
        object.__setattr__(
            self,
            "model_name",
            model_name,
        )
        object.__setattr__(
            self,
            "source_artifact_sha256",
            source_artifact_sha256,
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


class TopKLiveEvaluationSnapshotFactory:
    def build(
        self,
        *,
        request: TopKLiveEvaluationSnapshotBuildRequest,
    ) -> TopKLiveEvaluationSourceSnapshot:
        if not isinstance(
            request,
            TopKLiveEvaluationSnapshotBuildRequest,
        ):
            raise ContractError(
                "request must be "
                "TopKLiveEvaluationSnapshotBuildRequest"
            )

        prediction_result = request.prediction_result

        try:
            prediction_request = (
                prediction_result
                .generation
                .request
            )
            round_no = prediction_request.round_no
            top_k = prediction_request.top_k
        except (AttributeError, TypeError):
            raise ContractError(
                "prediction_result must expose request metadata"
            ) from None

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
            (str, bytes, bytearray),
        ):
            raise ContractError(
                "diversity selected output must be iterable"
            )

        try:
            selected_items = tuple(selected)
        except TypeError as exc:
            raise ContractError(
                "diversity selected output must be iterable"
            ) from exc

        if len(selected_items) != top_k:
            raise ContractError(
                "selected prediction count must equal request.top_k"
            )

        selected_sets: list[tuple[int, ...]] = []

        for item in selected_items:
            try:
                numbers = getattr(
                    item,
                    "numbers",
                )
            except Exception as exc:
                raise ContractError(
                    "selected prediction numbers are unavailable"
                ) from exc

            if numbers is None:
                raise ContractError(
                    "selected prediction numbers are unavailable"
                )

            try:
                selected_sets.append(
                    tuple(numbers)
                )
            except TypeError as exc:
                raise ContractError(
                    "selected prediction numbers must be iterable"
                ) from exc

        history_rounds = tuple(
            row.round_no
            for row in request.history_rows
        )

        return TopKLiveEvaluationSourceSnapshot(
            schema_version="1.0",
            round_no=round_no,
            top_k=top_k,
            selected_sets=tuple(selected_sets),
            model_name=request.model_name,
            history_rounds=history_rounds,
            regime_id=request.regime_id,
            strategy_name=request.strategy_name,
            generated_at_kst=prediction_result.generated_at_kst,
            source_artifact_sha256=(
                request.source_artifact_sha256
            ),
        )