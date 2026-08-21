from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lrp.contracts.exceptions import ContractError
from lrp.evaluation.topk_prediction_source_adapter import (
    TopKPredictionSourceRecord,
)
from lrp.io.draws import (
    HistoryRow,
    history_until_round,
)
from lrp.pipelines.models import PredictionResult


def _require_text(
    value: Any,
    *,
    name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise ContractError(
            f"{name} must be a non-empty string"
        )

    return value


def _require_optional_text(
    value: Any,
    *,
    name: str,
) -> str | None:
    if value is None:
        return None

    return _require_text(
        value,
        name=name,
    )


def _prediction_round(
    prediction_result: PredictionResult,
) -> int:
    if not isinstance(
        prediction_result,
        PredictionResult,
    ):
        raise ContractError(
            "prediction_result must be PredictionResult"
        )

    try:
        round_no = prediction_result.request.round_no

    except Exception as exc:
        raise ContractError(
            "prediction_result must expose a valid prediction round"
        ) from exc

    if (
        not isinstance(round_no, int)
        or isinstance(round_no, bool)
        or round_no < 1
    ):
        raise ContractError(
            "prediction round must be a positive integer"
        )

    return round_no


def _history_rounds(
    history_rows: Any,
    *,
    prediction_round: int,
) -> tuple[int, ...]:
    if not isinstance(
        history_rows,
        tuple,
    ):
        raise ContractError(
            "history_rows must be a tuple"
        )

    if not history_rows:
        raise ContractError(
            "history_rows must not be empty"
        )

    seen: set[int] = set()

    for row in history_rows:
        if not isinstance(
            row,
            HistoryRow,
        ):
            raise ContractError(
                "history_rows must contain only HistoryRow"
            )

        round_no = row.round_no

        if (
            not isinstance(round_no, int)
            or isinstance(round_no, bool)
            or round_no < 1
        ):
            raise ContractError(
                "history round must be a positive integer"
            )

        if round_no in seen:
            raise ContractError(
                "history_rows must not contain duplicate rounds"
            )

        seen.add(
            round_no
        )

        if round_no == prediction_round:
            raise ContractError(
                "history_rows must not contain prediction round"
            )

        if round_no > prediction_round:
            raise ContractError(
                "history_rows must not contain future rounds"
            )

    try:
        prior_rows = history_until_round(
            history_rows,
            target_round=prediction_round,
        )

    except Exception as exc:
        raise ContractError(
            "unable to derive prior prediction history"
        ) from exc

    rounds = tuple(
        row.round_no
        for row in prior_rows
    )

    if not rounds:
        raise ContractError(
            "no history exists before prediction round"
        )

    normalized = tuple(
        sorted(
            rounds
        )
    )

    if len(
        set(normalized)
    ) != len(normalized):
        raise ContractError(
            "derived history rounds must be unique"
        )

    if any(
        round_no >= prediction_round
        for round_no in normalized
    ):
        raise ContractError(
            "derived history contains current or future round"
        )

    if normalized != tuple(
        sorted(
            normalized
        )
    ):
        raise ContractError(
            "derived history rounds must be ascending"
        )

    return normalized


@dataclass(
    frozen=True,
    slots=True,
)
class TopKLivePredictionBindingRequest:
    prediction_result: PredictionResult
    history_rows: tuple[HistoryRow, ...]
    model_name: str
    regime_id: str | None = None
    strategy_name: str | None = None

    def __post_init__(
        self,
    ) -> None:
        if not isinstance(
            self.prediction_result,
            PredictionResult,
        ):
            raise ContractError(
                "prediction_result must be PredictionResult"
            )

        if not isinstance(
            self.history_rows,
            tuple,
        ):
            raise ContractError(
                "history_rows must be a tuple"
            )

        _require_text(
            self.model_name,
            name="model_name",
        )

        _require_optional_text(
            self.regime_id,
            name="regime_id",
        )

        _require_optional_text(
            self.strategy_name,
            name="strategy_name",
        )


@dataclass(
    frozen=True,
    slots=True,
)
class TopKLivePredictionBindingResult:
    source: TopKPredictionSourceRecord
    prediction_round: int
    history_rounds: tuple[int, ...]
    model_name: str

    def __post_init__(
        self,
    ) -> None:
        if not isinstance(
            self.source,
            TopKPredictionSourceRecord,
        ):
            raise ContractError(
                "source must be TopKPredictionSourceRecord"
            )

        if (
            not isinstance(
                self.prediction_round,
                int,
            )
            or isinstance(
                self.prediction_round,
                bool,
            )
            or self.prediction_round < 1
        ):
            raise ContractError(
                "prediction_round must be a positive integer"
            )

        if (
            not isinstance(
                self.history_rounds,
                tuple,
            )
            or not self.history_rounds
        ):
            raise ContractError(
                "history_rounds must be a non-empty tuple"
            )

        if any(
            (
                not isinstance(
                    round_no,
                    int,
                )
                or isinstance(
                    round_no,
                    bool,
                )
                or round_no < 1
            )
            for round_no in self.history_rounds
        ):
            raise ContractError(
                "history_rounds must contain positive integers"
            )

        if (
            tuple(
                sorted(
                    self.history_rounds
                )
            )
            != self.history_rounds
        ):
            raise ContractError(
                "history_rounds must be ascending"
            )

        if len(
            set(
                self.history_rounds
            )
        ) != len(
            self.history_rounds
        ):
            raise ContractError(
                "history_rounds must be unique"
            )

        if any(
            round_no >= self.prediction_round
            for round_no in self.history_rounds
        ):
            raise ContractError(
                "history_rounds must precede prediction_round"
            )

        _require_text(
            self.model_name,
            name="model_name",
        )


class TopKLivePredictionBinder:
    def bind(
        self,
        *,
        request: TopKLivePredictionBindingRequest,
    ) -> TopKLivePredictionBindingResult:
        if not isinstance(
            request,
            TopKLivePredictionBindingRequest,
        ):
            raise ContractError(
                "request must be TopKLivePredictionBindingRequest"
            )

        if not isinstance(
            request.prediction_result,
            PredictionResult,
        ):
            raise ContractError(
                "prediction_result must be PredictionResult"
            )

        model_name = _require_text(
            request.model_name,
            name="model_name",
        )

        regime_id = _require_optional_text(
            request.regime_id,
            name="regime_id",
        )

        strategy_name = _require_optional_text(
            request.strategy_name,
            name="strategy_name",
        )

        prediction_round = _prediction_round(
            request.prediction_result
        )

        history_rounds = _history_rounds(
            request.history_rows,
            prediction_round=prediction_round,
        )

        try:
            source = TopKPredictionSourceRecord(
                prediction_result=request.prediction_result,
                model_name=model_name,
                history_rounds=history_rounds,
                regime_id=regime_id,
                strategy_name=strategy_name,
            )

        except ContractError:
            raise

        except Exception as exc:
            raise ContractError(
                "downstream source contract rejected binding"
            ) from exc

        return TopKLivePredictionBindingResult(
            source=source,
            prediction_round=prediction_round,
            history_rounds=history_rounds,
            model_name=model_name,
        )
