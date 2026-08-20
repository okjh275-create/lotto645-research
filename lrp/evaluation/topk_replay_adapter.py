from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from lrp.contracts import ContractError


_MIN_LOTTO_NUMBER = 1
_MAX_LOTTO_NUMBER = 45
_LOTTO_SET_SIZE = 6
_MIN_PREDICTION_COUNT = 10


def _require_round_no(
    value: Any,
    *,
    field: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
    ):
        raise ContractError(
            f"{field} must be a positive integer"
        )

    return value


def _require_optional_string(
    value: Any,
    *,
    field: str,
) -> str | None:
    if value is None:
        return None

    if not isinstance(
        value,
        str,
    ):
        raise ContractError(
            f"{field} must be str or None"
        )

    if not value.strip():
        raise ContractError(
            f"{field} must be non-empty when provided"
        )

    return value


def _require_model_name(
    value: Any,
) -> str:
    if (
        not isinstance(
            value,
            str,
        )
        or not value.strip()
    ):
        raise ContractError(
            "model_name must be a non-empty string"
        )

    return value


def _canonical_number_set(
    values: Iterable[Any],
    *,
    field: str,
) -> tuple[int, ...]:
    try:
        numbers = tuple(
            values
        )
    except TypeError as exc:
        raise ContractError(
            f"{field} must be iterable"
        ) from exc

    if len(numbers) != _LOTTO_SET_SIZE:
        raise ContractError(
            f"{field} must contain exactly six numbers"
        )

    for number in numbers:
        if (
            isinstance(number, bool)
            or not isinstance(number, int)
        ):
            raise ContractError(
                f"{field} must contain integers"
            )

        if (
            number < _MIN_LOTTO_NUMBER
            or number > _MAX_LOTTO_NUMBER
        ):
            raise ContractError(
                f"{field} number out of range"
            )

    if len(set(numbers)) != len(numbers):
        raise ContractError(
            f"{field} must not contain duplicate numbers"
        )

    return tuple(
        sorted(numbers)
    )


def _canonical_predictions(
    values: Iterable[Iterable[Any]],
) -> tuple[tuple[int, ...], ...]:
    try:
        rows = tuple(
            values
        )
    except TypeError as exc:
        raise ContractError(
            "predictions must be iterable"
        ) from exc

    if len(rows) < _MIN_PREDICTION_COUNT:
        raise ContractError(
            "predictions must contain at least ten sets"
        )

    return tuple(
        _canonical_number_set(
            row,
            field="prediction",
        )
        for row in rows
    )


def _validated_history_rounds(
    values: Iterable[Any],
    *,
    prediction_round: int,
) -> tuple[int, ...]:
    try:
        rounds = tuple(
            values
        )
    except TypeError as exc:
        raise ContractError(
            "history_rounds must be iterable"
        ) from exc

    if not rounds:
        raise ContractError(
            "history_rounds must be non-empty"
        )

    normalized: list[int] = []

    for round_no in rounds:
        current = _require_round_no(
            round_no,
            field="history round",
        )

        if current >= prediction_round:
            raise ContractError(
                "history rounds must be strictly before prediction round"
            )

        normalized.append(
            current
        )

    result = tuple(
        normalized
    )

    if len(set(result)) != len(result):
        raise ContractError(
            "history_rounds must not contain duplicates"
        )

    if result != tuple(
        sorted(result)
    ):
        raise ContractError(
            "history_rounds must be strictly ascending"
        )

    return result


@dataclass(
    frozen=True,
)
class TopKReplayPrediction:
    round_no: int
    history_rounds: tuple[int, ...]
    predictions: tuple[tuple[int, ...], ...]
    model_name: str
    regime_id: str | None = None
    strategy_name: str | None = None

    def __post_init__(
        self,
    ) -> None:
        round_no = _require_round_no(
            self.round_no,
            field="round_no",
        )

        object.__setattr__(
            self,
            "round_no",
            round_no,
        )

        object.__setattr__(
            self,
            "history_rounds",
            _validated_history_rounds(
                self.history_rounds,
                prediction_round=round_no,
            ),
        )

        object.__setattr__(
            self,
            "predictions",
            _canonical_predictions(
                self.predictions
            ),
        )

        object.__setattr__(
            self,
            "model_name",
            _require_model_name(
                self.model_name
            ),
        )

        object.__setattr__(
            self,
            "regime_id",
            _require_optional_string(
                self.regime_id,
                field="regime_id",
            ),
        )

        object.__setattr__(
            self,
            "strategy_name",
            _require_optional_string(
                self.strategy_name,
                field="strategy_name",
            ),
        )


@dataclass(
    frozen=True,
)
class TopKReplayRow:
    round_no: int
    history_rounds: tuple[int, ...]
    actual_numbers: tuple[int, ...]
    predictions: tuple[tuple[int, ...], ...]
    model_name: str
    regime_id: str | None = None
    strategy_name: str | None = None

    def __post_init__(
        self,
    ) -> None:
        round_no = _require_round_no(
            self.round_no,
            field="round_no",
        )

        object.__setattr__(
            self,
            "round_no",
            round_no,
        )

        object.__setattr__(
            self,
            "history_rounds",
            _validated_history_rounds(
                self.history_rounds,
                prediction_round=round_no,
            ),
        )

        object.__setattr__(
            self,
            "actual_numbers",
            _canonical_number_set(
                self.actual_numbers,
                field="actual_numbers",
            ),
        )

        object.__setattr__(
            self,
            "predictions",
            _canonical_predictions(
                self.predictions
            ),
        )

        object.__setattr__(
            self,
            "model_name",
            _require_model_name(
                self.model_name
            ),
        )

        object.__setattr__(
            self,
            "regime_id",
            _require_optional_string(
                self.regime_id,
                field="regime_id",
            ),
        )

        object.__setattr__(
            self,
            "strategy_name",
            _require_optional_string(
                self.strategy_name,
                field="strategy_name",
            ),
        )


class TopKReplayBaselineProvider:
    def __init__(
        self,
        rows: Iterable[TopKReplayRow],
    ) -> None:
        index: dict[
            int,
            TopKReplayRow,
        ] = {}

        for row in tuple(
            rows
        ):
            if not isinstance(
                row,
                TopKReplayRow,
            ):
                raise ContractError(
                    "baseline rows must be TopKReplayRow instances"
                )

            if row.round_no in index:
                raise ContractError(
                    "duplicate baseline round"
                )

            index[
                row.round_no
            ] = row

        self._rows = index

    def get(
        self,
        round_no: int,
    ) -> TopKReplayRow:
        normalized = _require_round_no(
            round_no,
            field="round_no",
        )

        try:
            return self._rows[
                normalized
            ]

        except KeyError as exc:
            raise ContractError(
                "baseline round missing"
            ) from exc


class TopKReplayAdapter:
    def adapt(
        self,
        *,
        prediction_rows: Iterable[TopKReplayPrediction],
        actual_draws: Iterable[Any],
    ) -> tuple[TopKReplayRow, ...]:
        prediction_index: dict[
            int,
            TopKReplayPrediction,
        ] = {}

        for prediction in tuple(
            prediction_rows
        ):
            if not isinstance(
                prediction,
                TopKReplayPrediction,
            ):
                raise ContractError(
                    "prediction_rows must contain TopKReplayPrediction"
                )

            if (
                prediction.round_no
                in prediction_index
            ):
                raise ContractError(
                    "duplicate prediction round"
                )

            prediction_index[
                prediction.round_no
            ] = prediction

        draw_index: dict[
            int,
            tuple[int, ...],
        ] = {}

        for draw in tuple(
            actual_draws
        ):
            try:
                round_value = getattr(
                    draw,
                    "round_no",
                )

                numbers_value = getattr(
                    draw,
                    "numbers",
                )

            except AttributeError as exc:
                raise ContractError(
                    "actual draw must expose round_no and numbers"
                ) from exc

            draw_round = _require_round_no(
                round_value,
                field="actual draw round_no",
            )

            if draw_round in draw_index:
                raise ContractError(
                    "duplicate actual draw round"
                )

            draw_index[
                draw_round
            ] = _canonical_number_set(
                numbers_value,
                field="actual draw numbers",
            )

        rows: list[
            TopKReplayRow
        ] = []

        for round_no in sorted(
            prediction_index
        ):
            prediction = prediction_index[
                round_no
            ]

            if round_no not in draw_index:
                raise ContractError(
                    "actual draw missing for prediction round"
                )

            rows.append(
                TopKReplayRow(
                    round_no=prediction.round_no,
                    history_rounds=prediction.history_rounds,
                    actual_numbers=draw_index[
                        round_no
                    ],
                    predictions=prediction.predictions,
                    model_name=prediction.model_name,
                    regime_id=prediction.regime_id,
                    strategy_name=prediction.strategy_name,
                )
            )

        return tuple(
            rows
        )

    def baseline_provider(
        self,
        baseline_rows: Iterable[TopKReplayRow],
    ) -> TopKReplayBaselineProvider:
        return TopKReplayBaselineProvider(
            baseline_rows
        )
