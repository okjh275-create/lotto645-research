from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable

from lrp.contracts import ContractError


_SUPPORTED_K = (
    3,
    5,
    10,
)


def _require_int(
    value: object,
    *,
    name: str,
    minimum: int | None = None,
) -> int:
    if (
        not isinstance(
            value,
            int,
        )
        or isinstance(
            value,
            bool,
        )
    ):
        raise ContractError(
            f"{name} must be an integer"
        )

    if (
        minimum is not None
        and value < minimum
    ):
        raise ContractError(
            f"{name} must be >= {minimum}"
        )

    return value


def _require_float(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        not isinstance(
            value,
            (int, float),
        )
        or isinstance(
            value,
            bool,
        )
    ):
        raise ContractError(
            f"{name} must be numeric"
        )

    result = float(
        value
    )

    if not isfinite(
        result
    ):
        raise ContractError(
            f"{name} must be finite"
        )

    if (
        minimum is not None
        and result < minimum
    ):
        raise ContractError(
            f"{name} must be >= {minimum}"
        )

    if (
        maximum is not None
        and result > maximum
    ):
        raise ContractError(
            f"{name} must be <= {maximum}"
        )

    return result


def _validate_numbers(
    values: Iterable[object],
    *,
    name: str,
) -> tuple[int, ...]:
    try:
        numbers = tuple(
            values
        )

    except TypeError as exc:
        raise ContractError(
            f"{name} must be iterable"
        ) from exc

    if len(
        numbers
    ) != 6:
        raise ContractError(
            f"{name} must contain six numbers"
        )

    normalized = tuple(
        _require_int(
            value,
            name=name,
            minimum=1,
        )
        for value in numbers
    )

    if any(
        value > 45
        for value in normalized
    ):
        raise ContractError(
            f"{name} numbers must be <= 45"
        )

    if len(
        set(
            normalized
        )
    ) != 6:
        raise ContractError(
            f"{name} numbers must be unique"
        )

    return normalized


@dataclass(
    frozen=True
)
class HitDistribution:
    hit_0: int
    hit_1: int
    hit_2: int
    hit_3: int
    hit_4: int
    hit_5: int
    hit_6: int

    def __post_init__(
        self,
    ) -> None:
        for name in (
            "hit_0",
            "hit_1",
            "hit_2",
            "hit_3",
            "hit_4",
            "hit_5",
            "hit_6",
        ):
            _require_int(
                getattr(
                    self,
                    name,
                ),
                name=name,
                minimum=0,
            )

    @property
    def total_count(
        self,
    ) -> int:
        return (
            self.hit_0
            + self.hit_1
            + self.hit_2
            + self.hit_3
            + self.hit_4
            + self.hit_5
            + self.hit_6
        )

    @property
    def at_least_3_count(
        self,
    ) -> int:
        return (
            self.hit_3
            + self.hit_4
            + self.hit_5
            + self.hit_6
        )

    @property
    def at_least_4_count(
        self,
    ) -> int:
        return (
            self.hit_4
            + self.hit_5
            + self.hit_6
        )

    @property
    def at_least_5_count(
        self,
    ) -> int:
        return (
            self.hit_5
            + self.hit_6
        )

    @property
    def six_hit_count(
        self,
    ) -> int:
        return self.hit_6

    def _rate(
        self,
        count: int,
    ) -> float:
        if self.total_count == 0:
            return 0.0

        return (
            count
            / self.total_count
        )

    @property
    def at_least_3_rate(
        self,
    ) -> float:
        return self._rate(
            self.at_least_3_count
        )

    @property
    def at_least_4_rate(
        self,
    ) -> float:
        return self._rate(
            self.at_least_4_count
        )

    @property
    def at_least_5_rate(
        self,
    ) -> float:
        return self._rate(
            self.at_least_5_count
        )

    @property
    def six_hit_rate(
        self,
    ) -> float:
        return self._rate(
            self.six_hit_count
        )

    def as_dict(
        self,
    ) -> dict[str, object]:
        return {
            "hit_0":
                self.hit_0,

            "hit_1":
                self.hit_1,

            "hit_2":
                self.hit_2,

            "hit_3":
                self.hit_3,

            "hit_4":
                self.hit_4,

            "hit_5":
                self.hit_5,

            "hit_6":
                self.hit_6,

            "total_count":
                self.total_count,

            "at_least_3_count":
                self.at_least_3_count,

            "at_least_4_count":
                self.at_least_4_count,

            "at_least_5_count":
                self.at_least_5_count,

            "six_hit_count":
                self.six_hit_count,

            "at_least_3_rate":
                self.at_least_3_rate,

            "at_least_4_rate":
                self.at_least_4_rate,

            "at_least_5_rate":
                self.at_least_5_rate,

            "six_hit_rate":
                self.six_hit_rate,
        }


@dataclass(
    frozen=True
)
class TopKEvaluation:
    k: int
    round_count: int
    set_count: int
    mean_best_hits: float
    mean_set_hits: float
    best_hit_distribution: HitDistribution
    set_hit_distribution: HitDistribution
    baseline_delta_mean_best_hits: float
    baseline_delta_3plus_rate: float
    baseline_delta_4plus_rate: float

    def __post_init__(
        self,
    ) -> None:
        if self.k not in _SUPPORTED_K:
            raise ContractError(
                "unsupported Top-K value"
            )

        _require_int(
            self.round_count,
            name="round_count",
            minimum=1,
        )

        _require_int(
            self.set_count,
            name="set_count",
            minimum=1,
        )

        _require_float(
            self.mean_best_hits,
            name="mean_best_hits",
            minimum=0.0,
            maximum=6.0,
        )

        _require_float(
            self.mean_set_hits,
            name="mean_set_hits",
            minimum=0.0,
            maximum=6.0,
        )

        if not isinstance(
            self.best_hit_distribution,
            HitDistribution,
        ):
            raise ContractError(
                "best_hit_distribution must be HitDistribution"
            )

        if not isinstance(
            self.set_hit_distribution,
            HitDistribution,
        ):
            raise ContractError(
                "set_hit_distribution must be HitDistribution"
            )

        _require_float(
            self.baseline_delta_mean_best_hits,
            name="baseline_delta_mean_best_hits",
        )

        _require_float(
            self.baseline_delta_3plus_rate,
            name="baseline_delta_3plus_rate",
        )

        _require_float(
            self.baseline_delta_4plus_rate,
            name="baseline_delta_4plus_rate",
        )

    def as_dict(
        self,
    ) -> dict[str, object]:
        return {
            "k":
                self.k,

            "round_count":
                self.round_count,

            "set_count":
                self.set_count,

            "mean_best_hits":
                self.mean_best_hits,

            "mean_set_hits":
                self.mean_set_hits,

            "best_hit_distribution":
                self.best_hit_distribution.as_dict(),

            "set_hit_distribution":
                self.set_hit_distribution.as_dict(),

            "baseline_delta_mean_best_hits":
                self.baseline_delta_mean_best_hits,

            "baseline_delta_3plus_rate":
                self.baseline_delta_3plus_rate,

            "baseline_delta_4plus_rate":
                self.baseline_delta_4plus_rate,
        }


@dataclass(
    frozen=True
)
class WalkForwardRoundEvaluation:
    round_no: int
    history_end_round: int
    actual_numbers: tuple[int, ...]
    model_name: str
    regime_id: str | None
    strategy_name: str | None
    top3: TopKEvaluation
    top5: TopKEvaluation
    top10: TopKEvaluation

    def __post_init__(
        self,
    ) -> None:
        _require_int(
            self.round_no,
            name="round_no",
            minimum=1,
        )

        _require_int(
            self.history_end_round,
            name="history_end_round",
            minimum=1,
        )

        if (
            self.history_end_round
            >= self.round_no
        ):
            raise ContractError(
                "history_end_round must be less than round_no"
            )

        actual = _validate_numbers(
            self.actual_numbers,
            name="actual_numbers",
        )

        object.__setattr__(
            self,
            "actual_numbers",
            actual,
        )

        if (
            not isinstance(
                self.model_name,
                str,
            )
            or not self.model_name
        ):
            raise ContractError(
                "model_name must be non-empty"
            )

        if self.top3.k != 3:
            raise ContractError(
                "top3 must have k=3"
            )

        if self.top5.k != 5:
            raise ContractError(
                "top5 must have k=5"
            )

        if self.top10.k != 10:
            raise ContractError(
                "top10 must have k=10"
            )

    def as_dict(
        self,
    ) -> dict[str, object]:
        return {
            "round_no":
                self.round_no,

            "history_end_round":
                self.history_end_round,

            "actual_numbers":
                list(
                    self.actual_numbers
                ),

            "model_name":
                self.model_name,

            "regime_id":
                self.regime_id,

            "strategy_name":
                self.strategy_name,

            "top3":
                self.top3.as_dict(),

            "top5":
                self.top5.as_dict(),

            "top10":
                self.top10.as_dict(),
        }


@dataclass(
    frozen=True
)
class WalkForwardSlice:
    name: str
    round_count: int
    top3: TopKEvaluation
    top5: TopKEvaluation
    top10: TopKEvaluation

    def as_dict(
        self,
    ) -> dict[str, object]:
        return {
            "name":
                self.name,

            "round_count":
                self.round_count,

            "top3":
                self.top3.as_dict(),

            "top5":
                self.top5.as_dict(),

            "top10":
                self.top10.as_dict(),
        }


@dataclass(
    frozen=True
)
class WalkForwardEvaluation:
    window: Any
    rounds: tuple[
        WalkForwardRoundEvaluation,
        ...
    ]
    top3: TopKEvaluation
    top5: TopKEvaluation
    top10: TopKEvaluation
    model_name: str
    regime_slices: tuple[
        WalkForwardSlice,
        ...
    ]
    strategy_slices: tuple[
        WalkForwardSlice,
        ...
    ]

    def __post_init__(
        self,
    ) -> None:
        if not self.rounds:
            raise ContractError(
                "walk-forward rounds must not be empty"
            )

        round_numbers = tuple(
            value.round_no
            for value in self.rounds
        )

        if len(
            set(
                round_numbers
            )
        ) != len(
            round_numbers
        ):
            raise ContractError(
                "walk-forward round numbers must be unique"
            )

        if round_numbers != tuple(
            sorted(
                round_numbers
            )
        ):
            raise ContractError(
                "walk-forward rounds must be chronological"
            )

        try:
            start_round = int(
                self.window.start_round
            )

            end_round = int(
                self.window.end_round
            )

        except (
            AttributeError,
            TypeError,
            ValueError,
        ) as exc:
            raise ContractError(
                "invalid evaluation window"
            ) from exc

        if any(
            (
                round_no < start_round
                or round_no > end_round
            )
            for round_no in round_numbers
        ):
            raise ContractError(
                "round outside evaluation window"
            )

        if self.top3.k != 3:
            raise ContractError(
                "top3 must have k=3"
            )

        if self.top5.k != 5:
            raise ContractError(
                "top5 must have k=5"
            )

        if self.top10.k != 10:
            raise ContractError(
                "top10 must have k=10"
            )

    def as_dict(
        self,
    ) -> dict[str, object]:
        return {
            "window": {
                "name":
                    getattr(
                        self.window,
                        "name",
                        None,
                    ),

                "start_round":
                    self.window.start_round,

                "end_round":
                    self.window.end_round,
            },

            "rounds": [
                value.as_dict()
                for value in self.rounds
            ],

            "top3":
                self.top3.as_dict(),

            "top5":
                self.top5.as_dict(),

            "top10":
                self.top10.as_dict(),

            "model_name":
                self.model_name,

            "regime_slices": [
                value.as_dict()
                for value in self.regime_slices
            ],

            "strategy_slices": [
                value.as_dict()
                for value in self.strategy_slices
            ],
        }


@dataclass(
    frozen=True
)
class _RowScore:
    row: Any
    hit_counts: tuple[int, ...]
    best_hits: int


def _distribution_from_hits(
    hits: Iterable[int],
) -> HitDistribution:
    counts = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]

    for value in hits:
        if (
            not isinstance(
                value,
                int,
            )
            or value < 0
            or value > 6
        ):
            raise ContractError(
                "hit count must be between 0 and 6"
            )

        counts[value] += 1

    return HitDistribution(
        hit_0=counts[0],
        hit_1=counts[1],
        hit_2=counts[2],
        hit_3=counts[3],
        hit_4=counts[4],
        hit_5=counts[5],
        hit_6=counts[6],
    )


def _mean(
    values: Iterable[int],
) -> float:
    materialized = tuple(
        values
    )

    if not materialized:
        return 0.0

    return (
        sum(
            materialized
        )
        / len(
            materialized
        )
    )


def _history_rounds(
    row: Any,
) -> tuple[int, ...]:
    try:
        values = tuple(
            row.history_rounds
        )

    except (
        AttributeError,
        TypeError,
    ) as exc:
        raise ContractError(
            "replay row history_rounds missing or invalid"
        ) from exc

    if not values:
        raise ContractError(
            "history_rounds must not be empty"
        )

    normalized = tuple(
        _require_int(
            value,
            name="history_round",
            minimum=1,
        )
        for value in values
    )

    return normalized


def _predictions(
    row: Any,
) -> tuple[
    tuple[int, ...],
    ...
]:
    try:
        values = tuple(
            row.predictions
        )

    except (
        AttributeError,
        TypeError,
    ) as exc:
        raise ContractError(
            "replay row predictions missing or invalid"
        ) from exc

    if len(
        values
    ) < 10:
        raise ContractError(
            "at least 10 prediction sets are required"
        )

    return tuple(
        _validate_numbers(
            value,
            name="prediction",
        )
        for value in values
    )


def _round_no(
    row: Any,
) -> int:
    try:
        value = row.round_no

    except AttributeError as exc:
        raise ContractError(
            "replay row round_no missing"
        ) from exc

    return _require_int(
        value,
        name="round_no",
        minimum=1,
    )


def _actual_numbers(
    row: Any,
) -> tuple[int, ...]:
    try:
        values = row.actual_numbers

    except AttributeError as exc:
        raise ContractError(
            "replay row actual_numbers missing"
        ) from exc

    return _validate_numbers(
        values,
        name="actual_numbers",
    )


def _model_name(
    row: Any,
) -> str:
    try:
        value = row.model_name

    except AttributeError as exc:
        raise ContractError(
            "replay row model_name missing"
        ) from exc

    if (
        not isinstance(
            value,
            str,
        )
        or not value
    ):
        raise ContractError(
            "model_name must be non-empty"
        )

    return value


def _optional_string(
    row: Any,
    name: str,
) -> str | None:
    value = getattr(
        row,
        name,
        None,
    )

    if value is None:
        return None

    if not isinstance(
        value,
        str,
    ):
        raise ContractError(
            f"{name} must be str or None"
        )

    return value


class TopKWalkForwardEvaluator:
    def __init__(
        self,
        *,
        baseline_provider: Any,
    ) -> None:
        if baseline_provider is None:
            raise ContractError(
                "baseline_provider is required"
            )

        self._baseline_provider = (
            baseline_provider
        )

    @staticmethod
    def _validate_row(
        row: Any,
    ) -> None:
        round_no = _round_no(
            row
        )

        history = _history_rounds(
            row
        )

        if any(
            value >= round_no
            for value in history
        ):
            raise ContractError(
                "future history or prediction round leakage detected"
            )

        _actual_numbers(
            row
        )

        _predictions(
            row
        )

        _model_name(
            row
        )

        _optional_string(
            row,
            "regime_id",
        )

        _optional_string(
            row,
            "strategy_name",
        )

    @staticmethod
    def _score(
        row: Any,
        k: int,
    ) -> _RowScore:
        predictions = _predictions(
            row
        )

        actual = set(
            _actual_numbers(
                row
            )
        )

        hit_counts = tuple(
            len(
                set(
                    prediction
                )
                & actual
            )
            for prediction in predictions[:k]
        )

        return _RowScore(
            row=row,
            hit_counts=hit_counts,
            best_hits=max(
                hit_counts
            ),
        )

    @staticmethod
    def _build_topk(
        candidate_scores: tuple[
            _RowScore,
            ...
        ],
        baseline_scores: tuple[
            _RowScore,
            ...
        ],
        *,
        k: int,
    ) -> TopKEvaluation:
        candidate_best = tuple(
            score.best_hits
            for score in candidate_scores
        )

        baseline_best = tuple(
            score.best_hits
            for score in baseline_scores
        )

        candidate_all = tuple(
            value
            for score in candidate_scores
            for value in score.hit_counts
        )

        best_distribution = (
            _distribution_from_hits(
                candidate_best
            )
        )

        set_distribution = (
            _distribution_from_hits(
                candidate_all
            )
        )

        baseline_distribution = (
            _distribution_from_hits(
                baseline_best
            )
        )

        return TopKEvaluation(
            k=k,

            round_count=len(
                candidate_scores
            ),

            set_count=len(
                candidate_all
            ),

            mean_best_hits=_mean(
                candidate_best
            ),

            mean_set_hits=_mean(
                candidate_all
            ),

            best_hit_distribution=(
                best_distribution
            ),

            set_hit_distribution=(
                set_distribution
            ),

            baseline_delta_mean_best_hits=(
                _mean(
                    candidate_best
                )
                - _mean(
                    baseline_best
                )
            ),

            baseline_delta_3plus_rate=(
                best_distribution.at_least_3_rate
                - baseline_distribution.at_least_3_rate
            ),

            baseline_delta_4plus_rate=(
                best_distribution.at_least_4_rate
                - baseline_distribution.at_least_4_rate
            ),
        )

    def _baseline_for(
        self,
        row: Any,
    ) -> Any:
        round_no = _round_no(
            row
        )

        try:
            baseline = (
                self._baseline_provider.get(
                    round_no
                )
            )

        except ContractError:
            raise

        except Exception as exc:
            raise ContractError(
                "baseline round missing"
            ) from exc

        if baseline is None:
            raise ContractError(
                "baseline round missing"
            )

        self._validate_row(
            baseline
        )

        if _round_no(
            baseline
        ) != round_no:
            raise ContractError(
                "baseline round mismatch"
            )

        if _history_rounds(
            baseline
        ) != _history_rounds(
            row
        ):
            raise ContractError(
                "baseline walk-forward boundary mismatch"
            )

        return baseline

    def _slice(
        self,
        *,
        name: str,
        candidate_rows: tuple[
            Any,
            ...
        ],
        baseline_rows: tuple[
            Any,
            ...
        ],
    ) -> WalkForwardSlice:
        candidate3 = tuple(
            self._score(
                row,
                3,
            )
            for row in candidate_rows
        )

        candidate5 = tuple(
            self._score(
                row,
                5,
            )
            for row in candidate_rows
        )

        candidate10 = tuple(
            self._score(
                row,
                10,
            )
            for row in candidate_rows
        )

        baseline3 = tuple(
            self._score(
                row,
                3,
            )
            for row in baseline_rows
        )

        baseline5 = tuple(
            self._score(
                row,
                5,
            )
            for row in baseline_rows
        )

        baseline10 = tuple(
            self._score(
                row,
                10,
            )
            for row in baseline_rows
        )

        return WalkForwardSlice(
            name=name,

            round_count=len(
                candidate_rows
            ),

            top3=self._build_topk(
                candidate3,
                baseline3,
                k=3,
            ),

            top5=self._build_topk(
                candidate5,
                baseline5,
                k=5,
            ),

            top10=self._build_topk(
                candidate10,
                baseline10,
                k=10,
            ),
        )

    def evaluate(
        self,
        *,
        window: Any,
        replay_rows: Iterable[Any],
    ) -> WalkForwardEvaluation:
        rows = tuple(
            replay_rows
        )

        if not rows:
            raise ContractError(
                "replay_rows must not be empty"
            )

        for row in rows:
            self._validate_row(
                row
            )

        round_numbers = tuple(
            _round_no(
                row
            )
            for row in rows
        )

        if len(
            set(
                round_numbers
            )
        ) != len(
            round_numbers
        ):
            raise ContractError(
                "duplicate replay round"
            )

        if round_numbers != tuple(
            sorted(
                round_numbers
            )
        ):
            raise ContractError(
                "replay rows must be chronological"
            )

        try:
            start_round = int(
                window.start_round
            )

            end_round = int(
                window.end_round
            )

        except (
            AttributeError,
            TypeError,
            ValueError,
        ) as exc:
            raise ContractError(
                "invalid evaluation window"
            ) from exc

        if any(
            (
                round_no < start_round
                or round_no > end_round
            )
            for round_no in round_numbers
        ):
            raise ContractError(
                "replay row outside evaluation window"
            )

        model_names = tuple(
            _model_name(
                row
            )
            for row in rows
        )

        if len(
            set(
                model_names
            )
        ) != 1:
            raise ContractError(
                "model_name must be consistent"
            )

        baselines = tuple(
            self._baseline_for(
                row
            )
            for row in rows
        )

        candidate_scores = {
            k: tuple(
                self._score(
                    row,
                    k,
                )
                for row in rows
            )
            for k in _SUPPORTED_K
        }

        baseline_scores = {
            k: tuple(
                self._score(
                    row,
                    k,
                )
                for row in baselines
            )
            for k in _SUPPORTED_K
        }

        top3 = self._build_topk(
            candidate_scores[3],
            baseline_scores[3],
            k=3,
        )

        top5 = self._build_topk(
            candidate_scores[5],
            baseline_scores[5],
            k=5,
        )

        top10 = self._build_topk(
            candidate_scores[10],
            baseline_scores[10],
            k=10,
        )

        round_results = []

        for (
            index,
            row,
        ) in enumerate(
            rows
        ):
            round_results.append(
                WalkForwardRoundEvaluation(
                    round_no=_round_no(
                        row
                    ),

                    history_end_round=max(
                        _history_rounds(
                            row
                        )
                    ),

                    actual_numbers=(
                        _actual_numbers(
                            row
                        )
                    ),

                    model_name=(
                        _model_name(
                            row
                        )
                    ),

                    regime_id=(
                        _optional_string(
                            row,
                            "regime_id",
                        )
                    ),

                    strategy_name=(
                        _optional_string(
                            row,
                            "strategy_name",
                        )
                    ),

                    top3=self._build_topk(
                        (
                            candidate_scores[
                                3
                            ][index],
                        ),
                        (
                            baseline_scores[
                                3
                            ][index],
                        ),
                        k=3,
                    ),

                    top5=self._build_topk(
                        (
                            candidate_scores[
                                5
                            ][index],
                        ),
                        (
                            baseline_scores[
                                5
                            ][index],
                        ),
                        k=5,
                    ),

                    top10=self._build_topk(
                        (
                            candidate_scores[
                                10
                            ][index],
                        ),
                        (
                            baseline_scores[
                                10
                            ][index],
                        ),
                        k=10,
                    ),
                )
            )

        regime_names = sorted(
            {
                value
                for value in (
                    _optional_string(
                        row,
                        "regime_id",
                    )
                    for row in rows
                )
                if value is not None
            }
        )

        strategy_names = sorted(
            {
                value
                for value in (
                    _optional_string(
                        row,
                        "strategy_name",
                    )
                    for row in rows
                )
                if value is not None
            }
        )

        regime_slices = []

        for name in regime_names:
            indexes = tuple(
                index
                for (
                    index,
                    row,
                ) in enumerate(
                    rows
                )
                if (
                    _optional_string(
                        row,
                        "regime_id",
                    )
                    == name
                )
            )

            regime_slices.append(
                self._slice(
                    name=name,

                    candidate_rows=tuple(
                        rows[index]
                        for index in indexes
                    ),

                    baseline_rows=tuple(
                        baselines[index]
                        for index in indexes
                    ),
                )
            )

        strategy_slices = []

        for name in strategy_names:
            indexes = tuple(
                index
                for (
                    index,
                    row,
                ) in enumerate(
                    rows
                )
                if (
                    _optional_string(
                        row,
                        "strategy_name",
                    )
                    == name
                )
            )

            strategy_slices.append(
                self._slice(
                    name=name,

                    candidate_rows=tuple(
                        rows[index]
                        for index in indexes
                    ),

                    baseline_rows=tuple(
                        baselines[index]
                        for index in indexes
                    ),
                )
            )

        return WalkForwardEvaluation(
            window=window,

            rounds=tuple(
                round_results
            ),

            top3=top3,
            top5=top5,
            top10=top10,

            model_name=model_names[0],

            regime_slices=tuple(
                regime_slices
            ),

            strategy_slices=tuple(
                strategy_slices
            ),
        )
