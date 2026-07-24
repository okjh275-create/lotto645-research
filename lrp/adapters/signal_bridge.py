"""Bridge from Project C snapshots to Project D statistics signals.

The bridge depends only on the stable Project C snapshot attributes:

- snapshot.windows
- snapshot.numbers
- snapshot.relationships.affinity_graph.weighted_degree

It does not import Project C implementation modules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

from lrp.contracts import ContractError


_REQUIRED_NUMBERS = tuple(range(1, 46))
_SIGNAL_FIELDS = (
    "recency",
    "frequency",
    "gap_reversion",
    "pair_graph",
)


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{field} must be numeric")

    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{field} must be finite")

    return result


def _unit_interval(value: float) -> float:
    return min(1.0, max(0.0, value))


def _read_attribute(value: object, name: str) -> Any:
    if isinstance(value, Mapping):
        if name not in value:
            raise ContractError(f"missing snapshot field: {name}")
        return value[name]

    if not hasattr(value, name):
        raise ContractError(f"missing snapshot attribute: {name}")

    return getattr(value, name)


def _read_number_row(
    numbers: object,
    number: int,
) -> object:
    if not isinstance(numbers, Mapping):
        raise ContractError("snapshot.numbers must be a mapping")

    row = numbers.get(number)
    if row is None:
        row = numbers.get(str(number))

    if row is None:
        raise ContractError(
            f"snapshot.numbers is missing number {number}"
        )

    return row


def _read_numeric(
    row: object,
    field: str,
    *,
    number: int,
) -> float:
    try:
        value = _read_attribute(row, field)
    except ContractError as exc:
        raise ContractError(
            f"number {number}: missing field {field}"
        ) from exc

    return _finite_float(
        value,
        field=f"number {number}.{field}",
    )


def _normalize_windows(value: object) -> tuple[int, int, int]:
    if isinstance(value, Mapping):
        short = value.get("short", value.get("short_window"))
        mid = value.get("mid", value.get("mid_window"))
        long = value.get("long", value.get("long_window"))
        raw = (short, mid, long)
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        raw = tuple(value)
        if len(raw) < 3:
            raise ContractError(
                "snapshot.windows must contain short, mid and long"
            )
        raw = raw[:3]
    else:
        raise ContractError(
            "snapshot.windows must be a mapping or sequence"
        )

    normalized: list[int] = []
    for index, item in enumerate(raw):
        if (
            isinstance(item, bool)
            or not isinstance(item, int)
            or item <= 0
        ):
            raise ContractError(
                f"snapshot.windows[{index}] must be positive"
            )
        normalized.append(item)

    short, mid, long = normalized
    if not short <= mid <= long:
        raise ContractError(
            "snapshot windows must satisfy short <= mid <= long"
        )

    return short, mid, long


def _weighted_degree(snapshot: object) -> Mapping[Any, Any]:
    relationships = _read_attribute(snapshot, "relationships")
    affinity_graph = _read_attribute(
        relationships,
        "affinity_graph",
    )
    values = _read_attribute(
        affinity_graph,
        "weighted_degree",
    )

    if not isinstance(values, Mapping):
        raise ContractError(
            "affinity_graph.weighted_degree must be a mapping"
        )

    return values


def _minmax(
    values: Mapping[int, float],
) -> dict[int, float]:
    minimum = min(values.values())
    maximum = max(values.values())

    if math.isclose(minimum, maximum):
        return {number: 0.5 for number in _REQUIRED_NUMBERS}

    scale = maximum - minimum
    return {
        number: _unit_interval(
            (values[number] - minimum) / scale
        )
        for number in _REQUIRED_NUMBERS
    }


@dataclass(frozen=True, slots=True)
class SignalBridgeConfig:
    """Weights used to combine Project C frequency windows."""

    short_weight: float = 0.50
    mid_weight: float = 0.30
    long_weight: float = 0.20

    def __post_init__(self) -> None:
        weights = (
            self.short_weight,
            self.mid_weight,
            self.long_weight,
        )

        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in weights
        ):
            raise ContractError(
                "signal bridge weights must be finite and non-negative"
            )

        total = sum(float(value) for value in weights)
        if not math.isclose(total, 1.0, abs_tol=1e-12):
            raise ContractError(
                "signal bridge weights must sum to 1.0"
            )


@dataclass(frozen=True, slots=True)
class StatisticsSignalSnapshot:
    """Normalized Project C→D integration payload."""

    windows: tuple[int, int, int]
    signals: Mapping[int, Mapping[str, float]]

    def __post_init__(self) -> None:
        missing = tuple(
            number
            for number in _REQUIRED_NUMBERS
            if number not in self.signals
        )
        if missing:
            raise ContractError(
                f"signals are missing numbers: {missing}"
            )

        normalized: dict[int, dict[str, float]] = {}

        for number in _REQUIRED_NUMBERS:
            row = self.signals[number]
            if not isinstance(row, Mapping):
                raise ContractError(
                    f"signal row {number} must be a mapping"
                )

            converted: dict[str, float] = {}
            for field in _SIGNAL_FIELDS:
                if field not in row:
                    raise ContractError(
                        f"signal row {number} is missing {field}"
                    )

                value = _finite_float(
                    row[field],
                    field=f"signals[{number}].{field}",
                )
                if not 0.0 <= value <= 1.0:
                    raise ContractError(
                        f"signals[{number}].{field} "
                        "must be between 0 and 1"
                    )
                converted[field] = value

            normalized[number] = converted

        object.__setattr__(self, "signals", normalized)

    def to_dict(self) -> dict[int, dict[str, float]]:
        """Return the exact mapping required by Project D."""

        return {
            number: dict(self.signals[number])
            for number in _REQUIRED_NUMBERS
        }


def build_statistics_signals(
    snapshot: object,
    *,
    config: SignalBridgeConfig | None = None,
) -> StatisticsSignalSnapshot:
    """Build Project D-compatible signals from a Project C snapshot.

    Formulas:

    - recency:
      ``1 - min(gap, long_window) / long_window``
    - frequency:
      weighted short/mid/long occurrence rates
    - gap_reversion:
      ``min(gap, long_window) / long_window``
    - pair_graph:
      min-max-normalized affinity weighted degree

    The input snapshot must already have been produced with the correct
    ``until_round`` boundary. The bridge never reads raw future draws.
    """

    resolved = config or SignalBridgeConfig()
    windows = _normalize_windows(
        _read_attribute(snapshot, "windows")
    )
    short_window, mid_window, long_window = windows
    numbers = _read_attribute(snapshot, "numbers")
    degree_mapping = _weighted_degree(snapshot)

    raw_pair_graph: dict[int, float] = {}
    rows: dict[int, dict[str, float]] = {}

    for number in _REQUIRED_NUMBERS:
        row = _read_number_row(numbers, number)

        short_frequency = _read_numeric(
            row,
            "short_frequency",
            number=number,
        )
        mid_frequency = _read_numeric(
            row,
            "mid_frequency",
            number=number,
        )
        long_frequency = _read_numeric(
            row,
            "long_frequency",
            number=number,
        )
        gap = max(
            0.0,
            _read_numeric(row, "gap", number=number),
        )

        recency = 1.0 - min(gap, long_window) / long_window
        gap_reversion = min(gap, long_window) / long_window

        frequency = (
            float(resolved.short_weight)
            * _unit_interval(short_frequency / short_window)
            + float(resolved.mid_weight)
            * _unit_interval(mid_frequency / mid_window)
            + float(resolved.long_weight)
            * _unit_interval(long_frequency / long_window)
        )

        degree_value = degree_mapping.get(
            number,
            degree_mapping.get(str(number), 0.0),
        )
        raw_pair_graph[number] = max(
            0.0,
            _finite_float(
                degree_value,
                field=f"weighted_degree[{number}]",
            ),
        )

        rows[number] = {
            "recency": _unit_interval(recency),
            "frequency": _unit_interval(frequency),
            "gap_reversion": _unit_interval(gap_reversion),
            "pair_graph": 0.0,
        }

    normalized_pair_graph = _minmax(raw_pair_graph)

    for number in _REQUIRED_NUMBERS:
        rows[number]["pair_graph"] = normalized_pair_graph[number]

    return StatisticsSignalSnapshot(
        windows=windows,
        signals=rows,
    )
