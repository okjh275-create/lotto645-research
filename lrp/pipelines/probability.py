"""Probability-vector construction for the prediction pipeline."""

from __future__ import annotations

import math
from typing import Any, Mapping

from lrp.contracts import ContractError


_NUMBER_RANGE = tuple(range(1, 46))


def _read_signal(
    value: object,
    name: str,
    *,
    number: int,
) -> float:
    if isinstance(value, Mapping):
        raw = value.get(name)
    else:
        raw = getattr(value, name, None)

    if (
        isinstance(raw, bool)
        or not isinstance(raw, (int, float))
        or not math.isfinite(float(raw))
    ):
        raise ContractError(
            f"number signal {number}.{name} must be finite numeric"
        )

    result = float(raw)
    if not 0.0 <= result <= 1.0:
        raise ContractError(
            f"number signal {number}.{name} must be within 0..1"
        )

    return result


def _structural_prior(number: int) -> tuple[float, float, float]:
    """Return neutral terminal, sum-band and parity priors.

    These components cannot be fully evaluated before a six-number set is
    formed. At number-probability stage, bounded neutral priors prevent those
    score weights from being incorrectly discarded.

    - terminal dispersion: neutral 0.5
    - sum-band fit: preference for central lotto range
    - parity balance: neutral 0.5
    """

    terminal_dispersion = 0.5

    center = 23.0
    scale = 22.0
    sum_band = max(0.0, 1.0 - abs(number - center) / scale)

    parity_balance = 0.5

    return terminal_dispersion, sum_band, parity_balance


def build_probability_vector(
    number_signals: Mapping[int, object],
    *,
    weights: Mapping[str, float],
) -> dict[int, float]:
    """Build a normalized 1..45 probability vector.

    Number-level Project D signals contribute directly. Set-level components
    use bounded neutral priors and are evaluated fully during candidate
    scoring in the next pipeline stage.
    """

    missing = tuple(
        number for number in _NUMBER_RANGE
        if number not in number_signals
    )
    if missing:
        raise ContractError(
            f"number signals are missing numbers: {missing}"
        )

    required_weights = (
        "recency",
        "frequency",
        "gap_reversion",
        "pair_graph",
        "terminal_dispersion",
        "sum_band",
        "parity_balance",
    )

    missing_weights = tuple(
        name for name in required_weights
        if name not in weights
    )
    if missing_weights:
        raise ContractError(
            f"probability weights are missing: {missing_weights}"
        )

    raw: dict[int, float] = {}

    for number in _NUMBER_RANGE:
        signal = number_signals[number]
        terminal, sum_band, parity = _structural_prior(number)

        raw[number] = (
            float(weights["recency"])
            * _read_signal(signal, "recency", number=number)
            + float(weights["frequency"])
            * _read_signal(signal, "frequency", number=number)
            + float(weights["gap_reversion"])
            * _read_signal(signal, "gap_reversion", number=number)
            + float(weights["pair_graph"])
            * _read_signal(signal, "pair_graph", number=number)
            + float(weights["terminal_dispersion"]) * terminal
            + float(weights["sum_band"]) * sum_band
            + float(weights["parity_balance"]) * parity
        )

    floor = 1e-12
    adjusted = {
        number: max(floor, value)
        for number, value in raw.items()
    }

    total = sum(adjusted.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ContractError(
            "probability-vector total must be finite and positive"
        )

    normalized = {
        number: adjusted[number] / total
        for number in _NUMBER_RANGE
    }

    normalized_total = sum(normalized.values())
    if not math.isclose(
        normalized_total,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ContractError(
            "probability vector failed normalization"
        )

    return normalized
