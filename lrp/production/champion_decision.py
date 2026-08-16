"""Production-owned champion decision contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProductionChampionDecision:
    """Minimal production view of a champion decision."""

    selected_model: str | None

    @classmethod
    def from_payload(
        cls,
        payload: object,
    ) -> "ProductionChampionDecision":
        if not isinstance(
            payload,
            Mapping,
        ):
            raise TypeError(
                "payload must be a mapping"
            )

        if "selection" not in payload:
            raise ValueError(
                "selection is required"
            )

        selection = payload["selection"]

        if not isinstance(
            selection,
            Mapping,
        ):
            raise TypeError(
                "selection must be a mapping"
            )

        if "selected_model" not in selection:
            raise ValueError(
                "selected_model is required"
            )

        selected_model = selection[
            "selected_model"
        ]

        if (
            selected_model is not None
            and not isinstance(
                selected_model,
                str,
            )
        ):
            raise TypeError(
                "selected_model must be "
                "a string or None"
            )

        return cls(
            selected_model=selected_model
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "selected_model": (
                self.selected_model
            ),
        }
