"""Project D public-API adapter."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Any, Mapping

from lrp.contracts import CompatibilityError, ContractError


_REQUIRED_EXPORTS = (
    "generate_candidates",
    "number_signals_from_statistics",
    "rank_candidates",
    "score_candidates",
    "select_diverse_candidates",
    "select_practical_sets",
    "validate_statistics_contract",
)


@dataclass(frozen=True, slots=True)
class CandidateAdapter:
    """Thin adapter over Project D's public package."""

    module: ModuleType

    @classmethod
    def load(cls) -> "CandidateAdapter":
        try:
            module = import_module("lotto645_candidates")
        except ImportError as exc:
            raise CompatibilityError(
                "Project D Candidate Engine is not importable"
            ) from exc

        missing = tuple(
            name
            for name in _REQUIRED_EXPORTS
            if not hasattr(module, name)
        )
        if missing:
            raise CompatibilityError(
                "Candidate public API is incomplete: "
                + ", ".join(missing)
            )

        return cls(module=module)

    @property
    def version(self) -> str:
        value = getattr(self.module, "__version__", None)
        if not isinstance(value, str):
            raise CompatibilityError(
                "Candidate package does not expose __version__"
            )
        return value

    def validate_statistics(
        self,
        statistics: Mapping[int, Mapping[str, Any]],
    ) -> object:
        return self.module.validate_statistics_contract(statistics)

    def number_signals(
        self,
        statistics: Mapping[int, Mapping[str, Any]],
    ) -> Mapping[int, object]:
        result = self.module.number_signals_from_statistics(statistics)

        if not isinstance(result, Mapping):
            raise ContractError(
                "number_signals_from_statistics must return a mapping"
            )

        return result

    def generate_candidates(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        return self.module.generate_candidates(*args, **kwargs)

    def score_candidates(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        return self.module.score_candidates(*args, **kwargs)

    def rank_candidates(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        return self.module.rank_candidates(*args, **kwargs)

    def select_diverse_candidates(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        return self.module.select_diverse_candidates(*args, **kwargs)

    def select_practical_sets(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        return self.module.select_practical_sets(*args, **kwargs)
