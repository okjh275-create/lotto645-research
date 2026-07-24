"""Project C stable-public-API adapter."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Any, Iterable, Mapping

from lrp.contracts import CompatibilityError, ContractError


_REQUIRED_EXPORTS = (
    "AnalysisConfig",
    "AnalysisReport",
    "StatisticsEngine",
    "analyze_all",
    "build_feature_matrix",
    "feature_matrix_to_dict",
    "snapshot_to_dict",
)


@dataclass(frozen=True, slots=True)
class StatisticsAdapter:
    """Thin adapter over Project C's stable API 1.x."""

    module: ModuleType

    @classmethod
    def load(cls) -> "StatisticsAdapter":
        """Import and validate Project C."""

        try:
            module = import_module("lotto645_statistics")
        except ImportError as exc:
            raise CompatibilityError(
                "Project C Statistics Engine is not importable"
            ) from exc

        missing = tuple(
            name for name in _REQUIRED_EXPORTS
            if not hasattr(module, name)
        )
        if missing:
            raise CompatibilityError(
                "Statistics stable API is incomplete: "
                + ", ".join(missing)
            )

        api_version = getattr(module, "STABLE_API_VERSION", None)
        if api_version != "1.0":
            raise CompatibilityError(
                "unsupported Statistics stable API: "
                f"{api_version!r}"
            )

        return cls(module=module)

    @property
    def version(self) -> str:
        value = getattr(self.module, "__version__", None)
        if not isinstance(value, str):
            raise CompatibilityError(
                "Statistics package does not expose __version__"
            )
        return value

    @property
    def api_version(self) -> str:
        value = getattr(
            self.module,
            "STABLE_API_VERSION",
            None,
        )
        if not isinstance(value, str):
            raise CompatibilityError(
                "Statistics package does not expose "
                "STABLE_API_VERSION"
            )
        return value

    def create_config(self, **kwargs: Any) -> object:
        """Create Project C's public AnalysisConfig."""

        config_type = getattr(self.module, "AnalysisConfig")
        return config_type(**kwargs)

    def create_engine(self, *args: Any, **kwargs: Any) -> object:
        """Create Project C's public StatisticsEngine."""

        engine_type = getattr(self.module, "StatisticsEngine")
        return engine_type(*args, **kwargs)

    def analyze_all(
        self,
        draws: Iterable[object],
        *,
        config: object | None = None,
    ) -> object:
        """Run Project C's unified stable analysis function."""

        analyze = getattr(self.module, "analyze_all")

        if config is None:
            return analyze(draws)

        try:
            return analyze(draws, config=config)
        except TypeError:
            return analyze(draws, config)

    def build_feature_matrix(self, snapshot: object) -> object:
        """Build Project C's stable feature matrix."""

        build = getattr(self.module, "build_feature_matrix")
        return build(snapshot)

    def snapshot_to_dict(self, snapshot: object) -> Mapping[str, Any]:
        """Serialize a Project C snapshot through its stable serializer."""

        serializer = getattr(self.module, "snapshot_to_dict")
        result = serializer(snapshot)

        if not isinstance(result, Mapping):
            raise ContractError(
                "snapshot_to_dict must return a mapping"
            )

        return result

    def feature_matrix_to_dict(
        self,
        matrix: object,
    ) -> Mapping[str, Any]:
        """Serialize a Project C feature matrix."""

        serializer = getattr(
            self.module,
            "feature_matrix_to_dict",
        )
        result = serializer(matrix)

        if not isinstance(result, Mapping):
            raise ContractError(
                "feature_matrix_to_dict must return a mapping"
            )

        return result

    def public_exports(self) -> tuple[str, ...]:
        exports = getattr(self.module, "STABLE_EXPORTS", ())
        if not isinstance(exports, (tuple, list)):
            return ()
        return tuple(str(name) for name in exports)
