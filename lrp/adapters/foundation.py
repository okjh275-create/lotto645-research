"""Foundation public-API adapter for Project A."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Any, Mapping

from lrp.contracts import CompatibilityError, ContractError
from lrp.core import RuntimeContext


_REQUIRED_EXPORTS = (
    "ExecutionContext",
    "PluginRegistry",
    "check_foundation_compatibility",
)


@dataclass(frozen=True, slots=True)
class FoundationAdapter:
    """Thin adapter over Project B's stable public package."""

    module: ModuleType

    @classmethod
    def load(cls) -> "FoundationAdapter":
        """Import and validate the Foundation public package."""

        try:
            module = import_module("foundation")
        except ImportError as exc:
            raise CompatibilityError(
                "Project B Foundation is not importable"
            ) from exc

        missing = tuple(
            name for name in _REQUIRED_EXPORTS
            if not hasattr(module, name)
        )
        if missing:
            raise CompatibilityError(
                "Foundation public API is incomplete: "
                + ", ".join(missing)
            )

        return cls(module=module)

    @property
    def version(self) -> str:
        """Return the installed Foundation version."""

        value = getattr(self.module, "__version__", None)
        if not isinstance(value, str):
            raise CompatibilityError(
                "Foundation does not expose __version__"
            )
        return value

    def create_execution_context(
        self,
        runtime: RuntimeContext,
    ) -> object:
        """Convert Project A runtime metadata to Foundation context."""

        if not isinstance(runtime, RuntimeContext):
            raise ContractError(
                "runtime must be an lrp.core.RuntimeContext"
            )

        execution_context = getattr(
            self.module,
            "ExecutionContext",
        )

        return execution_context(
            execution_id=runtime.execution_id,
            seed=runtime.seed,
            parameters=dict(runtime.parameters),
            created_at_utc=runtime.created_at_utc,
        )

    def create_plugin_registry(self) -> object:
        """Create Project B's native PluginRegistry."""

        registry_type = getattr(self.module, "PluginRegistry")
        return registry_type()

    def check_compatibility(
        self,
        *,
        requested_version: str,
        artifact_version: str | None = None,
    ) -> object:
        """Delegate compatibility checking to Foundation."""

        check = getattr(
            self.module,
            "check_foundation_compatibility",
        )

        kwargs: dict[str, Any] = {
            "requested_version": requested_version,
        }
        if artifact_version is not None:
            kwargs["artifact_version"] = artifact_version

        try:
            return check(**kwargs)
        except TypeError:
            # Foundation 1.0 installations may expose a positional
            # compatibility helper. Keep this fallback inside the adapter.
            if artifact_version is None:
                return check(requested_version)
            return check(requested_version, artifact_version)

    def public_exports(self) -> tuple[str, ...]:
        """Return deterministic public names exposed by Foundation."""

        names = getattr(self.module, "__all__", ())
        if not isinstance(names, (tuple, list)):
            return ()
        return tuple(sorted(str(name) for name in names))
