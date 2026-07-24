"""Platform component discovery and compatibility assembly."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Iterable

from lrp import (
    CANDIDATE_REQUIRED_VERSION,
    FOUNDATION_REQUIRED_VERSION,
    STATISTICS_REQUIRED_API_VERSION,
)
from lrp.contracts import (
    CompatibilityError,
    ComponentDescriptor,
    ComponentKind,
    ReleaseChannel,
    is_major_compatible,
    same_release_line,
)

from .component_catalog import ComponentCatalog


@dataclass(frozen=True, slots=True)
class ComponentCheck:
    """Runtime compatibility result for one component."""

    descriptor: ComponentDescriptor
    module_name: str
    installed: bool
    compatible: bool
    actual_version: str | None
    actual_api_version: str | None = None
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AssemblyReport:
    """Aggregated platform assembly diagnostics."""

    checks: tuple[ComponentCheck, ...]

    @property
    def compatible(self) -> bool:
        return bool(self.checks) and all(
            check.compatible for check in self.checks
        )

    @property
    def missing_components(self) -> tuple[str, ...]:
        return tuple(
            check.descriptor.component_id
            for check in self.checks
            if not check.installed
        )

    @property
    def incompatible_components(self) -> tuple[str, ...]:
        return tuple(
            check.descriptor.component_id
            for check in self.checks
            if check.installed and not check.compatible
        )

    def require_compatible(self) -> None:
        """Raise when any required component is unavailable or incompatible."""

        if self.compatible:
            return

        messages: list[str] = []
        for check in self.checks:
            if check.compatible:
                continue

            detail = ", ".join(check.reasons) or "unknown reason"
            messages.append(
                f"{check.descriptor.component_id}: {detail}"
            )

        raise CompatibilityError(
            "platform assembly failed: " + "; ".join(messages)
        )


@dataclass(frozen=True, slots=True)
class PlatformAssembly:
    """Resolved B/C/D modules and their compatibility report."""

    catalog: ComponentCatalog
    modules: dict[ComponentKind, ModuleType]
    report: AssemblyReport

    def module(self, kind: ComponentKind) -> ModuleType:
        """Return a loaded module by component kind."""

        try:
            return self.modules[kind]
        except KeyError as exc:
            raise CompatibilityError(
                f"component module is unavailable: {kind.value}"
            ) from exc


_MODULE_NAMES = {
    ComponentKind.FOUNDATION: "foundation",
    ComponentKind.STATISTICS: "lotto645_statistics",
    ComponentKind.CANDIDATE: "lotto645_candidates",
}


def _read_version(module: ModuleType) -> str | None:
    value = getattr(module, "__version__", None)
    return value if isinstance(value, str) else None


def _read_statistics_api_version(
    module: ModuleType,
) -> str | None:
    value = getattr(module, "STABLE_API_VERSION", None)
    return value if isinstance(value, str) else None


def _check_foundation(
    descriptor: ComponentDescriptor,
    module: ModuleType,
) -> ComponentCheck:
    actual = _read_version(module)
    reasons: list[str] = []

    if actual is None:
        reasons.append("missing __version__")
    elif not is_major_compatible(
        actual,
        FOUNDATION_REQUIRED_VERSION,
    ):
        reasons.append(
            "unsupported Foundation version "
            f"{actual}; required {FOUNDATION_REQUIRED_VERSION}+ "
            "on the same major line"
        )

    required_exports = (
        "ExecutionContext",
        "PluginRegistry",
        "check_foundation_compatibility",
    )
    for name in required_exports:
        if not hasattr(module, name):
            reasons.append(f"missing public export: {name}")

    return ComponentCheck(
        descriptor=descriptor,
        module_name=module.__name__,
        installed=True,
        compatible=not reasons,
        actual_version=actual,
        reasons=tuple(reasons),
    )


def _check_statistics(
    descriptor: ComponentDescriptor,
    module: ModuleType,
) -> ComponentCheck:
    actual = _read_version(module)
    api_version = _read_statistics_api_version(module)
    reasons: list[str] = []

    if actual is None:
        reasons.append("missing __version__")
    elif not is_major_compatible(actual, descriptor.version):
        reasons.append(
            "unsupported Statistics version "
            f"{actual}; required {descriptor.version}+ "
            "on the same major line"
        )

    if api_version != STATISTICS_REQUIRED_API_VERSION:
        reasons.append(
            "Statistics API mismatch "
            f"{api_version!r}; required "
            f"{STATISTICS_REQUIRED_API_VERSION!r}"
        )

    stable_exports = getattr(module, "STABLE_EXPORTS", ())
    for name in (
        "StatisticsEngine",
        "build_feature_matrix",
        "snapshot_to_dict",
    ):
        if not hasattr(module, name):
            reasons.append(f"missing public export: {name}")
        if name not in stable_exports:
            reasons.append(f"missing stable API declaration: {name}")

    return ComponentCheck(
        descriptor=descriptor,
        module_name=module.__name__,
        installed=True,
        compatible=not reasons,
        actual_version=actual,
        actual_api_version=api_version,
        reasons=tuple(reasons),
    )


def _check_candidate(
    descriptor: ComponentDescriptor,
    module: ModuleType,
) -> ComponentCheck:
    actual = _read_version(module)
    reasons: list[str] = []

    if actual is None:
        reasons.append("missing __version__")
    elif descriptor.release_channel is ReleaseChannel.CANDIDATE:
        if not same_release_line(
            actual,
            CANDIDATE_REQUIRED_VERSION,
        ):
            reasons.append(
                "Candidate release-line mismatch "
                f"{actual}; required "
                f"{CANDIDATE_REQUIRED_VERSION}"
            )
    elif not is_major_compatible(actual, descriptor.version):
        reasons.append(
            "unsupported Candidate version "
            f"{actual}; required {descriptor.version}"
        )

    for name in (
        "generate_candidates",
        "rank_candidates",
        "select_diverse_candidates",
        "select_practical_sets",
        "number_signals_from_statistics",
        "validate_statistics_contract",
    ):
        if not hasattr(module, name):
            reasons.append(f"missing public export: {name}")

    return ComponentCheck(
        descriptor=descriptor,
        module_name=module.__name__,
        installed=True,
        compatible=not reasons,
        actual_version=actual,
        reasons=tuple(reasons),
    )


def _missing_check(
    descriptor: ComponentDescriptor,
    module_name: str,
    exc: ImportError,
) -> ComponentCheck:
    return ComponentCheck(
        descriptor=descriptor,
        module_name=module_name,
        installed=False,
        compatible=False,
        actual_version=None,
        reasons=(f"import failed: {exc}",),
    )


def inspect_platform(
    catalog: ComponentCatalog | None = None,
) -> AssemblyReport:
    """Inspect all required Project A components without raising."""

    resolved_catalog = catalog or ComponentCatalog.builtin()
    checks: list[ComponentCheck] = []

    for descriptor in resolved_catalog.descriptors:
        module_name = _MODULE_NAMES.get(descriptor.kind)
        if module_name is None:
            checks.append(
                ComponentCheck(
                    descriptor=descriptor,
                    module_name="",
                    installed=False,
                    compatible=False,
                    actual_version=None,
                    reasons=(
                        "no module mapping for component kind",
                    ),
                )
            )
            continue

        try:
            module = import_module(module_name)
        except ImportError as exc:
            checks.append(
                _missing_check(
                    descriptor,
                    module_name,
                    exc,
                )
            )
            continue

        if descriptor.kind is ComponentKind.FOUNDATION:
            check = _check_foundation(descriptor, module)
        elif descriptor.kind is ComponentKind.STATISTICS:
            check = _check_statistics(descriptor, module)
        elif descriptor.kind is ComponentKind.CANDIDATE:
            check = _check_candidate(descriptor, module)
        else:
            check = ComponentCheck(
                descriptor=descriptor,
                module_name=module_name,
                installed=True,
                compatible=False,
                actual_version=_read_version(module),
                reasons=("unsupported component kind",),
            )

        checks.append(check)

    return AssemblyReport(tuple(checks))


def assemble_platform(
    catalog: ComponentCatalog | None = None,
    *,
    require_compatible: bool = True,
) -> PlatformAssembly:
    """Import, validate and return the complete B/C/D assembly."""

    resolved_catalog = catalog or ComponentCatalog.builtin()
    report = inspect_platform(resolved_catalog)

    if require_compatible:
        report.require_compatible()

    modules: dict[ComponentKind, ModuleType] = {}

    for check in report.checks:
        if not check.installed:
            continue
        if require_compatible and not check.compatible:
            continue

        modules[check.descriptor.kind] = import_module(
            check.module_name
        )

    return PlatformAssembly(
        catalog=resolved_catalog,
        modules=modules,
        report=report,
    )


def format_assembly_report(
    report: AssemblyReport,
) -> str:
    """Render a compact deterministic diagnostics report."""

    lines = [
        "LRP Platform Assembly",
        f"compatible={report.compatible}",
    ]

    for check in report.checks:
        status = "PASS" if check.compatible else "FAIL"
        version = check.actual_version or "-"
        api_version = check.actual_api_version or "-"
        detail = ", ".join(check.reasons) or "ok"

        lines.append(
            f"[{status}] {check.descriptor.component_id} "
            f"module={check.module_name or '-'} "
            f"version={version} "
            f"api={api_version} "
            f"detail={detail}"
        )

    return "\n".join(lines)
