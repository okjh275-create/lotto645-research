"""Core platform assembly API."""

from .assembly import (
    AssemblyReport,
    ComponentCheck,
    PlatformAssembly,
    assemble_platform,
    format_assembly_report,
    inspect_platform,
)
from .component_catalog import (
    CANDIDATE_COMPONENT_ID,
    FOUNDATION_COMPONENT_ID,
    STATISTICS_COMPONENT_ID,
    ComponentCatalog,
    build_builtin_descriptors,
)
from .runtime import RuntimeContext

__all__ = [
    "AssemblyReport",
    "CANDIDATE_COMPONENT_ID",
    "ComponentCatalog",
    "ComponentCheck",
    "FOUNDATION_COMPONENT_ID",
    "PlatformAssembly",
    "RuntimeContext",
    "STATISTICS_COMPONENT_ID",
    "assemble_platform",
    "build_builtin_descriptors",
    "format_assembly_report",
    "inspect_platform",
]
