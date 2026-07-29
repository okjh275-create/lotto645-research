"""Runtime context management for Lotto645 Research Platform."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping
from uuid import uuid4

from lrp.contracts import ContractError


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _derive_seed(seed: int, namespace: str) -> int:
    digest = hashlib.sha256(
        _canonical_json_bytes(
            {
                "seed": seed,
                "namespace": namespace,
            }
        )
    ).hexdigest()
    return int(digest[:16], 16)


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    """Dependency-light Project A execution metadata.

    This model is used before Foundation is imported. Once Project B is
    available, ``to_foundation_context`` converts it to Foundation's stable
    ``ExecutionContext``.
    """

    execution_id: str
    seed: int
    parameters: Mapping[str, Any] = field(default_factory=dict)
    created_at_utc: str = field(
        default_factory=lambda: (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
    )

    def __post_init__(self) -> None:
        execution_id = (
            self.execution_id.strip()
            if isinstance(self.execution_id, str)
            else ""
        )

        if not execution_id:
            raise ContractError(
                "execution_id must be a non-empty string"
            )

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ContractError("seed must be an integer")

        if not isinstance(self.parameters, Mapping):
            raise ContractError("parameters must be a mapping")

        try:
            datetime.fromisoformat(
                self.created_at_utc.replace("Z", "+00:00")
            )
        except (TypeError, ValueError) as exc:
            raise ContractError(
                "created_at_utc must be an ISO-8601 timestamp"
            ) from exc

        object.__setattr__(self, "execution_id", execution_id)
        object.__setattr__(
            self,
            "parameters",
            dict(sorted(self.parameters.items())),
        )

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        execution_id: str | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> "RuntimeContext":
        """Create one reproducible platform execution context."""

        resolved_id = execution_id or (
            "lrp-" + uuid4().hex
        )

        return cls(
            execution_id=resolved_id,
            seed=seed,
            parameters={} if parameters is None else parameters,
        )

    def child(
        self,
        namespace: str,
        *,
        parameters: Mapping[str, Any] | None = None,
    ) -> "RuntimeContext":
        """Create a deterministic child context."""

        normalized = (
            namespace.strip()
            if isinstance(namespace, str)
            else ""
        )

        if not normalized:
            raise ContractError(
                "child namespace must be a non-empty string"
            )

        return RuntimeContext(
            execution_id=f"{self.execution_id}/{normalized}",
            seed=_derive_seed(self.seed, normalized),
            parameters=(
                self.parameters
                if parameters is None
                else parameters
            ),
            created_at_utc=self.created_at_utc,
        )

    def to_dict(self) -> dict[str, object]:
        """Return serializable runtime metadata."""

        return {
            "execution_id": self.execution_id,
            "seed": self.seed,
            "parameters": dict(self.parameters),
            "created_at_utc": self.created_at_utc,
        }

    def to_foundation_context(self) -> object:
        """Convert to Project B's stable ExecutionContext.

        Importing Foundation is deliberately delayed so the LRP package can
        still perform installation diagnostics when Project B is unavailable.
        """

        try:
            from foundation import ExecutionContext
        except ImportError as exc:
            raise ContractError(
                "Project B Foundation is not importable"
            ) from exc

        return ExecutionContext(
            execution_id=self.execution_id,
            seed=self.seed,
            parameters=dict(self.parameters),
            created_at_utc=self.created_at_utc,
        )
