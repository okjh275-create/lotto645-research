from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType

from lrp.operations.durable_replay_promotion_publication_request import (
    DurableReplayPromotionPublicationRequest,
)


_TRANSPORT_FIELDS = (
    "status",
    "round_count",
    "candidate_model_name",
    "baseline_model_name",
    "recommendation",
    "action",
    "window",
    "source_decision",
    "registry_root",
)


@dataclass(frozen=True)
class DurableReplayPublicationInvocationTransport:
    status: str
    round_count: int
    candidate_model_name: str
    baseline_model_name: str
    recommendation: str
    action: str
    window: Mapping[str, object]
    source_decision: str
    registry_root: str


class DurableReplayPublicationInvocationTransportCodec:
    def encode(
        self,
        request: DurableReplayPromotionPublicationRequest,
    ) -> DurableReplayPublicationInvocationTransport:
        if not isinstance(request, DurableReplayPromotionPublicationRequest):
            raise TypeError(
                "request must be DurableReplayPromotionPublicationRequest"
            )
        if not isinstance(request.window, Mapping):
            raise TypeError("request window must be a Mapping")

        window = self._readonly_window(request.window)

        return DurableReplayPublicationInvocationTransport(
            status=request.status,
            round_count=request.round_count,
            candidate_model_name=request.candidate_model_name,
            baseline_model_name=request.baseline_model_name,
            recommendation=request.recommendation,
            action=request.action,
            window=window,
            source_decision=str(request.source_decision),
            registry_root=str(request.registry_root),
        )

    def to_mapping(
        self,
        transport: DurableReplayPublicationInvocationTransport,
    ) -> Mapping[str, object]:
        self._require_transport(transport)
        return {
            "status": transport.status,
            "round_count": transport.round_count,
            "candidate_model_name": transport.candidate_model_name,
            "baseline_model_name": transport.baseline_model_name,
            "recommendation": transport.recommendation,
            "action": transport.action,
            "window": self._plain_window(transport.window),
            "source_decision": transport.source_decision,
            "registry_root": transport.registry_root,
        }

    def from_mapping(
        self,
        mapping: Mapping[str, object],
    ) -> DurableReplayPublicationInvocationTransport:
        if not isinstance(mapping, Mapping):
            raise TypeError("mapping must be a Mapping")

        keys = set(mapping)
        expected = set(_TRANSPORT_FIELDS)
        if keys != expected:
            missing = expected - keys
            unknown = keys - expected
            if missing:
                raise ValueError("transport mapping has missing fields")
            if unknown:
                raise ValueError("transport mapping has unknown fields")

        self._require_str(mapping["status"], "status")
        self._require_int(mapping["round_count"], "round_count")
        self._require_str(
            mapping["candidate_model_name"],
            "candidate_model_name",
        )
        self._require_str(
            mapping["baseline_model_name"],
            "baseline_model_name",
        )
        self._require_str(mapping["recommendation"], "recommendation")
        self._require_str(mapping["action"], "action")
        self._require_str(mapping["source_decision"], "source_decision")
        self._require_str(mapping["registry_root"], "registry_root")

        window_value = mapping["window"]
        if not isinstance(window_value, Mapping):
            raise TypeError("window must be a Mapping")

        return DurableReplayPublicationInvocationTransport(
            status=mapping["status"],
            round_count=mapping["round_count"],
            candidate_model_name=mapping["candidate_model_name"],
            baseline_model_name=mapping["baseline_model_name"],
            recommendation=mapping["recommendation"],
            action=mapping["action"],
            window=self._readonly_window(window_value),
            source_decision=mapping["source_decision"],
            registry_root=mapping["registry_root"],
        )

    def decode(
        self,
        transport: DurableReplayPublicationInvocationTransport,
    ) -> DurableReplayPromotionPublicationRequest:
        self._require_transport(transport)

        return DurableReplayPromotionPublicationRequest(
            status=transport.status,
            round_count=transport.round_count,
            candidate_model_name=transport.candidate_model_name,
            baseline_model_name=transport.baseline_model_name,
            recommendation=transport.recommendation,
            action=transport.action,
            window=self._readonly_window(transport.window),
            source_decision=transport.source_decision,
            registry_root=transport.registry_root,
        )

    @staticmethod
    def _require_transport(
        transport: DurableReplayPublicationInvocationTransport,
    ) -> None:
        if not isinstance(
            transport,
            DurableReplayPublicationInvocationTransport,
        ):
            raise TypeError(
                "transport must be "
                "DurableReplayPublicationInvocationTransport"
            )

    @staticmethod
    def _require_str(value: object, field_name: str) -> None:
        if not isinstance(value, str):
            raise TypeError(field_name + " must be str")

    @staticmethod
    def _require_int(value: object, field_name: str) -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(field_name + " must be int")

    @classmethod
    def _clone_value(cls, value: object) -> object:
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise TypeError("window float must be finite")
            return value
        if isinstance(value, Mapping):
            cloned: dict[str, object] = {}
            for key, nested in value.items():
                if not isinstance(key, str):
                    raise TypeError("window mapping keys must be str")
                cloned[key] = cls._clone_value(nested)
            return cloned
        if isinstance(value, list):
            return [cls._clone_value(item) for item in value]
        raise TypeError("window contains unsupported value type")

    @classmethod
    def _plain_window(
        cls,
        window: Mapping[str, object],
    ) -> dict[str, object]:
        if not isinstance(window, Mapping):
            raise TypeError("window must be a Mapping")
        cloned = cls._clone_value(window)
        if not isinstance(cloned, dict):
            raise TypeError("window must project to dict")
        return cloned

    @classmethod
    def _readonly_window(
        cls,
        window: Mapping[str, object],
    ) -> Mapping[str, object]:
        return MappingProxyType(cls._plain_window(window))