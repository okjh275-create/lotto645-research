from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import MappingProxyType

import pytest

import lrp.operations.durable_replay_result_artifact_promotion_publication_invocation_source_adapter as product
from lrp.operations.durable_replay_promotion_publication_request import (
    DurableReplayPromotionPublicationRequest,
)
from lrp.operations.durable_replay_publication_invocation_transport import (
    DurableReplayPublicationInvocationTransport,
)


MODULE_PATH = Path(
    "lrp/operations/"
    "durable_replay_result_artifact_promotion_"
    "publication_invocation_source_adapter.py"
)

CLASS_NAME = (
    "DurableReplayResultArtifactPromotion"
    "PublicationInvocationSourceAdapter"
)


class RecordingRequestSource:
    def __init__(self, request: DurableReplayPromotionPublicationRequest) -> None:
        self.request = request
        self.calls: list[
            tuple[
                object,
                object,
                object,
                object,
            ]
        ] = []

    def adapt(
        self,
        artifact_root,
        end_round,
        *,
        source_decision,
        registry_root,
    ):
        self.calls.append(
            (
                artifact_root,
                end_round,
                source_decision,
                registry_root,
            )
        )

        return self.request


class RecordingTransportCodec:
    def __init__(
        self,
        transport: DurableReplayPublicationInvocationTransport,
    ) -> None:
        self.transport = transport
        self.calls: list[object] = []

    def encode(self, request):
        self.calls.append(request)
        return self.transport


class RecordingFileCarrier:
    def __init__(self, returned_path: Path) -> None:
        self.returned_path = returned_path
        self.calls: list[
            tuple[
                object,
                object,
            ]
        ] = []

    def write(self, path, transport):
        self.calls.append(
            (
                path,
                transport,
            )
        )

        return self.returned_path


def _request() -> object:
    return object()


def _transport() -> object:
    return object()


def _adapter(
    *,
    request_source,
    transport_codec,
    file_carrier,
):
    cls = getattr(
        product,
        CLASS_NAME,
    )

    return cls(
        source_adapter=request_source,
        transport_codec=transport_codec,
        file_carrier=file_carrier,
    )


def test_module_exports_exact_product_class() -> None:
    assert hasattr(
        product,
        CLASS_NAME,
    )

    cls = getattr(
        product,
        CLASS_NAME,
    )

    assert inspect.isclass(cls)


def test_constructor_dependency_contract_is_exact() -> None:
    cls = getattr(
        product,
        CLASS_NAME,
    )

    signature = inspect.signature(
        cls.__init__
    )

    assert list(
        signature.parameters
    ) == [
        "self",
        "source_adapter",
        "transport_codec",
        "file_carrier",
    ]


def test_public_method_contract_is_exact() -> None:
    cls = getattr(
        product,
        CLASS_NAME,
    )

    public_methods = [
        name
        for name, member in inspect.getmembers(
            cls,
            predicate=inspect.isfunction,
        )
        if not name.startswith("_")
    ]

    assert public_methods == [
        "adapt",
    ]


def test_adapt_signature_is_exact() -> None:
    cls = getattr(
        product,
        CLASS_NAME,
    )

    signature = inspect.signature(
        cls.adapt
    )

    assert list(
        signature.parameters
    ) == [
        "self",
        "artifact_root",
        "end_round",
        "source_decision",
        "registry_root",
        "output_path",
    ]

    assert (
        signature.parameters[
            "source_decision"
        ].kind
        is inspect.Parameter.KEYWORD_ONLY
    )

    assert (
        signature.parameters[
            "registry_root"
        ].kind
        is inspect.Parameter.KEYWORD_ONLY
    )

    assert (
        signature.parameters[
            "output_path"
        ].kind
        is inspect.Parameter.KEYWORD_ONLY
    )


def test_adapt_composes_request_transport_and_file_write() -> None:
    request = _request()
    transport = _transport()
    returned_path = Path(
        "persisted-invocation.json"
    )

    request_source = RecordingRequestSource(
        request
    )

    transport_codec = RecordingTransportCodec(
        transport
    )

    file_carrier = RecordingFileCarrier(
        returned_path
    )

    adapter = _adapter(
        request_source=request_source,
        transport_codec=transport_codec,
        file_carrier=file_carrier,
    )

    result = adapter.adapt(
        "artifact-root",
        1234,
        source_decision="decision.json",
        registry_root="registry",
        output_path="invocation.json",
    )

    assert result == returned_path

    assert request_source.calls == [
        (
            "artifact-root",
            1234,
            "decision.json",
            "registry",
        )
    ]

    assert transport_codec.calls == [
        request,
    ]

    assert file_carrier.calls == [
        (
            "invocation.json",
            transport,
        )
    ]


@pytest.mark.parametrize(
    (
        "artifact_root",
        "end_round",
        "source_decision",
        "registry_root",
        "output_path",
    ),
    [
        (
            "relative-artifacts",
            1,
            "decision.json",
            "registry",
            "invocation.json",
        ),
        (
            Path("relative-artifacts"),
            9999,
            Path("decision.json"),
            Path("registry"),
            Path("invocation.json"),
        ),
        (
            r".\artifacts",
            777,
            r".\decision.json",
            r".\registry",
            r".\invocation.json",
        ),
        (
            r"..\artifacts",
            888,
            r"..\decision.json",
            r"..\registry",
            r"..\invocation.json",
        ),
    ],
)
def test_inputs_are_forwarded_without_rewrite(
    artifact_root,
    end_round,
    source_decision,
    registry_root,
    output_path,
) -> None:
    request = _request()
    transport = _transport()

    request_source = RecordingRequestSource(
        request
    )

    transport_codec = RecordingTransportCodec(
        transport
    )

    file_carrier = RecordingFileCarrier(
        Path("returned.json")
    )

    adapter = _adapter(
        request_source=request_source,
        transport_codec=transport_codec,
        file_carrier=file_carrier,
    )

    adapter.adapt(
        artifact_root,
        end_round,
        source_decision=source_decision,
        registry_root=registry_root,
        output_path=output_path,
    )

    assert request_source.calls == [
        (
            artifact_root,
            end_round,
            source_decision,
            registry_root,
        )
    ]

    assert file_carrier.calls == [
        (
            output_path,
            transport,
        )
    ]


def test_request_source_exception_propagates_by_identity() -> None:
    failure = RuntimeError(
        "request-source-failure"
    )

    class FailingSource:
        def adapt(self, *args, **kwargs):
            raise failure

    adapter = _adapter(
        request_source=FailingSource(),
        transport_codec=RecordingTransportCodec(
            _transport()
        ),
        file_carrier=RecordingFileCarrier(
            Path("unused.json")
        ),
    )

    with pytest.raises(
        RuntimeError
    ) as exc_info:
        adapter.adapt(
            "artifact-root",
            1234,
            source_decision="decision.json",
            registry_root="registry",
            output_path="invocation.json",
        )

    assert exc_info.value is failure


def test_transport_codec_exception_propagates_by_identity() -> None:
    failure = ValueError(
        "transport-failure"
    )

    class FailingCodec:
        def encode(self, request):
            raise failure

    adapter = _adapter(
        request_source=RecordingRequestSource(
            _request()
        ),
        transport_codec=FailingCodec(),
        file_carrier=RecordingFileCarrier(
            Path("unused.json")
        ),
    )

    with pytest.raises(
        ValueError
    ) as exc_info:
        adapter.adapt(
            "artifact-root",
            1234,
            source_decision="decision.json",
            registry_root="registry",
            output_path="invocation.json",
        )

    assert exc_info.value is failure


def test_file_carrier_exception_propagates_by_identity() -> None:
    failure = OSError(
        "write-failure"
    )

    class FailingCarrier:
        def write(self, path, transport):
            raise failure

    adapter = _adapter(
        request_source=RecordingRequestSource(
            _request()
        ),
        transport_codec=RecordingTransportCodec(
            _transport()
        ),
        file_carrier=FailingCarrier(),
    )

    with pytest.raises(
        OSError
    ) as exc_info:
        adapter.adapt(
            "artifact-root",
            1234,
            source_decision="decision.json",
            registry_root="registry",
            output_path="invocation.json",
        )

    assert exc_info.value is failure


def test_source_contains_no_execution_service_dependency() -> None:
    source = MODULE_PATH.read_text(
        encoding="utf-8-sig"
    )

    assert (
        "DurableReplayPublicationInvocationExecutionService"
        not in source
    )


def test_source_contains_no_lifecycle_execution_dependency() -> None:
    source = MODULE_PATH.read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        "DurableReplayPublicationLifecycleEntrypoint",
        "ProductionLifecycleStageResult",
        ".run(",
        ".execute(",
    )

    for token in forbidden:
        assert token not in source


def test_source_contains_no_publisher_or_registry_mutation() -> None:
    source = MODULE_PATH.read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        "ProductionChampionRegistryPublisher",
        ".publish(",
        "rollback",
        "registry mutation",
    )

    lowered = source.lower()

    for token in forbidden:
        assert token.lower() not in lowered


def test_source_contains_no_cli_surface() -> None:
    source = MODULE_PATH.read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        "argparse",
        "sys.argv",
        "sys.stdout",
        "sys.stderr",
        "parse_args",
        "parse_known_args",
        "_COMMANDS",
    )

    for token in forbidden:
        assert token not in source


def test_source_contains_no_path_normalization_or_discovery() -> None:
    source = MODULE_PATH.read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        ".resolve(",
        ".absolute(",
        ".expanduser(",
        "expandvars",
        "glob(",
        "rglob(",
        "discover",
        "latest",
    )

    for token in forbidden:
        assert token not in source


def test_ast_owns_exact_composition_calls() -> None:
    source = MODULE_PATH.read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    classes = [
        node
        for node in tree.body
        if isinstance(
            node,
            ast.ClassDef,
        )
    ]

    assert len(classes) == 1

    cls = classes[0]

    adapt_methods = [
        node
        for node in cls.body
        if isinstance(
            node,
            ast.FunctionDef,
        )
        and node.name == "adapt"
    ]

    assert len(adapt_methods) == 1

    calls = [
        ast.unparse(node.func)
        for node in ast.walk(
            adapt_methods[0]
        )
        if isinstance(
            node,
            ast.Call,
        )
    ]

    assert calls == [
        "self._source_adapter.adapt",
        "self._transport_codec.encode",
        "self._file_carrier.write",
    ]


def test_ast_has_no_exception_translation() -> None:
    source = MODULE_PATH.read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    assert not any(
        isinstance(
            node,
            (
                ast.Try,
                ast.Raise,
            ),
        )
        for node in ast.walk(tree)
    )