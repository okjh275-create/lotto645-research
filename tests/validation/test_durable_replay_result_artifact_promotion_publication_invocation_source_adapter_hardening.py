from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import lrp.operations.durable_replay_result_artifact_promotion_publication_invocation_source_adapter as product


MODULE_PATH = Path(
    "lrp/operations/"
    "durable_replay_result_artifact_promotion_"
    "publication_invocation_source_adapter.py"
)

CLASS_NAME = (
    "DurableReplayResultArtifactPromotion"
    "PublicationInvocationSourceAdapter"
)


class RequestSource:
    def __init__(self, request: object) -> None:
        self.request = request
        self.calls: list[tuple[object, object, object, object]] = []

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


class TransportCodec:
    def __init__(self, transport: object) -> None:
        self.transport = transport
        self.calls: list[object] = []

    def encode(self, request):
        self.calls.append(request)
        return self.transport


class FileCarrier:
    def __init__(self, returned_path: Path) -> None:
        self.returned_path = returned_path
        self.calls: list[tuple[object, object]] = []

    def write(self, path, transport):
        self.calls.append(
            (
                path,
                transport,
            )
        )
        return self.returned_path


def _class():
    return getattr(
        product,
        CLASS_NAME,
    )


def _source() -> str:
    return MODULE_PATH.read_text(
        encoding="utf-8-sig"
    )


def _tree() -> ast.Module:
    return ast.parse(
        _source()
    )


def _class_node() -> ast.ClassDef:
    classes = [
        node
        for node in _tree().body
        if isinstance(
            node,
            ast.ClassDef,
        )
    ]

    assert len(classes) == 1

    return classes[0]


def _method(name: str) -> ast.FunctionDef:
    cls = _class_node()

    nodes = [
        node
        for node in cls.body
        if isinstance(
            node,
            ast.FunctionDef,
        )
        and node.name == name
    ]

    assert len(nodes) == 1

    return nodes[0]


def test_product_class_name_is_exact() -> None:
    cls = _class()

    assert cls.__name__ == CLASS_NAME


def test_product_exposes_only_adapt_public_method() -> None:
    cls = _class()

    public = [
        name
        for name, member in inspect.getmembers(
            cls,
            predicate=inspect.isfunction,
        )
        if not name.startswith("_")
    ]

    assert public == [
        "adapt",
    ]


def test_constructor_parameter_order_is_exact() -> None:
    signature = inspect.signature(
        _class().__init__
    )

    assert list(
        signature.parameters
    ) == [
        "self",
        "source_adapter",
        "transport_codec",
        "file_carrier",
    ]


def test_adapt_parameter_order_and_keyword_contract_is_exact() -> None:
    signature = inspect.signature(
        _class().adapt
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

    for name in (
        "source_decision",
        "registry_root",
        "output_path",
    ):
        assert (
            signature.parameters[name].kind
            is inspect.Parameter.KEYWORD_ONLY
        )


def test_default_constructor_builds_exact_three_dependencies() -> None:
    init_method = _method(
        "__init__"
    )

    calls = [
        ast.unparse(node.func)
        for node in ast.walk(
            init_method
        )
        if isinstance(
            node,
            ast.Call,
        )
    ]

    assert calls == [
        "DurableReplayResultArtifactPromotionPublicationRequestSourceAdapter",
        "DurableReplayPublicationInvocationTransportCodec",
        "DurableReplayPublicationInvocationJsonFileCarrier",
    ]


def test_adapt_call_graph_is_exact_and_ordered() -> None:
    adapt_method = _method(
        "adapt"
    )

    calls = [
        ast.unparse(node.func)
        for node in ast.walk(
            adapt_method
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


def test_opaque_identity_flows_request_to_codec_to_carrier() -> None:
    request = object()
    transport = object()
    returned_path = Path(
        "returned.json"
    )

    request_source = RequestSource(
        request
    )
    codec = TransportCodec(
        transport
    )
    carrier = FileCarrier(
        returned_path
    )

    adapter = _class()(
        source_adapter=request_source,
        transport_codec=codec,
        file_carrier=carrier,
    )

    result = adapter.adapt(
        "artifact-root",
        100,
        source_decision="decision.json",
        registry_root="registry",
        output_path="invocation.json",
    )

    assert codec.calls == [
        request,
    ]

    assert carrier.calls == [
        (
            "invocation.json",
            transport,
        )
    ]

    assert result is returned_path


@pytest.mark.parametrize(
    (
        "artifact_root",
        "source_decision",
        "registry_root",
        "output_path",
    ),
    [
        (
            "",
            "",
            "",
            "",
        ),
        (
            ".",
            "./decision.json",
            "./registry",
            "./invocation.json",
        ),
        (
            "..",
            "../decision.json",
            "../registry",
            "../invocation.json",
        ),
        (
            r".\artifact-root",
            r".\decision.json",
            r".\registry",
            r".\invocation.json",
        ),
        (
            r"..\artifact-root",
            r"..\decision.json",
            r"..\registry",
            r"..\invocation.json",
        ),
        (
            r"%USERPROFILE%\artifact-root",
            r"%USERPROFILE%\decision.json",
            r"%USERPROFILE%\registry",
            r"%USERPROFILE%\invocation.json",
        ),
    ],
)
def test_string_inputs_are_forwarded_without_rewrite(
    artifact_root,
    source_decision,
    registry_root,
    output_path,
) -> None:
    request = object()
    transport = object()

    source = RequestSource(
        request
    )
    codec = TransportCodec(
        transport
    )
    carrier = FileCarrier(
        Path("returned.json")
    )

    adapter = _class()(
        source_adapter=source,
        transport_codec=codec,
        file_carrier=carrier,
    )

    adapter.adapt(
        artifact_root,
        321,
        source_decision=source_decision,
        registry_root=registry_root,
        output_path=output_path,
    )

    assert source.calls == [
        (
            artifact_root,
            321,
            source_decision,
            registry_root,
        )
    ]

    assert carrier.calls == [
        (
            output_path,
            transport,
        )
    ]


def test_path_inputs_are_forwarded_by_identity() -> None:
    artifact_root = Path(
        "artifact-root"
    )
    source_decision = Path(
        "decision.json"
    )
    registry_root = Path(
        "registry"
    )
    output_path = Path(
        "invocation.json"
    )

    request = object()
    transport = object()

    source = RequestSource(
        request
    )
    codec = TransportCodec(
        transport
    )
    carrier = FileCarrier(
        Path("returned.json")
    )

    adapter = _class()(
        source_adapter=source,
        transport_codec=codec,
        file_carrier=carrier,
    )

    adapter.adapt(
        artifact_root,
        555,
        source_decision=source_decision,
        registry_root=registry_root,
        output_path=output_path,
    )

    call = source.calls[0]

    assert call[0] is artifact_root
    assert call[2] is source_decision
    assert call[3] is registry_root

    assert carrier.calls[0][0] is output_path


@pytest.mark.parametrize(
    "end_round",
    [
        -1,
        0,
        1,
        999999,
    ],
)
def test_end_round_is_forwarded_without_validation(
    end_round: int,
) -> None:
    source = RequestSource(
        object()
    )

    adapter = _class()(
        source_adapter=source,
        transport_codec=TransportCodec(
            object()
        ),
        file_carrier=FileCarrier(
            Path("returned.json")
        ),
    )

    adapter.adapt(
        "artifact-root",
        end_round,
        source_decision="decision.json",
        registry_root="registry",
        output_path="invocation.json",
    )

    assert source.calls[0][1] == end_round


def test_request_source_failure_short_circuits_downstream() -> None:
    failure = RuntimeError(
        "request failure"
    )

    class FailingSource:
        def adapt(self, *args, **kwargs):
            raise failure

    codec = TransportCodec(
        object()
    )
    carrier = FileCarrier(
        Path("unused.json")
    )

    adapter = _class()(
        source_adapter=FailingSource(),
        transport_codec=codec,
        file_carrier=carrier,
    )

    with pytest.raises(
        RuntimeError
    ) as exc_info:
        adapter.adapt(
            "artifact-root",
            100,
            source_decision="decision.json",
            registry_root="registry",
            output_path="invocation.json",
        )

    assert exc_info.value is failure
    assert codec.calls == []
    assert carrier.calls == []


def test_transport_failure_short_circuits_file_write() -> None:
    failure = ValueError(
        "codec failure"
    )

    class FailingCodec:
        def encode(self, request):
            raise failure

    carrier = FileCarrier(
        Path("unused.json")
    )

    adapter = _class()(
        source_adapter=RequestSource(
            object()
        ),
        transport_codec=FailingCodec(),
        file_carrier=carrier,
    )

    with pytest.raises(
        ValueError
    ) as exc_info:
        adapter.adapt(
            "artifact-root",
            100,
            source_decision="decision.json",
            registry_root="registry",
            output_path="invocation.json",
        )

    assert exc_info.value is failure
    assert carrier.calls == []


def test_file_write_failure_propagates_by_identity() -> None:
    failure = OSError(
        "file failure"
    )

    class FailingCarrier:
        def write(self, path, transport):
            raise failure

    adapter = _class()(
        source_adapter=RequestSource(
            object()
        ),
        transport_codec=TransportCodec(
            object()
        ),
        file_carrier=FailingCarrier(),
    )

    with pytest.raises(
        OSError
    ) as exc_info:
        adapter.adapt(
            "artifact-root",
            100,
            source_decision="decision.json",
            registry_root="registry",
            output_path="invocation.json",
        )

    assert exc_info.value is failure


def test_no_exception_translation_exists_in_product_ast() -> None:
    tree = _tree()

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


def test_no_conditional_business_policy_in_adapt() -> None:
    adapt_method = _method(
        "adapt"
    )

    assert not any(
        isinstance(
            node,
            (
                ast.If,
                ast.Match,
                ast.For,
                ast.While,
            ),
        )
        for node in ast.walk(
            adapt_method
        )
    )


def test_product_does_not_execute_lifecycle_or_invocation() -> None:
    source = _source()

    forbidden = (
        "DurableReplayPublicationLifecycleEntrypoint",
        "DurableReplayPublicationInvocationExecutionService",
        ".execute(",
        ".run(",
    )

    for token in forbidden:
        assert token not in source


def test_product_does_not_own_publication_policy() -> None:
    source = _source()

    forbidden = (
        "ProductionChampionRegistryPublisher",
        ".publish(",
        "PromotionEligibility",
        "PromotionActionPlan",
        "candidate_advantage",
        "baseline_advantage",
        "baseline_delta",
        "rollback",
    )

    lowered = source.lower()

    for token in forbidden:
        assert token.lower() not in lowered


def test_product_does_not_own_cli_or_root_registration() -> None:
    source = _source()

    forbidden = (
        "argparse",
        "sys.argv",
        "sys.stdout",
        "sys.stderr",
        "_COMMANDS",
        "parse_args",
        "parse_known_args",
    )

    for token in forbidden:
        assert token not in source


def test_product_does_not_own_json_serialization() -> None:
    source = _source()

    forbidden = (
        "json.dumps",
        "json.dump",
        "json.loads",
        "json.load",
        "write_text(",
        "write_bytes(",
        "open(",
    )

    for token in forbidden:
        assert token not in source


def test_product_does_not_normalize_or_discover_paths() -> None:
    source = _source().lower()

    forbidden = (
        ".resolve(",
        ".absolute(",
        ".expanduser(",
        "expandvars",
        "getenv",
        "environ",
        "glob(",
        "rglob(",
        "discover",
        "latest",
    )

    for token in forbidden:
        assert token not in source


def test_product_import_surface_is_exact() -> None:
    tree = _tree()

    imports: list[str] = []

    for node in tree.body:
        if isinstance(
            node,
            ast.ImportFrom,
        ):
            if node.module == "__future__":
                continue

            imports.append(
                node.module or ""
            )

    assert imports == [
        "pathlib",
        "lrp.operations.durable_replay_publication_invocation_json_file_carrier",
        "lrp.operations.durable_replay_publication_invocation_transport",
        "lrp.operations.durable_replay_result_artifact_promotion_publication_request_source_adapter",
    ]


def test_product_contains_no_additional_top_level_functions() -> None:
    tree = _tree()

    top_level_functions = [
        node
        for node in tree.body
        if isinstance(
            node,
            ast.FunctionDef,
        )
    ]

    assert top_level_functions == []


def test_product_contains_no_additional_classes() -> None:
    classes = [
        node
        for node in _tree().body
        if isinstance(
            node,
            ast.ClassDef,
        )
    ]

    assert len(classes) == 1


def test_adapt_returns_direct_file_carrier_write_result() -> None:
    adapt_method = _method(
        "adapt"
    )

    returns = [
        node
        for node in adapt_method.body
        if isinstance(
            node,
            ast.Return,
        )
    ]

    assert len(returns) == 1

    value = returns[0].value

    assert isinstance(
        value,
        ast.Call,
    )

    assert ast.unparse(
        value.func
    ) == "self._file_carrier.write"