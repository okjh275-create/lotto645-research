from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest

import lrp.operations.durable_replay_result_artifact_promotion_publication_invocation_execution_bridge as product


MODULE_PATH = Path(
    "lrp/operations/"
    "durable_replay_result_artifact_promotion_"
    "publication_invocation_execution_bridge.py"
)

CLASS_NAME = (
    "DurableReplayResultArtifactPromotionPublication"
    "InvocationExecutionBridge"
)


class SourceSpy:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[
            tuple[
                tuple[object, object],
                dict[str, object],
            ]
        ] = []

    def adapt(
        self,
        artifact_root,
        end_round,
        *,
        source_decision,
        registry_root,
        output_path,
    ):
        self.calls.append(
            (
                (
                    artifact_root,
                    end_round,
                ),
                {
                    "source_decision": source_decision,
                    "registry_root": registry_root,
                    "output_path": output_path,
                },
            )
        )

        return self.result


class ExecutionSpy:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[object] = []

    def execute(self, path):
        self.calls.append(path)
        return self.result


def _class():
    return getattr(
        product,
        CLASS_NAME,
    )


def _source_text() -> str:
    return MODULE_PATH.read_text(
        encoding="utf-8-sig"
    )


def _tree() -> ast.Module:
    return ast.parse(
        _source_text()
    )


def _product_class_node() -> ast.ClassDef:
    classes = [
        node
        for node in _tree().body
        if isinstance(node, ast.ClassDef)
        and node.name == CLASS_NAME
    ]

    assert len(classes) == 1

    return classes[0]


def _method(name: str) -> ast.FunctionDef:
    cls = _product_class_node()

    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == name
    ]

    assert len(methods) == 1

    return methods[0]


def test_exact_product_class_exists() -> None:
    cls = _class()

    assert inspect.isclass(cls)
    assert cls.__name__ == CLASS_NAME


def test_execute_is_only_public_product_method() -> None:
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
        "execute",
    ]


def test_constructor_requires_exact_two_injected_dependencies() -> None:
    signature = inspect.signature(
        _class().__init__
    )

    assert list(
        signature.parameters
    ) == [
        "self",
        "source_adapter",
        "execution_service",
    ]

    assert (
        signature.parameters[
            "source_adapter"
        ].default
        is inspect.Parameter.empty
    )

    assert (
        signature.parameters[
            "execution_service"
        ].default
        is inspect.Parameter.empty
    )


def test_execute_signature_is_exact() -> None:
    signature = inspect.signature(
        _class().execute
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


def test_init_performs_no_dependency_construction() -> None:
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

    assert calls == []


def test_execute_call_graph_is_exact_two_calls() -> None:
    execute_method = _method(
        "execute"
    )

    calls = [
        ast.unparse(node.func)
        for node in ast.walk(
            execute_method
        )
        if isinstance(
            node,
            ast.Call,
        )
    ]

    assert calls == [
        "self._source_adapter.adapt",
        "self._execution_service.execute",
    ]


@pytest.mark.parametrize(
    "end_round",
    [
        -1,
        0,
        1,
        999999,
    ],
)
def test_end_round_is_forwarded_without_bridge_validation(
    end_round: int,
) -> None:
    source = SourceSpy(
        object()
    )

    bridge = _class()(
        source_adapter=source,
        execution_service=ExecutionSpy(
            object()
        ),
    )

    bridge.execute(
        "artifact-root",
        end_round,
        source_decision="decision.json",
        registry_root="registry",
        output_path="invocation.json",
    )

    assert source.calls[0][0][1] == end_round


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
            r".\artifacts",
            r".\decision.json",
            r".\registry",
            r".\invocation.json",
        ),
        (
            r"..\artifacts",
            r"..\decision.json",
            r"..\registry",
            r"..\invocation.json",
        ),
        (
            r"%USERPROFILE%\artifacts",
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
    source_result = object()

    source = SourceSpy(
        source_result
    )

    execution = ExecutionSpy(
        object()
    )

    bridge = _class()(
        source_adapter=source,
        execution_service=execution,
    )

    bridge.execute(
        artifact_root,
        123,
        source_decision=source_decision,
        registry_root=registry_root,
        output_path=output_path,
    )

    assert source.calls == [
        (
            (
                artifact_root,
                123,
            ),
            {
                "source_decision": source_decision,
                "registry_root": registry_root,
                "output_path": output_path,
            },
        )
    ]

    assert execution.calls == [
        source_result,
    ]


def test_path_inputs_preserve_identity_to_source_owner() -> None:
    artifact_root = Path(
        "artifacts"
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

    source = SourceSpy(
        Path("actual-result.json")
    )

    bridge = _class()(
        source_adapter=source,
        execution_service=ExecutionSpy(
            object()
        ),
    )

    bridge.execute(
        artifact_root,
        321,
        source_decision=source_decision,
        registry_root=registry_root,
        output_path=output_path,
    )

    positional, keywords = source.calls[0]

    assert positional[0] is artifact_root
    assert keywords["source_decision"] is source_decision
    assert keywords["registry_root"] is registry_root
    assert keywords["output_path"] is output_path


def test_exact_source_return_object_is_passed_to_execution() -> None:
    source_result = object()

    source = SourceSpy(
        source_result
    )

    execution = ExecutionSpy(
        object()
    )

    bridge = _class()(
        source_adapter=source,
        execution_service=execution,
    )

    bridge.execute(
        "artifact-root",
        123,
        source_decision="decision.json",
        registry_root="registry",
        output_path="requested.json",
    )

    assert execution.calls == [
        source_result,
    ]

    assert execution.calls[0] is source_result


def test_execution_result_is_returned_by_identity() -> None:
    execution_result = object()

    bridge = _class()(
        source_adapter=SourceSpy(
            object()
        ),
        execution_service=ExecutionSpy(
            execution_result
        ),
    )

    result = bridge.execute(
        "artifact-root",
        123,
        source_decision="decision.json",
        registry_root="registry",
        output_path="invocation.json",
    )

    assert result is execution_result


def test_source_failure_propagates_and_execution_is_not_called() -> None:
    failure = RuntimeError(
        "source failure"
    )

    class FailingSource:
        def adapt(self, *args: Any, **kwargs: Any):
            raise failure

    execution = ExecutionSpy(
        object()
    )

    bridge = _class()(
        source_adapter=FailingSource(),
        execution_service=execution,
    )

    with pytest.raises(
        RuntimeError
    ) as exc_info:
        bridge.execute(
            "artifact-root",
            123,
            source_decision="decision.json",
            registry_root="registry",
            output_path="invocation.json",
        )

    assert exc_info.value is failure
    assert execution.calls == []


def test_execution_failure_propagates_by_identity() -> None:
    source_result = object()
    failure = OSError(
        "execution failure"
    )

    class FailingExecution:
        def execute(self, path):
            assert path is source_result
            raise failure

    bridge = _class()(
        source_adapter=SourceSpy(
            source_result
        ),
        execution_service=FailingExecution(),
    )

    with pytest.raises(
        OSError
    ) as exc_info:
        bridge.execute(
            "artifact-root",
            123,
            source_decision="decision.json",
            registry_root="registry",
            output_path="invocation.json",
        )

    assert exc_info.value is failure


def test_product_ast_contains_no_exception_translation() -> None:
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


def test_execute_contains_no_business_branching() -> None:
    execute_method = _method(
        "execute"
    )

    assert not any(
        isinstance(
            node,
            (
                ast.If,
                ast.For,
                ast.While,
                ast.Match,
            ),
        )
        for node in ast.walk(
            execute_method
        )
    )


def test_product_does_not_construct_closed_owners() -> None:
    source = _source_text()

    forbidden = (
        "DurableReplayResultArtifactPromotionPublicationInvocationSourceAdapter(",
        "DurableReplayPublicationInvocationExecutionService(",
        "DurableReplayPublicationInvocationJsonFileCarrier(",
        "DurableReplayPublicationInvocationTransportCodec(",
        "DurableReplayPublicationLifecycleEntrypoint(",
        "DurableReplayPublicationLifecycleAdaptationService(",
    )

    for token in forbidden:
        assert token not in source


def test_product_does_not_own_file_or_transport_logic() -> None:
    source = _source_text()

    forbidden = (
        ".read(",
        ".write(",
        ".encode(",
        ".decode(",
        "json.dump",
        "json.dumps",
        "json.load",
        "json.loads",
        "open(",
        "write_text(",
        "write_bytes(",
    )

    for token in forbidden:
        assert token not in source


def test_product_does_not_own_lifecycle_or_publication_logic() -> None:
    source = _source_text()

    forbidden = (
        ".run(",
        ".publish(",
        "ProductionChampionRegistryPublisher",
        "PromotionEligibility",
        "PromotionActionPlan",
        "rollback",
    )

    lowered = source.lower()

    for token in forbidden:
        assert token.lower() not in lowered


def test_product_does_not_own_cli_or_root_surface() -> None:
    source = _source_text()

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


def test_product_does_not_normalize_or_discover_paths() -> None:
    source = _source_text().lower()

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


def test_execute_returns_direct_execution_call_result() -> None:
    execute_method = _method(
        "execute"
    )

    returns = [
        node
        for node in execute_method.body
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
    ) == "self._execution_service.execute"


def test_no_additional_public_top_level_functions_exist() -> None:
    tree = _tree()

    public_functions = [
        node.name
        for node in tree.body
        if isinstance(
            node,
            ast.FunctionDef,
        )
        and not node.name.startswith("_")
    ]

    assert public_functions == []


def test_protocols_are_private_support_types() -> None:
    tree = _tree()

    class_names = [
        node.name
        for node in tree.body
        if isinstance(
            node,
            ast.ClassDef,
        )
    ]

    assert CLASS_NAME in class_names

    support_classes = [
        name
        for name in class_names
        if name != CLASS_NAME
    ]

    assert support_classes == [
        "_InvocationSourceAdapter",
        "_InvocationExecutionService",
    ]

    assert all(
        name.startswith("_")
        for name in support_classes
    )