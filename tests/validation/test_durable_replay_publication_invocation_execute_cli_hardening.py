from __future__ import annotations

import ast
import inspect

import pytest

import lrp.cli.durable_replay_publication_invocation_execute as product


class _ExecutionStub:
    def __init__(
        self,
        *,
        result: object | None = None,
        failure: BaseException | None = None,
    ) -> None:
        self.result = result
        self.failure = failure
        self.calls: list[object] = []

    def execute(self, path: object) -> object:
        self.calls.append(path)

        if self.failure is not None:
            raise self.failure

        return self.result


@pytest.mark.parametrize(
    "path",
    [
        "",
        " ",
        "invocation.json",
        r".\invocation.json",
        r"..\payload\invocation.json",
        r"%USERPROFILE%\invocation.json",
        "payload/../payload/invocation.json",
        "./payload/invocation.json",
        "payload//invocation.json",
    ],
)
def test_unusual_input_identity_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    stub = _ExecutionStub(result=object())

    monkeypatch.setattr(
        product,
        "_build_execution_service",
        lambda: stub,
    )

    actual = product.main(
        [
            "--input",
            path,
        ]
    )

    assert actual == 0
    assert stub.calls == [path]
    assert stub.calls[0] is path


def test_execution_service_is_built_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _ExecutionStub(result=object())
    builds: list[None] = []

    def build():
        builds.append(None)
        return stub

    monkeypatch.setattr(
        product,
        "_build_execution_service",
        build,
    )

    actual = product.main(
        [
            "--input",
            "invocation.json",
        ]
    )

    assert actual == 0
    assert builds == [None]
    assert stub.calls == ["invocation.json"]


def test_execution_service_is_called_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _ExecutionStub(result=object())

    monkeypatch.setattr(
        product,
        "_build_execution_service",
        lambda: stub,
    )

    actual = product.main(
        [
            "--input",
            "invocation.json",
        ]
    )

    assert actual == 0
    assert stub.calls == ["invocation.json"]


def test_no_implicit_retry_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = RuntimeError("execution failure")
    stub = _ExecutionStub(failure=failure)

    monkeypatch.setattr(
        product,
        "_build_execution_service",
        lambda: stub,
    )

    with pytest.raises(RuntimeError) as exc_info:
        product.main(
            [
                "--input",
                "invocation.json",
            ]
        )

    assert exc_info.value is failure
    assert stub.calls == ["invocation.json"]


def test_repeated_main_calls_are_composition_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _ExecutionStub(result=object())

    monkeypatch.setattr(
        product,
        "_build_execution_service",
        lambda: stub,
    )

    first = product.main(
        [
            "--input",
            "invocation.json",
        ]
    )

    second = product.main(
        [
            "--input",
            "invocation.json",
        ]
    )

    assert first == 0
    assert second == 0

    assert stub.calls == [
        "invocation.json",
        "invocation.json",
    ]


def test_public_surface_remains_main_only() -> None:
    public_functions = {
        name
        for name, value in vars(product).items()
        if inspect.isfunction(value)
        and value.__module__ == product.__name__
        and not name.startswith("_")
    }

    assert public_functions == {"main"}


def test_private_helper_surface_remains_exact() -> None:
    private_functions = {
        name
        for name, value in vars(product).items()
        if inspect.isfunction(value)
        and value.__module__ == product.__name__
        and name.startswith("_")
    }

    assert private_functions == {
        "_parser",
        "_build_execution_service",
    }


def test_main_signature_remains_exact() -> None:
    assert str(inspect.signature(product.main)) == (
        "(argv: 'Sequence[str] | None' = None) -> 'int'"
    )


def test_parser_surface_remains_exact() -> None:
    parser = product._parser()

    args = parser.parse_args(
        [
            "--input",
            "invocation.json",
        ]
    )

    assert vars(args) == {
        "input": "invocation.json",
    }


@pytest.mark.parametrize(
    "extra_arg",
    [
        "--artifact-root",
        "--registry-root",
        "--source-decision",
        "--latest",
        "--discover",
        "--selector",
        "--stdin",
    ],
)
def test_forbidden_cli_arguments_are_rejected(
    extra_arg: str,
) -> None:
    parser = product._parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--input",
                "invocation.json",
                extra_arg,
                "x",
            ]
        )


def test_main_call_graph_remains_exact() -> None:
    tree = ast.parse(inspect.getsource(product))

    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "main"
    )

    calls = [
        ast.unparse(node.func)
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
    ]

    assert calls == [
        "_parser",
        "parser.parse_args",
        "_build_execution_service",
        "service.execute",
    ]


def test_dependency_construction_graph_remains_exact() -> None:
    tree = ast.parse(inspect.getsource(product))

    builder = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_build_execution_service"
    )

    calls = [
        ast.unparse(node.func)
        for node in ast.walk(builder)
        if isinstance(node, ast.Call)
    ]

    assert calls == [
        "DurableReplayPublicationInvocationJsonFileCarrier",
        "DurableReplayPublicationInvocationTransportCodec",
        "DurableReplayPublicationLifecycleAdaptationService",
        "DurableReplayPublicationLifecycleEntrypoint",
        "DurableReplayPublicationInvocationExecutionService",
    ]


def test_no_exception_translation_or_cleanup() -> None:
    tree = ast.parse(inspect.getsource(product))

    assert not any(
        isinstance(
            node,
            (
                ast.Try,
                ast.Raise,
                ast.With,
                ast.AsyncWith,
            ),
        )
        for node in ast.walk(tree)
    )


def test_no_direct_file_or_json_ownership() -> None:
    source = inspect.getsource(product).lower()

    forbidden = (
        "open(",
        "read_text(",
        "read_bytes(",
        "json.load",
        "json.loads",
        "json.dump",
        "json.dumps",
    )

    for token in forbidden:
        assert token not in source


def test_no_path_normalization_or_discovery() -> None:
    source = inspect.getsource(product).lower()

    forbidden = (
        "resolve(",
        "absolute(",
        "expanduser",
        "expandvars",
        "getenv",
        "environ",
        "cwd",
        "home(",
        "glob(",
        "rglob(",
        "discover",
        "latest",
    )

    for token in forbidden:
        assert token not in source


def test_no_direct_publication_or_policy_ownership() -> None:
    source = inspect.getsource(product).lower()

    forbidden = (
        "productionchampionregistrypublisher",
        "run_publication_stage",
        ".publish(",
        "promotioneligibility",
        "promotionactionplan",
        "candidate_advantage_count",
        "baseline_advantage_count",
        "baseline_delta",
        "rollback",
    )

    for token in forbidden:
        assert token not in source


def test_no_stdout_stdin_or_result_renderer() -> None:
    source = inspect.getsource(product).lower()

    forbidden = (
        "sys.stdout",
        "sys.stdin",
        "print(",
        "presentation",
        "serializer",
        "render",
    )

    for token in forbidden:
        assert token not in source


def test_no_inspection_cli_coupling() -> None:
    source = inspect.getsource(product)

    assert (
        "durable_replay_publication_invocation_json_file"
        not in source.replace(
            "durable_replay_publication_invocation_json_file_carrier",
            "",
        )
    )


def test_no_root_cli_coupling() -> None:
    source = inspect.getsource(product)

    assert "lrp.cli.__init__" not in source


def test_builder_returns_bs_execution_service() -> None:
    service = product._build_execution_service()

    assert service.__class__.__name__ == (
        "DurableReplayPublicationInvocationExecutionService"
    )


def test_builder_constructs_only_existing_authoritative_layers() -> None:
    source = inspect.getsource(
        product._build_execution_service
    )

    required = (
        "DurableReplayPublicationInvocationJsonFileCarrier",
        "DurableReplayPublicationInvocationTransportCodec",
        "DurableReplayPublicationLifecycleAdaptationService",
        "DurableReplayPublicationLifecycleEntrypoint",
        "DurableReplayPublicationInvocationExecutionService",
    )

    for token in required:
        assert token in source


def test_main_has_no_result_interpretation_branch() -> None:
    tree = ast.parse(inspect.getsource(product))

    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "main"
    )

    assert not any(
        isinstance(
            node,
            (
                ast.If,
                ast.Match,
            ),
        )
        for node in ast.walk(main)
    )