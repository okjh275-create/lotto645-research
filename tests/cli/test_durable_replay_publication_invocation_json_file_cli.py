from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from lrp.operations.durable_replay_promotion_publication_request import (
    DurableReplayPromotionPublicationRequest,
)
from lrp.operations.durable_replay_publication_invocation_json_file_carrier import (
    DurableReplayPublicationInvocationJsonFileCarrier,
)
from lrp.operations.durable_replay_publication_invocation_json_presentation import (
    DurableReplayPublicationInvocationJsonCodec,
)
from lrp.operations.durable_replay_publication_invocation_transport import (
    DurableReplayPublicationInvocationTransport,
    DurableReplayPublicationInvocationTransportCodec,
)

MODULE_NAME = "lrp.cli.durable_replay_publication_invocation_json_file"


def _module():
    return importlib.import_module(MODULE_NAME)


def _transport() -> DurableReplayPublicationInvocationTransport:
    request = DurableReplayPromotionPublicationRequest(
        status="PASS",
        round_count=9,
        candidate_model_name="candidate-model",
        baseline_model_name="baseline-model",
        recommendation="eligible",
        action="prepare_publish",
        window=MappingProxyType(
            {
                "start_round": 1223,
                "end_round": 1231,
                "unicode": "서울-✓",
            }
        ),
        source_decision=Path("explicit/source_decision.json"),
        registry_root=Path("explicit/registry_root"),
    )
    return DurableReplayPublicationInvocationTransportCodec().encode(request)


def test_cli_product_module_exists() -> None:
    assert importlib.util.find_spec(MODULE_NAME) is not None


def test_cli_public_main_exists() -> None:
    module = _module()
    assert hasattr(module, "main")
    assert callable(module.main)


def test_main_signature_is_exact() -> None:
    module = _module()
    signature = inspect.signature(module.main)
    assert str(signature) == "(argv: 'Sequence[str] | None' = None) -> 'int'"


def test_cli_parser_requires_input_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    with pytest.raises(SystemExit) as exc:
        module.main([])
    assert exc.value.code != 0


def test_cli_input_path_is_forwarded_unchanged_to_bg(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    seen: list[object] = []

    class Carrier:
        def read(self, path: object) -> DurableReplayPublicationInvocationTransport:
            seen.append(path)
            return _transport()

    monkeypatch.setattr(
        module,
        "DurableReplayPublicationInvocationJsonFileCarrier",
        lambda: Carrier(),
    )

    exit_code = module.main(["--input", r"relative\explicit\carrier.json"])
    assert exit_code == 0
    assert seen == [r"relative\explicit\carrier.json"]

    captured = capsys.readouterr()
    assert captured.err == ""


def test_cli_calls_bg_read_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    calls = 0

    class Carrier:
        def read(self, path: object) -> DurableReplayPublicationInvocationTransport:
            nonlocal calls
            calls += 1
            return _transport()

    monkeypatch.setattr(
        module,
        "DurableReplayPublicationInvocationJsonFileCarrier",
        lambda: Carrier(),
    )

    assert module.main(["--input", "carrier.json"]) == 0
    assert calls == 1
    capsys.readouterr()


def test_cli_reuses_bf_canonical_json_for_stdout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    transport = _transport()

    class Carrier:
        def read(self, path: object) -> DurableReplayPublicationInvocationTransport:
            return transport

    monkeypatch.setattr(
        module,
        "DurableReplayPublicationInvocationJsonFileCarrier",
        lambda: Carrier(),
    )

    assert module.main(["--input", "carrier.json"]) == 0

    expected = DurableReplayPublicationInvocationJsonCodec().encode(transport)
    captured = capsys.readouterr()
    assert captured.out == expected + "\n"
    assert captured.err == ""


def test_cli_stdout_is_valid_exact_nine_field_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()

    class Carrier:
        def read(self, path: object) -> DurableReplayPublicationInvocationTransport:
            return _transport()

    monkeypatch.setattr(
        module,
        "DurableReplayPublicationInvocationJsonFileCarrier",
        lambda: Carrier(),
    )

    assert module.main(["--input", "carrier.json"]) == 0
    payload = capsys.readouterr().out
    decoded = json.loads(payload)

    assert set(decoded) == {
        "status",
        "round_count",
        "candidate_model_name",
        "baseline_model_name",
        "recommendation",
        "action",
        "window",
        "source_decision",
        "registry_root",
    }


def test_cli_success_has_exact_one_terminal_newline(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()

    class Carrier:
        def read(self, path: object) -> DurableReplayPublicationInvocationTransport:
            return _transport()

    monkeypatch.setattr(
        module,
        "DurableReplayPublicationInvocationJsonFileCarrier",
        lambda: Carrier(),
    )

    assert module.main(["--input", "carrier.json"]) == 0
    output = capsys.readouterr().out
    assert output.endswith("\n")
    assert not output.endswith("\n\n")


def test_bg_failure_propagates_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    error = FileNotFoundError("carrier missing")

    class Carrier:
        def read(self, path: object) -> DurableReplayPublicationInvocationTransport:
            raise error

    monkeypatch.setattr(
        module,
        "DurableReplayPublicationInvocationJsonFileCarrier",
        lambda: Carrier(),
    )

    with pytest.raises(FileNotFoundError) as exc:
        module.main(["--input", "missing.json"])

    assert exc.value is error


def test_bf_encode_failure_propagates_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    error = ValueError("bf encode failure")

    class Carrier:
        def read(self, path: object) -> DurableReplayPublicationInvocationTransport:
            return _transport()

    class Codec:
        def encode(self, transport: DurableReplayPublicationInvocationTransport) -> str:
            raise error

    monkeypatch.setattr(
        module,
        "DurableReplayPublicationInvocationJsonFileCarrier",
        lambda: Carrier(),
    )
    monkeypatch.setattr(
        module,
        "DurableReplayPublicationInvocationJsonCodec",
        lambda: Codec(),
    )

    with pytest.raises(ValueError) as exc:
        module.main(["--input", "carrier.json"])

    assert exc.value is error


def test_cli_does_not_construct_transport_or_ba_request() -> None:
    source = Path("lrp/cli/durable_replay_publication_invocation_json_file.py").read_text(
        encoding="utf-8-sig"
    )
    assert "DurableReplayPromotionPublicationRequest(" not in source
    assert "DurableReplayPublicationInvocationTransport(" not in source


def test_cli_has_no_write_create_or_carrier_mutation_surface() -> None:
    import ast

    source = Path("lrp/cli/durable_replay_publication_invocation_json_file.py").read_text(
        encoding="utf-8-sig"
    )
    tree = ast.parse(source)

    def call_name(node: ast.Call) -> str:
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            parts: list[str] = []
            current: ast.expr = func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        return "<dynamic>"

    calls = {
        call_name(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }

    assert "sys.stdout.write" in calls

    forbidden_calls = {
        "open",
        "Path.open",
        "Path.write_text",
        "Path.write_bytes",
        "Path.mkdir",
        "Path.replace",
        "Path.unlink",
        "os.replace",
    }
    assert calls.isdisjoint(forbidden_calls)

    for token in [
        'open("w"',
        'open("x"',
        "write_text",
        "write_bytes",
        "mkdir(",
        "replace(",
        "unlink(",
    ]:
        assert token not in source

def test_cli_has_no_lifecycle_publisher_or_production_dependency() -> None:
    source = Path("lrp/cli/durable_replay_publication_invocation_json_file.py").read_text(
        encoding="utf-8-sig"
    )
    for token in [
        "DurableReplayPublicationLifecycleEntrypoint",
        "run_publication_stage",
        "ProductionChampionRegistryPublisher",
        "publish_champion",
        "lrp.production",
        ".publish(",
        ".run(",
    ]:
        assert token not in source


def test_cli_has_no_discovery_defaulting_policy_or_rollback_surface() -> None:
    source = Path("lrp/cli/durable_replay_publication_invocation_json_file.py").read_text(
        encoding="utf-8-sig"
    )
    for token in [
        "resolve()",
        "expanduser",
        "getenv",
        "environ",
        "discover",
        "latest",
        "glob(",
        "rglob(",
        "eligibility",
        "promotion_policy",
        "rollback",
        "source_decision=",
        "registry_root=",
    ]:
        assert token not in source


def test_cli_has_no_stdin_transport_input_surface() -> None:
    source = Path("lrp/cli/durable_replay_publication_invocation_json_file.py").read_text(
        encoding="utf-8-sig"
    )
    assert "stdin" not in source
    assert "sys.stdin" not in source


def test_cli_dependencies_are_exact() -> None:
    source = Path("lrp/cli/durable_replay_publication_invocation_json_file.py").read_text(
        encoding="utf-8-sig"
    )
    assert "import argparse" in source
    assert "import sys" in source
    assert (
        "lrp.operations.durable_replay_publication_invocation_json_file_carrier"
        in source
    )
    assert (
        "lrp.operations.durable_replay_publication_invocation_json_presentation"
        in source
    )


def test_cli_does_not_modify_existing_cli_owners() -> None:
    paths = [
        Path("lrp/cli/production_lifecycle.py"),
        Path("lrp/cli/publish_champion.py"),
        Path("lrp/cli/durable_replay_evaluation.py"),
    ]
    for path in paths:
        assert path.exists()
