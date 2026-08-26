from __future__ import annotations

import importlib
from pathlib import Path
from types import MappingProxyType

import pytest

from lrp.operations.durable_replay_promotion_publication_request import (
    DurableReplayPromotionPublicationRequest,
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


@pytest.mark.parametrize(
    "path_text",
    [
        r"relative\carrier.json",
        r".\relative\carrier.json",
        r"folder with spaces\carrier.json",
        r"%USERPROFILE%\carrier.json",
        r"~\carrier.json",
    ],
)
def test_input_path_text_is_forwarded_without_normalization(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    path_text: str,
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

    assert module.main(["--input", path_text]) == 0
    assert seen == [path_text]
    capsys.readouterr()


def test_success_stdout_contains_only_canonical_payload_and_one_lf(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    transport = _transport()

    class Carrier:
        def read(self, path: object) -> DurableReplayPublicationInvocationTransport:
            return transport

    class Codec:
        def encode(self, value: DurableReplayPublicationInvocationTransport) -> str:
            assert value is transport
            return '{"a":1}'

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

    assert module.main(["--input", "carrier.json"]) == 0
    captured = capsys.readouterr()

    assert captured.out == '{"a":1}\n'
    assert captured.err == ""


def test_missing_required_input_is_argparse_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()

    with pytest.raises(SystemExit) as exc:
        module.main([])

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--input" in captured.err


@pytest.mark.parametrize(
    "argv",
    [
        ["--unknown", "x"],
        ["--input"],
        ["--input", "a", "extra"],
    ],
)
def test_invalid_cli_shape_fails_in_argparse(
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
) -> None:
    module = _module()

    with pytest.raises(SystemExit) as exc:
        module.main(argv)

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err != ""


def test_bg_file_not_found_propagates_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    error = FileNotFoundError("missing carrier")

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


def test_bg_validation_error_propagates_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    error = ValueError("invalid file envelope")

    class Carrier:
        def read(self, path: object) -> DurableReplayPublicationInvocationTransport:
            raise error

    monkeypatch.setattr(
        module,
        "DurableReplayPublicationInvocationJsonFileCarrier",
        lambda: Carrier(),
    )

    with pytest.raises(ValueError) as exc:
        module.main(["--input", "bad.json"])

    assert exc.value is error


def test_bf_encode_error_propagates_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    transport = _transport()
    error = TypeError("invalid transport")

    class Carrier:
        def read(self, path: object) -> DurableReplayPublicationInvocationTransport:
            return transport

    class Codec:
        def encode(self, value: DurableReplayPublicationInvocationTransport) -> str:
            assert value is transport
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

    with pytest.raises(TypeError) as exc:
        module.main(["--input", "carrier.json"])

    assert exc.value is error


def test_cli_does_not_catch_domain_or_file_exceptions() -> None:
    source = Path("lrp/cli/durable_replay_publication_invocation_json_file.py").read_text(
        encoding="utf-8-sig"
    )
    assert "try:" not in source
    assert "except " not in source


def test_cli_has_exact_two_operational_dependencies() -> None:
    source = Path("lrp/cli/durable_replay_publication_invocation_json_file.py").read_text(
        encoding="utf-8-sig"
    )
    assert (
        "lrp.operations.durable_replay_publication_invocation_json_file_carrier"
        in source
    )
    assert (
        "lrp.operations.durable_replay_publication_invocation_json_presentation"
        in source
    )

    for token in [
        "durable_replay_publication_invocation_transport",
        "durable_replay_promotion_publication_request",
        "durable_replay_publication_lifecycle_entrypoint",
        "durable_replay_publication_lifecycle_adaptation",
        "durable_replay_promotion_publication_execution",
        "lrp.production",
    ]:
        assert token not in source


def test_cli_has_no_physical_file_io_of_its_own() -> None:
    source = Path("lrp/cli/durable_replay_publication_invocation_json_file.py").read_text(
        encoding="utf-8-sig"
    )
    for token in [
        "Path(",
        "open(",
        "read_text",
        "read_bytes",
        "write_text",
        "write_bytes",
    ]:
        assert token not in source


def test_cli_has_no_environment_or_path_discovery_surface() -> None:
    source = Path("lrp/cli/durable_replay_publication_invocation_json_file.py").read_text(
        encoding="utf-8-sig"
    )
    for token in [
        "resolve()",
        "expanduser",
        "getenv",
        "environ",
        "glob(",
        "rglob(",
        "cwd(",
        "home(",
        "latest",
        "discover",
    ]:
        assert token not in source


def test_cli_has_no_execution_or_mutation_surface() -> None:
    source = Path("lrp/cli/durable_replay_publication_invocation_json_file.py").read_text(
        encoding="utf-8-sig"
    )
    for token in [
        "DurableReplayPublicationLifecycleEntrypoint",
        "ProductionChampionRegistryPublisher",
        "run_publication_stage",
        "publish_champion",
        ".publish(",
        ".run(",
        "rollback",
        "eligibility",
        "promotion_policy",
        "source_decision=",
        "registry_root=",
    ]:
        assert token not in source


def test_existing_cli_owners_remain_independent() -> None:
    owners = {
        "lrp/cli/production_lifecycle.py": [
            "run_publication_stage",
        ],
        "lrp/cli/publish_champion.py": [
            "ProductionChampionRegistryPublisher",
        ],
        "lrp/cli/durable_replay_evaluation.py": [
            "argparse",
        ],
    }

    for path_text, required_tokens in owners.items():
        text = Path(path_text).read_text(encoding="utf-8-sig")
        assert "DurableReplayPublicationInvocationJsonFileCarrier" not in text
        for token in required_tokens:
            assert token in text
