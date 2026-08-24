from __future__ import annotations

import argparse
import importlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


_PRODUCT_MODULE = "lrp.cli.durable_replay_evaluation"

_DISPATCHER_PATH = Path(
    "lrp/cli/__init__.py"
)


def _product():
    return importlib.import_module(
        _PRODUCT_MODULE
    )


def _service_result() -> Any:
    return SimpleNamespace(
        evaluation={
            "mean_hits": 2.5,
            "windows": 4,
        },
        candidate_model_name="candidate-A",
        baseline_model_name="baseline-A",
        round_count=2,
    )


def _argv() -> list[str]:
    return [
        "--history",
        "data/history.json",
        "--window-name",
        "window-001",
        "--start-round",
        "1230",
        "--end-round",
        "1231",
        "--candidate",
        "artifacts/candidate-1.json|1230|candidate-A",
        "--candidate",
        (
            "artifacts/candidate-2.json"
            "|1231"
            "|candidate-A"
            "|regime-A"
            "|strategy-A"
        ),
        "--baseline",
        "artifacts/baseline-1.json|1230|baseline-A",
    ]


def _patch_service(
    monkeypatch: pytest.MonkeyPatch,
    product: Any,
    *,
    result: Any | None = None,
) -> list[Any]:
    calls: list[Any] = []

    returned = (
        _service_result()
        if result is None
        else result
    )

    def fake_execute(
        self: Any,
        *,
        request: Any,
    ) -> Any:
        calls.append(request)
        return returned

    monkeypatch.setattr(
        product.DurableReplayExecutionService,
        "execute",
        fake_execute,
    )

    return calls


def test_main_public_signature_is_exact() -> None:
    product = _product()

    assert str(
        inspect.signature(
            product.main
        )
    ) == (
        "(argv: 'Sequence[str] | None' = None) "
        "-> 'int'"
    )


def test_parser_requires_history() -> None:
    product = _product()
    argv = _argv()

    index = argv.index("--history")
    del argv[index:index + 2]

    with pytest.raises(SystemExit):
        product.main(argv)


def test_parser_requires_window_name() -> None:
    product = _product()
    argv = _argv()

    index = argv.index("--window-name")
    del argv[index:index + 2]

    with pytest.raises(SystemExit):
        product.main(argv)


def test_parser_requires_start_round() -> None:
    product = _product()
    argv = _argv()

    index = argv.index("--start-round")
    del argv[index:index + 2]

    with pytest.raises(SystemExit):
        product.main(argv)


def test_parser_requires_end_round() -> None:
    product = _product()
    argv = _argv()

    index = argv.index("--end-round")
    del argv[index:index + 2]

    with pytest.raises(SystemExit):
        product.main(argv)


def test_parser_candidate_is_mode_optional() -> None:
    product = _product()

    parser = product._parser()

    action = next(
        action
        for action in parser._actions
        if "--candidate" in action.option_strings
    )

    assert action.required is False


def test_parser_baseline_is_mode_optional() -> None:
    product = _product()

    parser = product._parser()

    action = next(
        action
        for action in parser._actions
        if "--baseline" in action.option_strings
    )

    assert action.required is False


def test_candidate_source_descriptor_minimal_form() -> None:
    product = _product()

    source = product._parse_source(
        "candidate.json|1230|candidate-A"
    )

    assert source.artifact_path == "candidate.json"
    assert source.round_no == 1230
    assert source.model_name == "candidate-A"
    assert source.regime_id is None
    assert source.strategy_name is None


def test_candidate_source_descriptor_full_form() -> None:
    product = _product()

    source = product._parse_source(
        (
            "candidate.json"
            "|1230"
            "|candidate-A"
            "|regime-A"
            "|strategy-A"
        )
    )

    assert source.artifact_path == "candidate.json"
    assert source.round_no == 1230
    assert source.model_name == "candidate-A"
    assert source.regime_id == "regime-A"
    assert source.strategy_name == "strategy-A"


def test_baseline_source_descriptor_minimal_form() -> None:
    product = _product()

    source = product._parse_source(
        "baseline.json|1229|baseline-A"
    )

    assert source.artifact_path == "baseline.json"
    assert source.round_no == 1229
    assert source.model_name == "baseline-A"
    assert source.regime_id is None
    assert source.strategy_name is None


def test_baseline_source_descriptor_full_form() -> None:
    product = _product()

    source = product._parse_source(
        (
            "baseline.json"
            "|1229"
            "|baseline-A"
            "|regime-B"
            "|strategy-B"
        )
    )

    assert source.artifact_path == "baseline.json"
    assert source.round_no == 1229
    assert source.model_name == "baseline-A"
    assert source.regime_id == "regime-B"
    assert source.strategy_name == "strategy-B"


@pytest.mark.parametrize(
    "value",
    [
        "only-path",
        "path|1230",
    ],
)
def test_source_descriptor_rejects_too_few_fields(
    value: str,
) -> None:
    product = _product()

    with pytest.raises(
        argparse.ArgumentTypeError
    ):
        product._parse_source(value)


@pytest.mark.parametrize(
    "value",
    [
        "path|1230|model|regime|strategy|extra",
        "a|1230|m|r|s|x|y",
    ],
)
def test_source_descriptor_rejects_too_many_fields(
    value: str,
) -> None:
    product = _product()

    with pytest.raises(
        argparse.ArgumentTypeError
    ):
        product._parse_source(value)


@pytest.mark.parametrize(
    "value",
    [
        "path|abc|model",
        "path|1.5|model",
        "path||model",
    ],
)
def test_source_descriptor_rejects_invalid_round(
    value: str,
) -> None:
    product = _product()

    with pytest.raises(
        argparse.ArgumentTypeError
    ):
        product._parse_source(value)


@pytest.mark.parametrize(
    (
        "value",
        "regime_id",
        "strategy_name",
    ),
    [
        (
            "path|1230|model||",
            None,
            None,
        ),
        (
            "path|1230|model||strategy-A",
            None,
            "strategy-A",
        ),
        (
            "path|1230|model|regime-A|",
            "regime-A",
            None,
        ),
    ],
)
def test_source_descriptor_maps_empty_optional_values_to_none(
    value: str,
    regime_id: str | None,
    strategy_name: str | None,
) -> None:
    product = _product()

    source = product._parse_source(value)

    assert source.regime_id == regime_id
    assert source.strategy_name == strategy_name


def test_candidate_order_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()

    calls = _patch_service(
        monkeypatch,
        product,
    )

    assert product.main(_argv()) == 0

    request = calls[0]

    assert tuple(
        source.artifact_path
        for source in request.candidate_sources
    ) == (
        "artifacts/candidate-1.json",
        "artifacts/candidate-2.json",
    )


def test_baseline_order_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()

    calls = _patch_service(
        monkeypatch,
        product,
    )

    argv = _argv() + [
        "--baseline",
        "artifacts/baseline-2.json|1231|baseline-A",
    ]

    assert product.main(argv) == 0

    request = calls[0]

    assert tuple(
        source.artifact_path
        for source in request.baseline_sources
    ) == (
        "artifacts/baseline-1.json",
        "artifacts/baseline-2.json",
    )


def test_main_constructs_exact_execution_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()

    calls = _patch_service(
        monkeypatch,
        product,
    )

    assert product.main(_argv()) == 0
    assert len(calls) == 1

    request = calls[0]

    assert isinstance(
        request,
        product.DurableReplayExecutionRequest,
    )

    assert request.history_path == "data/history.json"
    assert request.window_name == "window-001"
    assert request.start_round == 1230
    assert request.end_round == 1231
    assert len(request.candidate_sources) == 2
    assert len(request.baseline_sources) == 1


def test_main_invokes_execution_service_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()

    calls = _patch_service(
        monkeypatch,
        product,
    )

    result = product.main(_argv())

    assert result == 0
    assert len(calls) == 1


def test_main_returns_zero_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()

    _patch_service(
        monkeypatch,
        product,
    )

    assert product.main(_argv()) == 0


def test_main_renders_deterministic_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    product = _product()

    _patch_service(
        monkeypatch,
        product,
    )

    assert product.main(_argv()) == 0
    first = capsys.readouterr().out

    assert product.main(_argv()) == 0
    second = capsys.readouterr().out

    assert first == second

    payload = json.loads(first)

    assert payload["status"] == "PASS"
    assert payload["candidate_model_name"] == "candidate-A"
    assert payload["baseline_model_name"] == "baseline-A"
    assert payload["round_count"] == 2
    assert "evaluation" in payload


def test_main_preserves_candidate_model_name(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    product = _product()

    _patch_service(
        monkeypatch,
        product,
    )

    product.main(_argv())

    payload = json.loads(
        capsys.readouterr().out
    )

    assert payload["candidate_model_name"] == "candidate-A"


def test_main_preserves_baseline_model_name(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    product = _product()

    _patch_service(
        monkeypatch,
        product,
    )

    product.main(_argv())

    payload = json.loads(
        capsys.readouterr().out
    )

    assert payload["baseline_model_name"] == "baseline-A"


def test_main_preserves_round_count(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    product = _product()

    _patch_service(
        monkeypatch,
        product,
    )

    product.main(_argv())

    payload = json.loads(
        capsys.readouterr().out
    )

    assert payload["round_count"] == 2


def test_main_propagates_execution_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()

    sentinel = RuntimeError(
        "execution sentinel"
    )

    def fail(
        self: Any,
        *,
        request: Any,
    ) -> Any:
        raise sentinel

    monkeypatch.setattr(
        product.DurableReplayExecutionService,
        "execute",
        fail,
    )

    with pytest.raises(RuntimeError) as exc_info:
        product.main(_argv())

    assert exc_info.value is sentinel


def test_product_has_no_subprocess_dependency() -> None:
    product = _product()

    source = Path(
        product.__file__
    ).read_text(
        encoding="utf-8-sig"
    )

    assert "subprocess" not in source


def test_product_has_no_validation_tool_dependency() -> None:
    product = _product()

    source = Path(
        product.__file__
    ).read_text(
        encoding="utf-8-sig"
    )

    assert "tools.validation" not in source


def test_product_has_no_direct_history_dependency() -> None:
    product = _product()

    source = Path(
        product.__file__
    ).read_text(
        encoding="utf-8-sig"
    )

    assert "lrp.io.draws" not in source
    assert "load_history" not in source
    assert "history_until_round" not in source


def test_product_has_no_direct_durable_codec_dependency() -> None:
    product = _product()

    source = Path(
        product.__file__
    ).read_text(
        encoding="utf-8-sig"
    )

    assert "source_from_json" not in source
    assert "source_from_dict" not in source
    assert (
        "durable_prediction_evaluation_source"
        not in source
    )


def test_product_has_no_filesystem_write_dependency() -> None:
    product = _product()

    source = Path(
        product.__file__
    ).read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        "write_text",
        "write_bytes",
        ".mkdir(",
        "open(",
        "write_prediction_artifacts",
    )

    assert all(
        token not in source
        for token in forbidden
    )

    assert source.count(
        "write_operation_artifact"
    ) == 2


def test_product_has_no_artifact_discovery_dependency() -> None:
    product = _product()

    source = Path(
        product.__file__
    ).read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        "evaluation_source.json",
        "prediction-evaluation-sources",
        ".glob(",
        ".rglob(",
        ".iterdir(",
    )

    assert all(
        token not in source
        for token in forbidden
    )


def test_root_dispatch_registers_durable_replay_evaluation() -> None:
    source = _DISPATCHER_PATH.read_text(
        encoding="utf-8-sig"
    )

    assert (
        "durable-replay-evaluation"
        in source
    )

    assert (
        "durable_replay_evaluation"
        in source
    )


def test_root_dispatch_preserves_model_evaluation() -> None:
    source = _DISPATCHER_PATH.read_text(
        encoding="utf-8-sig"
    )

    assert '"model-evaluation"' in source
    assert "model_evaluation_main" in source

    model_source = Path(
        "lrp/cli/model_evaluation.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    assert (
        "tools.validation.run_model_evaluation"
        in model_source
    )
