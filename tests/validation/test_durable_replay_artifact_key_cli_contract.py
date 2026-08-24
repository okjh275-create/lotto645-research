from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path

import pytest

import lrp.cli.durable_replay_evaluation as module
from lrp.operations.durable_replay_artifact_discovery import (
    DurableReplayArtifactSelector,
)


def test_parse_selector_signature_remains_exact() -> None:
    assert str(inspect.signature(module._parse_selector)) == (
        "(value: 'str') -> 'DurableReplayArtifactSelector'"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        ("1|model-a", (1, "model-a", None, None, None)),
        ("1|model-a|regime-a", (1, "model-a", "regime-a", None, None)),
        (
            "1|model-a|regime-a|strategy-a",
            (1, "model-a", "regime-a", "strategy-a", None),
        ),
        (
            "1|model-a|regime-a|strategy-a|candidate-a",
            (1, "model-a", "regime-a", "strategy-a", "candidate-a"),
        ),
        ("1|model-a|||candidate-a", (1, "model-a", None, None, "candidate-a")),
        ("0007|model-b|||baseline_01", (7, "model-b", None, None, "baseline_01")),
    ),
)
def test_parse_selector_valid_matrix(value, expected) -> None:
    selector = module._parse_selector(value)
    assert (
        selector.round_no,
        selector.model_name,
        selector.regime_id,
        selector.strategy_name,
        selector.artifact_key,
    ) == expected


@pytest.mark.parametrize("value", ("", "1", "1|m|r|s|k|extra"))
def test_parse_selector_rejects_invalid_field_count(value: str) -> None:
    with pytest.raises(Exception):
        module._parse_selector(value)


@pytest.mark.parametrize(
    "value",
    ("+1|model", "-1|model", " 1|model", "1 |model", "1.0|model", "abc|model"),
)
def test_parse_selector_round_syntax_is_preserved(value: str) -> None:
    with pytest.raises(Exception):
        module._parse_selector(value)


@pytest.mark.parametrize(
    "artifact_key",
    (
        "", " ", ".", "..", "../x", "..\\x", "a/b", "a\\b",
        "/absolute", "\\absolute", "C:\\absolute", "a:b", "a b",
        "한글", "a@b", "a#b", "a" * 129,
    ),
)
def test_parse_selector_rejects_invalid_fifth_field_artifact_key(artifact_key: str) -> None:
    value = "1231|model-a|regime-a|strategy-a|" + artifact_key
    with pytest.raises(Exception):
        module._parse_selector(value)


def test_parse_selector_accepts_128_char_artifact_key() -> None:
    key = "a" * 128
    selector = module._parse_selector("1231|model-a|regime-a|strategy-a|" + key)
    assert selector.artifact_key == key


def test_existing_four_field_semantics_are_unchanged() -> None:
    selector = module._parse_selector("1231|model-a|regime-a|strategy-a")
    assert selector == DurableReplayArtifactSelector(
        round_no=1231,
        model_name="model-a",
        regime_id="regime-a",
        strategy_name="strategy-a",
        artifact_key=None,
    )


def test_fifth_field_maps_only_to_artifact_key() -> None:
    selector = module._parse_selector(
        "1231|model-a|regime-a|strategy-a|candidate-a"
    )
    assert selector == DurableReplayArtifactSelector(
        round_no=1231,
        model_name="model-a",
        regime_id="regime-a",
        strategy_name="strategy-a",
        artifact_key="candidate-a",
    )


def test_empty_optional_provenance_fields_are_preserved_as_none() -> None:
    selector = module._parse_selector("1231|model-a|||candidate-a")
    assert selector.regime_id is None
    assert selector.strategy_name is None
    assert selector.artifact_key == "candidate-a"


def test_selector_dataclass_field_order_is_exact() -> None:
    assert tuple(
        field.name for field in dataclasses.fields(DurableReplayArtifactSelector)
    ) == (
        "round_no",
        "model_name",
        "regime_id",
        "strategy_name",
        "artifact_key",
    )


def test_cli_product_contains_artifact_key_projection() -> None:
    source = Path("lrp/cli/durable_replay_evaluation.py").read_text(
        encoding="utf-8-sig"
    )
    assert "artifact_key" in source


def test_root_command_token_is_unchanged() -> None:
    source = Path("lrp/cli/__init__.py").read_text(encoding="utf-8-sig")
    assert "durable-replay-evaluation" in source