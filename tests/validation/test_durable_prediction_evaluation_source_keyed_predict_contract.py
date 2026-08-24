from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from lrp.cli import predict as module


def test_parser_has_artifact_key_option() -> None:
    parser = module._parser()
    actions = {
        option: action
        for action in parser._actions
        for option in action.option_strings
    }

    assert "--artifact-key" in actions


def test_artifact_key_option_is_optional() -> None:
    parser = module._parser()
    action = next(
        action
        for action in parser._actions
        if "--artifact-key" in action.option_strings
    )

    assert action.required is False
    assert action.default is None


def test_main_public_signature_is_unchanged() -> None:
    assert str(inspect.signature(module.main)) == (
        "(argv: 'Sequence[str] | None' = None) -> 'int'"
    )


def test_legacy_durable_source_path_contract_remains_present() -> None:
    source = Path(
        "lrp/cli/predict.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    assert "prediction-evaluation-sources" in source
    assert "evaluation_source.json" in source


def test_predict_product_mentions_artifact_key() -> None:
    source = Path(
        "lrp/cli/predict.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    assert "artifact_key" in source


def test_predict_product_mentions_artifact_key_cli_option() -> None:
    source = Path(
        "lrp/cli/predict.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    assert "--artifact-key" in source


@pytest.mark.parametrize(
    "artifact_key",
    (
        "candidate-a",
        "baseline_01",
        "seed.20260823",
        "A1",
        "a-b_c.d",
    ),
)
def test_parser_accepts_valid_artifact_key(
    artifact_key: str,
) -> None:
    parser = module._parser()

    args = parser.parse_args(
        [
            "--history",
            "history.json",
            "--round",
            "1231",
            "--seed",
            "20260823",
            "--output",
            "out",
            "--artifact-key",
            artifact_key,
        ]
    )

    assert args.artifact_key == artifact_key


@pytest.mark.parametrize(
    "artifact_key",
    (
        "",
        " ",
        ".",
        "..",
        "../x",
        "..\\x",
        "a/b",
        "a\\b",
        "/absolute",
        "\\absolute",
        "C:\\absolute",
        "a:b",
        "a b",
        "한글",
        "a@b",
        "a#b",
        "a" * 129,
    ),
)
def test_parser_rejects_invalid_artifact_key(
    artifact_key: str,
) -> None:
    parser = module._parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--history",
                "history.json",
                "--round",
                "1231",
                "--seed",
                "20260823",
                "--output",
                "out",
                "--artifact-key",
                artifact_key,
            ]
        )


def test_artifact_key_at_128_chars_is_accepted() -> None:
    parser = module._parser()

    key = "a" * 128

    args = parser.parse_args(
        [
            "--history",
            "history.json",
            "--round",
            "1231",
            "--seed",
            "20260823",
            "--output",
            "out",
            "--artifact-key",
            key,
        ]
    )

    assert args.artifact_key == key


def test_legacy_mode_does_not_require_artifact_key() -> None:
    parser = module._parser()

    args = parser.parse_args(
        [
            "--history",
            "history.json",
            "--round",
            "1231",
            "--seed",
            "20260823",
            "--output",
            "out",
        ]
    )

    assert args.artifact_key is None


def test_keyed_path_shape_is_encoded_in_product() -> None:
    source = Path(
        "lrp/cli/predict.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    assert "artifact_key" in source
    assert "prediction-evaluation-sources" in source
    assert "evaluation_source.json" in source