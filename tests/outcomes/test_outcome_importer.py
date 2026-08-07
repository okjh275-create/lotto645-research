from __future__ import annotations

import json
from pathlib import Path

import pytest

from lrp.outcomes import (
    OutcomeImporter,
    OutcomeImportError,
)


def prediction_payload() -> dict[str, object]:
    return {
        "round": 1232,
        "generated_at_kst": (
            "2026-08-08 20:45"
        ),
        "seed": 20260808,
        "params": {
            "temperature": 0.85,
            "weights": {
                "recency": 0.35,
                "frequency": 0.20,
            },
            "windows": {
                "short": 10,
                "mid": 20,
                "long": 50,
            },
            "K": 10,
        },
        "sets": [
            {
                "id": "S1",
                "numbers": [
                    4,
                    11,
                    19,
                    27,
                    34,
                    42,
                ],
                "score": 0.91,
                "risk_flags": [],
                "features": {
                    "sum": 137,
                    "odd_even": "3:3",
                },
            },
            {
                "id": "S2",
                "numbers": [
                    2,
                    13,
                    20,
                    28,
                    35,
                    41,
                ],
                "score": 0.84,
                "risk_flags": [
                    "example",
                ],
                "features": {
                    "sum": 139,
                    "odd_even": "3:3",
                },
            },
        ],
        "top5_practical": [
            "S1",
        ],
        "metadata": {
            "generated_candidates": 10000,
            "statistics_version": "1.0.0",
            "candidate_version": "0.8.0",
            "practical_complete": True,
        },
    }


def test_imports_prediction_records() -> None:
    importer = OutcomeImporter(
        model_name="lrp-v4.0.0",
    )

    records = importer.import_predictions(
        prediction_payload()
    )

    assert len(records) == 2

    first = records[0]

    assert first.round_no == 1232
    assert first.set_id == "S1"
    assert first.numbers == (
        4,
        11,
        19,
        27,
        34,
        42,
    )
    assert first.score == pytest.approx(
        0.91
    )
    assert first.model_name == (
        "lrp-v4.0.0"
    )
    assert first.seed == 20260808
    assert first.generated_at_kst == (
        "2026-08-08 20:45"
    )


def test_preserves_params_and_metadata() -> None:
    record = OutcomeImporter(
        model_name="lrp-v4.0.0",
    ).import_predictions(
        prediction_payload()
    )[0]

    assert record.parameters[
        "temperature"
    ] == pytest.approx(0.85)

    assert record.parameters[
        "windows"
    ] == {
        "short": 10,
        "mid": 20,
        "long": 50,
    }

    metadata = record.parameters[
        "prediction_metadata"
    ]

    assert metadata[
        "statistics_version"
    ] == "1.0.0"
    assert metadata[
        "candidate_version"
    ] == "0.8.0"


def test_enriches_practical_and_risk_features() -> None:
    first, second = OutcomeImporter(
        model_name="lrp-v4.0.0",
    ).import_predictions(
        prediction_payload()
    )

    assert first.features[
        "is_practical"
    ] is True
    assert second.features[
        "is_practical"
    ] is False

    assert first.features[
        "risk_flags"
    ] == ()
    assert second.features[
        "risk_flags"
    ] == ("example",)


def test_prediction_ids_are_deterministic() -> None:
    importer = OutcomeImporter(
        model_name="lrp-v4.0.0",
    )

    first = importer.import_predictions(
        prediction_payload()
    )
    second = importer.import_predictions(
        prediction_payload()
    )

    assert tuple(
        item.prediction_id
        for item in first
    ) == tuple(
        item.prediction_id
        for item in second
    )


def test_prediction_id_changes_with_seed() -> None:
    payload = prediction_payload()

    first = OutcomeImporter(
        model_name="lrp-v4.0.0",
    ).import_predictions(payload)[0]

    payload["seed"] = 999

    second = OutcomeImporter(
        model_name="lrp-v4.0.0",
    ).import_predictions(payload)[0]

    assert (
        first.prediction_id
        != second.prediction_id
    )


def test_imports_from_json_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "prediction.json"

    path.write_text(
        json.dumps(
            prediction_payload(),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    records = OutcomeImporter(
        model_name="lrp-v4.0.0",
    ).import_predictions(path)

    assert len(records) == 2
    assert records[0].set_id == "S1"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("round", 0),
        ("round", True),
        ("seed", True),
        ("generated_at_kst", ""),
    ],
)
def test_rejects_invalid_top_level_fields(
    field: str,
    value: object,
) -> None:
    payload = prediction_payload()
    payload[field] = value

    with pytest.raises(
        OutcomeImportError,
    ):
        OutcomeImporter(
            model_name="lrp-v4.0.0",
        ).import_predictions(payload)


def test_rejects_duplicate_set_ids() -> None:
    payload = prediction_payload()

    sets = payload["sets"]
    assert isinstance(sets, list)

    second = sets[1]
    assert isinstance(second, dict)

    second["id"] = "S1"

    with pytest.raises(
        OutcomeImportError,
        match="duplicate set id",
    ):
        OutcomeImporter(
            model_name="lrp-v4.0.0",
        ).import_predictions(payload)


def test_rejects_invalid_lotto_numbers() -> None:
    payload = prediction_payload()

    sets = payload["sets"]
    assert isinstance(sets, list)

    first = sets[0]
    assert isinstance(first, dict)

    first["numbers"] = [
        1,
        1,
        2,
        3,
        4,
        5,
    ]

    with pytest.raises(
        OutcomeImportError,
        match="invalid prediction set",
    ):
        OutcomeImporter(
            model_name="lrp-v4.0.0",
        ).import_predictions(payload)


def test_rejects_invalid_score() -> None:
    payload = prediction_payload()

    sets = payload["sets"]
    assert isinstance(sets, list)

    first = sets[0]
    assert isinstance(first, dict)

    first["score"] = 1.1

    with pytest.raises(
        OutcomeImportError,
        match="invalid prediction set",
    ):
        OutcomeImporter(
            model_name="lrp-v4.0.0",
        ).import_predictions(payload)


def test_requires_nonempty_model_name() -> None:
    with pytest.raises(
        OutcomeImportError,
        match="model_name",
    ):
        OutcomeImporter(
            model_name=" ",
        )


def test_public_exports() -> None:
    import lrp.outcomes as outcomes

    assert "OutcomeImporter" in outcomes.__all__
    assert "OutcomeImportError" in outcomes.__all__
