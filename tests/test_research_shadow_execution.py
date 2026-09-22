from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

import lrp.research_shadow_execution as rse


CHALLENGERS = [
    "CG00_A",
    "CG01_A",
    "CG02_A",
    "CG03_A",
    "CG04_A",
    "CG05_A",
    "HF00_A",
    "HF01_A",
    "HF02_A",
    "HF03_A",
    "HF04_A",
    "HF05_A",
    "LG00_A",
    "LG01_A",
    "LG02_A",
    "PS00_A",
    "PS01_A",
    "PS02_A",
    "PS03_A",
    "PS04_A",
    "SC00_A",
    "SC01_A",
    "SC02_A",
    "SC03_A",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest().upper()


def _write_registry(
    path: Path,
    ids=CHALLENGERS,
) -> str:
    payload = {
        "challengers": [
            {
                "id":
                    challenger_id,
            }
            for challenger_id
            in ids
        ]
    }

    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return _sha(path)


def _write_db(
    path: Path,
    rounds: list[int],
) -> None:
    connection = sqlite3.connect(
        path
    )

    try:
        connection.execute(
            "CREATE TABLE draw_history ("
            "round INTEGER PRIMARY KEY,"
            "n1 INTEGER,"
            "n2 INTEGER,"
            "n3 INTEGER,"
            "n4 INTEGER,"
            "n5 INTEGER,"
            "n6 INTEGER,"
            "bonus INTEGER"
            ")"
        )

        for round_no in rounds:
            connection.execute(
                "INSERT INTO draw_history "
                "(round,n1,n2,n3,n4,n5,n6,bonus) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    round_no,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                ),
            )

        connection.commit()

    finally:
        connection.close()


def _write_prediction(
    path: Path,
    round_no: int,
    *,
    set_count: int = 10,
) -> str:
    payload = {
        "round":
            round_no,

        "sets": [
            {
                "id":
                    f"S{index + 1}",
                "numbers":
                    [1, 2, 3, 4, 5, 6],
            }
            for index
            in range(set_count)
        ],

        "top5_practical": [
            "S1",
            "S2",
            "S3",
            "S4",
            "S5",
        ],
    }

    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return _sha(path)


def _control_inputs(
    tmp_path: Path,
):
    registry = (
        tmp_path
        / "registry.json"
    )

    database = (
        tmp_path
        / "history.db"
    )

    prediction = (
        tmp_path
        / "prediction.json"
    )

    registry_sha = (
        _write_registry(
            registry
        )
    )

    _write_db(
        database,
        [1242],
    )

    prediction_sha = (
        _write_prediction(
            prediction,
            1243,
        )
    )

    return (
        registry,
        database,
        prediction,
        registry_sha,
        prediction_sha,
    )


def test_validate_shadow_round_accepts_lower_bound():
    assert (
        rse.validate_shadow_round(
            1243
        )
        == 1243
    )


def test_validate_shadow_round_accepts_upper_bound():
    assert (
        rse.validate_shadow_round(
            1252
        )
        == 1252
    )


def test_validate_shadow_round_rejects_lower_outside():
    with pytest.raises(
        rse.ShadowExecutionContractError
    ):
        rse.validate_shadow_round(
            1242
        )


def test_validate_shadow_round_rejects_upper_outside():
    with pytest.raises(
        rse.ShadowExecutionContractError
    ):
        rse.validate_shadow_round(
            1253
        )


def test_challenger_id_validation():
    assert (
        rse.validate_challenger_id(
            "CG00_RANDOM_FILTERED"
        )
        == "CG00_RANDOM_FILTERED"
    )

    with pytest.raises(
        rse.RegistryContractError
    ):
        rse.validate_challenger_id(
            "BAD00"
        )


def test_seed_known_value():
    assert (
        rse.derive_shadow_seed(
            1243,
            "CG00_RANDOM_FILTERED",
        )
        == 1081018488
    )


def test_seed_is_deterministic():
    first = rse.derive_shadow_seed(
        1243,
        "CG00_RANDOM_FILTERED",
    )

    second = rse.derive_shadow_seed(
        1243,
        "CG00_RANDOM_FILTERED",
    )

    assert first == second


def test_registry_exact_contract(tmp_path):
    path = (
        tmp_path
        / "registry.json"
    )

    digest = _write_registry(
        path
    )

    loaded = rse.load_registry(
        path,
        expected_sha256=digest,
    )

    assert (
        len(
            loaded[
                "challenger_ids"
            ]
        )
        == 24
    )

    assert loaded[
        "prefix_counts"
    ] == {
        "CG": 6,
        "HF": 6,
        "LG": 3,
        "PS": 5,
        "SC": 4,
    }


def test_registry_sha_mismatch_rejected(tmp_path):
    path = (
        tmp_path
        / "registry.json"
    )

    _write_registry(path)

    with pytest.raises(
        rse.RegistryContractError
    ):
        rse.load_registry(
            path,
            expected_sha256=(
                "0" * 64
            ),
        )


def test_registry_count_mismatch_rejected(tmp_path):
    path = (
        tmp_path
        / "registry.json"
    )

    _write_registry(
        path,
        CHALLENGERS[:-1],
    )

    with pytest.raises(
        rse.RegistryContractError
    ):
        rse.load_registry(path)


def test_read_only_connection_sets_query_only(tmp_path):
    database = (
        tmp_path
        / "history.db"
    )

    _write_db(
        database,
        [1242],
    )

    connection = (
        rse.open_history_read_only(
            database
        )
    )

    try:
        value = connection.execute(
            "PRAGMA query_only"
        ).fetchone()[0]

        assert int(value) == 1

    finally:
        connection.close()


def test_read_only_connection_rejects_write(tmp_path):
    database = (
        tmp_path
        / "history.db"
    )

    _write_db(
        database,
        [1242],
    )

    connection = (
        rse.open_history_read_only(
            database
        )
    )

    try:
        with pytest.raises(
            sqlite3.OperationalError
        ):
            connection.execute(
                "INSERT INTO draw_history "
                "(round,n1,n2,n3,n4,n5,n6,bonus) "
                "VALUES (1243,1,2,3,4,5,6,7)"
            )

    finally:
        connection.close()


def test_history_cutoff_exact(tmp_path):
    database = (
        tmp_path
        / "history.db"
    )

    _write_db(
        database,
        [1242],
    )

    result = (
        rse.inspect_history_cutoff(
            database,
            1243,
        )
    )

    assert (
        result[
            "max_round"
        ]
        == 1242
    )

    assert (
        result[
            "target_count"
        ]
        == 0
    )


def test_history_target_result_rejected(tmp_path):
    database = (
        tmp_path
        / "history.db"
    )

    _write_db(
        database,
        [1242, 1243],
    )

    with pytest.raises(
        rse.HistoryContractError
    ):
        rse.inspect_history_cutoff(
            database,
            1243,
        )


def test_history_behind_cutoff_rejected(tmp_path):
    database = (
        tmp_path
        / "history.db"
    )

    _write_db(
        database,
        [1241],
    )

    with pytest.raises(
        rse.HistoryContractError
    ):
        rse.inspect_history_cutoff(
            database,
            1243,
        )


def test_frozen_prediction_valid(tmp_path):
    prediction = (
        tmp_path
        / "prediction.json"
    )

    digest = (
        _write_prediction(
            prediction,
            1243,
        )
    )

    result = (
        rse.load_frozen_prediction(
            prediction,
            1243,
            expected_sha256=digest,
        )
    )

    assert (
        result[
            "set_count"
        ]
        == 10
    )

    assert (
        result[
            "top5_count"
        ]
        == 5
    )


def test_frozen_prediction_wrong_round_rejected(tmp_path):
    prediction = (
        tmp_path
        / "prediction.json"
    )

    _write_prediction(
        prediction,
        1244,
    )

    with pytest.raises(
        rse.PredictionContractError
    ):
        rse.load_frozen_prediction(
            prediction,
            1243,
        )


def test_frozen_prediction_wrong_set_count_rejected(tmp_path):
    prediction = (
        tmp_path
        / "prediction.json"
    )

    _write_prediction(
        prediction,
        1243,
        set_count=9,
    )

    with pytest.raises(
        rse.PredictionContractError
    ):
        rse.load_frozen_prediction(
            prediction,
            1243,
        )


def test_artifact_plan_has_expected_paths_without_creation(tmp_path):
    plan = (
        rse.plan_artifact_transaction(
            tmp_path,
            1243,
        )
    )

    assert (
        plan.final_dir
        == tmp_path.resolve()
        / "artifacts"
        / "research"
        / "shadow"
        / "round_1243"
    )

    assert (
        plan.temp_dir
        == tmp_path.resolve()
        / "artifacts"
        / "research"
        / "shadow"
        / ".round_1243.tmp"
    )

    assert not plan.final_dir.exists()
    assert not plan.temp_dir.exists()


def test_artifact_final_collision_rejected(tmp_path):
    final_dir = (
        tmp_path
        / "artifacts"
        / "research"
        / "shadow"
        / "round_1243"
    )

    final_dir.mkdir(
        parents=True
    )

    with pytest.raises(
        rse.ArtifactContractError
    ):
        rse.plan_artifact_transaction(
            tmp_path,
            1243,
        )


def test_artifact_temp_collision_rejected(tmp_path):
    temp_dir = (
        tmp_path
        / "artifacts"
        / "research"
        / "shadow"
        / ".round_1243.tmp"
    )

    temp_dir.mkdir(
        parents=True
    )

    with pytest.raises(
        rse.ArtifactContractError
    ):
        rse.plan_artifact_transaction(
            tmp_path,
            1243,
        )


def test_build_control_plan_has_24_seeds(tmp_path):
    (
        registry,
        database,
        prediction,
        registry_sha,
        prediction_sha,
    ) = _control_inputs(
        tmp_path
    )

    plan = rse.build_control_plan(
        repo_root=tmp_path,
        target_round=1243,
        registry_path=registry,
        history_database=database,
        production_prediction_path=prediction,
        expected_registry_sha256=registry_sha,
        expected_prediction_sha256=prediction_sha,
    )

    assert (
        len(
            plan[
                "seeds"
            ]
        )
        == 24
    )

    assert (
        plan[
            "authorization"
        ][
            "challenger_generation"
        ]
        is False
    )

    assert (
        plan[
            "authorization"
        ][
            "challenger_execution"
        ]
        is False
    )


def test_build_control_plan_is_deterministic(tmp_path):
    (
        registry,
        database,
        prediction,
        registry_sha,
        prediction_sha,
    ) = _control_inputs(
        tmp_path
    )

    first = rse.build_control_plan(
        repo_root=tmp_path,
        target_round=1243,
        registry_path=registry,
        history_database=database,
        production_prediction_path=prediction,
        expected_registry_sha256=registry_sha,
        expected_prediction_sha256=prediction_sha,
    )

    second = rse.build_control_plan(
        repo_root=tmp_path,
        target_round=1243,
        registry_path=registry,
        history_database=database,
        production_prediction_path=prediction,
        expected_registry_sha256=registry_sha,
        expected_prediction_sha256=prediction_sha,
    )

    assert first == second


def test_module_has_no_challenger_execution_api():
    assert not hasattr(
        rse,
        "execute_challengers",
    )

    assert not hasattr(
        rse,
        "run_challengers",
    )

# === PHASE2_PARITY_HARNESS_TESTS_V1 BEGIN ===

import pytest as _phase2_pytest

from lrp import research_shadow_execution as _phase2_product


def _phase2_probe_snapshot(
    surface_id,
    *,
    output=None,
    output_schema=None,
    semantic_parity=True,
):
    contract = (
        _phase2_product.phase2_parity_contract(
            surface_id
        )
    )

    identity = {
        key: contract[key]
        for key in (
            "module",
            "path",
            "callable",
            "file_sha256",
            "git_blob",
        )
    }

    if output is None:
        output = {
            "value": surface_id,
        }

    if output_schema is None:
        output_schema = {
            "type": "mapping",
            "keys": ["value"],
        }

    return {
        "identity": identity,
        "output_schema": output_schema,
        "output": output,
        "semantic_parity": semantic_parity,
    }


def test_phase2_surface_ids_are_exact():
    assert (
        _phase2_product.phase2_parity_surface_ids()
        == (
            "CANDIDATE",
            "SCORING",
            "FILTER_FEATURE",
            "PRACTICAL_SELECTOR",
            "PAIR_FEATURE",
        )
    )


def test_phase2_authorization_is_fail_closed():
    authorization = (
        _phase2_product.phase2_parity_authorization()
    )

    assert authorization
    assert all(
        value is False
        for value in authorization.values()
    )


def test_phase2_contract_returns_defensive_copy():
    first = (
        _phase2_product.phase2_parity_contract(
            "CANDIDATE"
        )
    )

    first["classification"] = "MUTATED"

    second = (
        _phase2_product.phase2_parity_contract(
            "CANDIDATE"
        )
    )

    assert second["classification"] != "MUTATED"


def test_phase2_unknown_surface_fails_closed():
    with _phase2_pytest.raises(
        _phase2_product.ParityValidationError
    ):
        _phase2_product.phase2_parity_contract(
            "UNKNOWN"
        )


def test_phase2_candidate_can_be_parity_eligible_but_not_authorized():
    first = _phase2_probe_snapshot(
        "CANDIDATE"
    )

    second = _phase2_probe_snapshot(
        "CANDIDATE"
    )

    result = (
        _phase2_product.phase2_evaluate_probe_pair(
            "CANDIDATE",
            first,
            second,
        )
    )

    assert (
        result["decision"]
        == _phase2_product.PARITY_CONFIRMED_BINDING_ELIGIBLE
    )

    assert result["binding_eligible"] is True
    assert result["binding_authorized"] is False


def test_phase2_candidate_output_drift_fails_closed():
    first = _phase2_probe_snapshot(
        "CANDIDATE",
        output={"value": 1},
    )

    second = _phase2_probe_snapshot(
        "CANDIDATE",
        output={"value": 2},
    )

    result = (
        _phase2_product.phase2_evaluate_probe_pair(
            "CANDIDATE",
            first,
            second,
        )
    )

    assert (
        result["decision"]
        == _phase2_product.PARITY_UNRESOLVED_FAIL_CLOSED
    )

    assert result["deterministic"] is False
    assert result["binding_eligible"] is False


def test_phase2_candidate_schema_drift_fails_closed():
    first = _phase2_probe_snapshot(
        "CANDIDATE",
        output_schema={
            "type": "mapping",
            "keys": ["a"],
        },
    )

    second = _phase2_probe_snapshot(
        "CANDIDATE",
        output_schema={
            "type": "mapping",
            "keys": ["b"],
        },
    )

    result = (
        _phase2_product.phase2_evaluate_probe_pair(
            "CANDIDATE",
            first,
            second,
        )
    )

    assert (
        result["decision"]
        == _phase2_product.PARITY_UNRESOLVED_FAIL_CLOSED
    )

    assert result["output_schema_match"] is False


@_phase2_pytest.mark.parametrize(
    "surface_id",
    (
        "SCORING",
        "FILTER_FEATURE",
        "PRACTICAL_SELECTOR",
        "PAIR_FEATURE",
    ),
)
def test_phase2_advisory_helpers_cannot_become_direct_pipeline_bindings(
    surface_id,
):
    first = _phase2_probe_snapshot(
        surface_id
    )

    second = _phase2_probe_snapshot(
        surface_id
    )

    result = (
        _phase2_product.phase2_evaluate_probe_pair(
            surface_id,
            first,
            second,
        )
    )

    assert (
        result["decision"]
        == _phase2_product.PARITY_REJECTED_ADVISORY_ONLY
    )

    assert result["binding_eligible"] is False
    assert result["binding_authorized"] is False


def test_phase2_probe_matrix_requires_all_five_surfaces():
    probes = {}

    for surface_id in (
        _phase2_product.phase2_parity_surface_ids()
    ):
        snapshot = _phase2_probe_snapshot(
            surface_id
        )

        probes[surface_id] = {
            "first": snapshot,
            "second": snapshot,
        }

    results = (
        _phase2_product.phase2_validate_probe_matrix(
            probes
        )
    )

    assert len(results) == 5

    assert (
        _phase2_product.phase2_binding_decisions_fail_closed(
            results
        )
        is True
    )


def test_phase2_probe_matrix_rejects_missing_surface():
    probes = {}

    for surface_id in (
        _phase2_product.phase2_parity_surface_ids()[:-1]
    ):
        snapshot = _phase2_probe_snapshot(
            surface_id
        )

        probes[surface_id] = {
            "first": snapshot,
            "second": snapshot,
        }

    with _phase2_pytest.raises(
        _phase2_product.ParityValidationError
    ):
        _phase2_product.phase2_validate_probe_matrix(
            probes
        )


# === PHASE2_PARITY_HARNESS_TESTS_V1 END ===
