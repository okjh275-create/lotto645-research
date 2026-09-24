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

# === PHASE3_CHALLENGER_ENGINE_TESTS_V1 BEGIN ===


def test_phase3_exact_24_challenger_ids():
    ids = _phase2_product.phase3_challenger_ids()

    assert len(ids) == 24
    assert len(set(ids)) == 24

    assert ids[0] == "CG00_RANDOM_FILTERED"
    assert ids[-1] == "LG02_NO_GAP_HARD_RULE"


def test_phase3_category_counts_are_frozen():
    assert (
        _phase2_product.phase3_category_counts()
        == {
            "CG": 6,
            "PS": 5,
            "SC": 4,
            "HF": 6,
            "LG": 3,
        }
    )


def test_phase3_registry_validation_is_green():
    assert (
        _phase2_product.phase3_validate_registry()
        is True
    )


def test_phase3_unknown_challenger_fails_closed():
    with _phase2_pytest.raises(
        _phase2_product.ChallengerPlanError
    ):
        _phase2_product.phase3_challenger_definition(
            "UNKNOWN"
        )


def test_phase3_round_below_window_is_rejected():
    with _phase2_pytest.raises(
        _phase2_product.ChallengerPlanError
    ):
        _phase2_product.phase3_build_all_plans(
            1242
        )


def test_phase3_round_above_window_is_rejected():
    with _phase2_pytest.raises(
        _phase2_product.ChallengerPlanError
    ):
        _phase2_product.phase3_build_all_plans(
            1253
        )


def test_phase3_seed_matches_frozen_known_value():
    assert (
        _phase2_product.phase3_seed(
            1243,
            "CG00_RANDOM_FILTERED",
        )
        == 1081018488
    )


def test_phase3_seed_is_repeatable():
    first = _phase2_product.phase3_seed(
        1243,
        "PS01_RANDOM5",
    )

    second = _phase2_product.phase3_seed(
        1243,
        "PS01_RANDOM5",
    )

    assert first == second
    assert first > 0


def test_phase3_different_challengers_have_distinct_round1243_seeds():
    seeds = {
        _phase2_product.phase3_seed(
            1243,
            challenger_id,
        )
        for challenger_id
        in _phase2_product.phase3_challenger_ids()
    }

    assert len(seeds) == 24


def test_phase3_build_all_plans_is_exact_and_ordered():
    plans = (
        _phase2_product.phase3_build_all_plans(
            1243
        )
    )

    assert len(plans) == 24

    assert tuple(
        plan["challenger_id"]
        for plan in plans
    ) == _phase2_product.phase3_challenger_ids()

    assert all(
        plan["round"] == 1243
        for plan in plans
    )


def test_phase3_cg_temperature_contracts():
    low = (
        _phase2_product.phase3_challenger_definition(
            "CG01_TEMP_060"
        )
    )

    high = (
        _phase2_product.phase3_challenger_definition(
            "CG02_TEMP_110"
        )
    )

    assert low["parameters"]["temperature"] == 0.60
    assert high["parameters"]["temperature"] == 1.10


def test_phase3_cg_weight_ablation_contracts():
    equal = (
        _phase2_product.phase3_challenger_definition(
            "CG03_EQUAL_WEIGHTS"
        )
    )

    weights = equal["parameters"]["weights"]

    assert len(weights) == 7
    assert len(set(weights.values())) == 1
    assert abs(sum(weights.values()) - 1.0) < 1e-12

    no_recency = (
        _phase2_product.phase3_challenger_definition(
            "CG04_NO_RECENCY"
        )
    )

    no_pair = (
        _phase2_product.phase3_challenger_definition(
            "CG05_NO_PAIR_GRAPH"
        )
    )

    assert (
        tuple(
            no_recency["parameters"][
                "disabled_score_components"
            ]
        )
        == ("recency",)
    )

    assert (
        tuple(
            no_pair["parameters"][
                "disabled_score_components"
            ]
        )
        == ("pair_graph",)
    )


def test_phase3_practical_selector_contracts():
    expected = {
        "PS00_CURRENT_MMR": "current_mmr",
        "PS01_RANDOM5": "deterministic_random5",
        "PS02_SCORE_TOP5": "score_top5",
        "PS03_MAX_UNIQUE5": "max_unique5",
        "PS04_DIVERSITY5": "diversity5",
    }

    for challenger_id, selector in expected.items():
        definition = (
            _phase2_product.phase3_challenger_definition(
                challenger_id
            )
        )

        assert (
            definition["parameters"]["selector"]
            == selector
        )

        assert (
            definition["parameters"]["practical_k"]
            == 5
        )


def test_phase3_scoring_rank_contracts():
    assert (
        _phase2_product.phase3_challenger_definition(
            "SC00_CURRENT"
        )["parameters"]["ranking_mode"]
        == "current"
    )

    assert (
        _phase2_product.phase3_challenger_definition(
            "SC01_RANDOM_RANK"
        )["parameters"]["ranking_mode"]
        == "deterministic_random"
    )

    assert (
        _phase2_product.phase3_challenger_definition(
            "SC02_QUANTILE_DIAGNOSTIC"
        )["parameters"]["ranking_mode"]
        == "quantile_diagnostic"
    )

    prior = (
        _phase2_product.phase3_challenger_definition(
            "SC03_PREQUENTIAL_PRIOR_SHADOW"
        )
    )

    assert (
        prior["parameters"]["ranking_mode"]
        == "prequential_prior_shadow"
    )

    assert (
        prior["parameters"]["minimum_prior_sample"]
        == 3
    )


def test_phase3_hf_always_hard_filters_are_preserved():
    always_hard = {
        "sum",
        "low_high",
        "consecutive",
        "long_gap",
    }

    for challenger_id in (
        "HF00_CURRENT",
        "HF01_SOFT_ODD_EVEN",
        "HF02_SOFT_TERMINAL",
        "HF03_SOFT_PREVIOUS_OVERLAP",
        "HF04_SOFT_SAME_DECADE",
        "HF05_SOFT_ALL4",
    ):
        definition = (
            _phase2_product.phase3_challenger_definition(
                challenger_id
            )
        )

        assert always_hard.issubset(
            set(
                definition["parameters"][
                    "hard_filters"
                ]
            )
        )


def test_phase3_hf_soft_penalty_and_hf05_limit_are_exact():
    for challenger_id in (
        "HF01_SOFT_ODD_EVEN",
        "HF02_SOFT_TERMINAL",
        "HF03_SOFT_PREVIOUS_OVERLAP",
        "HF04_SOFT_SAME_DECADE",
        "HF05_SOFT_ALL4",
    ):
        definition = (
            _phase2_product.phase3_challenger_definition(
                challenger_id
            )
        )

        assert (
            definition["parameters"]["soft_penalty"]
            == 0.03
        )

    all4 = (
        _phase2_product.phase3_challenger_definition(
            "HF05_SOFT_ALL4"
        )
    )

    assert set(
        all4["parameters"]["soft_filters"]
    ) == {
        "odd_even",
        "terminal",
        "previous_overlap",
        "same_decade",
    }

    assert (
        all4["parameters"]["max_soft_violations"]
        == 2
    )


def test_phase3_long_gap_ablation_contracts():
    current = (
        _phase2_product.phase3_challenger_definition(
            "LG00_CURRENT"
        )["parameters"]
    )

    no_score = (
        _phase2_product.phase3_challenger_definition(
            "LG01_NO_GAP_SCORE"
        )["parameters"]
    )

    no_hard = (
        _phase2_product.phase3_challenger_definition(
            "LG02_NO_GAP_HARD_RULE"
        )["parameters"]
    )

    assert current == {
        "gap_score_enabled": True,
        "long_gap_hard_rule_enabled": True,
    }

    assert no_score == {
        "gap_score_enabled": False,
        "long_gap_hard_rule_enabled": True,
    }

    assert no_hard == {
        "gap_score_enabled": True,
        "long_gap_hard_rule_enabled": False,
    }


def test_phase3_plans_remain_execution_closed_and_phase2_remains_fail_closed():
    plans = (
        _phase2_product.phase3_build_all_plans(
            1243
        )
    )

    for plan in plans:
        assert plan["implementation_only"] is True
        assert all(
            value is False
            for value
            in plan["authorization"].values()
        )

    assert all(
        value is False
        for value in (
            _phase2_product
            .phase2_parity_authorization()
            .values()
        )
    )


# === PHASE3_CHALLENGER_ENGINE_TESTS_V1 END ===

# === PHASE4_RESEARCH_EXECUTOR_TESTS_V1 BEGIN ===


def test_phase4_executor_authorization_is_synthetic_only():
    auth = (
        _phase2_product.phase4_executor_authorization()
    )

    assert auth["research_executor"] is True
    assert auth["synthetic_fixture_execution"] is True

    assert auth["real_round_execution"] is False
    assert auth["real_database_shadow_execution"] is False
    assert auth["real_shadow_publish"] is False
    assert auth["production_helper_binding"] is False
    assert auth["production_prediction_regeneration"] is False
    assert auth["database_write"] is False
    assert auth["production_cli_registration"] is False


def test_phase4_request_is_deterministic_and_preserves_identity():
    first = _phase2_product.phase4_build_request(
        1243,
        "CG00_RANDOM_FILTERED",
    )

    second = _phase2_product.phase4_build_request(
        1243,
        "CG00_RANDOM_FILTERED",
    )

    assert first == second
    assert first["round"] == 1243
    assert (
        first["challenger_id"]
        == "CG00_RANDOM_FILTERED"
    )
    assert (
        first["seed"]
        == _phase2_product.phase3_seed(
            1243,
            "CG00_RANDOM_FILTERED",
        )
    )
    assert first["research_only"] is True
    assert first["synthetic_fixture"] is True


def test_phase4_real_execution_mode_is_rejected():
    with _phase2_pytest.raises(
        _phase2_product.ResearchExecutorError
    ):
        _phase2_product.phase4_build_request(
            1243,
            "CG00_RANDOM_FILTERED",
            execution_mode="real_round",
        )


def test_phase4_unknown_challenger_is_rejected():
    with _phase2_pytest.raises(
        _phase2_product.ChallengerPlanError
    ):
        _phase2_product.phase4_build_request(
            1243,
            "UNKNOWN",
        )


def test_phase4_out_of_window_round_is_rejected():
    with _phase2_pytest.raises(
        _phase2_product.ChallengerPlanError
    ):
        _phase2_product.phase4_build_request(
            1242,
            "CG00_RANDOM_FILTERED",
        )


def test_phase4_adapter_must_be_callable():
    request = (
        _phase2_product.phase4_build_request(
            1243,
            "PS01_RANDOM5",
        )
    )

    with _phase2_pytest.raises(
        _phase2_product.ResearchExecutorError
    ):
        _phase2_product.phase4_execute_synthetic(
            request,
            None,
        )


def test_phase4_adapter_output_must_be_mapping():
    request = (
        _phase2_product.phase4_build_request(
            1243,
            "PS01_RANDOM5",
        )
    )

    def adapter(_plan):
        return [1, 2, 3]

    with _phase2_pytest.raises(
        _phase2_product.ResearchExecutorError
    ):
        _phase2_product.phase4_execute_synthetic(
            request,
            adapter,
        )


def test_phase4_synthetic_result_preserves_request_identity():
    request = (
        _phase2_product.phase4_build_request(
            1243,
            "HF05_SOFT_ALL4",
        )
    )

    def adapter(plan):
        return {
            "fixture": "ok",
            "seed_seen": plan["seed"],
        }

    result = (
        _phase2_product.phase4_execute_synthetic(
            request,
            adapter,
        )
    )

    assert (
        result["request_id"]
        == request["request_id"]
    )
    assert result["round"] == request["round"]
    assert (
        result["challenger_id"]
        == request["challenger_id"]
    )
    assert result["seed"] == request["seed"]

    assert result["research_only"] is True
    assert result["synthetic_fixture"] is True

    assert result["real_round_execution"] is False
    assert result["database_write"] is False
    assert result["real_shadow_publish"] is False
    assert result["production_helper_binding"] is False

    assert (
        _phase2_product.phase4_validate_result(
            request,
            result,
        )
        is True
    )


def test_phase4_pure_adapter_execution_is_deterministic():
    request = (
        _phase2_product.phase4_build_request(
            1243,
            "LG01_NO_GAP_SCORE",
        )
    )

    def adapter(plan):
        return {
            "round": plan["round"],
            "seed": plan["seed"],
            "mode": "fixture",
        }

    first = (
        _phase2_product.phase4_execute_synthetic(
            request,
            adapter,
        )
    )

    second = (
        _phase2_product.phase4_execute_synthetic(
            request,
            adapter,
        )
    )

    assert first == second
    assert first["result_id"] == second["result_id"]


def test_phase4_adapter_receives_defensive_plan_copy():
    request = (
        _phase2_product.phase4_build_request(
            1243,
            "CG01_TEMP_060",
        )
    )

    original_plan = _phase2_product._phase3_copy(
        request["plan"]
    )

    def adapter(plan):
        plan["round"] = 9999
        plan["parameters"]["temperature"] = 9.9
        return {"mutated_fixture_copy": True}

    _phase2_product.phase4_execute_synthetic(
        request,
        adapter,
    )

    assert request["plan"] == original_plan
    assert request["round"] == 1243


def test_phase4_all_24_requests_are_deterministic_and_unique():
    requests = (
        _phase2_product.phase4_build_all_requests(
            1243
        )
    )

    assert len(requests) == 24

    assert tuple(
        item["challenger_id"]
        for item in requests
    ) == _phase2_product.phase3_challenger_ids()

    assert len({
        item["request_id"]
        for item in requests
    }) == 24

    assert len({
        item["seed"]
        for item in requests
    }) == 24

    assert all(
        item["execution_mode"]
        == "synthetic_fixture"
        for item in requests
    )


def test_phase4_does_not_open_phase2_or_phase3_execution_gates():
    assert all(
        value is False
        for value in (
            _phase2_product
            .phase2_parity_authorization()
            .values()
        )
    )

    phase3 = (
        _phase2_product
        .phase3_execution_authorization()
    )

    assert all(
        value is False
        for value in phase3.values()
    )

    phase4 = (
        _phase2_product
        .phase4_executor_authorization()
    )

    assert phase4["real_round_execution"] is False
    assert phase4["real_shadow_publish"] is False
    assert phase4["database_write"] is False
    assert phase4["production_helper_binding"] is False


# === PHASE4_RESEARCH_EXECUTOR_TESTS_V1 END ===

# === PHASE5_REAL_ROUND_ADAPTER_TESTS_V1 BEGIN ===


def _phase5_make_history_db(
    tmp_path,
    rounds=(1240, 1241, 1242),
):
    db_path = tmp_path / "phase5_history.sqlite"

    connection = (
        _phase2_product
        ._phase5_sqlite3
        .connect(str(db_path))
    )

    try:
        connection.execute(
            """
            CREATE TABLE draw_history (
                round INTEGER PRIMARY KEY,
                n1 INTEGER NOT NULL,
                n2 INTEGER NOT NULL,
                n3 INTEGER NOT NULL,
                n4 INTEGER NOT NULL,
                n5 INTEGER NOT NULL,
                n6 INTEGER NOT NULL,
                bonus INTEGER
            )
            """
        )

        for round_no in rounds:
            connection.execute(
                """
                INSERT INTO draw_history (
                    round,
                    n1,
                    n2,
                    n3,
                    n4,
                    n5,
                    n6,
                    bonus
                )
                VALUES (?, 1, 2, 3, 4, 5, 6, 7)
                """,
                (round_no,),
            )

        connection.commit()

    finally:
        connection.close()

    return db_path


def test_phase5_authorization_is_preparation_only():
    auth = (
        _phase2_product
        .phase5_real_round_authorization()
    )

    assert auth["research_only"] is True
    assert auth["real_round_adapter_implementation"] is True
    assert auth["real_round_preparation"] is True

    assert auth["real_round_execution"] is False
    assert auth["challenger_execution"] is False
    assert auth["real_database_shadow_execution"] is False
    assert auth["real_shadow_publish"] is False
    assert auth["database_write"] is False
    assert auth["production_helper_binding"] is False
    assert auth["production_learning"] is False


def test_phase5_target_round_1243_snapshot_is_accepted(
    tmp_path,
):
    db_path = _phase5_make_history_db(
        tmp_path
    )

    snapshot = (
        _phase2_product
        .phase5_load_history_snapshot(
            db_path,
            round_no=1243,
        )
    )

    assert snapshot["target_round"] == 1243
    assert snapshot["history_cutoff_max_round"] == 1242
    assert snapshot["history_max_round"] == 1242
    assert snapshot["database_max_round"] == 1242
    assert snapshot["database_read_only"] is True
    assert snapshot["database_query_only"] is True
    assert snapshot["target_future_leakage"] is False


def test_phase5_round_other_than_1243_is_rejected(
    tmp_path,
):
    db_path = _phase5_make_history_db(
        tmp_path
    )

    with _phase2_pytest.raises(
        _phase2_product.RealRoundAdapterError
    ):
        _phase2_product.phase5_load_history_snapshot(
            db_path,
            round_no=1244,
        )


def test_phase5_unknown_challenger_is_rejected(
    tmp_path,
):
    db_path = _phase5_make_history_db(
        tmp_path
    )

    with _phase2_pytest.raises(
        _phase2_product.ChallengerPlanError
    ):
        _phase2_product.phase5_prepare_real_round_request(
            db_path,
            "UNKNOWN_CHALLENGER",
        )


def test_phase5_target_round_leakage_is_rejected(
    tmp_path,
):
    db_path = _phase5_make_history_db(
        tmp_path,
        rounds=(1241, 1242, 1243),
    )

    with _phase2_pytest.raises(
        _phase2_product.RealRoundAdapterError
    ):
        _phase2_product.phase5_load_history_snapshot(
            db_path
        )


def test_phase5_future_round_leakage_is_rejected(
    tmp_path,
):
    db_path = _phase5_make_history_db(
        tmp_path,
        rounds=(1241, 1242, 1244),
    )

    with _phase2_pytest.raises(
        _phase2_product.RealRoundAdapterError
    ):
        _phase2_product.phase5_load_history_snapshot(
            db_path
        )


def test_phase5_database_connection_is_query_only_and_read_only(
    tmp_path,
):
    db_path = _phase5_make_history_db(
        tmp_path
    )

    connection = (
        _phase2_product
        ._phase5_open_read_only_database(
            db_path
        )
    )

    try:
        query_only = connection.execute(
            "PRAGMA query_only"
        ).fetchone()

        assert int(query_only[0]) == 1

        with _phase2_pytest.raises(
            _phase2_product._phase5_sqlite3.OperationalError
        ):
            connection.execute(
                """
                INSERT INTO draw_history (
                    round,
                    n1,
                    n2,
                    n3,
                    n4,
                    n5,
                    n6,
                    bonus
                )
                VALUES (1243, 1, 2, 3, 4, 5, 6, 7)
                """
            )

    finally:
        connection.close()


def test_phase5_prepared_request_is_deterministic(
    tmp_path,
):
    db_path = _phase5_make_history_db(
        tmp_path
    )

    first = (
        _phase2_product
        .phase5_prepare_real_round_request(
            db_path,
            "CG00_RANDOM_FILTERED",
        )
    )

    second = (
        _phase2_product
        .phase5_prepare_real_round_request(
            db_path,
            "CG00_RANDOM_FILTERED",
        )
    )

    assert first == second
    assert first["round"] == 1243
    assert (
        first["challenger_id"]
        == "CG00_RANDOM_FILTERED"
    )
    assert first["prepared_only"] is True
    assert first["execution_authorized"] is False

    assert (
        _phase2_product
        .phase5_validate_prepared_request(
            first
        )
        is True
    )


def test_phase5_all_24_requests_preserve_frozen_identity(
    tmp_path,
):
    db_path = _phase5_make_history_db(
        tmp_path
    )

    requests = (
        _phase2_product
        .phase5_prepare_all_real_round_requests(
            db_path
        )
    )

    assert len(requests) == 24

    assert tuple(
        item["challenger_id"]
        for item in requests
    ) == _phase2_product.phase3_challenger_ids()

    assert len({
        item["request_id"]
        for item in requests
    }) == 24

    assert len({
        item["seed"]
        for item in requests
    }) == 24

    for item in requests:
        assert (
            item["seed"]
            == _phase2_product.phase3_seed(
                1243,
                item["challenger_id"],
            )
        )

        assert item["round"] == 1243
        assert item["prepared_only"] is True
        assert item["execution_authorized"] is False


def test_phase5_history_tampering_is_detected(
    tmp_path,
):
    db_path = _phase5_make_history_db(
        tmp_path
    )

    request = (
        _phase2_product
        .phase5_prepare_real_round_request(
            db_path,
            "CG01_TEMP_060",
        )
    )

    request["history_snapshot"]["history_rows"][0][
        "nums"
    ][0] = 45

    with _phase2_pytest.raises(
        _phase2_product.RealRoundAdapterError
    ):
        _phase2_product.phase5_validate_prepared_request(
            request
        )


def test_phase5_seed_tampering_is_detected(
    tmp_path,
):
    db_path = _phase5_make_history_db(
        tmp_path
    )

    request = (
        _phase2_product
        .phase5_prepare_real_round_request(
            db_path,
            "HF05_SOFT_ALL4",
        )
    )

    request["seed"] += 1

    with _phase2_pytest.raises(
        _phase2_product.RealRoundAdapterError
    ):
        _phase2_product.phase5_validate_prepared_request(
            request
        )


def test_phase5_real_execution_fails_closed(
    tmp_path,
):
    db_path = _phase5_make_history_db(
        tmp_path
    )

    request = (
        _phase2_product
        .phase5_prepare_real_round_request(
            db_path,
            "PS01_RANDOM5",
        )
    )

    with _phase2_pytest.raises(
        _phase2_product.RealRoundAdapterError
    ):
        _phase2_product.phase5_execute_real_round(
            request,
            adapter=lambda plan: {
                "should_not_run": True
            },
        )


def test_phase5_does_not_reopen_phase4_real_execution(
    tmp_path,
):
    db_path = _phase5_make_history_db(
        tmp_path
    )

    request = (
        _phase2_product
        .phase5_prepare_real_round_request(
            db_path,
            "LG01_NO_GAP_SCORE",
        )
    )

    phase4 = (
        _phase2_product
        .phase4_executor_authorization()
    )

    phase5 = (
        _phase2_product
        .phase5_real_round_authorization()
    )

    assert phase4["real_round_execution"] is False
    assert phase4["real_shadow_publish"] is False

    assert phase5["real_round_execution"] is False
    assert phase5["challenger_execution"] is False
    assert phase5["real_shadow_publish"] is False
    assert phase5["database_write"] is False

    assert request["execution_authorized"] is False


def test_phase5_snapshot_does_not_mutate_database_file(
    tmp_path,
):
    db_path = _phase5_make_history_db(
        tmp_path
    )

    before = db_path.read_bytes()

    snapshot = (
        _phase2_product
        .phase5_load_history_snapshot(
            db_path
        )
    )

    after = db_path.read_bytes()

    assert before == after
    assert snapshot["history_max_round"] == 1242


# === PHASE5_REAL_ROUND_ADAPTER_TESTS_V1 END ===

# === PHASE6_RESEARCH_EXECUTION_ENGINE_TESTS_V1 BEGIN ===


def _phase6_prepared_request(
    tmp_path,
    challenger_id="CG00_RANDOM_FILTERED",
):
    db_path = _phase5_make_history_db(
        tmp_path
    )

    return (
        _phase2_product
        .phase5_prepare_real_round_request(
            db_path,
            challenger_id,
        )
    )


def test_phase6_authorization_is_implementation_only():
    auth = (
        _phase2_product
        .phase6_execution_engine_authorization()
    )

    assert auth["engine_implementation"] is True
    assert auth["synthetic_test_execution"] is True
    assert auth["prepared_request_test_execution"] is True

    assert auth["actual_round1243_execution"] is False
    assert auth["actual_challenger_execution"] is False
    assert auth["real_shadow_publish"] is False
    assert auth["database_access_inside_engine"] is False
    assert auth["database_write"] is False
    assert auth["production_helper_binding"] is False
    assert auth["production_learning"] is False


def test_phase6_execution_matrix_contains_all_24_ids_once():
    matrix = (
        _phase2_product
        .phase6_execution_matrix()
    )

    assert len(matrix) == 24

    ids = tuple(
        item["challenger_id"]
        for item in matrix
    )

    assert ids == (
        _phase2_product
        .phase3_challenger_ids()
    )

    assert len(set(ids)) == 24


def test_phase6_unknown_challenger_is_rejected():
    with _phase2_pytest.raises(
        _phase2_product.ResearchExecutionEngineError
    ):
        _phase2_product.phase6_challenger_config(
            "UNKNOWN_CHALLENGER"
        )


def test_phase6_execution_requires_explicit_test_mode(
    tmp_path,
):
    request = _phase6_prepared_request(
        tmp_path
    )

    with _phase2_pytest.raises(
        _phase2_product.ResearchExecutionEngineError
    ):
        _phase2_product.phase6_execute_prepared_request(
            request
        )


def test_phase6_actual_round_execution_stays_fail_closed():
    with _phase2_pytest.raises(
        _phase2_product.ResearchExecutionEngineError
    ):
        _phase2_product.phase6_execute_actual_round1243()


def test_phase6_result_is_deterministic_for_same_request(
    tmp_path,
):
    request = _phase6_prepared_request(
        tmp_path
    )

    first = (
        _phase2_product
        .phase6_execute_prepared_request(
            request,
            test_mode=True,
        )
    )

    second = (
        _phase2_product
        .phase6_execute_prepared_request(
            request,
            test_mode=True,
        )
    )

    assert first == second
    assert first["result_id"] == second["result_id"]


def test_phase6_result_contract_top10_and_practical5(
    tmp_path,
):
    request = _phase6_prepared_request(
        tmp_path,
        "PS02_SCORE_TOP5",
    )

    result = (
        _phase2_product
        .phase6_execute_prepared_request(
            request,
            test_mode=True,
        )
    )

    assert result["candidate_count"] == 10000
    assert result["top_k"] == 10
    assert result["practical_k"] == 5
    assert len(result["sets"]) == 10
    assert len(result["top5_practical"]) == 5

    assert (
        _phase2_product
        .phase6_validate_result(
            result
        )
        is True
    )


def test_phase6_top10_sets_are_legal_and_diverse(
    tmp_path,
):
    request = _phase6_prepared_request(
        tmp_path,
        "HF00_CURRENT",
    )

    result = (
        _phase2_product
        .phase6_execute_prepared_request(
            request,
            test_mode=True,
        )
    )

    for item in result["sets"]:
        numbers = item["numbers"]

        assert len(numbers) == 6
        assert len(set(numbers)) == 6
        assert numbers == sorted(numbers)
        assert all(
            1 <= number <= 45
            for number in numbers
        )

        features = item["features"]

        assert 90 <= features["sum"] <= 200
        assert features["odd_even"] in (
            "2:4",
            "3:3",
            "4:2",
        )
        assert features["max_consecutive_run"] <= 2
        assert features["max_same_ending"] <= 2
        assert features["previous_overlap"] <= 1
        assert features["long_gap_count"] >= 1
        assert features["max_same_decade"] <= 3

    assert (
        result["diversity"]["max_jaccard"]
        <= 0.33
    )


def test_phase6_engine_does_not_access_database(
    tmp_path,
    monkeypatch,
):
    request = _phase6_prepared_request(
        tmp_path
    )

    def explode(*args, **kwargs):
        raise AssertionError(
            "database access inside Phase-6 engine"
        )

    monkeypatch.setattr(
        _phase2_product,
        "_phase5_open_read_only_database",
        explode,
    )

    monkeypatch.setattr(
        _phase2_product._phase5_sqlite3,
        "connect",
        explode,
    )

    result = (
        _phase2_product
        .phase6_execute_prepared_request(
            request,
            test_mode=True,
        )
    )

    assert result["database_write"] is False


def test_phase6_tampered_prepared_request_is_rejected(
    tmp_path,
):
    request = _phase6_prepared_request(
        tmp_path
    )

    request["seed"] += 1

    with _phase2_pytest.raises(
        _phase2_product.ResearchExecutionEngineError
    ):
        _phase2_product.phase6_execute_prepared_request(
            request,
            test_mode=True,
        )


def test_phase6_candidate_generation_variants_change_only_generation_config():
    base = (
        _phase2_product
        .phase6_challenger_config(
            "CG01_TEMP_060"
        )
    )

    other = (
        _phase2_product
        .phase6_challenger_config(
            "CG02_TEMP_110"
        )
    )

    assert base["generation"] != other["generation"]
    assert base["scoring"] == other["scoring"]
    assert base["filters"] == other["filters"]
    assert (
        base["practical_selector"]
        == other["practical_selector"]
    )


def test_phase6_scoring_variants_change_only_scoring_config():
    base = (
        _phase2_product
        .phase6_challenger_config(
            "SC00_CURRENT"
        )
    )

    other = (
        _phase2_product
        .phase6_challenger_config(
            "SC01_RANDOM_RANK"
        )
    )

    assert base["generation"] == other["generation"]
    assert base["scoring"] != other["scoring"]
    assert base["filters"] == other["filters"]
    assert (
        base["practical_selector"]
        == other["practical_selector"]
    )


def test_phase6_filter_variants_change_only_filter_config():
    base = (
        _phase2_product
        .phase6_challenger_config(
            "HF00_CURRENT"
        )
    )

    other = (
        _phase2_product
        .phase6_challenger_config(
            "HF01_SOFT_ODD_EVEN"
        )
    )

    assert base["generation"] == other["generation"]
    assert base["scoring"] == other["scoring"]
    assert base["filters"] != other["filters"]
    assert (
        base["practical_selector"]
        == other["practical_selector"]
    )


def test_phase6_long_gap_variants_are_narrowly_scoped():
    current = (
        _phase2_product
        .phase6_challenger_config(
            "LG00_CURRENT"
        )
    )

    no_score = (
        _phase2_product
        .phase6_challenger_config(
            "LG01_NO_GAP_SCORE"
        )
    )

    no_hard = (
        _phase2_product
        .phase6_challenger_config(
            "LG02_NO_GAP_HARD_RULE"
        )
    )

    assert (
        current["generation"]
        == no_score["generation"]
        == no_hard["generation"]
    )

    assert (
        no_score["scoring"]
        != current["scoring"]
    )

    assert (
        no_hard["filters"]
        != current["filters"]
    )


def test_phase6_practical_variants_change_only_selector_config():
    base = (
        _phase2_product
        .phase6_challenger_config(
            "PS00_CURRENT_MMR"
        )
    )

    other = (
        _phase2_product
        .phase6_challenger_config(
            "PS01_RANDOM5"
        )
    )

    assert base["generation"] == other["generation"]
    assert base["scoring"] == other["scoring"]
    assert base["filters"] == other["filters"]

    assert (
        base["practical_selector"]
        != other["practical_selector"]
    )


def test_phase6_result_validation_detects_tampering(
    tmp_path,
):
    request = _phase6_prepared_request(
        tmp_path,
        "SC02_QUANTILE_DIAGNOSTIC",
    )

    result = (
        _phase2_product
        .phase6_execute_prepared_request(
            request,
            test_mode=True,
        )
    )

    result["sets"][0]["numbers"][0] = 45

    with _phase2_pytest.raises(
        _phase2_product.ResearchExecutionEngineError
    ):
        _phase2_product.phase6_validate_result(
            result
        )


# === PHASE6_RESEARCH_EXECUTION_ENGINE_TESTS_V1 END ===

# === PHASE7_ACTUAL_SHADOW_EXECUTION_BRIDGE_TESTS_V1 BEGIN ===


def test_phase7_bridge_authorization_is_implementation_only():
    authorization = (
        _phase2_product
        .phase7_execution_bridge_authorization()
    )

    assert authorization["research_only"] is True
    assert authorization["bridge_implementation"] is True

    assert (
        authorization[
            "explicit_runtime_authorization_required"
        ]
        is True
    )

    assert (
        authorization[
            "actual_round1243_execution"
        ]
        is False
    )

    assert (
        authorization[
            "actual_challenger_execution"
        ]
        is False
    )

    assert authorization["real_shadow_publish"] is False
    assert authorization["database_write"] is False
    assert authorization["production_cli_mutation"] is False
    assert authorization["production_learning"] is False


def test_phase7_shadow_prepared_request_requires_explicit_authorization(
    tmp_path,
):
    request = _phase6_prepared_request(
        tmp_path
    )

    with _phase2_pytest.raises(
        _phase2_product.ResearchExecutionEngineError
    ):
        (
            _phase2_product
            .phase7_execute_shadow_prepared_request(
                request
            )
        )


def test_phase7_shadow_prepared_request_uses_actual_shadow_context(
    tmp_path,
):
    request = _phase6_prepared_request(
        tmp_path
    )

    result = (
        _phase2_product
        .phase7_execute_shadow_prepared_request(
            request,
            execution_authorized=True,
        )
    )

    assert result["research_only"] is True

    assert (
        result[
            "execution_context"
        ]
        == "phase7_actual_shadow"
    )

    assert (
        result[
            "actual_round_execution"
        ]
        is True
    )

    assert result["real_shadow_publish"] is False
    assert result["database_write"] is False

    assert (
        _phase2_product
        .phase7_validate_shadow_result(
            result
        )
        is True
    )


def test_phase7_shadow_prepared_request_is_deterministic(
    tmp_path,
):
    request = _phase6_prepared_request(
        tmp_path,
        "CG00_RANDOM_FILTERED",
    )

    first = (
        _phase2_product
        .phase7_execute_shadow_prepared_request(
            request,
            execution_authorized=True,
        )
    )

    second = (
        _phase2_product
        .phase7_execute_shadow_prepared_request(
            request,
            execution_authorized=True,
        )
    )

    assert first == second
    assert first["result_id"] == second["result_id"]


def test_phase7_batch_requires_authorization_before_preparation(
    monkeypatch,
):
    calls = []

    def explode(
        db_path,
        *,
        round_no=1243,
    ):
        calls.append(
            (
                db_path,
                round_no,
            )
        )

        raise AssertionError(
            "preparation must not run"
        )

    monkeypatch.setattr(
        _phase2_product,
        "phase5_prepare_all_real_round_requests",
        explode,
    )

    with _phase2_pytest.raises(
        _phase2_product.ResearchExecutionEngineError
    ):
        (
            _phase2_product
            .phase7_execute_actual_shadow_round1243(
                "unused.db"
            )
        )

    assert calls == []


def test_phase7_batch_orchestration_is_exactly_24_in_registry_order(
    monkeypatch,
):
    challenger_ids = tuple(
        _phase2_product
        .phase3_challenger_ids()
    )

    assert len(challenger_ids) == 24

    requests = tuple(
        {
            "challenger_id":
                challenger_id,
        }
        for challenger_id
        in challenger_ids
    )

    preparation_calls = []

    def fake_prepare(
        db_path,
        *,
        round_no=1243,
    ):
        preparation_calls.append(
            (
                db_path,
                round_no,
            )
        )

        return requests

    validated_requests = []

    def fake_validate_request(
        request,
    ):
        validated_requests.append(
            request[
                "challenger_id"
            ]
        )

        return True

    executed = []

    def fake_execute(
        request,
        *,
        execution_authorized=False,
    ):
        assert execution_authorized is True

        challenger_id = request[
            "challenger_id"
        ]

        executed.append(
            challenger_id
        )

        return {
            "challenger_id":
                challenger_id,

            "result_id":
                "RID-" + challenger_id,
        }

    validated_results = []

    def fake_validate_result(
        result,
    ):
        validated_results.append(
            result[
                "challenger_id"
            ]
        )

        return True

    monkeypatch.setattr(
        _phase2_product,
        "phase5_prepare_all_real_round_requests",
        fake_prepare,
    )

    monkeypatch.setattr(
        _phase2_product,
        "_phase6_validate_prepared_request",
        fake_validate_request,
    )

    monkeypatch.setattr(
        _phase2_product,
        "phase7_execute_shadow_prepared_request",
        fake_execute,
    )

    monkeypatch.setattr(
        _phase2_product,
        "phase7_validate_shadow_result",
        fake_validate_result,
    )

    results = (
        _phase2_product
        .phase7_execute_actual_shadow_round1243(
            "unused.db",
            execution_authorized=True,
        )
    )

    assert preparation_calls == [
        (
            "unused.db",
            1243,
        )
    ]

    assert tuple(
        result[
            "challenger_id"
        ]
        for result
        in results
    ) == challenger_ids

    assert tuple(
        validated_requests
    ) == challenger_ids

    assert tuple(
        executed
    ) == challenger_ids

    assert tuple(
        validated_results
    ) == challenger_ids


def test_phase7_preserves_phase6_test_only_contract(
    tmp_path,
):
    request = _phase6_prepared_request(
        tmp_path
    )

    result = (
        _phase2_product
        .phase6_execute_prepared_request(
            request,
            test_mode=True,
        )
    )

    assert (
        result[
            "execution_context"
        ]
        == "phase6_test_only"
    )

    assert (
        result[
            "actual_round_execution"
        ]
        is False
    )

    assert result["real_shadow_publish"] is False
    assert result["database_write"] is False


def test_phase7_phase6_executor_still_rejects_without_test_mode(
    tmp_path,
):
    request = _phase6_prepared_request(
        tmp_path
    )

    with _phase2_pytest.raises(
        _phase2_product.ResearchExecutionEngineError
    ):
        (
            _phase2_product
            .phase6_execute_prepared_request(
                request
            )
        )


def test_phase7_preserves_phase6_actual_wrapper_fail_closed():
    with _phase2_pytest.raises(
        _phase2_product.ResearchExecutionEngineError
    ):
        (
            _phase2_product
            .phase6_execute_actual_round1243()
        )


# === PHASE7_ACTUAL_SHADOW_EXECUTION_BRIDGE_TESTS_V1 END ===
