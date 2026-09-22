from __future__ import annotations

import importlib
import json
import sqlite3
from pathlib import Path

import pytest

import lrp.research_shadow_protocol as protocol
import lrp.research_shadow_runner as runner


CHALLENGER_ID = "CG00_RANDOM_FILTERED"


def make_registry(
    path: Path,
    *,
    round_no: int = 1243,
    challenger_id: str = CHALLENGER_ID,
    seed: int | None = None,
) -> Path:
    if seed is None:
        seed = protocol.derive_shadow_seed(
            round_no,
            challenger_id,
        )

    payload = {
        "state":
            "CHALLENGER_DESIGN_FROZEN",

        "shadow_window": {
            "start":
                1243,

            "end":
                1252,
        },

        "challengers": {
            "candidate_generation": [
                {
                    "id":
                        challenger_id,
                }
            ],
        },

        "seed_registry": {
            str(round_no): {
                challenger_id:
                    seed,
            },
        },
    }

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return path


def make_database(
    path: Path,
    *,
    max_round: int,
    min_round: int | None = None,
) -> Path:
    if min_round is None:
        min_round = max_round

    connection = sqlite3.connect(
        path
    )

    try:
        connection.execute(
            """
            CREATE TABLE draw_history (
                round INTEGER PRIMARY KEY,
                n1 INTEGER,
                n2 INTEGER,
                n3 INTEGER,
                n4 INTEGER,
                n5 INTEGER,
                n6 INTEGER,
                bonus INTEGER
            )
            """
        )

        for round_no in range(
            min_round,
            max_round + 1,
        ):
            connection.execute(
                """
                INSERT INTO draw_history (
                    round,n1,n2,n3,n4,n5,n6,bonus
                )
                VALUES (?,?,?,?,?,?,?,?)
                """,
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

    return path


def test_import_safety_and_no_side_effect_contract():
    reloaded_protocol = importlib.reload(
        protocol
    )

    reloaded_runner = importlib.reload(
        runner
    )

    assert (
        reloaded_runner.CHALLENGER_EXECUTION_ENABLED
        is False
    )

    assert (
        reloaded_runner.DATABASE_WRITE_ENABLED
        is False
    )

    assert (
        reloaded_runner.PRODUCTION_CLI_REGISTRATION_ENABLED
        is False
    )

    assert (
        reloaded_protocol.SHADOW_START_ROUND
        == 1243
    )

    assert (
        reloaded_protocol.SHADOW_END_ROUND
        == 1252
    )


def test_deterministic_seed_is_stable():
    first = protocol.derive_shadow_seed(
        1243,
        CHALLENGER_ID,
    )

    second = protocol.derive_shadow_seed(
        1243,
        CHALLENGER_ID,
    )

    assert first == second

    assert (
        1
        <= first
        < 2147483647
    )


def test_registry_explicit_seed_must_match_frozen_rule(
    tmp_path: Path,
):
    wrong_seed = (
        protocol.derive_shadow_seed(
            1243,
            CHALLENGER_ID,
        )
        + 1
    )

    registry_path = make_registry(
        tmp_path / "registry.json",
        seed=wrong_seed,
    )

    registry = protocol.load_frozen_registry(
        registry_path
    )

    with pytest.raises(
        protocol.SeedContractError
    ):
        protocol.resolve_pre_registered_seed(
            registry,
            1243,
            CHALLENGER_ID,
        )


def test_challenger_id_validation_rejects_unknown(
    tmp_path: Path,
):
    registry_path = make_registry(
        tmp_path / "registry.json"
    )

    registry = protocol.load_frozen_registry(
        registry_path
    )

    with pytest.raises(
        protocol.ChallengerIdError
    ):
        protocol.validate_challenger_id(
            registry,
            "CG99_NOT_REGISTERED",
        )


@pytest.mark.parametrize(
    "round_no",
    [
        1242,
        1253,
    ],
)
def test_shadow_round_validation_rejects_outside_window(
    round_no: int,
):
    with pytest.raises(
        protocol.ShadowRoundError
    ):
        protocol.validate_shadow_round(
            round_no
        )


def test_database_is_read_only(
    tmp_path: Path,
):
    db_path = make_database(
        tmp_path / "lotto.db",
        max_round=1242,
    )

    connection = runner.open_database_read_only(
        db_path
    )

    try:
        with pytest.raises(
            sqlite3.OperationalError
        ):
            connection.execute(
                """
                INSERT INTO draw_history (
                    round,n1,n2,n3,n4,n5,n6,bonus
                )
                VALUES (1243,1,2,3,4,5,6,7)
                """
            )

    finally:
        connection.close()


def test_history_cutoff_exact_target_minus_one(
    tmp_path: Path,
):
    db_path = make_database(
        tmp_path / "lotto.db",
        min_round=1241,
        max_round=1242,
    )

    result = runner.inspect_history_cutoff(
        db_path,
        1243,
    )

    assert (
        result["max_round"]
        == 1242
    )

    assert (
        result["expected_cutoff"]
        == 1242
    )


def test_target_result_leakage_is_rejected(
    tmp_path: Path,
):
    db_path = make_database(
        tmp_path / "lotto.db",
        max_round=1243,
    )

    with pytest.raises(
        runner.ShadowDataLeakageError
    ):
        runner.inspect_history_cutoff(
            db_path,
            1243,
        )


def test_future_result_leakage_is_rejected(
    tmp_path: Path,
):
    db_path = make_database(
        tmp_path / "lotto.db",
        max_round=1244,
    )

    with pytest.raises(
        runner.ShadowDataLeakageError
    ):
        runner.inspect_history_cutoff(
            db_path,
            1243,
        )


def test_incomplete_history_cutoff_is_rejected(
    tmp_path: Path,
):
    db_path = make_database(
        tmp_path / "lotto.db",
        max_round=1241,
    )

    with pytest.raises(
        runner.HistoryCutoffError
    ):
        runner.inspect_history_cutoff(
            db_path,
            1243,
        )


def test_artifact_overwrite_is_rejected(
    tmp_path: Path,
):
    paths = protocol.construct_shadow_artifact_paths(
        tmp_path,
        1243,
    )

    paths["root"].mkdir(
        parents=True
    )

    with pytest.raises(
        protocol.ShadowOutputExistsError
    ):
        protocol.reject_existing_output(
            paths["root"]
        )


def test_dry_run_plan_has_no_execution_or_write_authority(
    tmp_path: Path,
):
    registry_path = make_registry(
        tmp_path / "registry.json"
    )

    db_path = make_database(
        tmp_path / "lotto.db",
        max_round=1242,
    )

    plan = runner.build_dry_run_plan(
        repo_root=tmp_path,
        registry_path=registry_path,
        database_path=db_path,
        target_round=1243,
        challenger_id=CHALLENGER_ID,
    )

    assert plan["mode"] == "SCAFFOLD_ONLY"

    assert (
        plan["database_mode"]
        == "READ_ONLY"
    )

    assert (
        plan["database_write_enabled"]
        is False
    )

    assert (
        plan["challenger_execution_enabled"]
        is False
    )

    assert (
        plan["production_cli_registration_enabled"]
        is False
    )

    assert (
        plan["production_output_mutation"]
        is False
    )

    assert (
        plan["execution_authorized"]
        is False
    )

    assert (
        plan["history_cutoff"]["max_round"]
        == 1242
    )

    assert not Path(
        plan["artifact_paths"]["root"]
    ).exists()


def test_dry_run_does_not_mutate_database(
    tmp_path: Path,
):
    registry_path = make_registry(
        tmp_path / "registry.json"
    )

    db_path = make_database(
        tmp_path / "lotto.db",
        max_round=1242,
    )

    before = db_path.read_bytes()

    runner.build_dry_run_plan(
        repo_root=tmp_path,
        registry_path=registry_path,
        database_path=db_path,
        target_round=1243,
        challenger_id=CHALLENGER_ID,
    )

    after = db_path.read_bytes()

    assert after == before


def test_production_style_paths_are_not_created_by_dry_run(
    tmp_path: Path,
):
    registry_path = make_registry(
        tmp_path / "registry.json"
    )

    db_path = make_database(
        tmp_path / "lotto.db",
        max_round=1242,
    )

    production_cli = (
        tmp_path
        / "lrp"
        / "cli.py"
    )

    production_cli.parent.mkdir(
        parents=True
    )

    production_cli.write_text(
        "ORIGINAL = True\n",
        encoding="utf-8",
    )

    before_cli = production_cli.read_bytes()

    runner.build_dry_run_plan(
        repo_root=tmp_path,
        registry_path=registry_path,
        database_path=db_path,
        target_round=1243,
        challenger_id=CHALLENGER_ID,
    )

    assert (
        production_cli.read_bytes()
        == before_cli
    )

    assert not (
        tmp_path
        / "artifacts"
        / "research"
        / "shadow"
        / "round_1243"
    ).exists()