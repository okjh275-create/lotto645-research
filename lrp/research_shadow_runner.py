"""Read-only scaffold for LRP controlled shadow research.

OP-127/OP-127R scope is validation and dry-run planning only.

This module does not:
- generate challenger Lotto combinations;
- invoke production prediction or review;
- write to data/lotto.db;
- mutate production artifacts;
- register a production CLI route;
- stage, commit, or push repository changes.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from lrp.research_shadow_protocol import (
    construct_shadow_artifact_paths,
    load_frozen_registry,
    reject_existing_output,
    resolve_pre_registered_seed,
    validate_challenger_id,
    validate_shadow_round,
)


CHALLENGER_EXECUTION_ENABLED = False
DATABASE_WRITE_ENABLED = False
PRODUCTION_CLI_REGISTRATION_ENABLED = False


class ShadowRunnerError(RuntimeError):
    """Base exception for shadow scaffold runner failures."""


class HistoryCutoffError(ShadowRunnerError):
    """Raised when history does not end at target_round - 1."""


class ShadowDataLeakageError(HistoryCutoffError):
    """Raised when target/future result data is visible."""


def open_database_read_only(
    database_path: str | Path,
) -> sqlite3.Connection:
    """Open SQLite in read-only mode and enforce query_only."""

    path = Path(
        database_path
    ).resolve()

    if not path.is_file():
        raise ShadowRunnerError(
            f"database not found: {path}"
        )

    uri = (
        path.as_uri()
        + "?mode=ro"
    )

    connection = sqlite3.connect(
        uri,
        uri=True,
    )

    try:
        connection.execute(
            "PRAGMA query_only = ON"
        )

        query_only = int(
            connection.execute(
                "PRAGMA query_only"
            ).fetchone()[0]
        )

        if query_only != 1:
            raise ShadowRunnerError(
                "SQLite query_only could not be enforced"
            )

    except Exception:
        connection.close()
        raise

    return connection


def inspect_history_cutoff(
    database_path: str | Path,
    target_round: int,
) -> dict[str, int]:
    """Require historical data to end exactly at target_round - 1."""

    validate_shadow_round(
        target_round
    )

    expected_cutoff = (
        target_round - 1
    )

    connection = open_database_read_only(
        database_path
    )

    try:
        table_exists = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM sqlite_master
                WHERE type = 'table'
                  AND name = 'draw_history'
                """
            ).fetchone()[0]
        )

        if table_exists != 1:
            raise HistoryCutoffError(
                "draw_history table is missing"
            )

        row = connection.execute(
            """
            SELECT
                MIN(round),
                MAX(round),
                COUNT(*),
                COUNT(DISTINCT round)
            FROM draw_history
            """
        ).fetchone()

    finally:
        connection.close()

    if row is None:
        raise HistoryCutoffError(
            "draw_history aggregate returned no row"
        )

    min_round, max_round, row_count, unique_rounds = row

    if max_round is None:
        raise HistoryCutoffError(
            "draw_history is empty"
        )

    min_round = int(
        min_round
    )

    max_round = int(
        max_round
    )

    row_count = int(
        row_count
    )

    unique_rounds = int(
        unique_rounds
    )

    if max_round >= target_round:
        raise ShadowDataLeakageError(
            "target/future result leakage detected: "
            f"target_round={target_round}, "
            f"history_max_round={max_round}"
        )

    if max_round != expected_cutoff:
        raise HistoryCutoffError(
            "history cutoff is incomplete: "
            f"target_round={target_round}, "
            f"expected={expected_cutoff}, "
            f"actual={max_round}"
        )

    if row_count != unique_rounds:
        raise HistoryCutoffError(
            "duplicate historical rounds detected"
        )

    return {
        "min_round":
            min_round,

        "max_round":
            max_round,

        "row_count":
            row_count,

        "unique_rounds":
            unique_rounds,

        "expected_cutoff":
            expected_cutoff,
    }


def build_dry_run_plan(
    *,
    repo_root: str | Path,
    registry_path: str | Path,
    database_path: str | Path,
    target_round: int,
    challenger_id: str,
) -> dict[str, Any]:
    """Build a no-write and no-generation shadow plan."""

    validate_shadow_round(
        target_round
    )

    registry = load_frozen_registry(
        registry_path
    )

    validate_challenger_id(
        registry,
        challenger_id,
    )

    seed = resolve_pre_registered_seed(
        registry,
        target_round,
        challenger_id,
    )

    history = inspect_history_cutoff(
        database_path,
        target_round,
    )

    paths = construct_shadow_artifact_paths(
        repo_root,
        target_round,
    )

    reject_existing_output(
        paths["root"]
    )

    return {
        "schema_version":
            1,

        "artifact_type":
            "shadow_dry_run_plan",

        "mode":
            "SCAFFOLD_ONLY",

        "target_round":
            target_round,

        "challenger_id":
            challenger_id,

        "seed":
            seed,

        "history_cutoff":
            history,

        "database_mode":
            "READ_ONLY",

        "database_write_enabled":
            False,

        "challenger_execution_enabled":
            False,

        "production_cli_registration_enabled":
            False,

        "production_output_mutation":
            False,

        "artifact_overwrite":
            False,

        "artifact_paths": {
            key:
                str(value)
            for key, value in paths.items()
        },

        "execution_authorized":
            False,

        "next_required_state":
            "SHADOW_EXECUTION_IMPLEMENTATION_NOT_YET_AUTHORIZED",
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the research-only direct module parser."""

    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "lrp.research_shadow_runner"
        ),
        description=(
            "LRP shadow-research scaffold. "
            "Produces a validation plan only."
        ),
    )

    parser.add_argument(
        "--repo-root",
        required=True,
    )

    parser.add_argument(
        "--registry",
        required=True,
    )

    parser.add_argument(
        "--database",
        required=True,
    )

    parser.add_argument(
        "--round",
        dest="target_round",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--challenger-id",
        required=True,
    )

    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    """Run the research-only dry-plan scaffold."""

    parser = build_argument_parser()

    args = parser.parse_args(
        argv
    )

    plan = build_dry_run_plan(
        repo_root=args.repo_root,
        registry_path=args.registry,
        database_path=args.database,
        target_round=args.target_round,
        challenger_id=args.challenger_id,
    )

    print(
        json.dumps(
            plan,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )