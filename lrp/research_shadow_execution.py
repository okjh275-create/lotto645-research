from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SHADOW_START = 1243
SHADOW_END = 1252
SEED_MODULUS = 2147483647

EXPECTED_CHALLENGER_COUNT = 24

EXPECTED_PREFIX_COUNTS = {
    "CG": 6,
    "HF": 6,
    "LG": 3,
    "PS": 5,
    "SC": 4,
}

CHALLENGER_ID_PATTERN = re.compile(
    r"^(?:CG|PS|SC|HF|LG)\d{2}(?:_[A-Z0-9_]+)?$"
)

REQUIRED_ARTIFACT_FILES = (
    "shadow_plan.json",
    "shadow_results.json",
    "shadow_manifest.json",
)


class ShadowExecutionContractError(RuntimeError):
    pass


class RegistryContractError(ShadowExecutionContractError):
    pass


class HistoryContractError(ShadowExecutionContractError):
    pass


class PredictionContractError(ShadowExecutionContractError):
    pass


class ArtifactContractError(ShadowExecutionContractError):
    pass


@dataclass(frozen=True)
class ArtifactTransactionPlan:
    final_dir: Path
    temp_dir: Path
    required_files: tuple[str, ...]


def sha256_file(path: str | Path) -> str:
    target = Path(path)

    h = hashlib.sha256()

    with target.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest().upper()


def validate_shadow_round(round_no: int) -> int:
    if isinstance(round_no, bool) or not isinstance(
        round_no,
        int,
    ):
        raise ShadowExecutionContractError(
            "shadow round must be an integer"
        )

    if not (
        SHADOW_START
        <= round_no
        <= SHADOW_END
    ):
        raise ShadowExecutionContractError(
            "shadow round outside frozen window: "
            f"{round_no}"
        )

    return round_no


def validate_challenger_id(
    challenger_id: str,
) -> str:
    if (
        not isinstance(
            challenger_id,
            str,
        )
        or not CHALLENGER_ID_PATTERN.fullmatch(
            challenger_id
        )
    ):
        raise RegistryContractError(
            "invalid challenger id: "
            f"{challenger_id!r}"
        )

    return challenger_id


def derive_shadow_seed(
    round_no: int,
    challenger_id: str,
) -> int:
    validate_shadow_round(round_no)
    validate_challenger_id(
        challenger_id
    )

    token = (
        "LRP-v4.0|shadow|"
        f"{round_no}|"
        f"{challenger_id}"
    )

    digest = hashlib.sha256(
        token.encode("utf-8")
    ).digest()

    value = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    seed = value % SEED_MODULUS

    if seed == 0:
        seed = 1

    return seed


def _walk_json(
    value: Any,
):
    yield value

    if isinstance(value, Mapping):
        for key, child in value.items():
            yield key
            yield from _walk_json(child)

    elif isinstance(
        value,
        Sequence,
    ) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for child in value:
            yield from _walk_json(child)


def collect_challenger_ids(
    payload: Any,
) -> tuple[str, ...]:
    found: set[str] = set()

    for value in _walk_json(
        payload
    ):
        if (
            isinstance(value, str)
            and CHALLENGER_ID_PATTERN.fullmatch(
                value
            )
        ):
            found.add(value)

    return tuple(
        sorted(found)
    )


def _prefix_counts(
    challenger_ids: Sequence[str],
) -> dict[str, int]:
    counts: dict[str, int] = {}

    for challenger_id in (
        challenger_ids
    ):
        prefix = challenger_id[:2]

        counts[prefix] = (
            counts.get(prefix, 0)
            + 1
        )

    return dict(
        sorted(counts.items())
    )


def load_registry(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    registry_path = Path(path)

    if not registry_path.is_file():
        raise RegistryContractError(
            "registry file missing: "
            f"{registry_path}"
        )

    actual_sha = sha256_file(
        registry_path
    )

    if (
        expected_sha256 is not None
        and actual_sha
        != expected_sha256.upper()
    ):
        raise RegistryContractError(
            "registry SHA256 mismatch"
        )

    try:
        payload = json.loads(
            registry_path.read_text(
                encoding="utf-8-sig"
            )
        )

    except Exception as exc:
        raise RegistryContractError(
            "registry JSON invalid"
        ) from exc

    challenger_ids = (
        collect_challenger_ids(
            payload
        )
    )

    if (
        len(challenger_ids)
        != EXPECTED_CHALLENGER_COUNT
    ):
        raise RegistryContractError(
            "registry must contain exactly "
            f"{EXPECTED_CHALLENGER_COUNT} "
            "unique challengers"
        )

    prefix_counts = _prefix_counts(
        challenger_ids
    )

    if (
        prefix_counts
        != EXPECTED_PREFIX_COUNTS
    ):
        raise RegistryContractError(
            "registry category counts mismatch"
        )

    return {
        "path":
            str(
                registry_path.resolve()
            ),

        "sha256":
            actual_sha,

        "challenger_ids":
            list(
                challenger_ids
            ),

        "prefix_counts":
            prefix_counts,

        "payload":
            payload,
    }


def open_history_read_only(
    path: str | Path,
) -> sqlite3.Connection:
    database_path = Path(path)

    if not database_path.is_file():
        raise HistoryContractError(
            "history database missing: "
            f"{database_path}"
        )

    uri = (
        database_path
        .resolve()
        .as_uri()
        + "?mode=ro"
    )

    try:
        connection = sqlite3.connect(
            uri,
            uri=True,
        )

        connection.execute(
            "PRAGMA query_only = ON"
        )

        query_only = int(
            connection.execute(
                "PRAGMA query_only"
            ).fetchone()[0]
        )

        if query_only != 1:
            connection.close()

            raise HistoryContractError(
                "query_only could not be enabled"
            )

        return connection

    except HistoryContractError:
        raise

    except Exception as exc:
        raise HistoryContractError(
            "unable to open read-only "
            "history database"
        ) from exc


def inspect_history_cutoff(
    path: str | Path,
    target_round: int,
) -> dict[str, Any]:
    validate_shadow_round(
        target_round
    )

    expected_cutoff = (
        target_round - 1
    )

    connection = open_history_read_only(
        path
    )

    try:
        integrity = connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        if integrity != "ok":
            raise HistoryContractError(
                "history database integrity "
                "check failed"
            )

        min_round, max_round, row_count, unique_rounds = (
            connection.execute(
                "SELECT "
                "MIN(round), "
                "MAX(round), "
                "COUNT(*), "
                "COUNT(DISTINCT round) "
                "FROM draw_history"
            ).fetchone()
        )

        if max_round is None:
            raise HistoryContractError(
                "history database is empty"
            )

        max_round = int(
            max_round
        )

        target_count = int(
            connection.execute(
                "SELECT COUNT(*) "
                "FROM draw_history "
                "WHERE round = ?",
                (target_round,),
            ).fetchone()[0]
        )

        future_count = int(
            connection.execute(
                "SELECT COUNT(*) "
                "FROM draw_history "
                "WHERE round > ?",
                (target_round,),
            ).fetchone()[0]
        )

        if (
            max_round
            != expected_cutoff
        ):
            raise HistoryContractError(
                "history cutoff mismatch: "
                f"expected {expected_cutoff}, "
                f"got {max_round}"
            )

        if target_count != 0:
            raise HistoryContractError(
                "target result leakage detected"
            )

        if future_count != 0:
            raise HistoryContractError(
                "future result leakage detected"
            )

        return {
            "database":
                str(
                    Path(path).resolve()
                ),

            "integrity":
                integrity,

            "target_round":
                target_round,

            "expected_cutoff":
                expected_cutoff,

            "min_round":
                (
                    int(min_round)
                    if min_round
                    is not None
                    else None
                ),

            "max_round":
                max_round,

            "row_count":
                int(row_count),

            "unique_rounds":
                int(unique_rounds),

            "target_count":
                target_count,

            "future_count":
                future_count,

            "query_only":
                True,

            "database_write":
                False,
        }

    finally:
        connection.close()


def load_frozen_prediction(
    path: str | Path,
    target_round: int,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    validate_shadow_round(
        target_round
    )

    prediction_path = Path(path)

    if not prediction_path.is_file():
        raise PredictionContractError(
            "frozen production prediction "
            "missing"
        )

    actual_sha = sha256_file(
        prediction_path
    )

    if (
        expected_sha256 is not None
        and actual_sha
        != expected_sha256.upper()
    ):
        raise PredictionContractError(
            "production prediction SHA256 "
            "mismatch"
        )

    try:
        payload = json.loads(
            prediction_path.read_text(
                encoding="utf-8-sig"
            )
        )

    except Exception as exc:
        raise PredictionContractError(
            "production prediction JSON "
            "invalid"
        ) from exc

    if not isinstance(
        payload,
        Mapping,
    ):
        raise PredictionContractError(
            "production prediction root "
            "must be an object"
        )

    if int(
        payload.get(
            "round",
            -1,
        )
    ) != target_round:
        raise PredictionContractError(
            "production prediction round "
            "mismatch"
        )

    sets = payload.get(
        "sets"
    )

    practical = payload.get(
        "top5_practical"
    )

    if (
        not isinstance(
            sets,
            list,
        )
        or len(sets) != 10
    ):
        raise PredictionContractError(
            "production prediction must "
            "contain exactly 10 sets"
        )

    if (
        not isinstance(
            practical,
            list,
        )
        or len(practical) != 5
    ):
        raise PredictionContractError(
            "production prediction must "
            "contain exactly 5 practical "
            "set ids"
        )

    return {
        "path":
            str(
                prediction_path.resolve()
            ),

        "sha256":
            actual_sha,

        "round":
            target_round,

        "set_count":
            len(sets),

        "top5_count":
            len(practical),

        "payload":
            payload,
    }


def plan_artifact_transaction(
    repo_root: str | Path,
    target_round: int,
) -> ArtifactTransactionPlan:
    validate_shadow_round(
        target_round
    )

    root = Path(
        repo_root
    ).resolve()

    shadow_root = (
        root
        / "artifacts"
        / "research"
        / "shadow"
    )

    final_dir = (
        shadow_root
        / f"round_{target_round}"
    )

    temp_dir = (
        shadow_root
        / f".round_{target_round}.tmp"
    )

    if final_dir.exists():
        raise ArtifactContractError(
            "final shadow artifact "
            "directory already exists"
        )

    if temp_dir.exists():
        raise ArtifactContractError(
            "temporary shadow artifact "
            "directory already exists"
        )

    return ArtifactTransactionPlan(
        final_dir=final_dir,
        temp_dir=temp_dir,
        required_files=(
            REQUIRED_ARTIFACT_FILES
        ),
    )


def build_control_plan(
    *,
    repo_root: str | Path,
    target_round: int,
    registry_path: str | Path,
    history_database: str | Path,
    production_prediction_path: str | Path,
    expected_registry_sha256: str | None = None,
    expected_prediction_sha256: str | None = None,
) -> dict[str, Any]:
    validate_shadow_round(
        target_round
    )

    registry = load_registry(
        registry_path,
        expected_sha256=(
            expected_registry_sha256
        ),
    )

    history = inspect_history_cutoff(
        history_database,
        target_round,
    )

    prediction = load_frozen_prediction(
        production_prediction_path,
        target_round,
        expected_sha256=(
            expected_prediction_sha256
        ),
    )

    transaction = (
        plan_artifact_transaction(
            repo_root,
            target_round,
        )
    )

    seeds = {
        challenger_id:
            derive_shadow_seed(
                target_round,
                challenger_id,
            )
        for challenger_id
        in registry[
            "challenger_ids"
        ]
    }

    if len(seeds) != 24:
        raise ShadowExecutionContractError(
            "control plan does not contain "
            "24 challenger seeds"
        )

    return {
        "schema_version":
            1,

        "phase":
            "CONTROL_AND_IO_CORE",

        "target_round":
            target_round,

        "history_cutoff":
            target_round - 1,

        "registry": {
            "path":
                registry["path"],

            "sha256":
                registry["sha256"],

            "challenger_count":
                len(
                    registry[
                        "challenger_ids"
                    ]
                ),

            "prefix_counts":
                registry[
                    "prefix_counts"
                ],
        },

        "production_prediction": {
            "path":
                prediction["path"],

            "sha256":
                prediction["sha256"],

            "round":
                prediction["round"],

            "set_count":
                prediction["set_count"],

            "top5_count":
                prediction["top5_count"],
        },

        "history":
            history,

        "seeds":
            seeds,

        "artifact_transaction": {
            "final_dir":
                str(
                    transaction.final_dir
                ),

            "temp_dir":
                str(
                    transaction.temp_dir
                ),

            "required_files":
                list(
                    transaction.required_files
                ),

            "overwrite":
                False,

            "partial_publish":
                False,
        },

        "authorization": {
            "challenger_generation":
                False,

            "challenger_execution":
                False,

            "database_write":
                False,

            "artifact_publish":
                False,
        },
    }


__all__ = [
    "ArtifactContractError",
    "ArtifactTransactionPlan",
    "HistoryContractError",
    "PredictionContractError",
    "RegistryContractError",
    "ShadowExecutionContractError",
    "SHADOW_END",
    "SHADOW_START",
    "build_control_plan",
    "collect_challenger_ids",
    "derive_shadow_seed",
    "inspect_history_cutoff",
    "load_frozen_prediction",
    "load_registry",
    "open_history_read_only",
    "plan_artifact_transaction",
    "sha256_file",
    "validate_challenger_id",
    "validate_shadow_round",
]
