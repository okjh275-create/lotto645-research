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

# === PHASE2_PARITY_HARNESS_V1 BEGIN ===
# Frozen by OP-147/OP-147R2.
#
# This harness does NOT execute challengers, mutate production configuration,
# register production CLI commands, or write to the database.
#
# It evaluates externally captured parity probe snapshots and fails closed.

import base64 as _phase2_base64
import json as _phase2_json
from collections.abc import Mapping as _Phase2Mapping


PARITY_CONFIRMED_BINDING_ELIGIBLE = "PARITY_CONFIRMED_BINDING_ELIGIBLE"
PARITY_REJECTED_ADVISORY_ONLY = "PARITY_REJECTED_ADVISORY_ONLY"
PARITY_UNRESOLVED_FAIL_CLOSED = "PARITY_UNRESOLVED_FAIL_CLOSED"


class ParityValidationError(ValueError):
    """Raised when a Phase-2 parity probe violates its frozen contract."""


_PHASE2_SURFACE_AUDIT_B64 = "eyJzY2hlbWFfdmVyc2lvbiI6MSwiYXJ0aWZhY3RfdHlwZSI6InBoYXNlMl9hZGFwdGVyX3N1cmZhY2VfYXVkaXQiLCJvcGVyYXRpb24iOiJPUC0xNDciLCJiYXNlbGluZV9jb21taXQiOiI1ZmJlMDlkNGY3N2JkZWZiNjhlYzYwNzE3MmFlNjA2MWM3Y2ZiY2ZlIiwiYmFzZWxpbmVfdHJlZSI6ImEzZTQ1Yzg5ZDlhY2NkNDU2MGQ4YjFmMmExNDg1OTYwNzRiNDVlNzYiLCJsZXhpY2FsX3ByaW1hcnlfYXV0aG9yaXRhdGl2ZSI6ZmFsc2UsInByb2R1Y3Rpb25fYmluZGluZ19wb2xpY3kiOiJQQVJJVFlfVEVTVF9SRVFVSVJFRCIsInN1cmZhY2VfY291bnQiOjUsInN1cmZhY2VzIjpbeyJpZCI6IkNBTkRJREFURSIsIm1vZHVsZSI6ImxycC5hZGFwdGVycy5jYW5kaWRhdGUiLCJwYXRoIjoibHJwL2FkYXB0ZXJzL2NhbmRpZGF0ZS5weSIsImNhbGxhYmxlIjoiQ2FuZGlkYXRlQWRhcHRlci5nZW5lcmF0ZV9jYW5kaWRhdGVzIiwiZmlsZV9zaGEyNTYiOiJGQ0UwNjlEMzUxMDZDN0QyODdBQUNBREUxRkYzNTBBQjgzRDZGRDZFQTgwRDNENEQ3Nzk5NUJERDI5OENERTJCIiwiZ2l0X2Jsb2IiOiIwNWQ4YmY4OTY5YTE3ZDQ2NDZiNTFlNjJhOThlNWVhNDlkMTJkNTcxIiwiZGVmaW5pdGlvbiI6eyJjbGFzc19uYW1lIjoiQ2FuZGlkYXRlQWRhcHRlciIsImNsYXNzX2xpbmUiOjI2LCJjbGFzc190ZXh0IjoiY2xhc3MgQ2FuZGlkYXRlQWRhcHRlcjoiLCJuYW1lIjoiZ2VuZXJhdGVfY2FuZGlkYXRlcyIsImRlZmluaXRpb25fbGluZSI6MTE2LCJkZWZpbml0aW9uX3RleHQiOiJkZWYgZ2VuZXJhdGVfY2FuZGlkYXRlcygifSwiY2xhc3NpZmljYXRpb24iOiJFTElHSUJMRV9BRFZJU09SWV9DQU5ESURBVEUiLCJsZXhpY2FsX21hdGNoX2F1dGhvcml0YXRpdmUiOmZhbHNlLCJkaXJlY3RfYmluZGluZ19hdXRob3JpemVkIjpmYWxzZSwicGFyaXR5X3Rlc3RfcmVxdWlyZWQiOnRydWV9LHsiaWQiOiJTQ09SSU5HIiwibW9kdWxlIjoibHJwLmVuc2VtYmxlLmFkYXB0ZXJzIiwicGF0aCI6ImxycC9lbnNlbWJsZS9hZGFwdGVycy5weSIsImNhbGxhYmxlIjoid2VpZ2h0c19mcm9tX3JhbmtpbmdzIiwiZmlsZV9zaGEyNTYiOiJDMTAwOUNBQ0Y5NTEyN0EyOUQwMjgzODNDRUNBQjc5N0ZBQzU3QzBDRTExNzBGNTI5Mzc0NDFEMUI3NURCOEUxIiwiZ2l0X2Jsb2IiOiIxZWU4MmUyNzg4YjcyMGE1ZDVmMjY1NTJiMmY2ZWNlZDI1MDA1MzE2IiwiZGVmaW5pdGlvbiI6eyJjbGFzc19uYW1lIjpudWxsLCJjbGFzc19saW5lIjpudWxsLCJjbGFzc190ZXh0IjpudWxsLCJuYW1lIjoid2VpZ2h0c19mcm9tX3JhbmtpbmdzIiwiZGVmaW5pdGlvbl9saW5lIjo0NTUsImRlZmluaXRpb25fdGV4dCI6ImRlZiB3ZWlnaHRzX2Zyb21fcmFua2luZ3MoIn0sImNsYXNzaWZpY2F0aW9uIjoiTkFSUk9XX0hFTFBFUl9BRFZJU09SWV9PTkxZIiwibGV4aWNhbF9tYXRjaF9hdXRob3JpdGF0aXZlIjpmYWxzZSwiZGlyZWN0X2JpbmRpbmdfYXV0aG9yaXplZCI6ZmFsc2UsInBhcml0eV90ZXN0X3JlcXVpcmVkIjp0cnVlfSx7ImlkIjoiRklMVEVSX0ZFQVRVUkUiLCJtb2R1bGUiOiJscnAuaW8uZHJhd3MiLCJwYXRoIjoibHJwL2lvL2RyYXdzLnB5IiwiY2FsbGFibGUiOiJsb25nX2dhcF9udW1iZXJzIiwiZmlsZV9zaGEyNTYiOiI1NDVCQzc1QzdENDY2MUREOUVENzI0RjFBQkNENTU2NUUxRTU3Q0MxOUZCRkE3MTQ1OTI1OTU3NEQ4MTg5NEU2IiwiZ2l0X2Jsb2IiOiIxMGE2YTdhYzU1MmY3YzAzZWQ0OGYwZDRlNjI5YjA1ZTU0MjY2MzEwIiwiZGVmaW5pdGlvbiI6eyJjbGFzc19uYW1lIjpudWxsLCJjbGFzc19saW5lIjpudWxsLCJjbGFzc190ZXh0IjpudWxsLCJuYW1lIjoibG9uZ19nYXBfbnVtYmVycyIsImRlZmluaXRpb25fbGluZSI6MzIxLCJkZWZpbml0aW9uX3RleHQiOiJkZWYgbG9uZ19nYXBfbnVtYmVycygifSwiY2xhc3NpZmljYXRpb24iOiJGRUFUVVJFX0hFTFBFUl9OT1RfRklMVEVSX1BJUEVMSU5FIiwibGV4aWNhbF9tYXRjaF9hdXRob3JpdGF0aXZlIjpmYWxzZSwiZGlyZWN0X2JpbmRpbmdfYXV0aG9yaXplZCI6ZmFsc2UsInBhcml0eV90ZXN0X3JlcXVpcmVkIjp0cnVlfSx7ImlkIjoiUFJBQ1RJQ0FMX1NFTEVDVE9SIiwibW9kdWxlIjoibHJwLmNsaS5kdXJhYmxlX3JlcGxheV9ldmFsdWF0aW9uIiwicGF0aCI6ImxycC9jbGkvZHVyYWJsZV9yZXBsYXlfZXZhbHVhdGlvbi5weSIsImNhbGxhYmxlIjoiX3BhcnNlX3NlbGVjdG9yIiwiZmlsZV9zaGEyNTYiOiJDNzgxNjRGN0VENEIzODBCNDQzNzU3NjA0QTUxMTNDODEzOEJDRUFEREVDMTVGQzc1M0JDQkUyQjcxMEIzNjk4IiwiZ2l0X2Jsb2IiOiJlYmQ5MWFjMmJkMGRjZDgyZGI3ZGZlMWIzNmU5MTY1ODVlNTRkOTU0IiwiZGVmaW5pdGlvbiI6eyJjbGFzc19uYW1lIjpudWxsLCJjbGFzc19saW5lIjpudWxsLCJjbGFzc190ZXh0IjpudWxsLCJuYW1lIjoiX3BhcnNlX3NlbGVjdG9yIiwiZGVmaW5pdGlvbl9saW5lIjo2NiwiZGVmaW5pdGlvbl90ZXh0IjoiZGVmIF9wYXJzZV9zZWxlY3Rvcih2YWx1ZTogc3RyKSAtXHUwMDNlIER1cmFibGVSZXBsYXlBcnRpZmFjdFNlbGVjdG9yOiJ9LCJjbGFzc2lmaWNhdGlvbiI6IlBBUlNFUl9IRUxQRVJfTk9UX1NFTEVDVE9SX1BJUEVMSU5FIiwibGV4aWNhbF9tYXRjaF9hdXRob3JpdGF0aXZlIjpmYWxzZSwiZGlyZWN0X2JpbmRpbmdfYXV0aG9yaXplZCI6ZmFsc2UsInBhcml0eV90ZXN0X3JlcXVpcmVkIjp0cnVlfSx7ImlkIjoiUEFJUl9GRUFUVVJFIiwibW9kdWxlIjoiZW5naW5lLnBhaXIiLCJwYXRoIjoiZW5naW5lL3BhaXIucHkiLCJjYWxsYWJsZSI6IlBhaXJFbmdpbmUucGFpcl9mcmVxdWVuY3kiLCJmaWxlX3NoYTI1NiI6IjRGQUJFNTQ5NzgwNjM1RDlCQzIwNDg5NTY1QzdDRDE5REJCMjJBNUJENUJBQzBEMkFDRTJBQ0JCNzQwN0Q0MEEiLCJnaXRfYmxvYiI6IjA4ZjcwMzUzMWQwZDgxZDc4N2RlZmViOTA2NTU3MjhhN2I0ZjViZDEiLCJkZWZpbml0aW9uIjp7ImNsYXNzX25hbWUiOiJQYWlyRW5naW5lIiwiY2xhc3NfbGluZSI6NywiY2xhc3NfdGV4dCI6ImNsYXNzIFBhaXJFbmdpbmU6IiwibmFtZSI6InBhaXJfZnJlcXVlbmN5IiwiZGVmaW5pdGlvbl9saW5lIjozOSwiZGVmaW5pdGlvbl90ZXh0IjoiZGVmIHBhaXJfZnJlcXVlbmN5KHNlbGYsIGxhc3Rfbj1Ob25lLCB1bnRpbF9yb3VuZD1Ob25lKToifSwiY2xhc3NpZmljYXRpb24iOiJFTElHSUJMRV9GRUFUVVJFX0hFTFBFUl9BRFZJU09SWV9PTkxZIiwibGV4aWNhbF9tYXRjaF9hdXRob3JpdGF0aXZlIjpmYWxzZSwiZGlyZWN0X2JpbmRpbmdfYXV0aG9yaXplZCI6ZmFsc2UsInBhcml0eV90ZXN0X3JlcXVpcmVkIjp0cnVlfV0sImNvbmNsdXNpb24iOiJTVEFUSUMgRElTQ09WRVJZIE9OTFk7IE5PIFBST0RVQ1RJT04gSEVMUEVSIElTIEJPVU5EIiwiY2hhbGxlbmdlcl9nZW5lcmF0aW9uIjpmYWxzZSwiY2hhbGxlbmdlcl9leGVjdXRpb24iOmZhbHNlfQ=="

_PHASE2_SURFACE_AUDIT = _phase2_json.loads(
    _phase2_base64.b64decode(
        _PHASE2_SURFACE_AUDIT_B64.encode("ascii")
    ).decode("utf-8")
)

_PHASE2_SURFACE_ORDER = tuple(
    str(item["id"])
    for item in _PHASE2_SURFACE_AUDIT["surfaces"]
)

_PHASE2_SURFACES = {
    str(item["id"]): item
    for item in _PHASE2_SURFACE_AUDIT["surfaces"]
}

_PHASE2_DIRECT_PIPELINE_REJECTIONS = frozenset(
    {
        "SCORING",
        "FILTER_FEATURE",
        "PRACTICAL_SELECTOR",
        "PAIR_FEATURE",
    }
)

_PHASE2_IDENTITY_FIELDS = (
    "module",
    "path",
    "callable",
    "file_sha256",
    "git_blob",
)


def _phase2_copy(value):
    """Return a JSON-safe defensive copy."""
    return _phase2_json.loads(
        _phase2_json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _phase2_canonical(value):
    return _phase2_json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def phase2_parity_surface_ids():
    """Return frozen parity surface IDs in contract order."""
    return tuple(_PHASE2_SURFACE_ORDER)


def phase2_parity_contract(surface_id):
    """Return a defensive copy of one frozen surface contract."""
    surface_key = str(surface_id)

    if surface_key not in _PHASE2_SURFACES:
        raise ParityValidationError(
            "unknown parity surface: "
            + surface_key
        )

    return _phase2_copy(
        _PHASE2_SURFACES[surface_key]
    )


def phase2_parity_authorization():
    """Return the fail-closed Phase-2 execution authorization."""
    return {
        "production_prediction_regeneration": False,
        "challenger_generation": False,
        "challenger_execution": False,
        "real_shadow_artifact_publish": False,
        "production_helper_binding": False,
        "database_write": False,
        "production_cli_registration": False,
    }


def _phase2_expected_identity(surface_id):
    contract = phase2_parity_contract(
        surface_id
    )

    return {
        field: contract[field]
        for field in _PHASE2_IDENTITY_FIELDS
    }


def _phase2_validate_probe_snapshot(
    surface_id,
    snapshot,
):
    if not isinstance(snapshot, _Phase2Mapping):
        raise ParityValidationError(
            "probe snapshot must be a mapping"
        )

    identity = snapshot.get("identity")

    if not isinstance(identity, _Phase2Mapping):
        raise ParityValidationError(
            "probe snapshot identity must be a mapping"
        )

    missing_identity = [
        field
        for field in _PHASE2_IDENTITY_FIELDS
        if field not in identity
    ]

    if missing_identity:
        raise ParityValidationError(
            "probe identity missing fields: "
            + ", ".join(missing_identity)
        )

    if "output_schema" not in snapshot:
        raise ParityValidationError(
            "probe snapshot missing output_schema"
        )

    if "output" not in snapshot:
        raise ParityValidationError(
            "probe snapshot missing output"
        )

    if "semantic_parity" not in snapshot:
        raise ParityValidationError(
            "probe snapshot missing semantic_parity"
        )

    return {
        "identity": {
            field: identity[field]
            for field in _PHASE2_IDENTITY_FIELDS
        },
        "output_schema": _phase2_copy(
            snapshot["output_schema"]
        ),
        "output": _phase2_copy(
            snapshot["output"]
        ),
        "semantic_parity": bool(
            snapshot["semantic_parity"]
        ),
    }


def phase2_evaluate_probe_pair(
    surface_id,
    first_snapshot,
    second_snapshot,
):
    """Evaluate two frozen-input observations for one advisory surface."""
    surface_key = str(surface_id)

    contract = phase2_parity_contract(
        surface_key
    )

    first = _phase2_validate_probe_snapshot(
        surface_key,
        first_snapshot,
    )

    second = _phase2_validate_probe_snapshot(
        surface_key,
        second_snapshot,
    )

    expected_identity = (
        _phase2_expected_identity(
            surface_key
        )
    )

    identity_match = (
        first["identity"]
        == expected_identity
        and second["identity"]
        == expected_identity
    )

    deterministic = (
        _phase2_canonical(
            first["output"]
        )
        == _phase2_canonical(
            second["output"]
        )
    )

    schema_match = (
        _phase2_canonical(
            first["output_schema"]
        )
        == _phase2_canonical(
            second["output_schema"]
        )
    )

    semantic_parity = (
        first["semantic_parity"]
        and second["semantic_parity"]
    )

    reasons = []

    if not identity_match:
        reasons.append(
            "surface_identity_mismatch"
        )

    if not deterministic:
        reasons.append(
            "non_deterministic_output"
        )

    if not schema_match:
        reasons.append(
            "output_schema_drift"
        )

    if not semantic_parity:
        reasons.append(
            "semantic_parity_not_proven"
        )

    if not identity_match:
        decision = (
            PARITY_UNRESOLVED_FAIL_CLOSED
        )
        binding_eligible = False

    elif surface_key in _PHASE2_DIRECT_PIPELINE_REJECTIONS:
        decision = (
            PARITY_REJECTED_ADVISORY_ONLY
        )
        binding_eligible = False

        reasons.append(
            "surface_is_not_complete_production_pipeline"
        )

    elif (
        deterministic
        and schema_match
        and semantic_parity
    ):
        decision = (
            PARITY_CONFIRMED_BINDING_ELIGIBLE
        )
        binding_eligible = True

    else:
        decision = (
            PARITY_UNRESOLVED_FAIL_CLOSED
        )
        binding_eligible = False

    return {
        "surface_id": surface_key,
        "classification": contract[
            "classification"
        ],
        "decision": decision,
        "binding_eligible": binding_eligible,
        "binding_authorized": False,
        "identity_match": identity_match,
        "deterministic": deterministic,
        "output_schema_match": schema_match,
        "semantic_parity": semantic_parity,
        "reasons": list(
            dict.fromkeys(reasons)
        ),
    }


def phase2_validate_probe_matrix(probes):
    """Validate an exact five-surface probe matrix."""
    if not isinstance(probes, _Phase2Mapping):
        raise ParityValidationError(
            "probe matrix must be a mapping"
        )

    actual_ids = set(
        str(key)
        for key in probes
    )

    expected_ids = set(
        _PHASE2_SURFACE_ORDER
    )

    if actual_ids != expected_ids:
        missing = sorted(
            expected_ids - actual_ids
        )
        extra = sorted(
            actual_ids - expected_ids
        )

        raise ParityValidationError(
            "probe matrix surface mismatch; "
            "missing="
            + repr(missing)
            + ", extra="
            + repr(extra)
        )

    results = []

    for surface_id in _PHASE2_SURFACE_ORDER:
        payload = probes[surface_id]

        if not isinstance(
            payload,
            _Phase2Mapping,
        ):
            raise ParityValidationError(
                "probe pair must be a mapping: "
                + surface_id
            )

        if (
            "first" not in payload
            or "second" not in payload
        ):
            raise ParityValidationError(
                "probe pair requires first and second: "
                + surface_id
            )

        results.append(
            phase2_evaluate_probe_pair(
                surface_id,
                payload["first"],
                payload["second"],
            )
        )

    return tuple(results)


def phase2_binding_decisions_fail_closed(results):
    """True only when no evaluation silently authorizes production binding."""
    for result in results:
        if not isinstance(
            result,
            _Phase2Mapping,
        ):
            raise ParityValidationError(
                "parity result must be a mapping"
            )

        if bool(
            result.get(
                "binding_authorized",
                False,
            )
        ):
            return False

    return True


# === PHASE2_PARITY_HARNESS_V1 END ===

# === PHASE3_CHALLENGER_ENGINE_V1 BEGIN ===

import hashlib as _phase3_hashlib


PHASE3_SHADOW_START_ROUND = 1243
PHASE3_SHADOW_END_ROUND = 1252

PHASE3_CHALLENGER_IDS = (
    "CG00_RANDOM_FILTERED",
    "CG01_TEMP_060",
    "CG02_TEMP_110",
    "CG03_EQUAL_WEIGHTS",
    "CG04_NO_RECENCY",
    "CG05_NO_PAIR_GRAPH",
    "PS00_CURRENT_MMR",
    "PS01_RANDOM5",
    "PS02_SCORE_TOP5",
    "PS03_MAX_UNIQUE5",
    "PS04_DIVERSITY5",
    "SC00_CURRENT",
    "SC01_RANDOM_RANK",
    "SC02_QUANTILE_DIAGNOSTIC",
    "SC03_PREQUENTIAL_PRIOR_SHADOW",
    "HF00_CURRENT",
    "HF01_SOFT_ODD_EVEN",
    "HF02_SOFT_TERMINAL",
    "HF03_SOFT_PREVIOUS_OVERLAP",
    "HF04_SOFT_SAME_DECADE",
    "HF05_SOFT_ALL4",
    "LG00_CURRENT",
    "LG01_NO_GAP_SCORE",
    "LG02_NO_GAP_HARD_RULE",
)

_PHASE3_SCORE_COMPONENTS = (
    "recency",
    "frequency",
    "gap_reversion",
    "pair_graph",
    "terminal_dispersion",
    "sum_band",
    "parity_balance",
)

_PHASE3_EQUAL_WEIGHT = 1.0 / len(
    _PHASE3_SCORE_COMPONENTS
)

_PHASE3_ALWAYS_HARD = (
    "sum",
    "low_high",
    "consecutive",
    "long_gap",
)

_PHASE3_ALL_CURRENT_HARD = (
    "sum",
    "odd_even",
    "low_high",
    "consecutive",
    "terminal",
    "previous_overlap",
    "long_gap",
    "same_decade",
)


class ChallengerPlanError(ValueError):
    """Raised when a Phase-3 challenger plan violates the frozen contract."""


def _phase3_definition(
    *,
    challenger_id,
    category,
    semantic,
    parameters,
):
    return {
        "challenger_id": challenger_id,
        "category": category,
        "semantic": semantic,
        "parameters": parameters,
        "implementation_only": True,
    }


PHASE3_CHALLENGER_DEFINITIONS = {
    "CG00_RANDOM_FILTERED": _phase3_definition(
        challenger_id="CG00_RANDOM_FILTERED",
        category="CG",
        semantic="filtered random control",
        parameters={
            "generation_mode": "random_filtered_control",
            "temperature": None,
        },
    ),
    "CG01_TEMP_060": _phase3_definition(
        challenger_id="CG01_TEMP_060",
        category="CG",
        semantic="temperature=0.60",
        parameters={
            "generation_mode": "weighted_sampling",
            "temperature": 0.60,
        },
    ),
    "CG02_TEMP_110": _phase3_definition(
        challenger_id="CG02_TEMP_110",
        category="CG",
        semantic="temperature=1.10",
        parameters={
            "generation_mode": "weighted_sampling",
            "temperature": 1.10,
        },
    ),
    "CG03_EQUAL_WEIGHTS": _phase3_definition(
        challenger_id="CG03_EQUAL_WEIGHTS",
        category="CG",
        semantic="equal component weights",
        parameters={
            "weights_mode": "equal",
            "weights": {
                name: _PHASE3_EQUAL_WEIGHT
                for name in _PHASE3_SCORE_COMPONENTS
            },
        },
    ),
    "CG04_NO_RECENCY": _phase3_definition(
        challenger_id="CG04_NO_RECENCY",
        category="CG",
        semantic="recency score component removed",
        parameters={
            "disabled_score_components": (
                "recency",
            ),
        },
    ),
    "CG05_NO_PAIR_GRAPH": _phase3_definition(
        challenger_id="CG05_NO_PAIR_GRAPH",
        category="CG",
        semantic="pair-graph score component removed",
        parameters={
            "disabled_score_components": (
                "pair_graph",
            ),
        },
    ),

    "PS00_CURRENT_MMR": _phase3_definition(
        challenger_id="PS00_CURRENT_MMR",
        category="PS",
        semantic="current practical MMR selector",
        parameters={
            "selector": "current_mmr",
            "practical_k": 5,
        },
    ),
    "PS01_RANDOM5": _phase3_definition(
        challenger_id="PS01_RANDOM5",
        category="PS",
        semantic="deterministic random five-set selector",
        parameters={
            "selector": "deterministic_random5",
            "practical_k": 5,
        },
    ),
    "PS02_SCORE_TOP5": _phase3_definition(
        challenger_id="PS02_SCORE_TOP5",
        category="PS",
        semantic="top five sets by score",
        parameters={
            "selector": "score_top5",
            "practical_k": 5,
        },
    ),
    "PS03_MAX_UNIQUE5": _phase3_definition(
        challenger_id="PS03_MAX_UNIQUE5",
        category="PS",
        semantic="maximize unique-number coverage",
        parameters={
            "selector": "max_unique5",
            "practical_k": 5,
        },
    ),
    "PS04_DIVERSITY5": _phase3_definition(
        challenger_id="PS04_DIVERSITY5",
        category="PS",
        semantic="diversity-oriented five-set selector",
        parameters={
            "selector": "diversity5",
            "practical_k": 5,
        },
    ),

    "SC00_CURRENT": _phase3_definition(
        challenger_id="SC00_CURRENT",
        category="SC",
        semantic="current score/rank behavior",
        parameters={
            "ranking_mode": "current",
        },
    ),
    "SC01_RANDOM_RANK": _phase3_definition(
        challenger_id="SC01_RANDOM_RANK",
        category="SC",
        semantic="deterministic random ranking control",
        parameters={
            "ranking_mode": "deterministic_random",
        },
    ),
    "SC02_QUANTILE_DIAGNOSTIC": _phase3_definition(
        challenger_id="SC02_QUANTILE_DIAGNOSTIC",
        category="SC",
        semantic="score quantile diagnostic ordering",
        parameters={
            "ranking_mode": "quantile_diagnostic",
        },
    ),
    "SC03_PREQUENTIAL_PRIOR_SHADOW": _phase3_definition(
        challenger_id="SC03_PREQUENTIAL_PRIOR_SHADOW",
        category="SC",
        semantic="prequential prior-only shadow",
        parameters={
            "ranking_mode": "prequential_prior_shadow",
            "minimum_prior_sample": 3,
        },
    ),

    "HF00_CURRENT": _phase3_definition(
        challenger_id="HF00_CURRENT",
        category="HF",
        semantic="current hard-filter behavior",
        parameters={
            "hard_filters": _PHASE3_ALL_CURRENT_HARD,
            "soft_filters": (),
            "soft_penalty": 0.03,
            "max_soft_violations": 0,
        },
    ),
    "HF01_SOFT_ODD_EVEN": _phase3_definition(
        challenger_id="HF01_SOFT_ODD_EVEN",
        category="HF",
        semantic="odd/even becomes soft",
        parameters={
            "hard_filters": tuple(
                item
                for item in _PHASE3_ALL_CURRENT_HARD
                if item != "odd_even"
            ),
            "soft_filters": ("odd_even",),
            "soft_penalty": 0.03,
            "max_soft_violations": 1,
        },
    ),
    "HF02_SOFT_TERMINAL": _phase3_definition(
        challenger_id="HF02_SOFT_TERMINAL",
        category="HF",
        semantic="terminal-dispersion becomes soft",
        parameters={
            "hard_filters": tuple(
                item
                for item in _PHASE3_ALL_CURRENT_HARD
                if item != "terminal"
            ),
            "soft_filters": ("terminal",),
            "soft_penalty": 0.03,
            "max_soft_violations": 1,
        },
    ),
    "HF03_SOFT_PREVIOUS_OVERLAP": _phase3_definition(
        challenger_id="HF03_SOFT_PREVIOUS_OVERLAP",
        category="HF",
        semantic="previous-round overlap becomes soft",
        parameters={
            "hard_filters": tuple(
                item
                for item in _PHASE3_ALL_CURRENT_HARD
                if item != "previous_overlap"
            ),
            "soft_filters": ("previous_overlap",),
            "soft_penalty": 0.03,
            "max_soft_violations": 1,
        },
    ),
    "HF04_SOFT_SAME_DECADE": _phase3_definition(
        challenger_id="HF04_SOFT_SAME_DECADE",
        category="HF",
        semantic="same-decade concentration becomes soft",
        parameters={
            "hard_filters": tuple(
                item
                for item in _PHASE3_ALL_CURRENT_HARD
                if item != "same_decade"
            ),
            "soft_filters": ("same_decade",),
            "soft_penalty": 0.03,
            "max_soft_violations": 1,
        },
    ),
    "HF05_SOFT_ALL4": _phase3_definition(
        challenger_id="HF05_SOFT_ALL4",
        category="HF",
        semantic="four research filters become soft",
        parameters={
            "hard_filters": _PHASE3_ALWAYS_HARD,
            "soft_filters": (
                "odd_even",
                "terminal",
                "previous_overlap",
                "same_decade",
            ),
            "soft_penalty": 0.03,
            "max_soft_violations": 2,
        },
    ),

    "LG00_CURRENT": _phase3_definition(
        challenger_id="LG00_CURRENT",
        category="LG",
        semantic="current gap score and hard rule",
        parameters={
            "gap_score_enabled": True,
            "long_gap_hard_rule_enabled": True,
        },
    ),
    "LG01_NO_GAP_SCORE": _phase3_definition(
        challenger_id="LG01_NO_GAP_SCORE",
        category="LG",
        semantic="gap score component removed; hard rule retained",
        parameters={
            "gap_score_enabled": False,
            "long_gap_hard_rule_enabled": True,
        },
    ),
    "LG02_NO_GAP_HARD_RULE": _phase3_definition(
        challenger_id="LG02_NO_GAP_HARD_RULE",
        category="LG",
        semantic="long-gap hard inclusion rule removed",
        parameters={
            "gap_score_enabled": True,
            "long_gap_hard_rule_enabled": False,
        },
    ),
}


def _phase3_copy(value):
    return _phase2_copy(value)


def phase3_challenger_ids():
    """Return the exact frozen 24-challenger registry."""
    return tuple(PHASE3_CHALLENGER_IDS)


def _phase3_validate_round(round_no):
    try:
        value = int(round_no)
    except (TypeError, ValueError) as exc:
        raise ChallengerPlanError(
            "round must be an integer"
        ) from exc

    if not (
        PHASE3_SHADOW_START_ROUND
        <= value
        <= PHASE3_SHADOW_END_ROUND
    ):
        raise ChallengerPlanError(
            "round outside frozen shadow window"
        )

    return value


def _phase3_validate_challenger_id(challenger_id):
    value = str(challenger_id)

    if value not in PHASE3_CHALLENGER_DEFINITIONS:
        raise ChallengerPlanError(
            "unknown challenger id: "
            + value
        )

    return value


def phase3_seed(round_no, challenger_id):
    """Derive the frozen deterministic OP-116 shadow seed."""
    round_value = _phase3_validate_round(
        round_no
    )

    challenger_value = (
        _phase3_validate_challenger_id(
            challenger_id
        )
    )

    token = (
        "LRP-v4.0|shadow|"
        + str(round_value)
        + "|"
        + challenger_value
    )

    digest = _phase3_hashlib.sha256(
        token.encode("utf-8")
    ).digest()

    seed = (
        int.from_bytes(
            digest[:8],
            byteorder="big",
            signed=False,
        )
        % 2147483647
    )

    return 1 if seed == 0 else seed


def phase3_challenger_definition(challenger_id):
    """Return a defensive copy of one frozen challenger definition."""
    challenger_value = (
        _phase3_validate_challenger_id(
            challenger_id
        )
    )

    return _phase3_copy(
        PHASE3_CHALLENGER_DEFINITIONS[
            challenger_value
        ]
    )


def phase3_execution_authorization():
    """Phase 3 is implementation-only and remains execution-closed."""
    return {
        "candidate_generation": False,
        "challenger_generation": False,
        "challenger_execution": False,
        "real_shadow_publish": False,
        "production_prediction_regeneration": False,
        "database_write": False,
        "production_cli_registration": False,
        "production_helper_binding_change": False,
    }


def phase3_build_challenger_plan(
    round_no,
    challenger_id,
):
    """Build a deterministic execution-disabled challenger plan."""
    round_value = _phase3_validate_round(
        round_no
    )

    challenger_value = (
        _phase3_validate_challenger_id(
            challenger_id
        )
    )

    definition = (
        phase3_challenger_definition(
            challenger_value
        )
    )

    return {
        "round": round_value,
        "challenger_id": challenger_value,
        "category": definition["category"],
        "semantic": definition["semantic"],
        "parameters": definition["parameters"],
        "seed": phase3_seed(
            round_value,
            challenger_value,
        ),
        "implementation_only": True,
        "authorization": (
            phase3_execution_authorization()
        ),
    }


def phase3_build_all_plans(round_no):
    """Build all 24 frozen plans without executing any challenger."""
    round_value = _phase3_validate_round(
        round_no
    )

    return tuple(
        phase3_build_challenger_plan(
            round_value,
            challenger_id,
        )
        for challenger_id
        in PHASE3_CHALLENGER_IDS
    )


def phase3_category_counts():
    counts = {
        "CG": 0,
        "PS": 0,
        "SC": 0,
        "HF": 0,
        "LG": 0,
    }

    for challenger_id in PHASE3_CHALLENGER_IDS:
        category = (
            PHASE3_CHALLENGER_DEFINITIONS[
                challenger_id
            ]["category"]
        )
        counts[category] += 1

    return counts


def phase3_validate_registry():
    """Fail closed if the embedded Phase-3 registry drifts."""
    ids = phase3_challenger_ids()

    if len(ids) != 24:
        raise ChallengerPlanError(
            "challenger registry count mismatch"
        )

    if len(set(ids)) != 24:
        raise ChallengerPlanError(
            "duplicate challenger ids"
        )

    if set(ids) != set(
        PHASE3_CHALLENGER_DEFINITIONS
    ):
        raise ChallengerPlanError(
            "challenger definition coverage mismatch"
        )

    expected_counts = {
        "CG": 6,
        "PS": 5,
        "SC": 4,
        "HF": 6,
        "LG": 3,
    }

    if phase3_category_counts() != expected_counts:
        raise ChallengerPlanError(
            "challenger category counts mismatch"
        )

    return True


# Validate immutable internal structure only.
# This performs no candidate generation or challenger execution.
phase3_validate_registry()

# === PHASE3_CHALLENGER_ENGINE_V1 END ===

# === PHASE4_RESEARCH_EXECUTOR_V1 BEGIN ===

import json as _phase4_json
from collections.abc import Mapping as _phase4_Mapping


PHASE4_SYNTHETIC_EXECUTION_MODE = "synthetic_fixture"


class ResearchExecutorError(ValueError):
    """Raised when the Phase-4 research executor contract is violated."""


def phase4_executor_authorization():
    """Return the frozen Phase-4 implementation-only authorization."""
    return {
        "research_executor": True,
        "synthetic_fixture_execution": True,
        "real_round_execution": False,
        "real_database_shadow_execution": False,
        "real_shadow_publish": False,
        "production_helper_binding": False,
        "production_prediction_regeneration": False,
        "database_write": False,
        "production_cli_registration": False,
    }


def _phase4_canonical_bytes(value):
    try:
        text = _phase4_json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    except (TypeError, ValueError) as exc:
        raise ResearchExecutorError(
            "value is not canonical-json serializable"
        ) from exc

    return text.encode("utf-8")


def _phase4_digest(value):
    return _phase3_hashlib.sha256(
        _phase4_canonical_bytes(value)
    ).hexdigest()


def phase4_build_request(
    round_no,
    challenger_id,
    *,
    execution_mode=PHASE4_SYNTHETIC_EXECUTION_MODE,
):
    """Build a deterministic research-only synthetic execution request."""
    if execution_mode != PHASE4_SYNTHETIC_EXECUTION_MODE:
        raise ResearchExecutorError(
            "only synthetic_fixture execution is authorized in Phase 4"
        )

    plan = phase3_build_challenger_plan(
        round_no,
        challenger_id,
    )

    identity = {
        "schema_version": 1,
        "round": plan["round"],
        "challenger_id": plan["challenger_id"],
        "seed": plan["seed"],
        "execution_mode": PHASE4_SYNTHETIC_EXECUTION_MODE,
    }

    request_id = _phase4_digest(identity)

    return {
        "schema_version": 1,
        "request_id": request_id,
        "round": plan["round"],
        "challenger_id": plan["challenger_id"],
        "seed": plan["seed"],
        "category": plan["category"],
        "execution_mode": PHASE4_SYNTHETIC_EXECUTION_MODE,
        "research_only": True,
        "synthetic_fixture": True,
        "plan": _phase3_copy(plan),
        "authorization": phase4_executor_authorization(),
    }


def phase4_build_all_requests(round_no):
    """Build all 24 synthetic requests without executing any adapter."""
    return tuple(
        phase4_build_request(
            round_no,
            challenger_id,
        )
        for challenger_id
        in phase3_challenger_ids()
    )


def _phase4_validate_request(request):
    if not isinstance(request, _phase4_Mapping):
        raise ResearchExecutorError(
            "request must be a mapping"
        )

    required = (
        "request_id",
        "round",
        "challenger_id",
        "seed",
        "execution_mode",
        "research_only",
        "synthetic_fixture",
        "plan",
    )

    missing = [
        key
        for key in required
        if key not in request
    ]

    if missing:
        raise ResearchExecutorError(
            "request missing required fields: "
            + ",".join(missing)
        )

    if (
        request["execution_mode"]
        != PHASE4_SYNTHETIC_EXECUTION_MODE
    ):
        raise ResearchExecutorError(
            "request execution mode is not authorized"
        )

    if request["research_only"] is not True:
        raise ResearchExecutorError(
            "request must be research-only"
        )

    if request["synthetic_fixture"] is not True:
        raise ResearchExecutorError(
            "request must be synthetic"
        )

    expected = phase4_build_request(
        request["round"],
        request["challenger_id"],
    )

    for key in (
        "request_id",
        "round",
        "challenger_id",
        "seed",
        "execution_mode",
    ):
        if request[key] != expected[key]:
            raise ResearchExecutorError(
                "request identity mismatch: "
                + key
            )

    if request["plan"] != expected["plan"]:
        raise ResearchExecutorError(
            "request plan mismatch"
        )

    return expected


def phase4_execute_synthetic(
    request,
    adapter,
):
    """Execute only an injected synthetic adapter.

    No production helper, database writer, CLI path, or real shadow
    publication surface is invoked here.
    """
    expected_request = _phase4_validate_request(
        request
    )

    if not callable(adapter):
        raise ResearchExecutorError(
            "synthetic adapter must be callable"
        )

    adapter_input = _phase3_copy(
        expected_request["plan"]
    )

    adapter_output = adapter(
        adapter_input
    )

    if not isinstance(
        adapter_output,
        _phase4_Mapping,
    ):
        raise ResearchExecutorError(
            "synthetic adapter output must be a mapping"
        )

    output_copy = _phase3_copy(
        dict(adapter_output)
    )

    result_identity = {
        "schema_version": 1,
        "request_id": expected_request["request_id"],
        "round": expected_request["round"],
        "challenger_id": expected_request["challenger_id"],
        "seed": expected_request["seed"],
        "execution_mode": PHASE4_SYNTHETIC_EXECUTION_MODE,
        "adapter_output": output_copy,
    }

    result_id = _phase4_digest(
        result_identity
    )

    return {
        "schema_version": 1,
        "result_id": result_id,
        "request_id": expected_request["request_id"],
        "round": expected_request["round"],
        "challenger_id": expected_request["challenger_id"],
        "seed": expected_request["seed"],
        "execution_mode": PHASE4_SYNTHETIC_EXECUTION_MODE,
        "research_only": True,
        "synthetic_fixture": True,
        "adapter_output": output_copy,
        "real_round_execution": False,
        "database_write": False,
        "real_shadow_publish": False,
        "production_helper_binding": False,
        "production_prediction_regeneration": False,
    }


def phase4_validate_result(
    request,
    result,
):
    """Validate identity preservation of one synthetic result."""
    expected_request = _phase4_validate_request(
        request
    )

    if not isinstance(result, _phase4_Mapping):
        raise ResearchExecutorError(
            "result must be a mapping"
        )

    required = (
        "result_id",
        "request_id",
        "round",
        "challenger_id",
        "seed",
        "execution_mode",
        "research_only",
        "synthetic_fixture",
        "adapter_output",
    )

    missing = [
        key
        for key in required
        if key not in result
    ]

    if missing:
        raise ResearchExecutorError(
            "result missing required fields: "
            + ",".join(missing)
        )

    for key in (
        "request_id",
        "round",
        "challenger_id",
        "seed",
        "execution_mode",
    ):
        if result[key] != expected_request[key]:
            raise ResearchExecutorError(
                "result identity mismatch: "
                + key
            )

    if result["research_only"] is not True:
        raise ResearchExecutorError(
            "result must remain research-only"
        )

    if result["synthetic_fixture"] is not True:
        raise ResearchExecutorError(
            "result must remain synthetic"
        )

    if not isinstance(
        result["adapter_output"],
        _phase4_Mapping,
    ):
        raise ResearchExecutorError(
            "result adapter_output must be a mapping"
        )

    expected_result_id = _phase4_digest({
        "schema_version": 1,
        "request_id": result["request_id"],
        "round": result["round"],
        "challenger_id": result["challenger_id"],
        "seed": result["seed"],
        "execution_mode": result["execution_mode"],
        "adapter_output": dict(
            result["adapter_output"]
        ),
    })

    if result["result_id"] != expected_result_id:
        raise ResearchExecutorError(
            "result digest mismatch"
        )

    return True


# === PHASE4_RESEARCH_EXECUTOR_V1 END ===

# === PHASE5_REAL_ROUND_ADAPTER_V1 BEGIN ===

import sqlite3 as _phase5_sqlite3
from pathlib import Path as _phase5_Path


PHASE5_REAL_ROUND_TARGET = 1243
PHASE5_HISTORY_CUTOFF_MAX = 1242
PHASE5_REAL_ROUND_PREPARED_MODE = "real_round_prepared"


class RealRoundAdapterError(ValueError):
    """Raised when the Phase-5 real-round preparation contract fails."""


def phase5_real_round_authorization():
    """Return the frozen Phase-5 preparation-only authorization."""
    return {
        "research_only": True,
        "real_round_adapter_implementation": True,
        "real_round_preparation": True,
        "synthetic_fixture_execution": True,
        "real_round_execution": False,
        "challenger_execution": False,
        "real_database_shadow_execution": False,
        "real_shadow_publish": False,
        "database_write": False,
        "production_helper_binding": False,
        "production_prediction_regeneration": False,
        "production_cli_registration": False,
        "production_learning": False,
    }


def _phase5_validate_target_round(round_no):
    if isinstance(round_no, bool) or not isinstance(round_no, int):
        raise RealRoundAdapterError(
            "target round must be integer 1243"
        )

    if round_no != PHASE5_REAL_ROUND_TARGET:
        raise RealRoundAdapterError(
            "Phase-5 adapter is locked to round 1243"
        )

    return round_no


def _phase5_database_path(db_path):
    path = _phase5_Path(db_path).expanduser().resolve()

    if not path.exists():
        raise RealRoundAdapterError(
            "database path does not exist"
        )

    if not path.is_file():
        raise RealRoundAdapterError(
            "database path is not a file"
        )

    return path


def _phase5_database_sha256(db_path):
    path = _phase5_database_path(db_path)

    digest = _phase3_hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def _phase5_open_read_only_database(db_path):
    """Open SQLite strictly read-only and force query_only."""
    path = _phase5_database_path(db_path)

    uri = path.as_uri() + "?mode=ro"

    try:
        connection = _phase5_sqlite3.connect(
            uri,
            uri=True,
        )
    except _phase5_sqlite3.Error as exc:
        raise RealRoundAdapterError(
            "unable to open database read-only"
        ) from exc

    try:
        connection.execute(
            "PRAGMA query_only = ON"
        )

        row = connection.execute(
            "PRAGMA query_only"
        ).fetchone()

        if not row or int(row[0]) != 1:
            raise RealRoundAdapterError(
                "SQLite query_only could not be enabled"
            )

        return connection

    except Exception:
        connection.close()
        raise


def phase5_load_history_snapshot(
    db_path,
    round_no=PHASE5_REAL_ROUND_TARGET,
):
    """Read the real history under a strict target-1 cutoff.

    This function performs no generation, scoring, selection,
    publication, or database mutation.
    """
    round_no = _phase5_validate_target_round(
        round_no
    )

    cutoff = round_no - 1

    if cutoff != PHASE5_HISTORY_CUTOFF_MAX:
        raise RealRoundAdapterError(
            "unexpected Phase-5 history cutoff"
        )

    database_sha256 = _phase5_database_sha256(
        db_path
    )

    connection = _phase5_open_read_only_database(
        db_path
    )

    try:
        query_only_row = connection.execute(
            "PRAGMA query_only"
        ).fetchone()

        if (
            not query_only_row
            or int(query_only_row[0]) != 1
        ):
            raise RealRoundAdapterError(
                "database is not query-only"
            )

        try:
            global_row = connection.execute(
                """
                SELECT
                    MIN(round),
                    MAX(round),
                    COUNT(*)
                FROM draw_history
                """
            ).fetchone()

            leakage_row = connection.execute(
                """
                SELECT round
                FROM draw_history
                WHERE round >= ?
                ORDER BY round ASC
                LIMIT 1
                """,
                (round_no,),
            ).fetchone()

            rows = connection.execute(
                """
                SELECT
                    round,
                    n1,
                    n2,
                    n3,
                    n4,
                    n5,
                    n6,
                    bonus
                FROM draw_history
                WHERE round <= ?
                ORDER BY round ASC
                """,
                (cutoff,),
            ).fetchall()

        except _phase5_sqlite3.Error as exc:
            raise RealRoundAdapterError(
                "draw_history query failed"
            ) from exc

    finally:
        connection.close()

    if leakage_row is not None:
        raise RealRoundAdapterError(
            "target/future leakage detected at round "
            + str(int(leakage_row[0]))
        )

    if not rows:
        raise RealRoundAdapterError(
            "history snapshot is empty"
        )

    global_min = (
        None
        if global_row[0] is None
        else int(global_row[0])
    )

    global_max = (
        None
        if global_row[1] is None
        else int(global_row[1])
    )

    global_count = int(global_row[2])

    if (
        global_max is not None
        and global_max > cutoff
    ):
        raise RealRoundAdapterError(
            "database max round exceeds target-1 cutoff"
        )

    history_rows = [
        {
            "round": int(row[0]),
            "nums": [
                int(row[1]),
                int(row[2]),
                int(row[3]),
                int(row[4]),
                int(row[5]),
                int(row[6]),
            ],
            "bonus": (
                None
                if row[7] is None
                else int(row[7])
            ),
        }
        for row in rows
    ]

    history_max = int(
        history_rows[-1]["round"]
    )

    if history_max > cutoff:
        raise RealRoundAdapterError(
            "history snapshot exceeds target-1 cutoff"
        )

    history_digest = _phase4_digest(
        history_rows
    )

    return {
        "schema_version": 1,
        "target_round": round_no,
        "history_cutoff_max_round": cutoff,
        "database_sha256": database_sha256,
        "database_read_only": True,
        "database_query_only": True,
        "target_future_leakage": False,
        "database_min_round": global_min,
        "database_max_round": global_max,
        "database_row_count": global_count,
        "history_min_round": int(
            history_rows[0]["round"]
        ),
        "history_max_round": history_max,
        "history_row_count": len(
            history_rows
        ),
        "history_digest": history_digest,
        "history_rows": history_rows,
    }


def _phase5_prepare_request_from_snapshot(
    snapshot,
    challenger_id,
):
    if not isinstance(
        snapshot,
        _phase4_Mapping,
    ):
        raise RealRoundAdapterError(
            "history snapshot must be a mapping"
        )

    round_no = _phase5_validate_target_round(
        snapshot.get("target_round")
    )

    if (
        snapshot.get("history_cutoff_max_round")
        != PHASE5_HISTORY_CUTOFF_MAX
    ):
        raise RealRoundAdapterError(
            "history cutoff mismatch"
        )

    if snapshot.get("database_read_only") is not True:
        raise RealRoundAdapterError(
            "database must be read-only"
        )

    if snapshot.get("database_query_only") is not True:
        raise RealRoundAdapterError(
            "database must be query-only"
        )

    if snapshot.get("target_future_leakage") is not False:
        raise RealRoundAdapterError(
            "leakage state is not clean"
        )

    if (
        int(snapshot.get("history_max_round"))
        > PHASE5_HISTORY_CUTOFF_MAX
    ):
        raise RealRoundAdapterError(
            "history max exceeds cutoff"
        )

    rows = snapshot.get("history_rows")

    if not isinstance(rows, list) or not rows:
        raise RealRoundAdapterError(
            "history rows missing"
        )

    if (
        snapshot.get("history_row_count")
        != len(rows)
    ):
        raise RealRoundAdapterError(
            "history row count mismatch"
        )

    if (
        snapshot.get("history_digest")
        != _phase4_digest(rows)
    ):
        raise RealRoundAdapterError(
            "history digest mismatch"
        )

    plan = phase3_build_challenger_plan(
        round_no,
        challenger_id,
    )

    identity = {
        "schema_version": 1,
        "round": round_no,
        "challenger_id": plan["challenger_id"],
        "seed": plan["seed"],
        "execution_mode": (
            PHASE5_REAL_ROUND_PREPARED_MODE
        ),
        "database_sha256": (
            snapshot["database_sha256"]
        ),
        "history_digest": (
            snapshot["history_digest"]
        ),
    }

    request_id = _phase4_digest(
        identity
    )

    return {
        "schema_version": 1,
        "request_id": request_id,
        "round": round_no,
        "challenger_id": plan["challenger_id"],
        "seed": plan["seed"],
        "category": plan["category"],
        "execution_mode": (
            PHASE5_REAL_ROUND_PREPARED_MODE
        ),
        "research_only": True,
        "prepared_only": True,
        "execution_authorized": False,
        "plan": _phase3_copy(plan),
        "history_snapshot": _phase3_copy(
            dict(snapshot)
        ),
        "authorization": (
            phase5_real_round_authorization()
        ),
    }


def phase5_prepare_real_round_request(
    db_path,
    challenger_id,
    *,
    round_no=PHASE5_REAL_ROUND_TARGET,
):
    """Prepare one real-data request without executing it."""
    snapshot = phase5_load_history_snapshot(
        db_path,
        round_no=round_no,
    )

    return _phase5_prepare_request_from_snapshot(
        snapshot,
        challenger_id,
    )


def phase5_prepare_all_real_round_requests(
    db_path,
    *,
    round_no=PHASE5_REAL_ROUND_TARGET,
):
    """Prepare all 24 frozen challenger requests.

    The database snapshot is loaded once and defensively copied
    into each request. No challenger is actually executed.
    """
    snapshot = phase5_load_history_snapshot(
        db_path,
        round_no=round_no,
    )

    return tuple(
        _phase5_prepare_request_from_snapshot(
            snapshot,
            challenger_id,
        )
        for challenger_id
        in phase3_challenger_ids()
    )


def phase5_validate_prepared_request(
    request,
):
    """Validate one prepared real-round request offline."""
    if not isinstance(
        request,
        _phase4_Mapping,
    ):
        raise RealRoundAdapterError(
            "prepared request must be a mapping"
        )

    required = (
        "request_id",
        "round",
        "challenger_id",
        "seed",
        "execution_mode",
        "research_only",
        "prepared_only",
        "execution_authorized",
        "plan",
        "history_snapshot",
    )

    missing = [
        key
        for key in required
        if key not in request
    ]

    if missing:
        raise RealRoundAdapterError(
            "prepared request missing fields: "
            + ",".join(missing)
        )

    round_no = _phase5_validate_target_round(
        request["round"]
    )

    if (
        request["execution_mode"]
        != PHASE5_REAL_ROUND_PREPARED_MODE
    ):
        raise RealRoundAdapterError(
            "prepared execution mode mismatch"
        )

    if request["research_only"] is not True:
        raise RealRoundAdapterError(
            "request must remain research-only"
        )

    if request["prepared_only"] is not True:
        raise RealRoundAdapterError(
            "request must remain preparation-only"
        )

    if request["execution_authorized"] is not False:
        raise RealRoundAdapterError(
            "real execution is not authorized"
        )

    snapshot = request["history_snapshot"]

    if not isinstance(
        snapshot,
        _phase4_Mapping,
    ):
        raise RealRoundAdapterError(
            "history snapshot must be a mapping"
        )

    rows = snapshot.get("history_rows")

    if not isinstance(rows, list) or not rows:
        raise RealRoundAdapterError(
            "history rows missing"
        )

    if (
        snapshot.get("history_cutoff_max_round")
        != PHASE5_HISTORY_CUTOFF_MAX
    ):
        raise RealRoundAdapterError(
            "history cutoff mismatch"
        )

    if (
        int(snapshot.get("history_max_round"))
        > PHASE5_HISTORY_CUTOFF_MAX
    ):
        raise RealRoundAdapterError(
            "history exceeds cutoff"
        )

    if snapshot.get("database_read_only") is not True:
        raise RealRoundAdapterError(
            "database read-only flag mismatch"
        )

    if snapshot.get("database_query_only") is not True:
        raise RealRoundAdapterError(
            "database query-only flag mismatch"
        )

    if snapshot.get("target_future_leakage") is not False:
        raise RealRoundAdapterError(
            "leakage flag mismatch"
        )

    if (
        snapshot.get("history_row_count")
        != len(rows)
    ):
        raise RealRoundAdapterError(
            "history row count mismatch"
        )

    if (
        snapshot.get("history_digest")
        != _phase4_digest(rows)
    ):
        raise RealRoundAdapterError(
            "history digest mismatch"
        )

    plan = phase3_build_challenger_plan(
        round_no,
        request["challenger_id"],
    )

    if request["seed"] != plan["seed"]:
        raise RealRoundAdapterError(
            "prepared request seed mismatch"
        )

    if request["plan"] != plan:
        raise RealRoundAdapterError(
            "prepared challenger plan mismatch"
        )

    identity = {
        "schema_version": 1,
        "round": round_no,
        "challenger_id": plan["challenger_id"],
        "seed": plan["seed"],
        "execution_mode": (
            PHASE5_REAL_ROUND_PREPARED_MODE
        ),
        "database_sha256": (
            snapshot["database_sha256"]
        ),
        "history_digest": (
            snapshot["history_digest"]
        ),
    }

    expected_request_id = _phase4_digest(
        identity
    )

    if request["request_id"] != expected_request_id:
        raise RealRoundAdapterError(
            "prepared request digest mismatch"
        )

    return True


def phase5_execute_real_round(
    request,
    adapter=None,
):
    """Fail closed until a separate exact round authorization exists."""
    phase5_validate_prepared_request(
        request
    )

    raise RealRoundAdapterError(
        "round-1243 real challenger execution "
        "is not authorized in Phase 5 adapter implementation"
    )


# === PHASE5_REAL_ROUND_ADAPTER_V1 END ===

# === PHASE6_RESEARCH_EXECUTION_ENGINE_V1 BEGIN ===

import math as _phase6_math
import random as _phase6_random
from collections import Counter as _phase6_Counter
from collections import defaultdict as _phase6_defaultdict


PHASE6_CANDIDATE_COUNT = 10000
PHASE6_TOP_K = 10
PHASE6_PRACTICAL_K = 5
PHASE6_JACCARD_MAX = 0.33
PHASE6_MAX_OVERLAP_BETWEEN_SETS = 3
PHASE6_SOFT_PENALTY = 0.03
PHASE6_MAX_ATTEMPTS = 500000

PHASE6_CURRENT_WEIGHTS = {
    "recency": 0.35,
    "frequency": 0.20,
    "gap_reversion": 0.15,
    "pair_graph": 0.10,
    "terminal_dispersion": 0.08,
    "sum_band": 0.07,
    "parity_balance": 0.05,
}

PHASE6_SCORE_COMPONENTS = (
    "recency",
    "frequency",
    "gap_reversion",
    "pair_graph",
    "terminal_dispersion",
    "sum_band",
    "parity_balance",
)


class ResearchExecutionEngineError(ValueError):
    pass


def _phase6_renormalize_weights(weights):
    total = sum(
        float(value)
        for value in weights.values()
    )

    if total <= 0:
        raise ResearchExecutionEngineError(
            "weights must sum to a positive value"
        )

    return {
        key: float(value) / total
        for key, value in weights.items()
    }


def phase6_execution_engine_authorization():
    return {
        "research_only": True,
        "engine_implementation": True,
        "synthetic_test_execution": True,
        "prepared_request_test_execution": True,
        "actual_round1243_execution": False,
        "actual_challenger_execution": False,
        "real_shadow_publish": False,
        "database_access_inside_engine": False,
        "database_write": False,
        "production_helper_binding": False,
        "production_cli_mutation": False,
        "production_learning": False,
    }


def phase6_challenger_config(challenger_id):
    if challenger_id not in phase3_challenger_ids():
        raise ResearchExecutionEngineError(
            "unknown challenger id"
        )

    config = {
        "challenger_id": challenger_id,
        "candidate_count": PHASE6_CANDIDATE_COUNT,
        "top_k": PHASE6_TOP_K,
        "practical_k": PHASE6_PRACTICAL_K,
        "generation": {
            "mode": "weighted",
            "temperature": 0.85,
            "weight_profile": "current",
        },
        "scoring": {
            "mode": "current",
            "weights": dict(PHASE6_CURRENT_WEIGHTS),
            "minimum_history_rows": 1,
        },
        "filters": {
            "soft_rules": (),
            "soft_penalty": PHASE6_SOFT_PENALTY,
            "max_soft_violations": 0,
            "long_gap_hard_rule": True,
        },
        "practical_selector": {
            "mode": "current_mmr",
        },
    }

    if challenger_id == "CG00_RANDOM_FILTERED":
        config["generation"] = {
            "mode": "uniform_filtered",
            "temperature": 1.0,
            "weight_profile": "uniform",
        }

    elif challenger_id == "CG01_TEMP_060":
        config["generation"]["temperature"] = 0.60

    elif challenger_id == "CG02_TEMP_110":
        config["generation"]["temperature"] = 1.10

    elif challenger_id == "CG03_EQUAL_WEIGHTS":
        config["generation"]["weight_profile"] = "equal_components"

    elif challenger_id == "CG04_NO_RECENCY":
        config["generation"]["weight_profile"] = "no_recency"

    elif challenger_id == "CG05_NO_PAIR_GRAPH":
        config["generation"]["weight_profile"] = "no_pair_graph"

    if challenger_id == "PS01_RANDOM5":
        config["practical_selector"]["mode"] = "deterministic_random5"

    elif challenger_id == "PS02_SCORE_TOP5":
        config["practical_selector"]["mode"] = "score_top5"

    elif challenger_id == "PS03_MAX_UNIQUE5":
        config["practical_selector"]["mode"] = "max_unique5"

    elif challenger_id == "PS04_DIVERSITY5":
        config["practical_selector"]["mode"] = "diversity5"

    if challenger_id == "SC01_RANDOM_RANK":
        config["scoring"]["mode"] = "deterministic_random_rank"

    elif challenger_id == "SC02_QUANTILE_DIAGNOSTIC":
        config["scoring"]["mode"] = "quantile_diagnostic"

    elif challenger_id == "SC03_PREQUENTIAL_PRIOR_SHADOW":
        config["scoring"]["mode"] = "prequential_prior_shadow"
        config["scoring"]["minimum_history_rows"] = 3

    soft_map = {
        "HF01_SOFT_ODD_EVEN": ("odd_even",),
        "HF02_SOFT_TERMINAL": ("terminal",),
        "HF03_SOFT_PREVIOUS_OVERLAP": ("previous_overlap",),
        "HF04_SOFT_SAME_DECADE": ("same_decade",),
        "HF05_SOFT_ALL4": (
            "odd_even",
            "terminal",
            "previous_overlap",
            "same_decade",
        ),
    }

    if challenger_id in soft_map:
        config["filters"]["soft_rules"] = soft_map[challenger_id]

        config["filters"]["max_soft_violations"] = (
            2
            if challenger_id == "HF05_SOFT_ALL4"
            else 1
        )

    if challenger_id == "LG01_NO_GAP_SCORE":
        weights = dict(
            config["scoring"]["weights"]
        )
        weights["gap_reversion"] = 0.0

        config["scoring"]["weights"] = (
            _phase6_renormalize_weights(
                weights
            )
        )

    elif challenger_id == "LG02_NO_GAP_HARD_RULE":
        config["filters"]["long_gap_hard_rule"] = False

    return _phase3_copy(config)


def phase6_execution_matrix():
    return tuple(
        phase6_challenger_config(
            challenger_id
        )
        for challenger_id
        in phase3_challenger_ids()
    )


def _phase6_validate_prepared_request(request):
    try:
        phase5_validate_prepared_request(
            request
        )
    except Exception as exc:
        raise ResearchExecutionEngineError(
            "prepared request validation failed"
        ) from exc

    if request["round"] != 1243:
        raise ResearchExecutionEngineError(
            "engine is locked to round 1243"
        )

    if request["execution_authorized"] is not False:
        raise ResearchExecutionEngineError(
            "prepared request unexpectedly authorizes execution"
        )

    snapshot = request[
        "history_snapshot"
    ]

    if (
        snapshot["history_cutoff_max_round"]
        != 1242
    ):
        raise ResearchExecutionEngineError(
            "history cutoff must remain 1242"
        )

    if snapshot["history_max_round"] > 1242:
        raise ResearchExecutionEngineError(
            "target/future leakage detected"
        )

    return True


def _phase6_normalize_number_map(values):
    numbers = range(1, 46)

    minimum = min(
        float(values.get(n, 0.0))
        for n in numbers
    )

    maximum = max(
        float(values.get(n, 0.0))
        for n in numbers
    )

    if _phase6_math.isclose(
        maximum,
        minimum,
    ):
        return {
            n: 0.5
            for n in numbers
        }

    span = maximum - minimum

    return {
        n: (
            float(values.get(n, 0.0))
            - minimum
        ) / span
        for n in numbers
    }


def _phase6_history_statistics(history_rows):
    if not history_rows:
        raise ResearchExecutionEngineError(
            "history rows are required"
        )

    numbers = range(1, 46)

    recent_rows = history_rows[-10:]
    mid_rows = history_rows[-20:]
    long_rows = history_rows[-50:]

    recency_raw = {
        n: 0.0
        for n in numbers
    }

    for age, row in enumerate(
        reversed(recent_rows)
    ):
        weight = 0.85 ** age

        for number in row["nums"]:
            recency_raw[int(number)] += weight

    frequency_raw = {
        n: 0.0
        for n in numbers
    }

    for window_rows in (
        recent_rows,
        mid_rows,
        long_rows,
    ):
        denominator = max(
            1,
            len(window_rows),
        )

        counts = _phase6_Counter(
            int(number)
            for row in window_rows
            for number in row["nums"]
        )

        for n in numbers:
            frequency_raw[n] += (
                counts[n] / denominator
            )

    gap_raw = {}

    for n in numbers:
        found_age = None

        for age, row in enumerate(
            reversed(history_rows)
        ):
            if n in row["nums"]:
                found_age = age
                break

        gap_raw[n] = (
            len(history_rows) + 5
            if found_age is None
            else found_age
        )

    pair_raw = _phase6_defaultdict(float)

    pair_degree_raw = {
        n: 0.0
        for n in numbers
    }

    for age, row in enumerate(
        reversed(long_rows)
    ):
        weight = 0.96 ** age

        values = sorted(
            int(n)
            for n in row["nums"]
        )

        for left_index in range(
            len(values)
        ):
            for right_index in range(
                left_index + 1,
                len(values),
            ):
                pair = (
                    values[left_index],
                    values[right_index],
                )

                pair_raw[pair] += weight
                pair_degree_raw[pair[0]] += weight
                pair_degree_raw[pair[1]] += weight

    recent5_seen = {
        int(number)
        for row in history_rows[-5:]
        for number in row["nums"]
    }

    long_gap_numbers = tuple(
        n
        for n in numbers
        if n not in recent5_seen
    )

    previous_numbers = tuple(
        sorted(
            int(number)
            for number
            in history_rows[-1]["nums"]
        )
    )

    historical_sums = [
        sum(
            int(n)
            for n in row["nums"]
        )
        for row in long_rows
    ]

    target_sum = (
        sum(historical_sums)
        / len(historical_sums)
        if historical_sums
        else 138.0
    )

    pair_max = max(
        [0.0]
        + list(pair_raw.values())
    )

    return {
        "recency":
            _phase6_normalize_number_map(
                recency_raw
            ),

        "frequency":
            _phase6_normalize_number_map(
                frequency_raw
            ),

        "gap_reversion":
            _phase6_normalize_number_map(
                gap_raw
            ),

        "pair_graph":
            _phase6_normalize_number_map(
                pair_degree_raw
            ),

        "pair_raw":
            dict(pair_raw),

        "pair_max":
            pair_max,

        "long_gap_numbers":
            long_gap_numbers,

        "previous_numbers":
            previous_numbers,

        "target_sum":
            target_sum,

        "history_row_count":
            len(history_rows),
    }


def _phase6_sampling_weights(
    stats,
    config,
):
    profile = (
        config["generation"][
            "weight_profile"
        ]
    )

    numbers = range(1, 46)

    if profile == "uniform":
        return {
            n: 1.0
            for n in numbers
        }

    base = {
        "recency": 0.35,
        "frequency": 0.20,
        "gap_reversion": 0.15,
        "pair_graph": 0.10,
    }

    if profile == "equal_components":
        base = {
            "recency": 0.25,
            "frequency": 0.25,
            "gap_reversion": 0.25,
            "pair_graph": 0.25,
        }

    elif profile == "no_recency":
        base["recency"] = 0.0

    elif profile == "no_pair_graph":
        base["pair_graph"] = 0.0

    base = _phase6_renormalize_weights(
        base
    )

    raw = {}

    for n in numbers:
        value = (
            base["recency"]
            * stats["recency"][n]

            + base["frequency"]
            * stats["frequency"][n]

            + base["gap_reversion"]
            * stats["gap_reversion"][n]

            + base["pair_graph"]
            * stats["pair_graph"][n]
        )

        raw[n] = (
            0.25
            + (
                0.75
                * max(
                    0.0,
                    value,
                )
            )
        )

    temperature = float(
        config["generation"][
            "temperature"
        ]
    )

    if temperature <= 0:
        raise ResearchExecutionEngineError(
            "temperature must be positive"
        )

    inverse_temperature = (
        1.0 / temperature
    )

    return {
        n: (
            max(
                raw[n],
                1e-12,
            )
            ** inverse_temperature
        )
        for n in numbers
    }


def _phase6_weighted_sample_without_replacement(
    rng,
    weights,
    k,
):
    pool = list(
        range(1, 46)
    )

    selected = []

    for _ in range(k):
        total = sum(
            float(weights[n])
            for n in pool
        )

        if total <= 0:
            raise ResearchExecutionEngineError(
                "sampling weights invalid"
            )

        target = (
            rng.random()
            * total
        )

        cumulative = 0.0
        chosen = pool[-1]

        for n in pool:
            cumulative += float(
                weights[n]
            )

            if target <= cumulative:
                chosen = n
                break

        selected.append(
            chosen
        )

        pool.remove(
            chosen
        )

    return tuple(
        sorted(selected)
    )


def _phase6_max_consecutive_run(
    numbers,
):
    maximum = 1
    current = 1

    for previous, current_number in zip(
        numbers,
        numbers[1:],
    ):
        if (
            current_number
            == previous + 1
        ):
            current += 1

            maximum = max(
                maximum,
                current,
            )
        else:
            current = 1

    return maximum


def _phase6_features(
    numbers,
    stats,
):
    numbers = tuple(
        sorted(
            int(n)
            for n in numbers
        )
    )

    odd = sum(
        1
        for n in numbers
        if n % 2 == 1
    )

    low = sum(
        1
        for n in numbers
        if n <= 22
    )

    endings = tuple(
        n % 10
        for n in numbers
    )

    ending_counts = _phase6_Counter(
            endings
        )

    decade_counts = _phase6_Counter(
            (n - 1) // 10
            for n in numbers
        )

    previous = set(
        stats["previous_numbers"]
    )

    long_gap = set(
        stats["long_gap_numbers"]
    )

    consecutive_pairs = tuple(
        (left, right)
        for left, right in zip(
            numbers,
            numbers[1:],
        )
        if right == left + 1
    )

    return {
        "sum":
            sum(numbers),

        "odd_count":
            odd,

        "even_count":
            6 - odd,

        "odd_even":
            f"{odd}:{6 - odd}",

        "low_count":
            low,

        "high_count":
            6 - low,

        "low_high":
            f"{low}:{6 - low}",

        "max_consecutive_run":
            _phase6_max_consecutive_run(
                numbers
            ),

        "consecutive_pairs":
            consecutive_pairs,

        "end_digits":
            endings,

        "max_same_ending":
            max(
                ending_counts.values()
            ),

        "previous_overlap":
            len(
                previous.intersection(
                    numbers
                )
            ),

        "long_gap_count":
            len(
                long_gap.intersection(
                    numbers
                )
            ),

        "max_same_decade":
            max(
                decade_counts.values()
            ),
    }


def _phase6_filter_candidate(
    numbers,
    stats,
    config,
):
    features = _phase6_features(
            numbers,
            stats,
        )

    violations = []

    if not (
        90
        <= features["sum"]
        <= 200
    ):
        violations.append(
            "sum"
        )

    if not (
        2
        <= features["odd_count"]
        <= 4
    ):
        violations.append(
            "odd_even"
        )

    if (
        features["low_count"] < 1
        or features["high_count"] < 1
    ):
        violations.append(
            "low_high"
        )

    if (
        features[
            "max_consecutive_run"
        ] > 2
    ):
        violations.append(
            "consecutive"
        )

    if (
        features[
            "max_same_ending"
        ] > 2
    ):
        violations.append(
            "terminal"
        )

    if (
        features[
            "previous_overlap"
        ] > 1
    ):
        violations.append(
            "previous_overlap"
        )

    if (
        config["filters"][
            "long_gap_hard_rule"
        ]
        and
        features[
            "long_gap_count"
        ] < 1
    ):
        violations.append(
            "long_gap"
        )

    if (
        features[
            "max_same_decade"
        ] > 3
    ):
        violations.append(
            "same_decade"
        )

    soft_rules = set(
        config["filters"][
            "soft_rules"
        ]
    )

    always_hard = {
        "sum",
        "low_high",
        "consecutive",
        "long_gap",
    }

    hard_violations = [
        item
        for item in violations
        if (
            item in always_hard
            or item not in soft_rules
        )
    ]

    soft_violations = [
        item
        for item in violations
        if item in soft_rules
    ]

    if hard_violations:
        return None

    if (
        len(soft_violations)
        >
        int(
            config["filters"][
                "max_soft_violations"
            ]
        )
    ):
        return None

    return {
        "numbers":
            numbers,

        "features":
            features,

        "risk_flags":
            tuple(
                soft_violations
            ),

        "soft_penalty":
            (
                float(
                    config["filters"][
                        "soft_penalty"
                    ]
                )
                * len(
                    soft_violations
                )
            ),
    }


def _phase6_pair_affinity(
    numbers,
    stats,
):
    values = tuple(
        sorted(numbers)
    )

    pair_values = []

    for left_index in range(
        len(values)
    ):
        for right_index in range(
            left_index + 1,
            len(values),
        ):
            pair = (
                values[left_index],
                values[right_index],
            )

            pair_values.append(
                float(
                    stats[
                        "pair_raw"
                    ].get(
                        pair,
                        0.0,
                    )
                )
            )

    if (
        not pair_values
        or stats["pair_max"] <= 0
    ):
        return 0.0

    return (
        sum(pair_values)
        / len(pair_values)
        / stats["pair_max"]
    )


def _phase6_component_scores(
    candidate,
    stats,
):
    numbers = candidate["numbers"]

    features = candidate["features"]

    recency = (
        sum(
            stats["recency"][n]
            for n in numbers
        )
        / 6.0
    )

    frequency = (
        sum(
            stats["frequency"][n]
            for n in numbers
        )
        / 6.0
    )

    gap = (
        sum(
            stats["gap_reversion"][n]
            for n in numbers
        )
        / 6.0
    )

    pair = _phase6_pair_affinity(
            numbers,
            stats,
        )

    terminal = (
        len(
            set(
                features[
                    "end_digits"
                ]
            )
        )
        / 6.0
    )

    sum_fit = max(
        0.0,
        (
            1.0
            -
            (
                abs(
                    features["sum"]
                    - stats[
                        "target_sum"
                    ]
                )
                / 110.0
            )
        ),
    )

    odd_count = features["odd_count"]

    if odd_count == 3:
        parity = 1.0

    elif odd_count in (
        2,
        4,
    ):
        parity = 0.85

    else:
        parity = 0.0

    return {
        "recency":
            recency,

        "frequency":
            frequency,

        "gap_reversion":
            gap,

        "pair_graph":
            pair,

        "terminal_dispersion":
            terminal,

        "sum_band":
            sum_fit,

        "parity_balance":
            parity,
    }


def _phase6_hash_unit_interval(
    seed,
    numbers,
):
    token = (
        str(seed)
        + "|"
        + ",".join(
            str(n)
            for n in numbers
        )
    )

    digest = _phase3_hashlib.sha256(
            token.encode(
                "utf-8"
            )
        ).digest()

    value = int.from_bytes(
            digest[:8],
            "big",
            signed=False,
        )

    return (
        value
        / float(2 ** 64)
    )


def _phase6_score_candidate(
    candidate,
    stats,
    config,
    seed,
):
    components = _phase6_component_scores(
            candidate,
            stats,
        )

    weights = config["scoring"][
            "weights"
        ]

    base_score = sum(
        float(weights[key])
        * float(
            components[key]
        )
        for key
        in PHASE6_SCORE_COMPONENTS
    )

    mode = config["scoring"][
            "mode"
        ]

    if mode == "deterministic_random_rank":
        raw_score = _phase6_hash_unit_interval(
                seed,
                candidate[
                    "numbers"
                ],
            )

    elif mode == "quantile_diagnostic":
        raw_score = (
            round(
                base_score
                * 10.0
            )
            / 10.0
        )

    elif mode == "prequential_prior_shadow":
        if (
            stats[
                "history_row_count"
            ]
            <
            int(
                config[
                    "scoring"
                ][
                    "minimum_history_rows"
                ]
            )
        ):
            raise ResearchExecutionEngineError(
                "prequential prior requires at least 3 history rows"
            )

        prior = components[
                "frequency"
            ]

        raw_score = (
            0.80
            * base_score
            +
            0.20
            * prior
        )

    else:
        raw_score = base_score

    raw_score -= float(
        candidate[
            "soft_penalty"
        ]
    )

    result = dict(
        candidate
    )

    result[
        "components"
    ] = components

    result[
        "raw_score"
    ] = raw_score

    return result


def _phase6_minmax_scores(
    candidates,
):
    values = [
        float(
            item["raw_score"]
        )
        for item in candidates
    ]

    minimum = min(values)

    maximum = max(values)

    if _phase6_math.isclose(
        minimum,
        maximum,
    ):
        normalized = [
            0.5
            for _ in values
        ]

    else:
        span = (
            maximum
            - minimum
        )

        normalized = [
            (
                value
                - minimum
            )
            / span
            for value in values
        ]

    result = []

    for item, score in zip(
        candidates,
        normalized,
    ):
        enriched = dict(
            item
        )

        enriched[
            "score"
        ] = float(score)

        result.append(
            enriched
        )

    return result


def _phase6_jaccard(
    left,
    right,
):
    left_set = set(left)

    right_set = set(right)

    union = left_set.union(
            right_set
        )

    if not union:
        return 0.0

    return (
        len(
            left_set.intersection(
                right_set
            )
        )
        / len(union)
    )


def _phase6_pairwise_allowed(
    left,
    right,
):
    overlap = len(
        set(left).intersection(
            right
        )
    )

    jaccard = _phase6_jaccard(
            left,
            right,
        )

    return (
        overlap
        <= PHASE6_MAX_OVERLAP_BETWEEN_SETS

        and

        jaccard
        <= PHASE6_JACCARD_MAX
    )


def _phase6_select_top10(
    scored,
):
    ordered = sorted(
        scored,
        key=lambda item: (
            -float(
                item["score"]
            ),
            tuple(
                item["numbers"]
            ),
        ),
    )

    selected = []

    candidate_pool = ordered[:4000]

    while (
        len(selected)
        < PHASE6_TOP_K
    ):
        best = None
        best_value = None

        for item in candidate_pool:
            if item in selected:
                continue

            if any(
                not _phase6_pairwise_allowed(
                    item["numbers"],
                    existing[
                        "numbers"
                    ],
                )
                for existing
                in selected
            ):
                continue

            max_similarity = max(
                [
                    _phase6_jaccard(
                        item[
                            "numbers"
                        ],
                        existing[
                            "numbers"
                        ],
                    )
                    for existing
                    in selected
                ]
                or [0.0]
            )

            mmr_value = (
                float(
                    item[
                        "score"
                    ]
                )
                -
                (
                    0.15
                    * max_similarity
                )
            )

            tie_key = (
                mmr_value,
                float(
                    item[
                        "score"
                    ]
                ),
                tuple(
                    -n
                    for n
                    in item[
                        "numbers"
                    ]
                ),
            )

            if (
                best_value is None
                or tie_key
                > best_value
            ):
                best = item
                best_value = tie_key

        if best is None:
            raise ResearchExecutionEngineError(
                "unable to select 10 diverse sets"
            )

        selected.append(
            best
        )

    return selected


def _phase6_practical_indices(
    top10,
    config,
    seed,
):
    mode = config[
            "practical_selector"
        ][
            "mode"
        ]

    if mode == "deterministic_random5":
        rng = _phase6_random.Random(
                int(seed)
                ^ 0x5A17
            )

        return tuple(
            sorted(
                rng.sample(
                    range(
                        PHASE6_TOP_K
                    ),
                    PHASE6_PRACTICAL_K,
                )
            )
        )

    if mode == "score_top5":
        return tuple(
            index
            for index, _item
            in sorted(
                enumerate(
                    top10
                ),
                key=lambda pair: (
                    -float(
                        pair[1][
                            "score"
                        ]
                    ),
                    pair[0],
                ),
            )[
                :PHASE6_PRACTICAL_K
            ]
        )

    if mode == "max_unique5":
        chosen = []
        used_numbers = set()

        while (
            len(chosen)
            < PHASE6_PRACTICAL_K
        ):
            choices = []

            for index, item in enumerate(
                top10
            ):
                if index in chosen:
                    continue

                new_count = len(
                    set(
                        item[
                            "numbers"
                        ]
                    )
                    - used_numbers
                )

                choices.append(
                    (
                        new_count,
                        float(
                            item[
                                "score"
                            ]
                        ),
                        -index,
                        index,
                    )
                )

            (
                _new_count,
                _score,
                _neg_index,
                index,
            ) = max(
                choices
            )

            chosen.append(
                index
            )

            used_numbers.update(
                top10[index][
                    "numbers"
                ]
            )

        return tuple(
            sorted(chosen)
        )

    if mode == "diversity5":
        chosen = [0]

        while (
            len(chosen)
            < PHASE6_PRACTICAL_K
        ):
            choices = []

            for index, item in enumerate(
                top10
            ):
                if index in chosen:
                    continue

                max_similarity = max(
                    _phase6_jaccard(
                        item[
                            "numbers"
                        ],
                        top10[
                            chosen_index
                        ][
                            "numbers"
                        ],
                    )
                    for chosen_index
                    in chosen
                )

                choices.append(
                    (
                        -max_similarity,
                        float(
                            item[
                                "score"
                            ]
                        ),
                        -index,
                        index,
                    )
                )

            (
                _neg_similarity,
                _score,
                _neg_index,
                index,
            ) = max(
                choices
            )

            chosen.append(
                index
            )

        return tuple(
            sorted(chosen)
        )

    return tuple(
        range(
            PHASE6_PRACTICAL_K
        )
    )


def _phase6_generate_candidates(
    request,
    stats,
    config,
):
    rng = _phase6_random.Random(
            int(
                request[
                    "seed"
                ]
            )
        )

    weights = _phase6_sampling_weights(
            stats,
            config,
        )

    accepted = {}
    attempts = 0

    while (
        len(accepted)
        < PHASE6_CANDIDATE_COUNT
        and
        attempts
        < PHASE6_MAX_ATTEMPTS
    ):
        attempts += 1

        if (
            config[
                "generation"
            ][
                "mode"
            ]
            == "uniform_filtered"
        ):
            numbers = tuple(
                sorted(
                    rng.sample(
                        range(1, 46),
                        6,
                    )
                )
            )

        else:
            numbers = (
                _phase6_weighted_sample_without_replacement(
                    rng,
                    weights,
                    6,
                )
            )

        if numbers in accepted:
            continue

        filtered = _phase6_filter_candidate(
                numbers,
                stats,
                config,
            )

        if filtered is not None:
            accepted[
                numbers
            ] = filtered

    if (
        len(accepted)
        != PHASE6_CANDIDATE_COUNT
    ):
        raise ResearchExecutionEngineError(
            "unable to retain exactly 10000 legal candidates"
        )

    return (
        list(
            accepted.values()
        ),
        attempts,
    )


def _phase6_render_top_sets(
    top10,
):
    rendered = []

    for index, item in enumerate(
        top10,
        start=1,
    ):
        features = item["features"]

        rendered.append({
            "id":
                f"S{index}",

            "numbers":
                list(
                    item[
                        "numbers"
                    ]
                ),

            "score":
                float(
                    item[
                        "score"
                    ]
                ),

            "raw_score":
                float(
                    item[
                        "raw_score"
                    ]
                ),

            "components": {
                key:
                    float(value)
                for key, value
                in item[
                    "components"
                ].items()
            },

            "risk_flags":
                list(
                    item[
                        "risk_flags"
                    ]
                ),

            "features": {
                "sum":
                    int(
                        features[
                            "sum"
                        ]
                    ),

                "odd_even":
                    features[
                        "odd_even"
                    ],

                "low_high":
                    features[
                        "low_high"
                    ],

                "consecutives": [
                    list(pair)
                    for pair
                    in features[
                        "consecutive_pairs"
                    ]
                ],

                "end_digits":
                    list(
                        features[
                            "end_digits"
                        ]
                    ),

                "max_consecutive_run":
                    int(
                        features[
                            "max_consecutive_run"
                        ]
                    ),

                "max_same_ending":
                    int(
                        features[
                            "max_same_ending"
                        ]
                    ),

                "previous_overlap":
                    int(
                        features[
                            "previous_overlap"
                        ]
                    ),

                "long_gap_count":
                    int(
                        features[
                            "long_gap_count"
                        ]
                    ),

                "max_same_decade":
                    int(
                        features[
                            "max_same_decade"
                        ]
                    ),
            },
        })

    return rendered


def _phase6_diversity(
    top_sets,
):
    pairs = []

    for left_index in range(
        len(top_sets)
    ):
        for right_index in range(
            left_index + 1,
            len(top_sets),
        ):
            pairs.append(
                _phase6_jaccard(
                    top_sets[
                        left_index
                    ][
                        "numbers"
                    ],
                    top_sets[
                        right_index
                    ][
                        "numbers"
                    ],
                )
            )

    unique_numbers = len({
        number
        for item in top_sets
        for number
        in item["numbers"]
    })

    return {
        "avg_jaccard":
            (
                sum(pairs)
                / len(pairs)
                if pairs
                else 0.0
            ),

        "max_jaccard":
            (
                max(pairs)
                if pairs
                else 0.0
            ),

        "unique_numbers":
            unique_numbers,
    }


def _phase7_execute_request_core(
    request,
    *,
    execution_context,
    actual_round_execution,
):
    _phase6_validate_prepared_request(
        request
    )

    config = phase6_challenger_config(request['challenger_id'])

    history_rows = request['history_snapshot']['history_rows']

    stats = _phase6_history_statistics(history_rows)

    candidates, attempts = _phase6_generate_candidates(request, stats, config)

    scored = [_phase6_score_candidate(candidate, stats, config, request['seed']) for candidate in candidates]

    normalized = _phase6_minmax_scores(scored)

    top10_internal = _phase6_select_top10(normalized)

    practical_indices = _phase6_practical_indices(top10_internal, config, request['seed'])

    top_sets = _phase6_render_top_sets(top10_internal)

    practical_ids = [f'S{index + 1}' for index in practical_indices]

    payload = {'schema_version': 1, 'request_id': request['request_id'], 'round': request['round'], 'challenger_id': request['challenger_id'], 'seed': request['seed'], 'research_only': True, 'execution_context': execution_context, 'actual_round_execution': actual_round_execution, 'real_shadow_publish': False, 'database_write': False, 'candidate_count': PHASE6_CANDIDATE_COUNT, 'candidate_attempts': attempts, 'top_k': PHASE6_TOP_K, 'practical_k': PHASE6_PRACTICAL_K, 'config': config, 'sets': top_sets, 'top5_practical': practical_ids, 'diversity': _phase6_diversity(top_sets)}

    result = dict(payload)

    result['result_id'] = _phase4_digest(payload)

    return result

def phase6_execute_prepared_request(
    request,
    *,
    test_mode=False,
):
    _phase6_validate_prepared_request(
        request
    )

    if test_mode is not True:
        raise ResearchExecutionEngineError(
            "Phase-6 execution is authorized only for tests"
        )

    result = _phase7_execute_request_core(
        request,
        execution_context="phase6_test_only",
        actual_round_execution=False,
    )

    phase6_validate_result(
        result
    )

    return result


def phase6_validate_result(
    result,
):
    if not isinstance(
        result,
        _phase4_Mapping,
    ):
        raise ResearchExecutionEngineError(
            "result must be a mapping"
        )

    required = (
        "result_id",
        "request_id",
        "round",
        "challenger_id",
        "seed",
        "research_only",
        "execution_context",
        "actual_round_execution",
        "real_shadow_publish",
        "database_write",
        "candidate_count",
        "top_k",
        "practical_k",
        "sets",
        "top5_practical",
        "diversity",
    )

    missing = [
        key
        for key in required
        if key not in result
    ]

    if missing:
        raise ResearchExecutionEngineError(
            "result missing fields: "
            + ",".join(
                missing
            )
        )

    if result["round"] != 1243:
        raise ResearchExecutionEngineError(
            "result round mismatch"
        )

    if (
        result[
            "challenger_id"
        ]
        not in phase3_challenger_ids()
    ):
        raise ResearchExecutionEngineError(
            "unknown result challenger"
        )

    if (
        result[
            "research_only"
        ]
        is not True
    ):
        raise ResearchExecutionEngineError(
            "result must remain research-only"
        )

    if (
        result[
            "execution_context"
        ]
        != "phase6_test_only"
    ):
        raise ResearchExecutionEngineError(
            "execution context mismatch"
        )

    if (
        result[
            "actual_round_execution"
        ]
        is not False
    ):
        raise ResearchExecutionEngineError(
            "actual round execution must remain false"
        )

    if (
        result[
            "real_shadow_publish"
        ]
        is not False
    ):
        raise ResearchExecutionEngineError(
            "shadow publish must remain false"
        )

    if (
        result[
            "database_write"
        ]
        is not False
    ):
        raise ResearchExecutionEngineError(
            "database write must remain false"
        )

    if (
        result[
            "candidate_count"
        ]
        != PHASE6_CANDIDATE_COUNT
    ):
        raise ResearchExecutionEngineError(
            "candidate count mismatch"
        )

    if (
        result[
            "top_k"
        ]
        != PHASE6_TOP_K
    ):
        raise ResearchExecutionEngineError(
            "Top-K mismatch"
        )

    if (
        result[
            "practical_k"
        ]
        != PHASE6_PRACTICAL_K
    ):
        raise ResearchExecutionEngineError(
            "Practical-K mismatch"
        )

    sets = result["sets"]

    if (
        not isinstance(
            sets,
            list,
        )
        or
        len(sets)
        != PHASE6_TOP_K
    ):
        raise ResearchExecutionEngineError(
            "result must contain 10 sets"
        )

    ids = []

    for index, item in enumerate(
        sets,
        start=1,
    ):
        if (
            item.get(
                "id"
            )
            != f"S{index}"
        ):
            raise ResearchExecutionEngineError(
                "set id mismatch"
            )

        numbers = item.get(
                "numbers"
            )

        if (
            not isinstance(
                numbers,
                list,
            )
            or
            len(numbers) != 6
            or
            len(
                set(numbers)
            ) != 6
            or
            numbers
            != sorted(numbers)
            or
            any(
                isinstance(
                    n,
                    bool,
                )
                or
                not isinstance(
                    n,
                    int,
                )
                or
                not (
                    1
                    <= n
                    <= 45
                )
                for n
                in numbers
            )
        ):
            raise ResearchExecutionEngineError(
                "illegal six-number set"
            )

        ids.append(
            item["id"]
        )

    for left_index in range(
        len(sets)
    ):
        for right_index in range(
            left_index + 1,
            len(sets),
        ):
            left = (
                sets[
                    left_index
                ][
                    "numbers"
                ]
            )

            right = (
                sets[
                    right_index
                ][
                    "numbers"
                ]
            )

            overlap = len(
                set(left).intersection(
                    right
                )
            )

            if (
                overlap
                >
                PHASE6_MAX_OVERLAP_BETWEEN_SETS
            ):
                raise ResearchExecutionEngineError(
                    "set overlap exceeds maximum"
                )

            if (
                _phase6_jaccard(
                    left,
                    right,
                )
                >
                PHASE6_JACCARD_MAX
            ):
                raise ResearchExecutionEngineError(
                    "Jaccard exceeds maximum"
                )

    practical = result["top5_practical"]

    if (
        not isinstance(
            practical,
            list,
        )
        or
        len(practical)
        != PHASE6_PRACTICAL_K
        or
        len(
            set(practical)
        )
        != PHASE6_PRACTICAL_K
        or
        any(
            item not in ids
            for item
            in practical
        )
    ):
        raise ResearchExecutionEngineError(
            "invalid practical Top5"
        )

    payload = {
        key:
            _phase3_copy(
                value
            )
        for key, value
        in result.items()
        if key != "result_id"
    }

    expected_result_id = _phase4_digest(
            payload
        )

    if (
        result[
            "result_id"
        ]
        != expected_result_id
    ):
        raise ResearchExecutionEngineError(
            "result digest mismatch"
        )

    return True


def phase6_execute_actual_round1243(
    *args,
    **kwargs,
):
    raise ResearchExecutionEngineError(
        "actual round-1243 execution remains unauthorized"
    )


# === PHASE6_RESEARCH_EXECUTION_ENGINE_V1 END ===

# === PHASE7_ACTUAL_SHADOW_EXECUTION_BRIDGE_V1 BEGIN ===

PHASE7_EXECUTION_CONTEXT = (
    "phase7_actual_shadow"
)


def phase7_execution_bridge_authorization():
    return {
        "research_only": True,
        "bridge_implementation": True,
        "explicit_runtime_authorization_required": True,
        "actual_round1243_execution": False,
        "actual_challenger_execution": False,
        "real_shadow_publish": False,
        "database_write": False,
        "production_helper_binding": False,
        "production_cli_mutation": False,
        "production_model_mutation": False,
        "production_learning": False,
    }


def phase7_validate_shadow_result(
    result,
):
    if not isinstance(
        result,
        _phase4_Mapping,
    ):
        raise ResearchExecutionEngineError(
            "shadow result must be a mapping"
        )

    if result.get(
        "research_only"
    ) is not True:
        raise ResearchExecutionEngineError(
            "shadow result must remain research-only"
        )

    if (
        result.get(
            "execution_context"
        )
        != PHASE7_EXECUTION_CONTEXT
    ):
        raise ResearchExecutionEngineError(
            "shadow execution context mismatch"
        )

    if result.get(
        "actual_round_execution"
    ) is not True:
        raise ResearchExecutionEngineError(
            "shadow result must mark actual round execution"
        )

    if result.get(
        "real_shadow_publish"
    ) is not False:
        raise ResearchExecutionEngineError(
            "shadow publication must remain disabled"
        )

    if result.get(
        "database_write"
    ) is not False:
        raise ResearchExecutionEngineError(
            "shadow database write must remain disabled"
        )

    payload = {
        key:
            _phase3_copy(
                value
            )
        for key, value
        in result.items()
        if key != "result_id"
    }

    expected_result_id = _phase4_digest(
        payload
    )

    if (
        result.get(
            "result_id"
        )
        != expected_result_id
    ):
        raise ResearchExecutionEngineError(
            "shadow result digest mismatch"
        )

    normalized = {
        key:
            _phase3_copy(
                value
            )
        for key, value
        in result.items()
    }

    normalized[
        "execution_context"
    ] = "phase6_test_only"

    normalized[
        "actual_round_execution"
    ] = False

    normalized_payload = {
        key:
            _phase3_copy(
                value
            )
        for key, value
        in normalized.items()
        if key != "result_id"
    }

    normalized[
        "result_id"
    ] = _phase4_digest(
        normalized_payload
    )

    phase6_validate_result(
        normalized
    )

    return True


def phase7_execute_shadow_prepared_request(
    request,
    *,
    execution_authorized=False,
):
    _phase6_validate_prepared_request(
        request
    )

    if execution_authorized is not True:
        raise ResearchExecutionEngineError(
            "Phase-7 shadow execution requires explicit authorization"
        )

    result = _phase7_execute_request_core(
        request,
        execution_context=PHASE7_EXECUTION_CONTEXT,
        actual_round_execution=True,
    )

    phase7_validate_shadow_result(
        result
    )

    return result


def phase7_execute_actual_shadow_round1243(
    db_path,
    *,
    execution_authorized=False,
):
    if execution_authorized is not True:
        raise ResearchExecutionEngineError(
            "Phase-7 actual round-1243 shadow execution "
            "requires explicit authorization"
        )

    requests = (
        phase5_prepare_all_real_round_requests(
            db_path,
            round_no=1243,
        )
    )

    expected_ids = tuple(
        phase3_challenger_ids()
    )

    if (
        not isinstance(
            requests,
            tuple,
        )
        or len(requests) != 24
        or len(requests)
        != len(expected_ids)
    ):
        raise ResearchExecutionEngineError(
            "Phase-7 bridge requires exactly 24 prepared requests"
        )

    actual_ids = tuple(
        request.get(
            "challenger_id"
        )
        for request
        in requests
    )

    if actual_ids != expected_ids:
        raise ResearchExecutionEngineError(
            "prepared challenger order mismatch"
        )

    results = []

    for (
        expected_id,
        request,
    ) in zip(
        expected_ids,
        requests,
    ):
        _phase6_validate_prepared_request(
            request
        )

        if (
            request[
                "challenger_id"
            ]
            != expected_id
        ):
            raise ResearchExecutionEngineError(
                "prepared challenger identity mismatch"
            )

        result = (
            phase7_execute_shadow_prepared_request(
                request,
                execution_authorized=True,
            )
        )

        phase7_validate_shadow_result(
            result
        )

        results.append(
            result
        )

    result_ids = [
        result[
            "result_id"
        ]
        for result
        in results
    ]

    if len(
        set(result_ids)
    ) != 24:
        raise ResearchExecutionEngineError(
            "Phase-7 shadow result ids must be unique"
        )

    return tuple(
        results
    )


# === PHASE7_ACTUAL_SHADOW_EXECUTION_BRIDGE_V1 END ===
