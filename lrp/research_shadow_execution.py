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
