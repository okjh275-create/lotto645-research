"""Frozen protocol helpers for LRP shadow research.

This module is research-only. It validates the frozen shadow experiment
contract and never mutates production prediction logic, the production
database, or production artifacts.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


SHADOW_START_ROUND = 1243
SHADOW_END_ROUND = 1252
SHADOW_ROUND_COUNT = 10

SEED_NAMESPACE = "LRP-v4.0|shadow"

CHALLENGER_ID_PATTERN = re.compile(
    r"^(?:CG|PS|SC|HF|LG)\d{2}(?:_[A-Z0-9_]+)?$"
)


class ShadowProtocolError(ValueError):
    """Base exception for frozen shadow-protocol violations."""


class ShadowRoundError(ShadowProtocolError):
    """Raised when a round is outside the frozen shadow window."""


class ChallengerIdError(ShadowProtocolError):
    """Raised when a challenger ID is invalid or unregistered."""


class SeedContractError(ShadowProtocolError):
    """Raised when a stored seed conflicts with the frozen seed rule."""


class ShadowOutputExistsError(ShadowProtocolError):
    """Raised when a shadow artifact target already exists."""


def validate_shadow_round(round_no: int) -> int:
    """Validate a round against the frozen 1243..1252 window."""

    if isinstance(round_no, bool) or not isinstance(round_no, int):
        raise ShadowRoundError(
            "shadow round must be an integer"
        )

    if not SHADOW_START_ROUND <= round_no <= SHADOW_END_ROUND:
        raise ShadowRoundError(
            "shadow round outside frozen window "
            f"{SHADOW_START_ROUND}..{SHADOW_END_ROUND}: "
            f"{round_no}"
        )

    return round_no


def derive_shadow_seed(
    round_no: int,
    challenger_id: str,
) -> int:
    """Derive the deterministic seed frozen in OP-116."""

    validate_shadow_round(
        round_no
    )

    if not isinstance(challenger_id, str) or not challenger_id:
        raise ChallengerIdError(
            "challenger_id must be a non-empty string"
        )

    token = (
        f"{SEED_NAMESPACE}|"
        f"{round_no}|"
        f"{challenger_id}"
    ).encode("utf-8")

    digest = hashlib.sha256(
        token
    ).digest()

    seed = (
        int.from_bytes(
            digest[:8],
            byteorder="big",
            signed=False,
        )
        % 2147483647
    )

    return seed if seed != 0 else 1


def load_frozen_registry(
    registry_path: str | Path,
) -> dict[str, Any]:
    """Load the frozen OP-116 challenger registry."""

    path = Path(
        registry_path
    )

    if not path.is_file():
        raise ShadowProtocolError(
            f"frozen registry not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        payload = json.load(
            handle
        )

    if not isinstance(payload, dict):
        raise ShadowProtocolError(
            "registry root must be a JSON object"
        )

    state = payload.get(
        "state"
    )

    if (
        state is not None
        and state != "CHALLENGER_DESIGN_FROZEN"
    ):
        raise ShadowProtocolError(
            f"unexpected registry state: {state!r}"
        )

    return payload


def _walk_json(
    value: Any,
    path: tuple[str, ...] = (),
) -> Iterable[
    tuple[
        tuple[str, ...],
        Any,
    ]
]:
    yield path, value

    if isinstance(
        value,
        Mapping,
    ):
        for key, child in value.items():
            yield from _walk_json(
                child,
                path + (str(key),),
            )

    elif isinstance(
        value,
        list,
    ):
        for index, child in enumerate(
            value
        ):
            yield from _walk_json(
                child,
                path + (str(index),),
            )


def collect_challenger_ids(
    registry: Mapping[str, Any],
) -> tuple[str, ...]:
    """Discover challenger IDs without coupling to one JSON layout."""

    found: set[str] = set()

    for path, value in _walk_json(
        registry
    ):
        for component in path:
            if CHALLENGER_ID_PATTERN.fullmatch(
                component
            ):
                found.add(
                    component
                )

        if (
            isinstance(value, str)
            and CHALLENGER_ID_PATTERN.fullmatch(
                value
            )
        ):
            found.add(
                value
            )

    return tuple(
        sorted(found)
    )


def validate_challenger_id(
    registry: Mapping[str, Any],
    challenger_id: str,
) -> str:
    """Require challenger_id to exist in the frozen registry."""

    if (
        not isinstance(
            challenger_id,
            str,
        )
        or not CHALLENGER_ID_PATTERN.fullmatch(
            challenger_id
        )
    ):
        raise ChallengerIdError(
            "invalid challenger ID format: "
            f"{challenger_id!r}"
        )

    registered = set(
        collect_challenger_ids(
            registry
        )
    )

    if challenger_id not in registered:
        raise ChallengerIdError(
            "challenger ID not found in frozen registry: "
            f"{challenger_id}"
        )

    return challenger_id


def _explicit_seed_candidates(
    registry: Mapping[str, Any],
    round_no: int,
    challenger_id: str,
) -> tuple[int, ...]:
    """Locate explicit registry seeds, if present."""

    candidates: set[int] = set()

    round_text = str(
        round_no
    )

    for _, value in _walk_json(
        registry
    ):
        if not isinstance(
            value,
            Mapping,
        ):
            continue

        record_round = value.get(
            "round"
        )

        record_id = (
            value.get(
                "challenger_id"
            )
            or value.get(
                "id"
            )
            or value.get(
                "challenger"
            )
        )

        seed = value.get(
            "seed"
        )

        try:
            record_round_int = int(
                record_round
            )
        except (
            TypeError,
            ValueError,
        ):
            record_round_int = None

        if (
            record_round_int == round_no
            and record_id == challenger_id
            and isinstance(seed, int)
            and not isinstance(seed, bool)
        ):
            candidates.add(
                seed
            )

    for path, value in _walk_json(
        registry
    ):
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
        ):
            continue

        lower_path = tuple(
            component.lower()
            for component in path
        )

        if (
            round_text in path
            and challenger_id in path
            and any(
                "seed" in component
                for component in lower_path
            )
        ):
            candidates.add(
                value
            )

    return tuple(
        sorted(candidates)
    )


def resolve_pre_registered_seed(
    registry: Mapping[str, Any],
    round_no: int,
    challenger_id: str,
) -> int:
    """Resolve and verify the deterministic OP-116 seed."""

    validate_shadow_round(
        round_no
    )

    validate_challenger_id(
        registry,
        challenger_id,
    )

    derived = derive_shadow_seed(
        round_no,
        challenger_id,
    )

    explicit = _explicit_seed_candidates(
        registry,
        round_no,
        challenger_id,
    )

    conflicting = [
        seed
        for seed in explicit
        if seed != derived
    ]

    if conflicting:
        raise SeedContractError(
            "explicit registry seed conflicts with "
            "the frozen seed rule: "
            f"round={round_no}, "
            f"challenger={challenger_id}, "
            f"derived={derived}, "
            f"explicit={list(explicit)}"
        )

    return derived


def construct_shadow_artifact_paths(
    repo_root: str | Path,
    round_no: int,
) -> dict[str, Path]:
    """Construct research artifact paths without creating them."""

    validate_shadow_round(
        round_no
    )

    root = (
        Path(repo_root)
        / "artifacts"
        / "research"
        / "shadow"
        / f"round_{round_no}"
    )

    return {
        "root":
            root,

        "plan":
            root
            / "shadow_plan.json",

        "manifest":
            root
            / "shadow_manifest.json",

        "results":
            root
            / "shadow_results.json",
    }


def reject_existing_output(
    output_root: str | Path,
) -> None:
    """Reject overwrite of a pre-existing research output directory."""

    path = Path(
        output_root
    )

    if path.exists():
        raise ShadowOutputExistsError(
            "shadow output already exists; "
            f"overwrite is forbidden: {path}"
        )


__all__ = [
    "SHADOW_START_ROUND",
    "SHADOW_END_ROUND",
    "SHADOW_ROUND_COUNT",
    "ShadowProtocolError",
    "ShadowRoundError",
    "ChallengerIdError",
    "SeedContractError",
    "ShadowOutputExistsError",
    "validate_shadow_round",
    "derive_shadow_seed",
    "load_frozen_registry",
    "collect_challenger_ids",
    "validate_challenger_id",
    "resolve_pre_registered_seed",
    "construct_shadow_artifact_paths",
    "reject_existing_output",
]