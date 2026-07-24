"""Dependency-free semantic-version handling for Project A.

The parser intentionally supports the version forms currently used by the
integrated Lotto645 projects:

- 1.0
- 1.0.0
- 0.8.0.dev0
- 1.2.0rc1
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from .exceptions import VersionError


_VERSION_PATTERN = re.compile(
    r"^\s*"
    r"(?P<major>0|[1-9]\d*)"
    r"\."
    r"(?P<minor>0|[1-9]\d*)"
    r"(?:\.(?P<patch>0|[1-9]\d*))?"
    r"(?:(?P<separator>[.-]?)"
    r"(?P<label>dev|a|alpha|b|beta|rc)"
    r"(?P<number>0|[1-9]\d*)?)?"
    r"\s*$",
    re.IGNORECASE,
)

_PRE_RELEASE_ORDER = {
    "dev": 0,
    "a": 1,
    "alpha": 1,
    "b": 2,
    "beta": 2,
    "rc": 3,
}


@dataclass(frozen=True, slots=True)
class Version:
    """Normalized dependency-free semantic version."""

    major: int
    minor: int
    patch: int = 0
    pre_release: str | None = None
    pre_release_number: int = 0

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.major, self.minor, self.patch)
        ):
            raise VersionError("version components must be non-negative integers")

        label = self.pre_release
        if label is not None:
            normalized = label.strip().lower()
            if normalized not in _PRE_RELEASE_ORDER:
                raise VersionError(f"unsupported pre-release label: {label}")
            if (
                isinstance(self.pre_release_number, bool)
                or not isinstance(self.pre_release_number, int)
                or self.pre_release_number < 0
            ):
                raise VersionError(
                    "pre_release_number must be a non-negative integer"
                )
            object.__setattr__(self, "pre_release", normalized)

    @property
    def release_tuple(self) -> tuple[int, int, int]:
        """Return only the numeric release portion."""

        return self.major, self.minor, self.patch

    @property
    def is_pre_release(self) -> bool:
        """Return whether this version represents a development build."""

        return self.pre_release is not None

    @property
    def base_version(self) -> str:
        """Return major.minor.patch without a pre-release suffix."""

        return f"{self.major}.{self.minor}.{self.patch}"

    def comparison_key(self) -> tuple[int, int, int, int, int]:
        """Return a deterministic ordering key."""

        if self.pre_release is None:
            pre_order = 4
            pre_number = 0
        else:
            pre_order = _PRE_RELEASE_ORDER[self.pre_release]
            pre_number = self.pre_release_number

        return (
            self.major,
            self.minor,
            self.patch,
            pre_order,
            pre_number,
        )

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self.comparison_key() < other.comparison_key()

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self.comparison_key() <= other.comparison_key()

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self.comparison_key() > other.comparison_key()

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self.comparison_key() >= other.comparison_key()

    def __str__(self) -> str:
        if self.pre_release is None:
            return self.base_version
        return (
            f"{self.base_version}."
            f"{self.pre_release}{self.pre_release_number}"
        )


def parse_version(value: str) -> Version:
    """Parse a supported version string."""

    if not isinstance(value, str):
        raise VersionError("version must be a string")

    match = _VERSION_PATTERN.fullmatch(value)
    if match is None:
        raise VersionError(
            "version must use major.minor[.patch] with an optional "
            "dev, alpha, beta, or rc suffix"
        )

    label = match.group("label")
    number = match.group("number")

    return Version(
        major=int(match.group("major")),
        minor=int(match.group("minor")),
        patch=int(match.group("patch") or 0),
        pre_release=label.lower() if label else None,
        pre_release_number=int(number or 0),
    )


def same_release_line(actual: str | Version, required: str | Version) -> bool:
    """Return whether actual belongs to the required numeric release line.

    This deliberately treats ``0.8.0.dev0`` as belonging to the ``0.8.0``
    release line. Project D currently uses this exact development-version
    arrangement.
    """

    actual_version = (
        parse_version(actual) if isinstance(actual, str) else actual
    )
    required_version = (
        parse_version(required) if isinstance(required, str) else required
    )

    if not isinstance(actual_version, Version):
        raise VersionError("actual must be a string or Version")
    if not isinstance(required_version, Version):
        raise VersionError("required must be a string or Version")

    return actual_version.release_tuple == required_version.release_tuple


def is_major_compatible(
    actual: str | Version,
    required: str | Version,
) -> bool:
    """Return whether actual is on the same compatible major line.

    A compatible version must:

    - use the same major version;
    - not be numerically older than the required release;
    - not be a pre-release of the minimum required release.
    """

    actual_version = (
        parse_version(actual) if isinstance(actual, str) else actual
    )
    required_version = (
        parse_version(required) if isinstance(required, str) else required
    )

    if not isinstance(actual_version, Version):
        raise VersionError("actual must be a string or Version")
    if not isinstance(required_version, Version):
        raise VersionError("required must be a string or Version")

    if actual_version.major != required_version.major:
        return False

    if actual_version.release_tuple < required_version.release_tuple:
        return False

    if (
        actual_version.release_tuple == required_version.release_tuple
        and actual_version.is_pre_release
        and not required_version.is_pre_release
    ):
        return False

    return True
