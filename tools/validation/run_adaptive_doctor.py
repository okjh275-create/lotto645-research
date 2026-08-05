"""Generate adaptive automation doctor JSON and Markdown reports."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from lrp.evolution.feedback import (
    AdaptiveAutomationDoctor,
    AdaptiveAutomationDoctorMarkdownRenderer,
    AdaptiveAutomationDoctorReportWriter,
    AdaptiveAutomationRepository,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adaptive-doctor",
        description=(
            "Inspect an adaptive automation repository "
            "and generate JSON and Markdown reports."
        ),
    )

    parser.add_argument(
        "--repository",
        required=True,
        type=Path,
        help=(
            "Adaptive automation repository root."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help=(
            "Directory for generated doctor reports."
        ),
    )
    parser.add_argument(
        "--stem",
        default="adaptive_doctor",
        help=(
            "Base filename for JSON and Markdown reports."
        ),
    )
    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help=(
            "Return exit code 1 when the doctor result fails."
        ),
    )

    return parser


def run(
    argv: Sequence[str] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        stem = _validate_stem(
            args.stem
        )

        repository = (
            AdaptiveAutomationRepository(
                args.repository
            )
        )

        report = (
            AdaptiveAutomationDoctor()
            .inspect(repository)
        )

        output_root = Path(
            args.output
        )

        if (
            output_root.exists()
            and not output_root.is_dir()
        ):
            raise NotADirectoryError(
                output_root
            )

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        json_path = (
            output_root
            / f"{stem}.json"
        )
        markdown_path = (
            output_root
            / f"{stem}.md"
        )

        json_result = (
            AdaptiveAutomationDoctorReportWriter()
            .write(
                report,
                json_path,
            )
        )

        markdown = (
            AdaptiveAutomationDoctorMarkdownRenderer()
            .render(report)
        )

        markdown_result = _write_markdown(
            path=markdown_path,
            content=markdown,
        )

        payload = {
            "status": (
                "PASS"
                if report.overall_ok
                else "FAIL"
            ),
            "overall_ok": report.overall_ok,
            "latest_revision": (
                report.latest_revision
            ),
            "rollback_count": (
                report.rollback_count
            ),
            "recommendation_count": (
                report.recommendation_count
            ),
            "error_count": (
                report.error_count
            ),
            "warning_count": (
                report.warning_count
            ),
            "json_path": str(
                json_result.path
            ),
            "json_created": (
                json_result.created
            ),
            "json_changed": (
                json_result.changed
            ),
            "markdown_path": str(
                markdown_result["path"]
            ),
            "markdown_created": (
                markdown_result["created"]
            ),
            "markdown_changed": (
                markdown_result["changed"]
            ),
        }

    except (
        FileNotFoundError,
        FileExistsError,
        IsADirectoryError,
        NotADirectoryError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        parser.exit(
            status=2,
            message=f"error: {exc}\n",
        )

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    if (
        args.fail_on_issues
        and not report.overall_ok
    ):
        return 1

    return 0


def _validate_stem(
    value: object,
) -> str:
    if not isinstance(value, str):
        raise TypeError(
            "stem must be a string"
        )

    normalized = value.strip()

    if not normalized:
        raise ValueError(
            "stem must not be empty"
        )

    allowed = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789._-"
    )

    if any(
        character not in allowed
        for character in normalized
    ):
        raise ValueError(
            "stem contains unsupported characters"
        )

    return normalized


def _write_markdown(
    *,
    path: Path,
    content: str,
) -> dict[str, object]:
    if not isinstance(content, str):
        raise TypeError(
            "Markdown content must be a string"
        )

    if not content.endswith("\n"):
        raise ValueError(
            "Markdown content must end "
            "with a newline"
        )

    serialized = content.encode(
        "utf-8"
    )

    if path.exists() and path.is_dir():
        raise IsADirectoryError(path)

    existed_before = path.exists()

    if existed_before:
        existing = path.read_bytes()

        if existing == serialized:
            return {
                "path": path,
                "created": False,
                "changed": False,
            }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name + ".tmp"
    )

    if temporary.exists():
        if temporary.is_dir():
            raise IsADirectoryError(
                temporary
            )

        temporary.unlink()

    temporary.write_bytes(
        serialized
    )
    temporary.replace(path)

    written = path.read_bytes()

    if written != serialized:
        raise RuntimeError(
            "Markdown report verification failed"
        )

    if written.startswith(
        b"\xef\xbb\xbf"
    ):
        raise RuntimeError(
            "Markdown report must not "
            "contain a UTF-8 BOM"
        )

    return {
        "path": path,
        "created": not existed_before,
        "changed": True,
    }


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
