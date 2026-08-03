"""Integrated validation-report generation service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tools.validation.validation_markdown_renderer import (
    ValidationMarkdownRenderer,
)
from tools.validation.validation_report_models import (
    ValidationReport,
)
from tools.validation.validation_report_writer import (
    ValidationReportWriter,
)


@dataclass(frozen=True, slots=True)
class ValidationReportingResult:
    """Result paths and report metadata."""

    report: ValidationReport
    json_path: Path
    markdown_path: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "report": self.report.as_dict(),
            "json_path": str(self.json_path),
            "markdown_path": str(
                self.markdown_path
            ),
        }


class ValidationReportingService:
    """Generate JSON and Markdown validation reports."""

    def __init__(
        self,
        *,
        writer: ValidationReportWriter | None = None,
        renderer: ValidationMarkdownRenderer | None = None,
    ) -> None:
        if (
            writer is not None
            and not isinstance(
                writer,
                ValidationReportWriter,
            )
        ):
            raise TypeError(
                "writer must be a "
                "ValidationReportWriter or None"
            )

        if (
            renderer is not None
            and not isinstance(
                renderer,
                ValidationMarkdownRenderer,
            )
        ):
            raise TypeError(
                "renderer must be a "
                "ValidationMarkdownRenderer or None"
            )

        self._writer = (
            writer
            if writer is not None
            else ValidationReportWriter()
        )
        self._renderer = (
            renderer
            if renderer is not None
            else ValidationMarkdownRenderer()
        )

    @property
    def writer(
        self,
    ) -> ValidationReportWriter:
        return self._writer

    @property
    def renderer(
        self,
    ) -> ValidationMarkdownRenderer:
        return self._renderer

    def generate(
        self,
        *,
        source_root: Path,
        output_root: Path,
        generated_at_utc: datetime | None = None,
        stem: str = "validation_report",
    ) -> ValidationReportingResult:
        source_root = Path(source_root)
        output_root = Path(output_root)

        normalized_stem = self._normalize_stem(
            stem
        )

        report = self.writer.build(
            source_root=source_root,
            generated_at_utc=generated_at_utc,
        )

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        json_path = self.writer.write_json(
            report=report,
            output=(
                output_root
                / f"{normalized_stem}.json"
            ),
        )

        markdown_path = self.renderer.write(
            report=report,
            output=(
                output_root
                / f"{normalized_stem}.md"
            ),
        )

        return ValidationReportingResult(
            report=report,
            json_path=json_path,
            markdown_path=markdown_path,
        )

    @staticmethod
    def _normalize_stem(
        stem: str,
    ) -> str:
        if not isinstance(stem, str):
            raise TypeError(
                "stem must be a string"
            )

        normalized = stem.strip()

        if not normalized:
            raise ValueError(
                "stem must not be empty"
            )

        if (
            "/" in normalized
            or "\\" in normalized
        ):
            raise ValueError(
                "stem must not contain "
                "path separators"
            )

        if normalized in {
            ".",
            "..",
        }:
            raise ValueError(
                "stem must be a file stem"
            )

        return normalized
