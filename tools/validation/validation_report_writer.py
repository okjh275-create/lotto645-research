"""Build and write deterministic validation reports."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from tools.validation.validation_report_models import (
    ValidationReport,
    ValidationReportBuilder,
)
from tools.validation.validation_run_discovery import (
    ValidationRunDiscovery,
)


class ValidationReportWriter:
    """Discover validation runs and write a JSON report."""

    def __init__(
        self,
        *,
        discovery: ValidationRunDiscovery | None = None,
        builder: ValidationReportBuilder | None = None,
    ) -> None:
        if (
            discovery is not None
            and not isinstance(
                discovery,
                ValidationRunDiscovery,
            )
        ):
            raise TypeError(
                "discovery must be a "
                "ValidationRunDiscovery or None"
            )

        if (
            builder is not None
            and not isinstance(
                builder,
                ValidationReportBuilder,
            )
        ):
            raise TypeError(
                "builder must be a "
                "ValidationReportBuilder or None"
            )

        self._discovery = (
            discovery
            if discovery is not None
            else ValidationRunDiscovery()
        )
        self._builder = (
            builder
            if builder is not None
            else ValidationReportBuilder()
        )

    @property
    def discovery(
        self,
    ) -> ValidationRunDiscovery:
        return self._discovery

    @property
    def builder(
        self,
    ) -> ValidationReportBuilder:
        return self._builder

    def build(
        self,
        *,
        source_root: Path,
        generated_at_utc: datetime | None = None,
    ) -> ValidationReport:
        source_root = Path(source_root)

        runs = self.discovery.discover(
            source_root
        )

        return self.builder.build(
            source_root=source_root,
            runs=runs,
            generated_at_utc=generated_at_utc,
        )

    def write_json(
        self,
        *,
        report: ValidationReport,
        output: Path,
    ) -> Path:
        if not isinstance(
            report,
            ValidationReport,
        ):
            raise TypeError(
                "report must be a ValidationReport"
            )

        output = Path(output)

        if output.exists() and output.is_dir():
            raise IsADirectoryError(output)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output.write_text(
            json.dumps(
                report.as_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        return output

    def build_and_write_json(
        self,
        *,
        source_root: Path,
        output: Path,
        generated_at_utc: datetime | None = None,
    ) -> tuple[
        ValidationReport,
        Path,
    ]:
        report = self.build(
            source_root=source_root,
            generated_at_utc=generated_at_utc,
        )

        output_path = self.write_json(
            report=report,
            output=output,
        )

        return report, output_path
