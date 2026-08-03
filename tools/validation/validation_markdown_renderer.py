"""Render validation reports as deterministic Markdown."""

from __future__ import annotations

from pathlib import Path

from tools.validation.validation_report_models import (
    ValidationReport,
)


class ValidationMarkdownRenderer:
    """Render a ValidationReport to Markdown."""

    def render(
        self,
        report: ValidationReport,
    ) -> str:
        if not isinstance(
            report,
            ValidationReport,
        ):
            raise TypeError(
                "report must be a ValidationReport"
            )

        summary = report.summary

        lines = [
            "# Validation Report",
            "",
            "## Overview",
            "",
            f"- Schema version: {report.schema_version}",
            (
                "- Generated at UTC: "
                f"{report.generated_at_utc.isoformat()}"
            ),
            f"- Source root: `{report.source_root}`",
            f"- Total runs: {summary.run_count}",
            f"- Replay runs: {summary.replay_count}",
            (
                "- Policy comparison runs: "
                f"{summary.policy_comparison_count}"
            ),
            f"- Complete runs: {summary.pass_count}",
            (
                "- Incomplete runs: "
                f"{summary.incomplete_count}"
            ),
            (
                "- Earliest round: "
                f"{self._optional(summary.earliest_round)}"
            ),
            (
                "- Latest round: "
                f"{self._optional(summary.latest_round)}"
            ),
            (
                "- Duplicate windows: "
                f"{summary.duplicate_window_count}"
            ),
            "",
            "## Runs",
            "",
        ]

        if not report.runs:
            lines.extend(
                [
                    "No validation runs were discovered.",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    (
                        "| Type | Window | Rounds | "
                        "Status | Missing files | Root |"
                    ),
                    (
                        "|---|---:|---:|---|---|---|"
                    ),
                ]
            )

            for run in report.runs:
                missing = (
                    ", ".join(run.missing_files)
                    if run.missing_files
                    else "-"
                )

                window = (
                    f"{run.start_round}"
                    f"–{run.end_round}"
                )

                lines.append(
                    "| "
                    f"{run.run_type} | "
                    f"{window} | "
                    f"{run.round_count} | "
                    f"{run.status} | "
                    f"{missing} | "
                    f"`{run.root}` |"
                )

            lines.append("")

        lines.extend(
            [
                "## Duplicate Windows",
                "",
            ]
        )

        if not report.duplicate_windows:
            lines.extend(
                [
                    "No duplicate validation windows were found.",
                    "",
                ]
            )
        else:
            for (
                start_round,
                end_round,
                run_type,
            ) in report.duplicate_windows:
                lines.append(
                    "- "
                    f"{run_type}: "
                    f"{start_round}–{end_round}"
                )

            lines.append("")

        lines.extend(
            [
                "## Interpretation",
                "",
                (
                    "- PASS means every required artifact "
                    "for that run type was found."
                ),
                (
                    "- INCOMPLETE means at least one "
                    "required artifact was missing."
                ),
                (
                    "- Duplicate windows indicate multiple "
                    "runs of the same type and round range."
                ),
                "",
            ]
        )

        return "\n".join(lines)

    def write(
        self,
        *,
        report: ValidationReport,
        output: Path,
    ) -> Path:
        output = Path(output)

        if output.exists() and output.is_dir():
            raise IsADirectoryError(output)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output.write_text(
            self.render(report),
            encoding="utf-8",
        )

        return output

    @staticmethod
    def _optional(
        value: int | None,
    ) -> str:
        return (
            str(value)
            if value is not None
            else "-"
        )
