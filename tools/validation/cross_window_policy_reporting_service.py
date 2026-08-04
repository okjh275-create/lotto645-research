"""Integrated cross-window policy reporting service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from tools.validation.cross_window_policy_aggregator import (
    CrossWindowPolicyAggregator,
)
from tools.validation.cross_window_policy_markdown_renderer import (
    CrossWindowPolicyMarkdownRenderer,
)


@dataclass(frozen=True, slots=True)
class CrossWindowPolicyReportingResult:
    """Cross-window report and generated paths."""

    report: dict[str, Any]
    json_path: Path
    markdown_path: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "report": dict(self.report),
            "json_path": str(self.json_path),
            "markdown_path": str(
                self.markdown_path
            ),
        }


class CrossWindowPolicyReportingService:
    """Generate JSON and Markdown cross-window policy reports."""

    def __init__(
        self,
        *,
        aggregator: (
            CrossWindowPolicyAggregator | None
        ) = None,
        renderer: (
            CrossWindowPolicyMarkdownRenderer
            | None
        ) = None,
    ) -> None:
        if (
            aggregator is not None
            and not isinstance(
                aggregator,
                CrossWindowPolicyAggregator,
            )
        ):
            raise TypeError(
                "aggregator must be a "
                "CrossWindowPolicyAggregator or None"
            )

        if (
            renderer is not None
            and not isinstance(
                renderer,
                CrossWindowPolicyMarkdownRenderer,
            )
        ):
            raise TypeError(
                "renderer must be a "
                "CrossWindowPolicyMarkdownRenderer "
                "or None"
            )

        self._aggregator = (
            aggregator
            if aggregator is not None
            else CrossWindowPolicyAggregator()
        )
        self._renderer = (
            renderer
            if renderer is not None
            else CrossWindowPolicyMarkdownRenderer()
        )

    @property
    def aggregator(
        self,
    ) -> CrossWindowPolicyAggregator:
        return self._aggregator

    @property
    def renderer(
        self,
    ) -> CrossWindowPolicyMarkdownRenderer:
        return self._renderer

    def generate_from_paths(
        self,
        *,
        paths: Sequence[Path],
        output_root: Path,
        stem: str = "cross_window_policy_report",
    ) -> CrossWindowPolicyReportingResult:
        normalized_stem = self._normalize_stem(
            stem
        )

        report = self.aggregator.aggregate(
            paths
        )

        return self._write(
            report=report,
            output_root=output_root,
            stem=normalized_stem,
        )

    def discover_and_generate(
        self,
        *,
        source_root: Path,
        output_root: Path,
        stem: str = "cross_window_policy_report",
    ) -> CrossWindowPolicyReportingResult:
        normalized_stem = self._normalize_stem(
            stem
        )

        report = (
            self.aggregator
            .discover_and_aggregate(
                source_root
            )
        )

        return self._write(
            report=report,
            output_root=output_root,
            stem=normalized_stem,
        )

    def _write(
        self,
        *,
        report: dict[str, Any],
        output_root: Path,
        stem: str,
    ) -> CrossWindowPolicyReportingResult:
        output_root = Path(output_root)

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        json_path = (
            self.aggregator.write_json(
                report=report,
                output=(
                    output_root
                    / f"{stem}.json"
                ),
            )
        )

        markdown_path = (
            self.renderer.write(
                report=report,
                output=(
                    output_root
                    / f"{stem}.md"
                ),
            )
        )

        return CrossWindowPolicyReportingResult(
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
