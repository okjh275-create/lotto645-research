"""Render cross-window policy aggregation reports as Markdown."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


class CrossWindowPolicyMarkdownRenderer:
    """Render aggregated policy results as Markdown."""

    def render(
        self,
        report: Mapping[str, Any],
    ) -> str:
        if not isinstance(
            report,
            Mapping,
        ):
            raise TypeError(
                "report must be a mapping"
            )

        window_count = self._integer(
            report,
            "window_count",
        )
        total_round_count = self._integer(
            report,
            "total_round_count",
        )
        policy_count = self._integer(
            report,
            "policy_count",
        )

        policies = report.get(
            "policies"
        )

        if not isinstance(
            policies,
            Mapping,
        ):
            raise TypeError(
                "policies must be an object"
            )

        ranking = report.get(
            "ranking"
        )

        if not isinstance(
            ranking,
            list,
        ):
            raise TypeError(
                "ranking must be a list"
            )

        windows = report.get(
            "windows"
        )

        if not isinstance(
            windows,
            list,
        ):
            raise TypeError(
                "windows must be a list"
            )

        lines = [
            "# Cross-Window Policy Report",
            "",
            "## Overview",
            "",
            f"- Validation windows: {window_count}",
            f"- Total rounds: {total_round_count}",
            f"- Policies: {policy_count}",
            "",
            "## Overall Ranking",
            "",
            (
                "| Rank | Policy | Practical Δ | "
                "Best Δ | First-place windows | "
                "Average probability L1 |"
            ),
            (
                "|---:|---|---:|---:|---:|---:|"
            ),
        ]

        for row in ranking:
            if not isinstance(
                row,
                Mapping,
            ):
                raise TypeError(
                    "ranking rows must be objects"
                )

            lines.append(
                "| "
                f"{self._integer(row, 'rank')} | "
                f"{self._text(row, 'policy_name')} | "
                f"{self._format_number(row, 'practical_hit_mean_delta')} | "
                f"{self._format_number(row, 'best_hit_mean_delta')} | "
                f"{self._integer(row, 'first_place_count')} | "
                f"{self._format_number(row, 'average_probability_l1_delta')} |"
            )

        lines.extend(
            [
                "",
                "## Policy Aggregates",
                "",
            ]
        )

        for policy_name in sorted(
            policies
        ):
            values = policies[
                policy_name
            ]

            if not isinstance(
                values,
                Mapping,
            ):
                raise TypeError(
                    "policy values must be objects"
                )

            weights = values.get(
                "mean_final_weights"
            )

            if not isinstance(
                weights,
                Mapping,
            ):
                raise TypeError(
                    "mean_final_weights must be an object"
                )

            lines.extend(
                [
                    f"### {policy_name}",
                    "",
                    (
                        "- Windows: "
                        f"{self._integer(values, 'window_count')}"
                    ),
                    (
                        "- Total rounds: "
                        f"{self._integer(values, 'total_round_count')}"
                    ),
                    (
                        "- Best hit mean delta: "
                        f"{self._format_number(values, 'best_hit_mean_delta')}"
                    ),
                    (
                        "- Practical hit mean delta: "
                        f"{self._format_number(values, 'practical_hit_mean_delta')}"
                    ),
                    (
                        "- First-place windows: "
                        f"{self._integer(values, 'first_place_count')}"
                    ),
                    (
                        "- Best adaptive/no-op/ties: "
                        f"{self._integer(values, 'best_adaptive_wins')}/"
                        f"{self._integer(values, 'best_noop_wins')}/"
                        f"{self._integer(values, 'best_ties')}"
                    ),
                    (
                        "- Practical adaptive/no-op/ties: "
                        f"{self._integer(values, 'practical_adaptive_wins')}/"
                        f"{self._integer(values, 'practical_noop_wins')}/"
                        f"{self._integer(values, 'practical_ties')}"
                    ),
                    (
                        "- Average probability L1 delta: "
                        f"{self._format_number(values, 'average_probability_l1_delta')}"
                    ),
                    (
                        "- Average changed set count: "
                        f"{self._format_number(values, 'average_changed_set_count')}"
                    ),
                    "",
                    "| Mean final weight | Value |",
                    "|---|---:|",
                ]
            )

            for field in (
                "hot_weight",
                "cold_weight",
                "gap_weight",
                "trend_weight",
                "transition_weight",
                "learning_weight",
                "adaptive_weight",
            ):
                lines.append(
                    "| "
                    f"{field} | "
                    f"{self._format_number(weights, field)} |"
                )

            lines.append("")

        lines.extend(
            [
                "## Validation Windows",
                "",
                "| Window | Rounds | First-ranked policy |",
                "|---:|---:|---|",
            ]
        )

        for window in windows:
            if not isinstance(
                window,
                Mapping,
            ):
                raise TypeError(
                    "window rows must be objects"
                )

            window_ranking = window.get(
                "ranking"
            )

            if not isinstance(
                window_ranking,
                list,
            ):
                raise TypeError(
                    "window ranking must be a list"
                )

            winner = self._winner(
                window_ranking
            )

            lines.append(
                "| "
                f"{self._integer(window, 'start_round')}"
                "–"
                f"{self._integer(window, 'end_round')} | "
                f"{self._integer(window, 'round_count')} | "
                f"{winner or '-'} |"
            )

        trends = report.get(
            "weight_trends"
        )

        if isinstance(
            trends,
            Mapping,
        ):
            trend_policies = trends.get(
                "policies"
            )

            if not isinstance(
                trend_policies,
                Mapping,
            ):
                raise TypeError(
                    "trend policies must be an object"
                )

            lines.extend(
                [
                    "## Weight Trends",
                    "",
                ]
            )

            for policy_name in sorted(
                trend_policies
            ):
                policy_trends = (
                    trend_policies[
                        policy_name
                    ]
                )

                if not isinstance(
                    policy_trends,
                    Mapping,
                ):
                    raise TypeError(
                        "policy trend values "
                        "must be objects"
                    )

                weight_trends = (
                    policy_trends.get(
                        "weights"
                    )
                )

                if not isinstance(
                    weight_trends,
                    Mapping,
                ):
                    raise TypeError(
                        "weight trends must "
                        "be an object"
                    )

                lines.extend(
                    [
                        f"### {policy_name}",
                        "",
                        (
                            "| Weight | Direction | "
                            "First | Last | Net change |"
                        ),
                        "|---|---|---:|---:|---:|",
                    ]
                )

                for field in (
                    "hot_weight",
                    "cold_weight",
                    "gap_weight",
                    "trend_weight",
                    "transition_weight",
                    "learning_weight",
                    "adaptive_weight",
                ):
                    values = weight_trends.get(
                        field
                    )

                    if not isinstance(
                        values,
                        Mapping,
                    ):
                        raise TypeError(
                            "weight trend rows "
                            "must be objects"
                        )

                    direction = values.get(
                        "direction"
                    )

                    if not isinstance(
                        direction,
                        str,
                    ):
                        raise TypeError(
                            "trend direction "
                            "must be a string"
                        )

                    lines.append(
                        "| "
                        f"{field} | "
                        f"{direction} | "
                        f"{self._optional_number(values.get('first'))} | "
                        f"{self._optional_number(values.get('last'))} | "
                        f"{self._optional_number(values.get('net_change'))} |"
                    )

                lines.append("")

        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                (
                    "- Mean deltas are weighted by each "
                    "window's round count."
                ),
                (
                    "- Win, loss, tie, and changed-portfolio "
                    "counts are summed across windows."
                ),
                (
                    "- First-place count is based on each "
                    "window's stored policy ranking."
                ),
                (
                    "- This report summarizes observed replay "
                    "results and does not establish predictive "
                    "causality or statistical significance."
                ),
                "",
            ]
        )

        return "\n".join(lines)

    def write(
        self,
        *,
        report: Mapping[str, Any],
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
    def _winner(
        ranking: list[Any],
    ) -> str | None:
        for row in ranking:
            if not isinstance(
                row,
                Mapping,
            ):
                continue

            if row.get("rank") != 1:
                continue

            scenario = row.get(
                "scenario"
            )

            if isinstance(
                scenario,
                str,
            ):
                return scenario

            policy_name = row.get(
                "policy_name"
            )

            if isinstance(
                policy_name,
                str,
            ):
                return policy_name

        return None

    @staticmethod
    def _integer(
        values: Mapping[str, Any],
        key: str,
    ) -> int:
        value = values.get(key)

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise TypeError(
                f"{key} must be an integer"
            )

        return value

    @staticmethod
    def _text(
        values: Mapping[str, Any],
        key: str,
    ) -> str:
        value = values.get(key)

        if not isinstance(
            value,
            str,
        ):
            raise TypeError(
                f"{key} must be a string"
            )

        return value

    @staticmethod
    def _optional_number(
        value: Any,
    ) -> str:
        if value is None:
            return "-"

        if (
            isinstance(value, bool)
            or not isinstance(
                value,
                (int, float),
            )
        ):
            raise TypeError(
                "optional numeric value must "
                "be numeric or None"
            )

        return f"{float(value):.6f}"

    @staticmethod
    def _format_number(
        values: Mapping[str, Any],
        key: str,
    ) -> str:
        value = values.get(key)

        if (
            isinstance(value, bool)
            or not isinstance(
                value,
                (int, float),
            )
        ):
            raise TypeError(
                f"{key} must be numeric"
            )

        return f"{float(value):.6f}"
