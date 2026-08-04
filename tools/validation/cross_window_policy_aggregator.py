"""Aggregate adaptive-policy results across validation windows."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any


class CrossWindowPolicyAggregator:
    """Aggregate multiple policy-comparison reports."""

    def aggregate(
        self,
        paths: Sequence[Path],
    ) -> dict[str, Any]:
        normalized_paths = tuple(
            Path(path)
            for path in paths
        )

        if not normalized_paths:
            raise ValueError(
                "paths must not be empty"
            )

        if len(normalized_paths) != len(
            set(
                path.resolve()
                for path in normalized_paths
            )
        ):
            raise ValueError(
                "paths must be unique"
            )

        windows = [
            self._load_window(path)
            for path in normalized_paths
        ]

        windows.sort(
            key=lambda item: (
                item["start_round"],
                item["end_round"],
                item["path"],
            )
        )

        self._validate_non_overlapping(
            windows
        )

        policy_names = sorted(
            {
                result["scenario"]["name"]
                for window in windows
                for result in window["results"]
            }
        )

        aggregates = {
            name: self._aggregate_policy(
                name=name,
                windows=windows,
            )
            for name in policy_names
        }

        ranking = self._rank(
            aggregates
        )

        return {
            "schema_version": 1,
            "window_count": len(windows),
            "total_round_count": sum(
                window["round_count"]
                for window in windows
            ),
            "policy_count": len(
                policy_names
            ),
            "windows": [
                {
                    "path": window["path"],
                    "start_round": (
                        window["start_round"]
                    ),
                    "end_round": (
                        window["end_round"]
                    ),
                    "round_count": (
                        window["round_count"]
                    ),
                    "ranking": window["ranking"],
                }
                for window in windows
            ],
            "policies": aggregates,
            "ranking": ranking,
        }

    def discover_and_aggregate(
        self,
        root: Path,
    ) -> dict[str, Any]:
        root = Path(root)

        if not root.exists():
            raise FileNotFoundError(root)

        if not root.is_dir():
            raise NotADirectoryError(root)

        paths = sorted(
            root.rglob(
                "policy_comparison.json"
            )
        )

        if not paths:
            raise ValueError(
                "no policy comparison reports found"
            )

        return self.aggregate(paths)

    def write_json(
        self,
        *,
        report: Mapping[str, Any],
        output: Path,
    ) -> Path:
        if not isinstance(
            report,
            Mapping,
        ):
            raise TypeError(
                "report must be a mapping"
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
                dict(report),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        return output

    def _load_window(
        self,
        path: Path,
    ) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(path)

        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(
            payload,
            Mapping,
        ):
            raise TypeError(
                f"{path.name} must contain "
                "a JSON object"
            )

        config = self._mapping(
            payload,
            "config",
        )

        start_round = self._integer(
            config,
            "start_round",
        )
        end_round = self._integer(
            config,
            "end_round",
        )

        if start_round < 1:
            raise ValueError(
                "start_round must be positive"
            )

        if end_round < start_round:
            raise ValueError(
                "end_round must not be less "
                "than start_round"
            )

        results = payload.get(
            "results"
        )

        if not isinstance(
            results,
            list,
        ):
            raise TypeError(
                "results must be a list"
            )

        if not results:
            raise ValueError(
                "results must not be empty"
            )

        normalized_results = [
            self._normalize_result(
                result
            )
            for result in results
        ]

        names = [
            result["scenario"]["name"]
            for result in normalized_results
        ]

        if len(names) != len(set(names)):
            raise ValueError(
                "scenario names must be unique "
                "within each window"
            )

        ranking = payload.get(
            "ranking",
            []
        )

        if not isinstance(
            ranking,
            list,
        ):
            raise TypeError(
                "ranking must be a list"
            )

        return {
            "path": str(path),
            "start_round": start_round,
            "end_round": end_round,
            "round_count": (
                end_round - start_round + 1
            ),
            "results": normalized_results,
            "ranking": ranking,
        }

    def _normalize_result(
        self,
        result: Any,
    ) -> dict[str, Any]:
        if not isinstance(
            result,
            Mapping,
        ):
            raise TypeError(
                "policy result must be an object"
            )

        scenario = self._mapping(
            result,
            "scenario",
        )

        name = scenario.get(
            "name"
        )

        if not isinstance(name, str):
            raise TypeError(
                "scenario name must be a string"
            )

        normalized_name = name.strip()

        if not normalized_name:
            raise ValueError(
                "scenario name must not be empty"
            )

        effectiveness = self._mapping(
            result,
            "effectiveness",
        )

        nested_summary = effectiveness.get(
            "summary"
        )

        summary = (
            nested_summary
            if isinstance(
                nested_summary,
                Mapping,
            )
            else effectiveness
        )

        profile = self._mapping(
            result,
            "final_profile",
        )
        weights = self._mapping(
            profile,
            "weights",
        )

        fields = (
            "round_count",
            "best_hit_mean_delta",
            "practical_hit_mean_delta",
            "best_adaptive_wins",
            "best_noop_wins",
            "best_ties",
            "practical_adaptive_wins",
            "practical_noop_wins",
            "practical_ties",
            "average_probability_l1_delta",
            "average_changed_set_count",
            "changed_portfolio_round_count",
        )

        normalized_summary = {
            field: self._numeric(
                summary,
                field,
            )
            for field in fields
        }

        round_count = normalized_summary[
            "round_count"
        ]

        if (
            isinstance(round_count, float)
            and not round_count.is_integer()
        ):
            raise ValueError(
                "round_count must be an integer"
            )

        normalized_summary[
            "round_count"
        ] = int(round_count)

        weight_fields = (
            "hot_weight",
            "cold_weight",
            "gap_weight",
            "trend_weight",
            "transition_weight",
            "learning_weight",
            "adaptive_weight",
        )

        normalized_weights = {
            field: float(
                self._numeric(
                    weights,
                    field,
                )
            )
            for field in weight_fields
        }

        total_weight = sum(
            normalized_weights.values()
        )

        if abs(
            total_weight - 1.0
        ) > 1e-9:
            raise ValueError(
                "final profile weights must "
                "sum to 1.0"
            )

        return {
            "scenario": {
                "name": normalized_name,
                "adjustment_scale": (
                    scenario.get(
                        "adjustment_scale"
                    )
                ),
                "minimum_weight": (
                    scenario.get(
                        "minimum_weight"
                    )
                ),
            },
            "summary": normalized_summary,
            "weights": normalized_weights,
        }

    def _aggregate_policy(
        self,
        *,
        name: str,
        windows: list[
            dict[str, Any]
        ],
    ) -> dict[str, Any]:
        matches = []

        for window in windows:
            result = next(
                (
                    item
                    for item in window[
                        "results"
                    ]
                    if item["scenario"][
                        "name"
                    ] == name
                ),
                None,
            )

            if result is not None:
                matches.append(
                    {
                        "window": window,
                        "result": result,
                    }
                )

        total_rounds = sum(
            item["result"]["summary"][
                "round_count"
            ]
            for item in matches
        )

        if total_rounds < 1:
            raise ValueError(
                f"policy {name} has no rounds"
            )

        weighted_fields = (
            "best_hit_mean_delta",
            "practical_hit_mean_delta",
            "average_probability_l1_delta",
            "average_changed_set_count",
        )

        weighted = {
            field: sum(
                item["result"]["summary"][
                    field
                ]
                * item["result"]["summary"][
                    "round_count"
                ]
                for item in matches
            )
            / total_rounds
            for field in weighted_fields
        }

        count_fields = (
            "best_adaptive_wins",
            "best_noop_wins",
            "best_ties",
            "practical_adaptive_wins",
            "practical_noop_wins",
            "practical_ties",
            "changed_portfolio_round_count",
        )

        counts = {
            field: int(
                sum(
                    item["result"]["summary"][
                        field
                    ]
                    for item in matches
                )
            )
            for field in count_fields
        }

        weight_fields = (
            "hot_weight",
            "cold_weight",
            "gap_weight",
            "trend_weight",
            "transition_weight",
            "learning_weight",
            "adaptive_weight",
        )

        mean_weights = {
            field: fmean(
                item["result"]["weights"][
                    field
                ]
                for item in matches
            )
            for field in weight_fields
        }

        first_place_count = sum(
            self._window_winner(
                item["window"]
            ) == name
            for item in matches
        )

        return {
            "policy_name": name,
            "window_count": len(matches),
            "total_round_count": total_rounds,
            **weighted,
            **counts,
            "first_place_count": (
                first_place_count
            ),
            "mean_final_weights": (
                mean_weights
            ),
            "windows": [
                {
                    "start_round": item[
                        "window"
                    ]["start_round"],
                    "end_round": item[
                        "window"
                    ]["end_round"],
                    "round_count": item[
                        "result"
                    ]["summary"]["round_count"],
                    "best_hit_mean_delta": (
                        item["result"]["summary"][
                            "best_hit_mean_delta"
                        ]
                    ),
                    "practical_hit_mean_delta": (
                        item["result"]["summary"][
                            "practical_hit_mean_delta"
                        ]
                    ),
                    "final_weights": dict(
                        item["result"]["weights"]
                    ),
                }
                for item in matches
            ],
        }

    @staticmethod
    def _window_winner(
        window: Mapping[str, Any],
    ) -> str | None:
        ranking = window.get(
            "ranking"
        )

        if not isinstance(
            ranking,
            list,
        ):
            return None

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

        return None

    @staticmethod
    def _rank(
        policies: Mapping[
            str,
            Mapping[str, Any],
        ],
    ) -> list[dict[str, Any]]:
        rows = [
            {
                "policy_name": name,
                "practical_hit_mean_delta": (
                    values[
                        "practical_hit_mean_delta"
                    ]
                ),
                "best_hit_mean_delta": (
                    values[
                        "best_hit_mean_delta"
                    ]
                ),
                "first_place_count": (
                    values[
                        "first_place_count"
                    ]
                ),
                "average_probability_l1_delta": (
                    values[
                        "average_probability_l1_delta"
                    ]
                ),
            }
            for name, values in policies.items()
        ]

        rows.sort(
            key=lambda row: (
                -float(
                    row[
                        "practical_hit_mean_delta"
                    ]
                ),
                -float(
                    row[
                        "best_hit_mean_delta"
                    ]
                ),
                -int(
                    row[
                        "first_place_count"
                    ]
                ),
                float(
                    row[
                        "average_probability_l1_delta"
                    ]
                ),
                str(
                    row["policy_name"]
                ),
            )
        )

        return [
            {
                "rank": index,
                **row,
            }
            for index, row in enumerate(
                rows,
                start=1,
            )
        ]

    @staticmethod
    def _validate_non_overlapping(
        windows: list[
            Mapping[str, Any]
        ],
    ) -> None:
        for previous, current in zip(
            windows,
            windows[1:],
        ):
            if (
                current["start_round"]
                <= previous["end_round"]
            ):
                raise ValueError(
                    "validation windows must "
                    "not overlap"
                )

    @staticmethod
    def _mapping(
        values: Mapping[str, Any],
        key: str,
    ) -> Mapping[str, Any]:
        value = values.get(key)

        if not isinstance(
            value,
            Mapping,
        ):
            raise TypeError(
                f"{key} must be an object"
            )

        return value

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
    def _numeric(
        values: Mapping[str, Any],
        key: str,
    ) -> float:
        if key not in values:
            raise ValueError(
                f"missing field: {key}"
            )

        value = values[key]

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

        return float(value)
