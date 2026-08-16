from __future__ import annotations

import json
from pathlib import Path

import pytest

from lrp.evaluation import (
    ChampionPromotionDecision,
    ChampionRanking,
    ChampionSelection,
    ModelRankingEntry,
)
from tools.validation.model_evaluation_champion import (
    HistoricalChampionSelection,
)
from tools.validation.model_evaluation_matrix import (
    HistoricalEvaluationMatrix,
)


def _historical_selection() -> HistoricalChampionSelection:
    entry = ModelRankingEntry(
        rank=1,
        model_name="calibration",
        practical_score=0.20,
        best_score=0.20,
        stability_score=0.20,
        significance_score=0.20,
        composite_score=0.20,
        eligible=True,
        exclusion_reasons=(),
    )

    ranking = ChampionRanking(
        entries=(entry,),
        champion="calibration",
    )

    matrix = HistoricalEvaluationMatrix(
        windows=(),
        evaluations=(),
        ranking=ranking,
    )

    decision = ChampionPromotionDecision(
        candidate="calibration",
        promoted=False,
        promoted_model=None,
        composite_margin=0.005,
        rejection_reasons=(
            "significance_below_minimum",
        ),
    )

    selection = ChampionSelection(
        ranking_champion="calibration",
        promotion=decision,
        selected_model=None,
    )

    return HistoricalChampionSelection(
        matrix=matrix,
        selection=selection,
    )


def test_writer_writes_deterministic_json(
    tmp_path: Path,
) -> None:
    from tools.validation.champion_decision_report_writer import (
        ChampionDecisionReportWriter,
    )

    report = _historical_selection()

    output = (
        tmp_path
        / "nested"
        / "champion_decision.json"
    )

    result = ChampionDecisionReportWriter().write_json(
        report=report,
        output=output,
    )

    assert result == output
    assert output.is_file()

    expected = (
        json.dumps(
            report.as_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    assert output.read_text(
        encoding="utf-8"
    ) == expected


def test_writer_creates_parent_directory(
    tmp_path: Path,
) -> None:
    from tools.validation.champion_decision_report_writer import (
        ChampionDecisionReportWriter,
    )

    output = (
        tmp_path
        / "a"
        / "b"
        / "decision.json"
    )

    ChampionDecisionReportWriter().write_json(
        report=_historical_selection(),
        output=output,
    )

    assert output.is_file()


def test_writer_rejects_wrong_report_type(
    tmp_path: Path,
) -> None:
    from tools.validation.champion_decision_report_writer import (
        ChampionDecisionReportWriter,
    )

    with pytest.raises(
        TypeError,
        match="HistoricalChampionSelection",
    ):
        ChampionDecisionReportWriter().write_json(
            report={},
            output=tmp_path / "decision.json",
        )


def test_writer_rejects_directory_output(
    tmp_path: Path,
) -> None:
    from tools.validation.champion_decision_report_writer import (
        ChampionDecisionReportWriter,
    )

    with pytest.raises(
        IsADirectoryError,
    ):
        ChampionDecisionReportWriter().write_json(
            report=_historical_selection(),
            output=tmp_path,
        )
