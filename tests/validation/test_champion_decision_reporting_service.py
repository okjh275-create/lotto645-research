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


def _report() -> HistoricalChampionSelection:
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


def test_service_writes_standard_json_artifact(
    tmp_path: Path,
) -> None:
    from tools.validation.champion_decision_reporting_service import (
        ChampionDecisionReportingService,
    )

    report = _report()

    service = ChampionDecisionReportingService()

    result = service.write(
        report=report,
        output_root=tmp_path,
    )

    expected = (
        tmp_path
        / "champion_decision.json"
    )

    assert result == expected
    assert expected.is_file()

    payload = json.loads(
        expected.read_text(
            encoding="utf-8"
        )
    )

    assert payload == report.as_dict()


def test_service_creates_output_root(
    tmp_path: Path,
) -> None:
    from tools.validation.champion_decision_reporting_service import (
        ChampionDecisionReportingService,
    )

    output_root = (
        tmp_path
        / "nested"
        / "report"
    )

    result = ChampionDecisionReportingService().write(
        report=_report(),
        output_root=output_root,
    )

    assert result == (
        output_root
        / "champion_decision.json"
    )

    assert result.is_file()


def test_service_rejects_wrong_report_type(
    tmp_path: Path,
) -> None:
    from tools.validation.champion_decision_reporting_service import (
        ChampionDecisionReportingService,
    )

    with pytest.raises(
        TypeError,
        match="HistoricalChampionSelection",
    ):
        ChampionDecisionReportingService().write(
            report=object(),
            output_root=tmp_path,
        )


def test_service_rejects_file_as_output_root(
    tmp_path: Path,
) -> None:
    from tools.validation.champion_decision_reporting_service import (
        ChampionDecisionReportingService,
    )

    file_root = tmp_path / "already-file"

    file_root.write_text(
        "x",
        encoding="utf-8",
    )

    with pytest.raises(
        NotADirectoryError,
    ):
        ChampionDecisionReportingService().write(
            report=_report(),
            output_root=file_root,
        )
