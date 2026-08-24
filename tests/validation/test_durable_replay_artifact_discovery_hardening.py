from __future__ import annotations

import ast
from pathlib import Path

import pytest

from lrp.contracts.exceptions import ContractError
from lrp.operations.durable_replay_artifact_discovery import (
    DurableReplayArtifactDiscoveryRequest,
    DurableReplayArtifactDiscoveryService,
    DurableReplayArtifactSelector,
)
from lrp.operations.durable_replay_execution import (
    DurableReplayExecutionSource,
)


def test_multi_root_canonical_path_projection() -> None:
    service = DurableReplayArtifactDiscoveryService()

    cases = (
        (
            "predictions",
            1,
            Path("predictions")
            / "prediction-evaluation-sources"
            / "round_0001"
            / "evaluation_source.json",
        ),
        (
            Path("output"),
            42,
            Path("output")
            / "prediction-evaluation-sources"
            / "round_0042"
            / "evaluation_source.json",
        ),
        (
            Path("custom"),
            1234,
            Path("custom")
            / "prediction-evaluation-sources"
            / "round_1234"
            / "evaluation_source.json",
        ),
    )

    for root, round_no, expected in cases:
        request = DurableReplayArtifactDiscoveryRequest(
            artifact_root=root,
            candidate_selectors=(
                DurableReplayArtifactSelector(
                    round_no=round_no,
                    model_name="model",
                ),
            ),
            baseline_selectors=(),
        )

        candidate, baseline = service.discover(
            request=request
        )

        assert candidate[0].artifact_path == expected
        assert baseline == ()


@pytest.mark.parametrize(
    ("round_no", "expected_name"),
    [
        (1, "round_0001"),
        (9, "round_0009"),
        (10, "round_0010"),
        (99, "round_0099"),
        (100, "round_0100"),
        (999, "round_0999"),
        (1000, "round_1000"),
    ],
)
def test_low_round_zero_padding_is_canonical(
    round_no: int,
    expected_name: str,
) -> None:
    request = DurableReplayArtifactDiscoveryRequest(
        artifact_root="root",
        candidate_selectors=(
            DurableReplayArtifactSelector(
                round_no=round_no,
                model_name="model",
            ),
        ),
        baseline_selectors=(),
    )

    candidate, _ = (
        DurableReplayArtifactDiscoveryService().discover(
            request=request
        )
    )

    assert candidate[0].artifact_path.parent.name == expected_name


def test_mixed_candidate_and_baseline_order_is_preserved() -> None:
    candidate_rounds = (
        1300,
        1001,
        1250,
        1100,
    )

    baseline_rounds = (
        1400,
        900,
        1200,
    )

    request = DurableReplayArtifactDiscoveryRequest(
        artifact_root="root",
        candidate_selectors=tuple(
            DurableReplayArtifactSelector(
                round_no=value,
                model_name=f"candidate-{value}",
            )
            for value in candidate_rounds
        ),
        baseline_selectors=tuple(
            DurableReplayArtifactSelector(
                round_no=value,
                model_name=f"baseline-{value}",
            )
            for value in baseline_rounds
        ),
    )

    candidate, baseline = (
        DurableReplayArtifactDiscoveryService().discover(
            request=request
        )
    )

    assert tuple(
        item.round_no
        for item in candidate
    ) == candidate_rounds

    assert tuple(
        item.round_no
        for item in baseline
    ) == baseline_rounds


def test_duplicate_selectors_remain_duplicate() -> None:
    selector = DurableReplayArtifactSelector(
        round_no=1234,
        model_name="same-model",
        regime_id="same-regime",
        strategy_name="same-strategy",
    )

    request = DurableReplayArtifactDiscoveryRequest(
        artifact_root="root",
        candidate_selectors=(
            selector,
            selector,
            selector,
        ),
        baseline_selectors=(),
    )

    candidate, baseline = (
        DurableReplayArtifactDiscoveryService().discover(
            request=request
        )
    )

    assert len(candidate) == 3
    assert candidate[0] == candidate[1] == candidate[2]
    assert baseline == ()


def test_exact_provenance_passthrough() -> None:
    request = DurableReplayArtifactDiscoveryRequest(
        artifact_root="root",
        candidate_selectors=(
            DurableReplayArtifactSelector(
                round_no=4321,
                model_name="model-x",
                regime_id="regime-x",
                strategy_name="strategy-x",
            ),
        ),
        baseline_selectors=(),
    )

    candidate, _ = (
        DurableReplayArtifactDiscoveryService().discover(
            request=request
        )
    )

    assert candidate == (
        DurableReplayExecutionSource(
            artifact_path=(
                Path("root")
                / "prediction-evaluation-sources"
                / "round_4321"
                / "evaluation_source.json"
            ),
            round_no=4321,
            model_name="model-x",
            regime_id="regime-x",
            strategy_name="strategy-x",
        ),
    )


def test_none_provenance_passthrough() -> None:
    request = DurableReplayArtifactDiscoveryRequest(
        artifact_root="root",
        candidate_selectors=(
            DurableReplayArtifactSelector(
                round_no=1,
                model_name="model",
            ),
        ),
        baseline_selectors=(),
    )

    candidate, _ = (
        DurableReplayArtifactDiscoveryService().discover(
            request=request
        )
    )

    assert candidate[0].regime_id is None
    assert candidate[0].strategy_name is None


def test_empty_shapes_remain_stable() -> None:
    request = DurableReplayArtifactDiscoveryRequest(
        artifact_root="root",
        candidate_selectors=(),
        baseline_selectors=(),
    )

    first = (
        DurableReplayArtifactDiscoveryService().discover(
            request=request
        )
    )

    second = (
        DurableReplayArtifactDiscoveryService().discover(
            request=request
        )
    )

    assert first == ((), ())
    assert second == ((), ())
    assert first == second


@pytest.mark.parametrize(
    "value",
    [
        None,
        object(),
        {},
        [],
        (),
        "request",
        1,
        True,
    ],
)
def test_invalid_request_types_fail_closed(
    value,
) -> None:
    with pytest.raises(
        ContractError,
        match=(
            "request must be "
            "DurableReplayArtifactDiscoveryRequest"
        ),
    ):
        DurableReplayArtifactDiscoveryService().discover(
            request=value
        )


@pytest.mark.parametrize(
    ("label", "value"),
    [
        ("candidate", None),
        ("candidate", object()),
        ("candidate", {}),
        ("candidate", []),
        ("candidate", ()),
        ("candidate", "selector"),
        ("candidate", 1),
        ("candidate", True),
        ("baseline", None),
        ("baseline", object()),
        ("baseline", {}),
        ("baseline", []),
        ("baseline", ()),
        ("baseline", "selector"),
        ("baseline", 1),
        ("baseline", True),
    ],
)
def test_invalid_selector_types_fail_closed(
    label: str,
    value,
) -> None:
    kwargs = {
        "artifact_root": "root",
        "candidate_selectors": (),
        "baseline_selectors": (),
    }

    kwargs[
        f"{label}_selectors"
    ] = (
        value,
    )

    request = DurableReplayArtifactDiscoveryRequest(
        **kwargs
    )

    with pytest.raises(
        ContractError,
        match=(
            rf"{label} selector must be "
            r"DurableReplayArtifactSelector"
        ),
    ):
        DurableReplayArtifactDiscoveryService().discover(
            request=request
        )


def test_missing_artifact_path_is_not_probed() -> None:
    root = Path(
        "definitely-does-not-exist"
    )

    request = DurableReplayArtifactDiscoveryRequest(
        artifact_root=root,
        candidate_selectors=(
            DurableReplayArtifactSelector(
                round_no=7777,
                model_name="model",
            ),
        ),
        baseline_selectors=(),
    )

    candidate, _ = (
        DurableReplayArtifactDiscoveryService().discover(
            request=request
        )
    )

    assert candidate[0].artifact_path == (
        root
        / "prediction-evaluation-sources"
        / "round_7777"
        / "evaluation_source.json"
    )


def test_repeated_execution_is_semantically_stable() -> None:
    request = DurableReplayArtifactDiscoveryRequest(
        artifact_root="root",
        candidate_selectors=(
            DurableReplayArtifactSelector(
                round_no=1234,
                model_name="candidate",
                regime_id="r1",
                strategy_name="s1",
            ),
        ),
        baseline_selectors=(
            DurableReplayArtifactSelector(
                round_no=1200,
                model_name="baseline",
                regime_id="r2",
                strategy_name="s2",
            ),
        ),
    )

    service = DurableReplayArtifactDiscoveryService()

    first = service.discover(
        request=request
    )

    second = service.discover(
        request=request
    )

    assert first == second


def test_request_and_selectors_are_not_mutated() -> None:
    candidate_selectors = (
        DurableReplayArtifactSelector(
            round_no=1234,
            model_name="candidate",
            regime_id="r1",
            strategy_name="s1",
        ),
    )

    baseline_selectors = (
        DurableReplayArtifactSelector(
            round_no=1200,
            model_name="baseline",
            regime_id="r2",
            strategy_name="s2",
        ),
    )

    request = DurableReplayArtifactDiscoveryRequest(
        artifact_root="root",
        candidate_selectors=candidate_selectors,
        baseline_selectors=baseline_selectors,
    )

    before_request = repr(request)
    before_candidate = repr(candidate_selectors)
    before_baseline = repr(baseline_selectors)

    DurableReplayArtifactDiscoveryService().discover(
        request=request
    )

    assert repr(request) == before_request
    assert repr(candidate_selectors) == before_candidate
    assert repr(baseline_selectors) == before_baseline


def test_product_has_exact_dependency_boundary() -> None:
    path = Path(
        "lrp/operations/durable_replay_artifact_discovery.py"
    )

    source = path.read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(
                alias.name
                for alias in node.names
            )

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(
                    node.module
                )

    assert set(imports) == {
        "__future__",
        "dataclasses",
        "pathlib",
        "re",
        "lrp.contracts.exceptions",
        "lrp.operations.durable_replay_execution",
    }


def test_product_preserves_legacy_owned_raise_sites() -> None:
    source = Path(
        "lrp/operations/durable_replay_artifact_discovery.py"
    ).read_text(
        encoding="utf-8-sig"
    )
    tree = ast.parse(source)
    raises = tuple(
        ast.unparse(node.exc)
        for node in ast.walk(tree)
        if isinstance(node, ast.Raise)
        and node.exc is not None
    )
    assert (
        "ContractError('request must be "
        "DurableReplayArtifactDiscoveryRequest')"
        in raises
    )
    assert (
        "ContractError(f'{label} selector must be "
        "DurableReplayArtifactSelector')"
        in raises
    )


def test_product_has_no_exception_normalization_layer() -> None:
    source = Path(
        "lrp/operations/durable_replay_artifact_discovery.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    handlers = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.ExceptHandler,
        )
    )

    assert handlers == ()


def test_product_has_no_filesystem_probe_or_io() -> None:
    source = Path(
        "lrp/operations/durable_replay_artifact_discovery.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        "read_text",
        "read_bytes",
        ".open(",
        ".exists(",
        ".is_file(",
        ".stat(",
        ".glob(",
        ".rglob(",
        ".iterdir(",
        "write_text",
        "write_bytes",
        ".mkdir(",
    )

    assert not any(
        token in source
        for token in forbidden
    )


def test_hardening_freezes_no_direct_json_dependency() -> None:
    source = Path(
        "lrp/operations/durable_replay_artifact_discovery.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(
                alias.name
                for alias in node.names
            )

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(
                    node.module
                )

    assert "json" not in imports

    assert "json.loads" not in source
    assert "json.dumps" not in source


def test_product_has_no_lower_layer_ownership_leak() -> None:
    source = Path(
        "lrp/operations/durable_replay_artifact_discovery.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        "DurableReplayExecutionService",
        "DurableReplayEvaluationOrchestrator",
        "TopKReplayEvaluationService",
        ".execute(",
        ".evaluate(",
        "lrp.cli",
        "argparse",
        "tools.validation",
        "manifest",
        "load_latest",
    )

    assert not any(
        token in source
        for token in forbidden
    )


def test_structural_call_contract_is_exact() -> None:
    source = Path(
        "lrp/operations/durable_replay_artifact_discovery.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    calls = tuple(
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    )

    assert sum(
        call == "Path"
        for call in calls
    ) == 1

    assert sum(
        call == "DurableReplayExecutionSource"
        for call in calls
    ) == 1

    assert sum(
        call.endswith("._execution_source")
        for call in calls
    ) == 2


def test_hardening_freezes_minimal_public_surface() -> None:
    source = Path(
        "lrp/operations/durable_replay_artifact_discovery.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    classes = tuple(
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    )

    functions = tuple(
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    )

    assert classes == (
        "DurableReplayArtifactSelector",
        "DurableReplayArtifactDiscoveryRequest",
        "DurableReplayArtifactDiscoveryService",
    )

    assert functions == ()