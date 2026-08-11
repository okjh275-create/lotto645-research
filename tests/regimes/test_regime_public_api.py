from __future__ import annotations

import subprocess
import sys


EXPECTED_REGIME_PUBLIC_API = [
    "SUPPORTED_REGIMES",
    "RegimeDecision",
    "RegimeDetector",
    "RegimeDetectorConfig",
    "RegimeFeatureExtractor",
    "RegimeFeatureSnapshot",
    "RegimeStabilityConfig",
    "RegimeStabilityPolicy",
]


def test_regime_public_api_is_stable() -> None:
    import lrp.regimes as regimes

    assert regimes.__all__ == (
        EXPECTED_REGIME_PUBLIC_API
    )

    for name in EXPECTED_REGIME_PUBLIC_API:
        assert hasattr(regimes, name)


def test_learning_and_persistence_types_remain_layered() -> None:
    import lrp.regimes as regimes

    internal = {
        "AdaptiveLearningRatePolicy",
        "RegimeBayesianRepository",
        "RegimeBayesianSnapshot",
        "RegimeBayesianState",
        "RegimeBayesianUpdater",
        "RegimeCalibrationRepository",
        "RegimeCalibrationSnapshot",
        "RegimeCalibrationUpdater",
        "RegimeReward",
        "RegimeRewardCalculator",
        "RegimeBayesianNotFoundError",
        "RegimeCalibrationNotFoundError",
    }

    assert internal.isdisjoint(
        set(regimes.__all__)
    )


def test_regime_integration_public_api_exposes_providers() -> None:
    import lrp.regimes.integration as integration

    expected = {
        "ActiveGlobalRegimeAdjustmentAdapter",
        "GlobalRegimeAdjustmentAdapter",
        "NoOpGlobalRegimeAdjustmentAdapter",
        "ProbabilityVectorAdjuster",
        "RegimeAdjustmentConfig",
        "RegimeBayesianProvider",
        "RepositoryRegimeBayesianProvider",
        "StaticRegimeBayesianProvider",
        "RegimeCalibrationProvider",
        "RepositoryRegimeCalibrationProvider",
        "StaticRegimeCalibrationProvider",
    }

    assert expected <= set(integration.__all__)

    for name in expected:
        assert hasattr(integration, name)


def test_regime_integration_import_is_service_independent() -> None:
    code = """
import sys
import lrp.regimes.integration

assert (
    "lrp.evolution.services.review_learning_service"
    not in sys.modules
)
"""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        result.stdout + result.stderr
    )
