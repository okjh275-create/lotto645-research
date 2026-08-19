"""Production champion operational audit contracts and service."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

from lrp.production.champion_registry import (
    ProductionChampionRegistry,
)
from lrp.pipelines.prediction import PredictionPipeline
from lrp.production.prediction_configuration import (
    ProductionPredictionConfiguration,
)
from lrp.production.champion_active_snapshot import (
    ProductionChampionActiveSnapshotReader,
)
from lrp.production.champion_decision import (
    ProductionChampionDecision,
)


_AUDIT_STATUSES = frozenset(
    {
        "PASS",
        "WARN",
        "FAIL",
    }
)

_CHECK_STATUSES = frozenset(
    {
        "PASS",
        "WARN",
        "FAIL",
    }
)

_ISSUE_SEVERITIES = frozenset(
    {
        "WARN",
        "FAIL",
    }
)


@dataclass(frozen=True)
class ProductionChampionAuditCheck:
    """One deterministic production audit check."""

    name: str
    status: str
    detail: str

    def __post_init__(
        self,
    ) -> None:
        if not self.name:
            raise ValueError(
                "name must not be empty"
            )

        if (
            self.status
            not in _CHECK_STATUSES
        ):
            raise ValueError(
                "invalid audit check status"
            )


@dataclass(frozen=True)
class ProductionChampionAuditIssue:
    """One production audit issue."""

    code: str
    severity: str
    message: str

    def __post_init__(
        self,
    ) -> None:
        if not self.code:
            raise ValueError(
                "code must not be empty"
            )

        if (
            self.severity
            not in _ISSUE_SEVERITIES
        ):
            raise ValueError(
                "invalid audit issue severity"
            )


@dataclass(frozen=True)
class ProductionChampionAuditResult:
    """Deterministic production champion audit result."""

    status: str
    selected_model: str | None
    resolved_model: str
    fallback_applied: bool
    fallback_reason: str | None
    checks: tuple[
        ProductionChampionAuditCheck,
        ...,
    ]
    issues: tuple[
        ProductionChampionAuditIssue,
        ...,
    ]

    def __post_init__(
        self,
    ) -> None:
        if (
            self.status
            not in _AUDIT_STATUSES
        ):
            raise ValueError(
                "invalid audit result status"
            )

        if not self.resolved_model:
            raise ValueError(
                "resolved_model must not be empty"
            )

        if (
            self.fallback_applied
            and self.fallback_reason is None
        ):
            raise ValueError(
                "fallback_reason is required "
                "when fallback is applied"
            )

        if (
            not self.fallback_applied
            and self.fallback_reason
            is not None
        ):
            raise ValueError(
                "fallback_reason must be None "
                "when fallback is not applied"
            )

    def as_dict(
        self,
    ) -> dict[str, object]:
        """Serialize deterministically."""

        return {
            "status": self.status,
            "selected_model": (
                self.selected_model
            ),
            "resolved_model": (
                self.resolved_model
            ),
            "fallback_applied": (
                self.fallback_applied
            ),
            "fallback_reason": (
                self.fallback_reason
            ),
            "checks": [
                {
                    "name": check.name,
                    "status": check.status,
                    "detail": check.detail,
                }
                for check in self.checks
            ],
            "issues": [
                {
                    "code": issue.code,
                    "severity": (
                        issue.severity
                    ),
                    "message": (
                        issue.message
                    ),
                }
                for issue in self.issues
            ],
        }


class ProductionChampionAudit:
    """Audit active champion production state.

    Q-03 scope intentionally stops before physical
    snapshot validation and PredictionPipeline.load().
    """

    def audit(
        self,
        *,
        registry_root: str | Path,
        snapshot_root: str | Path,
    ) -> ProductionChampionAuditResult:
        registry_root_path = Path(
            registry_root
        )

        snapshot_root_path = Path(
            snapshot_root
        )

        checks: list[
            ProductionChampionAuditCheck
        ] = []

        issues: list[
            ProductionChampionAuditIssue
        ] = []

        # --------------------------------------------------
        # Registry root
        # --------------------------------------------------

        if not registry_root_path.exists():
            return self._failure(
                code="registry_missing",
                message=(
                    "production registry root "
                    "does not exist"
                ),
                check_name="registry",
                detail=(
                    "production registry root "
                    "is missing"
                ),
            )

        if not registry_root_path.is_dir():
            return self._failure(
                code="registry_invalid",
                message=(
                    "production registry root "
                    "is not a directory"
                ),
                check_name="registry",
                detail=(
                    "production registry root "
                    "is invalid"
                ),
            )

        checks.append(
            ProductionChampionAuditCheck(
                name="registry",
                status="PASS",
                detail=(
                    "production registry "
                    "is available"
                ),
            )
        )

        registry = (
            ProductionChampionRegistry(
                registry_root_path
            )
        )

        active_decision_path = (
            registry.active_decision_path
        )

        publication_path = (
            registry_root_path
            / "active"
            / "publication.json"
        )

        # --------------------------------------------------
        # Active decision presence
        # --------------------------------------------------

        if not active_decision_path.exists():
            return self._failure(
                code="active_decision_missing",
                message=(
                    "active champion decision "
                    "is missing"
                ),
                check_name="active_decision",
                detail=(
                    "active champion decision "
                    "is missing"
                ),
                checks=checks,
            )

        if not active_decision_path.is_file():
            return self._failure(
                code="active_decision_invalid",
                message=(
                    "active champion decision "
                    "is not a file"
                ),
                check_name="active_decision",
                detail=(
                    "active champion decision "
                    "is invalid"
                ),
                checks=checks,
            )

        # --------------------------------------------------
        # Publication presence
        # --------------------------------------------------

        if not publication_path.exists():
            return self._failure(
                code="publication_missing",
                message=(
                    "publication provenance "
                    "is missing"
                ),
                check_name="publication",
                detail=(
                    "publication provenance "
                    "is missing"
                ),
                checks=checks,
            )

        if not publication_path.is_file():
            return self._failure(
                code="publication_invalid",
                message=(
                    "publication provenance "
                    "is not a file"
                ),
                check_name="publication",
                detail=(
                    "publication provenance "
                    "is invalid"
                ),
                checks=checks,
            )

        # --------------------------------------------------
        # Read active decision
        # --------------------------------------------------

        try:
            active_snapshot = (
                ProductionChampionActiveSnapshotReader()
                .read(
                    registry_root_path
                )
            )

            decision_payload = json.loads(
                active_snapshot.decision_bytes.decode(
                    "utf-8-sig"
                )
            )

            decision = (
                ProductionChampionDecision
                .from_payload(
                    decision_payload
                )
            )
        except Exception:
            return self._failure(
                code="active_decision_invalid",
                message=(
                    "active champion decision "
                    "cannot be read"
                ),
                check_name="active_decision",
                detail=(
                    "active champion decision "
                    "schema is invalid"
                ),
                checks=checks,
            )

        checks.append(
            ProductionChampionAuditCheck(
                name="active_decision",
                status="PASS",
                detail=(
                    "active champion decision "
                    "is readable"
                ),
            )
        )

        # --------------------------------------------------
        # Read publication JSON
        # --------------------------------------------------

        try:
            publication_object = json.loads(
                active_snapshot.publication_bytes.decode(
                    "utf-8-sig"
                )
            )
        except (
            UnicodeError,
            json.JSONDecodeError,
        ):
            return self._failure(
                code="publication_invalid",
                message=(
                    "publication provenance "
                    "JSON is invalid"
                ),
                check_name="publication",
                detail=(
                    "publication provenance "
                    "cannot be decoded"
                ),
                checks=checks,
                selected_model=(
                    decision.selected_model
                ),
            )

        if not isinstance(
            publication_object,
            Mapping,
        ):
            return self._failure(
                code="publication_invalid",
                message=(
                    "publication provenance "
                    "must be a mapping"
                ),
                check_name="publication",
                detail=(
                    "publication provenance "
                    "schema is invalid"
                ),
                checks=checks,
                selected_model=(
                    decision.selected_model
                ),
            )

        publication = publication_object

        required_fields = {
            "source_path",
            "source_sha256",
            "published_path",
            "published_at_kst",
            "selected_model",
        }

        if not required_fields.issubset(
            publication
        ):
            return self._failure(
                code="publication_invalid",
                message=(
                    "publication provenance "
                    "schema is incomplete"
                ),
                check_name="publication",
                detail=(
                    "publication provenance "
                    "required fields are missing"
                ),
                checks=checks,
                selected_model=(
                    decision.selected_model
                ),
            )

        published_model = publication[
            "selected_model"
        ]

        if (
            published_model is not None
            and not isinstance(
                published_model,
                str,
            )
        ):
            return self._failure(
                code="publication_invalid",
                message=(
                    "publication selected_model "
                    "is invalid"
                ),
                check_name="publication",
                detail=(
                    "publication provenance "
                    "schema is invalid"
                ),
                checks=checks,
                selected_model=(
                    decision.selected_model
                ),
            )

        checks.append(
            ProductionChampionAuditCheck(
                name="publication",
                status="PASS",
                detail=(
                    "publication provenance "
                    "is readable"
                ),
            )
        )

        # --------------------------------------------------
        # Decision / provenance agreement
        # --------------------------------------------------

        if (
            published_model
            != decision.selected_model
        ):
            return self._failure(
                code="selected_model_mismatch",
                message=(
                    "active decision and "
                    "publication selected_model "
                    "do not agree"
                ),
                check_name="model_agreement",
                detail=(
                    "selected_model mismatch"
                ),
                checks=checks,
                selected_model=(
                    decision.selected_model
                ),
            )

        checks.append(
            ProductionChampionAuditCheck(
                name="model_agreement",
                status="PASS",
                detail=(
                    "active decision and "
                    "publication selected_model "
                    "agree"
                ),
            )
        )

        # --------------------------------------------------
        # Provenance SHA-256 format
        # --------------------------------------------------

        source_sha256 = publication[
            "source_sha256"
        ]

        if (
            not isinstance(
                source_sha256,
                str,
            )
            or not self._is_sha256(
                source_sha256
            )
        ):
            return self._failure(
                code="source_sha256_invalid",
                message=(
                    "publication source_sha256 "
                    "is invalid"
                ),
                check_name="source_sha256",
                detail=(
                    "source_sha256 must be "
                    "64 hexadecimal characters"
                ),
                checks=checks,
                selected_model=(
                    decision.selected_model
                ),
            )

        checks.append(
            ProductionChampionAuditCheck(
                name="source_sha256",
                status="PASS",
                detail=(
                    "publication source_sha256 "
                    "format is valid"
                ),
            )
        )

        # --------------------------------------------------
        # Active decision hash agreement
        # --------------------------------------------------

        active_sha256 = (
            hashlib.sha256(
                active_snapshot.decision_bytes
            )
            .hexdigest()
        )

        if (
            active_sha256.lower()
            != source_sha256.lower()
        ):
            return self._failure(
                code="active_hash_mismatch",
                message=(
                    "active champion decision "
                    "hash does not match "
                    "publication provenance"
                ),
                check_name="active_hash",
                detail=(
                    "active decision SHA-256 "
                    "does not match provenance"
                ),
                checks=checks,
                selected_model=(
                    decision.selected_model
                ),
            )

        checks.append(
            ProductionChampionAuditCheck(
                name="active_hash",
                status="PASS",
                detail=(
                    "active decision SHA-256 "
                    "matches provenance"
                ),
            )
        )

        # --------------------------------------------------
        # Reuse existing production activation contract.
        #
        # Physical snapshot existence and actual pipeline
        # loading are deliberately deferred to Q-04/Q-05.
        # --------------------------------------------------

        try:
            configuration = (
                ProductionPredictionConfiguration
                .from_registry(
                    registry_root=(
                        registry_root_path
                    ),
                    snapshot_root=(
                        snapshot_root_path
                    ),
                )
            )
        except Exception:
            return self._failure(
                code="activation_invalid",
                message=(
                    "production activation "
                    "could not be resolved"
                ),
                check_name="activation",
                detail=(
                    "production activation "
                    "resolution failed"
                ),
                checks=checks,
                selected_model=(
                    decision.selected_model
                ),
            )

        checks.append(
            ProductionChampionAuditCheck(
                name="activation",
                status="PASS",
                detail=(
                    "production activation "
                    "resolved deterministically"
                ),
            )
        )

        # --------------------------------------------------
        # Required production snapshots.
        #
        # Requirements come from the resolved production
        # configuration. Q-04 validates filesystem readiness
        # only; actual PredictionPipeline.load() remains
        # deferred to Q-05.
        # --------------------------------------------------

        snapshot_failed = False

        calibration_snapshot = (
            configuration
            .regime_calibration_snapshot_root
        )

        if calibration_snapshot is not None:
            if not calibration_snapshot.exists():
                snapshot_failed = True

                checks.append(
                    ProductionChampionAuditCheck(
                        name=(
                            "regime_calibration_snapshot"
                        ),
                        status="FAIL",
                        detail=(
                            "required regime calibration "
                            "snapshot directory is missing"
                        ),
                    )
                )

                issues.append(
                    ProductionChampionAuditIssue(
                        code=(
                            "calibration_snapshot_missing"
                        ),
                        severity="FAIL",
                        message=(
                            "required regime calibration "
                            "snapshot is missing"
                        ),
                    )
                )

            elif not calibration_snapshot.is_dir():
                snapshot_failed = True

                checks.append(
                    ProductionChampionAuditCheck(
                        name=(
                            "regime_calibration_snapshot"
                        ),
                        status="FAIL",
                        detail=(
                            "regime calibration snapshot "
                            "path is not a directory"
                        ),
                    )
                )

                issues.append(
                    ProductionChampionAuditIssue(
                        code=(
                            "calibration_snapshot_invalid"
                        ),
                        severity="FAIL",
                        message=(
                            "regime calibration snapshot "
                            "path is invalid"
                        ),
                    )
                )

            else:
                checks.append(
                    ProductionChampionAuditCheck(
                        name=(
                            "regime_calibration_snapshot"
                        ),
                        status="PASS",
                        detail=(
                            "required regime calibration "
                            "snapshot directory is available"
                        ),
                    )
                )

        bayesian_snapshot = (
            configuration
            .regime_bayesian_snapshot_root
        )

        if bayesian_snapshot is not None:
            if not bayesian_snapshot.exists():
                snapshot_failed = True

                checks.append(
                    ProductionChampionAuditCheck(
                        name=(
                            "regime_bayesian_snapshot"
                        ),
                        status="FAIL",
                        detail=(
                            "required regime bayesian "
                            "snapshot directory is missing"
                        ),
                    )
                )

                issues.append(
                    ProductionChampionAuditIssue(
                        code=(
                            "bayesian_snapshot_missing"
                        ),
                        severity="FAIL",
                        message=(
                            "required regime bayesian "
                            "snapshot is missing"
                        ),
                    )
                )

            elif not bayesian_snapshot.is_dir():
                snapshot_failed = True

                checks.append(
                    ProductionChampionAuditCheck(
                        name=(
                            "regime_bayesian_snapshot"
                        ),
                        status="FAIL",
                        detail=(
                            "regime bayesian snapshot "
                            "path is not a directory"
                        ),
                    )
                )

                issues.append(
                    ProductionChampionAuditIssue(
                        code=(
                            "bayesian_snapshot_invalid"
                        ),
                        severity="FAIL",
                        message=(
                            "regime bayesian snapshot "
                            "path is invalid"
                        ),
                    )
                )

            else:
                checks.append(
                    ProductionChampionAuditCheck(
                        name=(
                            "regime_bayesian_snapshot"
                        ),
                        status="PASS",
                        detail=(
                            "required regime bayesian "
                            "snapshot directory is available"
                        ),
                    )
                )

        if snapshot_failed:
            return ProductionChampionAuditResult(
                status="FAIL",
                selected_model=(
                    configuration.requested_model
                ),
                resolved_model=(
                    configuration.resolved_model
                ),
                fallback_applied=(
                    configuration.fallback_applied
                ),
                fallback_reason=(
                    configuration.fallback_reason
                ),
                checks=tuple(
                    checks
                ),
                issues=tuple(
                    issues
                ),
            )

        # --------------------------------------------------
        # Production pipeline load.
        #
        # Snapshot filesystem readiness has already been
        # validated above. Use the production configuration
        # contract directly so this audit exercises the same
        # pipeline-loading boundary used by production.
        # --------------------------------------------------

        try:
            PredictionPipeline.load(
                **configuration.pipeline_kwargs()
            )
        except Exception:
            checks.append(
                ProductionChampionAuditCheck(
                    name="pipeline_load",
                    status="FAIL",
                    detail=(
                        "production prediction "
                        "pipeline could not be loaded"
                    ),
                )
            )

            issues.append(
                ProductionChampionAuditIssue(
                    code="pipeline_load_failed",
                    severity="FAIL",
                    message=(
                        "production prediction "
                        "pipeline load failed"
                    ),
                )
            )

            return ProductionChampionAuditResult(
                status="FAIL",
                selected_model=(
                    configuration.requested_model
                ),
                resolved_model=(
                    configuration.resolved_model
                ),
                fallback_applied=(
                    configuration.fallback_applied
                ),
                fallback_reason=(
                    configuration.fallback_reason
                ),
                checks=tuple(
                    checks
                ),
                issues=tuple(
                    issues
                ),
            )

        checks.append(
            ProductionChampionAuditCheck(
                name="pipeline_load",
                status="PASS",
                detail=(
                    "production prediction "
                    "pipeline loaded successfully"
                ),
            )
        )

        # --------------------------------------------------
        # Classification
        # --------------------------------------------------

        if configuration.fallback_applied:
            issues.append(
                ProductionChampionAuditIssue(
                    code="baseline_fallback",
                    severity="WARN",
                    message=(
                        "production resolved "
                        "to baseline fallback"
                    ),
                )
            )

            status = "WARN"
        else:
            status = "PASS"

        return ProductionChampionAuditResult(
            status=status,
            selected_model=(
                configuration.requested_model
            ),
            resolved_model=(
                configuration.resolved_model
            ),
            fallback_applied=(
                configuration.fallback_applied
            ),
            fallback_reason=(
                configuration.fallback_reason
            ),
            checks=tuple(
                checks
            ),
            issues=tuple(
                issues
            ),
        )

    @staticmethod
    def _is_sha256(
        value: str,
    ) -> bool:
        if len(value) != 64:
            return False

        try:
            int(
                value,
                16,
            )
        except ValueError:
            return False

        return True

    @staticmethod
    def _failure(
        *,
        code: str,
        message: str,
        check_name: str,
        detail: str,
        checks: list[
            ProductionChampionAuditCheck
        ] | None = None,
        selected_model: str | None = None,
        resolved_model: str = "unresolved",
        fallback_applied: bool = False,
        fallback_reason: str | None = None,
    ) -> ProductionChampionAuditResult:
        current_checks = list(
            checks or ()
        )

        current_checks.append(
            ProductionChampionAuditCheck(
                name=check_name,
                status="FAIL",
                detail=detail,
            )
        )

        return ProductionChampionAuditResult(
            status="FAIL",
            selected_model=selected_model,
            resolved_model=resolved_model,
            fallback_applied=fallback_applied,
            fallback_reason=fallback_reason,
            checks=tuple(
                current_checks
            ),
            issues=(
                ProductionChampionAuditIssue(
                    code=code,
                    severity="FAIL",
                    message=message,
                ),
            ),
        )
