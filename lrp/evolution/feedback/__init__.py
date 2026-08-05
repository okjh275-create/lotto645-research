"""Adaptive feedback contracts and services."""

from lrp.evolution.feedback.analyzer import (
    AdaptiveFeedbackAnalyzer,
)
from lrp.evolution.feedback.automation import (
    AdaptiveAutomationResult,
    AdaptiveAutomationService,
)
from lrp.evolution.feedback.contracts import (
    AdaptiveAction,
    AdaptiveDecision,
    AdaptiveFeedback,
    AdaptiveRecommendation,
)
from lrp.evolution.feedback.doctor import (
    AdaptiveAutomationDoctor,
    AdaptiveAutomationDoctorReport,
)
from lrp.evolution.feedback.doctor_report_writer import (
    AdaptiveAutomationDoctorReportWriter,
    AdaptiveDoctorReportWriteResult,
)
from lrp.evolution.feedback.profile_integrity import (
    AdaptiveProfileIntegrityDoctor,
    AdaptiveProfileIntegrityReport,
)
from lrp.evolution.feedback.profile_update import (
    AdaptiveProfileUpdatePlan,
    AdaptiveProfileUpdatePlanner,
)
from lrp.evolution.feedback.recommendation import (
    AdaptiveRecommendationEngine,
)
from lrp.evolution.feedback.repository import (
    AdaptiveAutomationRepository,
    AdaptiveAutomationSaveResult,
)
from lrp.evolution.feedback.repository_status_analyzer import (
    AdaptiveRepositoryStatusAnalyzer,
)
from lrp.evolution.feedback.rollback import (
    AdaptiveRollbackDiff,
    AdaptiveRollbackManager,
    AdaptiveRollbackPlan,
)
from lrp.evolution.feedback.rollback_repository import (
    AdaptiveRollbackRepository,
    AdaptiveRollbackSaveResult,
)
from lrp.evolution.feedback.safety import (
    AdaptiveSafetyGuard,
    AdaptiveSafetyResult,
)
from lrp.evolution.feedback.status import (
    AdaptiveStatusIssue,
    AdaptiveStatusReport,
)
from lrp.evolution.feedback.runner import (
    RevisionAwareAutomationResult,
    RevisionAwareAutomationRunner,
)

__all__ = [
    "AdaptiveAction",
    "AdaptiveAutomationDoctor",
    "AdaptiveAutomationDoctorReport",
    "AdaptiveAutomationDoctorReportWriter",
    "AdaptiveAutomationRepository",
    "AdaptiveAutomationResult",
    "AdaptiveAutomationSaveResult",
    "AdaptiveAutomationService",
    "AdaptiveDecision",
    "AdaptiveDoctorReportWriteResult",
    "AdaptiveFeedback",
    "AdaptiveFeedbackAnalyzer",
    "AdaptiveProfileIntegrityDoctor",
    "AdaptiveProfileIntegrityReport",
    "AdaptiveProfileUpdatePlan",
    "AdaptiveProfileUpdatePlanner",
    "AdaptiveRecommendation",
    "AdaptiveRecommendationEngine",
    "AdaptiveRepositoryStatusAnalyzer",
    "AdaptiveRollbackDiff",
    "AdaptiveRollbackManager",
    "AdaptiveRollbackPlan",
    "AdaptiveRollbackRepository",
    "AdaptiveRollbackSaveResult",
    "AdaptiveSafetyGuard",
    "AdaptiveSafetyResult",
    "AdaptiveStatusIssue",
    "AdaptiveStatusReport",
    "RevisionAwareAutomationResult",
    "RevisionAwareAutomationRunner",
]
