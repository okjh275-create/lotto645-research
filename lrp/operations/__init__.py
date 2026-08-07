"""Operational services for Lotto645 Research Platform."""

from .review import prize_rank, review_prediction
from .round_completion_repository import (
    RoundCompletionRepository,
)
from .round_completion_summary import (
    RoundCompletionSummary,
    summarize_round_completions,
)
from .runtime import (
    append_operation_log,
    atomic_write,
    create_backup,
    restore_backup,
    sha256_file,
    verify_manifest,
    write_operation_artifact,
)

__all__ = [
    "append_operation_log",
    "atomic_write",
    "create_backup",
    "prize_rank",
    "restore_backup",
    "RoundCompletionRepository",
    "RoundCompletionSummary",
    "review_prediction",
    "sha256_file",
    "summarize_round_completions",
    "verify_manifest",
    "write_operation_artifact",
]
