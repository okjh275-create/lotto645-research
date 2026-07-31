from __future__ import annotations

import pytest

from lrp.evolution.repositories.snapshot_repository import (
    SnapshotRepository,
)


def test_repository_contract_is_abstract() -> None:
    with pytest.raises(TypeError):
        SnapshotRepository()
