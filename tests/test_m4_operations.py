"""Sprint M4 operation-layer smoke tests."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import zipfile

from lrp.operations import (
    create_backup,
    restore_backup,
    review_prediction,
    verify_manifest,
    write_operation_artifact,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        prediction = {
            "round": 1234,
            "seed": 20260721,
            "sets": [
                {"id": "S1", "numbers": [1, 2, 3, 4, 5, 6], "score": 1.0},
                {"id": "S2", "numbers": [7, 8, 9, 10, 11, 12], "score": 0.5},
            ],
            "top5_practical": ["S1"],
            "metadata": {},
        }
        review = review_prediction(
            prediction,
            winning_numbers=[1, 2, 3, 20, 21, 22],
            bonus=6,
        )
        assert review["summary"]["best_main_hits"] == 3
        artifact = write_operation_artifact(
            review,
            output_root=root / "operations",
            artifact_type="review",
            round_no=1234,
            filename="review.json",
        )
        verified = verify_manifest(artifact["manifest_path"])
        assert verified["status"] == "PASS"

        source = root / "source"
        (source / "nested").mkdir(parents=True)
        (source / "nested" / "sample.txt").write_text("hello", encoding="utf-8")
        backup = create_backup(
            source_root=source,
            destination_root=root / "backups",
            label="smoke",
        )
        assert zipfile.is_zipfile(backup["archive"])
        restored = restore_backup(
            archive_path=backup["archive"],
            destination_root=root / "restored",
        )
        assert restored["restored_count"] == 1
        assert (root / "restored" / "nested" / "sample.txt").read_text(encoding="utf-8") == "hello"

        print(json.dumps({
            "status": "PASS",
            "review_best_hits": review["summary"]["best_main_hits"],
            "manifest_status": verified["status"],
            "backup": backup["archive"],
            "restored_count": restored["restored_count"],
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
