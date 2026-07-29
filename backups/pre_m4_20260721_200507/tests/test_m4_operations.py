"""M4 operation-layer smoke test."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from lrp.operations import create_backup, review_prediction, verify_manifest, write_operation_artifact


def main() -> None:
    prediction = {
        "round": 9999,
        "seed": 20260721,
        "sets": [
            {"id": "S1", "numbers": [1, 2, 3, 4, 5, 6], "score": 0.9},
            {"id": "S2", "numbers": [1, 7, 8, 9, 10, 11], "score": 0.8},
        ],
        "top5_practical": ["S1", "S2"],
        "metadata": {},
    }
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        review = review_prediction(prediction, winning_numbers=[1, 2, 3, 20, 21, 22], bonus=4)
        assert review["summary"]["best_main_hits"] == 3
        artifact = write_operation_artifact(review, output_root=root / "snapshots", artifact_type="reviews", round_no=9999, filename="review.json")
        verified = verify_manifest(artifact["manifest_path"])
        assert verified["status"] == "PASS"
        backup = create_backup(source_root=root / "snapshots", destination_root=root / "backups", label="smoke")
        assert Path(backup["archive"]).is_file()
        print(json.dumps({"status": "PASS", "best_hits": 3, "manifest": verified["status"], "backup": backup["archive"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
