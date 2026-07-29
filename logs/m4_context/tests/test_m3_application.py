"""Sprint M3 fast smoke tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

from lrp.io import (
    history_until_round,
    load_history_csv,
    long_gap_numbers,
    previous_numbers,
    sha256_file,
    write_prediction_artifacts,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        history_path = root / "history.csv"

        rows = [
            "round,n1,n2,n3,n4,n5,n6,bonus"
        ]

        for round_no in range(1, 61):
            numbers = sorted(
                {
                    ((round_no * 7 + offset * 6) % 45) + 1
                    for offset in range(6)
                }
            )

            if len(numbers) != 6:
                raise AssertionError(
                    f"invalid synthetic row: {numbers}"
                )

            bonus = next(
                number
                for number in range(1, 46)
                if number not in numbers
            )

            rows.append(
                ",".join(
                    [
                        str(round_no),
                        *[str(number) for number in numbers],
                        str(bonus),
                    ]
                )
            )

        history_path.write_text(
            "\n".join(rows) + "\n",
            encoding="utf-8",
        )

        history = load_history_csv(history_path)
        bounded = history_until_round(
            history,
            target_round=61,
        )

        assert len(history) == 60
        assert len(bounded) == 60
        assert len(previous_numbers(bounded)) == 6
        assert long_gap_numbers(bounded)

        payload = {
            "round": 61,
            "seed": 20260721,
            "sets": [],
            "metadata": {
                "statistics_version": "test",
                "candidate_version": "test",
            },
        }

        artifacts = write_prediction_artifacts(
            payload,
            output_root=root / "predictions",
        )

        prediction_path = Path(
            artifacts["prediction_path"]
        )
        manifest_path = Path(
            artifacts["manifest_path"]
        )

        assert prediction_path.is_file()
        assert manifest_path.is_file()

        manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

        assert (
            manifest["files"]["prediction.json"][
                "sha256"
            ]
            == sha256_file(prediction_path)
        )

        print(
            json.dumps(
                {
                    "status": "PASS",
                    "history_draws": len(history),
                    "previous_numbers": sorted(
                        previous_numbers(bounded)
                    ),
                    "long_gap_count": len(
                        long_gap_numbers(bounded)
                    ),
                    "prediction_path": str(
                        prediction_path
                    ),
                    "manifest_path": str(
                        manifest_path
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
