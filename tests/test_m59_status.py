"""M5.9 status collector smoke test."""

from __future__ import annotations

from pathlib import Path
import tempfile

from lrp.management import collect_platform_status


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        (root / "lrp").mkdir()
        (root / "tests").mkdir()
        (root / "tools").mkdir()
        (root / "config.yaml").write_text(
            "project: test\n",
            encoding="utf-8",
        )

        prediction = (
            root
            / "predictions"
            / "round_1234"
            / "seed_1"
            / "prediction.json"
        )
        prediction.parent.mkdir(parents=True)
        prediction.write_text(
            "{}\n",
            encoding="utf-8",
        )

        review = (
            root
            / "snapshots"
            / "reviews"
            / "round_1233"
            / "review.json"
        )
        review.parent.mkdir(parents=True)
        review.write_text(
            "{}\n",
            encoding="utf-8",
        )

        backup = root / "backups" / "test.zip"
        backup.parent.mkdir(parents=True)
        backup.write_bytes(b"test")

        result = collect_platform_status(
            project_root=root,
        )

        assert result["status"] == "PASS"
        assert result["counts"]["predictions"] == 1
        assert result["counts"]["reviews"] == 1
        assert result["counts"]["backups"] == 1
        assert result["latest"]["prediction"] is not None
        assert result["latest"]["review"] is not None
        assert result["latest"]["backup"] is not None

        print("PASS: M5.9 status collector")


if __name__ == "__main__":
    main()
