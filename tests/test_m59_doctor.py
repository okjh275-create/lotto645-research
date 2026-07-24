"""M5.9 doctor smoke test."""

from __future__ import annotations

from pathlib import Path
import tempfile

from lrp.management import run_doctor


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

        result = run_doctor(
            project_root=root,
        )

        assert result["status"] == "PASS"
        assert result["failure_count"] == 0

        assert any(
            item["name"] == "sqlite"
            and item["status"] == "PASS"
            for item in result["checks"]
        )

        assert any(
            item["name"] == "filesystem_write"
            and item["status"] == "PASS"
            for item in result["checks"]
        )

        print("PASS: M5.9 doctor")


if __name__ == "__main__":
    main()
