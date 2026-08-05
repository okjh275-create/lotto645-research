"""Write adaptive automation doctor reports as deterministic JSON."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lrp.evolution.feedback.doctor import (
    AdaptiveAutomationDoctorReport,
)


@dataclass(frozen=True, slots=True)
class AdaptiveDoctorReportWriteResult:
    """Result of writing an adaptive doctor JSON report."""

    path: Path
    created: bool
    changed: bool
    byte_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "created": self.created,
            "changed": self.changed,
            "byte_count": self.byte_count,
        }


class AdaptiveAutomationDoctorReportWriter:
    """Persist adaptive doctor reports as deterministic JSON."""

    def write(
        self,
        report: AdaptiveAutomationDoctorReport,
        output_path: Path,
    ) -> AdaptiveDoctorReportWriteResult:
        if not isinstance(
            report,
            AdaptiveAutomationDoctorReport,
        ):
            raise TypeError(
                "report must be an "
                "AdaptiveAutomationDoctorReport"
            )

        path = Path(output_path)

        if path.exists() and path.is_dir():
            raise IsADirectoryError(path)

        serialized = self.serialize(report)
        existed_before = path.exists()

        if existed_before:
            existing = path.read_bytes()

            if existing == serialized:
                self._verify_written_report(
                    path=path,
                    expected_payload=report.as_dict(),
                    expected_bytes=serialized,
                )

                return AdaptiveDoctorReportWriteResult(
                    path=path,
                    created=False,
                    changed=False,
                    byte_count=len(serialized),
                )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = path.with_name(
            path.name + ".tmp"
        )

        if temporary.exists():
            if temporary.is_dir():
                raise IsADirectoryError(temporary)

            temporary.unlink()

        try:
            with temporary.open("wb") as stream:
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())

            os.replace(
                temporary,
                path,
            )
        finally:
            if temporary.exists():
                temporary.unlink()

        self._verify_written_report(
            path=path,
            expected_payload=report.as_dict(),
            expected_bytes=serialized,
        )

        return AdaptiveDoctorReportWriteResult(
            path=path,
            created=not existed_before,
            changed=True,
            byte_count=len(serialized),
        )

    @staticmethod
    def serialize(
        report: AdaptiveAutomationDoctorReport,
    ) -> bytes:
        if not isinstance(
            report,
            AdaptiveAutomationDoctorReport,
        ):
            raise TypeError(
                "report must be an "
                "AdaptiveAutomationDoctorReport"
            )

        text = (
            json.dumps(
                report.as_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )

        return text.encode("utf-8")

    @staticmethod
    def _verify_written_report(
        *,
        path: Path,
        expected_payload: dict[str, Any],
        expected_bytes: bytes,
    ) -> None:
        actual_bytes = path.read_bytes()

        if actual_bytes != expected_bytes:
            raise RuntimeError(
                "doctor report byte verification failed"
            )

        if actual_bytes.startswith(
            b"\xef\xbb\xbf"
        ):
            raise RuntimeError(
                "doctor report must not contain "
                "a UTF-8 BOM"
            )

        if not actual_bytes.endswith(b"\n"):
            raise RuntimeError(
                "doctor report must end with "
                "a newline"
            )

        loaded = json.loads(
            actual_bytes.decode("utf-8")
        )

        if not isinstance(loaded, dict):
            raise RuntimeError(
                "doctor report must contain "
                "a JSON object"
            )

        if loaded != expected_payload:
            raise RuntimeError(
                "doctor report payload "
                "verification failed"
            )
