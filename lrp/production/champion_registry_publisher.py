"""Production champion registry publication."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import tempfile

from lrp.production.champion_decision_reader import (
    ProductionChampionDecisionReader,
)
from lrp.production.champion_registry import (
    ProductionChampionRegistry,
)
from lrp.production.champion_registry_reader import (
    ProductionChampionRegistryReader,
)
from lrp.production.production_registry_lock import (
    ProductionRegistryWriterLock,
)


_KST = timezone(
    timedelta(hours=9)
)


@dataclass(frozen=True)
class ProductionChampionPublicationResult:
    """Result and provenance of a champion publication."""

    source_path: Path
    source_sha256: str
    published_path: Path
    published_at_kst: str
    selected_model: str | None

    def as_dict(
        self,
    ) -> dict[str, object]:
        return {
            "source_path": str(
                self.source_path
            ),
            "source_sha256": (
                self.source_sha256
            ),
            "published_path": str(
                self.published_path
            ),
            "published_at_kst": (
                self.published_at_kst
            ),
            "selected_model": (
                self.selected_model
            ),
        }


class ProductionChampionRegistryPublisher:
    """Publish a persisted decision into the active registry."""

    def publish(
        self,
        *,
        source_decision: str | Path,
        registry_root: str | Path,
    ) -> ProductionChampionPublicationResult:
        source = Path(
            source_decision
        )

        if not source.exists():
            raise FileNotFoundError(
                source
            )

        if source.is_dir():
            raise IsADirectoryError(
                source
            )

        if not source.is_file():
            raise FileNotFoundError(
                source
            )

        registry = (
            ProductionChampionRegistry(
                registry_root
            )
        )

        root = registry.root

        if (
            root.exists()
            and not root.is_dir()
        ):
            raise NotADirectoryError(
                root
            )

        try:
            decision = (
                ProductionChampionDecisionReader()
                .read(source)
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "invalid champion decision schema"
            ) from exc

        source_bytes = source.read_bytes()
        source_sha256 = (
            hashlib.sha256(
                source_bytes
            )
            .hexdigest()
        )

        with ProductionRegistryWriterLock(
            root,
        ):
            root.mkdir(
                parents=True,
                exist_ok=True,
            )

            active_root = (
                root
                / "active"
            )

            active_root.mkdir(
                parents=True,
                exist_ok=True,
            )

            target = registry.active_decision_path

            temporary_path: Path | None = None

            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=target.parent,
                    prefix=(
                        f".{target.name}."
                    ),
                    suffix=".tmp",
                    delete=False,
                ) as temporary:
                    temporary.write(
                        source_bytes
                    )
                    temporary.flush()
                    os.fsync(
                        temporary.fileno()
                    )

                    temporary_path = Path(
                        temporary.name
                    )

                os.replace(
                    temporary_path,
                    target,
                )

                temporary_path = None

                verified = (
                    ProductionChampionRegistryReader()
                    .read(root)
                )

                if (
                    verified.selected_model
                    != decision.selected_model
                ):
                    raise RuntimeError(
                        "published champion decision "
                        "verification failed"
                    )

            finally:
                if (
                    temporary_path is not None
                    and temporary_path.exists()
                ):
                    temporary_path.unlink()

            result = (
                ProductionChampionPublicationResult(
                    source_path=source,
                    source_sha256=source_sha256,
                    published_path=target,
                    published_at_kst=(
                        datetime.now(
                            _KST
                        )
                        .isoformat()
                    ),
                    selected_model=(
                        decision.selected_model
                    ),
                )
            )

            self._write_publication_record(
                result=result,
                active_root=active_root,
            )

            self._write_decision_revision(
                source_bytes=source_bytes,
                source_sha256=source_sha256,
                registry_root=root,
            )

            self._write_publication_revision(
                result=result,
                registry_root=root,
            )

        return result

    @staticmethod
    def _write_decision_revision(
        *,
        source_bytes: bytes,
        source_sha256: str,
        registry_root: Path,
    ) -> None:
        decision_history_root = (
            registry_root
            / "history"
            / "decisions"
        )

        decision_history_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        revision_path = (
            decision_history_root
            / f"{source_sha256}.json"
        )

        if revision_path.exists():
            if (
                revision_path.read_bytes()
                != source_bytes
            ):
                raise RuntimeError(
                    "decision revision hash collision"
                )

            return

        with revision_path.open(
            "xb"
        ) as revision:
            revision.write(
                source_bytes
            )

    @staticmethod
    def _write_publication_revision(
        *,
        result: ProductionChampionPublicationResult,
        registry_root: Path,
    ) -> None:
        history_root = (
            registry_root
            / "history"
        )

        history_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = (
            json.dumps(
                result.as_dict(),
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        ).encode(
            "utf-8"
        )

        revision_sha256 = (
            hashlib.sha256(
                payload
            )
            .hexdigest()
        )

        revision_path = (
            history_root
            / f"{revision_sha256}.json"
        )

        if revision_path.exists():
            if (
                revision_path.read_bytes()
                != payload
            ):
                raise RuntimeError(
                    "publication revision hash collision"
                )

            return

        with revision_path.open(
            "xb"
        ) as revision:
            revision.write(
                payload
            )

    @staticmethod
    def _write_publication_record(
        *,
        result: ProductionChampionPublicationResult,
        active_root: Path,
    ) -> None:
        publication_path = (
            active_root
            / "publication.json"
        )

        temporary_path: Path | None = None

        payload = (
            json.dumps(
                result.as_dict(),
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        ).encode(
            "utf-8"
        )

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=active_root,
                prefix=".publication.json.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(
                    payload
                )

                temporary.flush()

                os.fsync(
                    temporary.fileno()
                )

                temporary_path = Path(
                    temporary.name
                )

            os.replace(
                temporary_path,
                publication_path,
            )

            temporary_path = None

        finally:
            if (
                temporary_path is not None
                and temporary_path.exists()
            ):
                temporary_path.unlink()
