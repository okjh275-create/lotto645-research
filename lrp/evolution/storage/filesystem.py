from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4


class AtomicTextFileSystem:
    """Small filesystem adapter with atomic text replacement."""

    def write_atomic(
        self,
        path: Path,
        content: str,
        *,
        overwrite: bool = False,
    ) -> None:
        if not isinstance(path, Path):
            raise TypeError("path must be a pathlib.Path")

        if not isinstance(content, str):
            raise TypeError("content must be a string")

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if path.exists() and not overwrite:
            raise FileExistsError(str(path))

        temporary_path = path.with_name(
            f".{path.name}.{uuid4().hex}.tmp"
        )

        try:
            with temporary_path.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())

            if path.exists() and not overwrite:
                raise FileExistsError(str(path))

            temporary_path.replace(path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def read_text(
        self,
        path: Path,
    ) -> str:
        if not isinstance(path, Path):
            raise TypeError("path must be a pathlib.Path")

        return path.read_text(encoding="utf-8")

    def list_files(
        self,
        directory: Path,
        *,
        pattern: str,
    ) -> tuple[Path, ...]:
        if not isinstance(directory, Path):
            raise TypeError(
                "directory must be a pathlib.Path"
            )

        if not directory.exists():
            return ()

        return tuple(
            sorted(
                (
                    path
                    for path in directory.glob(pattern)
                    if path.is_file()
                ),
                key=lambda item: item.name,
            )
        )